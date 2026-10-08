"""AS-16: actual HTTP and Google cryptographic verifier, synthetic data only.

--local-route compiles the exact Google/me/logout HTTP routes, request model,
dependencies and session helpers from server.py into real FastAPI. It replaces
neither Google's verifier nor the route/DB functions. Only certificate transport
returns an ephemeral RSA public key; Supabase is a loopback REST fixture. This
mode is NOT evidence of full-server import, ML startup or real Google login.

Without --local-route the runner imports FULL server.app and runs real lifespan
startup in an isolated Linux image, with read-only /app/models and writable
tmpfs cache. Use --network none and NEVER --env-file/production credentials.

Prepared example:
  python -B tests_v4/google_security_http_candidate.py --local-route \
    --policy prepared --evidence /private/http-prepared.safe.json
Original10b54e7 example:
  python -B tests_v4/google_security_http_candidate.py --local-route \
    --policy baseline --server-source /private/old-server.py \
    --evidence /private/http-baseline.safe.json
Baseline success means expected unsafe behavior was reproduced locally, NOT
that the baseline is secure. Output contains only statuses/booleans/counts and
source hashes: no private keys, ID/session tokens, users or captured logs.

Dependencies: fastapi, uvicorn, requests, google-auth, cryptography. Full mode
also requires the unchanged existing ML/server image. --self-test is stdlib only.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import copy
from datetime import datetime, timedelta, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import secrets
import socket
import sys
import threading
import time
import types
from urllib.parse import parse_qs, urlparse
from uuid import uuid4


KEY = "AS16_SYNTHETIC_NOT_A_PRODUCTION_CREDENTIAL"
AUDIENCE = "as16-synthetic.apps.googleusercontent.com"
SUBJECT = "as16_synthetic_subject_a"
EMAIL = "as16.synthetic.a@gmail.com"
OWNER = "00000000-0000-4000-8000-000000000001"


class SafeFailure(Exception):
    """Only fixed check IDs; never response, key, token or database detail."""


def require(condition, check):
    if not condition:
        raise SafeFailure(check)


def user(**changes):
    return {"id": OWNER, "email": EMAIL, "google_sub": SUBJECT,
            "name": "Synthetic profile A", "role": "user", "provider": "google",
            "password_hash": "SYNTHETIC_HASH", "salt": "SYNTHETIC_SALT",
            "created_at": "2026-01-01T00:00:00Z", **changes}


class Fixture:
    def __init__(self):
        self.lock = threading.RLock()
        self.users, self.sessions = [], []
        self.identity_writes = self.training_writes = self.training_reads = 0
        self.unexpected_calls = self.calls = self.storage_list_reads = 0
        self.reset()

    def reset(self, users=None):
        with self.lock:
            self.users = copy.deepcopy([user()] if users is None else users)
            self.sessions = []

    @staticmethod
    def matches(row, filters):
        for key, value in filters.items():
            if key in {"select", "limit", "order"}:
                continue
            if key == "or":
                # Exact old synthetic OR query, not a SQL/PostgREST engine.
                clauses = value.strip("()").split(",")
                if not any(".eq." in c and row.get(c.split(".eq.", 1)[0]) ==
                           c.split(".eq.", 1)[1] for c in clauses):
                    return False
            elif value == "is.null":
                if row.get(key) is not None:
                    return False
            elif value.startswith("eq."):
                if row.get(key) != value[3:]:
                    return False
            elif value.startswith("gt."):
                if not isinstance(row.get(key), str) or row[key] <= value[3:]:
                    return False
            else:
                return False
        return True

    def dispatch(self, method, path, filters, body):
        with self.lock:
            self.calls += 1
            if path == "training_state":
                if method != "GET":
                    self.training_writes += 1
                    return 403, {"error": "synthetic_training_read_only"}
                self.training_reads += 1
                return 200, [{"id": 1, "active_model_version": 4, "last_model_version": 4,
                              "training_in_progress": False, "retrain_requested": False}]
            if path == "users":
                selected = [r for r in self.users if self.matches(r, filters)]
                if method == "GET":
                    return 200, copy.deepcopy(selected[:int(filters.get("limit", "100"))])
                if method == "PATCH" and isinstance(body, dict):
                    self.identity_writes += 1
                    for row in selected:
                        row.update(body)
                    return 200, copy.deepcopy(selected)
                if method == "POST" and isinstance(body, dict):
                    self.identity_writes += 1
                    if any(r.get("email") == body.get("email") or
                           (body.get("google_sub") and r.get("google_sub") == body["google_sub"])
                           for r in self.users):
                        return 409, {"error": "synthetic_unique_constraint"}
                    self.users.append(copy.deepcopy(body))
                    return 201, [copy.deepcopy(body)]
            if path == "auth_sessions":
                selected = [r for r in self.sessions if self.matches(r, filters)]
                if method == "GET":
                    result = []
                    for session in selected[:int(filters.get("limit", "100"))]:
                        owners = [r for r in self.users if r["id"] == session["user_id"]]
                        result.append({**copy.deepcopy(session),
                                       "users": copy.deepcopy(owners[0]) if owners else None})
                    return 200, result
                if method == "POST" and isinstance(body, dict):
                    self.sessions.append(copy.deepcopy(body))
                    return 201, None
                if method == "DELETE":
                    self.sessions = [r for r in self.sessions if r not in selected]
                    return 204, None
            self.unexpected_calls += 1
            return 403, {"error": "non_fixture_path_refused"}


def fixture_server(fixture):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def handle_request(self):
            if (self.headers.get("apikey") != KEY or
                    self.headers.get("Authorization") != "Bearer " + KEY):
                self.send_response(403)
                self.end_headers()
                return
            parsed = urlparse(self.path)
            if (parsed.path == "/storage/v1/object/list/arborscan-models"
                    and self.command == "POST"):
                # Supabase Storage discovery uses POST for a read-only list.
                with fixture.lock:
                    fixture.storage_list_reads += 1
                status, result = 200, []
            elif parsed.path.startswith("/rest/v1/"):
                size = int(self.headers.get("Content-Length", "0"))
                if size > 65536:
                    self.send_response(413)
                    self.end_headers()
                    return
                body = json.loads(self.rfile.read(size)) if size else None
                filters = {k: v[0] for k, v in parse_qs(parsed.query, keep_blank_values=True).items()}
                status, result = fixture.dispatch(self.command, parsed.path[9:], filters, body)
            else:
                with fixture.lock:
                    fixture.unexpected_calls += 1
                status, result = 403, {"error": "non_fixture_path_refused"}
            encoded = b"" if result is None else json.dumps(result).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
        do_GET = do_POST = do_PATCH = do_DELETE = handle_request
    service = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    service.daemon_threads = True
    thread = threading.Thread(target=service.serve_forever, daemon=True)
    thread.start()
    return service, thread


def clean_environment():
    for name in ("SUPABASE_SERVICE_KEY", "SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_URL",
                 "PLANTNET_API_KEY", "OPENWEATHER_API_KEY", "WEATHER_API_KEY",
                 "OPENWEATHERMAP_API_KEY", "DATABASE_URL", "PGPASSWORD"):
        require(not os.environ.get(name), "production_or_external_environment_refused")
    require(os.environ.get("ARBORSCAN_GOOGLE_IDENTITY_CLAIMS_ENABLED", "false").lower()
            == "false", "claims_flag_must_be_false")


def readonly_mount(path):
    resolved, best = Path(path).resolve(), None
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        fields = line.split()
        mount = Path(fields[4].replace("\\040", " ").replace("\\134", "\\"))
        if resolved == mount or mount in resolved.parents:
            if best is None or len(str(mount)) > len(str(best[0])):
                best = mount, fields[5].split(",")
    return bool(best and "ro" in best[1])


@contextlib.contextmanager
def loopback_only(ports):
    import requests
    original = requests.sessions.Session.request
    def guarded(session, method, url, *args, **kwargs):
        parsed = urlparse(url)
        require(parsed.scheme == "http" and parsed.hostname == "127.0.0.1"
                and parsed.port in ports, "non_fixture_network_request_refused")
        session.trust_env = False
        return original(session, method, url, *args, **kwargs)
    requests.sessions.Session.request = guarded
    try:
        yield
    finally:
        requests.sessions.Session.request = original


class Tokens:
    def __init__(self):
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from google.auth import crypt, jwt
        self.jwt, self.certificate_calls = jwt, 0
        private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        wrong = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        def pem(key):
            return key.private_bytes(serialization.Encoding.PEM,
                                     serialization.PrivateFormat.PKCS8,
                                     serialization.NoEncryption())
        self.signer = crypt.RSASigner.from_string(pem(private), key_id="as16-synthetic-key")
        self.wrong_signer = crypt.RSASigner.from_string(pem(wrong), key_id="as16-synthetic-key")
        self.public = private.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo).decode()
        # Ephemeral private keys exist only in process RAM, never files/logs.

    def token(self, changes=None, remove=(), wrong_signature=False):
        now = int(time.time())
        claims = {"iss": "https://accounts.google.com", "aud": AUDIENCE,
                  "iat": now - 5, "exp": now + 600, "sub": SUBJECT, "email": EMAIL,
                  "email_verified": True, "name": "Synthetic verified A"}
        claims.update(changes or {})
        for field in remove:
            claims.pop(field, None)
        return self.jwt.encode(self.wrong_signer if wrong_signature else self.signer,
                               claims).decode("ascii")

    def certificates(self, url, method="GET", **_):
        require(url == "https://www.googleapis.com/oauth2/v1/certs" and method == "GET",
                "unexpected_certificate_request")
        self.certificate_calls += 1
        return types.SimpleNamespace(status=200,
            data=json.dumps({"as16-synthetic-key": self.public}).encode())


def local_routes(source, prepared, fixture_url):
    """Bounded exact AST alternative, explicitly not complete-server startup."""
    from fastapi import FastAPI, Depends, Header, HTTPException
    from pydantic import BaseModel
    from google.oauth2 import id_token
    from google.auth.transport import requests as google_requests
    import requests
    names = {"AuthGoogleRequest", "_sb_headers", "_supabase_is_configured", "training_state_get",
             "_now_iso", "_email_norm", "_hash_password", "_user_public", "_db_json", "_create_session",
             "_get_user_by_token", "_extract_bearer_token", "_resolve_auth_token",
             "require_authenticated_user", "auth_google", "auth_me", "auth_logout"}
    parsed = ast.parse(source.read_text(encoding="utf-8"))
    selected = [node for node in parsed.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and node.name in names]
    require({node.name for node in selected} == names, "exact_route_source_incomplete")
    module = types.ModuleType("as16_synthetic_http_routes")
    module.__dict__.update({"__file__": str(source), "app": FastAPI(), "BaseModel": BaseModel,
        "Depends": Depends, "Header": Header, "HTTPException": HTTPException,
        "requests": requests, "google_id_token": id_token, "google_requests": google_requests,
        "GOOGLE_CLIENT_ID": AUDIENCE, "SUPABASE_URL": fixture_url,
        "SUPABASE_DB_BASE": fixture_url + "/rest/v1", "SUPABASE_SERVICE_KEY": KEY,
        "AUTH_TOKEN_TTL_DAYS": 30, "datetime": datetime, "timedelta": timedelta,
        "uuid4": uuid4, "hashlib": hashlib, "secrets": secrets, "json": json})
    sys.modules[module.__name__] = module
    if prepared:
        from auth_google_identity import GoogleIdentityError, resolve_google_user, verified_google_identity
        module.__dict__.update({"GoogleIdentityError": GoogleIdentityError,
            "resolve_google_user": resolve_google_user, "verified_google_identity": verified_google_identity,
            "GOOGLE_IDENTITY_CLAIMS_ENABLED": False})
        require(any(isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and
            target.id == "GOOGLE_IDENTITY_CLAIMS_ENABLED" for target in node.targets)
            for node in parsed.body), "prepared_source_flag_missing")
    else:
        require(not any(isinstance(node, ast.ImportFrom) and node.module == "auth_google_identity"
                        for node in parsed.body), "baseline_must_be_original_source")
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec",
                 dont_inherit=True), module.__dict__)
    require(module.training_state_get()["active_model_version"] == 4,
            "synthetic_training_state_read_failed")
    return module


class HTTPChecks:
    def __init__(self, url, fixture, tokens, prepared, full_server):
        import requests
        self.client = requests.Session()
        self.client.trust_env = False
        self.url, self.fixture, self.tokens = url, fixture, tokens
        self.prepared, self.full_server, self.checks = prepared, full_server, []

    def request(self, method, path, **kwargs):
        return self.client.request(method, self.url + path, timeout=30, **kwargs)

    def record(self, name, callback):
        try:
            observation = callback()
            self.checks.append({"check": name, "passed": True,
                                **(observation if isinstance(observation, dict) else {})})
        except SafeFailure as exc:
            self.checks.append({"check": name, "passed": False, "category": str(exc)})
        except Exception:
            self.checks.append({"check": name, "passed": False, "category": "http_or_harness_error"})

    def denied(self, body, expected, no_db=False):
        self.fixture.reset()
        before, calls = copy.deepcopy(self.fixture.users), self.fixture.calls
        result = self.request("POST", "/auth/google", json=body)
        require(result.status_code == expected, "unexpected_http_status")
        require(self.fixture.users == before and not self.fixture.sessions,
                "denied_request_changed_identity_or_issued_session")
        if no_db:
            require(self.fixture.calls == calls, "verifier_failure_reached_database")
        return {"http_status": result.status_code, "identity_modified": False,
                "session_issued": False, "database_calls": self.fixture.calls - calls}

    def policy(self, body, rows, secure_status):
        self.fixture.reset(rows)
        before = copy.deepcopy(self.fixture.users)
        result = self.request("POST", "/auth/google", json=body)
        require(result.status_code == (secure_status if self.prepared else 200),
                "unexpected_policy_http_status")
        if self.prepared:
            require(self.fixture.users == before and not self.fixture.sessions,
                    "refused_policy_changed_identity_or_issued_session")
        else:
            require(bool(self.fixture.sessions) and self.fixture.users != before,
                    "unsafe_baseline_identity_and_session_not_reproduced")
        return {"http_status": result.status_code, "required_secure_http_status": secure_status,
                "identity_modified": self.fixture.users != before,
                "session_issued": bool(self.fixture.sessions),
                "unsafe_baseline_reproduced": not self.prepared}

    def linked(self):
        self.fixture.reset()
        before = copy.deepcopy(self.fixture.users)
        result = self.request("POST", "/auth/google", json={
            "id_token": self.tokens.token(), "email": "as16.spoof@gmail.com",
            "name": "Synthetic client spoof", "photo_url": "https://example.invalid/spoof",
            "role": "admin"})
        require(result.status_code == 200, "linked_login_failed")
        data = result.json()
        require(data["user"]["id"] == OWNER and data["user"]["email"] == EMAIL and
                data["user"]["role"] == "user", "canonical_owner_or_role_changed")
        require(isinstance(data.get("token"), str) and len(data["token"]) > 32,
                "opaque_session_not_issued")
        if self.prepared:
            require(self.fixture.users == before, "linked_identity_modified")
        return {"http_status": result.status_code, "canonical_owner_unchanged": True,
                "server_role_unchanged": True, "identity_modified": self.fixture.users != before}

    def sessions(self):
        b = user(id="00000000-0000-4000-8000-000000000002",
                 email="as16.synthetic.b@gmail.com", google_sub="as16_synthetic_subject_b")
        self.fixture.reset([user(), b])
        a_result = self.request("POST", "/auth/google", json={"id_token": self.tokens.token()})
        b_result = self.request("POST", "/auth/google", json={"id_token": self.tokens.token({
            "email": b["email"], "sub": b["google_sub"]})})
        require(a_result.status_code == b_result.status_code == 200, "two_owner_login_failed")
        a_token, b_token = a_result.json()["token"], b_result.json()["token"]
        require(a_token != b_token, "session_token_reused")
        require(self.request("GET", "/auth/me").status_code == 401, "unauthenticated_me_allowed")
        require(self.request("GET", "/auth/me", headers={"Authorization": "Bearer synthetic-unknown"})
                .status_code == 401, "unknown_bearer_allowed")
        for token, owner in ((a_token, OWNER), (b_token, b["id"])):
            me = self.request("GET", "/auth/me", headers={"Authorization": "Bearer " + token})
            require(me.status_code == 200 and me.json()["user"]["id"] == owner,
                    "session_owner_isolation_failed")
        me = self.request("GET", "/auth/me", params={"token": b_token},
                          headers={"Authorization": "Bearer " + a_token})
        require(me.status_code == 200 and me.json()["user"]["id"] == OWNER,
                "bearer_query_precedence_failed")
        require(self.request("POST", "/auth/logout", headers={"Authorization": "Bearer " + a_token})
                .status_code == 200, "logout_failed")
        require(self.request("GET", "/auth/me", headers={"Authorization": "Bearer " + a_token})
                .status_code == 401, "logged_out_session_still_valid")
        require(self.request("GET", "/auth/me", headers={"Authorization": "Bearer " + b_token})
                .status_code == 200, "logout_revoked_other_owner")
        with self.fixture.lock:
            for row in self.fixture.sessions:
                if row["token"] == b_token:
                    row["expires_at"] = "2000-01-01T00:00:00Z"
        require(self.request("GET", "/auth/me", headers={"Authorization": "Bearer " + b_token})
                .status_code == 401, "expired_session_still_valid")
        require(self.request("POST", "/auth/logout").status_code == 401,
                "unauthenticated_logout_allowed")

    def run(self):
        if self.full_server:
            self.record("full_server_models_ready", lambda: require(
                self.request("GET", "/health").json().get("models_ready") is True,
                "full_server_models_not_ready"))
        for name, body, expected in (("missing_id_token", {}, 422),
                ("empty_id_token", {"id_token": ""}, 401),
                ("malformed_id_token", {"id_token": "not-a-jwt"}, 401)):
            self.record(name, lambda body=body, expected=expected: self.denied(body, expected, True))
        self.record("real_rsa_signature_refused", lambda: self.denied({
            "id_token": self.tokens.token(wrong_signature=True)}, 401, True))
        variants = {"audience": {"aud": "other-synthetic.apps.googleusercontent.com"},
                    "issuer": {"iss": "https://issuer.example.invalid"},
                    "expiry": {"iat": int(time.time()) - 300, "exp": int(time.time()) - 120},
                    "future_iat": {"iat": int(time.time()) + 300, "exp": int(time.time()) + 600}}
        for name, changes in variants.items():
            self.record("real_verifier_" + name + "_refused", lambda changes=changes: self.denied(
                {"id_token": self.tokens.token(changes)}, 401, True))
        self.record("missing_email_client_fallback_policy", lambda: self.policy({
            "id_token": self.tokens.token(remove=("email",)), "email": EMAIL}, [user()], 401))
        for value, name in ((False, "false"), (None, "missing"), ("true", "string")):
            self.record("email_verified_" + name + "_policy", lambda value=value: self.policy(
                {"id_token": self.tokens.token(remove=("email_verified",)) if value is None
                    else self.tokens.token({"email_verified": value})}, [user()], 401))
        self.record("wrong_subject_reassignment_policy", lambda: self.policy({
            "id_token": self.tokens.token({"sub": "as16_synthetic_other_subject"})}, [user()], 409))
        self.record("ambiguous_subject_owner_policy", lambda: self.policy({"id_token": self.tokens.token()},
            [user(), user(id="00000000-0000-4000-8000-000000000003",
                          email="as16.synthetic.c@gmail.com")], 409))
        self.record("claims_false_unlinked_policy", lambda: self.policy({"id_token": self.tokens.token()},
                                                                       [user(google_sub=None)], 503))
        self.record("claims_false_new_profile_policy", lambda: self.policy({"id_token": self.tokens.token()}, [], 503))
        self.record("linked_owner_client_identity_and_role_ignored", self.linked)
        self.record("bearer_me_logout_expiry_two_owner_isolation", self.sessions)
        self.record("training_state_read_without_writes", lambda: require(
            self.fixture.training_reads > 0 and self.fixture.training_writes == 0,
            "training_state_contract_failed"))
        self.record("no_unexpected_rest_paths", lambda: require(self.fixture.unexpected_calls == 0,
                                                               "unexpected_rest_path_used"))
        if self.prepared:
            self.record("zero_identity_writes_all_requests", lambda: require(self.fixture.identity_writes == 0,
                                                                            "prepared_identity_write_detected"))
        self.client.close()


def self_test():
    fixture = Fixture()
    require(fixture.dispatch("GET", "training_state", {}, None)[1][0]["active_model_version"] == 4,
            "selftest_training_state")
    require(fixture.dispatch("POST", "training_state", {}, {})[0] == 403 and fixture.training_writes == 1,
            "selftest_training_write_observed_and_refused")
    require(fixture.dispatch("GET", "users", {"google_sub": "eq." + SUBJECT}, None)[1][0]["id"] == OWNER,
            "selftest_subject_filter")
    require(len(fixture.dispatch("GET", "users", {"or": "(google_sub.eq.absent,email.eq." + EMAIL + ")"}, None)[1]) == 1,
            "selftest_legacy_or")
    fixture.dispatch("POST", "auth_sessions", {}, {"token": "synthetic-session", "user_id": OWNER,
                                                     "expires_at": "2099-01-01T00:00:00Z"})
    result = fixture.dispatch("GET", "auth_sessions", {"token": "eq.synthetic-session",
                              "expires_at": "gt.2026-01-01T00:00:00Z"}, None)[1]
    require(result[0]["users"]["id"] == OWNER, "selftest_session_join")
    fixture.dispatch("DELETE", "auth_sessions", {"token": "eq.synthetic-session"}, None)
    require(not fixture.sessions, "selftest_logout")
    require(fixture.dispatch("GET", "not_fixture", {}, None)[0] == 403, "selftest_path_guard")
    previous = os.environ.get("SUPABASE_URL")
    try:
        os.environ["SUPABASE_URL"] = "https://production.example.invalid"
        try:
            clean_environment()
        except SafeFailure:
            pass
        else:
            raise SafeFailure("selftest_production_environment_not_refused")
    finally:
        if previous is None:
            os.environ.pop("SUPABASE_URL", None)
        else:
            os.environ["SUPABASE_URL"] = previous
    return {"self_test": True, "scope": "stdlib fixture only; not HTTP or crypto proof", "passed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("prepared", "baseline"), default="prepared")
    parser.add_argument("--local-route", action="store_true")
    parser.add_argument("--server-source", type=Path, help="Exact baseline source in local-route mode only")
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    proof = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "policy": args.policy,
             "python_version": sys.version.split()[0], "platform": sys.platform,
             "synthetic_data_only": True, "production_credentials_used": False,
             "production_mutations": False, "production_migration_executed": False,
             "real_google_login_proven": False, "verifier_replaced": False,
             "certificate_transport_synthetic": True, "passed": False}
    started, captured = time.monotonic(), io.StringIO()
    fixture_service = fixture_thread = bound = api = api_thread = server = None
    previous_request_factory = None
    exit_code = 1
    try:
        if args.self_test:
            proof.update(self_test())
            exit_code = 0
        else:
            clean_environment()
            require("server" not in sys.modules and "config" not in sys.modules, "server_preimport_refused")
            if not args.local_route:
                require(args.server_source is None and sys.platform.startswith("linux"),
                        "full_server_requires_linux_without_source_override")
                model_dir = Path(os.environ.get("MODEL_DIR", "/app/models")).resolve()
                require(readonly_mount(model_dir), "models_require_readonly_mount")
                model_files = sorted(model_dir.glob("*.pt"))
                model_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in model_files}
                require("model_v4.pt" in model_hashes and "stick_model.pt" in model_hashes, "model_fixture_missing")
            fixture = Fixture()
            fixture_service, fixture_thread = fixture_server(fixture)
            fixture_port = fixture_service.server_address[1]
            fixture_url = f"http://127.0.0.1:{fixture_port}"
            os.environ.update({"SUPABASE_URL": fixture_url, "SUPABASE_SERVICE_KEY": KEY,
                               "GOOGLE_CLIENT_ID": AUDIENCE, "ARBORSCAN_GOOGLE_IDENTITY_CLAIMS_ENABLED": "false",
                               "ACTIVE_MODEL_VERSION": "4", "PRELOAD_REMBG": "false", "SUPABASE_ENABLE_QUEUE": "false"})
            if not args.local_route:
                os.environ["MODEL_DIR"] = str(model_dir)
            bound = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            bound.bind(("127.0.0.1", 0))
            api_port = bound.getsockname()[1]
            repository = Path(__file__).resolve().parents[1]
            sys.path.insert(0, str(repository))
            source = args.server_source.resolve() if args.server_source else repository / "server.py"
            helper = repository / "auth_google_identity.py"
            proof.update({"server_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                          "harness_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          "identity_helper_used": args.policy == "prepared",
                          "identity_helper_sha256": hashlib.sha256(helper.read_bytes()).hexdigest()
                              if args.policy == "prepared" else None})
            with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured), loopback_only({fixture_port, api_port}):
                proof["dependency_versions"] = {name: importlib.metadata.version(name)
                    for name in ("fastapi", "uvicorn", "requests", "google-auth", "cryptography")}
                server = local_routes(source, args.policy == "prepared", fixture_url) if args.local_route \
                    else importlib.import_module("server")
                require(server.google_id_token is not None, "real_google_verifier_unavailable")
                require(getattr(server, "GOOGLE_IDENTITY_CLAIMS_ENABLED", None) is False if args.policy == "prepared"
                        else not hasattr(server, "GOOGLE_IDENTITY_CLAIMS_ENABLED"), "source_policy_mode_mismatch")
                tokens = Tokens()
                previous_request_factory = server.google_requests.Request
                server.google_requests.Request = lambda: tokens.certificates
                import uvicorn
                api = uvicorn.Server(uvicorn.Config(server.app, log_config=None, access_log=False, lifespan="on"))
                api_thread = threading.Thread(target=api.run, kwargs={"sockets": [bound]}, daemon=True)
                api_thread.start()
                deadline = time.monotonic() + 180
                while not api.started and api_thread.is_alive() and time.monotonic() < deadline:
                    time.sleep(0.05)
                require(api.started, "http_startup_failed_or_timed_out")
                checks = HTTPChecks(f"http://127.0.0.1:{api_port}", fixture, tokens,
                                    args.policy == "prepared", full_server=not args.local_route)
                checks.run()
                proof.update({"full_server_imported": not args.local_route,
                              "exact_source_routes_compiled": args.local_route,
                              "lifespan_startup_executed": not args.local_route,
                              "actual_loopback_http": True, "claims_enabled": False,
                              "models_read_only": True if not args.local_route else None,
                              "checks": checks.checks, "checks_run": len(checks.checks),
                              "certificate_transport_calls": tokens.certificate_calls,
                              "identity_write_attempts": fixture.identity_writes,
                              "training_state_reads": fixture.training_reads,
                              "training_write_attempts": fixture.training_writes,
                              "synthetic_storage_listing_reads": fixture.storage_list_reads,
                              "unexpected_rest_calls": fixture.unexpected_calls,
                              "baseline_unsafe_policy_expected": args.policy == "baseline"})
                proof["passed"] = all(c["passed"] for c in checks.checks)
                api.should_exit = True
                api_thread.join(timeout=30)
                require(not api_thread.is_alive(), "own_api_thread_shutdown_failed")
                if not args.local_route:
                    require({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in model_files} == model_hashes,
                            "model_files_changed")
                    proof["model_file_sha_unchanged"] = True
                exit_code = 0 if proof["passed"] else 1
    except SafeFailure as exc:
        proof.update({"passed": False, "category": str(exc)})
    except ImportError:
        proof.update({"passed": False, "category": "candidate_dependency_unavailable"})
    except Exception:
        proof.update({"passed": False, "category": "harness_environment_error"})
    finally:
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            if api:
                api.should_exit = True
            if api_thread:
                api_thread.join(timeout=15)
                if api_thread.is_alive():
                    proof.update({"passed": False, "cleanup_category": "own_api_thread_not_stopped"})
                    exit_code = 1
            if previous_request_factory is not None:
                server.google_requests.Request = previous_request_factory
            if fixture_service:
                fixture_service.shutdown()
                fixture_service.server_close()
            if fixture_thread:
                fixture_thread.join(timeout=5)
            if bound:
                bound.close()
        captured.close()  # Captured startup/error details are never emitted.
        proof["duration_seconds"] = round(time.monotonic() - started, 3)
        proof["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        if args.evidence:
            args.evidence.parent.mkdir(parents=True, exist_ok=True)
            args.evidence.write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(proof, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
