"""Security regressions on synthetic identities, never production user data.

The exact auth_google route is compiled without importing ML models or runtime
configuration. Only Google's verifier and the database transport are substituted;
these tests prove the server policy, not a real Google sign-in or SQL migration.
"""

import ast
import asyncio
import copy
from datetime import datetime
import hashlib
from pathlib import Path
import secrets
import types
import unittest
from uuid import uuid4

from auth_google_identity import (
    GoogleIdentityError, resolve_google_user, verified_google_identity,
)


def claims(**changes):
    return {"sub": "subject_a", "email": "owner@gmail.com", "email_verified": True,
            "name": "Verified owner", "picture": "https://example.invalid/avatar",
            **changes}


def row(**changes):
    return {"id": "owner-a", "email": "owner@gmail.com", "name": "Existing",
            "google_sub": "subject_a", "role": "user", "password_hash": "old",
            "salt": "old", **changes}


class Database:
    """Model unique-subject/unique-email prerequisites for guarded claims."""

    def __init__(self, rows=()):
        self.rows = copy.deepcopy(list(rows))
        self.calls = []
        self.before_patch = None
        self.before_insert = None

    def matches(self, item, params):
        for key, value in params.items():
            if key in {"select", "limit"}:
                continue
            if value == "is.null":
                if item.get(key) is not None:
                    return False
            elif item.get(key) != value.removeprefix("eq."):
                return False
        return True

    def __call__(self, method, path, *, params=None, json_body=None, **kwargs):
        assert path == "users"
        self.calls.append((method, copy.deepcopy(params), copy.deepcopy(json_body)))
        if method == "GET":
            selected = [r for r in self.rows if self.matches(r, params)]
            return copy.deepcopy(selected[:int(params["limit"])])
        if method == "PATCH":
            if self.before_patch:
                callback, self.before_patch = self.before_patch, None
                callback()
            selected = [r for r in self.rows if self.matches(r, params)]
            for r in selected:
                r.update(json_body)
            return copy.deepcopy(selected)
        if method == "POST":
            if self.before_insert:
                callback, self.before_insert = self.before_insert, None
                callback(json_body)
            if any(r["email"] == json_body["email"] or (
                    json_body.get("google_sub") and
                    r.get("google_sub") == json_body["google_sub"])
                    for r in self.rows):
                raise RuntimeError("synthetic unique constraint")
            self.rows.append(copy.deepcopy(json_body))
            return [copy.deepcopy(json_body)]
        raise AssertionError(method)

    @property
    def writes(self):
        return [c for c in self.calls if c[0] != "GET"]


class SafeHTTPException(Exception):
    def __init__(self, status_code, detail, **kwargs):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


def route(db, info, *, enable=False, verifier_error=None):
    source = Path(__file__).resolve().parents[1] / "server.py"
    selected = []
    for node in ast.parse(source.read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in {
                "auth_google", "_user_public", "_now_iso", "_hash_password"}:
            node.decorator_list = []
            selected.append(node)
    verified = []
    issued = []

    def verify(token, transport, audience):
        verified.append(audience)
        if verifier_error:
            raise verifier_error
        return info

    def session(owner):
        issued.append(owner)
        return {"token": "synthetic-session", "expires_at": "synthetic-expiry"}

    namespace = {
        "AuthGoogleRequest": object, "SUPABASE_DB_BASE": "synthetic-only",
        "GOOGLE_CLIENT_ID": "web-client.apps.googleusercontent.com",
        "GOOGLE_IDENTITY_CLAIMS_ENABLED": enable,
        "google_id_token": types.SimpleNamespace(verify_oauth2_token=verify),
        "google_requests": types.SimpleNamespace(Request=lambda: object()),
        "HTTPException": SafeHTTPException, "datetime": datetime,
        "hashlib": hashlib, "secrets": secrets, "uuid4": uuid4,
        "GoogleIdentityError": GoogleIdentityError,
        "verified_google_identity": verified_google_identity,
        "resolve_google_user": resolve_google_user,
        "_db_json": db, "_create_session": session,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["auth_google"], verified, issued


class GoogleIdentityTests(unittest.TestCase):
    def resolve(self, db, info=None, enable=False):
        identity = verified_google_identity(info or claims())
        return resolve_google_user(db, lambda _: self.fail("unexpected creation"),
                                   identity, allow_identity_claims=enable,
                                   updated_at="synthetic-time")

    def test_existing_stable_subject_keeps_owner_history_password_and_role(self):
        db = Database([row(role="admin")])
        before = copy.deepcopy(db.rows)
        owner = self.resolve(db)
        self.assertEqual(owner["id"], "owner-a")
        self.assertEqual(owner["role"], "admin")
        self.assertEqual(db.rows, before)
        self.assertEqual(db.writes, [])

    def test_claims_are_disabled_for_new_or_unlinked_profiles(self):
        for rows in ([], [row(google_sub=None)]):
            with self.subTest(rows=bool(rows)):
                db = Database(rows)
                with self.assertRaises(GoogleIdentityError) as err:
                    self.resolve(db)
                self.assertEqual(err.exception.status_code, 503)
                self.assertEqual(db.writes, [])

    def test_verified_gmail_links_password_profile_without_changing_owner(self):
        db = Database([row(google_sub=None, role="admin")])
        owner = self.resolve(db, enable=True)
        self.assertEqual(owner["id"], "owner-a")
        self.assertEqual(owner["role"], "admin")
        self.assertEqual(owner["password_hash"], "old")
        self.assertEqual(owner["salt"], "old")
        method, params, body = db.writes[0]
        self.assertEqual(method, "PATCH")
        self.assertEqual(params["google_sub"], "is.null")
        self.assertEqual(params["email"], "eq.owner@gmail.com")
        self.assertFalse(set(body) & {"id", "role", "email", "password_hash", "salt"})

    def test_empty_legacy_subject_uses_exact_empty_compare_and_set(self):
        db = Database([row(google_sub="")])
        self.resolve(db, enable=True)
        self.assertEqual(db.writes[0][1]["google_sub"], "eq.")

    def test_workspace_authoritative_link_and_non_google_email_rejection(self):
        db = Database([row(email="owner@example.invalid", google_sub=None)])
        info = claims(email="owner@example.invalid", hd="example.invalid")
        self.assertEqual(self.resolve(db, info, enable=True)["id"], "owner-a")
        db = Database([row(email="owner@example.invalid", google_sub=None)])
        with self.assertRaises(GoogleIdentityError) as err:
            self.resolve(db, claims(email="owner@example.invalid"), enable=True)
        self.assertEqual(err.exception.status_code, 409)
        self.assertEqual(db.writes, [])

    def test_google_email_change_preserves_original_owner_without_email_reassignment(self):
        db = Database([row(email="previous@gmail.com")])
        owner = self.resolve(db)
        self.assertEqual(owner["email"], "previous@gmail.com")
        self.assertEqual(db.writes, [])

    def test_other_linked_subject_and_ambiguous_owner_are_rejected(self):
        for rows in ([row(google_sub="subject_b")],
                     [row(email="previous@gmail.com"), row(id="owner-b", google_sub="subject_b")],
                     [row(), row(id="owner-b", email="another@gmail.com")]):
            with self.subTest(count=len(rows)):
                db = Database(rows)
                with self.assertRaises(GoogleIdentityError) as err:
                    self.resolve(db, enable=True)
                self.assertEqual(err.exception.status_code, 409)
                self.assertEqual(db.writes, [])

    def test_concurrent_other_subject_is_never_overwritten(self):
        db = Database([row(google_sub=None)])
        db.before_patch = lambda: db.rows[0].update(google_sub="subject_b")
        with self.assertRaises(GoogleIdentityError) as err:
            self.resolve(db, enable=True)
        self.assertEqual(err.exception.status_code, 409)
        self.assertEqual(db.rows[0]["google_sub"], "subject_b")
        self.assertEqual(len(db.writes), 1)

    def test_concurrent_same_subject_claim_returns_same_owner_without_second_write(self):
        db = Database([row(google_sub=None)])
        db.before_patch = lambda: db.rows[0].update(google_sub="subject_a")
        self.assertEqual(self.resolve(db, enable=True)["id"], "owner-a")
        self.assertEqual(len(db.writes), 1)

    def test_missing_unverified_and_malformed_claims_are_rejected_before_db(self):
        for changes in ({"email": None}, {"sub": None}, {"sub": "subject_a),role.eq.admin"},
                        {"email_verified": False}, {"email_verified": None},
                        {"email_verified": "true"}, {"email": "a@gmail.com),role.eq.admin"}):
            with self.subTest(changes=changes):
                db = Database([row()])
                with self.assertRaises(GoogleIdentityError) as err:
                    self.resolve(db, claims(**changes), enable=True)
                self.assertEqual(err.exception.status_code, 401)
                self.assertEqual(db.calls, [])

    def test_malformed_verified_payload_is_rejected(self):
        for info in (None, [], "malformed"):
            with self.subTest(type=type(info).__name__):
                with self.assertRaises(GoogleIdentityError) as err:
                    verified_google_identity(info)
                self.assertEqual(err.exception.status_code, 401)

    def test_route_ignores_spoofed_client_identity_and_role(self):
        db = Database([row()])
        fn, verified, issued = route(db, claims())
        payload = types.SimpleNamespace(id_token="synthetic-id-token", email="another@gmail.com",
                                        name="Spoofed", photo_url="https://example.invalid/spoof",
                                        role="admin")
        result = asyncio.run(fn(payload))
        self.assertEqual(result["user"]["id"], "owner-a")
        self.assertEqual(result["user"]["role"], "user")
        self.assertEqual(result["user"]["name"], "Existing")
        self.assertEqual(verified, ["web-client.apps.googleusercontent.com"])
        self.assertEqual(issued, ["owner-a"])
        self.assertEqual(db.writes, [])

    def test_route_missing_claim_email_cannot_fall_back_to_client_email(self):
        db = Database([row(role="admin")])
        fn, _, issued = route(db, claims(email=None))
        payload = types.SimpleNamespace(id_token="synthetic", email="owner@gmail.com",
                                        name="Spoofed", photo_url=None)
        with self.assertRaises(SafeHTTPException) as err:
            asyncio.run(fn(payload))
        self.assertEqual(err.exception.status_code, 401)
        self.assertEqual(issued, [])
        self.assertEqual(db.calls, [])

    def test_route_verifier_failure_is_safe_and_performs_no_db_work(self):
        db = Database([row()])
        fn, verified, issued = route(db, claims(), verifier_error=ValueError("private-token-detail"))
        with self.assertRaises(SafeHTTPException) as err:
            asyncio.run(fn(types.SimpleNamespace(id_token="synthetic")))
        self.assertEqual(err.exception.status_code, 401)
        self.assertNotIn("private-token-detail", err.exception.detail)
        self.assertEqual(verified, ["web-client.apps.googleusercontent.com"])
        self.assertEqual(issued, [])
        self.assertEqual(db.calls, [])

    def test_new_profile_role_is_server_assigned_and_identity_comes_from_claims(self):
        db = Database()
        fn, _, issued = route(db, claims(), enable=True)
        result = asyncio.run(fn(types.SimpleNamespace(
            id_token="synthetic", role="admin", email="spoofed@gmail.com")))
        self.assertEqual(len(db.rows), 1)
        self.assertEqual(db.rows[0]["role"], "user")
        self.assertEqual(db.rows[0]["email"], "owner@gmail.com")
        self.assertEqual(db.rows[0]["google_sub"], "subject_a")
        self.assertEqual(issued, [result["user"]["id"]])

    def test_same_identity_insert_race_reuses_unique_winner(self):
        db = Database()
        db.before_insert = lambda _: db.rows.append(row())
        fn, _, issued = route(db, claims(), enable=True)
        result = asyncio.run(fn(types.SimpleNamespace(id_token="synthetic")))
        self.assertEqual(result["user"]["id"], "owner-a")
        self.assertEqual(len(db.rows), 1)
        self.assertEqual(issued, ["owner-a"])

    def test_other_identity_email_insert_race_never_returns_its_session(self):
        db = Database()
        db.before_insert = lambda _: db.rows.append(row(google_sub="subject_b"))
        fn, _, issued = route(db, claims(), enable=True)
        with self.assertRaises(SafeHTTPException) as err:
            asyncio.run(fn(types.SimpleNamespace(id_token="synthetic")))
        self.assertEqual(err.exception.status_code, 503)
        self.assertEqual(db.rows[0]["google_sub"], "subject_b")
        self.assertEqual(issued, [])


if __name__ == "__main__":
    unittest.main()
