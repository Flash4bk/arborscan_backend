"""Real PostgreSQL integration checks using an isolated, synthetic-only Docker DB.

No production connection or credentials are accepted. The SSH option runs Docker
on an existing authorized laboratory host; it does not connect to that host's DB.
Raw SQL output is deliberately kept out of stdout and the public evidence file.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import shlex
import subprocess
import time
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path


IMAGE = "sha256:178f0976b54a39237096bfa310c1a352dbc82fb1b08dda45cdb8acb5d40c1426"
SCOPE = "arborscan-google-index-synthetic"


def valid_ssh_host(value):
    return (isinstance(value, str) and 0 < len(value) <= 255
            and not value.startswith("-")
            and re.fullmatch(r"[A-Za-z0-9_.@-]+", value) is not None)


class SandboxFailure(RuntimeError):
    """Only fixed stage names, never SQL output, are exposed to test reports."""


class PgSandbox:
    def __init__(self, ssh_host: str | None):
        self.ssh_host = ssh_host
        self.name = "arborscan-google-index-test-" + uuid.uuid4().hex
        self.container_id = None
        self.removed = False

    def command(self, arguments):
        if self.ssh_host:
            return ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
                    "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=20",
                    "-o", "ServerAliveCountMax=3", self.ssh_host,
                    shlex.join(["docker", *arguments])]
        return ["docker", *arguments]

    def call(self, arguments, sql=None, timeout=90):
        try:
            return subprocess.run(self.command(arguments), input=sql, text=True,
                                  capture_output=True, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SandboxFailure("docker_or_transport_unavailable") from exc

    def start(self):
        bootstrap = ("initdb -D /tmp/pgdata -A trust --no-locale -E UTF8 "
                     ">/tmp/init.log 2>&1 && exec postgres -D /tmp/pgdata "
                     "-c listen_addresses='' -c unix_socket_directories=/tmp "
                     "-c max_connections=12 -c shared_buffers=32MB "
                     ">/tmp/postgres.log 2>&1")
        result = self.call(["run", "-d", "--name", self.name, "--label", SCOPE + "=1",
                            "--network", "none", "--read-only", "--user", "postgres",
                            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                            "--pids-limit", "64", "--memory", "512m", "--cpus", "1",
                            "--tmpfs", "/tmp:rw,nosuid,nodev,noexec,mode=1777,size=256m",
                            "--entrypoint", "/bin/sh", IMAGE, "-c", bootstrap])
        candidate = result.stdout.strip()
        if result.returncode or not re.fullmatch(r"[0-9a-f]{64}", candidate):
            raise SandboxFailure("synthetic_container_start_failed")
        self.container_id = candidate
        for _ in range(30):
            query = self.sql("SELECT 1;", timeout=30)
            if query.returncode == 0 and query.stdout.strip() == "1":
                return
            time.sleep(0.25)
        raise SandboxFailure("synthetic_database_start_failed")

    def psql_arguments(self):
        if not self.container_id:
            raise SandboxFailure("synthetic_container_not_started")
        return ["exec", "-i", self.container_id, "psql", "-X", "-qAt", "-h", "/tmp",
                "-U", "postgres", "-d", "postgres", "-v", "ON_ERROR_STOP=1",
                "-v", "VERBOSITY=sqlstate"]

    def sql(self, text, timeout=90):
        return self.call(self.psql_arguments(), text, timeout)

    def successful(self, text):
        result = self.sql(text)
        if result.returncode:
            raise SandboxFailure("synthetic_sql_unexpected_failure")
        return result.stdout.strip()

    def reset(self, column="text"):
        if column not in {"text", "varchar(255)"}:
            raise SandboxFailure("unsupported_synthetic_type")
        self.successful("DROP TABLE IF EXISTS public.users CASCADE; "
                        "DROP TABLE IF EXISTS public.users_google_sub_unique; "
                        f"CREATE TABLE public.users(id integer PRIMARY KEY, google_sub {column});")

    def cleanup(self):
        # A successful remote docker run can lose its stdout when SSH closes.
        # Look up exactly this run's UUID name, never a prefix or a broad list.
        checked = self.call(["inspect", self.container_id or self.name])
        if checked.returncode:
            not_found = {"Error: No such object: " + self.name,
                         "Error: No such container: " + self.name}
            if (not self.container_id and checked.returncode == 1
                    and checked.stderr.strip() in not_found):
                return
            raise SandboxFailure("own_container_cleanup_identity_unavailable")
        try:
            item, = json.loads(checked.stdout)
            candidate_id = item["Id"]
            identity_ok = (re.fullmatch(r"[0-9a-f]{64}", candidate_id) is not None
                           and (not self.container_id or candidate_id == self.container_id)
                           and item["Name"] == "/" + self.name
                           and item["Image"] == IMAGE
                           and item["Config"]["Labels"].get(SCOPE) == "1"
                           and item["HostConfig"]["NetworkMode"] == "none")
        except (ValueError, TypeError, KeyError):
            identity_ok = False
        if not identity_ok:
            raise SandboxFailure("own_container_cleanup_identity_mismatch")
        self.container_id = candidate_id
        if self.call(["rm", "-f", self.container_id]).returncode:
            raise SandboxFailure("own_container_cleanup_failed")
        self.removed = True


class UniqueIndexChecks(unittest.TestCase):
    sandbox: PgSandbox
    migration: str

    def setUp(self):
        self.sandbox.reset()

    def migrate(self):
        return self.sandbox.sql(self.migration)

    def assert_denied(self, result, state="P0001"):
        self.assertNotEqual(result.returncode, 0)
        self.assertRegex(result.stderr, r"\b" + re.escape(state) + r"\b")

    def index_state(self):
        return self.sandbox.successful(
            "SELECT indisunique AND indisvalid AND indisready FROM pg_index "
            "WHERE indexrelid=to_regclass('public.users_google_sub_unique');")

    def test_first_run_and_repeat_keep_same_ready_index(self):
        self.sandbox.successful("INSERT INTO users VALUES(1,'synthetic_a'),(2,'synthetic_b');")
        self.assertEqual(self.migrate().returncode, 0)
        oid = self.sandbox.successful("SELECT 'public.users_google_sub_unique'::regclass::oid;")
        self.assertEqual(self.index_state(), "t")
        self.assertEqual(self.migrate().returncode, 0)
        self.assertEqual(self.sandbox.successful(
            "SELECT 'public.users_google_sub_unique'::regclass::oid;"), oid)
        self.assertEqual(self.sandbox.successful("SELECT count(*) FROM users;"), "2")

    def test_null_and_empty_legacy_values_remain_allowed(self):
        self.sandbox.successful("INSERT INTO users VALUES(1,NULL),(2,NULL),(3,''),(4,'');")
        self.assertEqual(self.migrate().returncode, 0)
        self.sandbox.successful("INSERT INTO users VALUES(5,NULL),(6,'');")
        self.assertEqual(self.sandbox.successful("SELECT count(*) FROM users;"), "6")

    def test_duplicate_subjects_abort_before_index_creation(self):
        self.sandbox.successful("INSERT INTO users VALUES(1,'synthetic_a'),(2,'synthetic_a');")
        self.assert_denied(self.migrate())
        self.assertEqual(self.sandbox.successful(
            "SELECT to_regclass('public.users_google_sub_unique') IS NULL;"), "t")
        self.assertEqual(self.sandbox.successful("SELECT count(*) FROM users;"), "2")

    def test_invalid_and_overlong_subjects_are_rejected(self):
        for expression in ("'synthetic space'", "repeat('a',256)"):
            with self.subTest(value_class="invalid_or_overlong"):
                self.sandbox.reset()
                self.sandbox.successful(f"INSERT INTO users VALUES(1,{expression});")
                self.assert_denied(self.migrate())
                self.assertEqual(self.sandbox.successful(
                    "SELECT to_regclass('public.users_google_sub_unique') IS NULL;"), "t")

    def test_nonunique_existing_index_is_rejected(self):
        self.sandbox.successful("CREATE INDEX users_google_sub_unique ON users(google_sub) "
                                "WHERE google_sub IS NOT NULL AND google_sub <> '';")
        self.assert_denied(self.migrate())
        self.assertEqual(self.sandbox.successful(
            "SELECT indisunique FROM pg_index WHERE indexrelid='users_google_sub_unique'::regclass;"), "f")

    def test_wrong_column_existing_index_is_rejected(self):
        self.sandbox.successful("CREATE UNIQUE INDEX users_google_sub_unique ON users(id) "
                                "WHERE google_sub IS NOT NULL AND google_sub <> '';")
        self.assert_denied(self.migrate())

    def test_wrong_predicate_existing_index_is_rejected(self):
        self.sandbox.successful("CREATE UNIQUE INDEX users_google_sub_unique ON users(google_sub) "
                                "WHERE google_sub IS NOT NULL;")
        self.assert_denied(self.migrate())

    def test_same_named_nonindex_relation_is_rejected(self):
        self.sandbox.successful("CREATE TABLE users_google_sub_unique(id integer);")
        self.assert_denied(self.migrate())

    def test_invalid_concurrent_index_is_rejected(self):
        self.sandbox.successful("INSERT INTO users VALUES(1,'synthetic_a'),(2,'synthetic_a');")
        self.assert_denied(self.sandbox.sql(
            "CREATE UNIQUE INDEX CONCURRENTLY users_google_sub_unique ON users(google_sub) "
            "WHERE google_sub IS NOT NULL AND google_sub <> '';"), "23505")
        self.sandbox.successful("DELETE FROM users WHERE id=2;")
        self.assertEqual(self.index_state(), "f")
        self.assert_denied(self.migrate())
        self.assertEqual(self.index_state(), "f")

    def test_duplicate_after_preflight_does_not_publish_ready_index(self):
        self.sandbox.successful("INSERT INTO users VALUES(1,'synthetic_a');")
        before, marker, after = self.migration.partition("CREATE UNIQUE INDEX CONCURRENTLY")
        self.assertTrue(marker, "concurrent_index_creation_missing")
        self.assertEqual(self.sandbox.sql(before).returncode, 0)
        self.sandbox.successful("INSERT INTO users VALUES(2,'synthetic_a');")
        self.assert_denied(self.sandbox.sql(marker + after), "23505")
        self.assertEqual(self.index_state(), "f")
        self.assertEqual(self.sandbox.successful("SELECT count(*) FROM users;"), "2")
        self.assert_denied(self.migrate())

    def test_unsupported_subject_type_is_rejected_before_creation(self):
        self.sandbox.successful("DROP TABLE users; CREATE TABLE users(id integer PRIMARY KEY, "
                                "google_sub char(255));")
        self.assert_denied(self.migrate())
        self.assertEqual(self.sandbox.successful(
            "SELECT to_regclass('public.users_google_sub_unique') IS NULL;"), "t")

    def test_transaction_wrapping_is_rejected(self):
        self.assert_denied(self.sandbox.sql("BEGIN;\n" + self.migration + "\nCOMMIT;"), "25001")
        self.assertEqual(self.sandbox.successful(
            "SELECT to_regclass('public.users_google_sub_unique') IS NULL;"), "t")

    def concurrent_conflict(self, first_sql, second_sql):
        self.assertEqual(self.migrate().returncode, 0)
        first = subprocess.Popen(self.sandbox.command(self.sandbox.psql_arguments()),
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True)
        try:
            first.stdin.write("BEGIN; " + first_sql + "; SELECT 'ready';\n")
            first.stdin.flush()
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                ready = pool.submit(first.stdout.readline)
                try:
                    readiness = ready.result(timeout=30).strip()
                except concurrent.futures.TimeoutError as exc:
                    first.kill()
                    raise SandboxFailure("first_transaction_readiness_timeout") from exc
                self.assertEqual(readiness, "ready")
                contender = pool.submit(self.sandbox.sql,
                                        "SET statement_timeout='20s'; " + second_sql + ";")
                blocked = False
                for _ in range(50):
                    blocked = self.sandbox.successful(
                        "SELECT EXISTS(SELECT 1 FROM pg_stat_activity "
                        "WHERE wait_event_type='Lock' AND wait_event='transactionid');") == "t"
                    if blocked:
                        break
                    time.sleep(0.1)
                first.stdin.write("COMMIT;\n")
                first.stdin.flush()
                first.stdin.close()
                first.stdin = None
                first.communicate(timeout=30)
                self.assertEqual(first.returncode, 0)
                second = contender.result(timeout=30)
                self.assertTrue(blocked, "contender_did_not_wait_for_first_transaction")
                self.assert_denied(second, "23505")
        finally:
            if first.poll() is None:
                first.kill()
                first.communicate(timeout=30)

    def test_concurrent_insert_same_subject_has_one_winner(self):
        self.concurrent_conflict("INSERT INTO users VALUES(1,'synthetic_race')",
                                 "INSERT INTO users VALUES(2,'synthetic_race')")
        self.assertEqual(self.sandbox.successful("SELECT count(*) FROM users;"), "1")

    def test_concurrent_claim_of_two_legacy_rows_has_one_winner(self):
        self.sandbox.successful("INSERT INTO users VALUES(1,NULL),(2,NULL);")
        self.concurrent_conflict("UPDATE users SET google_sub='synthetic_race' WHERE id=1",
                                 "UPDATE users SET google_sub='synthetic_race' WHERE id=2")
        self.assertEqual(self.sandbox.successful(
            "SELECT count(*) FROM users WHERE google_sub IS NOT NULL;"), "1")
        self.assertEqual(self.sandbox.successful("SELECT count(*) FROM users;"), "2")

    def test_case_distinct_subjects_are_not_coalesced(self):
        self.assertEqual(self.migrate().returncode, 0)
        self.sandbox.successful("INSERT INTO users VALUES(1,'synthetic_a'),(2,'Synthetic_a');")
        self.assertEqual(self.sandbox.successful("SELECT count(*) FROM users;"), "2")

    def test_varchar_schema_first_run_and_repeat(self):
        self.sandbox.reset("varchar(255)")
        self.assertEqual(self.migrate().returncode, 0)
        self.assertEqual(self.index_state(), "t")
        self.assertEqual(self.migrate().returncode, 0)


class SafeResults(unittest.TestResult):
    def __init__(self):
        super().__init__()
        self.checks = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.checks.append({"check": test._testMethodName, "passed": True})

    def addFailure(self, test, err):
        self.failures.append((test, "assertion_failed"))
        self.checks.append({"check": test._testMethodName, "passed": False,
                            "category": "assertion_failed"})

    def addError(self, test, err):
        self.errors.append((test, "test_environment_error"))
        self.checks.append({"check": test._testMethodName, "passed": False,
                            "category": "test_environment_error",
                            "exception_type": err[0].__name__,
                            "stage": str(err[1]) if isinstance(err[1], SandboxFailure) else None})

    def addSubTest(self, test, subtest, err):
        if err is not None:
            self.failures.append((test, "subtest_assertion_failed"))
            self.checks.append({"check": test._testMethodName, "passed": False,
                                "category": "subtest_assertion_failed"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docker-ssh-host", help="Existing authorized Docker sandbox host; no DB URI")
    parser.add_argument("--evidence", required=True, type=Path)
    args = parser.parse_args()
    if args.docker_ssh_host and not valid_ssh_host(args.docker_ssh_host):
        parser.error("unsupported SSH host format")
    repository = Path(__file__).resolve().parents[1]
    migration_path = repository / "deploy-vps/google-identity-unique-index.sql"
    migration_bytes = migration_path.read_bytes()
    sandbox = PgSandbox(args.docker_ssh_host)
    started = time.monotonic()
    proof = {"started_utc": datetime.now(timezone.utc).isoformat(),
             "environment": "isolated Docker PostgreSQL on existing VPS" if args.docker_ssh_host
             else "isolated local Docker PostgreSQL", "synthetic_data_only": True,
             "production_migration_executed": False, "backup_restore_repeated": False,
             "network": "none", "root_filesystem": "read-only", "database_storage": "tmpfs",
             "image_digest": IMAGE, "migration_sha256": hashlib.sha256(migration_bytes).hexdigest()}
    exit_code = 1
    try:
        sandbox.start()
        proof["server_version"] = sandbox.successful("SHOW server_version;")
        proof["server_version_num"] = sandbox.successful("SHOW server_version_num;")
        UniqueIndexChecks.sandbox = sandbox
        UniqueIndexChecks.migration = migration_bytes.decode("utf-8")
        results = SafeResults()
        unittest.defaultTestLoader.loadTestsFromTestCase(UniqueIndexChecks).run(results)
        proof["checks"] = results.checks
        proof["tests_run"] = results.testsRun
        proof["passed"] = results.wasSuccessful()
        exit_code = 0 if results.wasSuccessful() else 1
    except SandboxFailure as exc:
        proof["passed"] = False
        proof["category"] = str(exc)
    finally:
        try:
            sandbox.cleanup()
        except SandboxFailure as exc:
            proof["cleanup_category"] = str(exc)
            exit_code = 1
        proof["own_test_container_removed"] = sandbox.removed
        proof["duration_seconds"] = round(time.monotonic() - started, 3)
        proof["finished_utc"] = datetime.now(timezone.utc).isoformat()
        args.evidence.parent.mkdir(parents=True, exist_ok=True)
        args.evidence.write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(proof, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
