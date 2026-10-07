"""No-Docker regression tests for the synthetic PG runner's transport boundaries."""
import contextlib
import importlib.util
import io
import json
import subprocess
import unittest
from pathlib import Path
from unittest import mock


_path = Path(__file__).with_name("google_identity_unique_index_postgres.py")
_spec = importlib.util.spec_from_file_location("google_index_runner", _path)
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)


class RunnerBoundaries(unittest.TestCase):
    def sandbox(self):
        return runner.PgSandbox(None)

    def identity(self, sandbox):
        return {"Id": "a" * 64, "Name": "/" + sandbox.name, "Image": runner.IMAGE,
                "Config": {"Labels": {runner.SCOPE: "1"}},
                "HostConfig": {"NetworkMode": "none"}}

    def response(self, item):
        return subprocess.CompletedProcess([], 0, json.dumps([item]), "")

    def test_ssh_host_options_separators_credentials_and_overlong_values_denied(self):
        for value in ("-V", "--", "-oProxyCommand", "host;command", "host path",
                      "postgres://user:password@host/database", "a" * 256, ""):
            with self.subTest(input_class="unsafe_host"):
                self.assertFalse(runner.valid_ssh_host(value))
        self.assertTrue(runner.valid_ssh_host("synthetic-user@test-host.invalid"))

    def test_cli_option_host_denied_before_docker_or_file_read(self):
        with mock.patch("sys.argv", [str(_path), "--docker-ssh-host=-V",
                                     "--evidence", "unused.json"]), \
             mock.patch.object(runner, "PgSandbox") as sandbox, \
             mock.patch.object(Path, "read_bytes") as read, \
             contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                runner.main()
            self.assertEqual(error.exception.code, 2)
            sandbox.assert_not_called()
            read.assert_not_called()

    def test_lost_run_stdout_recovers_exact_owned_name_before_removal(self):
        sandbox = self.sandbox()
        identity = self.identity(sandbox)
        sandbox.call = mock.Mock(side_effect=[self.response(identity),
                                              subprocess.CompletedProcess([], 0, "", "")])
        sandbox.cleanup()
        self.assertEqual(sandbox.call.call_args_list,
                         [mock.call(["inspect", sandbox.name]), mock.call(["rm", "-f", "a" * 64])])
        self.assertTrue(sandbox.removed)

    def test_unknown_id_cleanup_refuses_any_ownership_guard_mismatch(self):
        for key in ("Id", "Name", "Image", "Labels", "NetworkMode"):
            with self.subTest(guard=key):
                sandbox = self.sandbox()
                item = self.identity(sandbox)
                if key == "Labels": item["Config"][key] = {}
                elif key == "NetworkMode": item["HostConfig"][key] = "host"
                else: item[key] = "mismatch"
                sandbox.call = mock.Mock(return_value=self.response(item))
                with self.assertRaisesRegex(runner.SandboxFailure, "identity_mismatch"):
                    sandbox.cleanup()
                self.assertEqual(sandbox.call.call_count, 1)
                self.assertFalse(sandbox.removed)

    def test_known_id_cleanup_never_removes_another_id(self):
        sandbox = self.sandbox()
        sandbox.container_id = "b" * 64
        sandbox.call = mock.Mock(return_value=self.response(self.identity(sandbox)))
        with self.assertRaisesRegex(runner.SandboxFailure, "identity_mismatch"):
            sandbox.cleanup()
        sandbox.call.assert_called_once_with(["inspect", "b" * 64])

    def test_missing_exact_owned_name_is_a_safe_noop(self):
        sandbox = self.sandbox()
        sandbox.call = mock.Mock(return_value=subprocess.CompletedProcess(
            [], 1, "", "Error: No such object: " + sandbox.name + "\n"))
        sandbox.cleanup()
        sandbox.call.assert_called_once_with(["inspect", sandbox.name])
        self.assertFalse(sandbox.removed)

    def test_transport_failure_is_not_mistaken_for_absent_owned_container(self):
        sandbox = self.sandbox()
        sandbox.call = mock.Mock(return_value=subprocess.CompletedProcess([], 255, "", "closed"))
        with self.assertRaisesRegex(runner.SandboxFailure, "identity_unavailable"):
            sandbox.cleanup()
        self.assertEqual(sandbox.call.call_count, 1)

    def test_unexpected_lookup_output_does_not_trigger_removal(self):
        sandbox = self.sandbox()
        sandbox.call = mock.Mock(return_value=subprocess.CompletedProcess([], 0, "not-json", ""))
        with self.assertRaisesRegex(runner.SandboxFailure, "identity_mismatch"):
            sandbox.cleanup()
        self.assertEqual(sandbox.call.call_count, 1)


if __name__ == "__main__":
    unittest.main()
