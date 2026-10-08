"""Offline checks of the security overlay; no Docker, identity or DB operations."""
import ast
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock


REPO = Path(__file__).resolve().parents[1]
PATH = REPO / "deploy-vps" / "ops_google_security_candidate.py"
SPEC = importlib.util.spec_from_file_location("google_security_candidate", PATH)
candidate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(candidate)


class CandidateBoundaries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prepared = candidate.prepared_sources(REPO)
        cls.baseline = subprocess.check_output(["git", "show", "10b54e7:server.py"], cwd=REPO)
        cls.image = "sha256:" + "a" * 64

    def test_exact_live_auth_function_guard_matches_previous_verified_baseline(self):
        node = candidate.auth_node(ast.parse(self.baseline.decode()))
        self.assertEqual(candidate.sha(candidate.segment(self.baseline, node)),
                         candidate.BASELINE_AUTH_SHA)

    def test_non_auth_bytes_unicode_comments_and_other_functions_are_preserved(self):
        before = b"# PRIVATE_TEST_ONLY: \xd0\xb5\xd0\xbb\xd1\x8c\n"
        after = b"\n# PRIVATE_TEST_ONLY_end\nasync def synthetic_other():\n    return 713\n"
        baseline = before + self.baseline + after
        transformed, preserved = candidate.transform(baseline, candidate.sha(baseline), self.prepared)
        old_tree, new_tree = ast.parse(baseline.decode()), ast.parse(transformed.decode())
        old_auth, new_auth = candidate.auth_node(old_tree), candidate.auth_node(new_tree)
        old_start, old_end = candidate.span(baseline, old_auth)
        new_start, new_end = candidate.span(transformed, new_auth)
        policy = next(n for n in new_tree.body if isinstance(n, ast.ImportFrom)
                      and n.module == "auth_google_identity")
        flag = next(n for n in new_tree.body if candidate.is_flag(n))
        policy_start, _ = candidate.span(transformed, policy)
        _, flag_end = candidate.span(transformed, flag)
        # Revert exactly the two authorized edits, including their introduced
        # newlines. Every remaining byte must be the independently supplied live
        # baseline, even if it differs from today's full repository server.py.
        reverted = (transformed[:policy_start - 1] + transformed[flag_end + 1:new_start] +
                    baseline[old_start:old_end] + transformed[new_end:])
        self.assertEqual(reverted, baseline)
        self.assertEqual(transformed[:len(before)], before)
        self.assertEqual(transformed[-len(after):], after)
        self.assertEqual(preserved, candidate.sha(baseline[:old_start] + baseline[old_end:]))

    def test_whole_baseline_digest_and_handler_tamper_are_independent_guards(self):
        with self.assertRaisesRegex(ValueError, "baseline_digest_mismatch"):
            candidate.transform(self.baseline + b"\n", candidate.sha(self.baseline), self.prepared)
        altered = self.baseline.replace(b"async def auth_google(payload: AuthGoogleRequest):",
                                       b"async def auth_google(payload: AuthGoogleRequest):\n    pass")
        with self.assertRaisesRegex(ValueError, "unexpected_baseline_auth_digest"):
            candidate.transform(altered, candidate.sha(altered), self.prepared)

    def test_prepared_policy_or_handler_change_is_rejected(self):
        for name in candidate.PREPARED_SHA:
            changed = dict(self.prepared)
            changed[name] += b"\n# modified\n"
            with self.subTest(source=name), self.assertRaisesRegex(
                    ValueError, "prepared_source_digest_mismatch"):
                candidate.transform(self.baseline, candidate.sha(self.baseline), changed)

    def test_duplicate_auth_route_and_duplicate_config_anchor_are_rejected(self):
        auth = candidate.segment(self.baseline, candidate.auth_node(ast.parse(self.baseline.decode())))
        variants = (self.baseline + b"\n" + auth,
                    self.baseline + b'\n@app.post("/auth/google")\nasync def other(payload):\n    pass\n',
                    self.baseline + b"\nfrom config import settings\n")
        for source in variants:
            with self.subTest(case=candidate.sha(source)), self.assertRaises(ValueError):
                candidate.transform(source, candidate.sha(source), self.prepared)

    def test_already_fixed_policy_or_claims_flag_is_rejected(self):
        for extra in (b"\nfrom auth_google_identity import GoogleIdentityError\n",
                      b"\nGOOGLE_IDENTITY_CLAIMS_ENABLED = True\n"):
            source = self.baseline + extra
            with self.subTest(case=candidate.sha(extra)), self.assertRaises(ValueError):
                candidate.transform(source, candidate.sha(source), self.prepared)
        fixed = self.prepared["server.py"]
        with self.assertRaises(ValueError):
            candidate.transform(fixed, candidate.sha(fixed), self.prepared)

    def test_git_objects_pinned_and_modified_checkout_refused(self):
        with mock.patch.object(candidate.subprocess, "run", return_value=
                subprocess.CompletedProcess([], 0, b"changed", b"")):
            with self.assertRaisesRegex(ValueError, "prepared_git_object_digest_mismatch"):
                candidate.prepared_sources(REPO)
        with mock.patch.object(candidate, "read_source", return_value=b"modified checkout"):
            with self.assertRaisesRegex(ValueError, "prepared_checkout_modified"):
                candidate.prepared_sources(REPO)
        # Windows Git checkout CRLF is representation, not a policy/source change.
        with mock.patch.object(candidate, "read_source", side_effect=[
                self.prepared[name].replace(b"\n", b"\r\n") for name in candidate.PREPARED_SHA]):
            self.assertEqual(candidate.prepared_sources(REPO), self.prepared)

    def test_minimal_context_manifest_and_dockerfile_inherit_runtime(self):
        with tempfile.TemporaryDirectory() as parent:
            parent = Path(parent).resolve()
            baseline = parent / "baseline.private.py"
            baseline.write_bytes(self.baseline + b"\n# PRIVATE_TEST_ONLY\n")
            output = parent / "candidate"
            manifest = candidate.prepare(REPO, baseline, candidate.sha(baseline.read_bytes()),
                                         self.image, output)
            self.assertEqual(set(p.name for p in output.iterdir()),
                             {"server.py", "auth_google_identity.py", "Dockerfile", "provenance.json"})
            self.assertEqual(json.loads((output / "provenance.json").read_text()), manifest)
            self.assertFalse(manifest["base_tag_mapping_verified_by_this_tool"])
            self.assertFalse(manifest["production_mutations"])
            self.assertFalse(manifest["identity_claims_default_enabled"])
            self.assertNotIn("PRIVATE_TEST_ONLY", json.dumps(manifest))
            self.assertNotIn(str(parent), json.dumps(manifest))
            for name, digest in manifest["file_sha256"].items():
                self.assertEqual(candidate.sha((output / name).read_bytes()), digest)
            commands = [line.split()[0] for line in (output / "Dockerfile").read_text().splitlines()
                        if line and not line.startswith("#")]
            self.assertEqual(commands, ["FROM", "COPY", "ENV"])
            self.assertIn(candidate.base_tag(self.image), (output / "Dockerfile").read_text())
            self.assertIn("IDENTITY_CLAIMS_ENABLED=false", (output / "Dockerfile").read_text())

    def test_output_overwrite_relative_inside_repository_and_parent_escape_refused(self):
        with tempfile.TemporaryDirectory() as parent:
            parent = Path(parent).resolve()
            baseline = parent / "baseline.private.py"
            baseline.write_bytes(self.baseline)
            existing = parent / "existing"
            existing.mkdir()
            marker = existing / "keep"
            marker.write_bytes(b"unchanged")
            outputs = (existing, Path("relative"), REPO / "output-candidate-never-created",
                       parent / "child" / ".." / "escape")
            for output in outputs:
                with self.subTest(case=str(output)), self.assertRaises(ValueError):
                    candidate.prepare(REPO, baseline, candidate.sha(self.baseline), self.image, output)
            self.assertEqual(marker.read_bytes(), b"unchanged")

    def test_symlink_input_and_output_parent_refused(self):
        with tempfile.TemporaryDirectory() as parent:
            parent = Path(parent).resolve()
            real = parent / "real"
            real.mkdir()
            source = real / "baseline.py"
            source.write_bytes(self.baseline)
            link = parent / "linked"
            try:
                link.symlink_to(real, target_is_directory=True)
            except OSError:
                self.skipTest("OS does not grant symlink creation for this test process")
            for baseline, output in ((link / "baseline.py", parent / "candidate"),
                                     (source, link / "candidate")):
                with self.assertRaisesRegex(ValueError, "symlink_or_junction_not_allowed"):
                    candidate.prepare(REPO, baseline, candidate.sha(self.baseline), self.image, output)
            self.assertFalse((real / "candidate").exists())

    def test_docker_image_id_cannot_inject_instruction_or_pull_registry_image(self):
        for image in ("latest", "docker.io/server:latest", "sha256:" + "a" * 64 + "\nRUN true",
                      "sha256:" + "A" * 64, "sha256:" + "a" * 63, "-V"):
            with self.subTest(case=image), self.assertRaises(ValueError):
                candidate.base_tag(image)

    def test_cli_does_not_expose_source_private_path_or_exception_content(self):
        args = [str(PATH), "--repo", str(REPO), "--baseline", "private",
                "--baseline-sha256", "a" * 64, "--base-image-id", self.image,
                "--output", "output"]
        error = io.StringIO()
        with mock.patch("sys.argv", args), mock.patch.object(candidate, "prepare",
                side_effect=ValueError("PRIVATE_TEST_ONLY source/path/token")), \
                contextlib.redirect_stderr(error):
            with self.assertRaises(SystemExit) as raised:
                candidate.main()
            self.assertEqual(raised.exception.code, 1)
        self.assertEqual(error.getvalue(), "candidate_preparation_refused\n")


if __name__ == "__main__":
    unittest.main()
