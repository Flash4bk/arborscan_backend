"""Signing safety with synthetic SDK output; real APK/device checks are separate."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

source = Path(__file__).resolve().parents[1] / "deploy-vps/ops_android_signing.py"
spec = importlib.util.spec_from_file_location("android_signing", source)
ops = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ops)

OLD = "a" * 64
NEW = "b" * 64
SECRET = "synthetic-private-password-must-not-appear"


class SigningSafetyTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.password = self.file("password.txt", SECRET.encode())
        self.old = self.file("old.p12", b"old synthetic key")
        self.new = self.file("new.p12", b"new synthetic key")
        self.lineage = self.file("lineage", b"synthetic rotation lineage")
        self.apk = self.file("candidate.apk", b"synthetic unsigned apk")
        self.config = {
            "format_version": 1, "application_id": "com.example.arborscan_app", "min_sdk": 24,
            "rotation_min_sdk": 33, "build_tools": str(self.root), "java_home": str(self.root),
            "lineage": str(self.lineage), "old_signer": self.signer(self.old, OLD),
            "new_signer": self.signer(self.new, NEW)}
        self.manifest = "package: name='com.example.arborscan_app' versionCode='16' versionName='release'\n" \
            "sdkVersion:'24'\ntargetSdkVersion:'36'\nuses-permission: name='android.permission.INTERNET'\n"
        self.calls = []

    def file(self, name, content):
        path = self.root / name
        path.write_bytes(content)
        path.chmod(0o600)
        return path

    def signer(self, path, certificate):
        return {"keystore": str(path), "alias": "synthetic", "sha256": certificate,
                "sha1": certificate[:40], "store_password": "file:" + str(self.password),
                "key_password": "file:" + str(self.password)}

    def runner(self, config, tool, args, operation):
        self.calls.append((tool, args, operation))
        if tool == "aapt":
            if "xmltree" in args:
                return 'A: android:targetPackage(0x1)="com.example.arborscan_app"\n' \
                       'A: android:name(0x2)="com.example.arborscan_app.DataPreservationInstrumentation"\n'
            return self.manifest
        if tool == "zipalign":
            if "-c" not in args:
                Path(args[-1]).write_bytes(Path(args[-2]).read_bytes())
            return "alignment checked"
        if args[0] == "rotate":
            Path(args[args.index("--out") + 1]).write_bytes(b"verified synthetic lineage")
            return ""
        if args[0] == "lineage":
            result = []
            for number, signer, rollback in ((1, OLD, "false"), (2, NEW, "true")):
                result.append(f"Signer #{number} in lineage certificate DN: Synthetic Public Certificate\n"
                    f"Signer #{number} in lineage certificate SHA-256 digest: {signer}\n"
                    f"Signer #{number} in lineage certificate SHA-1 digest: {signer[:40]}\n"
                    f"Has installed data capability: true\nHas rollback capability : {rollback}\n")
            return "".join(result)
        if args[0] == "sign":
            Path(args[args.index("--out") + 1]).write_bytes(b"verified synthetic signed apk")
            return ""
        minimum = int(args[args.index("--min-sdk-version") + 1])
        signer = OLD if minimum < 33 else NEW
        scheme = "v2" if minimum < 28 else "v3" if minimum < 33 else "v3.1"
        return f"Verifies\nVerified using {scheme} scheme (APK Signature Scheme): true\n" \
               f"Signer #1 certificate SHA-256 digest: {signer}\n" \
               f"Signer #1 certificate SHA-1 digest: {signer[:40]}\n"

    def save_config(self, config=None):
        return self.file("config.json", json.dumps(config or self.config).encode())

    def test_private_configuration_accepts_file_passwords_without_reading_the_secret(self):
        loaded = ops.load_configuration(self.save_config())
        self.assertEqual(NEW, loaded["new_signer"]["sha256"])
        self.assertNotIn(SECRET, json.dumps(loaded))
        self.assertNotIn(SECRET, repr(ops.signer_arguments(loaded["new_signer"])))

    def test_inline_password_and_unknown_fields_are_rejected(self):
        for change in ("inline", "unknown"):
            with self.subTest(change=change):
                config = copy.deepcopy(self.config)
                if change == "inline":
                    config["new_signer"]["store_password"] = "pass:" + SECRET
                else:
                    config["store_password"] = SECRET
                with self.assertRaises(ops.SigningError) as raised:
                    ops.load_configuration(self.save_config(config))
                self.assertNotIn(SECRET, str(raised.exception))

    def test_missing_key_env_and_in_repo_material_fail_closed(self):
        config = copy.deepcopy(self.config)
        config["new_signer"]["key_password"] = "env:ARBORSCAN_SIGNING_TEST_MISSING"
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ops.SigningError):
            ops.load_configuration(self.save_config(config))
        with self.assertRaises(ops.SigningError):
            ops.outside_repository(source)

    def test_rotation_policy_cannot_silently_drop_supported_devices(self):
        for sdk in (28, 32, 34):
            config = copy.deepcopy(self.config)
            config["rotation_min_sdk"] = sdk
            with self.subTest(sdk=sdk), self.assertRaises(ops.SigningError):
                ops.load_configuration(self.save_config(config))

    def test_other_git_checkout_is_not_private_key_storage(self):
        other_repository = self.root / "other-repository"
        other_repository.mkdir()
        (other_repository / ".git").write_text("gitdir: elsewhere")
        with self.assertRaises(ops.SigningError):
            ops.outside_repository(other_repository / "private-key.p12")

    def test_apk_effective_signer_checked_at_old_and_new_sdk_boundaries(self):
        result = ops.verify_apk(self.config, self.apk, 16, self.runner)
        self.assertEqual(["old", "old", "new", "new"],
                         [row["signer_role"] for row in result["platform_signatures"]])
        self.assertFalse(result["device_installation_verified"])
        self.assertNotIn(SECRET, json.dumps(result))
        self.assertNotIn(str(self.password), json.dumps(result))

    def test_wrong_effective_signer_fails_even_when_signature_is_valid(self):
        def wrong(config, tool, args, operation):
            return self.runner(config, tool, args, operation).replace(NEW, OLD)
        with self.assertRaisesRegex(ops.SigningError, "API 33"):
            ops.verify_apk(self.config, self.apk, 16, wrong)

    def test_wrong_sha1_for_api_restrictions_is_not_published(self):
        config = copy.deepcopy(self.config)
        config["new_signer"]["sha1"] = "c" * 40
        with self.assertRaises(ops.SigningError):
            ops.verify_apk(config, self.apk, 16, self.runner)

    def test_debuggable_missing_internet_wrong_id_sdk_and_version_rejected(self):
        changes = [self.manifest + "application-debuggable\n",
                   self.manifest.replace("uses-permission", "missing-permission"),
                   self.manifest.replace("com.example.arborscan_app", "other.application"),
                   self.manifest.replace("sdkVersion:'24'", "sdkVersion:'33'"),
                   self.manifest.replace("versionCode='16'", "versionCode='15'")]
        for manifest in changes:
            with self.subTest(manifest=manifest):
                self.manifest = manifest
                with self.assertRaises(ops.SigningError):
                    ops.verify_apk(self.config, self.apk, 16, self.runner)
                self.assertFalse(any(tool == "apksigner" for tool, _, _ in self.calls))
                self.calls.clear()

    def test_signing_publishes_only_verified_staged_apk_and_prevents_overwrite(self):
        output = self.root / "published.apk"
        metadata = ops.sign(self.config, self.apk, output, 16, self.runner)
        self.assertEqual(metadata["apk_sha256"], ops.sha256(output))
        with self.assertRaises(ops.SigningError):
            ops.sign(self.config, self.apk, output, 16, self.runner)
        arguments = next(args for tool, args, _ in self.calls if tool == "apksigner" and args[0] == "sign")
        self.assertEqual("33", arguments[arguments.index("--rotation-min-sdk-version") + 1])
        self.assertNotIn(SECRET, repr(arguments))

    def test_signing_validation_failure_does_not_publish_apk(self):
        output = self.root / "refused.apk"
        def wrong(config, tool, args, operation):
            return self.runner(config, tool, args, operation).replace(NEW, OLD)
        with self.assertRaises(ops.SigningError):
            ops.sign(self.config, self.apk, output, 16, wrong)
        self.assertFalse(output.exists())

    def test_interrupted_signing_does_not_publish_apk(self):
        output = self.root / "interrupted.apk"
        def interrupted(config, tool, args, operation):
            if tool == "apksigner" and args[0] == "sign":
                raise ops.SigningError("Synthetic interruption")
            return self.runner(config, tool, args, operation)
        with self.assertRaises(ops.SigningError):
            ops.sign(self.config, self.apk, output, 16, interrupted)
        self.assertFalse(output.exists())

    def test_lineage_not_overwritten_and_old_key_rollback_disabled(self):
        with self.assertRaises(ops.SigningError):
            ops.rotate(self.config, self.runner)
        config = copy.deepcopy(self.config)
        config["lineage"] = str(self.root / "new-lineage")
        ops.rotate(config, self.runner)
        arguments = next(args for tool, args, _ in self.calls if tool == "apksigner" and args[0] == "rotate")
        self.assertEqual("true", arguments[arguments.index("--set-installed-data") + 1])
        self.assertEqual("false", arguments[arguments.index("--set-rollback") + 1])
        self.assertTrue(Path(config["lineage"]).is_file())

    def test_tool_error_does_not_expose_command_password_or_stderr(self):
        tool = self.file("apksigner.bat" if os.name == "nt" else "apksigner", b"")
        completed = subprocess.CompletedProcess([str(tool)], 2, stdout=SECRET, stderr=SECRET)
        with patch.object(ops.subprocess, "run", return_value=completed):
            with self.assertRaises(ops.SigningError) as raised:
                ops.execute(self.config, "apksigner", ["sign", "file:private"], "APK signing")
        self.assertNotIn(SECRET, str(raised.exception))

    def test_public_metadata_is_atomic_and_never_overwritten(self):
        destination = self.root / "public.json"
        ops.write_public_metadata(destination, {"public": True})
        self.assertEqual({"public": True}, json.loads(destination.read_text()))
        with self.assertRaises(ops.SigningError):
            ops.write_public_metadata(destination, {"public": False})

    def test_instrumentation_cannot_masquerade_as_production_application(self):
        with self.assertRaises(ops.SigningError):
            ops.verify_apk(self.config, self.apk, 1, self.runner, instrumentation=True)
        self.manifest = self.manifest.replace("name='com.example.arborscan_app'", "name='com.example.arborscan_app.test'")
        self.manifest += "application-debuggable\n"
        result = ops.verify_apk(self.config, self.apk, 1, self.runner, instrumentation=True)
        self.assertEqual("instrumentation", result["artifact_kind"])
        with self.assertRaises(ops.SigningError):
            ops.verify_apk(self.config, self.apk, 1, self.runner)

    def test_instrumentation_target_and_runner_must_match_audit_contract(self):
        self.manifest = self.manifest.replace("name='com.example.arborscan_app'", "name='com.example.arborscan_app.test'")
        def wrong_target(config, tool, args, operation):
            result = self.runner(config, tool, args, operation)
            return result.replace('="com.example.arborscan_app"', '="other.application"') if "xmltree" in args else result
        with self.assertRaises(ops.SigningError):
            ops.verify_apk(self.config, self.apk, 1, wrong_target, instrumentation=True)

    def test_gradle_release_is_unsigned_and_audit_uses_release_target(self):
        gradle = (source.parents[1] / "arborscan_app/android/app/build.gradle").read_text()
        self.assertIn("signingConfig null", gradle)
        self.assertIn("debuggable false", gradle)
        self.assertIn('testBuildType "release"', gradle)
        self.assertNotIn("signingConfig signingConfigs.debug", gradle)

    def test_untrusted_lineage_capabilities_fail_before_publishing(self):
        output = self.root / "bad-lineage.apk"
        def wrong(config, tool, args, operation):
            result = self.runner(config, tool, args, operation)
            return result.replace("Has rollback capability : false", "Has rollback capability : true") if args[0] == "lineage" else result
        with self.assertRaisesRegex(ops.SigningError, "old-key rollback"):
            ops.sign(self.config, self.apk, output, 16, wrong)
        self.assertFalse(output.exists())

    def test_apk_embedded_lineage_must_match_data_preserving_configuration(self):
        def wrong(config, tool, args, operation):
            result = self.runner(config, tool, args, operation)
            if args[0] == "lineage" and args[args.index("--in") + 1] == str(self.apk):
                result = result.replace("Has installed data capability: true", "Has installed data capability: false", 1)
            return result
        with self.assertRaisesRegex(ops.SigningError, "preserve installed data"):
            ops.verify_apk(self.config, self.apk, 16, wrong)

    def test_windows_sdk_crlf_and_extra_cr_do_not_change_certificate_identity(self):
        for newline in ("\r\n", "\r\r\n", "\r"):
            with self.subTest(newline=repr(newline)):
                def windows(config, tool, args, operation):
                    return self.runner(config, tool, args, operation).replace("\n", newline)
                result = ops.verify_apk(self.config, self.apk, 16, windows)
                self.assertEqual([OLD, OLD, NEW, NEW],
                                 [row["sha256"] for row in result["platform_signatures"]])

    def test_real_windows_apksigner_old_sdk_output_is_parsed_without_losing_identity(self):
        fixture = source.parents[1] / "tests_v4/fixtures/apksigner-old-api24-27.txt"
        # Bytes preserve CRLF; Path.read_text() would conceal the original bug.
        output = fixture.read_bytes().decode("utf-8")
        result = ops.parse_signatures(output)
        self.assertEqual("68ff9861f52fdb22744b30ee27092008097f9eff4ead7eb38f80e662115f5b97", result["sha256"])
        self.assertEqual("2c08b4d637574c16be5cba1a595a4baac8541a4c", result["sha1"])
        self.assertTrue(result["schemes"]["v2"])
        self.assertFalse(result["schemes"]["v3.1"])

    def test_real_v31_sdk_targeted_signers_are_selected_by_verified_platform_range(self):
        fixture = source.parents[1] / "tests_v4/fixtures/apksigner-rotation-api33.txt"
        output = fixture.read_bytes().decode("utf-8")
        new = ops.parse_signatures(output, 33, 33)
        old = ops.parse_signatures(output, 24, 32)
        self.assertEqual("7fb94ede4099199db9166609c34d6278ebe51c7d80bd6ce1cd0208bf863863b8", new["sha256"])
        self.assertEqual("68ff9861f52fdb22744b30ee27092008097f9eff4ead7eb38f80e662115f5b97", old["sha256"])
        self.assertTrue(new["schemes"]["v3.1"])
        with self.assertRaises(ops.SigningError):
            ops.parse_signatures(output)
        with self.assertRaises(ops.SigningError):
            ops.parse_signatures(output, 32, 33)

    def test_overlapping_sdk_targeted_signers_are_rejected(self):
        fixture = source.parents[1] / "tests_v4/fixtures/apksigner-rotation-api33.txt"
        output = fixture.read_bytes().decode("utf-8").replace("maxSdkVersion=32", "maxSdkVersion=34")
        with self.assertRaises(ops.SigningError):
            ops.parse_signatures(output, 33, 33)


if __name__ == "__main__":
    unittest.main()
