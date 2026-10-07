"""Generated escrow fixtures only; no actual ArborScan keys or passwords read."""
import base64
from concurrent.futures import ThreadPoolExecutor
import contextlib
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.serialization import pkcs12
from cryptography.x509.oid import NameOID

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy-vps"))
import ops_signing_key_recovery as recovery


class SigningKeyRecoveryTests(unittest.TestCase):
    def setUp(self):
        # The actual source key files are never opened. Tests get their own
        # random private sibling directory on D when that drive is available.
        private_parent = Path(r"D:\ArborScanSigning") if os.name == "nt" else None
        if private_parent is not None and not private_parent.is_dir():
            private_parent = None
        self.temporary = tempfile.TemporaryDirectory(prefix="generated-key-recovery-", dir=private_parent)
        self.root = Path(self.temporary.name)
        recovery.secure_owned_directory(self.root)
        self.key = AESGCM.generate_key(bit_length=256)
        self.password = "generated-fixture-only-password"
        signing_key = ec.generate_private_key(ec.SECP256R1())
        subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Recovery TEST ONLY")])
        now = datetime.datetime.now(datetime.timezone.utc)
        certificate = (x509.CertificateBuilder().subject_name(subject).issuer_name(subject)
                       .public_key(signing_key.public_key()).serial_number(x509.random_serial_number())
                       .not_valid_before(now-datetime.timedelta(minutes=1))
                       .not_valid_after(now+datetime.timedelta(days=1)).sign(signing_key, hashes.SHA256()))
        p12 = pkcs12.serialize_key_and_certificates(
            b"generated-release", signing_key, certificate, None,
            serialization.BestAvailableEncryption(self.password.encode()))
        self.files = {
            "release.p12": p12,
            "legacy-debug.keystore": b"generated-old-store\0fixture",
            "signing.lineage": b"generated-lineage\0fixture",
            "credentials.private.json": json.dumps({"release_password": self.password}).encode(),
        }
        self.escrow = self.root / "key-backup.aesgcm"
        self.secret = self.root / "recovery-secret.private"
        self.destination = self.root / "restored"
        self._write(self.secret, base64.b64encode(self.key) + b"\n")
        self._encrypt(self._tar())

    def tearDown(self):
        self.temporary.cleanup()

    def _write(self, path, raw):
        path.write_bytes(raw)
        if os.name != "nt":
            path.chmod(0o600)

    def _tar(self, rows=None):
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w") as archive:
            for row in rows if rows is not None else [(n, d, tarfile.REGTYPE, "") for n, d in self.files.items()]:
                name, raw, kind, link = row
                member = tarfile.TarInfo(name)
                member.type = kind
                member.linkname = link
                member.size = len(raw) if kind == tarfile.REGTYPE else 0
                archive.addfile(member, io.BytesIO(raw) if kind == tarfile.REGTYPE else None)
        return stream.getvalue()

    def _encrypt(self, plain, aad=recovery.AAD):
        nonce = os.urandom(recovery.NONCE_BYTES)
        self._write(self.escrow, recovery.PREFIX + nonce + AESGCM(self.key).encrypt(nonce, plain, aad))

    def _assert_rejected_cleanly(self):
        with self.assertRaises(recovery.RecoveryError):
            recovery.recover(self.escrow, self.secret, self.destination)
        self.assertFalse(self.destination.exists())
        self.assertEqual(list(self.root.glob(".signing-recovery-*")), [])

    def test_real_aes_gcm_round_trip_and_private_output_without_certificate_claim(self):
        source_hash = hashlib.sha256(self.escrow.read_bytes()).hexdigest()
        result = recovery.recover(self.escrow, self.secret, self.destination)
        self.assertEqual(result["files_recovered"], 4)
        self.assertFalse(result["certificate_verified"])
        self.assertFalse(result["apk_signed"])
        self.assertFalse(result["existing_keys_replaced"])
        self.assertEqual(result["escrow_sha256"], source_hash)
        self.assertEqual({p.name for p in self.destination.iterdir()}, recovery.FILES)
        for name, data in self.files.items():
            target = self.destination / name
            self.assertEqual(target.read_bytes(), data)
            recovery._private(target)
        recovery._private(self.destination, directory=True)
        _, certificate, _ = pkcs12.load_key_and_certificates(
            (self.destination / "release.p12").read_bytes(), self.password.encode())
        self.assertEqual(certificate.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value, "Recovery TEST ONLY")
        self.assertEqual(hashlib.sha256(self.escrow.read_bytes()).hexdigest(), source_hash)

    def test_existing_destination_and_repeat_never_replace_material(self):
        recovery.recover(self.escrow, self.secret, self.destination)
        before = {p.name: p.read_bytes() for p in self.destination.iterdir()}
        with self.assertRaises(recovery.RecoveryError):
            recovery.recover(self.escrow, self.secret, self.destination)
        self.assertEqual({p.name: p.read_bytes() for p in self.destination.iterdir()}, before)
        self.assertEqual(list(self.root.glob(".signing-recovery-*")), [])

    def test_empty_existing_destination_is_not_replaced(self):
        self.destination.mkdir()
        with self.assertRaises(recovery.RecoveryError):
            recovery.recover(self.escrow, self.secret, self.destination)
        self.assertEqual(list(self.destination.iterdir()), [])

    def test_wrong_key_and_corrupt_ciphertext_never_write_plaintext(self):
        original = self.escrow.read_bytes()
        self._write(self.secret, base64.b64encode(AESGCM.generate_key(bit_length=256)))
        self._assert_rejected_cleanly()
        self._write(self.secret, base64.b64encode(self.key))
        modified = bytearray(original)
        modified[-1] ^= 1
        self._write(self.escrow, bytes(modified))
        self._assert_rejected_cleanly()

    def test_aad_and_nonce_are_authenticated(self):
        self._encrypt(self._tar(), aad=b"unrelated application")
        self._assert_rejected_cleanly()
        self._encrypt(self._tar())
        frame = bytearray(self.escrow.read_bytes())
        frame[len(recovery.PREFIX)] ^= 1
        self._write(self.escrow, bytes(frame))
        self._assert_rejected_cleanly()

    def test_prefix_and_truncated_ciphertext_are_refused(self):
        for frame in (b"OTHER-VERSION\n" + b"x" * 100, recovery.PREFIX,
                      recovery.PREFIX + b"x" * (recovery.NONCE_BYTES + recovery.TAG_BYTES - 1)):
            with self.subTest(frame_bytes=len(frame)):
                self._write(self.escrow, frame)
                self._assert_rejected_cleanly()

    def test_invalid_base64_and_wrong_key_lengths_are_refused(self):
        for raw in (b"not base64!", base64.b64encode(b"short"), base64.b64encode(b"x" * 64)):
            with self.subTest(secret_size=len(raw)):
                self._write(self.secret, raw)
                self._assert_rejected_cleanly()

    def test_source_sizes_are_bounded_before_decryption(self):
        with patch.object(recovery, "MAX_ENCRYPTED_BYTES", 16), patch.object(recovery, "_decrypt") as decrypt:
            self._assert_rejected_cleanly()
            decrypt.assert_not_called()
        self._write(self.secret, b"x" * (recovery.MAX_SECRET_BYTES + 1))
        self._assert_rejected_cleanly()

    def test_member_size_and_empty_members_are_refused(self):
        for payload in (b"", b"x" * (recovery.MAX_MEMBER_BYTES + 1)):
            with self.subTest(member_size=len(payload)):
                self.files["release.p12"] = payload
                self._encrypt(self._tar())
                self._assert_rejected_cleanly()

    def test_missing_unexpected_and_duplicate_members_are_refused(self):
        rows = [(n, d, tarfile.REGTYPE, "") for n, d in self.files.items()]
        for changed in (rows[:-1], rows + [("extra.txt", b"extra", tarfile.REGTYPE, "")], rows + [rows[0]]):
            with self.subTest(count=len(changed)):
                self._encrypt(self._tar(changed))
                self._assert_rejected_cleanly()

    def test_traversal_absolute_and_windows_member_names_are_refused(self):
        rows = [(n, d, tarfile.REGTYPE, "") for n, d in self.files.items()]
        for name in ("../release.p12", "/release.p12", "C:/release.p12", "sub\\release.p12", "sub/release.p12"):
            with self.subTest(name=name):
                self._encrypt(self._tar([(name, rows[0][1], tarfile.REGTYPE, ""), *rows[1:]]))
                self._assert_rejected_cleanly()

    def test_links_directories_and_devices_are_refused(self):
        rows = [(n, d, tarfile.REGTYPE, "") for n, d in self.files.items()]
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.DIRTYPE, tarfile.CHRTYPE):
            with self.subTest(member_type=kind):
                self._encrypt(self._tar([("release.p12", b"", kind, "outside"), *rows[1:]]))
                self._assert_rejected_cleanly()

    def test_json_schema_duplicate_fields_and_password_bounds(self):
        invalid = (b"{", b"[]", b"{}", b'{"release_password":123}', b'{"release_password":""}',
                   b'{"release_password":"fixture","extra":1}',
                   b'{"release_password":"one","release_password":"two"}',
                   json.dumps({"release_password": "x" * 4097}).encode(),
                   json.dumps({"release_password": "x\0y"}).encode(),
                   b'{"release_password":' + b'[' * 2048 + b'0' + b']' * 2048 + b'}')
        for raw in invalid:
            with self.subTest(json_bytes=len(raw)):
                self.files["credentials.private.json"] = raw
                self._encrypt(self._tar())
                self._assert_rejected_cleanly()

    def test_truncated_tar_and_hidden_trailing_data_are_refused(self):
        original = self._tar()
        with tarfile.open(fileobj=io.BytesIO(original), mode="r:") as archive:
            last = archive.getmembers()[-1]
            end = ((last.offset_data + last.size + 511) // 512) * 512
        for plain in (original[:end + 512], original + b"hidden trailing data"):
            with self.subTest(tar_bytes=len(plain)):
                self._encrypt(plain)
                self._assert_rejected_cleanly()

    def test_write_interruption_cleans_only_own_staging_and_retry_succeeds(self):
        untouched = self.root / ".signing-recovery-unrelated"
        untouched.mkdir()
        (untouched / "keep").write_bytes(b"another operation")
        actual_write = recovery._write_private
        calls = 0
        def interrupted(path, raw):
            nonlocal calls
            calls += 1
            actual_write(path, raw)
            if calls == 2:
                raise OSError("simulated interruption with a private secret")
        with patch.object(recovery, "_write_private", side_effect=interrupted):
            with self.assertRaises(OSError):
                recovery.recover(self.escrow, self.secret, self.destination)
        self.assertFalse(self.destination.exists())
        self.assertEqual(list(self.root.glob(".signing-recovery-*")), [untouched])
        self.assertEqual((untouched / "keep").read_bytes(), b"another operation")
        recovery.recover(self.escrow, self.secret, self.destination)
        self.assertEqual((self.destination / "release.p12").read_bytes(), self.files["release.p12"])

    def test_publication_error_cleans_staging_and_retry_succeeds(self):
        with patch.object(recovery, "_publish", side_effect=OSError("simulated filesystem failure")):
            with self.assertRaises(OSError):
                recovery.recover(self.escrow, self.secret, self.destination)
        self.assertFalse(self.destination.exists())
        self.assertEqual(list(self.root.glob(".signing-recovery-*")), [])
        self.assertEqual(recovery.recover(self.escrow, self.secret, self.destination)["files_recovered"], 4)

    def test_disk_content_corruption_is_rejected_before_publication(self):
        actual_write = recovery._write_private
        def corrupted(path, raw):
            actual_write(path, raw)
            if path.name == "release.p12":
                path.write_bytes(b"unexpected disk contents")
        with patch.object(recovery, "_write_private", side_effect=corrupted):
            self._assert_rejected_cleanly()
        self.assertEqual(recovery.recover(self.escrow, self.secret, self.destination)["files_recovered"], 4)

    def test_publication_race_preserves_destination_created_by_other_writer(self):
        actual_publish = recovery._publish
        def race(stage, destination):
            destination.mkdir()
            (destination / "owned-by-other").write_bytes(b"keep")
            actual_publish(stage, destination)
        with patch.object(recovery, "_publish", side_effect=race):
            with self.assertRaises(recovery.RecoveryError):
                recovery.recover(self.escrow, self.secret, self.destination)
        self.assertEqual((self.destination / "owned-by-other").read_bytes(), b"keep")
        self.assertEqual(list(self.root.glob(".signing-recovery-*")), [])

    def test_two_real_concurrent_recoveries_publish_exactly_once(self):
        def one():
            try:
                return recovery.recover(self.escrow, self.secret, self.destination)["status"]
            except recovery.RecoveryError:
                return "refused"
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda _: one(), range(2)))
        self.assertCountEqual(results, ["recovered_private_signing_material", "refused"])
        self.assertEqual({p.name: p.read_bytes() for p in self.destination.iterdir()}, self.files)
        self.assertEqual(list(self.root.glob(".signing-recovery-*")), [])

    def test_git_directory_and_worktree_git_file_destinations_are_refused(self):
        for kind in ("directory", "file"):
            repo = self.root / ("test-git-" + kind)
            repo.mkdir()
            marker = repo / ".git"
            marker.mkdir() if kind == "directory" else marker.write_text("gitdir: generated-fixture")
            with self.assertRaises(recovery.RecoveryError):
                recovery.recover(self.escrow, self.secret, repo / "private-keys")
            self.assertFalse((repo / "private-keys").exists())

    def test_symlink_source_and_destination_parent_are_refused(self):
        alias = self.root / "linked-secret"
        try:
            alias.symlink_to(self.secret)
        except (OSError, NotImplementedError):
            self.skipTest("creating test symlinks is not permitted on this Windows host")
        with self.assertRaises(recovery.RecoveryError):
            recovery.recover(self.escrow, alias, self.destination)
        parent_alias = self.root / "linked-parent"
        parent_alias.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(recovery.RecoveryError):
            recovery.recover(self.escrow, self.secret, parent_alias / "restored")
        self.assertFalse(self.destination.exists())

    def test_nonprivate_parent_is_refused_using_real_platform_permissions(self):
        nonprivate = self.root / "public-parent"
        nonprivate.mkdir()
        if os.name == "nt":
            subprocess.run(["icacls.exe", str(nonprivate), "/grant", "*S-1-1-0:(OI)(CI)R"],
                           capture_output=True, check=True)
        else:
            nonprivate.chmod(0o755)
        with self.assertRaises(recovery.RecoveryError):
            recovery.recover(self.escrow, self.secret, nonprivate / "restored")
        self.assertFalse((nonprivate / "restored").exists())

    def test_cli_never_prints_private_values_paths_or_decoded_bytes(self):
        self._write(self.escrow, b"private-untrusted-archive-with-secret-value")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = recovery.main(["--escrow", str(self.escrow), "--recovery-secret-file", str(self.secret),
                                    "--destination", str(self.destination)])
        self.assertEqual(status, 1)
        self.assertNotIn(self.password, output.getvalue())
        self.assertNotIn(str(self.root), output.getvalue())
        self.assertNotIn("private-untrusted", output.getvalue())
        self.assertEqual(json.loads(output.getvalue())["status"], "failed")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(recovery.main(["--unexpected", "do-not-print-this-value"]), 1)
        self.assertNotIn("do-not-print-this-value", output.getvalue())


if __name__ == "__main__":
    unittest.main()
