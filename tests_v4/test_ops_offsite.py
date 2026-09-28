import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('ops_offsite', Path(__file__).parents[1] / 'deploy-vps/ops_verify_offsite.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class OffsiteVerificationTest(unittest.TestCase):
    def fixture(self, root):
        names = ['application.tar', 'local-files.tar', 'arborscan.env.private', 'containers.private.json']
        for name in names:
            (root / name).write_bytes(b'synthetic')
        digest = hashlib.sha256(b'synthetic').hexdigest()
        (root / 'SHA256SUMS').write_text(''.join(f'{digest}  ./{name}\n' for name in names))

    def test_verified_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            self.assertEqual(module.verify(root)['verified_files'], 4)
            self.assertTrue((root / 'OFFSITE_VERIFIED.json').exists())

    def test_corruption_not_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            module.verify(root)
            (root / 'application.tar').write_bytes(b'truncated')
            with self.assertRaises(ValueError):
                module.verify(root)
            self.assertFalse((root / 'OFFSITE_VERIFIED.json').exists())

    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'SHA256SUMS').write_text('0' * 64 + '  ../outside\n')
            with self.assertRaises(ValueError):
                module.verify(root)

    def test_native_postgres_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = ['database.dump','roles.sql','source.json','archive-list.private.txt']
            lines = []
            for name in names:
                raw = ('synthetic '+name).encode()
                (root/name).write_bytes(raw)
                lines.append(hashlib.sha256(raw).hexdigest()+'  '+name+'\n')
            (root/'SHA256SUMS').write_text(''.join(lines))
            self.assertEqual(module.verify(root, postgres=True)['verified_files'],4)
            with self.assertRaises(ValueError):
                module.verify(root)  # Cannot call it an application/Storage backup.
            self.assertFalse((root/'OFFSITE_VERIFIED.json').exists())


if __name__ == '__main__':
    unittest.main()
