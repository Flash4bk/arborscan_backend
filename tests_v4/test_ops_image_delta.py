import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('image_delta', Path(__file__).parents[1]/'deploy-vps/ops_image_delta.py')
delta = importlib.util.module_from_spec(spec)
spec.loader.exec_module(delta)


def archive(path, files):
    with tarfile.open(path, 'w') as out:
        for name, value in files.items():
            item = tarfile.TarInfo(name)
            item.size = len(value)
            out.addfile(item, io.BytesIO(value))


class ImageDeltaTest(unittest.TestCase):
    def test_metadata_replaced_and_shared_layer_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, target, patch, merged = [root/name for name in ('base.tar', 'target.tar', 'patch.tar.gz', 'merged.tar')]
            archive(base, {'index.json': b'old', 'blobs/shared': b'same'})
            archive(target, {'index.json': b'new', 'blobs/shared': b'same', 'blobs/new': b'added'})
            result = delta.create(base, target, patch)
            self.assertEqual(result['changed_members'], 2)
            delta.apply(base, patch, merged)
            with tarfile.open(merged) as restored:
                self.assertEqual(restored.extractfile('index.json').read(), b'new')
                self.assertEqual(restored.extractfile('blobs/shared').read(), b'same')
                self.assertEqual(len(restored.getnames()), 3)
            with self.assertRaises(FileExistsError):
                delta.apply(base, patch, merged)

    def test_corrupt_base_cannot_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, target, patch, merged = [root/name for name in ('base.tar', 'target.tar', 'patch.tar.gz', 'merged.tar')]
            archive(base, {'index.json': b'old'})
            archive(target, {'index.json': b'new'})
            delta.create(base, target, patch)
            base.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                delta.apply(base, patch, merged)
            self.assertFalse(merged.exists())

    def test_unsafe_path_refused(self):
        member = tarfile.TarInfo('../outside')
        with self.assertRaises(ValueError):
            delta.safe(member)


if __name__ == '__main__':
    unittest.main()
