"""A data backup must not silently depend on an unavailable runtime image."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    'release_bundle', Path(__file__).resolve().parents[1] / 'deploy-vps/ops_verify_release_bundle.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReleaseBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.backup = self.root / 'backup'
        self.backup.mkdir()
        (self.backup / 'COMPLETE').touch()
        self.image = 'sha256:' + 'a' * 64
        self.inventory(self.image)
        self.release = self.root / 'release'
        self.release.mkdir()
        (self.release / 'COMPLETE').touch()
        manifest = {'status': 'verified_complete', 'base_images': {'file': 'base.tar'},
                    'source': {'file': 'source.tar'},
                    'overlay': {'file': 'delta.tar', 'result_image': 'sha256:' + 'b' * 64}}
        for field in ('base_images', 'source', 'overlay'):
            manifest[field]['sha256'] = hashlib.sha256(b'fixture archive content').hexdigest()
        (self.release / 'release.json').write_text(json.dumps(manifest))
        (self.release / 'pinned-images.json').write_text(json.dumps({'images': {'api': self.image}}))
        for name in ('base.tar', 'source.tar', 'delta.tar', 'delta.tar.json'):
            (self.release / name).write_bytes(b'fixture archive content')
        (self.release / 'delta.tar.json').write_text(json.dumps({
            'version': 1, 'base_sha256': manifest['base_images']['sha256'],
            'delta_sha256': manifest['overlay']['sha256']}))
        self.index = {'version': 1, 'complete': True, 'images': {
            self.image: {'release': 'release/release.json', 'artifact': 'base_images'}}}
        self.save_index()
        self.checksums()

    def inventory(self, image):
        (self.backup / 'containers.private.json').write_text(json.dumps([
            {'Name': name, 'Image': image, 'Config': {'Env': ['SECRET=do-not-print']}}
            for name in ('/arborscan-api-v4', '/arborscan-quality-worker')]))

    def save_index(self):
        (self.root / 'RELEASE_INDEX.json').write_text(json.dumps(self.index))

    def checksums(self):
        files = [p for p in self.release.iterdir() if p.name not in ('COMPLETE', 'SHA256SUMS')]
        (self.release / 'SHA256SUMS').write_text('\n'.join(
            hashlib.sha256(p.read_bytes()).hexdigest() + '  ' + p.name for p in files))

    def test_resolves_shared_runtime_archive_without_copying_it(self):
        before = {p: p.stat().st_mtime_ns for p in self.release.iterdir()}
        result = module.verify(self.backup, self.root)
        self.assertEqual(result['runtime_images_mapped'], 1)
        self.assertEqual(before, {p: p.stat().st_mtime_ns for p in self.release.iterdir()})

    def test_new_deployment_without_archived_image_is_refused(self):
        self.inventory('sha256:' + 'c' * 64)
        with self.assertRaisesRegex(ValueError, 'no external release'):
            module.verify(self.backup, self.root)

    def test_candidate_overlay_and_base_can_share_one_release(self):
        candidate = 'sha256:' + 'b' * 64
        self.index['images'][candidate] = {
            'release': 'release/release.json', 'artifact': 'base_images+overlay'}
        self.save_index()
        inventory = json.loads((self.backup / 'containers.private.json').read_text())
        inventory[1]['Image'] = candidate
        (self.backup / 'containers.private.json').write_text(json.dumps(inventory))
        result = module.verify(self.backup, self.root)
        self.assertEqual(result['runtime_images_mapped'], 2)
        self.assertEqual(result['release_archives_verified'], 1)

    def test_unfinished_transfer_is_not_a_complete_release(self):
        (self.release / 'COMPLETE').unlink()
        with self.assertRaisesRegex(ValueError, 'Incomplete external release'):
            module.verify(self.backup, self.root)

    def test_corruption_is_refused_even_with_complete_marker(self):
        (self.release / 'base.tar').write_bytes(b'broken')
        with self.assertRaisesRegex(ValueError, 'Damaged'):
            module.verify(self.backup, self.root)

    def test_required_archive_cannot_be_omitted_from_checksums(self):
        (self.release / 'base.tar').unlink()
        self.checksums()
        with self.assertRaisesRegex(ValueError, 'omits required'):
            module.verify(self.backup, self.root)

    def test_path_outside_offsite_root_is_refused(self):
        self.index['images'][self.image]['release'] = '../outside/release.json'
        self.save_index()
        with self.assertRaisesRegex(ValueError, 'Invalid release path'):
            module.verify(self.backup, self.root)

    def test_overlay_from_different_base_is_refused(self):
        p = self.release / 'delta.tar.json'
        delta = json.loads(p.read_text())
        delta['base_sha256'] = '0' * 64
        p.write_text(json.dumps(delta))
        self.checksums()
        with self.assertRaisesRegex(ValueError, 'does not match release base'):
            module.verify(self.backup, self.root)


if __name__ == '__main__':
    unittest.main()
