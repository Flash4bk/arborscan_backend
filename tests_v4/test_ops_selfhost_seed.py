"""Physical seed guards with synthetic private files; not a live Storage test."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

scripts = Path(__file__).resolve().parents[1] / 'deploy-vps'
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location('selfhost_seed', scripts / 'ops_selfhost_seed.py')
ops = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ops)


class PhysicalSeedSafetyTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.snapshot = self.folder / 'restored'
        self.inventory = []
        manifest = {}
        for name, version in [('versioned/photo.json', '11111111-1111-4111-8111-111111111111'), ('legacy.json', None)]:
            raw = ('synthetic-' + name).encode()
            relative = 'objects/private-bucket/' + name
            path = self.snapshot / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            manifest[relative] = hashlib.sha256(raw).hexdigest()
            self.inventory.append({'bucket_id': 'private-bucket', 'name': name, 'version': version,
                                   'id': 'unchanged-storage-identity', 'owner_id': 'unchanged-owner',
                                   'metadata': {'size': len(raw), 'mimetype': 'application/json', 'cacheControl': '3600'}})
        (self.snapshot / 'manifest.json').write_text(json.dumps(manifest))
        self.environment = {'GLOBAL_S3_BUCKET': 'arborscan', 'TENANT_ID': 'arborscan',
                            'FILE_STORAGE_BACKEND_PATH': '/var/lib/storage', 'STORAGE_BACKEND': 'file'}

    def test_versioned_and_legacy_physical_paths_preserve_real_bytes_and_metadata(self):
        before = json.dumps(self.inventory, sort_keys=True)
        result = ops.seed_plan(self.snapshot, self.inventory, self.environment)
        self.assertEqual(result['objects_count'], 2)
        self.assertEqual(result['objects'][0]['physical'],
                         'arborscan/arborscan/private-bucket/versioned/photo.json/11111111-1111-4111-8111-111111111111')
        self.assertEqual(result['objects'][1]['physical'], 'arborscan/arborscan/private-bucket/legacy.json')
        self.assertEqual(result['objects'][0]['mimetype'], 'application/json')
        self.assertEqual(result['objects'][0]['cache_control'], '3600')
        self.assertEqual(json.dumps(self.inventory, sort_keys=True), before)
        self.assertNotIn('owner_id', json.dumps(result))
        self.assertNotIn('unchanged-storage-identity', json.dumps(result))

    def test_metadata_size_difference_fails_before_write(self):
        self.inventory[0]['metadata']['size'] += 1
        with self.assertRaises(ValueError):
            ops.seed_plan(self.snapshot, self.inventory, self.environment)

    def test_full_inventory_required_no_duplicates_missing_or_extra_objects(self):
        for rows in (self.inventory[:-1], self.inventory + [self.inventory[0]],
                     self.inventory + [{'bucket_id': 'private-bucket', 'name': 'missing.json', 'metadata': {'size': 1}}]):
            with self.subTest(count=len(rows)):
                with self.assertRaises(ValueError):
                    ops.seed_plan(self.snapshot, rows, self.environment)

    def test_path_escape_bucket_version_and_header_injection_rejected(self):
        for field, value in [('bucket_id', '..'), ('name', '../escape'), ('name', '/absolute'),
                             ('name', 'x//y'), ('name', 'x\\y'), ('name', 'x/./y'), ('version', '../escape')]:
            with self.subTest(field=field, value=value):
                rows = json.loads(json.dumps(self.inventory))
                rows[0][field] = value
                with self.assertRaises(ValueError):
                    ops.seed_plan(self.snapshot, rows, self.environment)
        rows = json.loads(json.dumps(self.inventory))
        rows[0]['metadata']['mimetype'] = 'application/json\r\nInjected: yes'
        with self.assertRaises(ValueError):
            ops.seed_plan(self.snapshot, rows, self.environment)

    def test_snapshot_bytes_tamper_rejected_and_no_header_metadata_defaults_are_explicit(self):
        self.inventory[0]['metadata'].pop('mimetype')
        self.inventory[0]['metadata'].pop('cacheControl')
        result = ops.seed_plan(self.snapshot, self.inventory, self.environment)
        self.assertEqual(result['objects'][0]['mimetype'], 'application/octet-stream')
        self.assertEqual(result['objects'][0]['cache_control'], 'no-cache')
        (self.snapshot / result['objects'][0]['source']).write_bytes(b'tampered')
        with self.assertRaises(ValueError):
            ops.seed_plan(self.snapshot, self.inventory, self.environment)

    def test_nonlocal_backend_or_wrong_root_rejected(self):
        for key, value in [('STORAGE_BACKEND', 's3'), ('FILE_STORAGE_BACKEND_PATH', '/production/files')]:
            changed = {**self.environment, key: value}
            with self.assertRaises(ValueError):
                ops.seed_plan(self.snapshot, self.inventory, changed)

    def seed_fixture(self):
        root = self.folder / 'selfhost-20261012T170000Z'
        root.mkdir()
        (root / 'PREPARED').touch()
        config = {'name': root.name.lower(), 'services': {'storage': {
            'image': 'sha256:' + 'a' * 64, 'environment': self.environment}}}
        (root / 'compose.private.json').write_text(json.dumps(config))
        (root / 'rootmanifest.safe.json').write_text(json.dumps({'root': str(root), 'storage_snapshot': str(self.snapshot),
                                                             'images': {'storage': 'sha256:' + 'a' * 64}}))
        metadata = self.folder / 'current-source-storage.private.json'
        metadata.write_text(json.dumps(self.inventory))
        args = SimpleNamespace(root=root, storage_snapshot=self.snapshot, source_storage_metadata=metadata,
                               backend_module=ops.BACKEND_MODULE)
        receipt = {'objects_seeded': 2, 'bytes': sum(row['metadata']['size'] for row in self.inventory),
                   'sha256_verified': True, 'database_rows_changed': False, 'metadata_upsert': False, 'network': 'none'}
        volume = root.name.lower() + '_storage'
        volume_info = [{'Name': volume, 'Labels': {'com.docker.compose.project': root.name.lower()}}]
        return root, args, receipt, volume_info

    def test_transient_importer_only_own_volume_sources_readonly_no_network_no_sql_or_upsert(self):
        root, args, receipt, volume_info = self.seed_fixture()
        runner = SimpleNamespace(run=Mock(side_effect=[SimpleNamespace(stdout=json.dumps(volume_info).encode()),
                                                       SimpleNamespace(stdout=json.dumps(receipt).encode())]))
        with patch.object(ops.stack, 'HOME', self.folder), patch.object(ops.stack, 'checked_resume_root', return_value=root), \
                patch.object(ops.stack, 'checked_private_file'), patch.object(ops.stack, 'Runner', return_value=runner), \
                patch.object(ops.stack, 'checked_infrastructure'):
            self.assertEqual(ops.seed(args), receipt)
        command = runner.run.call_args_list[-1].args[0]
        self.assertEqual(command[:3], ['docker', 'run', '--rm'])
        self.assertEqual(command[command.index('--network') + 1], 'none')
        self.assertEqual(command[command.index('--pull=never')], '--pull=never')
        self.assertEqual(command[command.index('--cap-add') + 1], 'DAC_OVERRIDE')
        mounts = [command[index + 1] for index, value in enumerate(command) if value == '-v']
        self.assertEqual(mounts, [root.name.lower() + '_storage:/var/lib/storage',
                                 str(self.snapshot) + ':/seed:ro', str(root / 'seed.plan.private.json') + ':/seed-plan.json:ro'])
        self.assertNotIn('--privileged', command)
        self.assertTrue((root / 'storage-seed.safe.json').is_file())
        self.assertEqual(len(runner.run.call_args_list), 2)
        self.assertNotIn('requests', ops.NODE_IMPORT)
        self.assertNotIn('upsert', ops.NODE_IMPORT.lower().replace('metadata_upsert', ''))
        self.assertNotIn('database', ops.NODE_IMPORT.lower().replace('database_rows_changed', ''))

    def test_wrong_project_volume_fails_before_any_importer(self):
        root, args, receipt, volume_info = self.seed_fixture()
        volume_info[0]['Labels']['com.docker.compose.project'] = 'production'
        runner = SimpleNamespace(run=Mock(return_value=SimpleNamespace(stdout=json.dumps(volume_info).encode())))
        with patch.object(ops.stack, 'HOME', self.folder), patch.object(ops.stack, 'checked_resume_root', return_value=root), \
                patch.object(ops.stack, 'checked_private_file'), patch.object(ops.stack, 'Runner', return_value=runner), \
                patch.object(ops.stack, 'checked_infrastructure'):
            with self.assertRaises(ValueError):
                ops.seed(args)
        self.assertEqual(len(runner.run.call_args_list), 1)
        self.assertFalse((root / 'seed.plan.private.json').exists())

    def test_retry_reuses_identical_private_plan_never_replaces_changed_input(self):
        path = self.folder / 'seed.plan.private.json'
        ops.identical_private_file(path, 'original')
        with patch.object(ops.stack, 'checked_private_file'):
            ops.identical_private_file(path, 'original')
            with self.assertRaises(ValueError):
                ops.identical_private_file(path, 'changed')
        self.assertEqual(path.read_text(), 'original')


if __name__ == '__main__':
    unittest.main()
