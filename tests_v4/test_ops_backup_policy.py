"""Destructive/interrupt/retention cases use only tiny isolated fixtures."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / 'deploy-vps'))
import ops_backup_policy as policy

API = 'sha256:' + '1' * 64
WORKER = 'sha256:' + '2' * 64


class BackupPolicyTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.home = Path(self.temp.name)
        self.root = self.home / 'backups'
        self.root.mkdir()
        self.asset = self.home / 'immutable-runtime.tar'
        self.asset.write_bytes(b'test runtime image archive')
        self.runtime = {'format': 1, 'images': {image: [
            {'path': str(self.asset), 'sha256': policy.digest(self.asset)}] for image in (API, WORKER)}}
        self.index = self.home / 'RUNTIME_INDEX.json'
        self.index.write_text(json.dumps(self.runtime))
        self.created = []

    def tearDown(self):
        self.temp.cleanup()

    def content(self, folder, native=True, runtime=True, automatic=True, operation=None):
        folder.mkdir(exist_ok=True)
        for name in policy.REQUIRED:
            (folder / name).write_bytes(('test-' + name).encode())
        (folder / 'containers.private.json').write_text(json.dumps([
            {'Name': '/arborscan-api-v4', 'Image': API},
            {'Name': '/arborscan-quality-worker', 'Image': WORKER}]))
        if native:
            postgres = folder / 'postgres'
            postgres.mkdir()
            for name in policy.NATIVE:
                (postgres / name).write_bytes(b'PGDMPfixture' if name == 'database.dump' else b'fixture')
            (postgres / 'SHA256SUMS').write_text(''.join(
                policy.digest(postgres / name) + '  ' + name + '\n' for name in sorted(policy.NATIVE)))
            (postgres / 'COMPLETE').touch()
        if runtime:
            (folder / 'RUNTIME_DEPENDENCIES.json').write_text(json.dumps(self.runtime))
        if automatic:
            (folder / 'AUTO_BACKUP.json').write_text(json.dumps({
                'format': 1, 'kind': 'scheduled_backup', 'name': folder.name,
                'operation_id': operation or folder.name}))
        policy.write_manifest(folder)
        (folder / 'COMPLETE').touch()
        return folder

    def existing(self, count=14, **options):
        return [self.content(self.root / ('20260101T%06dZ' % number), **options)
                for number in range(count)]

    def create(self, folder):
        self.created.append(folder)
        self.content(folder, runtime=False, automatic=False)
        (folder / 'COMPLETE').unlink()  # Stage cannot publish the parent COMPLETE.

    def run_backup(self, **kwargs):
        kwargs.setdefault('operation_id', 'test-operation')
        return policy.run_backup(self.root, self.create, self.index,
                                 minimum_free=1, **kwargs)

    def test_fourteen_then_full_new_backup_then_rotation_and_idempotent_repeat(self):
        before = self.existing()
        result = self.run_backup()
        self.assertEqual(result['event'], 'completed_verified')
        self.assertEqual(result['deleted'], [before[0].name])
        self.assertFalse(before[0].exists())
        self.assertEqual(len(policy.inventory(self.root)), 14)
        repeated = self.run_backup()
        self.assertEqual(repeated['event'], 'verified_existing')
        self.assertEqual(len(self.created), 1)
        self.assertEqual(repeated['name'], result['name'])

    def test_no_delete_until_native_and_runtime_are_verified(self):
        before = self.existing()
        def broken(folder):
            self.create(folder)
            (folder / 'postgres/database.dump').write_bytes(b'corrupt')
        with self.assertRaisesRegex(policy.PolicyError, 'postgres|checksum'):
            policy.run_backup(self.root, broken, self.index, 'failure', minimum_free=1)
        self.assertTrue(all(folder.exists() for folder in before))
        self.assertEqual(len(policy.inventory(self.root)), 14)
        self.assertFalse(any((p / 'COMPLETE').exists() for p in (self.root / '.staging').glob('*/*')))

    def test_interrupted_creation_is_retried_as_one_operation(self):
        before = self.existing()
        def interrupted(folder):
            self.create(folder)
            raise KeyboardInterrupt('simulated process interruption')
        with self.assertRaises(KeyboardInterrupt):
            policy.run_backup(self.root, interrupted, self.index, 'interrupt', minimum_free=1)
        self.assertTrue(all(folder.exists() for folder in before))
        active = json.loads((self.root / 'active-backup.json').read_text())
        result = self.run_backup(operation_id='interrupt')
        self.assertEqual(result['name'], active['name'])
        self.assertEqual(len(policy.inventory(self.root)), 14)
        self.assertEqual(len(list((self.root / '.staging' / active['name']).iterdir())), 1)

    def test_crash_after_publication_finishes_rotation_without_duplicate_even_next_day(self):
        self.existing()
        with patch.object(policy, 'rotate', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_backup(operation_id='previous-day')
        self.assertEqual(len(policy.inventory(self.root)), 15)
        result = self.run_backup(operation_id='next-day')
        self.assertEqual(result['event'], 'verified_existing')
        self.assertFalse((self.root / 'active-backup.json').exists())
        self.assertEqual(len(policy.inventory(self.root)), 14)
        self.assertEqual(len(self.created), 1)

    def test_low_disk_without_fresh_remote_proof_preserves_every_copy(self):
        before = self.existing()
        with self.assertRaisesRegex(policy.PolicyError, 'insufficient_disk'):
            self.run_backup(free=lambda _: shutil._ntuple_diskusage(1, 1, 0))
        self.assertTrue(all(folder.exists() for folder in before))
        self.assertEqual(self.created, [])

    def test_corrupt_set_is_not_successful_or_deleted(self):
        before = self.existing()
        (before[0] / 'application.tar').write_bytes(b'corrupt')
        result = self.run_backup()
        self.assertEqual(result['deleted'], [])
        rows = policy.inventory(self.root)
        self.assertEqual(sum(row['valid'] for row in rows), 14)
        self.assertTrue(before[0].exists())

    def test_pinned_manual_and_live_compose_sources_survive(self):
        before = self.existing()
        (before[0] / 'PINNED').touch()
        self.content(self.root / '20251201T000000Z', automatic=False)
        result = self.run_backup(live=[str(before[1] / 'reliability.yml')])
        self.assertTrue(before[0].exists())
        self.assertTrue(before[1].exists())
        self.assertTrue((self.root / '20251201T000000Z').exists())
        self.assertEqual(result['deleted'], [before[2].name, before[3].name])

    def test_all_sets_protected_fail_before_creating_or_deleting(self):
        before = self.existing(automatic=False)
        with self.assertRaisesRegex(policy.PolicyError, 'rotation_blocked'):
            self.run_backup()
        self.assertTrue(all(folder.exists() for folder in before))
        self.assertEqual(self.created, [])

    def test_nested_process_cannot_obtain_same_lock(self):
        with policy.locked(self.root):
            with self.assertRaisesRegex(policy.PolicyError, 'already_running'):
                self.run_backup()
        self.assertEqual(self.created, [])

    def test_runtime_base_dependency_is_retained(self):
        before = self.existing()
        base = before[0] / 'base-images.tar'
        base.write_bytes(b'base archive')
        policy.write_manifest(before[0])
        self.runtime['images'][API] = [{'path': str(base), 'sha256': policy.digest(base)}]
        self.index.write_text(json.dumps(self.runtime))
        result = self.run_backup()
        self.assertTrue(base.is_file())
        self.assertEqual(result['deleted'], [before[1].name])

    def test_runtime_archive_missing_or_changed_refuses_publication(self):
        before = self.existing()
        self.asset.write_bytes(b'changed runtime')
        with self.assertRaisesRegex(policy.PolicyError, 'runtime_archive_checksum'):
            self.run_backup()
        self.assertTrue(all(folder.exists() for folder in before))
        self.assertFalse(any((p / 'COMPLETE').exists() for p in (self.root / '.staging').glob('*/*')))

    def test_legacy_adoption_is_explicit_checksum_bound_and_nonmutating(self):
        before = self.existing(automatic=False)
        old_manifest = (before[0] / 'SHA256SUMS').read_bytes()
        evidence = {'format': 1, 'sets': [{'name': before[0].name,
                    'manifest_sha256': policy.digest(before[0] / 'SHA256SUMS'),
                    'source': 'observed_automatic_run', 'evidence_reference': 'isolated fake service fixture'}]}
        self.assertEqual(policy.adopt(self.root, evidence)['adopted_exact_manifests'], 1)
        result = self.run_backup()
        self.assertEqual(result['deleted'], [before[0].name])
        self.assertEqual((before[1] / 'SHA256SUMS').read_bytes(), old_manifest.replace(
            before[0].name.encode(), before[1].name.encode()))

    def test_adoption_requires_observed_evidence_and_matching_manifest(self):
        before = self.existing(1, automatic=False)
        for fields in ({'source': 'inferred_from_name', 'manifest_sha256': '0' * 64},
                       {'source': 'observed_automatic_run', 'manifest_sha256': '0' * 64}):
            item = {'name': before[0].name, 'evidence_reference': 'test', **fields}
            with self.assertRaises(policy.PolicyError):
                policy.adopt(self.root, {'format': 1, 'sets': [item]})
        self.assertFalse((self.root / 'automatic-sets.json').exists())

    def test_path_traversal_cannot_read_external_file(self):
        folder = self.existing(1)[0]
        (folder / 'SHA256SUMS').write_text('0' * 64 + '  ../secret\n')
        with self.assertRaisesRegex(policy.PolicyError, 'unsafe_manifest'):
            policy.verify_set(folder)

    def test_new_native_backup_manifest_must_be_covered_by_outer_hash(self):
        folder = self.existing(1)[0]
        manifest = folder / 'SHA256SUMS'
        manifest.write_text(''.join(line + '\n' for line in manifest.read_text().splitlines()
                                   if not line.endswith('  postgres/SHA256SUMS')))
        self.assertTrue(policy.verify_set(folder)['native_postgres'])  # Legacy can still be inventoried.
        self.assertIsNone(policy.backup_status(self.root)['latest_verified_full_backup'])
        with self.assertRaisesRegex(policy.PolicyError, 'native_manifest_not_covered'):
            policy.verify_set(folder, require_native=True)

    def test_unreadable_retained_dependency_metadata_prevents_rotation(self):
        before = self.existing(15)
        (before[0] / 'containers.private.json').write_bytes(b'not-json')
        # Corrupt material is not counted as successful, and cannot be removed.
        self.content(self.root / '20260201T000000Z')
        self.assertTrue(policy.plan(self.root)['blocked'])
        self.assertTrue(all(folder.exists() for folder in before))

    def test_changed_plan_is_refused_before_deletion(self):
        before = self.existing(15)
        proposal = policy.plan(self.root, retain=14, new_name=before[-1].name)
        (before[0] / 'PINNED').touch()
        with self.assertRaisesRegex(policy.PolicyError, 'rotation_plan_changed'):
            policy.rotate(self.root, proposal, before[-1].name)
        self.assertTrue(all(folder.exists() for folder in before))

    def test_status_does_not_treat_complete_marker_or_fake_external_as_success(self):
        before = self.existing(2)
        (before[1] / 'postgres/database.dump').write_bytes(b'corrupt')
        receipts = self.root / 'offsite-receipts'
        receipts.mkdir()
        (receipts / 'fake.json').write_text(json.dumps({'format': 1,
            'name': before[1].name, 'is_independent': True, 'actual_content_verified': False}))
        status = policy.backup_status(self.root)
        self.assertEqual(status['latest_verified_full_backup'], before[0].name)
        self.assertEqual(status['incomplete_or_corrupt_sets'], [before[1].name])
        self.assertIsNone(status['latest_confirmed_external'])

    def test_confirmed_external_age_is_separate_from_live_reachability(self):
        before = self.existing(1)
        receipts = self.root / 'offsite-receipts'
        receipts.mkdir()
        (receipts / 'verified.json').write_text(json.dumps({'format': 1,
            'name': before[0].name, 'manifest_sha256': policy.digest(before[0] / 'SHA256SUMS'),
            'is_independent': True, 'actual_content_verified': True,
            'source_content_verified': True, 'destination_id': 'isolated-test-receiver',
            'verified_at_utc': '2026-01-01T00:00:00+00:00'}))
        status = policy.backup_status(self.root, now=1767225610)
        self.assertEqual(status['latest_confirmed_external'], before[0].name)
        self.assertEqual(status['external_confirmation_age_seconds'], 10)
        self.assertFalse(status['external_current_reachability_verified'])

    def test_old_native_set_not_mislabelled_as_runtime_complete_contract(self):
        folder = self.existing(1, runtime=False)[0]
        status = policy.backup_status(self.root)
        self.assertEqual(status['latest_verified_native_backup'], folder.name)
        self.assertIsNone(status['latest_verified_full_backup'])
        self.assertEqual(status['verified_full_runtime_sets'], 0)

    def test_malformed_automatic_marker_does_not_allow_rotation(self):
        folder = self.existing(1)[0]
        for marker in ({'format': 1, 'kind': 'scheduled_backup', 'name': 'another', 'operation_id': 'valid'},
                       {'format': 1, 'kind': 'scheduled_backup', 'name': folder.name, 'operation_id': '../bad'}):
            (folder / 'AUTO_BACKUP.json').write_text(json.dumps(marker))
            policy.write_manifest(folder)
            row = policy.inventory(self.root)[0]
            self.assertFalse(row['automatic'])
            self.assertIn('manual_or_unadopted', row['protected'])

    def test_malformed_active_transaction_fails_without_new_stage(self):
        self.existing(1)
        (self.root / 'active-backup.json').write_text(json.dumps({'format': 1,
            'name': '20260101T000000Z', 'operation_id': '../outside', 'attempt': 1}))
        with self.assertRaisesRegex(policy.PolicyError, 'invalid_active_backup'):
            self.run_backup()
        self.assertFalse((self.root / '.staging').exists())
        self.assertEqual(self.created, [])

    @unittest.skipUnless(hasattr(__import__('os'), 'symlink'), 'symlink not supported')
    def test_symlink_copy_or_extra_descendant_cannot_be_removed(self):
        before = self.existing(15)
        try:
            (before[0] / 'unlisted-link').symlink_to(self.asset)
        except OSError:
            self.skipTest('OS forbids creation of symlink for this user')
        proposal = policy.plan(self.root, retain=14, new_name=before[-1].name)
        with self.assertRaisesRegex(policy.PolicyError, 'symlink_refused'):
            policy.rotate(self.root, proposal, before[-1].name)
        self.assertTrue(self.asset.is_file())
        self.assertTrue(before[0].exists())


if __name__ == '__main__':
    unittest.main()
