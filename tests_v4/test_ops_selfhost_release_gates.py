"""Release proof binding checks with private tiny files, never a real database."""
import copy
import datetime as dt
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'deploy-vps'))
import ops_selfhost_audit as audit
import ops_selfhost_release_gates as gates
import ops_selfhost_promote as promote


class ReleaseGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / 'selfhost-20261012T140855Z'
        self.root.mkdir(mode=0o700)
        self.now = dt.datetime(2026, 10, 12, 18, 40, tzinfo=dt.timezone.utc)
        self.proof_file = self.root / 'source-pre-cutover-proof-20261012T182930Z.json'
        public_rows = [{'table': name, 'rows': 1, 'sha256': audit.digest_bytes(name.encode())}
                       for name in ('users', 'auth_sessions', 'report_versions', 'contour_revisions')]
        public_rows += [{'versions': {'contours': 1, 'history': 1, 'quality': 1}},
                        {'relationships': {'orphan_sessions': 0, 'orphan_reports': 0, 'orphan_contours': 0}}]
        storage_rows = [{'storage_table': name, 'rows': count, 'sha256': audit.digest_bytes(name.encode())}
                        for name, count in (('objects', 12), ('buckets', 2))]
        public_file, storage_file = self.root / 'source-public.private.txt', self.root / 'source-storage.private.txt'
        public_file.write_text('\n'.join(json.dumps(row) for row in public_rows)); public_file.chmod(0o600)
        storage_file.write_text('\n'.join(json.dumps(row) for row in storage_rows)); storage_file.chmod(0o600)
        self.database = audit.parse_fingerprints(public_file.read_bytes())
        self.storage = audit.parse_storage_fingerprints(storage_file.read_bytes())
        audit.write_private(self.root / 'audit.expected.private.json', {'database': self.database, 'storage_database': self.storage})
        self.proof = {'captured_at_utc': '20261012T182930Z', 'public_file': str(public_file), 'storage_file': str(storage_file),
                      'source_public_exact': True, 'source_storage_exact': True, 'read_only_single_snapshot': True,
                      'sql_writes': False, 'tables': 4, 'public_rows': 4, 'storage_objects': 12, 'storage_buckets': 2,
                      'source_public_sha256': audit.digest_bytes(audit.canonical(self.database)),
                      'source_storage_sha256': audit.digest_bytes(audit.canonical(self.storage))}
        audit.write_private(self.proof_file, self.proof)
        keys = ['00000000-0000-4000-8000-000000000003_' + str(index) * 64 + '.json' for index in range(1, 5)]
        self.state = {'format': 1, 'tag': audit.FIXTURE_NAME, 'complete': True, 'image_sha256': 'a' * 64,
                      'accounts': [{'id': '00000000-0000-4000-8000-00000000000' + str(index),
                                    'email': 'demo-selfhost-' + letter * 32 + '@invalid.test', 'name': audit.FIXTURE_NAME,
                                    'password': 'synthetic-password', 'token': 'synthetic-token'}
                                   for index, letter in ((1, 'a'), (2, 'b'))],
                      'contours': [[key, status] for key, status in zip(keys[:3], ('rejected', 'accepted', 'draft'))]}
        audit.write_private(self.root / 'fixture.private.json', self.state)
        self.snapshot = {'format': 1, 'identity': {'tag': self.state['tag'], 'owner': self.state['accounts'][0]['id'],
                                                 'image_sha256': self.state['image_sha256']},
                         'records': {key: {'owner_id': self.state['accounts'][0]['id']} for key in keys},
                         'sql_metadata': [{'status': status} for _, status in self.state['contours']]}
        audit.write_private(self.root / 'quarantine.originals.private.json', self.snapshot)
        self.quarantine = {'objects_marked': 4, 'immutable_content_preserved': True, 'sql_decisions_preserved': True,
                           'source_accounts_absent': True, 'real_user_records_changed': False, 'training_started': False,
                           'snapshot_sha256': audit.digest_bytes(audit.canonical(self.snapshot)),
                           'actual_exporter_segmentation_classification_refusals': 8, 'dataset_created': False}
        audit.write_private(self.root / 'quarantine-result.json', self.quarantine)
        self.backup_root = Path(self.temporary.name) / 'ops-backups'; self.backup_root.mkdir(mode=0o700)
        self.backup = self.backup_root / '20261012T184000Z'; self.backup.mkdir(mode=0o700)
        (self.backup / 'COMPLETE').touch(mode=0o600)
        (self.backup / 'SHA256SUMS').write_bytes(b'verified tiny test manifest\n'); (self.backup / 'SHA256SUMS').chmod(0o600)
        self.marker = {'source_root': str(self.root), 'tables': {'auth_sessions': 3, 'ml_jobs': 0}, 'objects': 16}
        audit.write_private(self.backup / 'SELFHOST_BACKUP.json', self.marker)
        self.config = {'services': {'db': {'image': 'sha256:' + 'd' * 64}}}
        self.restored = {'event': 'isolated_selfhost_restore_verified', 'restored_backup': str(self.backup),
                         'restore_verified': True, 'backup_manifest_sha256': audit.digest_bytes((self.backup / 'SHA256SUMS').read_bytes()),
                         'native_sql_restore': True, 'roles_owner_acl_restored_by_pg_restore': True,
                         'network': 'none', 'production_credentials_mounted': False, 'worker_started': False, 'training_started': False,
                         'db_image': self.config['services']['db']['image'], 'migration_versions': [1, 1, 1],
                         'public_tables_matched': 2, 'storage_objects_matched': 16, 'auth_sessions_rows_matched': 3,
                         'quality_jobs_rows_matched': 0,
                         'raw_volumes': {name: {'ownership_restored': True, 'xattrs_restored': True,
                                                'files_restored': count, 'directories_restored': 2}
                                         for name, count in (('storage-files.tar', 16), ('runtime-files.tar', 1))}}
        evidence = self.root / 'backup-restore-checks' / '20261012T184100Z-123456abcdef'
        evidence.mkdir(mode=0o700, parents=True)
        self.restore_file = evidence / 'result.json'; audit.write_private(self.restore_file, self.restored)
        self.addCleanup(patch.stopall)
        patch.object(gates.backup, 'BACKUP_ROOT', self.backup_root).start()
        self.verify_backup = patch.object(gates.backup, 'verify_set', return_value=8).start()

    def test_actual_snapshot_quarantine_and_exact_restored_backup_are_all_required(self):
        result = gates.verify(self.root, self.backup, self.restore_file, self.proof_file, self.config, now=self.now)
        self.assertEqual(result['source_public_sha256'], self.proof['source_public_sha256'])
        self.assertEqual(result['restore_receipt_sha256'], audit.digest_bytes(self.restore_file.read_bytes()))
        self.assertEqual(result['backup_manifest_sha256'], self.restored['backup_manifest_sha256'])
        self.verify_backup.assert_called_once_with(self.backup)

    def test_stale_future_or_write_capable_source_snapshot_is_refused(self):
        for change in ({'captured_at_utc': '20261012T163000Z'}, {'captured_at_utc': '20261012T184200Z'},
                       {'sql_writes': True}, {'read_only_single_snapshot': False}, {'source_storage_exact': False}):
            with self.subTest(change=change):
                audit.write_private(self.proof_file, {**self.proof, **change})
                with self.assertRaises(gates.GateFailure): gates.source_gate(self.root, self.proof_file, now=self.now)

    def test_source_sha_and_counts_must_match_captured_bytes_and_imported_baseline(self):
        for change in ({'source_public_sha256': '0' * 64}, {'source_storage_sha256': '0' * 64},
                       {'tables': True}, {'public_rows': 5}):
            with self.subTest(change=change):
                audit.write_private(self.proof_file, {**self.proof, **change})
                with self.assertRaises(gates.GateFailure): gates.source_gate(self.root, self.proof_file, now=self.now)
        audit.write_private(self.proof_file, self.proof)
        other = copy.deepcopy(self.database); other['tables']['users']['sha256'] = 'e' * 64
        audit.write_private(self.root / 'audit.expected.private.json', {'database': other, 'storage_database': self.storage})
        with self.assertRaisesRegex(audit.AuditFailure, 'database_continuity_mismatch'):
            gates.source_gate(self.root, self.proof_file, now=self.now)

    def test_source_evidence_from_another_root_cannot_approve_this_candidate(self):
        external = self.root.parent / 'other-source-proof.json'; audit.write_private(external, self.proof)
        with self.assertRaises(gates.GateFailure): gates.source_gate(self.root, external, now=self.now)
        audit.write_private(self.proof_file, {**self.proof, 'public_file': str(self.root.parent / 'outside.txt')})
        with self.assertRaises(gates.GateFailure): gates.source_gate(self.root, self.proof_file, now=self.now)

    def test_no_quarantine_failed_exporter_or_real_record_changes_block_release(self):
        (self.root / 'quarantine-result.json').unlink()
        with self.assertRaises(gates.GateFailure): gates.quarantine_gate(self.root)
        for change in ({'objects_marked': 3}, {'actual_exporter_segmentation_classification_refusals': 7},
                       {'dataset_created': True}, {'real_user_records_changed': True}, {'sql_decisions_preserved': False}):
            with self.subTest(change=change):
                audit.write_private(self.root / 'quarantine-result.json', {**self.quarantine, **change})
                with self.assertRaises(gates.GateFailure): gates.quarantine_gate(self.root)

    def test_quarantine_receipt_cannot_be_reused_for_changed_or_unfinished_fixture(self):
        audit.write_private(self.root / 'quarantine.originals.private.json', {**self.snapshot, 'records': {}})
        with self.assertRaisesRegex(gates.GateFailure, 'snapshot_sha256'):
            gates.quarantine_gate(self.root)
        audit.write_private(self.root / 'quarantine.originals.private.json', self.snapshot)
        audit.write_private(self.root / 'fixture.private.json', {**self.state, 'complete': False})
        with self.assertRaises(gates.quarantine.QuarantineFailure): gates.quarantine_gate(self.root)

    def test_restore_is_bound_to_manifest_bytes_not_just_success_or_same_folder(self):
        (self.backup / 'SHA256SUMS').write_bytes(b'changed backup manifest\n')
        with self.assertRaisesRegex(gates.GateFailure, 'identity_sha256'):
            gates.restore_gate(self.root, self.backup, self.restore_file, self.config)

    def test_restore_receipt_counts_acl_files_and_isolation_are_required(self):
        for change in ({'restored_backup': str(self.backup_root / '20261011T184000Z')},
                       {'storage_objects_matched': 12}, {'auth_sessions_rows_matched': 2},
                       {'migration_versions': [True, 1, 1]}, {'roles_owner_acl_restored_by_pg_restore': False},
                       {'production_credentials_mounted': True}, {'network': 'bridge'}, {'worker_started': True},
                       {'raw_volumes': {'storage-files.tar': {}}}):
            with self.subTest(change=change):
                audit.write_private(self.restore_file, {**self.restored, **change})
                with self.assertRaises(gates.GateFailure):
                    gates.restore_gate(self.root, self.backup, self.restore_file, self.config)

    def test_unrelated_restore_receipt_or_source_backup_root_is_refused(self):
        other = self.root / 'unrelated-result.json'; audit.write_private(other, self.restored)
        with self.assertRaises(gates.GateFailure): gates.restore_gate(self.root, self.backup, other, self.config)
        audit.write_private(self.backup / 'SELFHOST_BACKUP.json', {**self.marker, 'source_root': 'another-stack'})
        with self.assertRaises(gates.GateFailure): gates.restore_gate(self.root, self.backup, self.restore_file, self.config)

    def test_sql_inventory_match_cannot_replace_actual_restoration_of_all_files(self):
        wrong = copy.deepcopy(self.restored)
        wrong['raw_volumes']['storage-files.tar']['files_restored'] = 0
        audit.write_private(self.restore_file, wrong)
        with self.assertRaisesRegex(gates.GateFailure, 'storage_file_count'):
            gates.restore_gate(self.root, self.backup, self.restore_file, self.config)

    def test_prepare_failed_gate_never_creates_candidate_or_runs_docker(self):
        (self.root / 'PREPARED').touch(mode=0o600)
        for name, value in [('compose.private.json', self.config), ('storage-seed.safe.json', {'sha256_verified': True}),
                            ('audit-before-fixtures.json', {'api_health': 'passed', 'exact_images_policy_models': 'identical'}),
                            ('audit-restart.json', {'smoke': {'restart_password_login_versions_contours': 'passed'}}),
                            ('inference-result.json', {'analysis_saved': True})]:
            audit.write_private(self.root / name, value)
        runner = Mock()
        with patch.object(promote.stack, 'checked_resume_root', return_value=self.root), \
             patch.object(promote.stack, 'live_rows', return_value={}), \
             patch.object(promote.stack, 'Runner', return_value=runner), \
             patch.object(promote.gates, 'verify', side_effect=gates.GateFailure('fresh_source_proof_expired')):
            with self.assertRaisesRegex(promote.PromotionFailure, 'fresh_source_proof_expired'):
                promote.prepare(self.root, self.backup, self.restore_file, self.proof_file)
        runner.run.assert_not_called()
        self.assertFalse((self.root / 'compose.production.private.json').exists())
        self.assertFalse((self.root / 'promotion.private.json').exists())

    def test_prepare_cannot_overwrite_promoted_or_interrupted_journal(self):
        for phase in ('promoted', 'parking_originals', 'local_api_starting', 'rolling_back'):
            with self.subTest(phase=phase):
                audit.write_private(self.root / 'promotion.private.json', {'phase': phase})
                with patch.object(promote.stack, 'checked_resume_root', return_value=self.root), \
                     patch.object(promote.stack, 'Runner') as runner:
                    with self.assertRaisesRegex(promote.PromotionFailure, 'cannot replace active journal'):
                        promote.prepare(self.root, self.backup, self.restore_file, self.proof_file)
                    runner.assert_not_called()
                self.assertEqual(audit.private_json(self.root / 'promotion.private.json')['phase'], phase)

    def test_prepared_activation_rejects_expired_source_proof_before_any_container_command(self):
        receipt = {'root': str(self.root), 'phase': 'prepared', 'release_gates': {'source_proof': str(self.proof_file)}}
        audit.write_private(self.root / 'promotion.private.json', receipt)
        with patch.object(promote.stack, 'checked_resume_root', return_value=self.root), \
             patch.object(promote.gates, 'source_gate', side_effect=gates.GateFailure('fresh_source_proof_expired')), \
             patch.object(promote.stack, 'Runner') as runner, patch.object(promote, 'continuity') as local:
            with self.assertRaisesRegex(promote.PromotionFailure, 'fresh_source_proof_expired'):
                promote.activate(self.root)
            runner.assert_not_called()
            local.assert_not_called()

    def test_prepared_activation_rejects_replaced_source_receipt_before_any_container_command(self):
        receipt = {'root': str(self.root), 'phase': 'prepared',
                   'release_gates': {'source_proof': str(self.proof_file), 'source_proof_sha256': 'original-proof'}}
        audit.write_private(self.root / 'promotion.private.json', receipt)
        with patch.object(promote.stack, 'checked_resume_root', return_value=self.root), \
             patch.object(promote.gates, 'source_gate', return_value={'source_proof_sha256': 'replaced-proof'}), \
             patch.object(promote.stack, 'Runner') as runner:
            with self.assertRaisesRegex(promote.PromotionFailure, 'source evidence changed'):
                promote.activate(self.root)
            runner.assert_not_called()


if __name__ == '__main__': unittest.main()
