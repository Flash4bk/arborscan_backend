"""Local safety tests with synthetic files and mocked Docker, not a VPS deploy."""
from contextlib import redirect_stdout
import copy
from datetime import datetime, timedelta, timezone
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

source = Path(__file__).resolve().parents[1] / 'deploy-vps/ops_integration_v4.py'
spec = importlib.util.spec_from_file_location('integration_rollout', source)
ops = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ops)

COMMIT = 'a' * 40
CANDIDATE = 'sha256:' + 'b' * 64
SECRET = 'synthetic-weather-secret-do-not-print'


class IntegrationSafetyTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.root = self.folder / 'private-plan'
        self.config = self.folder / 'operator.env'
        self.config.write_text('OPENWEATHER_API_KEY=' + SECRET + '\nDATABASE=unchanged\n')
        self.compose = self.folder / 'compose.yml'
        self.compose.write_text('services: {api-v4: {image: pinned}}\n')
        self.models = self.folder / 'models'
        self.models.mkdir()
        (self.models / 'baseline.pt').write_bytes(b'original unchanged model')
        self.https = self.folder / 'nginx'
        self.https.mkdir()
        (self.https / 'nginx.conf').write_text('unchanged https config')
        self.backups = self.folder / 'backups'
        self.backups.mkdir()
        self.backup = self.backup_fixture(datetime.now(timezone.utc) - timedelta(seconds=10))
        self.env = {'PATH': '/bin', 'DATABASE': 'unchanged', 'OPENWEATHER_API_KEY': SECRET}
        self.before = self.row(ops.CURRENT_IMAGE, ops.CURRENT_REVISION, 'initial-id', 'initial-start')
        self.current = copy.deepcopy(self.before)
        self.v3 = self.row('sha256:v3', 'v3-revision', 'v3-id', 'v3-start')
        self.worker = self.row('sha256:worker', 'worker-revision', 'worker-id', 'worker-start')
        self.patches = [patch.object(ops, 'CONFIG', self.config),
            patch.object(ops, 'BACKUPS', self.backups), patch.object(ops, 'MODELS', self.models),
            patch.object(ops, 'HTTPS_CONFIG', self.https),
            patch.object(ops.os, 'fchmod', create=True),
            patch.object(ops, 'inspect', side_effect=self.inspect),
            patch.object(ops, 'command', side_effect=self.command),
            patch.object(ops, 'current_api', side_effect=lambda: copy.deepcopy(self.current)),
            patch.object(ops, 'save_runtime_image', side_effect=self.save_image)]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def backup_fixture(self, stamp):
        root = self.backups / stamp.strftime('%Y%m%dT%H%M%SZ')
        (root / 'postgres').mkdir(parents=True)
        files = {'application.tar': b'synthetic storage/table archive', 'local-files.tar': b'local data',
                 'arborscan.env.private': b'private config', 'containers.private.json': b'[]',
                 'restore-result.json': json.dumps({'files_restored': 4,
                     'production_credentials_loaded': False}).encode(),
                 'postgres/database.dump': b'PGDMP synthetic database dump',
                 'postgres/roles.sql': b'roles', 'postgres/source.json': b'{}',
                 'postgres/archive-list.private.txt': b'archive entries', 'postgres/COMPLETE': b''}
        for name, data in files.items():
            (root / name).write_bytes(data)
        (root / 'postgres/SHA256SUMS').write_text(''.join(
            ops.sha(root / 'postgres' / name) + '  ' + name + '\n'
            for name in ('database.dump', 'roles.sql', 'source.json', 'archive-list.private.txt')))
        files['postgres/SHA256SUMS'] = b''
        (root / 'SHA256SUMS').write_text(''.join(
            ops.sha(root / name) + '  ./' + name + '\n' for name in sorted(files)))
        (root / 'COMPLETE').touch()
        return root

    def row(self, image, revision, identity, started):
        return {'Id': identity, 'Image': image,
            'State': {'StartedAt': started, 'Running': True, 'Health': {'Status': 'healthy'}},
            'Config': {'Env': [key + '=' + value for key, value in self.env.items()],
                'Labels': {'org.opencontainers.image.revision': revision,
                    'com.docker.compose.project.config_files': str(self.compose),
                    'com.docker.compose.project': 'arborscan-v4',
                    'com.docker.compose.project.working_dir': str(self.folder),
                    'com.docker.compose.service': 'api-v4'}}}

    def inspect(self, name):
        if name in (CANDIDATE, 'candidate-tag'):
            return {'Id': CANDIDATE, 'Size': 10, 'Config': {'Env': ['PATH=/bin'],
                'Labels': {'org.opencontainers.image.revision': COMMIT}}}
        if name == ops.CURRENT_IMAGE:
            return {'Id': ops.CURRENT_IMAGE, 'Size': 10, 'Config': {'Env': ['PATH=/bin'],
                'Labels': {'org.opencontainers.image.revision': ops.CURRENT_REVISION}}}
        if name == ops.API:
            return copy.deepcopy(self.current)
        if name == 'arborscan-api':
            return copy.deepcopy(self.v3)
        if name == 'arborscan-quality-worker':
            return copy.deepcopy(self.worker)
        raise AssertionError('Unexpected Docker inspect target')

    def command(self, argv, **kwargs):
        if 'config' in argv:
            overlay = Path(argv[argv.index('config') - 1])
            return json.dumps(json.loads(overlay.read_text())).encode()
        if argv == ['docker', 'image', 'ls', '--no-trunc', '-q']:
            return (ops.CURRENT_IMAGE + '\n' + CANDIDATE + '\n').encode()
        raise AssertionError('Unexpected command')

    def save_image(self, root, row):
        (root / 'runtime-image.private.tar').write_bytes(b'exact synthetic saved image')

    def prepare(self):
        output = io.StringIO()
        with redirect_stdout(output):
            ops.prepare(self.root, self.backup, COMMIT, 'candidate-tag')
        self.assertNotIn(SECRET, output.getvalue())
        return json.loads((self.root / 'plan.private.json').read_text())

    def mutate(self, argv, **kwargs):
        self.assertEqual(argv[-8:], ['up', '-d', '--no-build', '--pull', 'never',
                                  '--no-deps', '--force-recreate', 'api-v4'])
        overlay = Path(argv[argv.index('up') - 1])
        target = json.loads(overlay.read_text())['services']['api-v4']['image']
        revision = COMMIT if target == CANDIDATE else ops.CURRENT_REVISION
        self.current = self.row(target, revision, 'new-' + target, 'new-start')
        result = type('Result', (), {'returncode': 0})()
        return result

    def apply(self, mode, runner=None):
        output = io.StringIO()
        with patch.object(ops.subprocess, 'run', side_effect=runner or self.mutate) as process:
            with redirect_stdout(output):
                ops.apply_mode(self.root, mode)
        self.assertNotIn(SECRET, output.getvalue())
        return process, json.loads(output.getvalue())

    def test_prepare_is_read_only_and_private_copies_are_verified(self):
        with patch.object(ops.subprocess, 'run') as process:
            plan = self.prepare()
        process.assert_not_called()
        self.assertEqual(plan['candidate'], CANDIDATE)
        manifest = json.loads((self.root / 'backup-manifest.private.json').read_text())
        for name, expected in manifest.items():
            self.assertEqual(ops.sha(self.root / name), expected)
        self.assertEqual((self.root / 'operator-config.env.private').read_bytes(), self.config.read_bytes())
        self.assertEqual(json.loads((self.root / 'runtime-before.private.json').read_text()), self.before)
        for mode in ('deploy', 'rollback'):
            override = json.loads((self.root / (mode + '.private.yml')).read_text())
            self.assertEqual(set(override['services']), {'api-v4'})
            self.assertEqual(override['services']['api-v4']['environment'], self.env)

    def test_candidate_commit_mismatch_stops_before_any_snapshot(self):
        with self.assertRaisesRegex(RuntimeError, 'revision does not match'):
            ops.prepare(self.root, self.backup, 'c' * 40, 'candidate-tag')
        self.assertFalse(self.root.exists())

    def test_current_api_identity_must_be_the_audited_runtime(self):
        self.current['Image'] = 'unknown image'
        with self.assertRaisesRegex(RuntimeError, 'verified current image'):
            self.prepare()
        self.assertFalse(self.root.exists())

    def test_changed_effective_weather_or_other_environment_refuses_prepare(self):
        for key in ('OPENWEATHER_API_KEY', 'DATABASE'):
            root = self.folder / ('failed-' + key)
            bad = {**self.env, key: 'changed'}
            with patch.object(ops, 'effective_environment', return_value=bad):
                with self.assertRaisesRegex(RuntimeError, 'effective runtime environment'):
                    ops.prepare(root, self.backup, COMMIT, 'candidate-tag')
            self.assertFalse((root / 'PREPARED').exists())

    def test_corrupt_private_backup_stops_before_compose(self):
        self.prepare()
        (self.root / 'runtime-image.private.tar').write_bytes(b'corrupt')
        with patch.object(ops.subprocess, 'run') as process:
            with self.assertRaisesRegex(RuntimeError, 'backup changed'):
                ops.apply_mode(self.root, 'deploy')
        process.assert_not_called()

    def test_plan_or_private_manifest_tampering_stops_before_compose(self):
        self.prepare()
        (self.root / 'backup-manifest.private.json').write_text('{}')
        with patch.object(ops.subprocess, 'run') as process:
            with self.assertRaisesRegex(RuntimeError, 'manifest changed'):
                ops.apply_mode(self.root, 'deploy')
        process.assert_not_called()

    def test_configuration_model_or_https_drift_stops_before_compose(self):
        self.prepare()
        for target, message in [(self.config, 'Operator configuration'),
                (self.compose, 'Compose configuration'),
                (self.models / 'baseline.pt', 'Model files'),
                (self.https / 'nginx.conf', 'HTTPS configuration')]:
            before = target.read_bytes()
            target.write_bytes(b'changed')
            with patch.object(ops.subprocess, 'run') as process:
                with self.assertRaisesRegex(RuntimeError, message):
                    ops.apply_mode(self.root, 'deploy')
            process.assert_not_called()
            target.write_bytes(before)

    def test_worker_v3_or_api_recreation_refuses_mutation(self):
        self.prepare()
        for row, message in [(self.worker, 'Worker changed'), (self.v3, 'API-v3 changed'),
                             (self.current, 'outside this plan')]:
            row['Id'] += '-externally-recreated'
            with patch.object(ops.subprocess, 'run') as process:
                with self.assertRaisesRegex(RuntimeError, message):
                    ops.apply_mode(self.root, 'deploy')
            process.assert_not_called()
            row['Id'] = row['Id'].removesuffix('-externally-recreated')

    def test_deploy_then_rollback_target_only_v4_and_keep_existing_weather(self):
        self.prepare()
        deployed, result = self.apply('deploy')
        self.assertEqual(deployed.call_count, 1)
        self.assertEqual(result['image'], CANDIDATE)
        self.assertEqual(self.current['Image'], CANDIDATE)
        self.assertEqual(ops.environment(self.current), self.env)
        rolled, result = self.apply('rollback')
        self.assertEqual(rolled.call_count, 1)
        self.assertEqual(result['image'], ops.CURRENT_IMAGE)
        self.assertEqual(ops.environment(self.current), self.env)
        self.assertEqual(self.v3['Id'], 'v3-id')
        self.assertEqual(self.worker['Id'], 'worker-id')
        self.assertEqual(self.config.read_text(), 'OPENWEATHER_API_KEY=' + SECRET + '\nDATABASE=unchanged\n')
        self.assertEqual(json.loads((self.root / 'state.private.json').read_text()), ops.runtime_identity(self.current))

    def test_failed_compose_with_own_new_container_preserves_reviewed_rollback(self):
        self.prepare()
        def failed(argv, **kwargs):
            result = self.mutate(argv, **kwargs)
            result.returncode = 1
            return result
        with patch.object(ops.subprocess, 'run', side_effect=failed):
            with self.assertRaisesRegex(RuntimeError, 'prepared rollback is available'):
                ops.apply_mode(self.root, 'deploy')
        self.assertEqual(json.loads((self.root / 'state.private.json').read_text()), ops.runtime_identity(self.current))
        self.apply('rollback')
        self.assertEqual(self.current['Image'], ops.CURRENT_IMAGE)

    def test_own_candidate_restart_allows_rollback_but_not_unreviewed_deploy(self):
        self.prepare()
        self.apply('deploy')
        self.current['State']['StartedAt'] = 'automatic crash restart'
        with patch.object(ops.subprocess, 'run') as process:
            with self.assertRaisesRegex(RuntimeError, 'outside this plan'):
                ops.apply_mode(self.root, 'deploy')
        process.assert_not_called()
        self.apply('rollback')

    def test_failed_compose_without_runtime_allows_only_own_prepared_rollback(self):
        self.prepare()
        def failed(argv, **kwargs):
            self.current = None
            return type('Result', (), {'returncode': 1})()
        with patch.object(ops.subprocess, 'run', side_effect=failed):
            with self.assertRaisesRegex(RuntimeError, 'absent after operation'):
                ops.apply_mode(self.root, 'deploy')
        with patch.object(ops.subprocess, 'run') as process:
            with self.assertRaisesRegex(RuntimeError, 'outside this plan'):
                ops.apply_mode(self.root, 'deploy')
        process.assert_not_called()
        self.apply('rollback')
        self.assertEqual(self.current['Image'], ops.CURRENT_IMAGE)

    def test_latest_recent_complete_backup_and_postgres_checks_are_required(self):
        self.assertEqual(ops.checked_backup(self.backup), self.backup)
        older = self.backup_fixture(datetime.now(timezone.utc) - timedelta(hours=37))
        with self.assertRaisesRegex(RuntimeError, 'latest complete backup'):
            ops.checked_backup(older)
        with self.assertRaisesRegex(RuntimeError, '36 hours'):
            ops.checked_backup(older, require_latest=False)
        (self.backup / 'postgres/database.dump').write_bytes(b'corrupt')
        with self.assertRaisesRegex(RuntimeError, 'checksum mismatch'):
            ops.checked_backup(self.backup)

    def test_missing_postgres_complete_and_manifest_omission_are_rejected(self):
        (self.backup / 'postgres/COMPLETE').unlink()
        with self.assertRaisesRegex(RuntimeError, 'incomplete'):
            ops.checked_backup(self.backup)
        (self.backup / 'postgres/COMPLETE').touch()
        manifest = self.backup / 'SHA256SUMS'
        manifest.write_text('\n'.join(line for line in manifest.read_text().splitlines()
                                     if not line.endswith('postgres/database.dump')) + '\n')
        with self.assertRaisesRegex(RuntimeError, 'omits required'):
            ops.checked_backup(self.backup)

    def test_manifest_cannot_escape_backup_or_duplicate_same_entry(self):
        target = self.folder / 'outside'
        target.write_bytes(b'outside')
        manifest = self.backup / 'SHA256SUMS'
        manifest.write_text(ops.sha(target) + '  ../outside\n')
        with self.assertRaisesRegex(RuntimeError, 'Unsafe or missing'):
            ops.verify_manifest(self.backup, manifest)
        p = self.backup / 'application.tar'
        manifest.write_text(ops.sha(p) + '  application.tar\n' + ops.sha(p) + '  ./application.tar\n')
        with self.assertRaisesRegex(RuntimeError, 'Unsafe or missing'):
            ops.verify_manifest(self.backup, manifest)

    def test_rollback_restores_missing_cached_original_from_verified_private_archive(self):
        self.prepare()
        self.apply('deploy')
        calls = []
        def docker(argv, **kwargs):
            calls.append(argv)
            if argv == ['docker', 'image', 'ls', '--no-trunc', '-q']:
                return (CANDIDATE + '\n').encode()
            if argv[:3] == ['docker', 'image', 'load']:
                self.assertEqual(Path(argv[-1]), self.root / 'runtime-image.private.tar')
                return b'loaded original image'
            return self.command(argv, **kwargs)
        with patch.object(ops, 'command', side_effect=docker):
            self.apply('rollback')
        self.assertEqual(sum(argv[:3] == ['docker', 'image', 'load'] for argv in calls), 1)
        self.assertEqual(self.current['Image'], ops.CURRENT_IMAGE)

    def test_compose_timeout_records_own_candidate_and_allows_rollback(self):
        self.prepare()
        def timeout(argv, **kwargs):
            self.mutate(argv, **kwargs)
            raise ops.subprocess.TimeoutExpired(argv, 90)
        with patch.object(ops.subprocess, 'run', side_effect=timeout):
            with self.assertRaisesRegex(RuntimeError, 'Compose failed/timed out'):
                ops.apply_mode(self.root, 'deploy')
        self.apply('rollback')
        self.assertEqual(self.current['Image'], ops.CURRENT_IMAGE)

    def test_unhealthy_candidate_never_reports_success_but_rollback_remains(self):
        self.prepare()
        def unhealthy(argv, **kwargs):
            result = self.mutate(argv, **kwargs)
            self.current['State']['Health']['Status'] = 'unhealthy'
            return result
        with patch.object(ops.subprocess, 'run', side_effect=unhealthy):
            with patch.object(ops.time, 'monotonic', side_effect=[100, 191]):
                with self.assertRaisesRegex(RuntimeError, 'health timeout'):
                    ops.apply_mode(self.root, 'deploy')
        self.assertEqual(json.loads((self.root / 'operation.private.json').read_text())['phase'], 'attempted')
        self.apply('rollback')

    def test_extra_candidate_image_environment_is_rejected(self):
        original = self.inspect
        def changed(name):
            image = original(name)
            if name in (CANDIDATE, 'candidate-tag'):
                image['Config']['Env'].append('NEW_DEFAULT=unreviewed')
            return image
        with patch.object(ops, 'inspect', side_effect=changed):
            with self.assertRaisesRegex(RuntimeError, 'effective runtime environment'):
                self.prepare()
        self.assertFalse((self.root / 'PREPARED').exists())

    def test_valid_checksums_do_not_make_a_plain_file_a_postgres_dump(self):
        pg = self.backup / 'postgres'
        (pg / 'database.dump').write_bytes(b'not a custom archive')
        for manifest, root in [(pg / 'SHA256SUMS', pg), (self.backup / 'SHA256SUMS', self.backup)]:
            names = [line.split('  ', 1)[1] for line in manifest.read_text().splitlines()]
            manifest.write_text(''.join(ops.sha(root / name) + '  ' + name + '\n' for name in names))
        with self.assertRaisesRegex(RuntimeError, 'custom archive header'):
            ops.checked_backup(self.backup)

    def test_failed_original_image_save_never_publishes_prepared_marker(self):
        with patch.object(ops, 'save_runtime_image', side_effect=RuntimeError('image backup failed')):
            with self.assertRaisesRegex(RuntimeError, 'image backup failed'):
                self.prepare()
        self.assertFalse((self.root / 'PREPARED').exists())

    def test_global_nonblocking_lock_refuses_a_concurrent_plan(self):
        def busy(*args):
            raise BlockingIOError()
        fcntl = SimpleNamespace(LOCK_EX=2, LOCK_NB=4, flock=busy)
        entered = False
        with patch.dict(sys.modules, {'fcntl': fcntl}), patch.object(ops, 'HOME', self.folder):
            with self.assertRaisesRegex(RuntimeError, 'Another API-v4 rollout'):
                with ops.operation_lock(self.root):
                    entered = True
        self.assertFalse(entered)
        self.assertTrue((self.folder / '.as16-v4-rollout.private.lock').is_file())


if __name__ == '__main__':
    unittest.main()
