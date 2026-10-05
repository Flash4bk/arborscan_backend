"""Safety regressions for the AS-09 API-v4-only operator weather refresh.

All Docker calls/configuration are mocked; fixtures are temporary local files.
These tests do not constitute a production rollout or PostgreSQL restore.
"""
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


source = Path(__file__).resolve().parents[1]/'deploy-vps/ops_weather_v4_refresh.py'
spec = importlib.util.spec_from_file_location('weather_refresh', source)
ops = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ops)


class WeatherRefreshSafetyTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.config = self.folder/'operator.env'
        self.config.write_text('OPENWEATHER_API_KEY=synthetic-test-value\n')
        self.compose = self.folder/'compose.yml'
        self.compose.write_text('services: {api-v4: {image: pinned}}\n')
        self.root = self.folder/'private-plan'
        self.before = self.row({'PATH': '/bin', 'DATABASE_PLACEHOLDER': 'unchanged'})
        self.patches = [
            patch.object(ops, 'CONFIG', self.config),
            patch.object(ops.os, 'fchmod', create=True),
            patch.object(ops, 'checked_backup', return_value=self.folder/'complete-backup'),
            patch.object(ops, 'invariant', return_value={'image': 'unchanged', 'started_at': 'same'}),
            patch.object(ops, 'inspect', side_effect=self.inspect),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def row(self, environment):
        return {'Image': ops.IMAGE, 'State': {'StartedAt': 'initial', 'Running': True,
                'Health': {'Status': 'healthy'}}, 'Config': {
                'Env': [key+'='+value for key, value in environment.items()], 'Labels': {
                    'org.opencontainers.image.revision': ops.REVISION,
                    'com.docker.compose.project.config_files': str(self.compose),
                    'com.docker.compose.project': 'arborscan'}}}

    def inspect(self, name):
        if name == ops.OLD_APPLICATION:
            return {'Id': ops.OLD_APPLICATION, 'Config': {'Env': ['PATH=/bin']}}
        return self.before

    def effective(self, plan, override):
        value = {'PATH': '/bin', 'DATABASE_PLACEHOLDER': 'unchanged'}
        if override.name.startswith('rollback-weather'):
            value.update({key: '' for key in ops.ALIASES})
        else:
            value['OPENWEATHER_API_KEY'] = 'synthetic-test-value'
        return value

    def prepare(self):
        with patch.object(ops, 'effective_environment', side_effect=self.effective):
            with redirect_stdout(io.StringIO()):
                ops.prepare(self.root, self.folder/'complete-backup')

    def test_prepare_verifies_private_copies_and_never_recreates_container(self):
        with patch.object(ops.subprocess, 'run') as runner:
            self.prepare()
        runner.assert_not_called()
        self.assertTrue((self.root/'PREPARED').is_file())
        manifest = json.loads((self.root/'backup-manifest.private.json').read_text())
        for file, digest in manifest.items():
            self.assertEqual(ops.sha(self.root/file), digest)
        self.assertEqual(json.loads((self.root/'runtime-before.private.json').read_text()), self.before)
        self.assertEqual(self.config.read_bytes(), (self.root/'operator-config.env.private').read_bytes())

    def test_corrupt_copy_does_not_create_prepared_marker(self):
        real_sha = ops.sha
        def altered(path):
            return 'incorrect' if Path(path).name == 'operator-config.env.private' else real_sha(path)
        with patch.object(ops, 'sha', side_effect=altered):
            with self.assertRaisesRegex(RuntimeError, 'copy failed verification'):
                self.prepare()
        self.assertFalse((self.root/'PREPARED').exists())

    def test_nonweather_environment_change_is_rejected(self):
        def altered(plan, override):
            value = self.effective(plan, override)
            value['DATABASE_PLACEHOLDER'] = 'unexpected-change'
            return value
        with patch.object(ops, 'effective_environment', side_effect=altered):
            with self.assertRaisesRegex(RuntimeError, 'extend beyond weather'):
                ops.prepare(self.root, self.folder/'complete-backup')
        self.assertFalse((self.root/'PREPARED').exists())

    def test_missing_effective_key_is_rejected(self):
        with patch.object(ops, 'effective_environment', return_value={'PATH': '/bin'}):
            with self.assertRaisesRegex(RuntimeError, 'key is not present'):
                ops.prepare(self.root, self.folder/'complete-backup')
        self.assertFalse((self.root/'PREPARED').exists())

    def test_unverified_image_is_rejected_before_snapshot(self):
        self.before['Image'] = 'unverified-image'
        with self.assertRaisesRegex(RuntimeError, 'differs from verified'):
            self.prepare()
        self.assertFalse(self.root.exists())

    def test_rollbacks_clear_aliases_only_for_weather_and_keep_shared_config(self):
        self.prepare()
        weather = json.loads((self.root/'rollback-weather.private.yml').read_text())
        self.assertEqual(weather, {'services': {'api-v4': {
            'image': ops.IMAGE, 'environment': {key: '' for key in ops.ALIASES}}}})
        app = json.loads((self.root/'rollback-application.private.yml').read_text())
        self.assertEqual(app, {'services': {'api-v4': {'image': ops.OLD_APPLICATION}}})
        self.assertEqual(self.config.read_text(), 'OPENWEATHER_API_KEY=synthetic-test-value\n')

    def test_changed_operator_config_stops_before_any_mutation(self):
        self.prepare()
        self.config.write_text('unexpected operator edit')
        with patch.object(ops.subprocess, 'run') as runner:
            with self.assertRaisesRegex(RuntimeError, 'Operator config changed'):
                ops.apply_mode(self.root, 'deploy')
        runner.assert_not_called()

    def test_deploy_targets_only_v4_pinned_image_with_bounded_recreate(self):
        self.prepare()
        plan = json.loads((self.root/'plan.private.json').read_text())
        env = self.effective(plan, self.root/'deploy.private.yml')
        after = self.row(env)
        after['State']['StartedAt'] = 'new-container'
        with patch.object(ops, 'effective_environment', return_value=env):
            with patch.object(ops, 'inspect', side_effect=[self.before, after, after]):
                with patch.object(ops.subprocess, 'run') as runner:
                    runner.return_value.returncode = 0
                    capture = io.StringIO()
                    with redirect_stdout(capture):
                        ops.apply_mode(self.root, 'deploy')
        argv = runner.call_args.args[0]
        self.assertEqual(argv[-8:], ['up', '-d', '--no-build', '--pull', 'never',
                                    '--no-deps', '--force-recreate', 'api-v4'])
        self.assertEqual(runner.call_args.kwargs['timeout'], 90)
        self.assertNotIn('synthetic-test-value', str(argv)+capture.getvalue())
        self.assertNotIn('arborscan-api', argv)
        self.assertNotIn('arborscan-quality-worker', argv)
        self.assertEqual(json.loads((self.root/'state.private.json').read_text()),
                         ops.runtime_identity(after))


if __name__ == '__main__':
    unittest.main()
