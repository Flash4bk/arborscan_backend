"""AS-14/16 staging builder safety; Docker/SQL are fixtures, not a real restore."""
import base64
import hashlib
import hmac
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

source = Path(__file__).resolve().parents[1] / 'deploy-vps/ops_selfhost_stack.py'
spec = importlib.util.spec_from_file_location('selfhost_stack', source)
ops = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ops)


class SelfhostStagingSafetyTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.assets = self.root / 'assets'
        self.assets.mkdir()
        for name in ('models', 'model-quality', 'model-cache', 'u2net'):
            (self.assets / name).mkdir()
        self.images = {name: 'sha256:' + str(index) * 64
                       for index, name in enumerate(ops.PINNED, 1)}
        self.rows = {}
        for index, (service, name) in enumerate(ops.LIVE_NAMES.items(), 5):
            self.rows[service] = {'Name': name, 'Image': 'sha256:' + str(index) * 64,
                'Config': {'Env': ['PATH=/bin', 'SUPABASE_URL=https://restricted.invalid',
                    'SUPABASE_SERVICE_ROLE_KEY=old-secret',
                    'OPENWEATHER_API_KEY=synthetic${literal}$password=with\nnewline',
                    'PLANTNET_API_KEY=synthetic-key', 'MODEL_QUALITY_ENABLED=1']},
                'Mounts': [{'Type': 'bind', 'Source': '/opt/arborscan/models',
                            'Destination': '/app/models', 'RW': False},
                           {'Type': 'bind', 'Source': '/home/arborscan/model-quality',
                            'Destination': '/app/model-quality', 'RW': service == 'worker'}]}

    def plan(self):
        return ops.compose_plan(self.root / 'selfhost-20261012T170000Z', self.assets,
                                self.rows, self.images, 'a' * 64, 'b' * 64, issued=1000)

    def backup(self):
        root = self.root / 'backup'
        root.mkdir()
        (root / 'COMPLETE').touch()
        values = {'roles.sql': b'CREATE ROLE authenticator;\n',
                  'database.dump': b'native-dump-fixture', 'source.json': b'{"server_version":"17.6"}'}
        manifest = []
        for name, data in values.items():
            (root / name).write_bytes(data)
            manifest.append(hashlib.sha256(data).hexdigest() + '  ' + name)
        (root / 'SHA256SUMS').write_text('\n'.join(manifest))
        return root

    def snapshot(self):
        root = self.root / 'snapshot'
        file = root / 'objects' / 'private-bucket' / 'nested' / 'payload.json'
        file.parent.mkdir(parents=True)
        file.write_bytes(b'{"synthetic":"private"}')
        name = file.relative_to(root).as_posix()
        (root / 'manifest.json').write_text(json.dumps({name: hashlib.sha256(file.read_bytes()).hexdigest()}))
        return root, file

    def test_public_fingerprints_pin_cross_server_serialization_and_one_snapshot(self):
        run = Mock(return_value=SimpleNamespace(stdout=(
            '{"table":"auth_sessions","rows":29,"fingerprint":"' + 'a' * 32 + '"}\n'
            '{"table":"users","rows":9,"fingerprint":"' + 'b' * 32 + '"}\n').encode()))
        result = ops.public_fingerprints(run, ['psql', '-At'])
        self.assertEqual(result['auth_sessions']['rows'], 29)
        self.assertEqual(run.call_count, 1)
        command, query = run.call_args.args
        self.assertIn('-q', command)
        self.assertIn(b'REPEATABLE READ READ ONLY', query)
        self.assertIn(b"SET LOCAL TIME ZONE 'UTC'", query)
        self.assertIn(b'SET LOCAL extra_float_digits = 1', query)
        self.assertIn(b'COLLATE "C"', query)
        self.assertIn(b'\\gexec', query)
        self.assertTrue(query.rstrip().endswith(b'COMMIT;'))

    def test_public_fingerprints_reject_duplicate_negative_and_invalid_digest_records(self):
        valid = {'table': 'users', 'rows': 9, 'fingerprint': 'a' * 32}
        for records in ([], [valid, valid], [{**valid, 'rows': -1}],
                        [{**valid, 'fingerprint': 'unverified'}], [{**valid, 'table': 'unsafe-table'}]):
            with self.subTest(records=records):
                run = Mock(return_value=SimpleNamespace(stdout='\n'.join(json.dumps(row) for row in records).encode()))
                with self.assertRaises(ValueError):
                    ops.public_fingerprints(run, ['psql'])

    def test_storage_and_database_not_published_worker_requires_explicit_profile(self):
        plan = self.plan()
        self.assertTrue(plan['networks']['default']['internal'])
        self.assertNotIn('ports', plan['services']['db'])
        self.assertNotIn('ports', plan['services']['rest'])
        self.assertNotIn('ports', plan['services']['storage'])
        self.assertNotIn('providers', plan['services']['db'].get('networks', []))
        self.assertEqual(plan['services']['worker']['profiles'], ['manual-worker'])
        self.assertEqual(plan['services']['worker']['restart'], 'no')
        self.assertEqual(plan['services']['gateway']['ports'], ['127.0.0.1:19080:8000'])
        self.assertEqual(plan['services']['v3']['ports'], ['127.0.0.1:19000:8000'])
        self.assertEqual(plan['services']['api']['ports'], ['127.0.0.1:19001:8001'])
        self.assertEqual(plan['services']['api']['networks'], ['default', 'providers'])
        self.assertTrue(all(s['pull_policy'] == 'never' for s in plan['services'].values()))

    def test_external_provider_values_are_preserved_with_compose_dollar_escaping(self):
        value = self.plan()['services']['api']['environment']
        self.assertEqual(value['SUPABASE_URL'], 'http://gateway:8000')
        self.assertEqual(value['SUPABASE_SERVICE_ROLE_KEY'], value['SUPABASE_SERVICE_KEY'])
        self.assertEqual(value['OPENWEATHER_API_KEY'], 'synthetic$${literal}$$password=with\nnewline')
        self.assertEqual(value['PLANTNET_API_KEY'], 'synthetic-key')
        self.assertNotIn('old-secret', json.dumps(value))
        # Source inspect remains unchanged; no production env rewrite.
        self.assertIn('SUPABASE_SERVICE_ROLE_KEY=old-secret', self.rows['api']['Config']['Env'])

    def test_current_pinned_images_and_cloned_mount_readonly_are_retained(self):
        plan = self.plan()
        for service in self.rows:
            self.assertEqual(plan['services'][service]['image'], self.rows[service]['Image'])
            mounts = plan['services'][service]['volumes']
            self.assertTrue(mounts[0]['read_only'])
            self.assertEqual(mounts[1]['read_only'], service != 'worker')
            self.assertTrue(all(Path(m['source']).is_relative_to(self.assets) for m in mounts))

    def test_original_production_mount_and_unknown_mount_are_rejected(self):
        self.rows['api']['Mounts'][0]['Source'] = str(self.assets / 'models')
        with self.assertRaises(ValueError):
            self.plan()
        self.rows['api']['Mounts'][0]['Source'] = '/opt/arborscan/models'
        self.rows['api']['Mounts'].append({'Type': 'bind', 'Source': '/etc/arborscan',
                                         'Destination': '/etc/arborscan', 'RW': False})
        with self.assertRaises(ValueError):
            self.plan()

    def test_model_mount_must_be_readonly_and_clone_must_exist(self):
        self.rows['api']['Mounts'][0]['RW'] = True
        with self.assertRaises(ValueError):
            self.plan()
        self.rows['api']['Mounts'][0]['RW'] = False
        (self.assets / 'models').rmdir()
        with self.assertRaises(ValueError):
            self.plan()

    def test_jwt_signature_roles_expiry_and_minimum_key_strength(self):
        token = self.plan()['services']['storage']['environment']['SERVICE_KEY']
        header, payload, signature = token.split('.')
        decode = lambda value: base64.urlsafe_b64decode(value + '=' * (-len(value) % 4))
        self.assertEqual(json.loads(decode(header))['alg'], 'HS256')
        claims = json.loads(decode(payload))
        self.assertEqual(claims['role'], 'service_role')
        self.assertEqual(claims['exp'], 1000 + 3650 * 86400)
        self.assertEqual(decode(signature), hmac.new(b'a' * 64, (header + '.' + payload).encode(), hashlib.sha256).digest())
        with self.assertRaises(ValueError):
            ops.jwt('a' * 64, 'service_role', 1000, 3651)
        with self.assertRaises(ValueError):
            ops.compose_plan(self.root, self.assets, self.rows, self.images, 'short', 'b' * 64)

    def test_native_manifest_valid_bytes_and_tamper(self):
        root = self.backup()
        self.assertEqual(ops.verify_manifest(root, ('roles.sql', 'database.dump')), 3)
        (root / 'database.dump').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            ops.verify_manifest(root, ('database.dump',))

    def test_native_manifest_requires_complete_and_required_member(self):
        root = self.backup()
        with self.assertRaises(ValueError):
            ops.verify_manifest(root, ('missing',))
        (root / 'COMPLETE').unlink()
        with self.assertRaises(ValueError):
            ops.verify_manifest(root)

    def test_native_manifest_rejects_escape_and_duplicate(self):
        root = self.backup()
        manifest = root / 'SHA256SUMS'
        valid = manifest.read_text()
        manifest.write_text(valid + '\n' + 'a' * 64 + '  ../secret')
        with self.assertRaises(ValueError):
            ops.verify_manifest(root)
        manifest.write_text(valid + '\n' + valid.splitlines()[0])
        with self.assertRaises(ValueError):
            ops.verify_manifest(root)

    def test_storage_manifest_checks_real_bytes_extra_files_and_escape(self):
        root, file = self.snapshot()
        self.assertEqual(ops.verify_snapshot(root), 1)
        extra = file.with_name('unexpected.json')
        extra.write_bytes(b'extra')
        with self.assertRaises(ValueError):
            ops.verify_snapshot(root)
        extra.unlink()
        file.write_bytes(b'tampered')
        with self.assertRaises(ValueError):
            ops.verify_snapshot(root)
        (root / 'manifest.json').write_text(json.dumps({'../outside': 'a' * 64}))
        with self.assertRaises(ValueError):
            ops.verify_snapshot(root)

    def test_inspect_requires_all_three_current_images(self):
        path = self.root / 'inspect.private.json'
        path.write_text(json.dumps(list(self.rows.values())))
        self.assertEqual(set(ops.live_rows(path)), set(self.rows))
        path.write_text(json.dumps(list(self.rows.values())[:-1]))
        with self.assertRaises(ValueError):
            ops.live_rows(path)
        rows = list(self.rows.values())
        rows[0]['Image'] = 'latest'
        path.write_text(json.dumps(rows))
        with self.assertRaises(ValueError):
            ops.live_rows(path)

    def test_private_writes_cannot_replace_existing_config(self):
        path = self.root / 'compose.private.json'
        ops.private_write(path, 'original')
        with self.assertRaises(FileExistsError):
            ops.private_write(path, 'replacement')
        self.assertEqual(path.read_text(), 'original')
        if os.name != 'nt':
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_root_rejects_production_name_and_existing_plan(self):
        with self.assertRaises(ValueError):
            ops.checked_root(self.root / 'arborscan', parent=self.root)
        root = ops.checked_root(self.root / 'selfhost-20261012T170000Z', parent=self.root)
        ops.private_write(root / 'compose.private.json', '{}')
        with self.assertRaises(ValueError):
            ops.checked_root(root, parent=self.root)

    def test_roles_preserve_grants_and_reject_unknown_grantor(self):
        raw = ('CREATE ROLE supabase_admin;\nCREATE ROLE service_role;\n'
               'GRANT service_role TO authenticator GRANTED BY supabase_admin;\n')
        result = ops.role_restore_sql(raw)
        self.assertNotIn('CREATE ROLE supabase_admin;', result)
        self.assertIn('CREATE ROLE service_role;', result)
        self.assertIn('SET ROLE supabase_admin;\nGRANT service_role', result)
        with self.assertRaises(ValueError):
            ops.role_restore_sql(raw.replace('GRANTED BY supabase_admin', 'GRANTED BY unexpected'))

    def test_restore_preserves_sessions_jobs_and_checks_all_public_rows(self):
        backup = self.backup()
        calls = []
        def execute(args, data=None, **kwargs):
            calls.append((args, data, kwargs))
            raw = b'[1,1,1]' if data and b'SELECT json_build_array' in data else b''
            return SimpleNamespace(stdout=raw, returncode=0)
        runner = SimpleNamespace(run=execute)
        preserved = {'users': {'rows': 4, 'fingerprint': 'usershash'},
                     'auth_sessions': {'rows': 9, 'fingerprint': 'sessionshash'},
                     'ml_jobs': {'rows': 3, 'fingerprint': 'jobshash'}}
        with patch.object(ops, 'public_fingerprints', side_effect=[preserved, preserved]):
            result, versions = ops.restore_database(runner, ['docker', 'compose', '-f', 'new.private.json'], backup, 'b' * 64)
        self.assertEqual(result, preserved)
        self.assertEqual(versions, [1, 1, 1])
        queries = b'\n'.join(data for _, data, _ in calls if data)
        self.assertNotIn(b'DELETE ', queries)
        self.assertNotIn(b'UPDATE ', queries)
        self.assertIn(b'ALTER ROLE authenticator', queries)
        restore = next(call for call in calls if 'pg_restore' in call[0])
        self.assertIn('--single-transaction', restore[0])
        self.assertIn('--exit-on-error', restore[0])
        self.assertEqual(restore[1], b'native-dump-fixture')
        self.assertNotIn('--no-acl', restore[0])
        self.assertNotIn('--no-owner', restore[0])

    def test_restore_detects_changed_session_or_bad_migration(self):
        backup = self.backup()
        runner = SimpleNamespace(run=Mock(return_value=SimpleNamespace(stdout=b'[1,1,1]', returncode=0)))
        with patch.object(ops, 'public_fingerprints', side_effect=[{'sessions': 9}, {'sessions': 8}]):
            with self.assertRaises(RuntimeError):
                ops.restore_database(runner, ['docker', 'compose'], backup, 'b' * 64)
        runner.run.return_value.stdout = b'[1,0,1]'
        with patch.object(ops, 'public_fingerprints', return_value={'sessions': 9}):
            with self.assertRaises(RuntimeError):
                ops.restore_database(runner, ['docker', 'compose'], backup, 'b' * 64)

    def test_prepare_starts_only_new_private_infrastructure_and_never_pulls(self):
        backup = self.backup()
        snapshot, _ = self.snapshot()
        inventory = self.root / 'inspect.private.json'
        inventory.write_text(json.dumps(list(self.rows.values())))
        destination = self.root / 'selfhost-20261012T170000Z'
        calls = []
        def run(args, data=None, **kwargs):
            calls.append((args, data))
            if args[:3] == ['docker', 'image', 'inspect']:
                reference = args[3]
                identifier = next((self.images[name] for name, pin in ops.PINNED.items()
                                   if pin == reference), reference)
                return SimpleNamespace(stdout=identifier.encode(), returncode=0)
            return SimpleNamespace(stdout=b'', returncode=1 if args[:3] == ['docker', 'volume', 'inspect'] else 0)
        runner = SimpleNamespace(run=run)
        response = Mock()
        response.status = 200
        response.read.return_value = b'[]'
        context = Mock()
        context.__enter__ = Mock(return_value=response)
        context.__exit__ = Mock(return_value=False)
        args = SimpleNamespace(pg_backup=backup, storage_snapshot=snapshot, assets_root=self.assets,
                               live_inspect=inventory, root=destination, jwt_valid_days=3650)
        actual_chmod = os.chmod
        chmod_calls = []
        def chmod(path, mode):
            chmod_calls.append((Path(path), mode))
            actual_chmod(path, mode)
        with patch.object(ops, 'HOME', self.root), patch.object(ops, 'Runner', return_value=runner), \
                patch.object(ops, 'restore_database', return_value=({'auth_sessions': {'rows': 9}}, [1, 1, 1])), \
                patch.object(ops.urllib.request, 'urlopen', return_value=context), \
                patch.object(ops.os, 'chmod', side_effect=chmod):
            result = ops.prepare(args)
        starts = [command[command.index('up') + 1:] for command, _ in calls if 'up' in command]
        self.assertEqual(starts, [['-d', '--pull', 'never', 'db'],
                                  ['-d', '--pull', 'never', 'rest', 'storage', 'gateway']])
        self.assertFalse(result['api_started'])
        self.assertFalse(result['worker_started'])
        self.assertFalse(result['production_cutover'])
        self.assertFalse(result['storage_files_imported'])
        self.assertTrue((destination / 'PREPARED').is_file())
        safe = (destination / 'rootmanifest.safe.json').read_text()
        self.assertNotIn('synthetic-key', safe)
        self.assertNotIn('old-secret', safe)
        self.assertNotIn('JWT_SECRET', safe)
        self.assertNotIn('pgpass', safe)
        self.assertFalse(any(command[:2] == ['docker', 'pull'] for command, _ in calls))
        self.assertEqual(chmod_calls, [(destination / 'gateway.conf', 0o644)])
        self.assertNotIn('JWT', (destination / 'gateway.conf').read_text())
        if os.name != 'nt':
            self.assertEqual(destination.stat().st_mode & 0o777, 0o700)
            self.assertEqual((destination / 'gateway.conf').stat().st_mode & 0o777, 0o644)
            self.assertEqual((destination / 'compose.private.json').stat().st_mode & 0o777, 0o600)

    def test_prepare_does_not_reuse_existing_volume(self):
        backup = self.backup()
        snapshot, _ = self.snapshot()
        inventory = self.root / 'inspect.private.json'
        inventory.write_text(json.dumps(list(self.rows.values())))
        destination = self.root / 'selfhost-20261012T170000Z'
        calls = []
        def run(command, **kwargs):
            calls.append(command)
            if command[:3] == ['docker', 'image', 'inspect']:
                reference = command[3]
                identifier = next((self.images[name] for name, pin in ops.PINNED.items()
                                   if pin == reference), reference)
                return SimpleNamespace(stdout=identifier.encode(), returncode=0)
            return SimpleNamespace(stdout=b'pre-existing', returncode=0)
        args = SimpleNamespace(pg_backup=backup, storage_snapshot=snapshot, assets_root=self.assets,
                               live_inspect=inventory, root=destination, jwt_valid_days=3650)
        with patch.object(ops, 'HOME', self.root), patch.object(ops, 'Runner', return_value=SimpleNamespace(run=run)):
            with self.assertRaises(ValueError):
                ops.prepare(args)
        self.assertFalse((destination / 'compose.private.json').exists())
        self.assertFalse(any('up' in command or command[:3] == ['docker', 'volume', 'create'] for command in calls))

    def infrastructure(self, plan):
        rows = []
        for name in ('db', 'rest', 'storage', 'gateway'):
            networks = {plan['name'] + '_default': {}}
            if name == 'gateway':
                networks[plan['name'] + '_providers'] = {}
            rows.append({'Image': plan['services'][name]['image'], 'State': {'Running': True},
                'Config': {'Labels': {'com.docker.compose.project': plan['name'], 'com.docker.compose.service': name}},
                'NetworkSettings': {'Networks': networks},
                'HostConfig': {'NetworkMode': plan['name'] + '_default',
                              'PortBindings': {'8000/tcp': [{'HostIp': '127.0.0.1', 'HostPort': '19080'}]} if name == 'gateway' else {}}})
        return rows

    def test_resume_refuses_running_api_worker_external_database_and_public_gateway(self):
        plan = self.plan()
        rows = self.infrastructure(plan)
        runner = SimpleNamespace(run=Mock(side_effect=[SimpleNamespace(stdout=b'container-id'), SimpleNamespace(stdout=json.dumps(rows).encode())]))
        ops.checked_infrastructure(runner, plan)
        for service in ('api', 'worker'):
            with self.subTest(service=service):
                modified = self.infrastructure(plan)
                modified.append({'Config': {'Labels': {'com.docker.compose.project': plan['name'], 'com.docker.compose.service': service}},
                                 'State': {'Running': True}})
                runner.run = Mock(side_effect=[SimpleNamespace(stdout=b'container-id'), SimpleNamespace(stdout=json.dumps(modified).encode())])
                with self.assertRaises(ValueError):
                    ops.checked_infrastructure(runner, plan)
        rows = self.infrastructure(plan)
        rows[0]['NetworkSettings']['Networks'][plan['name'] + '_providers'] = {}
        runner.run = Mock(side_effect=[SimpleNamespace(stdout=b'container-id'), SimpleNamespace(stdout=json.dumps(rows).encode())])
        with self.assertRaises(ValueError):
            ops.checked_infrastructure(runner, plan)
        rows = self.infrastructure(plan)
        rows[-1]['HostConfig']['PortBindings']['8000/tcp'][0]['HostIp'] = '0.0.0.0'
        runner.run = Mock(side_effect=[SimpleNamespace(stdout=b'container-id'), SimpleNamespace(stdout=json.dumps(rows).encode())])
        with self.assertRaises(ValueError):
            ops.checked_infrastructure(runner, plan)

    def test_resume_token_rejects_tamper_wrong_role_and_expiry(self):
        issued = int(time.time())
        token = ops.jwt('a' * 64, 'service_role', issued, 3650)
        self.assertEqual(ops.checked_jwt(token, 'a' * 64, 'service_role')['iat'], issued)
        for invalid in (token + 'bad', ops.jwt('b' * 64, 'service_role', issued, 3650),
                        ops.jwt('a' * 64, 'anon', issued, 3650), ops.jwt('a' * 64, 'service_role', issued - 86401, 1)):
            with self.subTest(invalid=invalid[:10]):
                with self.assertRaises(ValueError):
                    ops.checked_jwt(invalid, 'a' * 64, 'service_role')

    def test_resume_finalizes_only_after_source_fingerprint_and_readiness_checks_no_database_writes(self):
        backup = self.backup()
        snapshot, _ = self.snapshot()
        inventory = self.root / 'inspect.private.json'
        inventory.write_text(json.dumps(list(self.rows.values())))
        destination = self.root / 'selfhost-20261012T170000Z'
        destination.mkdir()
        plan = ops.compose_plan(destination, self.assets, self.rows, self.images, 'a' * 64, 'b' * 64)
        ops.private_write(destination / 'compose.private.json', json.dumps(plan))
        ops.private_write(destination / 'gateway.conf', ops.GATEWAY)
        fingerprints = self.root / 'source-fingerprints.private.json'
        baseline = {'auth_sessions': {'rows': 9, 'fingerprint': 'preserved'}}
        fingerprints.write_text(json.dumps(baseline))
        calls = []
        def execute(command, data=None):
            calls.append((command, data))
            if command[:3] == ['docker', 'image', 'inspect']:
                reference = command[3]
                identifier = next(self.images[name] for name, pin in ops.PINNED.items() if pin == reference)
                return SimpleNamespace(stdout=identifier.encode())
            return SimpleNamespace(stdout=b'[1,1,1]')
        args = SimpleNamespace(pg_backup=backup, storage_snapshot=snapshot, assets_root=self.assets,
                               live_inspect=inventory, root=destination, expected_public_fingerprints=fingerprints)
        with patch.object(ops, 'HOME', self.root), patch.object(ops, 'checked_resume_root', return_value=destination), \
                patch.object(ops, 'checked_private_file'), patch.object(ops, 'Runner', return_value=SimpleNamespace(run=execute)), \
                patch.object(ops, 'checked_infrastructure') as infrastructure, \
                patch.object(ops, 'public_fingerprints', return_value=baseline), patch.object(ops, 'storage_ready') as ready:
            result = ops.verify_prepared(args)
            ready.assert_called_once()
            infrastructure.assert_called_once()
        self.assertTrue(result['sessions_preserved'])
        self.assertTrue((destination / 'PREPARED').is_file())
        self.assertFalse(any('pg_restore' in command or 'up' in command or 'create' in command for command, _ in calls))
        sql = [data for _, data in calls if data]
        self.assertTrue(all(data.startswith(b'SELECT ') for data in sql))
        self.assertEqual(json.loads((destination / 'rootmanifest.safe.json').read_text())['public_tables'], baseline)
        # Changed data must not be blessed, even when the infrastructure is healthy.
        (destination / 'PREPARED').unlink()
        (destination / 'rootmanifest.safe.json').unlink()
        with patch.object(ops, 'HOME', self.root), patch.object(ops, 'checked_resume_root', return_value=destination), \
                patch.object(ops, 'checked_private_file'), patch.object(ops, 'Runner', return_value=SimpleNamespace(run=execute)), \
                patch.object(ops, 'checked_infrastructure'), \
                patch.object(ops, 'public_fingerprints', return_value={'auth_sessions': {'rows': 8}}), patch.object(ops, 'storage_ready') as ready:
            with self.assertRaises(RuntimeError):
                ops.verify_prepared(args)
            ready.assert_not_called()
        self.assertFalse((destination / 'PREPARED').exists())


if __name__ == '__main__':
    unittest.main()
