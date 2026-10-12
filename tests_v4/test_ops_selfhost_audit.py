"""Self-host audit safeguards; all fixtures are tiny and locally isolated.

These tests check the audit tool, not an actual migrated PostgreSQL database.
Real stack/database/HTTP checks are recorded separately by its CLI runner.
"""
import copy
import base64
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import uuid

sys.path.insert(0, str(Path(__file__).parents[1] / 'deploy-vps'))
import ops_selfhost_audit as audit


def fingerprints():
    rows = [{'table': name, 'rows': 1, 'sha256': audit.digest_bytes(name.encode())}
            for name in ('users', 'auth_sessions', 'report_versions', 'contour_revisions')]
    rows += [{'versions': {'contours': 1, 'history': 1, 'quality': 1}},
             {'relationships': {'orphan_sessions': 0, 'orphan_reports': 0, 'orphan_contours': 0}}]
    return rows


def raw(rows):
    return '\n'.join(json.dumps(row) for row in rows).encode()


class AuditDataTests(unittest.TestCase):
    def storage_fingerprints(self):
        return [{'storage_table': name, 'rows': count, 'sha256': audit.digest_bytes(name.encode())}
                for name, count in (('objects', 2252), ('buckets', 4))]

    def test_all_storage_metadata_columns_preserved_not_only_blob_keys(self):
        expected = audit.parse_storage_fingerprints(raw(self.storage_fingerprints()))
        actual = audit.parse_storage_fingerprints(raw(list(reversed(self.storage_fingerprints()))))
        result = audit.compare_storage_database(expected, actual)
        self.assertEqual(result['objects'], 2252)
        self.assertEqual(result['all_columns_including_owners_versions_timestamps_metadata'], 'identical')
        # Same rows/key counts are insufficient when an import replaces IDs,
        # ownership, object versions, creation dates or content metadata.
        changed = copy.deepcopy(actual)
        changed['objects']['sha256'] = '0' * 64
        with self.assertRaisesRegex(audit.AuditFailure, 'storage_metadata_continuity_mismatch'):
            audit.compare_storage_database(expected, changed)
        self.assertIn('to_jsonb(t)', audit.STORAGE_FULL_FINGERPRINT_SQL)
        self.assertIn('FROM storage.buckets', audit.STORAGE_FULL_FINGERPRINT_SQL)

    def test_full_storage_fingerprint_requires_both_tables_and_valid_exact_counts(self):
        examples = [self.storage_fingerprints()[:1], self.storage_fingerprints() * 2,
                    [{**self.storage_fingerprints()[0], 'rows': True}, self.storage_fingerprints()[1]],
                    [{**self.storage_fingerprints()[0], 'sha256': None}, self.storage_fingerprints()[1]]]
        for rows in examples:
            with self.subTest(length=len(rows)), self.assertRaises(audit.AuditFailure):
                audit.parse_storage_fingerprints(raw(rows))

    def test_source_restore_fingerprints_order_independent_and_accounts_exact(self):
        expected = audit.parse_fingerprints(raw(fingerprints()))
        actual = audit.parse_fingerprints(raw(list(reversed(fingerprints()))))
        summary = audit.compare_database(expected, actual)
        self.assertEqual(summary['accounts_password_hashes_sessions_owners'], 'identical')
        self.assertEqual(summary['tables'], 4)
        actual['tables']['auth_sessions']['sha256'] = '0' * 64
        with self.assertRaisesRegex(audit.AuditFailure, 'database_continuity_mismatch'):
            audit.compare_database(expected, actual)

    def test_deleted_account_and_changed_moderation_are_not_ignored(self):
        expected = audit.parse_fingerprints(raw(fingerprints()))
        for table in ('users', 'contour_revisions'):
            with self.subTest(table=table):
                actual = copy.deepcopy(expected)
                actual['tables'][table]['rows'] = 0
                with self.assertRaises(audit.AuditFailure):
                    audit.compare_database(expected, actual)

    def test_incomplete_duplicate_non_integer_versions_and_orphans_fail(self):
        examples = [fingerprints()[1:], fingerprints() + [fingerprints()[0]]]
        wrong_version = fingerprints()
        wrong_version[-2]['versions']['history'] = True
        examples.append(wrong_version)
        orphan = fingerprints()
        orphan[-1]['relationships']['orphan_sessions'] = 1
        examples.append(orphan)
        for rows in examples:
            with self.subTest(rows=len(rows)), self.assertRaises(audit.AuditFailure):
                audit.parse_fingerprints(raw(rows))

    def test_readonly_repeatable_fingerprints_include_whole_rows(self):
        self.assertIn('REPEATABLE READ READ ONLY', audit.FINGERPRINT_SQL)
        self.assertIn("TIME ZONE 'UTC'", audit.FINGERPRINT_SQL)
        self.assertIn('sha256(convert_to', audit.FINGERPRINT_SQL)
        self.assertIn('to_jsonb(t)', audit.FINGERPRINT_SQL)
        self.assertIn('extra_float_digits = 1', audit.FINGERPRINT_SQL)
        self.assertIn('COLLATE "C"', audit.FINGERPRINT_SQL)
        self.assertNotIn('DELETE ', audit.FINGERPRINT_SQL)
        self.assertNotIn('UPDATE ', audit.FINGERPRINT_SQL)

    def test_storage_inventory_rejects_missing_extra_wrong_sizes(self):
        expected = [{'bucket': 'private', 'name': 'reports/test.json', 'size': 5, 'sha256': 'a' * 64}]
        exact = [{'bucket': 'private', 'name': 'reports/test.json', 'size': 5}]
        audit.compare_storage_keys(expected, json.dumps(exact).encode())
        for actual in ([], exact + [{'bucket': 'private', 'name': 'extra', 'size': 0}],
                       [{**exact[0], 'size': 4}]):
            with self.subTest(size=len(actual)), self.assertRaisesRegex(audit.AuditFailure, 'storage_inventory_mismatch'):
                audit.compare_storage_keys(expected, json.dumps(actual).encode())

    def test_manifest_rejects_traversal_duplicates_and_invalid_lengths(self):
        row = {'bucket': 'private', 'name': 'test/file', 'size': 3, 'sha256': 'a' * 64}
        for name in ('../file', '/file', 'a//file', 'a/./file', 'a\\file'):
            with self.subTest(name=name), self.assertRaises(audit.AuditFailure):
                audit.storage_manifest([{**row, 'name': name}])
        for bad in ({**row, 'size': True}, {**row, 'size': -1}, {**row, 'sha256': 'a' * 63}):
            with self.assertRaises(audit.AuditFailure):
                audit.storage_manifest([bad])
        with self.assertRaisesRegex(audit.AuditFailure, 'duplicate_object_manifest'):
            audit.storage_manifest([row, row])

    def test_google_and_model_policy_hashes_preserve_decision_changes(self):
        env = {'GOOGLE_CLIENT_ID': 'not-a-secret-test', 'GOOGLE_ALLOW_CREATE': 'false',
               'ACTIVE_MODEL_VERSION': '4', 'SUPABASE_URL': 'remote', 'SUPABASE_SERVICE_KEY': 'fixture-secret'}
        policy = audit.protected_policy(env)
        changed_storage = {**env, 'SUPABASE_URL': 'http://gateway:8000', 'SUPABASE_SERVICE_KEY': 'local'}
        self.assertEqual(policy, audit.protected_policy(changed_storage))
        self.assertNotEqual(policy, audit.protected_policy({**env, 'GOOGLE_ALLOW_CREATE': 'true'}))
        self.assertNotIn('fixture-secret', json.dumps(policy))
        self.assertNotIn('not-a-secret-test', json.dumps(policy))

    def test_compose_literal_dollars_are_unescaped_once_without_environment_evaluation(self):
        raw_env = {'GOOGLE_TEST_POLICY': '${NOT_EXPANDED}$value$$end'}
        compose_env = {key: value.replace('$', '$$') for key, value in raw_env.items()}
        self.assertEqual(audit.protected_policy(raw_env), audit.protected_policy(compose_env, compose_values=True))
        self.assertEqual(audit.compose_literal(compose_env['GOOGLE_TEST_POLICY']), raw_env['GOOGLE_TEST_POLICY'])

    def test_legacy_absolute_url_audit_never_exposes_private_paths_or_domains(self):
        rows = [{'url_table': 'analyses', 'urls': ['https://private-project.supabase.co/storage/v1/object/public/bucket/secret-owner?token=private',
                                                  'https://31.57.170.88/api/v4/image']},
                {'url_table': 'users', 'urls': ['https://lh3.googleusercontent.com/private-account']},
                {'url_table': 'users.avatar_url', 'urls': ['https://private-project.supabase.co/storage/v1/object/public/avatar/private']}]
        result = audit.classify_url_records(raw(rows))
        self.assertEqual(result['absolute_urls'], 3)
        self.assertEqual(result['old_supabase_urls'], 1)
        self.assertEqual(result['old_supabase_avatar_urls'], 1)
        serialized = json.dumps(result)
        for private in ('private-project', 'secret-owner', 'token=', 'private-account'):
            self.assertNotIn(private, serialized)
        payload = {'nested': [{'link': rows[0]['urls'][0]}], 'base64': 'aGVsbG8='}
        self.assertEqual(audit.payload_url_counts(payload), {'old_supabase': 1})

    def test_loopback_only_no_credentials_public_host_https_or_redirect_base(self):
        self.assertEqual(audit.loopback_base('http://127.0.0.1:19080/'), 'http://127.0.0.1:19080')
        for url in ('https://31.57.170.88', 'http://localhost:19080', 'https://127.0.0.1:19080',
                    'http://user:pass@127.0.0.1:19080', 'http://127.0.0.1:19080/a',
                    'http://127.0.0.1:19080?next=public'):
            with self.subTest(url=url), self.assertRaises(audit.AuditFailure):
                audit.loopback_base(url)


class AuditFilesystemTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.home = Path(self.temp.name)
        self.root = self.home / 'selfhost-20261012T000000Z'
        self.root.mkdir(mode=0o700)
        self.assets = self.home / 'cloned-private-assets'
        self.assets.mkdir()
        self.config = {'services': {key: {} for key in ('db', 'rest', 'storage', 'gateway', 'v3', 'api', 'worker')},
                       'volumes': {'db': {}, 'storage': {}}}
        self.config['services']['gateway'] = {'ports': ['127.0.0.1:19080:8000']}
        for name in ('v3', 'api', 'worker'):
            self.config['services'][name]['environment'] = {'SUPABASE_URL': 'http://gateway:8000'}

    def tearDown(self):
        self.temp.cleanup()

    def test_staging_shape_allows_only_cloned_private_mounts(self):
        self.config['services']['api']['volumes'] = [{'source': str(self.assets / 'models'), 'target': '/app/models'}]
        self.assertEqual(audit.guarded_stack(self.root, self.config, self.assets, self.home), 'http://127.0.0.1:19080')
        self.config['services']['api']['volumes'] = [{'source': '/opt/arborscan/models', 'target': '/app/models'}]
        with self.assertRaisesRegex(audit.AuditFailure, 'production_mount_forbidden'):
            audit.guarded_stack(self.root, self.config, self.assets, self.home)

    def test_public_ports_host_network_external_volumes_and_live_cloud_fail(self):
        changes = [lambda c: c['services']['gateway'].update(ports=['19080:8000']),
                   lambda c: c['services']['api'].update(network_mode='host'),
                   lambda c: c['volumes']['db'].update(external=True),
                   lambda c: c['services']['v3']['environment'].update(SUPABASE_URL='https://live.invalid')]
        for change in changes:
            c = copy.deepcopy(self.config)
            change(c)
            with self.assertRaises(audit.AuditFailure):
                audit.guarded_stack(self.root, c, None, self.home)

    def test_private_output_atomic_replaced_and_readable(self):
        path = self.root / 'fixture.private.json'
        audit.write_private(path, {'fixture': 1})
        audit.write_private(path, {'fixture': 2})
        self.assertEqual(audit.private_json(path), {'fixture': 2})
        self.assertEqual(list(self.root.glob('*.tmp-*')), [])

    def test_build_expected_checks_actual_seed_bytes_models_and_live_images(self):
        snapshot = self.root / 'seed'
        file = snapshot / 'objects' / 'private' / 'report-versions' / 'demo.json'
        file.parent.mkdir(parents=True)
        file.write_bytes(b'fixture')
        (snapshot / 'manifest.json').write_text(json.dumps({'objects/private/report-versions/demo.json': audit.digest_bytes(b'fixture')}))
        models = self.assets / 'models'
        models.mkdir()
        (models / 'fixture.pt').write_bytes(b'not-a-model')
        inspected = [{'Name': name, 'Image': 'sha256:' + str(index) * 64,
                      'Config': {'Env': ['GOOGLE_ALLOW_CREATE=false', 'ACTIVE_MODEL_VERSION=4']}}
                     for index, name in enumerate(('/arborscan-api', '/arborscan-api-v4', '/arborscan-quality-worker'), 1)]
        storage_database = audit.parse_storage_fingerprints(raw([
            {'storage_table': 'objects', 'rows': 1, 'sha256': 'a' * 64},
            {'storage_table': 'buckets', 'rows': 1, 'sha256': 'b' * 64}]))
        expected = audit.build_expected(audit.parse_fingerprints(raw(fingerprints())), snapshot, inspected, self.assets, storage_database)
        self.assertEqual(expected['storage'][0]['size'], 7)
        self.assertEqual(expected['storage_database'], storage_database)
        self.assertEqual(expected['models']['fixture.pt'], audit.digest_bytes(b'not-a-model'))
        file.write_bytes(b'changed')
        with self.assertRaisesRegex(audit.AuditFailure, 'seed_sha256_mismatch'):
            audit.build_expected(expected['database'], snapshot, inspected, self.assets)


class FixtureTests(unittest.TestCase):
    def fixture(self):
        return {'format': 1, 'tag': audit.FIXTURE_NAME, 'accounts': [
            {'id': str(uuid.uuid4()), 'name': audit.FIXTURE_NAME,
             'email': 'demo-selfhost-' + uuid.uuid4().hex + '@invalid.test', 'password': 'fixture', 'token': 'fixture'}
            for _ in range(2)]}

    def test_only_minted_tagged_test_accounts_pass(self):
        valid = self.fixture()
        audit.validate_fixture(valid)
        for field, value in (('email', 'actual-user@example.invalid'), ('name', 'actual user'), ('token', '')):
            state = copy.deepcopy(valid)
            state['accounts'][0][field] = value
            with self.subTest(field=field), self.assertRaises(audit.AuditFailure):
                audit.validate_fixture(state)
        invalid = copy.deepcopy(valid)
        invalid['accounts'][1]['id'] = invalid['accounts'][0]['id']
        with self.assertRaises(audit.AuditFailure):
            audit.validate_fixture(invalid)

    def test_interrupted_smoke_is_retained_and_not_claimed_as_restart_pass(self):
        # No HTTP operation may occur when partially initialized fixture state
        # exists; no real account is substituted to make the result pass.
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'fixture.private.json'
            audit.write_private(path, self.fixture())
            with patch.dict(sys.modules, {'PIL': __import__('types').SimpleNamespace(Image=None, ImageDraw=None)}):
                with self.assertRaisesRegex(audit.AuditFailure, 'incomplete_fixture_retained_for_review'):
                    audit.smoke(None, path)
            self.assertTrue(path.exists())

    def test_repeat_cli_uses_builder_manifest_without_source_expected_argument(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            audit.write_private(root / 'compose.private.json', {'services': {
                'api': {'environment': {'SUPABASE_SERVICE_KEY': 'fixture'}, 'ports': ['127.0.0.1:19001:8001']},
                'v3': {'ports': ['127.0.0.1:19000:8000']}}})
            (root / 'rootmanifest.safe.json').write_text(json.dumps({'assets_root': 'fixture-assets'}))
            audit.write_private(root / 'fixture.private.json', {'fixture': 'already-created'})
            with patch.object(sys, 'argv', ['ops_selfhost_audit.py', '--root', str(root), '--repeat-smoke']), \
                 patch.object(audit, 'guarded_stack', return_value='http://127.0.0.1:19080') as guard, \
                 patch.object(audit, 'StageHTTP'), \
                 patch.object(audit, 'smoke', return_value={'restart': 'passed'}), \
                 patch('builtins.print'):
                audit.main()
            self.assertEqual(guard.call_args.args[2], 'fixture-assets')
            self.assertEqual(audit.private_json(root / 'audit-result.json')['smoke']['restart'], 'passed')


class ActualContainerTests(unittest.TestCase):
    def fixture(self):
        config = {'name': 'selfhost-fixture', 'services': {}}
        rows = []
        for name in ('db', 'rest', 'storage', 'gateway', 'v3', 'api'):
            env = {'SUPABASE_URL': 'http://gateway:8000'}
            image = 'sha256:' + audit.digest_bytes(name.encode())
            config['services'][name] = {'image': image, 'environment': env}
            rows.append({'Image': image, 'State': {'Running': True},
                         'Config': {'Labels': {'com.docker.compose.service': name,
                                              'com.docker.compose.project': config['name']},
                                    'Env': [key + '=' + value for key, value in env.items()]},
                         'HostConfig': {'PortBindings': {}}})
        return rows, config

    def test_actual_images_and_application_env_must_match_compose(self):
        rows, config = self.fixture()
        self.assertEqual(len(audit.check_running_rows(rows, config)), 6)
        rows[-1]['Image'] = 'sha256:' + '0' * 64
        with self.assertRaisesRegex(audit.AuditFailure, 'running_container_image_mismatch'):
            audit.check_running_rows(rows, config)

    def test_running_environment_literal_dollar_is_compared_to_effective_compose(self):
        rows, config = self.fixture()
        config['services']['api']['environment']['GOOGLE_TEST_POLICY'] = '$${NOT_EXPANDED}$$value$$$$end'
        rows[-1]['Config']['Env'].append('GOOGLE_TEST_POLICY=${NOT_EXPANDED}$value$$end')
        audit.check_running_rows(rows, config)
        rows, config = self.fixture()
        rows[-1]['Config']['Env'] = ['SUPABASE_URL=https://cloud.invalid']
        with self.assertRaisesRegex(audit.AuditFailure, 'running_application_environment_mismatch'):
            audit.check_running_rows(rows, config)

    def test_running_worker_public_port_wrong_project_fail(self):
        rows, config = self.fixture()
        rows[-1]['HostConfig']['PortBindings'] = {'8001/tcp': [{'HostIp': '0.0.0.0', 'HostPort': '8001'}]}
        with self.assertRaisesRegex(audit.AuditFailure, 'running_public_port_forbidden'):
            audit.check_running_rows(rows, config)
        rows, config = self.fixture()
        rows[-1]['Config']['Labels']['com.docker.compose.project'] = 'production'
        with self.assertRaisesRegex(audit.AuditFailure, 'unexpected_staging_container'):
            audit.check_running_rows(rows, config)
        rows, config = self.fixture()
        worker = copy.deepcopy(rows[-1])
        worker['Config']['Labels']['com.docker.compose.service'] = 'worker'
        config['services']['worker'] = config['services']['api']
        with self.assertRaisesRegex(audit.AuditFailure, 'staging_worker_must_remain_stopped'):
            audit.check_running_rows(rows + [worker], config)


class ExistingPayloadTests(unittest.TestCase):
    def fixture(self):
        from PIL import Image
        image = io.BytesIO()
        Image.new('RGB', (4, 6), 'gray').save(image, 'PNG')
        image_raw = image.getvalue()
        mask = io.BytesIO()
        Image.new('L', (4, 6), 0).save(mask, 'PNG')
        mask_raw = mask.getvalue()
        owner = str(uuid.uuid4())
        report = {'image': {'original_base64': base64.b64encode(image_raw).decode(), 'width': 4, 'height': 6}}
        report_raw = audit.canonical(report)
        contour = {'owner_id': owner, 'image_sha256': audit.digest_bytes(image_raw),
                   'mask_sha256': audit.digest_bytes(mask_raw), 'width': 4, 'height': 6,
                   'original_image_base64': base64.b64encode(image_raw).decode(),
                   'mask_png_base64': base64.b64encode(mask_raw).decode()}
        links = {'reports': [{'owner_id': owner, 'payload_sha256': audit.digest_bytes(report_raw), 'image_sha256': audit.digest_bytes(image_raw)}],
                 'contours': [{'owner_id': owner, 'correction_id': 'fixture', 'image_sha256': audit.digest_bytes(image_raw)}]}

        class Response:
            def __init__(self, payload):
                self.payload = payload
                self.content = audit.canonical(payload)

            def json(self):
                return self.payload

        class HTTP:
            def request(self, method, path, **kwargs):
                assert method == 'GET' and kwargs.get('db') is True
                return Response(report if '/report-versions/' in path else contour)

        return HTTP(), links, contour

    def test_existing_report_and_contour_actual_originals_masks_are_decoded(self):
        http, links, _ = self.fixture()
        result = audit.verify_payload_links(http, json.dumps(links).encode(), 'private')
        self.assertEqual(result, {'report_photos_decoded': 1, 'indexed_contour_pairs_decoded': 1, 'stored_payload_url_categories': {}})

    def test_owner_swap_mask_tampering_and_source_photo_tampering_fail(self):
        for change in ('owner', 'mask', 'photo'):
            http, links, contour = self.fixture()
            if change == 'owner':
                contour['owner_id'] = str(uuid.uuid4())
            elif change == 'mask':
                contour['mask_png_base64'] = base64.b64encode(b'tampered').decode()
            else:
                links['reports'][0]['image_sha256'] = '0' * 64
            with self.subTest(change=change), self.assertRaises(audit.AuditFailure):
                audit.verify_payload_links(http, json.dumps(links).encode(), 'private')


class DirectAPIHTTPTests(unittest.TestCase):
    def fixture(self):
        from types import SimpleNamespace
        calls = []

        class Session:
            def request(self, method, url, **kwargs):
                calls.append((method, url, kwargs))
                return SimpleNamespace(status_code=200, json=lambda: {'ok': True})

        http = object.__new__(audit.StageHTTP)
        http.base, http.service_key, http.session = 'http://127.0.0.1:19080', 'fixture-service-key', Session()
        http.api_bases = {'/api/v3': 'http://127.0.0.1:19000', '/api/v4': 'http://127.0.0.1:19001'}
        return http, calls

    def test_auth_and_v4_api_use_direct_ports_while_storage_rest_use_gateway(self):
        http, calls = self.fixture()
        http.api('GET', '/api/v3/auth/me', token='fixture-user-token')
        http.api('GET', '/api/v4/v4/reports', token='fixture-user-token')
        http.request('GET', '/rest/v1/users', db=True)
        self.assertEqual([row[1] for row in calls], ['http://127.0.0.1:19000/auth/me',
                                                   'http://127.0.0.1:19001/v4/reports',
                                                   'http://127.0.0.1:19080/rest/v1/users'])
        self.assertNotIn('apikey', calls[0][2]['headers'])
        self.assertEqual(calls[2][2]['headers']['apikey'], 'fixture-service-key')
        self.assertTrue(all(row[2]['allow_redirects'] is False for row in calls))

    def test_unexpected_api_prefix_external_mapping_and_extra_api_ports_fail(self):
        http, calls = self.fixture()
        with self.assertRaises(audit.AuditFailure):
            http.api('GET', '/api/v2/health')
        http.api_bases['/api/v3'] = 'https://production.invalid'
        with self.assertRaises(audit.AuditFailure):
            http.api('GET', '/api/v3/auth/me')
        self.assertEqual(calls, [])
        config = {'services': {'v3': {'ports': ['127.0.0.1:8000:8000']},
                               'api': {'ports': ['127.0.0.1:8001:8001']}}}
        self.assertEqual(audit.api_bases(config)['/api/v4'], 'http://127.0.0.1:8001')
        config['services']['api']['ports'].append('0.0.0.0:9000:8001')
        with self.assertRaises(audit.AuditFailure):
            audit.api_bases(config)


if __name__ == '__main__':
    unittest.main()
