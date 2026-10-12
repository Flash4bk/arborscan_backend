"""Read-only continuity audit and explicitly isolated DEMO HTTP smoke.

The audit never prints user rows, credentials, object keys or upstream responses.
Capture FINGERPRINT_SQL against the source and restored database with the same
PostgreSQL client. Its output contains only counts and SHA256 digests. The HTTP
smoke is opt-in, targets a guarded localhost staging stack, and creates only new
DEMO accounts. Existing owners, sessions, decisions and jobs are not modified.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import stat
import subprocess
import sys
from urllib.parse import quote, urlsplit
import uuid


FINGERPRINT_SQL = r"""
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY;
SET LOCAL TIME ZONE 'UTC';
SET LOCAL DateStyle = 'ISO, YMD';
SET LOCAL IntervalStyle = 'postgres';
SET LOCAL extra_float_digits = 1;
SELECT format($q$SELECT json_build_object('table', %L, 'rows', count(*),
 'sha256', encode(sha256(convert_to(coalesce(jsonb_agg(to_jsonb(t)
 ORDER BY to_jsonb(t)::text COLLATE "C")::text, '[]'), 'UTF8')), 'hex')) FROM public.%I t$q$,
 tablename, tablename) FROM pg_tables WHERE schemaname='public' ORDER BY tablename
\gexec
SELECT json_build_object('versions',json_build_object(
 'contours',public.contour_workflow_version(),
 'history',public.server_history_version(),
 'quality',public.model_quality_version()));
SELECT json_build_object('relationships',json_build_object(
 'orphan_sessions',(SELECT count(*) FROM public.auth_sessions a LEFT JOIN public.users u ON u.id=a.user_id WHERE u.id IS NULL),
 'orphan_reports',(SELECT count(*) FROM public.report_versions a LEFT JOIN public.users u ON u.id=a.owner_id WHERE u.id IS NULL),
 'orphan_contours',(SELECT count(*) FROM public.contour_revisions a LEFT JOIN public.users u ON u.id=a.owner_id WHERE u.id IS NULL)));
COMMIT;
"""
STORAGE_FULL_FINGERPRINT_SQL = r"""
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY;
SET LOCAL TIME ZONE 'UTC';
SET LOCAL DateStyle = 'ISO, YMD';
SET LOCAL IntervalStyle = 'postgres';
SET LOCAL extra_float_digits = 1;
SELECT json_build_object('storage_table','objects','rows',count(*),
 'sha256',encode(sha256(convert_to(coalesce(jsonb_agg(to_jsonb(t)
 ORDER BY to_jsonb(t)::text COLLATE "C")::text,'[]'),'UTF8')),'hex')) FROM storage.objects t;
SELECT json_build_object('storage_table','buckets','rows',count(*),
 'sha256',encode(sha256(convert_to(coalesce(jsonb_agg(to_jsonb(t)
 ORDER BY to_jsonb(t)::text COLLATE "C")::text,'[]'),'UTF8')),'hex')) FROM storage.buckets t;
COMMIT;
"""
SHA256 = re.compile(r'^[0-9a-f]{64}$')
IMAGE_ID = re.compile(r'^sha256:[0-9a-f]{64}$')
FIXTURE_NAME = 'DEMO-selfhost-recovery'


class AuditFailure(Exception):
    """A fixed category safe for public evidence; never contains source values."""


def digest_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def private_json(path):
    """Reject world/group readable files on Linux before reading credentials."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise AuditFailure('private_input_required')
    if os.name != 'nt' and stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise AuditFailure('private_input_permissions')
    return json.loads(path.read_text(encoding='utf-8'))


def write_private(path, value):
    path = Path(path)
    if path.is_symlink():
        raise AuditFailure('unsafe_private_output')
    # O_EXCL creates a new temporary output; atomic replace avoids a truncated
    # fixture state if the operation is interrupted. The parent must be private.
    temporary = path.with_name(path.name + '.tmp-' + secrets.token_hex(8))
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(canonical(value))
        os.replace(temporary, path)
        if os.name != 'nt':
            path.chmod(0o600)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_fingerprints(raw):
    data = {'tables': {}}
    for line in raw.decode('utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if set(row) == {'table', 'rows', 'sha256'}:
            name = row['table']
            if not isinstance(name, str) or not re.fullmatch(r'[a-zA-Z_][a-zA-Z_0-9]*', name):
                raise AuditFailure('invalid_table_fingerprint')
            if name in data['tables'] or type(row['rows']) is not int or row['rows'] < 0 or not SHA256.fullmatch(row['sha256']):
                raise AuditFailure('invalid_table_fingerprint')
            data['tables'][name] = {'rows': row['rows'], 'sha256': row['sha256']}
        elif set(row) == {'versions'} and 'versions' not in data:
            data.update(row)
        elif set(row) == {'relationships'} and 'relationships' not in data:
            data.update(row)
        else:
            raise AuditFailure('invalid_fingerprint_record')
    required = {'users', 'auth_sessions', 'report_versions', 'contour_revisions'}
    if not required <= data['tables'].keys():
        raise AuditFailure('incomplete_database_fingerprint')
    if (data.get('versions') != {'contours': 1, 'history': 1, 'quality': 1}
            or any(type(value) is not int for value in data['versions'].values())):
        raise AuditFailure('migration_version_mismatch')
    if (data.get('relationships') != {'orphan_sessions': 0, 'orphan_reports': 0, 'orphan_contours': 0}
            or any(type(value) is not int for value in data['relationships'].values())):
        raise AuditFailure('ownership_relationship_mismatch')
    return data


def compare_database(expected, actual):
    if expected != actual:
        # Do not print a user ID or row as a diagnostic for a mismatch.
        raise AuditFailure('database_continuity_mismatch')
    return {'tables': len(actual['tables']),
            'rows': sum(row['rows'] for row in actual['tables'].values()),
            'accounts_password_hashes_sessions_owners': 'identical',
            'reports_contours_decisions_jobs': 'identical',
            'schema_versions': actual['versions'],
            'fingerprint_sha256': digest_bytes(canonical(actual))}


def parse_storage_fingerprints(raw):
    result = {}
    for line in raw.decode('utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (set(row) != {'storage_table', 'rows', 'sha256'} or row['storage_table'] not in ('objects', 'buckets')
                or row['storage_table'] in result or type(row['rows']) is not int or row['rows'] < 0
                or not isinstance(row['sha256'], str) or not SHA256.fullmatch(row['sha256'])):
            raise AuditFailure('invalid_full_storage_fingerprint')
        result[row['storage_table']] = {'rows': row['rows'], 'sha256': row['sha256']}
    if set(result) != {'objects', 'buckets'}:
        raise AuditFailure('incomplete_full_storage_fingerprint')
    return result


def compare_storage_database(expected, actual):
    if expected != actual:
        raise AuditFailure('storage_metadata_continuity_mismatch')
    return {'objects': actual['objects']['rows'], 'buckets': actual['buckets']['rows'],
            'all_columns_including_owners_versions_timestamps_metadata': 'identical',
            'checked_phase': 'before_http_reads_and_before_demo_writes',
            'fingerprint_sha256': digest_bytes(canonical(actual))}


def safe_object(row):
    if not isinstance(row, dict) or set(row) != {'bucket', 'name', 'size', 'sha256'}:
        raise AuditFailure('invalid_object_manifest')
    bucket, name = row['bucket'], row['name']
    if not isinstance(bucket, str) or not bucket or any(c in bucket for c in '/\\') or bucket in ('.', '..'):
        raise AuditFailure('unsafe_bucket')
    if not isinstance(name, str) or not name or '\\' in name or name.startswith('/'):
        raise AuditFailure('unsafe_object')
    if any(part in ('', '.', '..') for part in name.split('/')) or PurePosixPath(name).as_posix() != name:
        raise AuditFailure('unsafe_object')
    if type(row['size']) is not int or row['size'] < 0 or not SHA256.fullmatch(row['sha256']):
        raise AuditFailure('invalid_object_manifest')
    return row


def storage_manifest(rows):
    seen = set()
    result = []
    for row in rows:
        row = safe_object(row)
        key = (row['bucket'], row['name'])
        if key in seen:
            raise AuditFailure('duplicate_object_manifest')
        seen.add(key)
        result.append(row)
    return sorted(result, key=lambda row: (row['bucket'], row['name']))


def loopback_base(base):
    parsed = urlsplit(base)
    if (parsed.scheme != 'http' or parsed.hostname != '127.0.0.1' or
            parsed.username or parsed.password or parsed.path not in ('', '/') or
            parsed.query or parsed.fragment or parsed.port is None):
        raise AuditFailure('localhost_staging_required')
    return 'http://127.0.0.1:' + str(parsed.port)


def guarded_stack(root, config, assets_root=None, parent=None):
    root = Path(root)
    if root.is_symlink():
        raise AuditFailure('staging_root_symlink_forbidden')
    root = root.resolve()
    parent = Path('/home/arborscan').resolve() if parent is None else Path(parent).resolve()
    if root.parent != parent or not re.fullmatch(r'selfhost-\d{8}T\d{6}Z(?:-[a-z0-9]+)?', root.name):
        raise AuditFailure('staging_root_required')
    if os.name != 'nt' and stat.S_IMODE(root.stat().st_mode) & 0o077:
        raise AuditFailure('staging_root_permissions')
    expected_services = {'db', 'rest', 'storage', 'gateway', 'v3', 'api', 'worker'}
    if not expected_services <= config.get('services', {}).keys():
        raise AuditFailure('incomplete_staging_stack')
    if any(v.get('external') for v in config.get('volumes', {}).values() if isinstance(v, dict)):
        raise AuditFailure('external_volume_forbidden')
    allowed_mounts = [root]
    if assets_root is not None:
        assets = Path(assets_root).resolve(strict=True)
        if not assets.is_relative_to(parent) or assets == parent:
            raise AuditFailure('private_cloned_assets_required')
        allowed_mounts.append(assets)
    for service in config['services'].values():
        if service.get('network_mode') == 'host' or service.get('privileged'):
            raise AuditFailure('unsafe_staging_service')
        for mount in service.get('volumes', []):
            source = mount.get('source', '') if isinstance(mount, dict) else mount.split(':')[0]
            if source.startswith('/') and not any(Path(source).resolve().is_relative_to(p) for p in allowed_mounts):
                raise AuditFailure('production_mount_forbidden')
        for binding in service.get('ports', []):
            if not isinstance(binding, str) or not re.fullmatch(r'127\.0\.0\.1:\d+:\d+', binding):
                raise AuditFailure('public_staging_port_forbidden')
    gateway = config['services']['gateway'].get('ports', [])
    if len(gateway) != 1 or not gateway[0].endswith(':8000'):
        raise AuditFailure('staging_gateway_port_required')
    for name in ('v3', 'api', 'worker'):
        env = config['services'][name].get('environment', {})
        if not isinstance(env, dict) or env.get('SUPABASE_URL') != 'http://gateway:8000':
            raise AuditFailure('production_dependency_forbidden')
    return loopback_base('http://' + gateway[0].rsplit(':', 1)[0])


def run_sql(root, script=FINGERPRINT_SQL):
    command = ['docker', 'compose', '-f', str(root / 'compose.private.json'),
               'exec', '-T', 'db', 'psql', '-U', 'supabase_admin', '-d', 'postgres',
               '-v', 'ON_ERROR_STOP=1', '-qAt']
    result = subprocess.run(command, input=script.encode(), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=300)
    if result.returncode:
        raise AuditFailure('staging_sql_failed')
    return result.stdout


def running_stack(root, config):
    """Verify running containers, rather than trusting just a Compose file."""
    command = ['docker', 'compose', '-f', str(root / 'compose.private.json')]
    listed = subprocess.run(command + ['ps', '-q'], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=30)
    ids = listed.stdout.decode().split()
    if listed.returncode or not ids or any(not re.fullmatch('[0-9a-f]{12,64}', value) for value in ids):
        raise AuditFailure('staging_container_inventory_failed')
    inspected = subprocess.run(['docker', 'inspect', *ids], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, timeout=30)
    if inspected.returncode:
        raise AuditFailure('staging_container_inspect_failed')
    return check_running_rows(json.loads(inspected.stdout), config)


def check_running_rows(rows, config):
    seen, summary = set(), []
    for row in rows:
        labels = row['Config'].get('Labels') or {}
        service = labels.get('com.docker.compose.service')
        if service in seen or service not in config['services'] or labels.get('com.docker.compose.project') != config['name']:
            raise AuditFailure('unexpected_staging_container')
        seen.add(service)
        if service == 'worker':
            raise AuditFailure('staging_worker_must_remain_stopped')
        if row.get('Image') != config['services'][service]['image'] or not row['State'].get('Running'):
            raise AuditFailure('running_container_image_mismatch')
        if service in ('v3', 'api'):
            actual_env = dict(item.split('=', 1) for item in row['Config']['Env'])
            for key, value in config['services'][service]['environment'].items():
                if compose_literal(value) != actual_env.get(key):
                    raise AuditFailure('running_application_environment_mismatch')
        bindings = row['HostConfig'].get('PortBindings') or {}
        if any(binding.get('HostIp') != '127.0.0.1' for values in bindings.values() if values for binding in values):
            raise AuditFailure('running_public_port_forbidden')
        summary.append({'service': service, 'image': row['Image'], 'running': True,
                        'health': row['State'].get('Health', {}).get('Status')})
    if seen != {'db', 'rest', 'storage', 'gateway', 'v3', 'api'}:
        raise AuditFailure('staging_running_services_incomplete')
    return sorted(summary, key=lambda row: row['service'])


def compose_literal(value):
    """Undo exactly Compose's literal dollar escape. Never evaluate variables."""
    return str(value).replace('$$', '$')


def protected_policy(env, compose_values=False):
    """Compare every Google decision switch, plus explicit model selection.

    Exact image IDs protect implementation; key/value digests compare decisions
    while avoiding client IDs and secrets in public evidence.
    """
    return {key: digest_bytes((compose_literal(value) if compose_values else str(value)).encode()) for key, value in sorted(env.items())
            if key.startswith('GOOGLE_') or key in {
                'ACTIVE_MODEL_VERSION', 'AUTO_SELECT_LATEST_LOCAL_MODEL',
                'SUPABASE_ENABLE_QUEUE', 'MODEL_QUALITY_ENABLED'}}


def build_expected(database, snapshot, live_inspect, assets, storage_database=None):
    """Build the private expected contract from verified independent seed files.

    This does not capture a database itself: caller supplies parsed source SQL
    fingerprints matched to the native dump. File bytes must match the backup
    manifest, not merely an untrusted provided checksum.
    """
    snapshot, assets = Path(snapshot).resolve(strict=True), Path(assets).resolve(strict=True)
    manifest = json.loads((snapshot / 'manifest.json').read_text())
    rows = []
    for name, expected_digest in manifest.items():
        if not name.startswith('objects/'):
            continue
        relative = PurePosixPath(name)
        if len(relative.parts) < 3 or any(p in ('.', '..') for p in relative.parts) or '\\' in name or not SHA256.fullmatch(expected_digest):
            raise AuditFailure('invalid_seed_manifest')
        path = snapshot.joinpath(*relative.parts)
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(snapshot):
            raise AuditFailure('unsafe_seed_file')
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != expected_digest:
                raise AuditFailure('seed_sha256_mismatch')
        rows.append({'bucket': relative.parts[1], 'name': '/'.join(relative.parts[2:]),
                     'size': path.stat().st_size, 'sha256': expected_digest})
    inspected = live_inspect if isinstance(live_inspect, list) else private_json(live_inspect)
    names = {'v3': '/arborscan-api', 'api': '/arborscan-api-v4', 'worker': '/arborscan-quality-worker'}
    images, policy = {}, {}
    for service, name in names.items():
        matches = [row for row in inspected if row.get('Name') == name]
        if len(matches) != 1 or not IMAGE_ID.fullmatch(matches[0].get('Image', '')):
            raise AuditFailure('source_image_inventory_mismatch')
        row = matches[0]
        env = dict(item.split('=', 1) for item in row['Config']['Env'])
        images[service], policy[service] = row['Image'], protected_policy(env)
    models = {}
    models_root = assets / 'models'
    for path in sorted(models_root.rglob('*')):
        if path.is_symlink():
            raise AuditFailure('model_seed_symlink')
        if path.is_file():
            with path.open('rb') as stream:
                models[path.relative_to(models_root).as_posix()] = hashlib.file_digest(stream, 'sha256').hexdigest()
    if not models:
        raise AuditFailure('model_seed_missing')
    result = {'format': 1, 'database': database, 'storage': storage_manifest(rows),
            'images': images, 'policy': policy, 'models': models, 'assets_root': str(assets)}
    if storage_database is not None:
        result['storage_database'] = storage_database
    return result


def api_bases(config):
    result = {}
    for prefix, service, port in (('/api/v3', 'v3', 8000), ('/api/v4', 'api', 8001)):
        values = config['services'][service].get('ports', [])
        if len(values) != 1 or not isinstance(values[0], str):
            raise AuditFailure('single_localhost_api_port_required')
        match = re.fullmatch(r'127\.0\.0\.1:(\d+):' + str(port), values[0])
        if not match:
            raise AuditFailure('single_localhost_api_port_required')
        result[prefix] = loopback_base('http://127.0.0.1:' + match[1])
    return result


class StageHTTP:
    def __init__(self, base, service_key, api_bases):
        import requests
        self.base = loopback_base(base)
        self.session = requests.Session()
        # Proxy environment can otherwise forward a private loopback request.
        self.session.trust_env = False
        self.service_key = service_key
        if set(api_bases) != {'/api/v3', '/api/v4'}:
            raise AuditFailure('localhost_api_mapping_required')
        self.api_bases = {key: loopback_base(value) for key, value in api_bases.items()}

    def request(self, method, path, token=None, expected=200, db=False, **kwargs):
        return self.request_base(self.base, method, path, token, expected, db, **kwargs)

    def request_base(self, base, method, path, token=None, expected=200, db=False, **kwargs):
        base = loopback_base(base)
        if not path.startswith('/') or '://' in path:
            raise AuditFailure('invalid_staging_request')
        headers = {'Authorization': 'Bearer ' + token} if token else {}
        if db:
            headers = {'Authorization': 'Bearer ' + self.service_key, 'apikey': self.service_key}
        response = self.session.request(method, base + path, headers=headers,
                                        timeout=120, allow_redirects=False, **kwargs)
        if response.status_code != expected:
            raise AuditFailure('staging_http_status_' + str(response.status_code))
        return response

    def api(self, method, path, token=None, expected=200, **kwargs):
        for prefix, base in self.api_bases.items():
            if path.startswith(prefix + '/'):
                return self.request_base(base, method, path[len(prefix):], token, expected, **kwargs).json()
        raise AuditFailure('known_api_prefix_required')

    def objects(self, expected):
        rows = storage_manifest(expected)
        # The caller checks the SQL key set before reading bytes. No additional
        # public RPC or access policy is created by this audit.
        observed = []
        for row in rows:
            path = '/storage/v1/object/' + quote(row['bucket'], safe='') + '/' + quote(row['name'], safe='/')
            response = self.request('GET', path, db=True, stream=True)
            digest, length = hashlib.sha256(), 0
            try:
                for chunk in response.iter_content(1024 * 1024):
                    length += len(chunk)
                    if length > row['size']:
                        raise AuditFailure('storage_size_mismatch')
                    digest.update(chunk)
            finally:
                response.close()
            if length != row['size'] or digest.hexdigest() != row['sha256']:
                raise AuditFailure('storage_content_mismatch')
            observed.append(row)
        return {'objects': len(rows), 'bytes': sum(row['size'] for row in rows),
                'all_object_sha256': 'identical', 'manifest_sha256': digest_bytes(canonical(observed))}


def storage_sql_manifest(raw):
    rows = json.loads(raw)
    if not isinstance(rows, list) or any(set(row) != {'bucket', 'name', 'size'} for row in rows):
        raise AuditFailure('invalid_storage_sql_inventory')
    return sorted(rows, key=lambda row: (row['bucket'], row['name']))


STORAGE_SQL = """BEGIN READ ONLY; SELECT coalesce(json_agg(t ORDER BY bucket,name),'[]')
FROM (SELECT bucket_id AS bucket,name,coalesce((metadata->>'size')::bigint,0) AS size
FROM storage.objects) t; COMMIT;"""

LINK_SQL = """BEGIN READ ONLY; SELECT json_build_object(
 'reports',(SELECT coalesce(json_agg(t),'[]') FROM (SELECT owner_id,payload_sha256,image_sha256 FROM public.report_versions) t),
 'contours',(SELECT coalesce(json_agg(t),'[]') FROM (SELECT owner_id,correction_id,image_sha256 FROM public.contour_revisions) t)); COMMIT;"""

URL_SQL = r"""BEGIN READ ONLY;
SELECT format($q$SELECT json_build_object('url_table', %L, 'urls',
 coalesce(json_agg(m[1]),'[]')) FROM public.%I t CROSS JOIN LATERAL
 regexp_matches(to_jsonb(t)::text, 'https?://[^"[:space:]<>]+', 'g') m$q$,
 tablename,tablename) FROM pg_tables WHERE schemaname='public' ORDER BY tablename
\gexec
SELECT json_build_object('url_table','users.avatar_url','urls',
 coalesce(json_agg(to_jsonb(t)->>'avatar_url') FILTER
 (WHERE coalesce(to_jsonb(t)->>'avatar_url','')<>''),'[]')) FROM public.users t;
COMMIT;"""


def classify_url(url):
    """Return only a category; URL paths may include private IDs or tokens."""
    host = (urlsplit(url.rstrip('\\')).hostname or '').lower()
    if host.endswith('.supabase.co') or host == 'supabase.co':
        return 'old_supabase'
    if host.endswith('.googleusercontent.com'):
        return 'google_avatar'
    if host in ('gateway', '127.0.0.1', 'localhost'):
        return 'local'
    if host == '31.57.170.88':
        return 'arborscan_https'
    return 'other_external'


def classify_url_records(raw):
    result = {'absolute_urls': 0, 'old_supabase_urls': 0,
              'old_supabase_avatar_urls': 0, 'by_source': {}}
    for line in raw.decode().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if set(row) != {'url_table', 'urls'} or not isinstance(row['urls'], list):
            raise AuditFailure('invalid_url_inventory')
        table = row['url_table']
        if not isinstance(table, str) or not re.fullmatch(r'[a-zA-Z_][a-zA-Z_0-9.]+' , table):
            raise AuditFailure('invalid_url_inventory')
        counts = {}
        for url in row['urls']:
            if not isinstance(url, str) or not url.startswith(('http://', 'https://')):
                continue
            category = classify_url(url)
            counts[category] = counts.get(category, 0) + 1
        result['by_source'][table] = counts
        # The avatar-specific query repeats those references intentionally. Do
        # not count it a second time in the all-public-row URL totals.
        if table == 'users.avatar_url':
            result['old_supabase_avatar_urls'] = counts.get('old_supabase', 0)
        else:
            result['absolute_urls'] += sum(counts.values())
            result['old_supabase_urls'] += counts.get('old_supabase', 0)
    result['client_absolute_url_behavior'] = 'preserved_by_ApiConfig.imageUrl'
    result['current_client_network_images'] = 'profile_avatar_only; report_and_contour_images_use_base64'
    return result


def payload_url_counts(value):
    counts = {}

    def visit(node):
        if isinstance(node, dict):
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)
        elif isinstance(node, str):
            for match in re.finditer(r'https?://[^"\s<>]+', node):
                category = classify_url(match.group())
                counts[category] = counts.get(category, 0) + 1
    visit(value)
    return counts


def verify_payload_links(http, raw, bucket):
    """Read originals/masks referenced by every restored index, without signing
    in as their owners or printing their data. Service credentials stay internal.
    This is file integrity evidence, not a manual phone/history UI result.
    """
    from PIL import Image, ImageOps
    links = json.loads(raw)
    counts = {'report_photos_decoded': 0, 'indexed_contour_pairs_decoded': 0,
              'stored_payload_url_categories': {}}

    def urls(value):
        for category, count in payload_url_counts(value).items():
            counts['stored_payload_url_categories'][category] = counts['stored_payload_url_categories'].get(category, 0) + count

    def original(encoded, digest, dimensions):
        data = base64.b64decode(encoded, validate=True)
        if digest_bytes(data) != digest:
            raise AuditFailure('referenced_photo_sha256_mismatch')
        with Image.open(io.BytesIO(data)) as image:
            oriented = ImageOps.exif_transpose(image)
            if oriented.size != dimensions:
                raise AuditFailure('referenced_photo_dimensions_mismatch')

    for row in links['reports']:
        path = '/storage/v1/object/' + quote(bucket, safe='') + '/report-versions/' + quote(row['owner_id'], safe='') + '/' + row['payload_sha256'] + '.json'
        response = http.request('GET', path, db=True)
        if digest_bytes(response.content) != row['payload_sha256']:
            raise AuditFailure('referenced_report_payload_sha256_mismatch')
        payload = response.json()
        urls(payload)
        image = payload['image']
        original(image['original_base64'], row['image_sha256'], (image['width'], image['height']))
        counts['report_photos_decoded'] += 1
    for row in links['contours']:
        path = '/storage/v1/object/' + quote(bucket, safe='') + '/v4-corrections/' + quote(row['owner_id'], safe='') + '/' + quote(row['correction_id'], safe='')
        payload = http.request('GET', path, db=True).json()
        urls(payload)
        if payload['owner_id'] != row['owner_id'] or payload['image_sha256'] != row['image_sha256']:
            raise AuditFailure('referenced_contour_owner_mismatch')
        dimensions = (payload['width'], payload['height'])
        original(payload['original_image_base64'], row['image_sha256'], dimensions)
        mask = base64.b64decode(payload['mask_png_base64'], validate=True)
        if digest_bytes(mask) != payload['mask_sha256']:
            raise AuditFailure('referenced_mask_sha256_mismatch')
        with Image.open(io.BytesIO(mask)) as image:
            if image.format != 'PNG' or image.size != dimensions:
                raise AuditFailure('referenced_mask_format_mismatch')
            image.verify()
        counts['indexed_contour_pairs_decoded'] += 1
    return counts


def compare_storage_keys(expected, raw):
    expected = storage_manifest(expected)
    keys = [{k: row[k] for k in ('bucket', 'name', 'size')} for row in expected]
    if keys != storage_sql_manifest(raw):
        raise AuditFailure('storage_inventory_mismatch')


def validate_fixture(state):
    if not isinstance(state, dict) or state.get('tag') != FIXTURE_NAME or state.get('format') != 1:
        raise AuditFailure('invalid_fixture_state')
    if len(state.get('accounts', [])) != 2:
        raise AuditFailure('invalid_fixture_state')
    ids = set()
    for account in state['accounts']:
        try:
            uuid.UUID(account['id'])
        except (ValueError, KeyError, TypeError):
            raise AuditFailure('invalid_fixture_account') from None
        if (account['id'] in ids or not re.fullmatch(r'demo-selfhost-[0-9a-f]{32}@invalid\.test', account.get('email', ''))
                or account.get('name') != FIXTURE_NAME or not account.get('password') or not account.get('token')):
            raise AuditFailure('invalid_fixture_account')
        ids.add(account['id'])
    return state


def smoke(http, fixture_path):
    """New fixtures only, no reads as real owners and no training/worker calls.

    First run creates fixtures and writes private state after every registration.
    Subsequent runs prove password login and durable reads after restart. Records
    remain as clearly marked test data; deleting production-style records is not
    hidden in a finally block.
    """
    from PIL import Image, ImageDraw
    fixture_path = Path(fixture_path)
    if fixture_path.exists():
        state = validate_fixture(private_json(fixture_path))
        if state.get('complete') is not True:
            raise AuditFailure('incomplete_fixture_retained_for_review')
        owner, other = state['accounts']
        for account in state['accounts']:
            login = http.api('POST', '/api/v3/auth/login', json={k: account[k] for k in ('email', 'password')})
            if login['user']['id'] != account['id']:
                raise AuditFailure('fixture_login_owner_mismatch')
            account['token'] = login['token']
        for version in state['versions']:
            r = http.api('GET', '/api/v4/v4/reports/' + version, owner['token'])
            if not r['persisted'] or r['snapshot']['image']['sha256'] != state['image_sha256']:
                raise AuditFailure('fixture_report_restart_mismatch')
            http.api('GET', '/api/v4/v4/reports/' + version, other['token'], 404)
        for correction, status in state['contours']:
            r = http.api('GET', '/api/v4/v4/corrections/' + correction, owner['token'])
            if r['review_status'] != status or r['image_sha256'] != state['image_sha256']:
                raise AuditFailure('fixture_contour_restart_mismatch')
        write_private(fixture_path, state)
        return {'restart_password_login_versions_contours': 'passed', 'new_test_data': True}

    state = {'format': 1, 'tag': FIXTURE_NAME, 'accounts': [], 'versions': [], 'contours': []}
    for _ in range(2):
        account = {'name': FIXTURE_NAME, 'email': 'demo-selfhost-' + uuid.uuid4().hex + '@invalid.test',
                   'password': secrets.token_urlsafe(32)}
        result = http.api('POST', '/api/v3/auth/register', json=account)
        if result['user']['email'] != account['email'] or result['user']['name'] != FIXTURE_NAME:
            raise AuditFailure('fixture_registration_mismatch')
        account.update(id=result['user']['id'], token=result['token'])
        state['accounts'].append(account)
        write_private(fixture_path, state)
    validate_fixture(state)
    owner, admin = state['accounts']
    http.api('GET', '/api/v4/v4/reports', expected=401)
    http.api('GET', '/api/v4/v4/corrections', expected=401)
    http.api('GET', '/api/v3/auth/me', expected=401)
    http.api('POST', '/api/v3/auth/login', expected=401, json={'email': owner['email'], 'password': 'wrong'})
    for account in state['accounts']:
        result = http.api('POST', '/api/v3/auth/login', json={k: account[k] for k in ('email', 'password')})
        if result['user']['id'] != account['id']:
            raise AuditFailure('fixture_login_owner_mismatch')
        account['token'] = result['token']
    http.api('GET', '/api/v4/v4/corrections/workflow/queue', owner['token'], 403)
    # Only the newly registered fixture is promoted in this localhost COPY.
    http.request('PATCH', '/rest/v1/users', db=True, expected=204,
                 params={'id': 'eq.' + admin['id'], 'email': 'eq.' + admin['email'], 'name': 'eq.' + FIXTURE_NAME},
                 json={'role': 'admin'})
    first_caps = http.api('GET', '/api/v4/v4/reports/capabilities', owner['token'])
    contour_caps = http.api('GET', '/api/v4/v4/corrections/workflow/capabilities', owner['token'])
    if first_caps['history_version'] != 1 or contour_caps['workflow_version'] != 1:
        raise AuditFailure('fixture_capabilities_mismatch')

    image = io.BytesIO()
    Image.new('RGB', (48, 64), 'gray').save(image, 'PNG')
    raw = image.getvalue()
    mask = io.BytesIO()
    pixels = Image.new('L', (48, 64), 0)
    ImageDraw.Draw(pixels).rectangle((10, 6, 30, 54), fill=255)
    pixels.save(mask, 'PNG')
    state['image_sha256'] = digest_bytes(raw)
    aid = str(uuid.uuid4())
    line = lambda a, b: [{'x': a[0], 'y': a[1]}, {'x': b[0], 'y': b[1]}]
    snapshot = {'version': 1, 'kind': 'reference', 'change_source': 'reference',
                'captured_at': dt.datetime.now(dt.timezone.utc).isoformat(), 'report': {},
                'reference': {'version': 1, 'method': 'known_object_segment_v1',
                              'coordinates': 'normalized_oriented_image', 'width': 48, 'height': 64,
                              'same_plane': True, 'length_m': 1, 'reference': line((.1, .1), (.1, .3)),
                              'tree': line((.5, .1), (.5, .9)), 'crown': line((.2, .1), (.8, .1)),
                              'outline': [{'x': .1, 'y': .1}, {'x': .1, 'y': .3}, {'x': .2, 'y': .3}]}}
    for index in range(2):
        version = str(uuid.uuid4())
        data = {'analysis_id': aid, 'version_id': version, 'snapshot': json.dumps(snapshot)}
        if index:
            data['parent_id'] = state['versions'][0]
        saved = http.api('POST', '/api/v4/v4/reports', owner['token'],
                         data=data, files={'image': ('DEMO-fixture.png', raw, 'image/png')})
        if saved.get('saved') is not True or saved.get('persisted') is not True:
            raise AuditFailure('fixture_save_unconfirmed')
        repeated = http.api('POST', '/api/v4/v4/reports', owner['token'],
                            data=data, files={'image': ('DEMO-fixture.png', raw, 'image/png')})
        if repeated != saved:
            raise AuditFailure('fixture_report_idempotency')
        state['versions'].append(version)
        r = http.api('GET', '/api/v4/v4/reports/' + version, owner['token'])
        if abs(r['snapshot']['report']['height_m'] - 4) > 1e-8 or digest_bytes(base64.b64decode(r['snapshot']['image']['original_base64'], validate=True)) != state['image_sha256']:
            raise AuditFailure('fixture_report_roundtrip')
        http.api('GET', '/api/v4/v4/reports/' + version, admin['token'], 404)
        write_private(fixture_path, state)

    def contour(parent=None, x=.2, legacy=False, analysis=aid):
        data = {'analysis_id': analysis}
        if not legacy:
            data['editor_state'] = json.dumps({'version': 1, 'coordinates': 'normalized_oriented_image',
                'width': 48, 'height': 64, 'closed': True,
                'points': [{'x': x, 'y': .1}, {'x': .8, 'y': .1}, {'x': .5, 'y': .8}]})
        if parent:
            data['parent_id'] = parent
        saved = http.api('POST', '/api/v4/v4/corrections' + ('' if legacy else '/workflow'), owner['token'],
                         data=data, files={'image': ('DEMO-photo.png', raw, 'image/png'),
                                           'mask': ('DEMO-mask.png', mask.getvalue(), 'image/png')})
        if saved.get('saved') is not True:
            raise AuditFailure('fixture_contour_save_unconfirmed')
        return saved['correction_id']

    legacy = contour(legacy=True, analysis=str(uuid.uuid4()))
    legacy_read = http.api('GET', '/api/v4/v4/corrections/' + legacy, owner['token'])
    if 'editor_state' in legacy_read or digest_bytes(base64.b64decode(legacy_read['mask_png_base64'], validate=True)) != digest_bytes(mask.getvalue()):
        raise AuditFailure('fixture_legacy_png')
    first = contour()
    if contour() != first:
        raise AuditFailure('fixture_contour_idempotency')
    http.api('GET', '/api/v4/v4/corrections/' + first, admin['token'], 404)
    http.api('POST', '/api/v4/v4/corrections/workflow/review/' + owner['id'] + '/' + first,
             owner['token'], 403, json={'decision': 'accepted'})
    http.api('POST', '/api/v4/v4/corrections/' + first + '/submit', owner['token'])
    reviewed = http.api('POST', '/api/v4/v4/corrections/workflow/review/' + owner['id'] + '/' + first,
                        admin['token'], json={'decision': 'rejected', 'reason': FIXTURE_NAME + ' rejection'})
    if reviewed['status'] != 'rejected':
        raise AuditFailure('fixture_rejection')
    child = contour(first, .3)
    http.api('POST', '/api/v4/v4/corrections/' + child + '/submit', owner['token'])
    reviewed = http.api('POST', '/api/v4/v4/corrections/workflow/review/' + owner['id'] + '/' + child,
                        admin['token'], json={'decision': 'accepted'})
    if reviewed['status'] != 'accepted':
        raise AuditFailure('fixture_acceptance')
    draft = contour(child, .4)
    r = http.api('GET', '/api/v4/v4/corrections/' + draft, owner['token'])
    if r['review_status'] != 'draft' or r['eligible_for_training'] is not False:
        raise AuditFailure('fixture_accepted_to_draft')
    state['contours'] = [[first, 'rejected'], [child, 'accepted'], [draft, 'draft']]
    state['complete'] = True
    write_private(fixture_path, state)
    return {'new_demo_accounts': 2, 'password_login': 'passed', 'owner_isolation': 'passed',
            'unauthenticated_refusal': 'passed', 'private_report_versions': 2,
            'immutable_retries': 'passed', 'legacy_png': 'passed',
            'moderation_reject_accept_new_draft': 'passed', 'training_started': False,
            'real_user_records_changed': False, 'restart_check': 'pending_repeat_after_restart'}


def audit(root, expected):
    root = Path(root).resolve()
    config = private_json(root / 'compose.private.json')
    base = guarded_stack(root, config, expected.get('assets_root'))
    if expected.get('format') != 1:
        raise AuditFailure('invalid_expected_format')
    actual_database = parse_fingerprints(run_sql(root))
    result = {'database': compare_database(expected['database'], actual_database)}
    result['running_services'] = running_stack(root, config)
    if 'storage_database' not in expected:
        raise AuditFailure('full_source_storage_metadata_required')
    result['storage_database'] = compare_storage_database(expected['storage_database'],
            parse_storage_fingerprints(run_sql(root, STORAGE_FULL_FINGERPRINT_SQL)))
    for service in ('v3', 'api', 'worker'):
        if config['services'][service]['image'] != expected['images'][service] or not IMAGE_ID.fullmatch(expected['images'][service]):
            raise AuditFailure('application_image_mismatch')
        actual_policy = protected_policy(config['services'][service]['environment'], compose_values=True)
        if actual_policy != expected['policy'][service]:
            raise AuditFailure('application_policy_mismatch')
    model_mounts = [mount for mount in config['services']['api']['volumes']
                    if isinstance(mount, dict) and mount.get('target') == '/app/models']
    if len(model_mounts) != 1 or model_mounts[0].get('read_only') is not True:
        raise AuditFailure('read_only_models_required')
    models_root = Path(model_mounts[0]['source']).resolve(strict=True)
    actual_names = {path.relative_to(models_root).as_posix() for path in models_root.rglob('*') if path.is_file()}
    if actual_names != set(expected['models']):
        raise AuditFailure('model_inventory_mismatch')
    for name, expected_digest in expected['models'].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or any(p in ('.', '..') for p in relative.parts) or not SHA256.fullmatch(expected_digest):
            raise AuditFailure('unsafe_model_manifest')
        path = (models_root / name).resolve(strict=True)
        if not path.is_relative_to(models_root):
            raise AuditFailure('unsafe_model_path')
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != expected_digest:
                raise AuditFailure('model_sha256_mismatch')
    result['exact_images_policy_models'] = 'identical'
    compare_storage_keys(expected['storage'], run_sql(root, STORAGE_SQL))
    http = StageHTTP(base, compose_literal(config['services']['api']['environment']['SUPABASE_SERVICE_KEY']), api_bases(config))
    result['storage'] = http.objects(expected['storage'])
    bucket = config['services']['api']['environment'].get('V4_CORRECTIONS_BUCKET', 'arborscan-corrections')
    result['indexed_file_integrity'] = verify_payload_links(http, run_sql(root, LINK_SQL), bucket)
    result['stored_url_compatibility'] = classify_url_records(run_sql(root, URL_SQL))
    for version in ('v3', 'v4'):
        if http.api('GET', '/api/' + version + '/health').get('status') != 'ok':
            raise AuditFailure('staging_api_health')
    for path in ('/api/v3/auth/me', '/api/v4/v4/reports', '/api/v4/v4/corrections'):
        http.api('GET', path, expected=401)
    result.update(checked_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                  api_health='passed', authorization_without_token='refused',
                  source_user_data_printed=False, production_modified=False)
    return result, http


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fingerprint-sql', action='store_true')
    parser.add_argument('--storage-fingerprints-sql', action='store_true')
    parser.add_argument('--normalize-fingerprints', type=Path)
    parser.add_argument('--build-expected', action='store_true')
    parser.add_argument('--source-fingerprints', type=Path)
    parser.add_argument('--source-storage-fingerprints', type=Path)
    parser.add_argument('--storage-snapshot', type=Path)
    parser.add_argument('--live-inspect', type=Path)
    parser.add_argument('--assets-root', type=Path)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--expected', type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--repeat-smoke', action='store_true')
    args = parser.parse_args()
    if args.fingerprint_sql:
        print(FINGERPRINT_SQL)
        return
    if args.storage_fingerprints_sql:
        print(STORAGE_FULL_FINGERPRINT_SQL)
        return
    if args.normalize_fingerprints:
        print(json.dumps(parse_fingerprints(args.normalize_fingerprints.read_bytes()), sort_keys=True))
        return
    if args.build_expected:
        if not all((args.root, args.source_fingerprints, args.source_storage_fingerprints,
                    args.storage_snapshot, args.live_inspect, args.assets_root)):
            parser.error('--build-expected needs --root, --source-fingerprints, '
                         '--source-storage-fingerprints, --storage-snapshot, --live-inspect, --assets-root')
        os.umask(0o077)
        database = parse_fingerprints(args.source_fingerprints.read_bytes())
        storage_database = parse_storage_fingerprints(args.source_storage_fingerprints.read_bytes())
        expected = build_expected(database, args.storage_snapshot, args.live_inspect,
                                  args.assets_root, storage_database)
        output = args.root / 'audit.expected.private.json'
        write_private(output, expected)
        print(json.dumps({'expected_manifest_built': True, 'tables': len(database['tables']),
                          'objects': len(expected['storage']), 'model_files': len(expected['models']),
                          'private_manifest': True}))
        return
    if not args.root or (not args.expected and not args.repeat_smoke):
        parser.error('--root and --expected required for the first audit')
    os.umask(0o077)
    if args.repeat_smoke:
        # The continuity digest was captured before synthetic fixtures. Do not
        # assert unchanged row counts after this tool deliberately creates them.
        config = private_json(args.root / 'compose.private.json')
        manifest = json.loads((args.root / 'rootmanifest.safe.json').read_text())
        base = guarded_stack(args.root, config, manifest.get('assets_root'))
        http = StageHTTP(base, compose_literal(config['services']['api']['environment']['SUPABASE_SERVICE_KEY']), api_bases(config))
        if not (args.root / 'fixture.private.json').exists():
            raise AuditFailure('restart_fixture_required')
        result = {'smoke': smoke(http, args.root / 'fixture.private.json')}
    else:
        result, http = audit(args.root, private_json(args.expected))
        if args.smoke:
            result['smoke'] = smoke(http, args.root / 'fixture.private.json')
    write_private(args.root / 'audit-result.json', result)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # requests/SQL/path exceptions can include credentials or private keys.
        category = str(error) if isinstance(error, AuditFailure) else 'unexpected_' + type(error).__name__
        print(json.dumps({'selfhost_audit': 'failed', 'category': category}))
        raise SystemExit(1)
