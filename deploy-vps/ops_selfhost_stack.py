"""AS-14/16: restore NEW private self-host infrastructure, never cut over API.

Uses an already verified native PostgreSQL backup, cloned application assets and
private current Docker inspect metadata. Does not download restricted provider
data, delete sessions/jobs, start a worker, train, or touch production services.
Compose contains private credentials: retain mode 600 under a mode-700 root.
"""
from __future__ import annotations

import argparse
import base64
import datetime
import hashlib
import hmac
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import subprocess
import sys
import time
import urllib.request

PINNED = {
    'db': 'supabase/postgres@sha256:178f0976b54a39237096bfa310c1a352dbc82fb1b08dda45cdb8acb5d40c1426',
    'rest': 'postgrest/postgrest@sha256:c9dc201e555f5d8e37e7f39cdd4df0229774996e213bfd7de8d10ac609030f2c',
    'storage': 'supabase/storage-api@sha256:b393ac5759a45a934557a150ecdb97157bbdd419e0193e4c5dc5770b9ddd7ee1',
    'gateway': 'nginx@sha256:a8b39bd9cf0f83869a2162827a0caf6137ddf759d50a171451b335cecc87d236',
}
LIVE_NAMES = {'v3': '/arborscan-api', 'api': '/arborscan-api-v4',
              'worker': '/arborscan-quality-worker'}
MOUNT_TARGETS = {'/app/models': 'models', '/app/model-quality': 'model-quality',
                 '/tmp/arborscan-models': 'model-cache',
                 '/home/arborscan/.u2net': 'u2net'}
ROOT_NAME = re.compile(r'selfhost-\d{8}T\d{6}Z(?:-[a-z0-9]+)?')
IMAGE_ID = re.compile(r'sha256:[0-9a-f]{64}')
ENV_KEY = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')
HOME = Path('/home/arborscan')
PUBLIC_FINGERPRINT_SQL = r"""
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY;
SET LOCAL TIME ZONE 'UTC';
SET LOCAL DateStyle = 'ISO, YMD';
SET LOCAL IntervalStyle = 'postgres';
SET LOCAL extra_float_digits = 1;
SELECT format($q$SELECT json_build_object('table', %L, 'rows', count(*),
 'fingerprint', md5(coalesce(string_agg(row_to_json(t)::text, E'\n'
 ORDER BY row_to_json(t)::text COLLATE "C"), ''))) FROM public.%I t$q$,
 tablename, tablename) FROM pg_tables WHERE schemaname='public' ORDER BY tablename
\gexec
COMMIT;
"""


def private_write(path, payload):
    raw = payload.encode() if isinstance(payload, str) else payload
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw)


def checked_root(path, parent=None):
    parent = HOME if parent is None else parent
    root = Path(path).absolute()
    if root.parent != parent or not ROOT_NAME.fullmatch(root.name) or root.is_symlink():
        raise ValueError('New private selfhost root required')
    if root.exists():
        if not root.is_dir() or root.stat().st_mode & 0o077:
            raise ValueError('Selfhost root permissions must be 700')
        if hasattr(os, 'getuid') and root.stat().st_uid != os.getuid():
            raise ValueError('Selfhost root owner differs')
    else:
        root.mkdir(mode=0o700)
    if (root / 'compose.private.json').exists() or (root / 'PREPARED').exists():
        raise ValueError('Existing selfhost plan retained; choose a new root')
    return root


def verify_manifest(root, required=()):
    """Reject escape/symlink/duplicate/omitted required files; compare real bytes."""
    root = Path(root).resolve(strict=True)
    if not (root / 'COMPLETE').is_file():
        raise ValueError('Complete native backup required')
    seen = set()
    for line in (root / 'SHA256SUMS').read_text().splitlines():
        match = re.fullmatch(r'([0-9a-f]{64})  (.+)', line)
        if match is None:
            raise ValueError('Invalid backup manifest')
        digest, name = match.groups()
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or '\\' in name or name in seen:
            raise ValueError('Unsafe backup manifest')
        target = root.joinpath(*relative.parts)
        if (not target.is_file() or any(p.is_symlink() for p in [target, *target.parents] if p != root.parent)
                or not target.resolve().is_relative_to(root)):
            raise ValueError('Invalid backup member')
        with target.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != digest:
                raise ValueError('Backup checksum mismatch')
        seen.add(name)
    if not set(required).issubset(seen):
        raise ValueError('Required native backup members missing')
    return len(seen)


def live_rows(path):
    rows = json.loads(Path(path).read_text())
    if not isinstance(rows, list):
        raise ValueError('Private Docker inspect inventory required')
    selected = {}
    for service, name in LIVE_NAMES.items():
        matches = [r for r in rows if r.get('Name') == name]
        if len(matches) != 1 or not IMAGE_ID.fullmatch(matches[0].get('Image', '')):
            raise ValueError('Exact live application image required')
        selected[service] = matches[0]
    return selected


def verify_snapshot(root):
    root = Path(root).resolve(strict=True)
    expected = json.loads((root / 'manifest.json').read_text())
    if not isinstance(expected, dict) or not expected or not (root / 'objects').is_dir():
        raise ValueError('Verified restored Storage snapshot required')
    objects = set()
    for name, digest in expected.items():
        relative = PurePosixPath(name)
        if (relative.is_absolute() or '..' in relative.parts or '\\' in name
                or not re.fullmatch('[0-9a-f]{64}', digest)):
            raise ValueError('Unsafe Storage snapshot manifest')
        target = root.joinpath(*relative.parts)
        if (not target.is_file() or target.is_symlink() or not target.resolve().is_relative_to(root)
                or any(p.is_symlink() for p in target.parents if p.is_relative_to(root))):
            raise ValueError('Invalid Storage snapshot member')
        with target.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != digest:
                raise ValueError('Storage snapshot checksum mismatch')
        if relative.parts[0] == 'objects':
            objects.add(name)
    actual = {p.relative_to(root).as_posix() for p in (root / 'objects').rglob('*') if p.is_file()}
    if actual != objects:
        raise ValueError('Storage snapshot coverage mismatch')
    return len(objects)


def environment(row, service_key):
    result = {}
    for item in row['Config'].get('Env') or []:
        key, separator, value = item.partition('=')
        if not separator or not ENV_KEY.fullmatch(key) or key in result or '\x00' in value:
            raise ValueError('Invalid application environment')
        result[key] = value
    result['SUPABASE_URL'] = 'http://gateway:8000'
    result['SUPABASE_SERVICE_KEY'] = service_key
    if 'SUPABASE_SERVICE_ROLE_KEY' in result:
        result['SUPABASE_SERVICE_ROLE_KEY'] = service_key
    # Compose interpolates JSON too. Escape literal dollars, preserve all other
    # provider configuration exactly; do not print this private dictionary.
    return {key: value.replace('$', '$$') for key, value in result.items()}


def jwt(secret, role, issued, valid_days):
    if type(valid_days) is not int or not 1 <= valid_days <= 3650:
        raise ValueError('JWT validity must be 1..3650 days')
    enc = lambda raw: base64.urlsafe_b64encode(raw).rstrip(b'=')
    claims = {'role': role, 'iss': 'arborscan-selfhost', 'iat': issued,
              'exp': issued + valid_days * 86400}
    message = enc(b'{"alg":"HS256","typ":"JWT"}') + b'.' + enc(json.dumps(claims).encode())
    return (message + b'.' + enc(hmac.new(secret.encode(), message, hashlib.sha256).digest())).decode()


def checked_jwt(token, secret, role):
    try:
        header, body, signature = token.split('.')
        decode = lambda raw: base64.urlsafe_b64decode(raw + '=' * (-len(raw) % 4))
        expected = hmac.new(secret.encode(), (header + '.' + body).encode(), hashlib.sha256).digest()
        claims = json.loads(decode(body))
        if (json.loads(decode(header)) != {'alg': 'HS256', 'typ': 'JWT'}
                or not hmac.compare_digest(decode(signature), expected)
                or set(claims) != {'role', 'iss', 'iat', 'exp'}
                or claims['role'] != role or claims['iss'] != 'arborscan-selfhost'
                or type(claims['iat']) is not int or type(claims['exp']) is not int
                or claims['iat'] > time.time() + 60 or claims['exp'] <= time.time()
                or not 86400 <= claims['exp'] - claims['iat'] <= 3650 * 86400
                or (claims['exp'] - claims['iat']) % 86400):
            raise ValueError()
        return claims
    except (ValueError, TypeError, KeyError, UnicodeError, AttributeError):
        raise ValueError('Existing private service token invalid') from None


def compose_plan(root, assets, rows, images, secret, dbpass, valid_days=3650, issued=None):
    if len(secret) < 64 or len(dbpass) < 64 or not re.fullmatch('[0-9a-f]+', dbpass):
        raise ValueError('Generated 256-bit service credentials required')
    issued = int(time.time()) if issued is None else issued
    service_key = jwt(secret, 'service_role', issued, valid_days)
    anon_key = jwt(secret, 'anon', issued, valid_days)
    root, assets = Path(root), Path(assets).resolve(strict=True)
    common = {'restart': 'unless-stopped', 'init': True, 'pids_limit': 256,
              'security_opt': ['no-new-privileges:true'], 'cap_drop': ['ALL'],
              'logging': {'driver': 'json-file', 'options': {'max-size': '5m', 'max-file': '3'}}}

    def application(service, port=None):
        row = rows[service]
        mounts = []
        for mount in row.get('Mounts') or []:
            target = mount.get('Destination')
            if target not in MOUNT_TARGETS or mount.get('Type') not in ('bind', 'volume'):
                raise ValueError('Unsupported live mount; clone and review explicitly')
            source = assets / MOUNT_TARGETS[target]
            if not source.is_dir() or source.is_symlink() or source.resolve() == Path(mount['Source']).resolve():
                raise ValueError('Independent cloned assets required')
            if any(path.is_symlink() for path in source.rglob('*')):
                raise ValueError('Cloned assets must not contain symbolic links')
            mounts.append({'type': 'bind', 'source': str(source), 'target': target,
                           'read_only': not mount.get('RW', False)})
        if not any(m['target'] == '/app/models' and m['read_only'] for m in mounts):
            raise ValueError('Read-only cloned models required')
        value = {**common, 'image': row['Image'], 'pull_policy': 'never',
                 'environment': environment(row, service_key), 'volumes': mounts,
                 'networks': ['default', 'providers'], 'mem_limit': '2g', 'cpus': 1}
        if port:
            value['ports'] = [f'127.0.0.1:{19000 if service == "v3" else 19001}:{port}']
        if service == 'worker':
            value.update({'profiles': ['manual-worker'], 'command': ['python', '-m', 'arborscan_v4.quality_worker'],
                          'restart': 'no', 'mem_limit': '3g', 'cpus': 2})
        return value

    services = {
        'db': {**common, 'image': images['db'], 'pull_policy': 'never', 'user': 'postgres',
               'entrypoint': ['/bin/bash', '-c'], 'command': [
                   'test -f /var/lib/postgresql/data/selfhost/PG_VERSION || '
                   'initdb -D /var/lib/postgresql/data/selfhost -U supabase_admin '
                   '--auth-local=trust --auth-host=reject; '
                   'exec postgres -D /var/lib/postgresql/data/selfhost '
                   '-c listen_addresses=* -c shared_preload_libraries='],
               'volumes': ['db:/var/lib/postgresql/data'], 'mem_limit': '2g', 'cpus': 1},
        'rest': {**common, 'image': images['rest'], 'pull_policy': 'never', 'mem_limit': '256m',
                 'environment': {'PGRST_DB_URI': f'postgres://authenticator:{dbpass}@db:5432/postgres',
                                 'PGRST_DB_SCHEMAS': 'public', 'PGRST_DB_ANON_ROLE': 'anon',
                                 'PGRST_JWT_SECRET': secret}},
        'storage': {**common, 'image': images['storage'], 'pull_policy': 'never', 'mem_limit': '512m',
                    'environment': {'ANON_KEY': anon_key, 'SERVICE_KEY': service_key, 'AUTH_JWT_SECRET': secret,
                                    'POSTGREST_URL': 'http://rest:3000',
                                    'DATABASE_URL': f'postgres://supabase_storage_admin:{dbpass}@db:5432/postgres',
                                    'STORAGE_BACKEND': 'file', 'GLOBAL_S3_BUCKET': 'arborscan',
                                    'FILE_STORAGE_BACKEND_PATH': '/var/lib/storage', 'TENANT_ID': 'arborscan',
                                    'REGION': 'local', 'FILE_SIZE_LIMIT': '268435456',
                                    'ENABLE_IMAGE_TRANSFORMATION': 'false'},
                    'volumes': ['storage:/var/lib/storage']},
        'gateway': {**common, 'image': images['gateway'], 'pull_policy': 'never', 'mem_limit': '128m',
                    'cap_add': ['CHOWN', 'SETGID', 'SETUID'],
                    'networks': ['default', 'providers'], 'ports': ['127.0.0.1:19080:8000'],
                    'volumes': [{'type': 'bind', 'source': str(root / 'gateway.conf'),
                                 'target': '/etc/nginx/nginx.conf', 'read_only': True}]},
        'v3': application('v3', 8000), 'api': application('api', 8001), 'worker': application('worker'),
    }
    return {'name': root.name.lower(), 'services': services,
            'networks': {'default': {'internal': True}, 'providers': {}},
            'volumes': {'db': {}, 'storage': {}}}


GATEWAY = '''events {}\nhttp {
  access_log off;
  server { listen 8000; client_max_body_size 256m;
    location /rest/v1/ { proxy_pass http://rest:3000/; }
    location /storage/v1/ { proxy_pass http://storage:5000/; }
  }
}\n'''


def role_restore_sql(raw):
    lines = raw.splitlines()
    grants = [line for line in lines if line.startswith('GRANT ')]
    if any(not line.endswith(' GRANTED BY supabase_admin;') for line in grants):
        raise ValueError('Unexpected native role grantor')
    definitions = '\n'.join(line for line in lines if not line.startswith('GRANT ')
                            and line != 'CREATE ROLE supabase_admin;')
    return definitions + '\nSET ROLE supabase_admin;\n' + '\n'.join(grants) + '\nRESET ROLE;\n'


class Runner:
    def __init__(self, root):
        self.root = root

    def run(self, args, data=None, required=True, timeout=600):
        with (self.root / 'diagnostics.private.log').open('ab') as log:
            result = subprocess.run(args, input=data, stdout=subprocess.PIPE, stderr=log, timeout=timeout)
            os.chmod(self.root / 'diagnostics.private.log', 0o600)
        if required and result.returncode:
            raise RuntimeError('Private staging command failed')
        return result


def public_fingerprints(run, sql):
    # Locale, timezone and float display settings differ between managed and
    # self-hosted PostgreSQL. Pin serialization and byte ordering before hashing,
    # and capture every table in one consistent read-only snapshot.
    rows = run(sql + ['-q'], PUBLIC_FINGERPRINT_SQL.encode()).stdout
    result = {}
    for line in rows.decode().splitlines():
        row = json.loads(line)
        name = row.get('table')
        if (set(row) != {'table', 'rows', 'fingerprint'} or not isinstance(name, str)
                or not ENV_KEY.fullmatch(name) or name in result or type(row['rows']) is not int
                or row['rows'] < 0 or not isinstance(row['fingerprint'], str)
                or not re.fullmatch(r'[0-9a-f]{32}', row['fingerprint'])):
            raise ValueError('Unexpected public table fingerprint')
        result[name] = {'rows': row['rows'], 'fingerprint': row['fingerprint']}
    if not result:
        raise ValueError('Unexpected public table inventory')
    return result


def restore_database(runner, command, backup, dbpass):
    """Restores the whole native dump. Only local service passwords are changed."""
    sql = command + ['exec', '-T', 'db', 'psql', '-U', 'supabase_admin', '-d', 'postgres',
                     '-v', 'ON_ERROR_STOP=1', '-At']
    runner.run(sql, role_restore_sql((backup / 'roles.sql').read_text()).encode())
    runner.run(command + ['exec', '-T', 'db', 'pg_restore', '-U', 'supabase_admin', '-d', 'postgres',
                          '--exit-on-error', '--single-transaction'], (backup / 'database.dump').read_bytes(), timeout=1800)
    before = public_fingerprints(runner.run, sql)
    # Users, sessions, model pointers, jobs, reports and decisions are untouched.
    runner.run(sql, (f"ALTER ROLE authenticator PASSWORD '{dbpass}'; "
                     f"ALTER ROLE supabase_storage_admin PASSWORD '{dbpass}';").encode())
    runner.run(command + ['exec', '-T', 'db', 'sh', '-c',
                          "printf 'local all all trust\\nhost all all all scram-sha-256\\n' "
                          '> /var/lib/postgresql/data/selfhost/pg_hba.conf'])
    runner.run(sql, b'SELECT pg_reload_conf();')
    versions = json.loads(runner.run(sql, b"SELECT json_build_array(public.contour_workflow_version(), public.server_history_version(), public.model_quality_version());").stdout)
    if versions != [1, 1, 1] or before != public_fingerprints(runner.run, sql):
        raise RuntimeError('Restored application data or migration verification failed')
    return before, versions


def storage_ready(token, attempts=30):
    for _ in range(attempts):
        try:
            request = urllib.request.Request('http://127.0.0.1:19080/storage/v1/bucket',
                                              headers={'Authorization': 'Bearer ' + token, 'apikey': token})
            with urllib.request.urlopen(request, timeout=5) as response:
                if response.status == 200 and isinstance(json.load(response), list):
                    return
        except (OSError, ValueError):
            pass
        time.sleep(2)
    raise RuntimeError('Private Storage/PostgREST readiness failed')


def final_manifest(args, root, images, rows, before, versions, expires_at):
    return {'format': 1, 'root': str(root), 'postgres_backup': str(args.pg_backup.resolve()),
            'storage_snapshot': str(args.storage_snapshot.resolve()), 'assets_root': str(args.assets_root.resolve()),
            'images': {**images, **{name: row['Image'] for name, row in rows.items()}},
            'versions': versions, 'public_tables': before, 'sessions_preserved': True,
            'jobs_unchanged': True, 'api_started': False, 'worker_started': False,
            'storage_files_imported': False, 'production_cutover': False,
            'credential_rotation_due_before': datetime.datetime.fromtimestamp(expires_at, datetime.timezone.utc).isoformat()}


def publish_prepared(root, manifest):
    path = root / 'rootmanifest.safe.json'
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError('Existing preparation receipt differs; retained unchanged')
    else:
        private_write(path, json.dumps(manifest, indent=2))
    if not (root / 'PREPARED').exists():
        private_write(root / 'PREPARED', b'private infrastructure only\n')
    return {key: manifest[key] for key in ('root', 'images', 'versions', 'sessions_preserved',
            'jobs_unchanged', 'api_started', 'worker_started', 'storage_files_imported', 'production_cutover')}


def checked_infrastructure(runner, config):
    ids = runner.run(['docker', 'ps', '-aq', '--filter',
                      'label=com.docker.compose.project=' + config['name']]).stdout.decode().split()
    if not ids:
        raise RuntimeError('Existing private infrastructure missing')
    rows = json.loads(runner.run(['docker', 'inspect', *ids]).stdout)
    found = set()
    for row in rows:
        labels = row['Config']['Labels']
        service = labels.get('com.docker.compose.service')
        running = row['State']['Running']
        if service not in config['services'] or labels.get('com.docker.compose.project') != config['name']:
            raise ValueError('Unexpected private project service')
        if service in ('v3', 'api', 'worker'):
            if running:
                raise ValueError('Resume requires API and worker not running')
            continue
        if not running or service in found or row['Image'] != config['services'][service]['image']:
            raise RuntimeError('Private infrastructure state differs')
        found.add(service)
        networks = set(row['NetworkSettings']['Networks'])
        expected_networks = {config['name'] + '_default'}
        if service == 'gateway':
            expected_networks.add(config['name'] + '_providers')
        if networks != expected_networks or row['HostConfig'].get('NetworkMode') == 'host':
            raise ValueError('Private infrastructure network differs')
        bindings = row['HostConfig'].get('PortBindings') or {}
        expected_ports = {'8000/tcp': [{'HostIp': '127.0.0.1', 'HostPort': '19080'}]} if service == 'gateway' else {}
        if bindings != expected_ports:
            raise ValueError('Private infrastructure port bindings differ')
    if found != {'db', 'rest', 'storage', 'gateway'}:
        raise RuntimeError('Private infrastructure incomplete')


def checked_resume_root(path):
    root = Path(path).absolute()
    if (root.parent != HOME or not ROOT_NAME.fullmatch(root.name) or root.is_symlink()
            or not root.is_dir() or root.stat().st_mode & 0o077):
        raise ValueError('Existing private staging root required')
    if hasattr(os, 'getuid') and root.stat().st_uid != os.getuid():
        raise ValueError('Existing private root owner differs')
    return root


def checked_private_file(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o077:
        raise ValueError('Existing private file permissions differ')


def verify_prepared(args):
    """Read-only resume after an incomplete readiness probe; never restores DB."""
    os.umask(0o077)
    root = checked_resume_root(args.root)
    for source in (args.pg_backup, args.storage_snapshot, args.live_inspect,
                   args.assets_root, args.expected_public_fingerprints):
        if not source.resolve(strict=True).is_relative_to(HOME):
            raise ValueError('Private verified source required')
    compose = root / 'compose.private.json'
    checked_private_file(compose)
    if ((root / 'gateway.conf').is_symlink() or (root / 'gateway.conf').read_text() != GATEWAY):
        raise ValueError('Existing private configuration differs')
    config = json.loads(compose.read_text())
    secret = config['services']['rest']['environment']['PGRST_JWT_SECRET']
    uri = config['services']['rest']['environment']['PGRST_DB_URI']
    match = re.fullmatch(r'postgres://authenticator:([0-9a-f]{64})@db:5432/postgres', uri)
    if match is None:
        raise ValueError('Existing private database credentials invalid')
    token = config['services']['storage']['environment']['SERVICE_KEY']
    claims = checked_jwt(token, secret, 'service_role')
    checked_jwt(config['services']['storage']['environment']['ANON_KEY'], secret, 'anon')
    valid_days = (claims['exp'] - claims['iat']) // 86400
    rows = live_rows(args.live_inspect)
    runner = Runner(root)
    images = {}
    for service, reference in PINNED.items():
        actual = runner.run(['docker', 'image', 'inspect', reference, '--format', '{{.Id}}']).stdout.decode().strip()
        if not IMAGE_ID.fullmatch(actual):
            raise ValueError('Pinned dependency image unavailable')
        images[service] = actual
    expected_config = compose_plan(root, args.assets_root, rows, images, secret, match[1], valid_days, claims['iat'])
    if config != expected_config:
        raise ValueError('Existing private Compose differs from verified inputs')
    verify_manifest(args.pg_backup, ('roles.sql', 'database.dump', 'source.json'))
    verify_snapshot(args.storage_snapshot)
    checked_infrastructure(runner, config)
    command = ['docker', 'compose', '-f', str(compose)]
    sql = command + ['exec', '-T', 'db', 'psql', '-U', 'supabase_admin', '-d', 'postgres',
                     '-v', 'ON_ERROR_STOP=1', '-At']
    before = public_fingerprints(runner.run, sql)
    expected = json.loads(args.expected_public_fingerprints.read_text())
    if not expected or before != expected:
        raise RuntimeError('Restored data differ from independently captured source fingerprints')
    versions = json.loads(runner.run(sql, b"SELECT json_build_array(public.contour_workflow_version(), public.server_history_version(), public.model_quality_version());").stdout)
    if versions != [1, 1, 1]:
        raise RuntimeError('Private migration versions differ')
    storage_ready(token)
    return publish_prepared(root, final_manifest(args, root, images, rows, before, versions, claims['exp']))


def prepare(args):
    os.umask(0o077)
    backup = args.pg_backup.resolve(strict=True)
    snapshot = args.storage_snapshot.resolve(strict=True)
    assets = args.assets_root.resolve(strict=True)
    for source in (backup, snapshot, assets, args.live_inspect.resolve(strict=True)):
        if not source.is_relative_to(HOME):
            raise ValueError('Private sources under /home/arborscan required')
    verify_manifest(backup, ('roles.sql', 'database.dump', 'source.json'))
    verify_snapshot(snapshot)
    rows = live_rows(args.live_inspect)
    root = checked_root(args.root)
    runner = Runner(root)
    images = {}
    for service, reference in PINNED.items():
        actual = runner.run(['docker', 'image', 'inspect', reference, '--format', '{{.Id}}']).stdout.decode().strip()
        if not IMAGE_ID.fullmatch(actual):
            raise ValueError('Existing pinned dependency image required')
        images[service] = actual
    for row in rows.values():
        actual = runner.run(['docker', 'image', 'inspect', row['Image'], '--format', '{{.Id}}']).stdout.decode().strip()
        if actual != row['Image']:
            raise ValueError('Current application image missing')
    # No unknown pre-existing project volume may be reused accidentally.
    for suffix in ('db', 'storage'):
        result = runner.run(['docker', 'volume', 'inspect', root.name.lower() + '_' + suffix], required=False)
        if result.returncode == 0:
            raise ValueError('Existing staging volume retained; choose a new root')
    secret, dbpass = secrets.token_hex(32), secrets.token_hex(32)
    config = compose_plan(root, assets, rows, images, secret, dbpass, args.jwt_valid_days)
    compose = root / 'compose.private.json'
    private_write(compose, json.dumps(config, indent=2))
    private_write(root / 'gateway.conf', GATEWAY)
    # This file has no credentials. Container root has no DAC_OVERRIDE and
    # differs from the host owner, so nginx must be able to read the bind mount.
    # The enclosing directory stays 700; private Compose/credentials stay 600.
    os.chmod(root / 'gateway.conf', 0o644)
    command = ['docker', 'compose', '-f', str(compose)]
    runner.run(command + ['config', '--quiet'])
    runner.run(['docker', 'volume', 'create', config['name'] + '_db'])
    runner.run(['docker', 'run', '--rm', '--pull=never', '--network', 'none', '--user', 'root',
                '-v', config['name'] + '_db:/restore-db', '--entrypoint', 'sh', images['db'],
                '-c', 'chown postgres:postgres /restore-db'])
    runner.run(command + ['up', '-d', '--pull', 'never', 'db'])
    for _ in range(30):
        if runner.run(command + ['exec', '-T', 'db', 'pg_isready', '-U', 'supabase_admin'], required=False).returncode == 0:
            break
        time.sleep(1)
    else:
        raise RuntimeError('Private database not ready')
    before, versions = restore_database(runner, command, backup, dbpass)
    runner.run(command + ['up', '-d', '--pull', 'never', 'rest', 'storage', 'gateway'])
    token = config['services']['storage']['environment']['SERVICE_KEY']
    storage_ready(token)
    expires = checked_jwt(token, secret, 'service_role')['exp']
    return publish_prepared(root, final_manifest(args, root, images, rows, before, versions, expires))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'verify-prepared'):
        command = commands.add_parser(name)
        for option in ('pg-backup', 'storage-snapshot', 'live-inspect', 'assets-root', 'root'):
            command.add_argument('--' + option, type=Path, required=True)
        if name == 'prepare':
            command.add_argument('--jwt-valid-days', type=int, default=3650)
        else:
            command.add_argument('--expected-public-fingerprints', type=Path, required=True)
    args = parser.parse_args()
    try:
        action = prepare if args.command == 'prepare' else verify_prepared
        print(json.dumps(action(args)))
    except Exception:
        print('Selfhost preparation incomplete; private diagnostics retained; production unchanged.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
