"""Reviewed-command candidate: AS-09 weather configuration, API-v4 only.

prepare creates PRIVATE copies of operator config/runtime; it never changes
production. deploy/rollback require an explicit mode after the operator has
saved the key. No secret is printed. No source, DB, model, v3 or worker changes.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


API = 'arborscan-api-v4'
IMAGE = 'sha256:9dc0ad620110a347b7bbd5a050ad84a6d2453e16165abc15aee47550ff8e017b'
REVISION = '4483d48100514ec249bb7e72cff6565f0f7eb324'
OLD_APPLICATION = 'sha256:3fe28a5b7eb5716a71e984908520c31fff9ac325a9c5885c5e102e93996ac14a'
CONFIG = Path('/etc/arborscan/arborscan.env')
ALIASES = ('WEATHER_API_KEY', 'OPENWEATHER_API_KEY', 'OPENWEATHERMAP_API_KEY')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def private_write(path, data):
    path = Path(path)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(data if isinstance(data, bytes) else data.encode('utf-8'))


def private_json(path, value):
    private_write(path, json.dumps(value, indent=2, sort_keys=True))


def command(argv, timeout=30, environ=None):
    result = subprocess.run(argv, capture_output=True, timeout=timeout,
                            env=environ, check=False)
    require(result.returncode == 0, 'Read-only command failed; no secret output forwarded')
    return result.stdout


def inspect(name):
    rows = json.loads(command(['docker', 'inspect', name]))
    require(len(rows) == 1, 'Ambiguous Docker object')
    return rows[0]


def environment(row):
    return dict(item.split('=', 1) for item in row['Config'].get('Env', [])
                if '=' in item)


def env_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def nonweather(value):
    return {key: item for key, item in value.items() if key not in ALIASES}


def runtime_identity(row):
    return {'image': row['Image'], 'started_at': row['State']['StartedAt'],
            'environment_sha256': env_digest(environment(row))}


def invariant(name):
    row = inspect(name)
    return {'image': row['Image'], 'started_at': row['State']['StartedAt']}


def compose_prefix(plan):
    args = ['docker', 'compose', '-p', plan['project']]
    for file in plan['compose']:
        args += ['-f', file]
    return args


def process_environment(plan):
    # Existing worker interpolation is needed for config parsing only.
    return {**os.environ, 'MODEL_QUALITY_IMAGE': plan['worker']['image']}


def effective_environment(plan, override):
    raw = command(compose_prefix(plan) + ['-f', str(override),
                  'config', '--format', 'json'], environ=process_environment(plan))
    cfg = json.loads(raw)['services']['api-v4']
    require(cfg.get('image') in (IMAGE, OLD_APPLICATION), 'Unpinned effective API image')
    defaults = environment(inspect(cfg['image']))
    values = cfg.get('environment', {})
    require(isinstance(values, dict), 'Unexpected effective environment shape')
    defaults.update({key: '' if value is None else str(value) for key, value in values.items()})
    return defaults


def verify_invariants(plan):
    require(sha(CONFIG) == plan['operator_config_sha256'], 'Operator config changed after prepare')
    require(all(sha(path) == digest for path, digest in plan['compose_sha256'].items()),
            'Compose configuration changed after prepare')
    current_models = {str(p): sha(p) for p in Path('/opt/arborscan/models').glob('*.pt')}
    require(current_models == plan['models'],
            'Models changed after prepare')
    require(invariant('arborscan-api') == plan['v3'], 'API-v3 state changed; stop')
    require(invariant('arborscan-quality-worker') == plan['worker'], 'Worker state changed; stop')


def checked_backup(path):
    path = Path(path).resolve()
    require(path.parent == Path('/home/arborscan/ops-backups'), 'Invalid backup path')
    require(re.fullmatch(r'\d{8}T\d{6}Z', path.name) is not None, 'Unexpected backup name')
    stamp = datetime.strptime(path.name, '%Y%m%dT%H%M%SZ').replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - stamp).total_seconds()
    require(0 <= age <= 36 * 3600, 'Backup must be complete and at most 36 hours old')
    require((path/'COMPLETE').is_file() and (path/'postgres/COMPLETE').is_file(),
            'Backup lacks complete PostgreSQL/application markers')
    checked = subprocess.run(['sha256sum', '--quiet', '-c', 'SHA256SUMS'],
                             cwd=path, capture_output=True, timeout=120)
    require(checked.returncode == 0, 'Backup integrity verification failed')
    return path


def prepare(root, backup):
    backup = checked_backup(backup)
    row = inspect(API)
    require(row['Image'] == IMAGE, 'API-v4 image differs from verified AS-09 version')
    require(row['Config']['Labels'].get('org.opencontainers.image.revision') == REVISION,
            'API-v4 revision differs from verified AS-09 source')
    require(row['State'].get('Health', {}).get('Status') == 'healthy', 'API-v4 is not healthy')
    old_env = environment(row)
    require(not any(old_env.get(key, '').strip() for key in ALIASES),
            'Previous runtime already has a weather key; this procedure is for initial activation')
    labels = row['Config']['Labels']
    files = labels['com.docker.compose.project.config_files'].split(',')
    require(all(Path(path).is_file() for path in files), 'Compose source unavailable')
    require(inspect(OLD_APPLICATION)['Id'] == OLD_APPLICATION, 'Previous application image unavailable')
    plan = {'created_at': datetime.now(timezone.utc).isoformat(), 'backup': str(backup),
            'candidate': IMAGE, 'revision': REVISION, 'old_application': OLD_APPLICATION,
            'operator_config_sha256': sha(CONFIG), 'project': labels['com.docker.compose.project'],
            'compose': files, 'compose_sha256': {p: sha(p) for p in files},
            'models': {str(p): sha(p) for p in Path('/opt/arborscan/models').glob('*.pt')},
            'v3': invariant('arborscan-api'), 'worker': invariant('arborscan-quality-worker'),
            'override_sha256': {}}
    root.mkdir(mode=0o700, exist_ok=False)
    # This is a fresh configuration/runtime backup, not a restore or DB write.
    private_write(root/'operator-config.env.private', CONFIG.read_bytes())
    private_json(root/'runtime-before.private.json', row)
    for index, file in enumerate(files):
        private_write(root/f'compose-{index}.private', Path(file).read_bytes())
    require(sha(root/'operator-config.env.private') == plan['operator_config_sha256'],
            'Private operator configuration copy failed verification')
    require(json.loads((root/'runtime-before.private.json').read_text()) == row,
            'Private runtime copy failed verification')
    require(all(sha(root/f'compose-{index}.private') == plan['compose_sha256'][file]
                for index, file in enumerate(files)), 'Private Compose copy failed verification')
    backup_files = [root/'operator-config.env.private', root/'runtime-before.private.json',
                    *(root/f'compose-{index}.private' for index in range(len(files)))]
    private_json(root/'backup-manifest.private.json', {p.name: sha(p) for p in backup_files})
    for mode, image, fields in (
        ('deploy', IMAGE, {}),
        ('rollback-weather', IMAGE, {key: '' for key in ALIASES}),
        ('rollback-application', OLD_APPLICATION, {}),
    ):
        override = {'services': {'api-v4': {'image': image}}}
        if fields:
            override['services']['api-v4']['environment'] = fields
        private_json(root/(mode+'.private.yml'), override)
        plan['override_sha256'][mode] = sha(root/(mode+'.private.yml'))
        effective = effective_environment(plan, root/(mode+'.private.yml'))
        if mode == 'deploy':
            require(any(effective.get(key, '').strip() for key in ALIASES),
                    'Operator key is not present in the effective API-v4 configuration')
            require(nonweather(effective) == nonweather(old_env),
                    'Configuration changes extend beyond weather aliases; stop for review')
        elif mode == 'rollback-weather':
            require(not any(effective.get(key, '').strip() for key in ALIASES),
                    'Weather rollback does not clear every alias')
            require(nonweather(effective) == nonweather(old_env),
                    'Weather rollback changes another runtime setting')
        plan[mode+'_env_sha256'] = env_digest(effective)
    verify_invariants(plan)
    require(runtime_identity(inspect(API)) == runtime_identity(row),
            'API-v4 changed while prepare was running')
    private_json(root/'plan.private.json', plan)
    private_json(root/'state.private.json', runtime_identity(row))
    private_write(root/'PREPARED', datetime.now(timezone.utc).isoformat())
    print('Prepared API-v4-only weather plan; fresh private config/runtime copies verified; nothing restarted')


def apply_mode(root, mode):
    require((root/'PREPARED').is_file(), 'Private plan is not completely prepared')
    plan = json.loads((root/'plan.private.json').read_text())
    state = json.loads((root/'state.private.json').read_text())
    verify_invariants(plan)
    require(runtime_identity(inspect(API)) == state, 'API-v4 changed outside this plan; stop')
    override = root/(mode+'.private.yml')
    require(sha(override) == plan['override_sha256'][mode], 'Private override changed after prepare')
    require(env_digest(effective_environment(plan, override)) == plan[mode+'_env_sha256'],
            'Effective API environment differs from prepared plan')
    argv = compose_prefix(plan) + ['-f', str(override), 'up', '-d', '--no-build',
            '--pull', 'never', '--no-deps', '--force-recreate', 'api-v4']
    log_path = root/(mode+'.private.log')
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, 'ab') as log:
        os.fchmod(log.fileno(), 0o600)
        try:
            result = subprocess.run(argv, stdout=log, stderr=log, timeout=90,
                                    env=process_environment(plan), check=False)
            return_code = result.returncode
        except subprocess.TimeoutExpired:
            return_code = None
    after = inspect(API)
    expected = OLD_APPLICATION if mode == 'rollback-application' else IMAGE
    require(after['Image'] == expected, 'Unexpected API image after operation; inspect private diagnostics')
    require(env_digest(environment(after)) == plan[mode+'_env_sha256'],
            'Unexpected API environment after operation; inspect private diagnostics')
    private_json(root/'state.private.json', runtime_identity(after))
    verify_invariants(plan)
    require(return_code == 0, 'Compose failed or timed out; private log retained; reviewed rollback remains available')
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        after = inspect(API)
        if after['State'].get('Health', {}).get('Status') == 'healthy':
            break
        require(after['State'].get('Running'), 'API-v4 exited; reviewed rollback remains available')
        time.sleep(2)
    require(after['State'].get('Health', {}).get('Status') == 'healthy',
            'API-v4 did not become healthy within 90 seconds; reviewed rollback remains available')
    verify_invariants(plan)
    print('API-v4-only '+mode+' completed; public health/auth and real weather checks still required')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'deploy', 'rollback-weather', 'rollback-application'])
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--backup', type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    root = args.plan.resolve()
    require(root.parent == Path('/home/arborscan') and
            re.fullmatch(r'as09-weather-\d{8}T\d{6}Z', root.name), 'Invalid private plan path')
    if args.mode == 'prepare':
        require(args.backup is not None, 'Prepare requires a verified recent backup')
        prepare(root, args.backup)
    else:
        apply_mode(root, args.mode)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Raw subprocess stderr, config, provider keys and tracebacks stay private.
        safe = str(error) if isinstance(error, RuntimeError) else 'Operation stopped; no secret diagnostics forwarded'
        print(safe, file=sys.stderr)
        sys.exit(1)
