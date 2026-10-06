"""AS-16 guarded API-v4-only rollout. No SQL, worker, v3, HTTPS or model writes.

prepare verifies an existing recent complete backup and privately captures the
actual runtime/configuration plus its exact image. deploy and rollback are
explicit separate actions; neither builds code nor upgrades dependencies.
Raw command output, environment values and diagnostics are never printed.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

API = 'arborscan-api-v4'
CURRENT_IMAGE = 'sha256:9dc0ad620110a347b7bbd5a050ad84a6d2453e16165abc15aee47550ff8e017b'
CURRENT_REVISION = '4483d48100514ec249bb7e72cff6565f0f7eb324'
CONFIG = Path('/etc/arborscan/arborscan.env')
HOME = Path('/home/arborscan')
BACKUPS = HOME / 'ops-backups'
MODELS = Path('/opt/arborscan/models')
HTTPS_CONFIG = Path('/etc/nginx')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def private_write(path, raw):
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, 'O_NOFOLLOW', 0)
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(raw if isinstance(raw, bytes) else raw.encode('utf-8'))


def private_json(path, value):
    private_write(path, json.dumps(value, sort_keys=True, indent=2))


def command(argv, timeout=30, environ=None):
    result = subprocess.run(argv, capture_output=True, timeout=timeout,
                            env=environ, check=False)
    require(result.returncode == 0, 'Command failed; no private output forwarded')
    return result.stdout


def inspect(name):
    rows = json.loads(command(['docker', 'inspect', name]))
    require(isinstance(rows, list) and len(rows) == 1, 'Ambiguous Docker object')
    return rows[0]


def environment(row):
    return dict(item.split('=', 1) for item in row['Config'].get('Env', []) if '=' in item)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def runtime_identity(row):
    return {'id': row['Id'], 'image': row['Image'],
            'started_at': row['State']['StartedAt'],
            'environment_sha256': digest(environment(row))}


def invariant(name):
    return runtime_identity(inspect(name))


def tree_hashes(root):
    require(root.is_dir(), 'Invariant directory is unavailable')
    return {str(path): sha(path) for path in sorted(root.rglob('*')) if path.is_file()}


def verify_manifest(root, manifest):
    entries = {}
    for line in manifest.read_text().splitlines():
        match = re.fullmatch(r'([0-9a-f]{64}) [ *](.+)', line)
        require(match is not None, 'Invalid backup checksum manifest')
        name = match[2]
        path = (root / name).resolve()
        require(path.is_relative_to(root.resolve()) and path.is_file(),
                'Unsafe or missing backup manifest entry')
        normalized = path.relative_to(root.resolve()).as_posix()
        require(normalized not in entries, 'Unsafe or missing backup manifest entry')
        require(sha(path) == match[1], 'Backup checksum mismatch')
        entries[normalized] = match[1]
    require(bool(entries), 'Empty backup checksum manifest')
    return entries


def checked_backup(path=None, require_latest=True, require_recent=True):
    available = sorted(p for p in BACKUPS.iterdir() if p.is_dir() and
        re.fullmatch(r'\d{8}T\d{6}Z', p.name) and (p / 'COMPLETE').is_file())
    require(bool(available), 'No complete application backup is available')
    root = (Path(path) if path else available[-1]).resolve()
    require(root.parent == BACKUPS.resolve(), 'Invalid backup path')
    require(re.fullmatch(r'\d{8}T\d{6}Z', root.name) is not None, 'Invalid backup timestamp')
    if require_latest:
        require(root == available[-1].resolve(), 'Use the latest complete backup')
    stamp = datetime.strptime(root.name, '%Y%m%dT%H%M%SZ').replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - stamp).total_seconds()
    if require_recent:
        require(0 <= age <= 36 * 3600, 'Backup must be at most 36 hours old')
    require((root / 'COMPLETE').is_file() and (root / 'postgres/COMPLETE').is_file(),
            'Application/PostgreSQL backup is incomplete')
    checks = verify_manifest(root, root / 'SHA256SUMS')
    required = {'application.tar', 'local-files.tar', 'arborscan.env.private',
                'containers.private.json', 'restore-result.json',
                'postgres/database.dump', 'postgres/roles.sql', 'postgres/source.json',
                'postgres/archive-list.private.txt', 'postgres/SHA256SUMS', 'postgres/COMPLETE'}
    require(required <= checks.keys(), 'Backup manifest omits required application/PostgreSQL files')
    pg = verify_manifest(root / 'postgres', root / 'postgres/SHA256SUMS')
    require({'database.dump', 'roles.sql', 'source.json', 'archive-list.private.txt'} <= pg.keys(),
            'PostgreSQL manifest is incomplete')
    with (root / 'postgres/database.dump').open('rb') as stream:
        require(stream.read(5) == b'PGDMP', 'PostgreSQL custom archive header is invalid')
    restored = json.loads((root / 'restore-result.json').read_text())
    require(type(restored.get('files_restored')) is int and restored['files_restored'] > 0 and
            restored.get('production_credentials_loaded') is False,
            'Application offline-restore evidence is missing')
    return root


def compose_prefix(plan):
    args = ['docker', 'compose', '-p', plan['project'],
            '--project-directory', plan['working_directory']]
    for file in plan['compose']:
        args += ['-f', file]
    return args


def process_environment(plan):
    # Parse existing worker interpolation but never target/recreate that service.
    return {**os.environ, 'MODEL_QUALITY_IMAGE': plan['worker']['image']}


def effective_environment(plan, override, image):
    raw = command(compose_prefix(plan) + ['-f', str(override), 'config', '--format', 'json'],
                  environ=process_environment(plan))
    service = json.loads(raw)['services']['api-v4']
    require(service.get('image') == image, 'Effective API image is not exactly pinned')
    values = service.get('environment', {})
    require(isinstance(values, dict), 'Unexpected effective environment shape')
    defaults = environment(inspect(image))
    defaults.update({k: '' if v is None else str(v) for k, v in values.items()})
    return defaults


def verify_invariants(plan):
    require(sha(CONFIG) == plan['operator_config_sha256'], 'Operator configuration changed')
    require(all(sha(path) == value for path, value in plan['compose_sha256'].items()),
            'Compose configuration changed')
    require(tree_hashes(MODELS) == plan['models'], 'Model files changed')
    require(tree_hashes(HTTPS_CONFIG) == plan['https_configuration'], 'HTTPS configuration changed')
    require(invariant('arborscan-api') == plan['v3'], 'API-v3 changed; stop')
    require(invariant('arborscan-quality-worker') == plan['worker'], 'Worker changed; stop')


def save_runtime_image(root, row):
    image = inspect(row['Image'])
    require(image['Id'] == row['Image'], 'Original runtime image identity changed')
    require(shutil.disk_usage(root).free > image['Size'] + 1024 ** 3,
            'Insufficient space for the exact runtime-image backup')
    target = root / 'runtime-image.private.tar'
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as archive:
        result = subprocess.run(['docker', 'image', 'save', row['Image']],
                                stdout=archive, stderr=subprocess.PIPE, timeout=900, check=False)
    require(result.returncode == 0 and target.stat().st_size > 0,
            'Original runtime image backup failed')


def prepare(root, backup, commit, image):
    require(re.fullmatch(r'[0-9a-f]{40}', commit) is not None, 'A full exact commit is required')
    backup = checked_backup(backup)
    before = inspect(API)
    require(before['Image'] == CURRENT_IMAGE, 'API-v4 is not the verified current image')
    require(before['Config']['Labels'].get('org.opencontainers.image.revision') == CURRENT_REVISION,
            'API-v4 source revision differs from the verified current version')
    require(before['State'].get('Running') and
            before['State'].get('Health', {}).get('Status') == 'healthy', 'API-v4 is not healthy')
    candidate = inspect(image)
    require(re.fullmatch(r'sha256:[0-9a-f]{64}', candidate['Id']) is not None,
            'Candidate image lacks an immutable ID')
    require(candidate['Config']['Labels'].get('org.opencontainers.image.revision') == commit,
            'Candidate image revision does not match the exact requested commit')
    require(candidate['Id'] != before['Image'], 'Candidate must differ from the current image')
    labels = before['Config']['Labels']
    files = labels['com.docker.compose.project.config_files'].split(',')
    require(files and all(Path(p).is_file() for p in files), 'Compose sources are unavailable')
    require(labels.get('com.docker.compose.service') == 'api-v4', 'Unexpected API service identity')
    work = labels.get('com.docker.compose.project.working_dir')
    require(isinstance(work, str) and Path(work).is_dir(), 'Compose working directory is unavailable')
    plan = {'version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
            'backup': str(backup), 'backup_manifest_sha256': sha(backup / 'SHA256SUMS'),
            'candidate': candidate['Id'], 'revision': commit,
            'original_image': before['Image'], 'original_revision': CURRENT_REVISION,
            'operator_config_sha256': sha(CONFIG), 'project': labels['com.docker.compose.project'],
            'working_directory': work, 'compose': files, 'compose_sha256': {p: sha(p) for p in files},
            'models': tree_hashes(MODELS), 'https_configuration': tree_hashes(HTTPS_CONFIG),
            'v3': invariant('arborscan-api'), 'worker': invariant('arborscan-quality-worker'),
            'environment_sha256': digest(environment(before)), 'override_sha256': {},
            'script_sha256': sha(Path(__file__))}
    root.mkdir(mode=0o700, exist_ok=False)
    private_write(root / 'operator-config.env.private', CONFIG.read_bytes())
    private_json(root / 'runtime-before.private.json', before)
    for index, file in enumerate(files):
        private_write(root / f'compose-{index}.private', Path(file).read_bytes())
    save_runtime_image(root, before)
    copies = [root / 'operator-config.env.private', root / 'runtime-before.private.json',
              root / 'runtime-image.private.tar',
              *(root / f'compose-{i}.private' for i in range(len(files)))]
    require(sha(copies[0]) == plan['operator_config_sha256'], 'Operator copy verification failed')
    require(json.loads(copies[1].read_text()) == before, 'Runtime copy verification failed')
    require(all(sha(root / f'compose-{i}.private') == plan['compose_sha256'][p]
                for i, p in enumerate(files)), 'Compose copy verification failed')
    private_json(root / 'backup-manifest.private.json', {p.name: sha(p) for p in copies})
    plan['private_backup_manifest_sha256'] = sha(root / 'backup-manifest.private.json')
    for mode, target in [('deploy', plan['candidate']), ('rollback', plan['original_image'])]:
        override = root / (mode + '.private.yml')
        # Preserve every existing variable, including current weather aliases.
        private_json(override, {'services': {'api-v4': {
            'image': target, 'environment': environment(before)}}})
        require(digest(effective_environment(plan, override, target)) == plan['environment_sha256'],
                'Candidate/rollback would change the effective runtime environment')
        plan['override_sha256'][mode] = sha(override)
    verify_invariants(plan)
    require(runtime_identity(inspect(API)) == runtime_identity(before), 'API changed during prepare')
    private_json(root / 'plan.private.json', plan)
    private_json(root / 'state.private.json', runtime_identity(before))
    private_write(root / 'PREPARED', sha(root / 'plan.private.json'))
    print(json.dumps({'prepared': True, 'revision': commit, 'candidate_image': plan['candidate'],
                      'original_image': plan['original_image'], 'backup': backup.name,
                      'environment_unchanged': True, 'production_restarted': False}))


def current_api():
    ids = command(['docker', 'container', 'ls', '-aq', '--filter', 'name=^/' + API + '$']).split()
    require(len(ids) <= 1, 'Ambiguous API-v4 container identity')
    return inspect(API) if ids else None


def apply_mode(root, mode):
    require(mode in ('deploy', 'rollback'), 'Invalid operation')
    require((root / 'PREPARED').is_file(), 'Plan is not completely prepared')
    require((root / 'PREPARED').read_text() == sha(root / 'plan.private.json'), 'Private plan changed')
    plan = json.loads((root / 'plan.private.json').read_text())
    require(plan['script_sha256'] == sha(Path(__file__)), 'Rollout script changed after prepare')
    require(sha(root / 'backup-manifest.private.json') == plan['private_backup_manifest_sha256'],
            'Private backup manifest changed')
    for name, expected in json.loads((root / 'backup-manifest.private.json').read_text()).items():
        require(sha(root / name) == expected, 'Private runtime/configuration backup changed')
    verify_invariants(plan)
    if mode == 'deploy':
        backup = checked_backup(plan['backup'], require_latest=False)
        require(sha(backup / 'SHA256SUMS') == plan['backup_manifest_sha256'], 'Full backup changed')
    state = json.loads((root / 'state.private.json').read_text())
    before = current_api()
    same_container = before is not None and all(runtime_identity(before).get(key) == state.get(key)
        for key in ('id', 'image', 'environment_sha256'))
    require((before is None and mode == 'rollback' and state.get('missing_after_own_operation') is True) or
            (before is not None and runtime_identity(before) == state) or
            (mode == 'rollback' and same_container),
            'API-v4 changed outside this plan; stop')
    target = plan['candidate'] if mode == 'deploy' else plan['original_image']
    if mode == 'rollback':
        cached = command(['docker', 'image', 'ls', '--no-trunc', '-q']).decode().splitlines()
        if target not in cached:
            command(['docker', 'image', 'load', '-i', str(root / 'runtime-image.private.tar')], timeout=900)
    image = inspect(target)
    expected_revision = plan['revision'] if mode == 'deploy' else plan['original_revision']
    require(image['Id'] == target and image['Config']['Labels'].get('org.opencontainers.image.revision') == expected_revision,
            'Pinned runtime image identity/revision changed')
    override = root / (mode + '.private.yml')
    require(sha(override) == plan['override_sha256'][mode], 'Private API override changed')
    require(digest(effective_environment(plan, override, target)) == plan['environment_sha256'],
            'Effective environment differs from the prepared unchanged runtime')
    argv = compose_prefix(plan) + ['-f', str(override), 'up', '-d', '--no-build',
            '--pull', 'never', '--no-deps', '--force-recreate', 'api-v4']
    private_json(root / 'operation.private.json', {'mode': mode, 'phase': 'attempted',
                 'started_at': datetime.now(timezone.utc).isoformat(), 'prior_state': state})
    fd = os.open(root / (mode + '.private.log'), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, 'ab') as log:
        os.fchmod(log.fileno(), 0o600)
        try:
            result = subprocess.run(argv, stdout=log, stderr=log, timeout=90,
                                    env=process_environment(plan), check=False)
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = None
    after = current_api()
    if after is None:
        private_json(root / 'state.private.json', {'missing_after_own_operation': True})
        raise RuntimeError('API-v4 is absent after operation; prepared rollback is available')
    require(after['Image'] in (plan['candidate'], plan['original_image']) and
            digest(environment(after)) == plan['environment_sha256'],
            'Unexpected API image/environment after operation; inspect private diagnostics')
    actual_revision = plan['revision'] if after['Image'] == plan['candidate'] else plan['original_revision']
    require(after['Config']['Labels'].get('org.opencontainers.image.revision') == actual_revision,
            'Unexpected API source revision after operation; inspect private diagnostics')
    private_json(root / 'state.private.json', runtime_identity(after))
    verify_invariants(plan)
    require(code == 0 and after['Image'] == target,
            'Compose failed/timed out; private log retained and prepared rollback is available')
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        after = inspect(API)
        require(runtime_identity(after) == json.loads((root / 'state.private.json').read_text()),
                'API changed during health check')
        if after['State'].get('Health', {}).get('Status') == 'healthy':
            break
        require(after['State'].get('Running'), 'API-v4 exited; prepared rollback is available')
        time.sleep(2)
    require(after['State'].get('Health', {}).get('Status') == 'healthy',
            'API-v4 health timeout; prepared rollback is available')
    verify_invariants(plan)
    private_json(root / 'operation.private.json', {'mode': mode, 'phase': 'healthy',
                 'completed_at': datetime.now(timezone.utc).isoformat()})
    print(json.dumps({'mode': mode, 'image': target, 'revision': expected_revision,
                      'healthy': True, 'environment_unchanged': True,
                      'worker_v3_models_https_unchanged': True,
                      'public_and_authenticated_checks_required': True}))


@contextmanager
def operation_lock(root):
    import fcntl
    # Serialize this operator's different prepared plans as well as repeated calls.
    fd = os.open(HOME / '.as16-v4-rollout.private.lock',
                 os.O_RDWR | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    with os.fdopen(fd, 'a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another API-v4 rollout operation is active') from None
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'deploy', 'rollback'])
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--backup', type=Path)
    parser.add_argument('--commit')
    parser.add_argument('--image')
    args = parser.parse_args()
    os.umask(0o077)
    root = args.directory.resolve()
    require(root.parent == HOME and re.fullmatch(r'as16-v4-\d{8}T\d{6}Z', root.name),
            'Invalid private rollout directory')
    if args.mode == 'prepare':
        require(args.commit is not None and args.image is not None, 'Prepare requires --commit and --image')
        with operation_lock(root):
            prepare(root, args.backup, args.commit, args.image)
    else:
        require(args.commit is None and args.image is None and args.backup is None,
                'Deploy/rollback use only the pinned prepared plan')
        with operation_lock(root):
            apply_mode(root, args.mode)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        safe = str(error) if isinstance(error, RuntimeError) else 'Operation stopped; inspect private diagnostics'
        print(safe, file=sys.stderr)
        sys.exit(1)
