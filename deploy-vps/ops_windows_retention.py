"""Windows retention14 after current native/runtime verification; protected history stays.

Uses the existing timestamp sets and additive RELOCATED_RUNTIME map. It never
deletes shared runtime archives. Explicit adoption is bound to manifest bytes.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat

NAME = re.compile(r'\d{8}T\d{6}Z')
SHA = re.compile(r'[a-f0-9]{64}')
SERVICES = {'/arborscan-api', '/arborscan-api-v4', '/arborscan-quality-worker'}


@contextlib.contextmanager
def locked(root):
    path = safe_path(root, Path(root)/'pull.lock')
    with path.open('a+b') as stream:
        stream.seek(0); stream.write(b'0'); stream.flush(); stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise ValueError('retention_already_running') from None
        yield


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical_root(root):
    root = Path(root).absolute()
    for parent in [root, *root.parents]:
        if parent.is_symlink() or (parent.exists() and getattr(parent.lstat(),
                'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 1024)):
            raise ValueError('symlink_or_reparse_refused')
    return root.resolve(strict=True)


def safe_path(root, path):
    root = canonical_root(root)
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    # Resolve only after checking every component, including parents.
    for parent in [path, *path.parents]:
        if parent.is_symlink() or (parent.exists() and getattr(parent.lstat(),
                'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 1024)):
            raise ValueError('symlink_or_reparse_refused')
    if not path.resolve().is_relative_to(root):
        raise ValueError('unsafe_destination')
    return path


def verify_data(folder):
    folder = Path(folder)
    seen = {}
    for line in (folder/'SHA256SUMS').read_text(encoding='utf-8').splitlines():
        expected, relative = line.split('  ', 1)
        parsed = PurePosixPath(relative)
        canonical = str(parsed)
        if (not SHA.fullmatch(expected) or canonical in seen or '\\' in relative or ':' in relative
                or parsed.is_absolute() or '..' in parsed.parts or canonical == '.'):
            raise ValueError('invalid_manifest')
        path = safe_path(folder, canonical)
        if not path.is_file() or digest(path) != expected:
            raise ValueError('checksum_mismatch')
        seen[canonical] = expected
    if not {'application.tar', 'local-files.tar', 'arborscan.env.private',
            'containers.private.json'} <= seen.keys():
        raise ValueError('missing_required_archives')
    return seen


def verify_runtime(root, folder, require_auth=False):
    entries = verify_data(folder)
    if 'RUNTIME_DEPENDENCIES.json' not in entries:
        raise ValueError('runtime_archives_required')
    runtime = json.loads((folder/'RUNTIME_DEPENDENCIES.json').read_bytes())
    services = runtime.get('services', sorted(SERVICES - {'/arborscan-api'}))
    if (runtime.get('format') != 1 or not isinstance(services, list) or
            len(services) != len(set(services)) or
            set(services) not in (SERVICES, SERVICES - {'/arborscan-api'})):
        raise ValueError('invalid_runtime_services')
    if require_auth and set(services) != SERVICES:
        raise ValueError('auth_runtime_required')
    rows = json.loads((folder/'containers.private.json').read_bytes())
    selected = {row['Name']: row['Image'] for row in rows if row.get('Name') in services}
    if selected.keys() != set(services) or set(selected.values()) != set(runtime.get('images', {})):
        raise ValueError('runtime_index_image_mismatch')
    mapping = json.loads((folder/'RELOCATED_RUNTIME.json').read_bytes())
    if mapping.get('format') != 1:
        raise ValueError('invalid_runtime_relocation')
    relocated = {(row['original_path'], row['sha256']): row['restored_path']
                 for row in mapping['assets']}
    checked = {}
    for assets in runtime['images'].values():
        if not isinstance(assets, list) or not assets:
            raise ValueError('runtime_archive_missing')
        for asset in assets:
            expected = asset['sha256']
            if not SHA.fullmatch(expected):
                raise ValueError('unsafe_runtime_archive')
            key = (asset['path'], expected)
            if key not in relocated:
                raise ValueError('runtime_relocation_missing')
            path = safe_path(root, relocated[key])
            if path not in checked:
                checked[path] = digest(path)
            if checked[path] != expected:
                raise ValueError('runtime_archive_checksum_mismatch')
    return {'runtime_services': sorted(services), 'unique_runtime_archives': len(checked)}


def verify_replacement(root, name):
    if not NAME.fullmatch(name):
        raise ValueError('invalid_backup_name')
    folder = safe_path(root, name)
    if not (folder/'COMPLETE').is_file():
        raise ValueError('replacement_incomplete')
    entries = verify_data(folder)
    required = {'postgres/' + n for n in ('database.dump', 'roles.sql', 'source.json',
                'archive-list.private.txt', 'COMPLETE', 'SHA256SUMS')}
    if not required <= entries.keys():
        raise ValueError('native_postgres_required')
    native = {}
    for line in (folder/'postgres/SHA256SUMS').read_text().splitlines():
        expected, item = line.split('  ', 1)
        native[item] = expected
    if native.keys() != {'database.dump', 'roles.sql', 'source.json', 'archive-list.private.txt'}:
        raise ValueError('invalid_postgres_manifest')
    if any(entries['postgres/'+n] != d for n, d in native.items()):
        raise ValueError('postgres_manifest_mismatch')
    with (folder/'postgres/database.dump').open('rb') as stream:
        if stream.read(5) != b'PGDMP':
            raise ValueError('invalid_postgres_dump_header')
    return verify_runtime(root, folder, require_auth=True)


def plan(root, replacement, protected=(), retain=14):
    root = canonical_root(root)
    verify_replacement(root, replacement)
    policy_file = root/'WINDOWS_RETENTION_POLICY.private.json'
    adoption = {}
    if policy_file.is_file():
        policy = json.loads(safe_path(root, policy_file).read_bytes())
        if policy.get('format') != 1 or policy.get('retention') != 14:
            raise ValueError('invalid_retention_policy')
        for row in policy.get('automatic_sets', []):
            if (not NAME.fullmatch(row['name']) or not SHA.fullmatch(row['manifest_sha256']) or
                    row.get('source') != 'observed_automatic_run' or not row.get('evidence_reference')):
                raise ValueError('invalid_retention_adoption')
            adoption[row['name']] = row['manifest_sha256']
        protected = set(protected) | set(policy.get('protected_sets', []))
    rows = []
    dependencies = set(protected)
    complete_folders = [p for p in sorted(root.iterdir())
                        if NAME.fullmatch(p.name) and (p/'COMPLETE').is_file()]
    def references(value):
        if isinstance(value, dict):
            for child in value.values(): references(child)
        elif isinstance(value, list):
            for child in value: references(child)
        elif isinstance(value, str):
            for name in re.findall(r'(?:/ops-backups/|ArborScanBackups[\\/])(\d{8}T\d{6}Z)(?:[\\/]|$)',value):
                dependencies.add(name)
    for folder in complete_folders:
        for metadata in ('RUNTIME_DEPENDENCIES.json','RELOCATED_RUNTIME.json','containers.private.json'):
            path = folder/metadata
            if path.exists():
                try:
                    references(json.loads(safe_path(root,path).read_bytes()))
                except (OSError,ValueError,TypeError):
                    raise ValueError('dependency_inventory_unreadable') from None
    for folder in sorted(root.iterdir()):
        if not NAME.fullmatch(folder.name) or not (folder/'COMPLETE').is_file():
            continue
        safe_path(root, folder)
        try:
            entries = verify_data(folder)
        except (OSError, ValueError):
            rows.append({'name':folder.name,'verified':False,'protected':['invalid_or_incomplete']})
            continue
        manifest_sha = digest(folder/'SHA256SUMS')
        reasons = []
        for pin in ('PINNED', 'DO_NOT_ROTATE', 'ROLLBACK_PIN'):
            if (folder/pin).exists():
                reasons.append(pin)
        if folder.name in dependencies:
            reasons.append('live_or_runtime_dependency')
        if folder.name == replacement:
            reasons.append('verified_replacement')
        automatic = False
        if 'AUTO_BACKUP.json' in entries:
            marker = json.loads((folder/'AUTO_BACKUP.json').read_bytes())
            automatic = (marker.get('format') == 1 and marker.get('kind') == 'scheduled_backup'
                         and marker.get('name') == folder.name)
        if adoption.get(folder.name) == manifest_sha:
            automatic = True
        if not automatic:
            reasons.append('manual_or_unadopted_history')
        rows.append({'name':folder.name,'manifest_sha256':manifest_sha,'verified':True,
                     'protected':reasons,'automatic':automatic})
    verified = [r for r in rows if r['verified']]
    excess = max(0, len(verified)-retain)
    eligible = [r for r in verified if not r['protected'] and r['name'] < replacement]
    # An older replacement cannot authorize deletion of a newer completed set.
    if len(eligible) < excess:
        raise ValueError('rotation_blocked_all_remaining_sets_protected')
    return {'format':1,'replacement':replacement,'retain':retain,'before':len(verified),
            'delete':[r['name'] for r in eligible[:excess]],'sets':rows}


def apply(root, proposal, protected=(), lock_held=False):
    root = canonical_root(root)
    if not lock_held:
        with locked(root):
            return apply(root, proposal, protected, lock_held=True)
    current = plan(root, proposal['replacement'], protected, proposal['retain'])
    if current != proposal:
        raise ValueError('retention_plan_changed')
    deleted = []
    for name in proposal['delete']:
        # Re-read replacement and all references before each bounded removal.
        latest = plan(root, proposal['replacement'], protected, proposal['retain'])
        if name not in latest['delete']:
            raise ValueError('retention_plan_changed')
        target = safe_path(root, name)
        if target.parent != root or not NAME.fullmatch(target.name):
            raise ValueError('unsafe_rotation_target')
        shutil.rmtree(target)
        deleted.append(name)
    return {'deleted':deleted,'retained_verified_sets':proposal['before']-len(deleted)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(r'D:\ArborScanBackups'))
    parser.add_argument('--replacement', required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    try:
        proposal = plan(args.root, args.replacement)
        print(json.dumps(apply(args.root, proposal) if args.apply else proposal))
    except (OSError, ValueError, KeyError, TypeError):
        raise SystemExit('Retention refused; inspect private policy and usable replacement')
