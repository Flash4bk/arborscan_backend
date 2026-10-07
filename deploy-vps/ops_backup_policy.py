"""Private bounded backups: verify, publish atomically, then dependency-safe rotation.

No application/DB writes. Legacy COMPLETE directories are read-only unless their
exact manifest is explicitly adopted using recorded automatic-run evidence.
"""
import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import time

NAME = re.compile(r'\d{8}T\d{6}Z')
SHA = re.compile(r'[a-f0-9]{64}')
RETENTION = 14
MIN_FREE = 20 * 1024**3
REQUIRED = {'application.tar', 'local-files.tar', 'arborscan.env.private',
            'containers.private.json'}
NATIVE = {'database.dump', 'roles.sql', 'source.json', 'archive-list.private.txt'}


class PolicyError(Exception):
    pass


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe_file(root, relative):
    root = Path(root)
    name = PurePosixPath(relative)
    if name.is_absolute() or '..' in name.parts or '\\' in relative or ':' in relative:
        raise PolicyError('unsafe_manifest_path')
    path = root
    for part in name.parts:
        path /= part
        if path.is_symlink():
            raise PolicyError('symlink_refused')
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise PolicyError('missing_or_unsafe_file')
    return path


def manifest(root):
    path = safe_file(root, 'SHA256SUMS')
    entries = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            expected, name = line.split('  ', 1)
        except ValueError:
            raise PolicyError('invalid_manifest') from None
        name = str(PurePosixPath(name))
        if not SHA.fullmatch(expected) or name in entries:
            raise PolicyError('invalid_manifest')
        safe_file(root, name)
        entries[name] = expected
    if not entries:
        raise PolicyError('empty_manifest')
    return entries


def verify_set(folder, require_native=False, require_runtime=False, complete=True):
    folder = Path(folder)
    if folder.is_symlink() or not folder.is_dir():
        raise PolicyError('unsafe_backup_directory')
    if complete:
        safe_file(folder, 'COMPLETE')
    entries = manifest(folder)
    if not REQUIRED <= entries.keys():
        raise PolicyError('required_files_missing')
    total = 0
    for name, expected in entries.items():
        path = safe_file(folder, name)
        if digest(path) != expected:
            raise PolicyError('checksum_mismatch')
        total += path.stat().st_size
    native = 'postgres/database.dump' in entries
    if native:
        if not {'postgres/' + n for n in NATIVE | {'COMPLETE'}} <= entries.keys():
            raise PolicyError('incomplete_postgres_set')
        if require_native and 'postgres/SHA256SUMS' not in entries:
            raise PolicyError('native_manifest_not_covered_by_outer_manifest')
        native_entries = manifest(folder / 'postgres')
        if native_entries.keys() != NATIVE:
            raise PolicyError('invalid_postgres_manifest')
        for name, expected in native_entries.items():
            if entries['postgres/' + name] != expected:
                raise PolicyError('postgres_manifest_mismatch')
        with safe_file(folder, 'postgres/database.dump').open('rb') as dump:
            if dump.read(5) != b'PGDMP':
                raise PolicyError('invalid_postgres_dump_header')
    elif require_native:
        raise PolicyError('native_postgres_required')
    dependencies = []
    if 'RUNTIME_DEPENDENCIES.json' in entries:
        runtime = json.loads(safe_file(folder, 'RUNTIME_DEPENDENCIES.json').read_text())
        dependencies = verify_runtime(folder, runtime)
    elif require_runtime:
        raise PolicyError('runtime_archives_required')
    return {'name': folder.name, 'manifest_sha256': digest(folder / 'SHA256SUMS'),
            'verified_files': len(entries), 'verified_bytes': total,
            'native_postgres': native, 'dependencies': dependencies,
            'native_manifest_covered': 'postgres/SHA256SUMS' in entries,
            'runtime_archives_verified': 'RUNTIME_DEPENDENCIES.json' in entries}


def runtime_images(folder):
    rows = json.loads(safe_file(folder, 'containers.private.json').read_text())
    selected = {row['Name']: row['Image'] for row in rows if row.get('Name') in
                ('/arborscan-api-v4', '/arborscan-quality-worker')}
    if set(selected) != {'/arborscan-api-v4', '/arborscan-quality-worker'}:
        raise PolicyError('runtime_image_inventory_missing')
    if not all(re.fullmatch(r'sha256:[a-f0-9]{64}', image) for image in selected.values()):
        raise PolicyError('invalid_runtime_image_id')
    return set(selected.values())


def verify_runtime(folder, runtime):
    if runtime.get('format') != 1 or set(runtime.get('images', {})) != runtime_images(folder):
        raise PolicyError('runtime_index_image_mismatch')
    checked = {}
    for assets in runtime['images'].values():
        if not isinstance(assets, list) or not assets:
            raise PolicyError('runtime_archive_missing')
        for asset in assets:
            path = Path(asset['path'])
            expected = asset['sha256']
            if not path.is_absolute() or path.is_symlink() or not SHA.fullmatch(expected):
                raise PolicyError('unsafe_runtime_archive')
            for parent in path.parents:
                if parent.is_symlink():
                    raise PolicyError('symlink_refused')
            if path not in checked:
                checked[path] = digest(path)
            if checked[path] != expected:
                raise PolicyError('runtime_archive_checksum_mismatch')
    return [str(path) for path in checked]


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.writing')
    if temporary.is_symlink() or path.is_symlink():
        raise PolicyError('symlink_refused')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.chmod(0o600)
    os.replace(temporary, path)


def read_json(path, default):
    path = Path(path)
    if path.is_symlink():
        raise PolicyError('symlink_refused')
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


@contextlib.contextmanager
def locked(root):
    root = Path(root)
    if root.is_symlink() or any(p.is_symlink() for p in root.parents):
        raise PolicyError('symlink_refused')
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    root.chmod(0o700)
    lock = root / 'backup.lock'
    descriptor = os.open(lock, os.O_CREAT | os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    stream = os.fdopen(descriptor, 'a')
    try:
        if os.name == 'nt':
            import msvcrt
            stream.write('0'); stream.flush(); stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                raise PolicyError('already_running') from None
        else:
            import fcntl
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise PolicyError('already_running') from None
        yield
    finally:
        stream.close()


def live_dependencies():
    inventory = subprocess.run(['docker', 'ps', '-aq'], capture_output=True, timeout=30, check=True)
    names = inventory.stdout.decode().split()
    if not names:
        raise PolicyError('live_container_inventory_empty')
    inspect = subprocess.run(['docker', 'inspect', *names], capture_output=True, timeout=30, check=True)
    paths = []
    for row in json.loads(inspect.stdout):
        labels = row.get('Config', {}).get('Labels') or {}
        paths += [p for p in labels.get('com.docker.compose.project.config_files', '').split(',') if p]
        paths += [m['Source'] for m in row.get('Mounts', []) if m.get('Source')]
    return paths


def inventory(root, live=()):
    root = Path(root)
    registry = read_json(root / 'automatic-sets.json', {'format': 1, 'sets': {}})
    if registry.get('format') != 1:
        raise PolicyError('invalid_automatic_registry')
    rows = []
    references = list(live() if callable(live) else live)
    for folder in sorted(root.iterdir()):
        if not NAME.fullmatch(folder.name):
            continue
        row = {'name': folder.name, 'valid': False, 'native_postgres': False,
               'automatic': False, 'protected': [], 'bytes': 0}
        if folder.is_symlink():
            row['error'] = 'symlink_refused'; rows.append(row); continue
        if not (folder / 'COMPLETE').is_file():
            row['error'] = 'incomplete'; rows.append(row); continue
        # Read references even from a damaged set: retaining damaged material
        # must not silently remove the base archive/config it may need.
        try:
            metadata = read_json(folder / 'RUNTIME_DEPENDENCIES.json', {})
            references.extend(asset['path'] for assets in metadata.get('images', {}).values()
                              for asset in assets)
            rows_private = read_json(folder / 'containers.private.json', [])
            for container in rows_private:
                labels = container.get('Config', {}).get('Labels') or {}
                references.extend(p for p in labels.get('com.docker.compose.project.config_files', '').split(',') if p)
        except (OSError, ValueError, KeyError, TypeError, PolicyError):
            row['protected'].append('dependency_inventory_unreadable')
        try:
            checked = verify_set(folder)
            row.update(checked); row['valid'] = True
            row['bytes'] = sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())
            references.extend(checked['dependencies'])
            marker = read_json(folder / 'AUTO_BACKUP.json', {})
            declared = registry['sets'].get(folder.name, {})
            marker_covered = 'AUTO_BACKUP.json' in manifest(folder)
            row['automatic'] = (marker_covered and marker.get('format') == 1 and
                                marker.get('kind') == 'scheduled_backup' and
                                marker.get('name') == folder.name and
                                isinstance(marker.get('operation_id'), str) and
                                re.fullmatch(r'[A-Za-z0-9_.-]{1,100}', marker['operation_id']) is not None) or (
                                    declared.get('manifest_sha256') == checked['manifest_sha256'] and
                                    declared.get('source') == 'observed_automatic_run')
            for pin in ('PINNED', 'DO_NOT_ROTATE', 'ROLLBACK_PIN'):
                if (folder / pin).exists():
                    row['protected'].append('pinned')
            if not row['automatic']:
                row['protected'].append('manual_or_unadopted')
        except (PolicyError, OSError, ValueError, KeyError, TypeError):
            row['error'] = 'integrity_or_structure_failed'
        rows.append(row)
    # Any dependent set is retained, including references originating in another
    # pinned/legacy set. Invalid dependencies refuse rotation rather than guess.
    for row in rows:
        folder = root / row['name']
        for reference in references:
            path = Path(reference)
            if path.is_absolute() and path.is_relative_to(folder):
                row['protected'].append('live_or_runtime_dependency'); break
    return rows


def plan(root, live=(), retain=RETENTION, new_name=None):
    rows = inventory(root, live)
    valid = [row for row in rows if row['valid']]
    excess = max(0, len(valid) - retain)
    eligible = [row for row in valid if row['automatic'] and not row['protected'] and row['name'] != new_name]
    selected = eligible[:excess]
    unknown_dependencies = any('dependency_inventory_unreadable' in row['protected'] for row in rows)
    return {'retain': retain, 'verified_sets': len(valid), 'excess': excess,
            'candidates': [{'name': row['name'], 'reason': 'oldest_verified_automatic_unpinned',
                            'manifest_sha256': row['manifest_sha256'], 'bytes': row['bytes']}
                           for row in selected],
            'blocked': len(selected) < excess or (excess > 0 and unknown_dependencies), 'sets': rows}


def rotate(root, proposal, new_name, live=()):
    root = Path(root)
    verify_set(root / new_name, require_native=True, require_runtime=True)
    current = plan(root, live, proposal['retain'], new_name)
    if current['blocked']:
        raise PolicyError('rotation_blocked_all_remaining_sets_protected')
    if current['candidates'] != proposal['candidates']:
        raise PolicyError('rotation_plan_changed')
    deleted = []
    for row in current['candidates']:
        folder = root / row['name']
        # Re-check exact digest and all descendants immediately before unlinking;
        # shutil.rmtree's fd-based symlink protection supplements these guards.
        checked = verify_set(folder)
        if checked['manifest_sha256'] != row['manifest_sha256']:
            raise PolicyError('rotation_candidate_changed')
        if any(path.is_symlink() for path in folder.rglob('*')):
            raise PolicyError('symlink_refused')
        if folder.parent.resolve() != root.resolve() or not NAME.fullmatch(folder.name):
            raise PolicyError('unsafe_rotation_target')
        shutil.rmtree(folder)
        deleted.append(row['name'])
    return deleted


def write_manifest(folder):
    rows = []
    for path in sorted(Path(folder).rglob('*')):
        relative = path.relative_to(folder)
        if relative.parts[0].startswith('restored') or path.name in ('SHA256SUMS', 'COMPLETE') and len(relative.parts) == 1:
            continue
        if path.is_symlink():
            raise PolicyError('symlink_refused')
        if path.is_file():
            rows.append(digest(path) + '  ' + relative.as_posix() + '\n')
    (Path(folder) / 'SHA256SUMS').write_text(''.join(rows), encoding='utf-8')


def record_success(root, name, operation_id, checked, deleted, proposal, replayed=False):
    """Repair durable completion metadata without making old data look fresh."""
    root = Path(root)
    completed = dt.datetime.fromtimestamp((root / name / 'COMPLETE').stat().st_mtime,
                                         dt.timezone.utc)
    now = dt.datetime.now(dt.timezone.utc)
    previous = read_json(root / 'last-success.json', {})
    same = (previous.get('format') == 1 and previous.get('name') == name and
            previous.get('operation_id') == operation_id and
            previous.get('manifest_sha256') == checked['manifest_sha256'])
    verified_at = now.isoformat()
    if replayed:
        # After an interrupted publish the only durable completion time may be
        # the existing marker. Never substitute the later retry time for it.
        verified_at = completed.isoformat()
        if same:
            try:
                prior = dt.datetime.fromisoformat(previous['verified_at_utc'])
                if prior.tzinfo is not None and completed <= prior <= now:
                    verified_at = previous['verified_at_utc']
            except (KeyError, TypeError, ValueError):
                pass
    history = previous.get('deleted', []) if same else []
    if not isinstance(history, list) or any(not isinstance(n, str) or not NAME.fullmatch(n) for n in history):
        history = []
    atomic_json(root / 'last-success.json', {
        'format': 1, 'name': name, 'operation_id': operation_id,
        'completed_at_utc': completed.isoformat(), 'verified_at_utc': verified_at,
        'last_integrity_check_at_utc': now.isoformat(),
        'manifest_sha256': checked['manifest_sha256'],
        'retained_verified_sets': proposal['verified_sets'] - len(deleted),
        'deleted': list(dict.fromkeys([*history, *deleted]))})


def run_backup(root, create, runtime_index, operation_id=None, live=(), free=shutil.disk_usage,
               retain=RETENTION, minimum_free=MIN_FREE):
    root = Path(root)
    operation_id = operation_id or 'daily-' + dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d')
    if not re.fullmatch(r'[A-Za-z0-9_.-]{1,100}', operation_id):
        raise PolicyError('invalid_operation_id')
    with locked(root):
        active_path = root / 'active-backup.json'
        active = read_json(active_path, {})
        if active and (active.get('format') != 1 or not NAME.fullmatch(active.get('name', ''))
                       or not isinstance(active.get('operation_id'), str)
                       or not re.fullmatch(r'[A-Za-z0-9_.-]{1,100}', active['operation_id'])
                       or not isinstance(active.get('attempt'), int) or active['attempt'] < 1):
            raise PolicyError('invalid_active_backup')
        if active and active.get('operation_id') != operation_id:
            # Always finish the interrupted operation before starting another.
            operation_id = active['operation_id']
        rows = inventory(root, live)
        for row in rows:
            if not row['valid']:
                continue
            marker = read_json(root / row['name'] / 'AUTO_BACKUP.json', {})
            if marker.get('operation_id') == operation_id:
                checked = verify_set(root / row['name'], require_native=True, require_runtime=True)
                proposal = plan(root, live, retain, row['name'])
                atomic_json(root / 'rotation-dry-run.json', proposal)
                deleted = rotate(root, proposal, row['name'], live)
                record_success(root, row['name'], operation_id, checked, deleted, proposal, replayed=True)
                if active.get('name') == row['name']:
                    active_path.unlink(missing_ok=True)
                return {'event': 'verified_existing', 'name': row['name'], 'deleted': deleted}
        name = active.get('name') or dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        if not NAME.fullmatch(name):
            raise PolicyError('invalid_active_backup')
        if (root / name).exists():
            raise PolicyError('unowned_existing_backup_name')
        if free(root).free < minimum_free:
            raise PolicyError('insufficient_disk_no_fresh_external_verification_for_predelete')
        # Explain exactly which sets would rotate after a successful extra set.
        preflight = plan(root, live, max(0, retain - 1), name)
        atomic_json(root / 'rotation-dry-run.json', preflight)
        if preflight['blocked']:
            raise PolicyError('rotation_blocked_all_remaining_sets_protected')
        active.update({'format': 1, 'name': name, 'operation_id': operation_id,
                       'attempt': active.get('attempt', 0) + 1})
        atomic_json(active_path, active)
        staging = root / '.staging' / name
        if staging.is_symlink() or staging.parent.is_symlink():
            raise PolicyError('symlink_refused')
        staging.mkdir(mode=0o700, parents=True, exist_ok=True)
        attempt = staging / ('attempt-' + str(active['attempt']))
        attempt.mkdir(mode=0o700)
        create(attempt)
        selected = runtime_images(attempt)
        runtime = read_json(runtime_index, {})
        if runtime.get('format') != 1 or not selected <= runtime.get('images', {}).keys():
            raise PolicyError('runtime_archive_index_missing')
        runtime = {'format': 1, 'images': {image: runtime['images'][image] for image in selected}}
        verify_runtime(attempt, runtime)
        atomic_json(attempt / 'RUNTIME_DEPENDENCIES.json', runtime)
        atomic_json(attempt / 'AUTO_BACKUP.json', {'format': 1, 'kind': 'scheduled_backup',
                                                 'operation_id': operation_id, 'name': name})
        write_manifest(attempt)
        checked = verify_set(attempt, require_native=True, require_runtime=True, complete=False)
        (attempt / 'COMPLETE').touch(mode=0o600)
        os.replace(attempt, root / name)
        # A crash here is recovered by the same operation_id marker above.
        proposal = plan(root, live, retain, name)
        atomic_json(root / 'rotation-dry-run.json', proposal)
        deleted = rotate(root, proposal, name, live)
        record_success(root, name, operation_id, checked, deleted, proposal)
        active_path.unlink(missing_ok=True)
        return {'event': 'completed_verified', 'name': name, 'deleted': deleted}


def adopt(root, evidence):
    """Explicit migration of observed generated sets; never infer from the name."""
    if evidence.get('format') != 1 or not isinstance(evidence.get('sets'), list):
        raise PolicyError('invalid_adoption_evidence')
    with locked(root):
        registry = read_json(Path(root) / 'automatic-sets.json', {'format': 1, 'sets': {}})
        for item in evidence['sets']:
            name = item['name']
            if (not NAME.fullmatch(name) or item.get('source') != 'observed_automatic_run'
                    or not item.get('evidence_reference')):
                raise PolicyError('invalid_adoption_evidence')
            checked = verify_set(Path(root) / name)
            if checked['manifest_sha256'] != item['manifest_sha256']:
                raise PolicyError('adoption_manifest_mismatch')
            registry['sets'][name] = item
        atomic_json(Path(root) / 'automatic-sets.json', registry)
    return {'adopted_exact_manifests': len(evidence['sets'])}


def backup_status(root, now=None):
    """Actual full/native integrity and separately recorded external freshness."""
    root = Path(root)
    now = time.time() if now is None else now
    rows = inventory(root) if root.is_dir() else []
    native = [row for row in rows if row['valid'] and row['native_postgres']]
    full = [row for row in native if row.get('runtime_archives_verified') and row.get('native_manifest_covered')]
    latest_native = native[-1] if native else None
    latest = full[-1] if full else None
    result = {'verified_sets': sum(row['valid'] for row in rows),
              'verified_native_sets': len(native),
              'verified_full_runtime_sets': len(full),
              'incomplete_or_corrupt_sets': [row['name'] for row in rows if not row['valid']],
              'latest_verified_native_backup': latest_native['name'] if latest_native else None,
              'native_backup_age_seconds': round(now - (root / latest_native['name'] / 'COMPLETE').stat().st_mtime) if latest_native else None,
              'latest_verified_full_backup': latest['name'] if latest else None,
              'full_backup_age_seconds': round(now - (root / latest['name'] / 'COMPLETE').stat().st_mtime) if latest else None,
              'latest_confirmed_external': None, 'external_confirmation_age_seconds': None}
    staging = root / '.staging'
    result['unfinished_staged_attempts'] = sum(1 for path in staging.glob('*/attempt-*')
                                              if path.is_dir() and not path.is_symlink()) if staging.is_dir() else 0
    receipt_root = root / 'offsite-receipts'
    receipts = []
    if receipt_root.is_dir() and not receipt_root.is_symlink():
        for path in receipt_root.glob('*.json'):
            try:
                receipt = read_json(path, {})
                if (receipt.get('format') != 1 or not NAME.fullmatch(receipt.get('name', ''))
                        or receipt.get('is_independent') is not True
                        or receipt.get('actual_content_verified') is not True
                        or receipt.get('source_content_verified') is not True
                        or not SHA.fullmatch(receipt.get('manifest_sha256', ''))):
                    continue
                timestamp = dt.datetime.fromisoformat(receipt['verified_at_utc']).timestamp()
                if timestamp > now + 300 or not receipt.get('destination_id'):
                    continue
                receipts.append((timestamp, receipt['name']))
            except (OSError, ValueError, KeyError, TypeError, PolicyError):
                continue
    if receipts:
        timestamp, name = max(receipts)
        result.update({'latest_confirmed_external': name,
                       'external_confirmation_age_seconds': round(now - timestamp),
                       'external_current_reachability_verified': False})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['run', 'dry-run', 'adopt'])
    parser.add_argument('--root', type=Path, default=Path('/home/arborscan/ops-backups'))
    parser.add_argument('--runtime-index', type=Path, default=Path('/home/arborscan/ops-runtime/RUNTIME_INDEX.json'))
    parser.add_argument('--evidence', type=Path)
    parser.add_argument('--operation-id')
    args = parser.parse_args()
    os.umask(0o077)
    try:
        if args.mode == 'adopt':
            if not args.evidence:
                raise PolicyError('adoption_evidence_required')
            result = adopt(args.root, read_json(args.evidence, {}))
        else:
            live = live_dependencies
            if args.mode == 'dry-run':
                with locked(args.root):
                    result = plan(args.root, live)
            else:
                def create(folder):
                    script = Path(__file__).with_name('ops_create_backup.sh')
                    result = subprocess.run(['/bin/sh', str(script), str(folder)], timeout=2300)
                    if result.returncode:
                        raise PolicyError('snapshot_stage_failed')
                result = run_backup(args.root, create, args.runtime_index, args.operation_id, live)
        print(json.dumps(result, sort_keys=True))
        return 0
    except PolicyError as error:
        print(json.dumps({'event': 'refused', 'reason': str(error)}), file=sys.stderr)
        return 0 if str(error) == 'already_running' else 1
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        print(json.dumps({'event': 'failed', 'reason': 'private_diagnostics_required'}), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
