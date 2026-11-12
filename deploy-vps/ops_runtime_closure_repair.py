"""Prepare a reviewed immutable selfhost runtime index, never switch production.

The input inventory must describe actual Docker images, not a possibly stale
base Compose file. New images need a complete Docker-save archive whose SHA
and exact linux/amd64 image closure are checked. Historic mappings and files are
preserved. This tool does not export images, start containers, run a backup,
delete failed stages, or change SELFHOST_BACKUP_CONFIG.private.json.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import tarfile

from ops_runtime_contract import selfhost_inventory
from ops_selfhost_backup import verify_saved_image, BackupError
from ops_selfhost_runtime_preflight import coverage, read_index, private_path, PRIVATE_HOME


class RepairError(ValueError):
    """A fixed safe category, not contents of private input."""


def file_sha(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def index_plan(source, rows, *, private_home=PRIVATE_HOME):
    source = private_path(source, private_home)
    runtime = read_index(source, private_home)
    try:
        names, images = selfhost_inventory(rows)
    except (ValueError, TypeError, AttributeError):
        raise RepairError('invalid_selfhost_runtime_inventory') from None
    return {'format': 1, 'kind': 'selfhost_runtime_index_repair',
            'source_index_sha256': file_sha(source),
            'actual_services': len(names), 'actual_unique_images': len(images),
            'missing_image_ids': sorted(images - runtime['images'].keys()),
            'source_index_changed': False, 'configuration_changed': False,
            'backup_started': False, 'containers_started': False}


def build_candidate(source, rows, additions, *, private_home=PRIVATE_HOME):
    """Return checked index bytes and plan; callers choose a new private target."""
    source = private_path(source, private_home)
    before = source.read_bytes()
    runtime = read_index(source, private_home)
    plan = index_plan(source, rows, private_home=private_home)
    if not isinstance(additions, dict) or set(additions) != set(plan['missing_image_ids']):
        raise RepairError('exact_missing_image_archives_required')
    candidate = copy.deepcopy(runtime)
    for image, item in additions.items():
        if (not isinstance(item, dict) or set(item) != {'path', 'sha256'}
                or not isinstance(item['sha256'], str)
                or not re.fullmatch(r'[a-f0-9]{64}', item['sha256'])):
            raise RepairError('invalid_replacement_archive')
        archive = private_path(item['path'], private_home)
        if file_sha(archive) != item['sha256']:
            raise RepairError('replacement_archive_checksum_mismatch')
        try:
            verify_saved_image(archive, image)
        except (BackupError, KeyError, ValueError, OSError, tarfile.TarError):
            raise RepairError('replacement_archive_image_mismatch') from None
        candidate['images'][image] = [{'path': str(archive), 'sha256': item['sha256']}]
    # Keep prior image mappings too: historical complete sets may require them.
    candidate['selfhost'] = True
    candidate['service_images'] = {
        row['Config']['Labels']['com.docker.compose.service']: row['Image'] for row in rows}
    candidate['platform'] = {'os': 'linux', 'architecture': 'amd64'}
    stats = coverage(rows, candidate, private_home=private_home)
    verified = {}
    fingerprints = {}
    for assets in candidate['images'].values():
        if not isinstance(assets, list) or not assets:
            raise RepairError('historical_runtime_assets_invalid')
        for item in assets:
            if (not isinstance(item, dict) or not isinstance(item.get('path'), str)
                    or not isinstance(item.get('sha256'), str)
                    or not re.fullmatch(r'[a-f0-9]{64}', item['sha256'])):
                raise RepairError('historical_runtime_assets_invalid')
            path = private_path(item['path'], private_home)
            if str(path) not in verified:
                before_stat = path.stat()
                actual = file_sha(path)
                after_stat = path.stat()
                state = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
                if state(before_stat) != state(after_stat):
                    raise RepairError('runtime_asset_changed_during_repair')
                verified[str(path)] = actual
                fingerprints[str(path)] = state(after_stat)
            if verified[str(path)] != item['sha256']:
                raise RepairError('runtime_archive_checksum_mismatch')
    for name, fingerprint in fingerprints.items():
        path = private_path(name, private_home)
        stat = path.stat()
        if (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != fingerprint:
            raise RepairError('runtime_asset_changed_during_repair')
    if source.read_bytes() != before:
        raise RepairError('source_runtime_index_changed')
    raw = (json.dumps(candidate, sort_keys=True, indent=2) + '\n').encode('utf-8')
    plan.update(candidate_index_sha256=hashlib.sha256(raw).hexdigest(),
                archives_verified=len(verified),
                archive_bytes=sum(Path(path).stat().st_size for path in verified),
                current_closure=stats, integrity_verified=True,
                previous_image_entries_preserved=True)
    return raw, plan


def write_candidate(source, rows, additions, destination, *, private_home=PRIVATE_HOME):
    destination, home = Path(destination), Path(private_home)
    private_path(destination.parent, home, directory=True)
    if (not destination.is_absolute() or destination.parent != home / 'ops-runtime'
            or not re.fullmatch(r'repair-[A-Za-z0-9_-]{1,80}', destination.name)
            or destination.exists() or destination.is_symlink()
            or any(p.is_symlink() for p in destination.parents)
            or not destination.parent.is_dir()):
        raise RepairError('new_private_candidate_directory_required')
    raw, plan = build_candidate(source, rows, additions, private_home=home)
    # A source race cannot silently switch the object we just reviewed.
    if file_sha(source) != plan['source_index_sha256']:
        raise RepairError('source_runtime_index_changed')
    destination.mkdir(mode=0o700)
    for name, payload in (
            ('RUNTIME_INDEX.json', raw),
            ('REPAIR_PLAN.json', (json.dumps(plan, sort_keys=True, indent=2) + '\n').encode('utf-8'))):
        with (destination / name).open('xb') as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-index', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True,
                        help='Private actual Docker-inspect inventory, not base Compose')
    parser.add_argument('--additions', type=Path,
                        help='Private JSON mapping missing image ID to full Docker-save path/SHA')
    parser.add_argument('--destination', type=Path,
                        help='New /home/arborscan/ops-runtime/repair-NAME; omission is read-only')
    args = parser.parse_args()
    try:
        rows = json.loads(private_path(args.inventory, PRIVATE_HOME).read_bytes())
        if args.destination is None:
            result = index_plan(args.source_index, rows)
        else:
            if args.additions is None:
                raise RepairError('exact_missing_image_archives_required')
            additions = json.loads(private_path(args.additions, PRIVATE_HOME).read_bytes())
            result = write_candidate(args.source_index, rows, additions, args.destination)
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception:
        # Keep arbitrary exception, filenames, stderr and private inventory out.
        print(json.dumps({'event': 'runtime_index_repair_refused',
                          'reason': 'private_validation_required'}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
