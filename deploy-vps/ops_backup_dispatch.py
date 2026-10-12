"""Existing systemd entry: cloud until a private self-host config is installed.

Installing this module alone does not switch a backup source. There is no env
loading, automatic cutover, password prompt, worker launch or deletion here.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def dispatch(args, *, config=None, legacy=subprocess.run, scheduled=None):
    directory = Path(__file__).parent
    config = config or directory / 'SELFHOST_BACKUP_CONFIG.private.json'
    if not config.exists():
        command = [sys.executable, str(directory / 'ops_backup_policy.py'), 'run', '--root', str(args.root)]
        if args.operation_id:
            command += ['--operation-id', args.operation_id]
        if args.runtime_index:
            command += ['--runtime-index', str(args.runtime_index)]
        return legacy(command).returncode
    if config.is_symlink() or (os.name == 'posix' and config.stat().st_mode & 0o077):
        raise ValueError('private_backup_config_required')
    value = json.loads(config.read_bytes())
    if (not isinstance(value, dict) or set(value) != {'format', 'root', 'runtime_index', 'storage_layout'}
            or value['format'] != 1 or value['storage_layout'] not in {'version-directory', 'version-suffix', 'unversioned'}
            or not isinstance(value['root'], str) or not isinstance(value['runtime_index'], str)):
        raise ValueError('invalid_selfhost_backup_config')
    if args.runtime_index and str(args.runtime_index) != value['runtime_index']:
        raise ValueError('configured_runtime_index_required')
    from ops_selfhost_backup import scheduled_backup, safe_path, PRIVATE_HOME
    root, index = safe_path(value['root']), safe_path(value['runtime_index'])
    if root.parent != PRIVATE_HOME or not (root / 'PREPARED').is_file() or not index.is_relative_to(PRIVATE_HOME):
        raise ValueError('prepared_selfhost_root_required')
    result = (scheduled or scheduled_backup)(root, args.root, index,
        layout=value['storage_layout'], operation_id=args.operation_id)
    print(json.dumps(result))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/home/arborscan/ops-backups'))
    parser.add_argument('--operation-id')
    parser.add_argument('--runtime-index', type=Path)
    args = parser.parse_args()
    try:
        return dispatch(args)
    except Exception:
        print(json.dumps({'event': 'backup_dispatch_failed',
                          'reason': 'private_validation_or_runtime_error'}), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
