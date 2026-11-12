"""Read-only early runtime coverage checks for an already deployed selfhost stack.

This checks image identity, index shape and asset presence before taking another
database/files snapshot. It is NOT an integrity receipt: normal publication must
still read and verify every archive SHA. No container is started and no index,
backup, diagnostic file or configuration is written here.
"""
import json
from pathlib import Path
import re
import subprocess

from ops_runtime_contract import selfhost_inventory

PRIVATE_HOME = Path('/home/arborscan')
IMAGE = re.compile(r'sha256:[a-f0-9]{64}')
SHA = re.compile(r'[a-f0-9]{64}')


class RuntimePreflightError(ValueError):
    """Contains an allowlisted operational code, never command output."""


def private_path(value, private_home, *, directory=False):
    path, home = Path(value), Path(private_home)
    if (not path.is_absolute() or path == home or not path.is_relative_to(home)
            or '..' in path.parts or not path.resolve().is_relative_to(home.resolve())):
        raise RuntimePreflightError('unsafe_runtime_preflight_path')
    for item in [path, *path.parents]:
        if item.is_symlink() or (item.exists() and getattr(item.lstat(), 'st_file_attributes', 0) & 1024):
            raise RuntimePreflightError('unsafe_runtime_preflight_path')
    if not (path.is_dir() if directory else path.is_file()):
        raise RuntimePreflightError('runtime_preflight_file_missing')
    return path


def read_index(path, private_home=PRIVATE_HOME):
    path = private_path(path, private_home)
    if path.stat().st_size > 2 * 1024**2:
        raise RuntimePreflightError('invalid_runtime_preflight_index')
    try:
        value = json.loads(path.read_bytes())
    except (ValueError, OSError):
        raise RuntimePreflightError('invalid_runtime_preflight_index') from None
    if (not isinstance(value, dict) or value.get('format') != 1
            or not isinstance(value.get('images'), dict)):
        raise RuntimePreflightError('invalid_runtime_preflight_index')
    return value


def coverage(rows, runtime, *, private_home=PRIVATE_HOME):
    """Validate actual seven-service images, preserving extra historic entries."""
    try:
        names, selected = selfhost_inventory(rows)
    except (ValueError, TypeError, AttributeError):
        raise RuntimePreflightError('invalid_selfhost_runtime_inventory') from None
    if (not isinstance(runtime, dict) or runtime.get('format') != 1
            or not isinstance(runtime.get('images'), dict)):
        raise RuntimePreflightError('invalid_runtime_preflight_index')
    if not selected <= runtime['images'].keys():
        raise RuntimePreflightError('runtime_archive_index_missing')
    assets = {}
    for image in selected:
        values = runtime['images'][image]
        if not isinstance(values, list) or not values or len(values) > 64:
            raise RuntimePreflightError('runtime_archive_missing')
        for value in values:
            if (not isinstance(value, dict) or not isinstance(value.get('path'), str)
                    or not isinstance(value.get('sha256'), str)
                    or not SHA.fullmatch(value['sha256'])):
                raise RuntimePreflightError('invalid_runtime_archive_digest')
            path = private_path(value['path'], private_home)
            if path.stat().st_size <= 0:
                raise RuntimePreflightError('runtime_archive_missing')
            if str(path) in assets and assets[str(path)]['sha256'] != value['sha256']:
                raise RuntimePreflightError('conflicting_runtime_archive_digest')
            assets[str(path)] = {'sha256': value['sha256'], 'bytes': path.stat().st_size}
    return {'services': len(names), 'unique_images': len(selected),
            'unique_assets': len(assets),
            'asset_bytes': sum(a['bytes'] for a in assets.values()),
            'integrity_verified': False}


def live_inventory(root, *, private_home=PRIVATE_HOME, run=subprocess.run):
    root = private_path(root, private_home, directory=True)
    if root.parent != Path(private_home) or not re.fullmatch(r'selfhost-\d{8}T\d{6}Z', root.name):
        raise RuntimePreflightError('unexpected_scheduled_selfhost_root')
    compose = private_path(root / 'compose.private.json', private_home)

    def command(args):
        try:
            result = run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         timeout=120, check=False)
            if result.returncode:
                raise RuntimePreflightError('runtime_preflight_inventory_unavailable')
            return result.stdout
        except (OSError, subprocess.SubprocessError):
            raise RuntimePreflightError('runtime_preflight_inventory_unavailable') from None

    identifiers = command(['docker', 'compose', '-f', str(compose), '--profile',
                           'manual-worker', 'ps', '-aq']).decode().split()
    if len(identifiers) != 7 or any(not re.fullmatch(r'[a-f0-9]{12,64}', value) for value in identifiers):
        raise RuntimePreflightError('invalid_selfhost_runtime_inventory')
    try:
        rows = json.loads(command(['docker', 'inspect', *identifiers]))
    except (ValueError, UnicodeError):
        raise RuntimePreflightError('invalid_selfhost_runtime_inventory') from None
    return rows


def preflight(root, index, *, private_home=PRIVATE_HOME, run=subprocess.run):
    return coverage(live_inventory(root, private_home=private_home, run=run),
                    read_index(index, private_home), private_home=private_home)
