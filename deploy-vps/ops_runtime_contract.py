"""Additive runtime contract for cloud history and local self-host backups.

Readers call this only after checking the outer SHA256SUMS. A self-host marker
alone cannot authorize a seven-service replacement or deletion of old copies.
"""
import json
import hashlib
from pathlib import Path
import re

LEGACY_V4 = {'/arborscan-api-v4', '/arborscan-quality-worker'}
LEGACY_AUTH = '/arborscan-api'
SELFHOST_ROLES = {'db', 'rest', 'storage', 'gateway', 'v3', 'api', 'worker'}
IMAGE = re.compile(r'sha256:[a-f0-9]{64}')
TRANSITION_FILE = 'SELFHOST_TRANSITION_PROTECTION.private.json'


def selected_runtime(folder, runtime, covered):
    """Return the exact services/images backed by the covered container inventory."""
    folder = Path(folder)
    rows = json.loads((folder / 'containers.private.json').read_bytes())
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError('invalid_runtime_inventory')
    services = runtime.get('services', sorted(LEGACY_V4))
    if (runtime.get('format') != 1 or not isinstance(services, list) or
            any(not isinstance(n, str) for n in services) or len(services) != len(set(services))):
        raise ValueError('invalid_runtime_services')
    selfhost = runtime.get('selfhost') is True
    marker_exists = (folder / 'SELFHOST_BACKUP.json').exists()
    if selfhost:
        if 'SELFHOST_BACKUP.json' not in covered or not marker_exists:
            raise ValueError('selfhost_marker_not_covered')
        marker = json.loads((folder / 'SELFHOST_BACKUP.json').read_bytes())
        if (marker.get('format') != 1 or marker.get('kind') != 'local_selfhost_backup' or
                set(marker.get('services', [])) != SELFHOST_ROLES):
            raise ValueError('invalid_selfhost_marker')
        inventory = {}
        projects = set()
        for row in rows:
            labels = row.get('Config', {}).get('Labels') or {}
            role = labels.get('com.docker.compose.service')
            project = labels.get('com.docker.compose.project')
            if role not in SELFHOST_ROLES or role in inventory or not isinstance(project, str) or not project:
                raise ValueError('invalid_selfhost_runtime_inventory')
            inventory[role] = row
            projects.add(project)
        if set(inventory) != SELFHOST_ROLES or len(projects) != 1 or len(rows) != 7:
            raise ValueError('invalid_selfhost_runtime_inventory')
        selected = {row.get('Name'): row.get('Image') for row in inventory.values()}
        if set(selected) != set(services) or len(selected) != 7:
            raise ValueError('selfhost_runtime_services_mismatch')
    else:
        if marker_exists or runtime.get('selfhost') not in (None, False):
            raise ValueError('selfhost_contract_required')
        if set(services) not in (LEGACY_V4, LEGACY_V4 | {LEGACY_AUTH}):
            raise ValueError('invalid_runtime_services')
        selected_rows = [row for row in rows if row.get('Name') in services]
        selected = {row['Name']: row.get('Image') for row in selected_rows}
        if set(selected) != set(services) or len(selected_rows) != len(services):
            raise ValueError('runtime_image_inventory_missing')
    if any(not isinstance(image, str) or not IMAGE.fullmatch(image) for image in selected.values()):
        raise ValueError('invalid_runtime_image_id')
    if set(selected.values()) != set(runtime.get('images', {})):
        raise ValueError('runtime_index_image_mismatch')
    return {'services': sorted(services), 'images': set(selected.values()),
            'selfhost': selfhost, 'auth_present': selfhost or LEGACY_AUTH in selected}


def selfhost_inventory(rows):
    """Construct the seven-role identity without inventing stable container names."""
    if not isinstance(rows, list) or len(rows) != 7:
        raise ValueError('invalid_selfhost_runtime_inventory')
    names = [row.get('Name') for row in rows]
    roles = [((row.get('Config') or {}).get('Labels') or {}).get('com.docker.compose.service') for row in rows]
    projects = {((row.get('Config') or {}).get('Labels') or {}).get('com.docker.compose.project') for row in rows}
    if (len(set(names)) != 7 or any(not isinstance(n, str) or not n.startswith('/') for n in names)
            or set(roles) != SELFHOST_ROLES or len(set(roles)) != 7 or len(projects) != 1
            or not all(isinstance(p, str) and p for p in projects)):
        raise ValueError('invalid_selfhost_runtime_inventory')
    images = {row.get('Image') for row in rows}
    if any(not isinstance(image, str) or not IMAGE.fullmatch(image) for image in images):
        raise ValueError('invalid_runtime_image_id')
    return sorted(names), images


def transition_protection(root):
    """Exact old manifests stay outside the new rolling14 pool, never inferred."""
    path = Path(root) / TRANSITION_FILE
    if not path.exists():
        return {}
    if path.is_symlink():
        raise ValueError('unsafe_transition_protection')
    value = json.loads(path.read_bytes())
    if (value.get('format') != 1 or value.get('kind') != 'selfhost_backup_transition'
            or not isinstance(value.get('sets'), list)):
        raise ValueError('invalid_transition_protection')
    protected = {}
    for row in value['sets']:
        if (not isinstance(row, dict) or not re.fullmatch(r'\d{8}T\d{6}Z', row.get('name', ''))
                or not re.fullmatch(r'[a-f0-9]{64}', row.get('manifest_sha256', ''))
                or row['name'] in protected):
            raise ValueError('invalid_transition_protection')
        protected[row['name']] = row['manifest_sha256']
        manifest = Path(root) / row['name'] / 'SHA256SUMS'
        if (not manifest.is_file() or manifest.is_symlink() or manifest.parent.is_symlink()
                or hashlib.sha256(manifest.read_bytes()).hexdigest() != row['manifest_sha256']):
            raise ValueError('transition_protection_manifest_mismatch')
    return protected
