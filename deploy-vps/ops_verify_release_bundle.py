"""Check that a backup's actual API/worker images have a complete off-site release.

This supplements ops_verify_offsite.py: a verified data dump alone does not prove
the runtime images are recoverable. Never prints Docker environments or secrets.
Public bootstrap dependencies and TLS reprovisioning remain separate.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re


def contained(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name:
        raise ValueError('Invalid release path')
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError('Release path outside off-site storage')
    return path


def verify(backup, root):
    backup, root = Path(backup).resolve(strict=True), Path(root).resolve(strict=True)
    if not (backup / 'COMPLETE').is_file():
        raise ValueError('Incomplete data backup')
    rows = json.loads((backup / 'containers.private.json').read_text(encoding='utf-8'))
    required = {'/arborscan-api-v4', '/arborscan-quality-worker'}
    selected = {row['Name']: row['Image'] for row in rows if row['Name'] in required}
    if selected.keys() != required:
        raise ValueError('Missing API/worker image inventory')
    index = json.loads((root / 'RELEASE_INDEX.json').read_text(encoding='utf-8'))
    if index.get('version') != 1 or index.get('complete') is not True:
        raise ValueError('Incomplete release index')
    checked = set()
    for image_id in set(selected.values()):
        if not re.fullmatch(r'sha256:[0-9a-f]{64}', image_id):
            raise ValueError('Invalid runtime image ID')
        entry = index['images'].get(image_id)
        if not entry:
            raise ValueError('Runtime image has no external release archive')
        manifest_path = contained(root, entry['release'])
        if manifest_path.name != 'release.json':
            raise ValueError('Invalid release manifest name')
        release = manifest_path.parent
        if not (release / 'COMPLETE').is_file():
            raise ValueError('Incomplete external release')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('status') != 'verified_complete':
            raise ValueError('Release is not verified')
        if entry['artifact'] == 'base_images':
            pinned = json.loads((release / 'pinned-images.json').read_text(encoding='utf-8'))
            if pinned['images']['api'] != image_id:
                raise ValueError('Base image inventory mismatch')
        elif entry['artifact'] == 'base_images+overlay':
            if manifest['overlay']['result_image'] != image_id:
                raise ValueError('Overlay image inventory mismatch')
        else:
            raise ValueError('Unknown release artifact mapping')
        if release not in checked:
            names = set()
            verified_digests = {}
            for line in (release / 'SHA256SUMS').read_text(encoding='utf-8').splitlines():
                expected, name = line.split('  ', 1)
                if name in names or not re.fullmatch(r'[0-9a-f]{64}', expected):
                    raise ValueError('Invalid release checksum manifest')
                names.add(name)
                path = contained(release, name)
                with path.open('rb') as stream:
                    actual = hashlib.file_digest(stream, 'sha256').hexdigest()
                if actual != expected:
                    raise ValueError('Damaged or incomplete release artifact')
                verified_digests[name] = actual
            required_files = {'release.json', 'pinned-images.json'}
            for field in ('base_images', 'overlay', 'source'):
                required_files.add(manifest[field]['file'])
            required_files.add(manifest['overlay']['file'] + '.json')
            if not required_files.issubset(names):
                raise ValueError('Release checksum manifest omits required artifacts')
            for field in ('base_images', 'overlay', 'source'):
                item = manifest[field]
                if verified_digests[item['file']] != item['sha256']:
                    raise ValueError('Release metadata checksum mismatch')
            delta = json.loads((release / (manifest['overlay']['file'] + '.json')).read_text(encoding='utf-8'))
            if (delta.get('version') != 1 or
                    delta.get('base_sha256') != manifest['base_images']['sha256'] or
                    delta.get('delta_sha256') != manifest['overlay']['sha256']):
                raise ValueError('Overlay does not match release base')
            checked.add(release)
    return {'runtime_images_mapped': len(set(selected.values())),
            'release_archives_verified': len(checked),
            'scope': 'API/worker archive availability and integrity; verify data separately',
            'new_vm_or_public_https_verified': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('backup', type=Path)
    parser.add_argument('offsite_root', type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.backup, args.offsite_root)))
    except (OSError, ValueError, KeyError, TypeError):
        raise SystemExit('Release bundle verification failed; inspect private manifests locally')
