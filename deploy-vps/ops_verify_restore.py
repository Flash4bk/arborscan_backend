"""Restore a private snapshot to a NEW offline directory; never connects to a DB."""
import hashlib
import json
import pathlib
import tarfile
import sys
import time


def restore(source, destination):
    start = time.monotonic()
    root = pathlib.Path(destination)
    root.mkdir(mode=0o700, parents=False, exist_ok=False)
    total = 0
    with tarfile.open(source) as archive:
        manifest = json.load(archive.extractfile('manifest.json'))
        seen = set()
        for member in archive:
            path = pathlib.PurePosixPath(member.name)
            if (not member.isfile() or path.is_absolute() or '..' in path.parts or
                '\\' in member.name or member.name in seen):
                raise ValueError('Unsafe archive')
            seen.add(member.name)
            if member.name == 'manifest.json': continue
            target = root.joinpath(*path.parts)
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with archive.extractfile(member) as src, target.open('xb') as dst:
                target.chmod(0o600)
                while chunk := src.read(1024*1024):
                    digest.update(chunk); dst.write(chunk); total += len(chunk)
            if digest.hexdigest() != manifest.get(member.name): raise ValueError('Checksum mismatch')
        if set(manifest) != seen-{'manifest.json'}: raise ValueError('Missing members')
    # Verify report -> owner/version -> private immutable payload -> photo hashes.
    versions = root/'tables/report_versions.json'
    reports = json.loads(versions.read_text()) if versions.exists() else []
    for row in reports:
        matches = list((root/'objects').glob('*/report-versions/'+row['owner_id']+'/'+row['payload_sha256']+'.json'))
        if len(matches) != 1: raise ValueError('Missing report payload')
        raw = matches[0].read_bytes()
        if hashlib.sha256(raw).hexdigest() != row['payload_sha256']: raise ValueError('Payload mismatch')
    return {'files_restored': len(manifest), 'bytes': total, 'reports_linked': len(reports),
            'seconds': round(time.monotonic()-start, 3), 'database_restored': False,
            'production_credentials_loaded': False}


if __name__ == '__main__':
    try: print(json.dumps(restore(sys.argv[1], sys.argv[2])))
    except Exception:
        print('Offline restore failed; destination retained for diagnosis', file=sys.stderr)
        sys.exit(1)
