"""Small, checksum-bound Docker save overlays; archives remain private, outside Git.

The full base archive is required. apply reconstructs a Docker-loadable tar; it
does not use an installed Docker image or contact a registry.
"""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@contextlib.contextmanager
def reader(path):
    if str(path).endswith('.zst'):
        process = subprocess.Popen(['zstd', '-q', '-d', '-c', str(path)], stdout=subprocess.PIPE)
        try:
            with tarfile.open(fileobj=process.stdout, mode='r|') as archive:
                yield archive
        finally:
            process.stdout.close()
            if process.wait() != 0:
                raise ValueError('Archive decompression failed')
    else:
        with tarfile.open(path, 'r:*') as archive:
            yield archive


def safe(member):
    p = PurePosixPath(member.name)
    if p.is_absolute() or '..' in p.parts or '\\' in member.name or not (member.isfile() or member.isdir()):
        raise ValueError('Unsafe image archive')


def create(base, target, output):
    known = {}
    with reader(base) as archive:
        for member in archive:
            safe(member)
            if member.isfile():
                known[member.name] = hashlib.file_digest(archive.extractfile(member), 'sha256').hexdigest()
    changed = {}
    with reader(target) as archive, tarfile.open(output, 'x:gz') as delta:
        for member in archive:
            safe(member)
            if not member.isfile():
                continue
            data = archive.extractfile(member).read()
            sha = hashlib.sha256(data).hexdigest()
            if known.get(member.name) != sha:
                import io
                delta.addfile(member, io.BytesIO(data))
                changed[member.name] = sha
    manifest = {'version': 1, 'base_sha256': digest(base), 'delta_sha256': digest(output), 'changed': changed}
    Path(str(output)+'.json').write_text(json.dumps(manifest, indent=2))
    return {'changed_members': len(changed), 'delta_bytes': Path(output).stat().st_size}


def apply(base, delta, output):
    manifest = json.loads(Path(str(delta)+'.json').read_text())
    if manifest['version'] != 1 or digest(base) != manifest['base_sha256'] or digest(delta) != manifest['delta_sha256']:
        raise ValueError('Image archive checksum mismatch')
    seen = set()
    with tarfile.open(output, 'x') as restored:
        with reader(base) as archive:
            for member in archive:
                safe(member)
                if member.name not in manifest['changed']:
                    restored.addfile(member, archive.extractfile(member) if member.isfile() else None)
        with reader(delta) as archive:
            for member in archive:
                safe(member)
                if not member.isfile() or member.name in seen:
                    raise ValueError('Invalid delta member')
                data = archive.extractfile(member).read()
                if hashlib.sha256(data).hexdigest() != manifest['changed'].get(member.name):
                    raise ValueError('Delta member checksum mismatch')
                import io
                restored.addfile(member, io.BytesIO(data))
                seen.add(member.name)
    if seen != set(manifest['changed']):
        raise ValueError('Incomplete delta')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['create', 'apply'])
    parser.add_argument('base', type=Path)
    parser.add_argument('other', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(json.dumps((create if args.mode == 'create' else apply)(args.base, args.other, args.output)))
