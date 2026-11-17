"""Small, checksum-bound Docker save overlays; archives remain private, outside Git.

The full base archive is required. apply reconstructs a Docker-loadable tar; it
does not use an installed Docker image or contact a registry.
"""
import argparse
import base64
import contextlib
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import re
import shutil
import tempfile


MAX_DELTA_MANIFEST = 4 * 1024**2
MAX_DELTA_MEMBERS = 4096
MAX_EXTENDED_HEADER = 64 * 1024
MAX_EXTENDED_BYTES = 1024**2
MAX_DELTA_NAME = 1024
RESERVED_FREE_BYTES = 4 * 1024**3


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


def _delta_name(name):
    if (not isinstance(name, str) or not name or len(name.encode('utf-8')) > MAX_DELTA_NAME
            or any(ord(char) < 32 for char in name) or ':' in name or '\\' in name
            or any(part in ('', '.', '..') for part in name.split('/'))
            or PurePosixPath(name).is_absolute()):
        raise ValueError('Unsafe member delta path')


def _check_pax(payload):
    # Do this before TarInfo._proc_pax can interpret a size/sparse override or
    # allocate a payload based on it. Ordinary small mtime/path attributes stay
    # compatible with Docker-save headers; image layer bytes remain opaque.
    offset, count = 0, 0
    while offset < len(payload):
        split = payload.find(b' ', offset, offset + 12)
        if split < 0 or not payload[offset:split].isdigit():
            raise ValueError('Invalid PAX delta header')
        length = int(payload[offset:split])
        end = offset + length
        if length <= split - offset + 3 or end > len(payload) or payload[end-1:end] != b'\n':
            raise ValueError('Invalid PAX delta header')
        key, separator, value = payload[split+1:end-1].partition(b'=')
        if (not separator or len(key) > 256 or key == b'size'
                or key.startswith(b'GNU.sparse') or key == b'SCHILY.realsize'):
            raise ValueError('Unsupported PAX delta header override')
        if key in (b'path', b'linkpath'):
            _delta_name(value.decode('utf-8'))
        count += 1
        if count > 512:
            raise ValueError('PAX delta header budget exceeded')
        offset = end


@contextlib.contextmanager
def checked_member_delta(path, directory, max_changed_bytes):
    """Inflate once into a bounded private temporary, before tarfile scans it.

    Inspect each 512-byte raw header before reading its payload. In particular,
    a giant PAX/longname declaration cannot make tarfile allocate it. The raw
    archive has independent payload, metadata, header-count and padding caps.
    Subsequent member hashing/copying uses the uncompressed seekable temporary,
    avoiding repeated gzip inflation on backwards seeks. No extraction occurs.
    """
    if type(max_changed_bytes) is not int or not 0 <= max_changed_bytes <= 512 * 1024**2:
        raise ValueError('Invalid member delta payload budget')
    ceiling = (max_changed_bytes + MAX_DELTA_MANIFEST + MAX_EXTENDED_BYTES
               + MAX_DELTA_MEMBERS * 2 * 1024 + tarfile.RECORDSIZE)
    if shutil.disk_usage(directory).free < ceiling + RESERVED_FREE_BYTES:
        raise ValueError('Insufficient member delta temporary space')
    with gzip.open(path, 'rb') as source, tempfile.TemporaryFile(dir=directory) as checked:
        inflated = headers = members = payload_bytes = extended_bytes = 0

        def read(count):
            nonlocal inflated
            if count > 64 * 1024:
                raise ValueError('Unbounded member delta read refused')
            raw = source.read(count)
            inflated += len(raw)
            if inflated > ceiling:
                raise ValueError('Member delta inflate budget exceeded')
            if shutil.disk_usage(directory).free < RESERVED_FREE_BYTES + len(raw):
                raise ValueError('Insufficient member delta temporary space')
            checked.write(raw)
            return raw

        while True:
            raw = read(tarfile.BLOCKSIZE)
            if len(raw) != tarfile.BLOCKSIZE:
                raise ValueError('Truncated member delta header')
            if raw == bytes(tarfile.BLOCKSIZE):
                # Standard tar padding is bounded too, including concatenated
                # gzip members; silently ignoring a trailing inflate bomb is
                # not a successful integrity check.
                padding = len(raw)
                while tail := read(64 * 1024):
                    padding += len(tail)
                    if padding > tarfile.RECORDSIZE or any(tail):
                        raise ValueError('Member delta inflate padding budget exceeded')
                break
            headers += 1
            if headers > MAX_DELTA_MEMBERS * 2:
                raise ValueError('Member delta header budget exceeded')
            member = tarfile.TarInfo.frombuf(raw, 'utf-8', 'surrogateescape')
            if member.size < 0:
                raise ValueError('Invalid member delta header size')
            extended = member.type in (tarfile.XHDTYPE, tarfile.XGLTYPE,
                                       tarfile.GNUTYPE_LONGNAME, tarfile.GNUTYPE_LONGLINK)
            if extended:
                extended_bytes += member.size
                if member.size > MAX_EXTENDED_HEADER or extended_bytes > MAX_EXTENDED_BYTES:
                    raise ValueError('Member delta extended header budget exceeded')
            else:
                if member.type not in (tarfile.REGTYPE, tarfile.AREGTYPE):
                    raise ValueError('Non-file member delta header')
                _delta_name(member.name)
                members += 1
                if members > MAX_DELTA_MEMBERS:
                    raise ValueError('Member delta member budget exceeded')
                if member.name == 'DELTA_MANIFEST.json':
                    if member.size > MAX_DELTA_MANIFEST:
                        raise ValueError('Member delta manifest budget exceeded')
                else:
                    payload_bytes += member.size
                    if payload_bytes > max_changed_bytes:
                        raise ValueError('Member delta payload budget exceeded')
            remaining = (member.size + 511) // 512 * 512
            metadata = bytearray()
            while remaining:
                chunk = read(min(64 * 1024, remaining))
                if not chunk:
                    raise ValueError('Truncated member delta payload')
                if extended:
                    metadata.extend(chunk)
                remaining -= len(chunk)
            if extended:
                content = bytes(metadata[:member.size])
                if member.type in (tarfile.XHDTYPE, tarfile.XGLTYPE):
                    _check_pax(content)
                else:
                    _delta_name(content.rstrip(b'\0').decode('utf-8'))
        checked.seek(0)
        with tarfile.open(fileobj=checked, mode='r:') as archive:
            yield archive


def _write_raw_members(original, changes, delta_members, info, output, max_changed_bytes,
                       expected_archive_sha256):
    """Validate native framing, then write an opaque archive; never extract it."""
    if (not isinstance(expected_archive_sha256, str)
            or not re.fullmatch(r'[a-f0-9]{64}', expected_archive_sha256)):
        raise ValueError('Raw archive expected checksum required')
    trailer = info.get('trailing_zero_bytes')
    if (type(trailer) is not int or not 1024 <= trailer <= tarfile.RECORDSIZE
            or trailer % tarfile.BLOCKSIZE or type(info.get('target_bytes')) is not int):
        raise ValueError('Invalid raw member framing metadata or EOF')
    seen, files, changed, prepared = set(), set(), set(), []
    target_bytes, changed_bytes = trailer, 0
    required = {'name', 'bytes', 'sha256', 'base', 'type', 'header_b64'}
    for row in info['members']:
        if (not isinstance(row, dict) or set(row) != required
                or not isinstance(row['name'], str) or type(row['bytes']) is not int
                or row['bytes'] < 0 or type(row['base']) is not bool
                or not isinstance(row['sha256'], str)
                or not re.fullmatch(r'[a-f0-9]{64}', row['sha256'])):
            raise ValueError('Invalid raw member target metadata')
        _delta_name(row['name'])
        if row['name'] in seen:
            raise ValueError('Duplicate raw member target')
        seen.add(row['name'])
        encoded = row['header_b64']
        if not isinstance(encoded, str) or len(encoded) != 684:
            raise ValueError('Invalid raw member header length')
        try:
            header = base64.b64decode(encoded, validate=True)
            if len(header) != tarfile.BLOCKSIZE:
                raise ValueError('Invalid raw member header length')
            member = tarfile.TarInfo.frombuf(header, 'utf-8', 'strict')
        except (ValueError, UnicodeError, tarfile.HeaderError):
            raise ValueError('Invalid raw member header') from None
        if (member.name != row['name'] or member.size != row['bytes'] or member.linkname
                or member.type not in (tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE)):
            raise ValueError('Invalid raw member header name size or type')
        source, payload = None, None
        if row['type'] == 'directory' and member.isdir():
            if row['bytes'] != 0 or row['base'] or row['sha256'] != hashlib.sha256(b'').hexdigest():
                raise ValueError('Invalid raw directory size or payload')
        elif row['type'] == 'file' and member.isfile():
            files.add(row['name'])
            source = original if row['base'] else changes
            try:
                payload = source.getmember(row['name'])
            except KeyError:
                raise ValueError('Missing raw member payload') from None
            safe(payload)
            if payload.type not in (tarfile.REGTYPE, tarfile.AREGTYPE) or payload.size != row['bytes']:
                raise ValueError('Raw member payload size mismatch')
            if not row['base']:
                changed.add(row['name'])
                changed_bytes += row['bytes']
        else:
            raise ValueError('Invalid raw member header type')
        target_bytes += tarfile.BLOCKSIZE + (row['bytes'] + 511) // 512 * 512
        prepared.append((row, header, source, payload))
    for name in seen:
        if any(parent.as_posix() in files for parent in PurePosixPath(name).parents
               if parent != PurePosixPath('.')):
            raise ValueError('Raw file directory collision')
    if ({m.name for m in delta_members} != changed | {'DELTA_MANIFEST.json'}
            or changed_bytes != info['changed_bytes'] or changed_bytes > max_changed_bytes):
        raise ValueError('Raw member delta entries or budget mismatch')
    if target_bytes != info['target_bytes']:
        raise ValueError('Raw member target size mismatch')
    directory = Path(output).absolute().parent
    if shutil.disk_usage(directory).free < target_bytes + RESERVED_FREE_BYTES:
        raise ValueError('Insufficient raw archive output space')
    written = 0
    with Path(output).open('xb') as restored:
        def write(raw):
            nonlocal written
            if written + len(raw) > target_bytes:
                raise ValueError('Raw member output size exceeded')
            if shutil.disk_usage(directory).free < RESERVED_FREE_BYTES + len(raw):
                raise ValueError('Insufficient raw archive output space')
            restored.write(raw)
            written += len(raw)
        for row, header, source, payload in prepared:
            write(header)
            if source is not None:
                checksum = hashlib.sha256()
                remaining = row['bytes']
                with source.extractfile(payload) as stream:
                    while remaining:
                        raw = stream.read(min(1024**2, remaining))
                        if not raw:
                            raise ValueError('Truncated raw member payload')
                        checksum.update(raw)
                        write(raw)
                        remaining -= len(raw)
                if checksum.hexdigest() != row['sha256']:
                    raise ValueError('Member delta checksum mismatch')
                write(bytes(-row['bytes'] % tarfile.BLOCKSIZE))
        write(bytes(trailer))
        restored.flush()
        os.fsync(restored.fileno())
    return seen, changed_bytes


def _member_delta_result(output, image_id, seen, total_changed, expected_archive_sha256):
    from ops_selfhost_backup import verify_saved_image
    verify_saved_image(Path(output).resolve(), image_id)
    sha = digest(output)
    if expected_archive_sha256 is not None and sha != expected_archive_sha256:
        raise ValueError('Reconstructed archive checksum mismatch')
    return {'image_id': image_id, 'archive_sha256': sha,
            'archive_bytes': Path(output).stat().st_size, 'target_members': len(seen),
            'changed_payload_bytes': total_changed, 'image_closure_verified': True,
            'container_started': False, 'production_configuration_changed': False}


def apply_member_delta(base, delta, output, *, image_id, base_sha256,
                       expected_archive_sha256=None, max_changed_bytes=512 * 1024**2):
    """Read the original file-only format; new native framing is opt-in."""
    return _apply_member_delta(base, delta, output, image_id=image_id,
        base_sha256=base_sha256, expected_archive_sha256=expected_archive_sha256,
        max_changed_bytes=max_changed_bytes, format_version=1)


def apply_raw_member_delta(base, delta, output, *, image_id, base_sha256,
                           expected_archive_sha256=None, max_changed_bytes=512 * 1024**2):
    """Explicit format 2: exact native headers/order/zero padding, no extraction."""
    return _apply_member_delta(base, delta, output, image_id=image_id,
        base_sha256=base_sha256, expected_archive_sha256=expected_archive_sha256,
        max_changed_bytes=max_changed_bytes, format_version=2)


def _apply_member_delta(base, delta, output, *, image_id, base_sha256,
                        expected_archive_sha256, max_changed_bytes, format_version):
    """Reconstruct the exact target member set from a bounded OCI stream delta.

    A partial is not a success receipt. Outer base/member hashes and the exact
    pinned saved-image closure must pass before this returns. No Docker starts,
    registry requests or production configuration changes are performed.
    """
    if (not re.fullmatch(r'sha256:[a-f0-9]{64}', image_id)
            or not re.fullmatch(r'[a-f0-9]{64}', base_sha256)
            or digest(base) != base_sha256):
        raise ValueError('Invalid member delta base or image identity')
    with checked_member_delta(delta, Path(output).absolute().parent, max_changed_bytes) as changes, tarfile.open(base) as original:
        delta_members = changes.getmembers()
        if len(delta_members) > MAX_DELTA_MEMBERS or len({m.name for m in delta_members}) != len(delta_members):
            raise ValueError('Duplicate member delta entry')
        for member in delta_members:
            safe(member)
            _delta_name(member.name)
            if not member.isfile():
                raise ValueError('Non-file member delta entry')
        metadata = changes.getmember('DELTA_MANIFEST.json')
        if metadata.size > MAX_DELTA_MANIFEST:
            raise ValueError('Oversize member delta metadata')
        info = json.load(changes.extractfile(metadata))
        if (info.get('format') != format_version
                or (format_version == 2 and type(info.get('format')) is not int)
                or info.get('kind') != 'docker_save_verified_member_delta'
                or info.get('image_id') != image_id or info.get('base_sha256') != base_sha256
                or type(info.get('changed_bytes')) is not int
                or not 0 <= info['changed_bytes'] <= max_changed_bytes
                or not isinstance(info.get('members'), list)
                or len(info['members']) > MAX_DELTA_MEMBERS):
            raise ValueError('Invalid member delta metadata')
        if info['format'] == 2:
            seen, total_changed = _write_raw_members(original, changes, delta_members, info, output,
                                                     max_changed_bytes, expected_archive_sha256)
            result = _member_delta_result(output, image_id, seen, total_changed, expected_archive_sha256)
            result.update(delta_format=2, raw_headers_preserved=True)
            return result
        seen, changed, total_changed = set(), set(), 0
        with tarfile.open(output, 'x') as restored:
            for row in info['members']:
                if (not isinstance(row, dict) or not isinstance(row.get('name'), str)
                        or type(row.get('base')) is not bool or type(row.get('bytes')) is not int
                        or row['bytes'] < 0 or not isinstance(row.get('sha256'), str)
                        or not re.fullmatch(r'[a-f0-9]{64}', row['sha256'])
                        or row['name'] in seen):
                    raise ValueError('Invalid member delta target')
                seen.add(row['name'])
                _delta_name(row['name'])
                source = original if row['base'] else changes
                member = source.getmember(row['name'])
                safe(member)
                if not member.isfile() or member.size != row['bytes']:
                    raise ValueError('Member delta size mismatch')
                with source.extractfile(member) as stream:
                    if hashlib.file_digest(stream, 'sha256').hexdigest() != row['sha256']:
                        raise ValueError('Member delta checksum mismatch')
                if not row['base']:
                    changed.add(member.name)
                    total_changed += member.size
                    if total_changed > max_changed_bytes:
                        raise ValueError('Member delta payload budget exceeded')
                member.uid = member.gid = 0
                member.uname = member.gname = ''
                member.pax_headers = {}
                restored.addfile(member, source.extractfile(member))
        if ({m.name for m in delta_members} != changed | {'DELTA_MANIFEST.json'}
                or total_changed != info['changed_bytes']):
            raise ValueError('Member delta entries or budget mismatch')
    return _member_delta_result(output, image_id, seen, total_changed, expected_archive_sha256)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['create', 'apply', 'apply-member', 'apply-raw-member'])
    parser.add_argument('base', type=Path)
    parser.add_argument('other', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--image-id')
    parser.add_argument('--base-sha256')
    parser.add_argument('--expected-archive-sha256')
    args = parser.parse_args()
    if args.mode in ('apply-member', 'apply-raw-member'):
        function = apply_raw_member_delta if args.mode == 'apply-raw-member' else apply_member_delta
        result = function(args.base, args.other, args.output,
            image_id=args.image_id, base_sha256=args.base_sha256,
            expected_archive_sha256=args.expected_archive_sha256)
    else:
        result = (create if args.mode == 'create' else apply)(args.base, args.other, args.output)
    print(json.dumps(result))
