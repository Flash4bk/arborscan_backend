"""Verify/transport an explicitly reviewed exact-byte failed-stage quarantine.

Private receipts, absolute source paths, account data and secrets are inputs,
never source constants. This does not create a fresh backup or restore SQL.
Default CLI verifies only. Receiving requires explicit absent target paths,
an inactive service, an existing backup lock and the separately pinned proof.
"""
import argparse
import base64
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tarfile
import shutil

SHA = re.compile(r'[a-f0-9]{64}')
MAX_JSON = 32 * 1024**2


def digest(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def private_file(path, root):
    path, root = Path(path), Path(root)
    if not path.is_absolute() or not path.is_relative_to(root) or '..' in path.parts:
        raise ValueError('unsafe_quarantine_asset')
    for item in [path, *path.parents]:
        if item.is_symlink() or (item.exists() and getattr(item.lstat(), 'st_file_attributes', 0) & 1024):
            raise ValueError('unsafe_quarantine_asset')
    if not path.is_file():
        raise ValueError('missing_quarantine_asset')
    return path


def decode_json(raw, *, compressed=False):
    if len(raw) > MAX_JSON:
        raise ValueError('oversize_quarantine_metadata')
    if compressed:
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as source:
            raw = source.read(MAX_JSON + 1)
        if len(raw) > MAX_JSON:
            raise ValueError('oversize_quarantine_metadata')
    return json.loads(raw)


def relative_name(name, attempts):
    if not isinstance(name, str):
        raise ValueError('unsafe_quarantine_name')
    path = PurePosixPath(name)
    if (path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name
            or not path.parts or path.parts[0] not in attempts
            or any(part in ('', '.') for part in name.split('/'))):
        raise ValueError('unsafe_quarantine_name')
    return path


def read_blobs(path, missing):
    blobs, receipt = {}, None
    with tarfile.open(path, 'r:gz') as archive:
        for member in archive:
            if not member.isfile() or member.size > 16 * 1024**2 or member.name in blobs:
                raise ValueError('invalid_quarantine_blob')
            if member.name == 'COPY_RECEIPT.json':
                if receipt is not None or member.size > MAX_JSON:
                    raise ValueError('duplicate_quarantine_receipt')
                receipt = json.load(archive.extractfile(member))
                continue
            if member.name not in missing or not SHA.fullmatch(member.name):
                raise ValueError('unrequested_quarantine_blob')
            raw = archive.extractfile(member).read()
            if (len(raw) != missing[member.name]['size']
                    or hashlib.sha256(raw).hexdigest() != member.name):
                raise ValueError('quarantine_blob_checksum')
            blobs[member.name] = raw
            if sum(map(len, blobs.values())) > 32 * 1024**2:
                raise ValueError('quarantine_blob_budget')
    if (set(blobs) != set(missing) or not isinstance(receipt, dict)
            or receipt.get('source_before_and_after_cas') is not True):
        raise ValueError('incomplete_quarantine_blobs')
    return blobs


def segment_chunks(row, local, blobs):
    sha, size = row.get('sha256'), row.get('size', row.get('bytes'))
    if not isinstance(sha, str) or not SHA.fullmatch(sha) or type(size) is not int or size < 0:
        raise ValueError('invalid_quarantine_segment')
    result, count = hashlib.sha256(), 0
    if 'inline' in row or sha in blobs:
        raw = base64.b64decode(row['inline'], validate=True) if 'inline' in row else blobs[sha]
        result.update(raw); count = len(raw)
        if count != size or result.hexdigest() != sha:
            raise ValueError('quarantine_segment_checksum')
        for offset in range(0, len(raw), 1024**2):
            yield raw[offset:offset + 1024**2]
    elif sha in local:
        item = local[sha]
        if item['size'] != size:
            raise ValueError('quarantine_segment_size')
        with item['path'].open('rb') as source:
            source.seek(item['offset'])
            while count < size:
                raw = source.read(min(1024**2, size - count))
                if not raw:
                    raise ValueError('truncated_quarantine_base')
                count += len(raw); result.update(raw)
                yield raw
        if result.hexdigest() != sha:
            raise ValueError('quarantine_segment_checksum')
    else:
        raise ValueError('missing_quarantine_segment')


def file_parts(row, tars):
    if row['relative'].endswith('.tar'):
        return tars[row['sha256']]['parts']
    return row.get('blocks') or [row]


def file_chunks(row, tars, local, blobs):
    total, whole = 0, hashlib.sha256()
    for part in file_parts(row, tars):
        if 'offset' in part and part['offset'] != total:
            raise ValueError('quarantine_reconstruction_gap')
        for raw in segment_chunks(part, local, blobs):
            total += len(raw); whole.update(raw)
            yield raw
    if total != row['bytes'] or whole.hexdigest() != row['sha256']:
        raise ValueError('quarantine_file_checksum')


def load_bundle(root, proof_sha256):
    root = Path(root).absolute()
    if not SHA.fullmatch(proof_sha256):
        raise ValueError('exact_quarantine_proof_required')
    proof_path = private_file(root / 'QUARANTINE_RECONSTRUCTION.private.json', root)
    proof_raw = proof_path.read_bytes()
    if (hashlib.sha256(proof_raw).hexdigest() != proof_sha256
            or private_file(root / 'QUARANTINE_COMPLETE', root).read_text().strip() != proof_sha256):
        raise ValueError('quarantine_proof_checksum')
    proof = decode_json(proof_raw)
    asset_names = {'quarantine-blobs.private.tar.gz': 'blob_pack_sha256',
                   'FAILED_LAYOUT.private.json.gz': 'layout_sha256',
                   'TAR_AND_NATIVE_LOCAL_REUSE.private.json': 'local_reuse_index_sha256',
                   'MISSING_QUARANTINE_BLOBS.private.json': 'missing_blob_index_sha256'}
    for name, key in asset_names.items():
        if digest(private_file(root / name, root)) != proof[key]:
            raise ValueError('quarantine_asset_checksum')
    layout_raw = (root / 'FAILED_LAYOUT.private.json.gz').read_bytes()
    layout = decode_json(layout_raw, compressed=True)
    local = decode_json((root / 'TAR_AND_NATIVE_LOCAL_REUSE.private.json').read_bytes())
    missing = decode_json((root / 'MISSING_QUARANTINE_BLOBS.private.json').read_bytes())
    blobs = read_blobs(root / 'quarantine-blobs.private.tar.gz', missing)
    pinned, states = {}, {}
    for row in local.values():
        sha = row['archive_sha256']
        if not isinstance(sha, str) or not SHA.fullmatch(sha):
            raise ValueError('quarantine_base_identity')
        # Relocatable: private absolute paths in the old receipt are not trusted
        # as locations. Only the hash-named pin inside this bundle is read.
        path = private_file(root / 'pinned-local-bases' / (sha + '.archive'), root)
        if path not in pinned:
            before = path.stat()
            if digest(path) != sha:
                raise ValueError('quarantine_base_checksum')
            pinned[path] = sha
            states[path] = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
                            getattr(before, 'st_birthtime_ns', before.st_ctime_ns))
        if (type(row['offset']) is not int or type(row['size']) is not int
                or row['offset'] < 0 or row['size'] < 0
                or row['offset'] + row['size'] > path.stat().st_size):
            raise ValueError('quarantine_base_bounds')
        row['path'] = path
    if set(pinned.values()) != set(proof['pinned_local_base_sha'].values()):
        raise ValueError('quarantine_base_set')
    parent = PurePosixPath(proof['source_parent'])
    attempts = {PurePosixPath(path).name for path in proof['attempts']}
    if (not parent.is_absolute() or not attempts
            or any(not re.fullmatch(r'attempt-[1-9][0-9]*', value) for value in attempts)
            or any(PurePosixPath(path).parent != parent for path in proof['attempts'])):
        raise ValueError('invalid_quarantine_scope')
    tars = {row['sha256']: row for row in layout['tars']}
    files, verified_tars = {}, set()
    for row in layout['files']:
        relative_name(row['relative'], attempts)
        if row['relative'] in files:
            raise ValueError('duplicate_quarantine_file')
        if row['relative'].endswith('.tar') and row['sha256'] in verified_tars:
            if row['bytes'] != tars[row['sha256']]['bytes']:
                raise ValueError('quarantine_tar_reference')
        else:
            for _ in file_chunks(row, tars, local, blobs):
                pass
            if row['relative'].endswith('.tar'):
                verified_tars.add(row['sha256'])
        files[row['relative']] = {'bytes': row['bytes'], 'sha256': row['sha256']}
    if files != proof['source_file_sha']:
        raise ValueError('quarantine_file_set')
    for row in layout['directories']:
        relative_name(row['relative'], attempts)
    for path, state in states.items():
        after = path.stat()
        if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
                getattr(after, 'st_birthtime_ns', after.st_ctime_ns)) != state:
            raise ValueError('quarantine_base_changed')
    return {'root': root, 'proof_raw': proof_raw, 'proof': proof, 'layout_raw': layout_raw,
            'layout': layout, 'tars': tars, 'local': local, 'blobs': blobs}


class ChunkReader:
    def __init__(self, chunks):
        self.chunks, self.pending = iter(chunks), b''

    def read(self, size=-1):
        if size < 0:
            raise ValueError('bounded_quarantine_read_required')
        result = bytearray()
        while len(result) < size:
            if not self.pending:
                self.pending = next(self.chunks, b'')
                if not self.pending:
                    break
            take = min(size - len(result), len(self.pending))
            result.extend(self.pending[:take]); self.pending = self.pending[take:]
        return bytes(result)


def emit_bundle(bundle, target):
    with tarfile.open(fileobj=target, mode='w|') as archive:
        raw = json.dumps({'proof_raw': base64.b64encode(bundle['proof_raw']).decode(),
                          'layout_gzip': base64.b64encode(bundle['layout_raw']).decode()}).encode()
        member = tarfile.TarInfo('RESTORE_METADATA.private.json')
        member.size, member.mode = len(raw), 0o600
        archive.addfile(member, io.BytesIO(raw))
        for row in bundle['layout']['files']:
            member = tarfile.TarInfo(row['relative'])
            member.size, member.mode = row['bytes'], 0o600
            archive.addfile(member, ChunkReader(file_chunks(row, bundle['tars'], bundle['local'], bundle['blobs'])))


def receive_bundle(stream, *, proof_sha256, target_parent, backup_root, service):
    """Linux-only reviewed rollback; refuse every existing target tree."""
    import fcntl
    root, parent = Path(backup_root), Path(target_parent)
    if (parent.parent != root / '.staging' or parent.resolve() != parent
            or any(p.is_symlink() for p in [parent, *parent.parents])):
        raise ValueError('unsafe_quarantine_target')
    state = subprocess.run(['systemctl', 'show', service, '-p', 'ActiveState', '--value'],
                           capture_output=True, timeout=15, check=True)
    if state.stdout.strip() not in [b'inactive', b'failed']:
        raise ValueError('quarantine_service_active')
    with os.fdopen(os.open(root / 'backup.lock', os.O_RDWR | os.O_NOFOLLOW), 'r+') as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with tarfile.open(fileobj=stream, mode='r|') as archive:
            entries = iter(archive)
            first = next(entries)
            if first.name != 'RESTORE_METADATA.private.json' or not first.isfile() or first.size > MAX_JSON:
                raise ValueError('invalid_quarantine_transport')
            info = json.load(archive.extractfile(first))
            raw, layout_raw = base64.b64decode(info['proof_raw']), base64.b64decode(info['layout_gzip'])
            if hashlib.sha256(raw).hexdigest() != proof_sha256:
                raise ValueError('quarantine_proof_checksum')
            proof = decode_json(raw)
            if proof['source_parent'] != str(parent) or hashlib.sha256(layout_raw).hexdigest() != proof['layout_sha256']:
                raise ValueError('quarantine_target_or_layout')
            layout = decode_json(layout_raw, compressed=True)
            attempts = {PurePosixPath(path).name for path in proof['attempts']}
            if (not attempts or any(not re.fullmatch(r'attempt-[1-9][0-9]*', name) for name in attempts)
                    or any(PurePosixPath(path).parent != PurePosixPath(str(parent)) for path in proof['attempts'])):
                raise ValueError('invalid_quarantine_scope')
            files = {row['relative']: row for row in layout['files']}
            directories = {row['relative']: row for row in layout['directories']}
            if set(files) != set(proof['source_file_sha']):
                raise ValueError('quarantine_file_set')
            for name, row in {**directories, **files}.items():
                relative_name(name, attempts)
                if row['uid'] != os.getuid() or row['gid'] not in [os.getgid(), *os.getgroups()]:
                    raise ValueError('quarantine_owner_unavailable')
            if any((parent / name).exists() or (parent / name).is_symlink() for name in attempts):
                raise ValueError('existing_quarantine_target_preserved')
            reserve = 4 * 1024**3
            required = sum(row['bytes'] for row in files.values())
            if shutil.disk_usage(parent).free < required + reserve:
                raise ValueError('insufficient_quarantine_rollback_space')
            for name in sorted(directories, key=lambda n: len(PurePosixPath(n).parts)):
                (parent / name).mkdir(mode=0o700)
            seen = set()
            for member in entries:
                relative_name(member.name, attempts)
                if not member.isfile() or member.name not in files or member.name in seen:
                    raise ValueError('invalid_quarantine_transport')
                row, expected = files[member.name], proof['source_file_sha'][member.name]
                if member.size != row['bytes'] or member.size != expected['bytes'] or row['sha256'] != expected['sha256']:
                    raise ValueError('quarantine_file_size')
                path, whole = parent / member.name, hashlib.sha256()
                with path.open('xb') as target, archive.extractfile(member) as source:
                    while raw := source.read(1024**2):
                        if shutil.disk_usage(parent).free < reserve + len(raw):
                            raise ValueError('insufficient_quarantine_rollback_space')
                        target.write(raw); whole.update(raw)
                    target.flush(); os.fsync(target.fileno())
                if whole.hexdigest() != expected['sha256']:
                    raise ValueError('quarantine_file_checksum')
                restore_metadata(path, row)
                seen.add(member.name)
            if seen != set(files):
                raise ValueError('incomplete_quarantine_transport')
            for name in sorted(directories, key=lambda n: len(PurePosixPath(n).parts), reverse=True):
                restore_metadata(parent / name, directories[name])
    return {'event': 'failed_stage_rollback_verified', 'files': len(seen),
            'database_restored': False, 'api_or_worker_started': False}


def restore_metadata(path, row):
    if path.is_symlink():
        raise ValueError('unsafe_quarantine_target')
    os.chown(path, row['uid'], row['gid']); os.chmod(path, row['mode'])
    for key, value in row['xattrs'].items():
        os.setxattr(path, key, base64.b64decode(value), follow_symlinks=False)
    os.utime(path, ns=(row['mtime_ns'], row['mtime_ns']), follow_symlinks=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['verify', 'emit', 'receive'])
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--proof-sha256', required=True)
    parser.add_argument('--target-parent', type=Path)
    parser.add_argument('--backup-root', type=Path)
    parser.add_argument('--service', default='arborscan-backup.service')
    args = parser.parse_args()
    try:
        if args.mode == 'receive':
            result = receive_bundle(sys.stdin.buffer, proof_sha256=args.proof_sha256,
                target_parent=args.target_parent, backup_root=args.backup_root, service=args.service)
        else:
            bundle = load_bundle(args.bundle, args.proof_sha256)
            if args.mode == 'emit':
                emit_bundle(bundle, sys.stdout.buffer)
                return 0
            result = {'event': 'failed_stage_quarantine_verified',
                      'files': len(bundle['proof']['source_file_sha']),
                      'logical_bytes': bundle['proof']['logical_bytes'],
                      'proof_sha256': args.proof_sha256, 'source_modified': False,
                      'rollback_executed': False}
        print(json.dumps(result), file=sys.stderr if args.mode == 'receive' else sys.stdout)
        return 0
    except Exception:
        print(json.dumps({'event': 'failed_stage_quarantine_refused', 'reason': 'private_validation_required'}),
              file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
