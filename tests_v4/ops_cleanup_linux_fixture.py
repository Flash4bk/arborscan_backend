"""Standalone Linux fixture: temporary synthetic trees only, no production paths.

Run with the exact reviewed ops modules on PYTHONPATH. No pytest installation,
Docker, SQL, systemd or real backup directory is used. Every created path is
inside a newly generated tempfile with an explicit arborscan fixture prefix.
"""
import base64
import gzip
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import io
import tarfile
from types import SimpleNamespace

import ops_failed_stage_cleanup as cleanup
import ops_failed_stage_quarantine as quarantine


def quarantine_roundtrip(tmp):
    root = tmp / 'roundtrip' / 'backups'
    parent = root / '.staging/20260101T000000Z'
    target = parent / 'attempt-1'
    target.mkdir(parents=True)
    (parent / 'attempt-8').mkdir()
    (root / 'backup.lock').touch(mode=0o600)
    file = target / 'opaque.bin'
    file.write_bytes(b'synthetic-opaque-payload')
    os.setxattr(file, 'user.arborscan_fixture', b'synthetic')
    def metadata(path):
        s = path.stat()
        return {'relative': path.relative_to(parent).as_posix(), 'uid': s.st_uid,
                'gid': s.st_gid, 'mode': stat.S_IMODE(s.st_mode), 'mtime_ns': s.st_mtime_ns,
                'xattrs': {key: base64.b64encode(os.getxattr(path, key)).decode() for key in os.listxattr(path)}}
    row = metadata(file); row.update(bytes=file.stat().st_size, sha256=hashlib.sha256(file.read_bytes()).hexdigest())
    layout = {'files': [row], 'directories': [metadata(target)], 'tars': []}
    bundle = tmp / 'roundtrip' / 'bundle'; bundle.mkdir()
    (bundle / 'FAILED_LAYOUT.private.json.gz').write_bytes(gzip.compress(json.dumps(layout).encode()))
    (bundle / 'TAR_AND_NATIVE_LOCAL_REUSE.private.json').write_text('{}')
    (bundle / 'MISSING_QUARANTINE_BLOBS.private.json').write_text(json.dumps({row['sha256']: {'size': row['bytes']}}))
    with tarfile.open(bundle / 'quarantine-blobs.private.tar.gz', 'w:gz') as archive:
        for name, raw in {row['sha256']: file.read_bytes(),
                'COPY_RECEIPT.json': json.dumps({'source_before_and_after_cas': True}).encode()}.items():
            member = tarfile.TarInfo(name); member.size = len(raw); archive.addfile(member, io.BytesIO(raw))
    proof = {'source_parent': str(parent), 'attempts': [str(target)],
             'source_file_sha': {row['relative']: {'bytes': row['bytes'], 'sha256': row['sha256']}},
             'logical_bytes': row['bytes'], 'pinned_local_base_sha': {}}
    for name, key in [('quarantine-blobs.private.tar.gz', 'blob_pack_sha256'),
                     ('FAILED_LAYOUT.private.json.gz', 'layout_sha256'),
                     ('TAR_AND_NATIVE_LOCAL_REUSE.private.json', 'local_reuse_index_sha256'),
                     ('MISSING_QUARANTINE_BLOBS.private.json', 'missing_blob_index_sha256')]:
        proof[key] = quarantine.digest(bundle / name)
    raw = json.dumps(proof).encode(); expected = hashlib.sha256(raw).hexdigest()
    (bundle / 'QUARANTINE_RECONSTRUCTION.private.json').write_bytes(raw)
    (bundle / 'QUARANTINE_COMPLETE').write_text(expected + '\n')
    loaded = quarantine.load_bundle(bundle, expected)
    wire = io.BytesIO(); quarantine.emit_bundle(loaded, wire)
    assert target.resolve().is_relative_to(tmp.resolve())
    file.unlink(); target.rmdir()
    original = quarantine.subprocess.run
    def inactive(command, **kwargs):
        assert command[:3] == ['systemctl', 'show', 'fixture.service']
        return SimpleNamespace(returncode=0, stdout=b'inactive')
    quarantine.subprocess.run = inactive
    try:
        result = quarantine.receive_bundle(io.BytesIO(wire.getvalue()), proof_sha256=expected,
            target_parent=parent, backup_root=root, service='fixture.service')
        assert result['files'] == 1 and result['database_restored'] is False
        assert file.read_bytes() == b'synthetic-opaque-payload'
        assert file.stat().st_mtime_ns == row['mtime_ns']
        assert stat.S_IMODE(file.stat().st_mode) == row['mode']
        assert os.getxattr(file, 'user.arborscan_fixture') == b'synthetic'
        try:
            quarantine.receive_bundle(io.BytesIO(wire.getvalue()), proof_sha256=expected,
                target_parent=parent, backup_root=root, service='fixture.service')
            raise AssertionError('Existing rollback target must be preserved')
        except ValueError as error:
            assert 'existing_quarantine_target' in str(error)
        assert file.read_bytes() == b'synthetic-opaque-payload'
    finally:
        quarantine.subprocess.run = original


def prepare(tmp, label):
    root = tmp / label / 'backups'
    parent = root / '.staging/20260101T000000Z'
    target = parent / 'attempt-1'
    (target / 'postgres').mkdir(parents=True)
    (target / 'postgres/database.dump').write_bytes(b'synthetic-only')
    (target / 'postgres/COMPLETE').touch()
    (parent / 'attempt-8').mkdir()
    (parent / 'attempt-8/keep').write_bytes(b'protected-fixture')
    active = {'name': parent.name, 'operation_id': 'fixture-operation', 'attempt': 8}
    (root / 'active-backup.json').write_text(json.dumps(active))
    for n in ['20260101T000100Z', '20260101T000200Z']:
        (root / n).mkdir(); (root / n / 'COMPLETE').touch()
    assert target.resolve().is_relative_to(tmp.resolve())
    files, directories = {}, []
    for path in [target, *sorted(target.rglob('*'))]:
        s = path.stat(); row = {'relative': path.relative_to(parent).as_posix(),
            'uid': s.st_uid, 'gid': s.st_gid, 'mode': stat.S_IMODE(s.st_mode),
            'mtime_ns': s.st_mtime_ns, 'xattrs': {}}
        if path.is_dir():
            directories.append(row)
        else:
            row.update(bytes=s.st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                       inode=s.st_ino, ctime_ns=s.st_ctime_ns)
            files[row['relative']] = row
    layout_raw = gzip.compress(json.dumps({'files': list(files.values()), 'directories': directories}).encode())
    proof_raw = json.dumps({'source_parent': str(parent), 'attempts': [str(target)],
        'layout_sha256': hashlib.sha256(layout_raw).hexdigest(),
        'source_file_sha': {name: {'bytes': r['bytes'], 'sha256': r['sha256']} for name, r in files.items()}}).encode()
    proof_sha = hashlib.sha256(proof_raw).hexdigest()
    request = {'proof_sha256': proof_sha, 'source_parent': str(parent), 'files': files,
        'directories': directories, 'proof_raw_base64': base64.b64encode(proof_raw).decode(),
        'layout_gzip_base64': base64.b64encode(layout_raw).decode()}
    scope = cleanup.Scope(root, parent, ('attempt-1',), proof_sha, 'fixture-operation', 8, 2)
    plan, states, active_bytes, completed = cleanup.verify_tree(request, scope)
    assert plan['source_files_sha_verified'] == 2 and len(states) == 4
    return scope, states, target, active_bytes, completed


def run():
    if os.name != 'posix':
        raise RuntimeError('Linux fixture requires real POSIX filesystem semantics')
    with tempfile.TemporaryDirectory(prefix='arborscan-ops-fixture-') as folder:
        tmp = Path(folder)
        assert tmp.resolve() == tmp and tmp.parent == Path(tempfile.gettempdir()).resolve()
        assert tmp.name.startswith('arborscan-ops-fixture-')
        scope, states, target, active, completed = prepare(tmp, 'positive')
        cleanup.delete_verified_entries(scope, states)
        assert not target.exists()
        assert (scope.parent / 'attempt-8/keep').read_bytes() == b'protected-fixture'
        assert (scope.root / 'active-backup.json').read_bytes() == active
        assert all((scope.root / name / 'COMPLETE').exists() for name in completed)
        scope, states, target, _, _ = prepare(tmp, 'between-plan-apply')
        (target / 'new-not-copied').write_bytes(b'must-survive')
        try:
            cleanup.delete_verified_entries(scope, states)
            raise AssertionError('New member must reject apply')
        except ValueError as error:
            assert 'membership' in str(error)
        assert (target / 'new-not-copied').read_bytes() == b'must-survive'
        assert (target / 'postgres/database.dump').read_bytes() == b'synthetic-only'
        scope, states, target, _, _ = prepare(tmp, 'during-delete')
        original = Path.unlink
        def insert_new(path, *args, **kwargs):
            if path.name == 'database.dump':
                (target / 'new-during-unlink').write_bytes(b'must-survive')
            return original(path, *args, **kwargs)
        Path.unlink = insert_new
        try:
            try:
                cleanup.delete_verified_entries(scope, states)
                raise AssertionError('rmdir must preserve new member')
            except OSError:
                pass
        finally:
            Path.unlink = original
        assert (target / 'new-during-unlink').read_bytes() == b'must-survive'
        assert (scope.parent / 'attempt-8/keep').read_bytes() == b'protected-fixture'
        quarantine_roundtrip(tmp)
    return {'event': 'linux_cleanup_fixture_passed', 'cases': 3,
            'positive_exact_delete': True, 'plan_apply_mutation_refused_before_delete': True,
            'concurrent_new_entry_preserved': True, 'production_paths_used': False,
            'systemd_or_docker_called': False, 'temporary_fixture_removed': True,
            'synthetic_quarantine_emit_receive_flock_metadata_xattr_passed': True,
            'existing_rollback_target_preserved': True,
            'production_database_restore_performed': False}


if __name__ == '__main__':
    print(json.dumps(run()))
