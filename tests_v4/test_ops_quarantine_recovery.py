"""Synthetic private bundles only; never reads production/real backup roots."""
import base64
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'deploy-vps'))
import ops_failed_stage_quarantine as q
import ops_failed_stage_cleanup as cleanup
from ops_failed_stage_cleanup import Scope, verify_tree


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / 'bundle'; root.mkdir(); pins = root / 'pinned-local-bases'; pins.mkdir()
    old, new, secret = b'opaque base payload', b'new native block', b'SYNTHETIC_PRIVATE=not-real'
    (pins / (sha(old) + '.archive')).write_bytes(old)
    source = '/home/arborscan/ops-backups/.staging/20260101T000000Z'
    def row(name, raw):
        return {'relative': name, 'bytes': len(raw), 'sha256': sha(raw),
                'uid': 1000, 'gid': 1000, 'mode': 0o600, 'mtime_ns': 1, 'xattrs': {}}
    files = [row('attempt-1/application.tar', b'head' + old + b'tail'),
             row('attempt-1/postgres/database.dump', new), row('attempt-1/arborscan.env.private', secret)]
    files[1]['blocks'] = [{'offset': 0, 'size': len(new), 'sha256': sha(new)}]
    parts, offset = [], 0
    for raw, inline in [(b'head', True), (old, False), (b'tail', True)]:
        part = {'offset': offset, 'size': len(raw), 'sha256': sha(raw)}
        if inline:
            part['inline'] = base64.b64encode(raw).decode()
        parts.append(part); offset += len(raw)
    layout = {'files': files, 'directories': [{'relative': 'attempt-1'}, {'relative': 'attempt-1/postgres'}],
              'tars': [{'sha256': files[0]['sha256'], 'bytes': files[0]['bytes'], 'parts': parts}]}
    local = {sha(old): {'path': 'obsolete-machine-location', 'offset': 0, 'size': len(old), 'archive_sha256': sha(old)}}
    missing = {sha(new): {'size': len(new)}, sha(secret): {'size': len(secret)}}
    (root / 'FAILED_LAYOUT.private.json.gz').write_bytes(gzip.compress(json.dumps(layout).encode()))
    (root / 'TAR_AND_NATIVE_LOCAL_REUSE.private.json').write_text(json.dumps(local))
    (root / 'MISSING_QUARANTINE_BLOBS.private.json').write_text(json.dumps(missing))
    with tarfile.open(root / 'quarantine-blobs.private.tar.gz', 'w:gz') as archive:
        for name, raw in {sha(new): new, sha(secret): secret,
                          'COPY_RECEIPT.json': json.dumps({'source_before_and_after_cas': True}).encode()}.items():
            m = tarfile.TarInfo(name); m.size = len(raw); archive.addfile(m, io.BytesIO(raw))
    proof = {'source_parent': source, 'attempts': [source + '/attempt-1'],
             'source_file_sha': {r['relative']: {'bytes': r['bytes'], 'sha256': r['sha256']} for r in files},
             'logical_bytes': sum(r['bytes'] for r in files),
             'pinned_local_base_sha': {'obsolete-machine-location': sha(old)}}
    for name, key in [('quarantine-blobs.private.tar.gz', 'blob_pack_sha256'),
                      ('FAILED_LAYOUT.private.json.gz', 'layout_sha256'),
                      ('TAR_AND_NATIVE_LOCAL_REUSE.private.json', 'local_reuse_index_sha256'),
                      ('MISSING_QUARANTINE_BLOBS.private.json', 'missing_blob_index_sha256')]:
        proof[key] = q.digest(root / name)
    raw = json.dumps(proof).encode(); expected = sha(raw)
    (root / 'QUARANTINE_RECONSTRUCTION.private.json').write_bytes(raw)
    (root / 'QUARANTINE_COMPLETE').write_text(expected + '\n')
    return root, expected, files, old


def test_relocated_bundle_raw_tar_native_and_opaque_secret_survive(bundle):
    root, expected, files, old = bundle
    loaded = q.load_bundle(root, expected)
    output = io.BytesIO(); q.emit_bundle(loaded, output)
    with tarfile.open(fileobj=io.BytesIO(output.getvalue())) as archive:
        assert archive.getnames()[0] == 'RESTORE_METADATA.private.json'
        for row in files:
            data = archive.extractfile(row['relative']).read()
            assert len(data) == row['bytes'] and sha(data) == row['sha256']
    assert b''.join(q.file_chunks(files[0], loaded['tars'], loaded['local'], loaded['blobs'])) == b'head' + old + b'tail'


@pytest.mark.parametrize('asset', ['pin', 'blob', 'layout', 'marker', 'proof'])
def test_corrupted_proof_asset_never_emits(bundle, asset):
    root, expected, _, _ = bundle
    if asset == 'pin':
        path = next((root / 'pinned-local-bases').iterdir())
    else:
        path = root / {'blob': 'quarantine-blobs.private.tar.gz', 'layout': 'FAILED_LAYOUT.private.json.gz',
                       'marker': 'QUARANTINE_COMPLETE', 'proof': 'QUARANTINE_RECONSTRUCTION.private.json'}[asset]
    path.write_bytes(b'corrupt')
    with pytest.raises((ValueError, OSError)):
        q.load_bundle(root, expected)


def test_segment_hash_and_size_cannot_be_bypassed():
    raw = b'correct'
    with pytest.raises(ValueError, match='checksum'):
        list(q.segment_chunks({'sha256': sha(raw), 'size': len(raw), 'inline': base64.b64encode(b'changed').decode()}, {}, {}))


@pytest.mark.parametrize('name', ['../attempt-1/a', '/attempt-1/a', 'attempt-2/a', 'attempt-1/../a', 'attempt-1\\a'])
def test_transport_scope_rejects_other_paths(name):
    with pytest.raises(ValueError):
        q.relative_name(name, {'attempt-1'})


def test_decompression_budget_rejects_bomb():
    compressed = gzip.compress(b'x' * (q.MAX_JSON + 1))
    with pytest.raises(ValueError, match='oversize'):
        q.decode_json(compressed, compressed=True)


@pytest.mark.parametrize('case', ['active', 'duplicate', 'traversal', 'parent', 'proof', 'protected'])
def test_cleanup_scope_protects_active_and_complete_sets(tmp_path, case):
    root = tmp_path / 'backups'; parent = root / '.staging/20260101T000000Z'
    names, proof, minimum = ('attempt-1',), 'a' * 64, 2
    if case == 'active': names = ('attempt-8',)
    if case == 'duplicate': names = ('attempt-1', 'attempt-1')
    if case == 'traversal': names = ('attempt-1/../../',)
    if case == 'parent': parent = root / 'actual-live-data'
    if case == 'proof': proof = 'not-a-sha'
    if case == 'protected': minimum = 1
    with pytest.raises(ValueError):
        Scope(root, parent, names, proof, 'fixture-operation', 8, minimum).validate()


def test_valid_explicit_cleanup_scope_has_no_side_effect(tmp_path):
    root = tmp_path / 'backups'
    Scope(root, root / '.staging/20260101T000000Z', ('attempt-1',), 'a' * 64,
          'fixture-operation', 8, 2).validate()
    assert not root.exists()


@pytest.mark.parametrize('case', ['proof', 'layout', 'files', 'metadata'])
def test_cleanup_request_must_bind_to_actual_offhost_receipt_before_source_access(tmp_path, case):
    root = tmp_path / 'backups'; parent = root / '.staging/20260101T000000Z'
    row = {'relative': 'attempt-1/opaque', 'bytes': 1, 'sha256': sha(b'x'),
           'uid': 1000, 'gid': 1000, 'mode': 0o600, 'mtime_ns': 1, 'xattrs': {}}
    layout = {'files': [row], 'directories': []}
    layout_raw = gzip.compress(json.dumps(layout).encode())
    proof = {'source_parent': str(parent), 'attempts': [str(parent / 'attempt-1')],
             'source_file_sha': {row['relative']: {'bytes': row['bytes'], 'sha256': row['sha256']}},
             'layout_sha256': sha(layout_raw)}
    proof_raw = json.dumps(proof).encode(); expected = sha(proof_raw)
    request = {'proof_sha256': expected, 'source_parent': str(parent),
               'proof_raw_base64': base64.b64encode(proof_raw).decode(),
               'layout_gzip_base64': base64.b64encode(layout_raw).decode(),
               'files': {row['relative']: dict(row)}, 'directories': []}
    if case == 'proof': request['proof_raw_base64'] = base64.b64encode(b'changed').decode()
    if case == 'layout': request['layout_gzip_base64'] = base64.b64encode(b'changed').decode()
    if case == 'files': request['files'][row['relative']]['sha256'] = 'f' * 64
    if case == 'metadata': request['files'][row['relative']]['uid'] += 1
    with pytest.raises(ValueError, match='proof_checksum|layout_checksum|not_bound'):
        verify_tree(request, Scope(root, parent, ('attempt-1',), expected, 'fixture-operation', 8))
    assert not root.exists()


def test_chunk_reader_handles_boundaries_and_refuses_unbounded_read():
    reader = q.ChunkReader(iter([b'a', b'bc']))
    assert reader.read(2) == b'ab' and reader.read(2) == b'c' and reader.read(2) == b''
    with pytest.raises(ValueError): reader.read()


@pytest.fixture
def planned_cleanup_tree(tmp_path, monkeypatch):
    root = tmp_path / 'backups'; parent = root / '.staging/20260101T000000Z'
    target = parent / 'attempt-1'; (target / 'postgres').mkdir(parents=True)
    (target / 'postgres/database.dump').write_bytes(b'synthetic-original')
    (parent / 'attempt-8').mkdir(); (parent / 'attempt-8/protected').write_bytes(b'keep')
    scope = Scope(root, parent, ('attempt-1',), 'a' * 64, 'fixture-operation', 8)
    assert root.resolve().is_relative_to(tmp_path.resolve())
    # Ownership/xattrs are checked separately in the real Linux fixture. This
    # portable fixture exercises exact membership/deletion semantics on Windows.
    def safe_fixture(path):
        assert path.resolve().is_relative_to(tmp_path.resolve())
        if path.is_symlink(): raise ValueError('unsafe_source')
        return path.stat()
    monkeypatch.setattr(cleanup, 'safe', safe_fixture)
    states = {str(path): (cleanup.fingerprint(path.stat()), path.is_dir())
              for path in [target, *target.rglob('*')]}
    return scope, states, target


@pytest.mark.parametrize('mutation', ['file', 'directory', 'complete'])
def test_added_entry_after_plan_refuses_before_any_remove(planned_cleanup_tree, mutation):
    scope, states, target = planned_cleanup_tree
    if mutation == 'directory': (target / 'new-unrecorded').mkdir()
    else: (target / ('COMPLETE' if mutation == 'complete' else 'new-unrecorded')).write_bytes(b'new')
    with pytest.raises(ValueError, match='membership|completion'):
        cleanup.delete_verified_entries(scope, states)
    assert (target / 'postgres/database.dump').read_bytes() == b'synthetic-original'
    assert (scope.parent / 'attempt-8/protected').read_bytes() == b'keep'


def test_mutation_at_apply_boundary_is_rechecked(planned_cleanup_tree, monkeypatch):
    scope, states, target = planned_cleanup_tree
    original, calls = cleanup.recheck_tree, []
    def mutate(scope, states, name):
        calls.append(name)
        if len(calls) == 2: (target / 'new-at-boundary').write_bytes(b'preserve-new')
        return original(scope, states, name)
    monkeypatch.setattr(cleanup, 'recheck_tree', mutate)
    with pytest.raises(ValueError, match='membership'):
        cleanup.delete_verified_entries(scope, states)
    assert (target / 'postgres/database.dump').is_file()
    assert (target / 'new-at-boundary').read_bytes() == b'preserve-new'


def test_new_entry_during_unlink_is_never_recursively_removed(planned_cleanup_tree, monkeypatch):
    scope, states, target = planned_cleanup_tree
    original = Path.unlink
    def add_new(path, *args, **kwargs):
        if path.name == 'database.dump': (target / 'new-during-delete').write_bytes(b'preserve-new')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'unlink', add_new)
    with pytest.raises(OSError): cleanup.delete_verified_entries(scope, states)
    assert (target / 'new-during-delete').read_bytes() == b'preserve-new'
    assert (scope.parent / 'attempt-8/protected').read_bytes() == b'keep'


def test_exact_known_entries_removed_and_active_attempt_retained(planned_cleanup_tree):
    scope, states, target = planned_cleanup_tree
    cleanup.delete_verified_entries(scope, states)
    assert not target.exists()
    assert (scope.parent / 'attempt-8/protected').read_bytes() == b'keep'
