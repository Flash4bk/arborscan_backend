"""Per-invocation read budget and mutation guards, using disposable files only.

The forced Linux gate tests control flow on Windows, not Linux ctime semantics.
The two restored-mtime tests require a real Linux filesystem and run there only.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'deploy-vps'))
import ops_backup_policy as policy
import ops_selfhost_backup as selfhost

LINUX = sys.platform.startswith('linux')
IMAGE = 'sha256:' + 'a' * 64


def file_tar(path, *, filesystem=False):
    members = {'fixture': b'synthetic bytes'}
    if filesystem:
        row = {'uid': 0, 'gid': 0, 'mode': 0o600, 'mtime_ns': 1, 'xattrs': {}}
        members['filesystem-metadata.json'] = selfhost.canonical({
            'format': 1, 'files': {'fixture': row}, 'directories': {'.': row}})
    members['manifest.json'] = selfhost.canonical({
        name: hashlib.sha256(raw).hexdigest() for name, raw in members.items()})
    with tarfile.open(path, 'w') as archive:
        for name, raw in members.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(raw)
            archive.addfile(entry, io.BytesIO(raw))


@pytest.fixture
def fixture(tmp_path):
    root = tmp_path / 'backups'
    root.mkdir()
    asset = tmp_path / 'image.tar'
    asset.write_bytes(b'synthetic runtime image')
    runtime = {'format': 1, 'images': {IMAGE: [
        {'path': str(asset), 'sha256': policy.digest(asset)}]}}
    index = tmp_path / 'runtime-index.json'
    index.write_text(json.dumps(runtime))

    def make(name, *, full_selfhost=False, operation=None):
        folder = root / name
        folder.mkdir()
        for required in policy.REQUIRED:
            (folder / required).write_bytes(b'synthetic')
        rows = [{'Name': name, 'Image': IMAGE} for name in sorted(policy.RUNTIME_SERVICES)]
        selected = dict(runtime)
        if full_selfhost:
            rows = [{'Name': '/fixture-' + role, 'Image': IMAGE, 'Config': {'Labels': {
                'com.docker.compose.service': role,
                'com.docker.compose.project': 'disposable'}}} for role in sorted(selfhost.SERVICES)]
            selected.update(selfhost=True, services=[row['Name'] for row in rows])
            (folder / 'SELFHOST_BACKUP.json').write_text(json.dumps({
                'format': 1, 'kind': 'local_selfhost_backup', 'services': sorted(selfhost.SERVICES)}))
            for name in ('storage-files.tar', 'runtime-files.tar', 'local-files.tar'):
                file_tar(folder / name, filesystem=name != 'local-files.tar')
        (folder / 'containers.private.json').write_text(json.dumps(rows))
        (folder / 'RUNTIME_DEPENDENCIES.json').write_text(json.dumps(selected))
        postgres = folder / 'postgres'
        postgres.mkdir()
        for name in policy.NATIVE:
            (postgres / name).write_bytes(b'PGDMPfixture' if name == 'database.dump' else b'synthetic')
        (postgres / 'SHA256SUMS').write_text(''.join(
            policy.digest(postgres / name) + '  ' + name + '\n' for name in sorted(policy.NATIVE)))
        (postgres / 'COMPLETE').touch()
        (folder / 'AUTO_BACKUP.json').write_text(json.dumps({
            'format': 1, 'kind': 'scheduled_backup', 'name': folder.name,
            'operation_id': operation or folder.name}))
        policy.write_manifest(folder)
        (folder / 'COMPLETE').touch()
        return folder

    return root, asset, index, make


@pytest.fixture
def linux_gate(monkeypatch):
    # Exercise the cache's Linux branch locally without changing os.name/Path.
    monkeypatch.setattr(sys, 'platform', 'linux')


def test_replay_reads_each_snapshot_only_once_per_invocation(fixture, linux_gate):
    root, _, index, make = fixture
    folders = [make(f'20260101T00000{i}Z', full_selfhost=True,
                    operation='replay' if i == 2 else None) for i in range(3)]
    reads = []
    original = hashlib.file_digest

    def measured(stream, *args, **kwargs):
        name = getattr(stream, 'name', None)
        if isinstance(name, str) and Path(name).name == 'storage-files.tar':
            reads.append(Path(name))
        return original(stream, *args, **kwargs)

    with patch.object(hashlib, 'file_digest', side_effect=measured):
        for _ in range(2):
            result = policy.run_backup(root, lambda _: pytest.fail('replay created a snapshot'),
                                       index, operation_id='replay', retain=3)
            assert result == {'event': 'verified_existing', 'name': folders[2].name, 'deleted': []}
    # First verification retains outer SHA AND the selfhost full SHA scan.
    # A second invocation must read them all again, with no persistent claims.
    assert {folder.name: reads.count(folder / 'storage-files.tar') for folder in folders} == {
        folder.name: 4 for folder in folders}


@pytest.mark.parametrize('change', ['add', 'remove', 'replace', 'replace-root', 'mode'])
def test_changed_tree_is_never_reused(fixture, linux_gate, change):
    root, _, _, make = fixture
    folder = make('20260101T000000Z')
    cache = policy._VerifiedSetCache()
    policy.verify_set(folder, verified_cache=cache)
    target = folder / 'application.tar'
    if change == 'add':
        (folder / 'not-in-manifest').touch()
    elif change == 'remove':
        target.unlink()
    elif change == 'replace':
        replacement = folder / 'replacement'
        replacement.write_bytes(target.read_bytes())
        os.replace(replacement, target)
    elif change == 'replace-root':
        folder.rename(root / 'moved')
        make(folder.name)
    else:
        target.chmod(0o400 if target.stat().st_mode & 0o200 else 0o600)
    with pytest.raises(policy.PolicyError, match='changed|unsafe'):
        policy.verify_set(folder, verified_cache=cache)
    if target.exists():
        target.chmod(0o600)


def test_mutation_during_full_verification_is_not_cached(fixture, linux_gate):
    _, _, _, make = fixture
    folder = make('20260101T000000Z')
    original = policy.digest

    def mutate(path):
        result = original(path)
        if Path(path).name == 'application.tar':
            (folder / 'uncovered').touch()
        return result

    with patch.object(policy, 'digest', side_effect=mutate):
        with pytest.raises(policy.PolicyError, match='changed'):
            policy.verify_set(folder, verified_cache=policy._VerifiedSetCache())


def test_mutation_during_hit_is_refused(fixture, linux_gate):
    _, _, _, make = fixture
    folder = make('20260101T000000Z')
    cache = policy._VerifiedSetCache()
    policy.verify_set(folder, verified_cache=cache)
    original = policy.verify_runtime

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        (folder / 'uncovered').touch()
        return result

    with patch.object(policy, 'verify_runtime', side_effect=mutate):
        with pytest.raises(policy.PolicyError, match='changed'):
            policy.verify_set(folder, verified_cache=cache)


def test_cache_hit_revalidates_external_runtime(fixture, linux_gate):
    _, asset, _, make = fixture
    folder = make('20260101T000000Z')
    cache = policy._VerifiedSetCache()
    policy.verify_set(folder, verified_cache=cache)
    asset.write_bytes(b'changed runtime image')
    with pytest.raises(policy.PolicyError, match='runtime_archive_.*(changed|mismatch)'):
        policy.verify_set(folder, verified_cache=cache)


@pytest.mark.skipif(not LINUX, reason='Requires Linux change-time semantics')
@pytest.mark.parametrize('external', [False, True])
def test_same_size_rewrite_with_restored_mtime_is_refused(fixture, external):
    _, asset, _, make = fixture
    folder = make('20260101T000000Z')
    cache = policy._VerifiedSetCache()
    policy.verify_set(folder, verified_cache=cache)
    target = asset if external else folder / 'application.tar'
    before = target.stat()
    target.write_bytes(b'x' * before.st_size)
    os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(policy.PolicyError, match='changed'):
        policy.verify_set(folder, verified_cache=cache)


def test_stricter_native_request_and_result_mutation_cannot_poison_cache(fixture, linux_gate):
    _, _, _, make = fixture
    folder = make('20260101T000000Z')
    manifest = folder / 'SHA256SUMS'
    manifest.write_text(''.join(line + '\n' for line in manifest.read_text().splitlines()
                                if not line.endswith('  postgres/SHA256SUMS')))
    cache = policy._VerifiedSetCache()
    checked = policy.verify_set(folder, verified_cache=cache)
    checked['native_manifest_covered'] = True
    checked['dependencies'].clear()
    with pytest.raises(policy.PolicyError, match='native_manifest_not_covered'):
        policy.verify_set(folder, require_native=True, verified_cache=cache)
    assert policy.verify_set(folder, verified_cache=cache)['dependencies']


def test_initial_full_verification_keeps_tar_semantic_checks(fixture, linux_gate):
    _, _, _, make = fixture
    folder = make('20260101T000000Z', full_selfhost=True)
    target = folder / 'storage-files.tar'
    target.write_bytes(target.read_bytes().replace(b'synthetic bytes', b'corrupted bytes', 1))
    policy.write_manifest(folder)  # Valid outer SHA cannot conceal bad member SHA.
    with pytest.raises(policy.PolicyError, match='file_archive_checksum_mismatch'):
        policy.verify_set(folder, verified_cache=policy._VerifiedSetCache())


@pytest.mark.parametrize('failure,code', [
    ('checksum', 'checksum_mismatch'),
    ('native-header', 'invalid_postgres_dump_header'),
    ('runtime', 'runtime_archive_checksum_mismatch'),
])
def test_initial_full_checks_are_mandatory(fixture, linux_gate, failure, code):
    _, asset, _, make = fixture
    folder = make('20260101T000000Z')
    if failure == 'checksum':
        (folder / 'application.tar').write_bytes(b'corrupted')
    elif failure == 'native-header':
        (folder / 'postgres/database.dump').write_bytes(b'invalid')
        (folder / 'postgres/SHA256SUMS').write_text(''.join(
            policy.digest(folder / 'postgres' / name) + '  ' + name + '\n'
            for name in sorted(policy.NATIVE)))
        policy.write_manifest(folder)
    else:
        asset.write_bytes(b'changed image')
    with pytest.raises(policy.PolicyError, match=code):
        policy.verify_set(folder, require_native=True, require_runtime=True,
                          verified_cache=policy._VerifiedSetCache())


def test_stricter_runtime_request_cannot_reuse_legacy_success(fixture, linux_gate):
    _, _, _, make = fixture
    folder = make('20260101T000000Z')
    (folder / 'RUNTIME_DEPENDENCIES.json').unlink()
    policy.write_manifest(folder)
    cache = policy._VerifiedSetCache()
    assert not policy.verify_set(folder, verified_cache=cache)['runtime_archives_verified']
    with pytest.raises(policy.PolicyError, match='runtime_archives_required'):
        policy.verify_set(folder, require_runtime=True, verified_cache=cache)


def test_unpublished_verification_always_reads_again(fixture, linux_gate):
    _, _, _, make = fixture
    folder = make('20260101T000000Z')
    (folder / 'COMPLETE').unlink()
    cache = policy._VerifiedSetCache()
    with patch.object(policy, 'digest', wraps=policy.digest) as measured:
        for _ in range(2):
            policy.verify_set(folder, complete=False, verified_cache=cache)
    assert sum(Path(call.args[0]) == folder / 'application.tar' for call in measured.call_args_list) == 2


def test_rotation_candidate_always_gets_fresh_full_verification(fixture, linux_gate):
    root, _, _, make = fixture
    old = make('20260101T000000Z')
    new = make('20260101T000001Z')
    cache = policy._VerifiedSetCache()
    proposal = policy.plan(root, retain=1, new_name=new.name, verified_cache=cache)
    with patch.object(policy, 'digest', wraps=policy.digest) as measured:
        assert policy.rotate(root, proposal, new.name, verified_cache=cache) == [old.name]
    assert sum(Path(call.args[0]) == old / 'application.tar' for call in measured.call_args_list) == 1
    assert new.is_dir() and not old.exists()


@pytest.mark.parametrize('platform', ['win32', 'darwin'])
def test_unvalidated_platforms_keep_full_reads(fixture, monkeypatch, platform):
    _, _, _, make = fixture
    folder = make('20260101T000000Z')
    monkeypatch.setattr(sys, 'platform', platform)
    cache = policy._VerifiedSetCache()
    with patch.object(policy, 'digest', wraps=policy.digest) as measured:
        policy.verify_set(folder, verified_cache=cache)
        policy.verify_set(folder, verified_cache=cache)
    assert sum(Path(call.args[0]) == folder / 'application.tar' for call in measured.call_args_list) == 2
