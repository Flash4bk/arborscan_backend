"""Read-only manifest hints save remote chunking, never replace SHA checks."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'deploy-vps'))
import ops_pull_backups as pull
import ops_backup_policy as policy
from test_ops_selfhost_backup import environment
from test_ops_selfhost_readers import local_set


def inventory(local_set, monkeypatch, capsys, *, known):
    env, folder = local_set
    calls = []
    def run(args, **kwargs):
        raw = 'unlabelled-test\n' if args[1] == 'ps' else '[{"Config":{"Labels":null},"Mounts":[]}]'
        return subprocess.CompletedProcess(args, 0, raw, '')
    monkeypatch.setattr(subprocess, 'run', run)
    original = Path.open
    def open(path, *args, **kwargs):
        if path == folder / 'application.tar':
            calls.append(path)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', open)
    code = pull.INVENTORY.replace("pathlib.Path('/home/arborscan/ops-backups')",
                                   'pathlib.Path(' + repr(str(env.output)) + ')')
    exec(code, {'KNOWN_LOCAL_MANIFESTS': known})
    return json.loads(capsys.readouterr().out)[0], calls


@pytest.fixture
def large_set(local_set):
    env, folder = local_set
    # Valid tar padding preserves the logical payload while exceeding the real
    # 32-MiB chunking threshold; its complete bytes are covered by the outer SHA.
    with (folder / 'application.tar').open('ab') as stream:
        stream.write(b'\0' * (33 * 1024**2))
    policy.write_manifest(folder)
    return local_set


def test_matching_manifest_skips_only_unused_chunk_hashing(large_set, monkeypatch, capsys):
    env, folder = large_set
    known = pull.known_local_manifests(env.output)
    assert known == {folder.name: hashlib.sha256((folder / 'SHA256SUMS').read_bytes()).hexdigest()}
    record, opened = inventory(large_set, monkeypatch, capsys, known=known)
    assert record['blocks'] == {} and not opened
    assert record['runtime_assets'][0]['sha256'] == policy.digest(env.asset)
    # The fixture's remote image path is a temporary Windows path. Runtime
    # relocation is exercised separately by the real reader fixtures; this
    # assertion isolates covered local-file verification after a cheap hint.
    monkeypatch.setattr(pull, 'transfer_runtime', lambda *args, **kwargs: None)
    assert pull.transfer(env.output, record, lambda *_: pytest.fail('existing bytes redownloaded'),
        download_many=lambda *_: pytest.fail('existing images redownloaded')) == 'verified_existing'
    assert opened == [folder / 'application.tar']


@pytest.mark.parametrize('known', [{}, {'20261012T170000Z': '0' * 64}])
def test_missing_or_changed_hint_keeps_full_chunk_hashes(large_set, monkeypatch, capsys, known):
    _, folder = large_set
    record, opened = inventory(large_set, monkeypatch, capsys, known=known)
    assert opened == [folder / 'application.tar']
    chunks = record['blocks']['application.tar']
    assert len(chunks) >= 9 and sum(c['size'] for c in chunks) == (folder / 'application.tar').stat().st_size


def test_a_hint_cannot_hide_damaged_local_data(local_set, monkeypatch, capsys):
    env, folder = local_set
    known = pull.known_local_manifests(env.output)
    record, _ = inventory(local_set, monkeypatch, capsys, known=known)
    app = folder / 'application.tar'; raw = app.read_bytes()
    app.write_bytes(raw.replace(b'abc', b'bad', 1))
    assert pull.known_local_manifests(env.output) == known
    with pytest.raises(ValueError, match='damaged backup file'):
        pull.transfer(env.output, record, lambda *_: pytest.fail('must not silently rewrite old data'))
    assert app.read_bytes() != raw


def test_a_hint_does_not_skip_corrupt_runtime_archive_detection(local_set, monkeypatch, capsys):
    env, _ = local_set
    known = pull.known_local_manifests(env.output)
    env.asset.write_bytes(b'corrupted image archive')
    with pytest.raises(ValueError, match='runtime_archive_checksum_mismatch'):
        inventory(local_set, monkeypatch, capsys, known=known)


def test_hints_and_recovery_timeout_are_bounded_and_keep_secure_command():
    command = ['ssh', '-o', 'StrictHostKeyChecking=yes', 'test-host', 'python3 -']
    known = {'20261012T170000Z': 'a' * 64}; calls = []
    def run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, b'[{"name":"20261012T170000Z"}]', b'')
    pull.read_inventory(command, known=known, timeout=600, run=run)
    assert calls[0][0] == command and calls[0][1]['timeout'] == 600
    assert b'KNOWN_LOCAL_MANIFESTS' in calls[0][1]['input']
    for invalid in (0, 601, True):
        with pytest.raises(ValueError, match='bounded_inventory_timeout_required'):
            pull.read_inventory(command, known=known, timeout=invalid, run=run)
    for invalid in ({'../escape': 'a' * 64}, {'20261012T170000Z': 'not SHA'}):
        with pytest.raises(ValueError, match='invalid_local_manifest_hint'):
            pull.read_inventory(command, known=invalid, run=run)
    assert len(calls) == 1
