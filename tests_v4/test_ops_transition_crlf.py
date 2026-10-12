"""Legacy offsite newline translation is explicit, hash-bound and non-destructive.

All archives and links are tiny isolated fixtures, not installed backup files.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'deploy-vps'))
import ops_backup_policy as policy
import ops_pull_backups as pull
import ops_runtime_contract as contract
import test_ops_backup_policy as fixtures


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def legacy():
    fixture = fixtures.BackupPolicyTest()
    fixture.setUp()
    try:
        yield fixture
    finally:
        fixture.tearDown()


def original(legacy, number=0, crlf=True):
    folder = legacy.content(legacy.root / ('20261001T%06dZ' % number))
    manifest = folder / 'SHA256SUMS'
    source = manifest.read_bytes().replace(b'\r\n', b'\n')
    manifest.write_bytes(source.replace(b'\n', b'\r\n') if crlf else source)
    return folder, {'name': folder.name, 'manifest_sha256': sha(source)}


def records(*rows):
    return [{'transition_protection': {'format': 1, 'kind': 'selfhost_backup_transition',
                                      'sets': list(rows)}}]


def ledger(legacy):
    return legacy.root / contract.TRANSITION_FILE


def test_mixed_fourteen_originals_keep_exact_bytes_and_are_outside_rolling_pool(legacy):
    originals = [original(legacy, i, crlf=i >= 9) for i in range(14)]
    before = {folder.name: (folder / 'SHA256SUMS').read_bytes() for folder, _ in originals}
    trusted = records(*(row for _, row in originals))
    assert pull.preserve_transition_history(legacy.root, trusted)
    protected = json.loads(ledger(legacy).read_bytes())
    assert sum('local_manifest_sha256' in row for row in protected['sets']) == 5
    for row, (_, source_row) in zip(protected['sets'], originals):
        assert row['manifest_sha256'] == source_row['manifest_sha256']
        if 'local_manifest_sha256' in row:
            assert row['local_manifest_sha256'] == sha(before[row['name']])
    assert contract.transition_protection(legacy.root) == {name: sha(raw) for name, raw in before.items()}
    proposal = policy.plan(legacy.root)
    assert proposal['transition_protected_sets'] == 14
    assert proposal['retention_pool_sets'] == 0 and proposal['candidates'] == []
    assert all(row['valid'] for row in proposal['sets'])
    published = ledger(legacy).read_bytes()
    assert pull.preserve_transition_history(legacy.root, trusted)
    assert ledger(legacy).read_bytes() == published
    assert {folder.name: (folder / 'SHA256SUMS').read_bytes() for folder, _ in originals} == before
    assert all((folder / 'COMPLETE').is_file() for folder, _ in originals)


def test_lf_registry_stays_exact_existing_format_without_local_annotation(legacy):
    folder, row = original(legacy, crlf=False)
    trusted = records(row)
    assert pull.preserve_transition_history(legacy.root, trusted)
    assert json.loads(ledger(legacy).read_bytes()) == trusted[0]['transition_protection']
    assert contract.transition_protection(legacy.root) == {folder.name: row['manifest_sha256']}


def test_crlf_hint_is_source_digest_and_never_proves_covered_data(legacy):
    folder, row = original(legacy)
    assert pull.known_local_manifests(legacy.root) == {folder.name: row['manifest_sha256']}
    (folder / 'application.tar').write_bytes(b'damaged archive with unchanged manifest')
    assert pull.known_local_manifests(legacy.root) == {folder.name: row['manifest_sha256']}
    with pytest.raises(ValueError, match='damaged backup file'):
        pull.preserve_transition_history(legacy.root, records(row))
    assert not ledger(legacy).exists() and (folder / 'COMPLETE').exists()


@pytest.mark.parametrize('repeat', [False, True])
def test_full_covered_sha_is_required_on_adoption_and_repeat(legacy, repeat):
    folder, row = original(legacy)
    if repeat:
        pull.preserve_transition_history(legacy.root, records(row))
    prior = ledger(legacy).read_bytes() if repeat else None
    manifest = (folder / 'SHA256SUMS').read_bytes()
    (folder / 'postgres/database.dump').write_bytes(b'changed native archive')
    with pytest.raises(ValueError, match='damaged backup file'):
        pull.preserve_transition_history(legacy.root, records(row))
    assert (folder / 'SHA256SUMS').read_bytes() == manifest
    assert (ledger(legacy).read_bytes() if ledger(legacy).exists() else None) == prior
    assert (folder / 'COMPLETE').exists()


@pytest.mark.parametrize('defect', ['source', 'local_newline', 'local_content'])
def test_changed_source_or_local_manifest_cannot_rebind_existing_registry(legacy, defect):
    folder, row = original(legacy)
    trusted = records(row)
    pull.preserve_transition_history(legacy.root, trusted)
    prior = ledger(legacy).read_bytes()
    manifest = folder / 'SHA256SUMS'
    if defect == 'source':
        trusted[0]['transition_protection']['sets'][0]['manifest_sha256'] = '0' * 64
    elif defect == 'local_newline':
        manifest.write_bytes(manifest.read_bytes().replace(b'\r\n', b'\n'))
    else:
        manifest.write_bytes(manifest.read_bytes().replace(b'  application.tar', b'  different-file.tar'))
    current = manifest.read_bytes()
    if defect != 'source':
        with pytest.raises(ValueError, match='manifest_mismatch'):
            contract.transition_protection(legacy.root)
    with pytest.raises(ValueError):
        pull.preserve_transition_history(legacy.root, trusted)
    assert ledger(legacy).read_bytes() == prior and manifest.read_bytes() == current


@pytest.mark.parametrize('defect', ['wrong_local', 'wrong_source', 'same_as_source', 'non_string'])
def test_registry_read_requires_both_raw_and_source_sha(legacy, defect):
    folder, row = original(legacy)
    pull.preserve_transition_history(legacy.root, records(row))
    value = json.loads(ledger(legacy).read_bytes())
    annotated = value['sets'][0]
    if defect == 'wrong_local':
        annotated['local_manifest_sha256'] = '0' * 64
    elif defect == 'wrong_source':
        annotated['manifest_sha256'] = '0' * 64
    elif defect == 'same_as_source':
        annotated['local_manifest_sha256'] = annotated['manifest_sha256']
    else:
        annotated['local_manifest_sha256'] = 123
    ledger(legacy).write_text(json.dumps(value))
    with pytest.raises(ValueError, match='transition_protection'):
        contract.transition_protection(legacy.root)
    assert (folder / 'COMPLETE').exists()


def test_crlf_is_the_only_allowed_manifest_translation(legacy):
    folder, row = original(legacy)
    manifest = folder / 'SHA256SUMS'
    manifest.write_bytes(manifest.read_bytes().replace(b'\r\n', b'\r'))
    with pytest.raises(ValueError, match='not_verified_locally'):
        pull.preserve_transition_history(legacy.root, records(row))
    assert not ledger(legacy).exists()


def test_different_remote_registries_are_refused_without_local_adoption(legacy):
    _, row = original(legacy)
    trusted = records(row)
    altered = copy.deepcopy(trusted[0])
    altered['transition_protection']['sets'][0]['manifest_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='invalid_transition_protection'):
        pull.preserve_transition_history(legacy.root, [trusted[0], altered])
    assert not ledger(legacy).exists()


@pytest.mark.parametrize('destination', ['writing', 'verification_receipt'])
def test_mutable_registry_staging_and_receipt_hardlinks_refused_before_write(legacy, destination):
    folder, row = original(legacy)
    protected = legacy.home / 'protected-original'
    protected.write_bytes(b'original private metadata')
    target = (ledger(legacy).with_name(contract.TRANSITION_FILE + '.writing')
              if destination == 'writing' else folder / 'OFFSITE_VERIFIED.json')
    os.link(protected, target)
    before = protected.read_bytes(), protected.stat().st_mtime_ns
    with pytest.raises(ValueError, match='multiply_linked_writable_destination'):
        pull.preserve_transition_history(legacy.root, records(row))
    assert (protected.read_bytes(), protected.stat().st_mtime_ns) == before
    assert not ledger(legacy).exists()


@pytest.mark.parametrize('destination', ['registry', 'writing'])
def test_registry_symlink_refused_without_mutating_target(legacy, destination):
    _, row = original(legacy)
    protected = legacy.home / 'protected-original'
    protected.write_bytes(b'original private metadata')
    target = ledger(legacy) if destination == 'registry' else ledger(legacy).with_name(contract.TRANSITION_FILE + '.writing')
    try:
        target.symlink_to(protected)
    except OSError:
        pytest.skip('Symlink creation unavailable in this environment')
    with pytest.raises(ValueError, match='symlink_or_reparse_refused'):
        pull.preserve_transition_history(legacy.root, records(row))
    assert protected.read_bytes() == b'original private metadata'


def test_legitimate_interrupted_registry_writing_resumes_after_full_verification(legacy):
    folder, row = original(legacy)
    writing = ledger(legacy).with_name(contract.TRANSITION_FILE + '.writing')
    writing.write_bytes(b'{"interrupted')
    manifest = (folder / 'SHA256SUMS').read_bytes()
    assert pull.preserve_transition_history(legacy.root, records(row))
    assert not writing.exists()
    assert contract.transition_protection(legacy.root)[folder.name] == sha(manifest)
    assert (folder / 'SHA256SUMS').read_bytes() == manifest


def test_registry_destination_rechecked_before_atomic_publication(legacy, monkeypatch):
    _, row = original(legacy)
    protected = legacy.home / 'protected-original'
    protected.write_bytes(b'original private metadata')
    original_fsync = os.fsync
    def publish_alias(fd):
        original_fsync(fd)
        os.link(protected, ledger(legacy))
    monkeypatch.setattr(pull.os, 'fsync', publish_alias)
    with pytest.raises(ValueError, match='multiply_linked_writable_destination'):
        pull.preserve_transition_history(legacy.root, records(row))
    assert protected.read_bytes() == b'original private metadata'
    assert ledger(legacy).read_bytes() == b'original private metadata'


def test_absent_trusted_registry_never_adopts_local_copies(legacy):
    _, _ = original(legacy)
    assert pull.preserve_transition_history(legacy.root, [{}]) is False
    assert not ledger(legacy).exists() and contract.transition_protection(legacy.root) == {}


def test_main_refuses_hardlinked_retention_proposal_before_apply(legacy, monkeypatch):
    import ops_windows_job
    import ops_windows_retention as windows
    protected = legacy.home / 'protected-original'
    protected.write_bytes(b'original private proposal')
    os.link(protected, legacy.root / 'retention-dry-run.private.json')
    before = protected.read_bytes(), protected.stat().st_mtime_ns
    monkeypatch.setattr(sys, 'argv', ['pull', '--root', str(legacy.root)])
    monkeypatch.setenv('WINDIR', str(legacy.home))
    if os.name != 'nt':
        monkeypatch.setitem(sys.modules, 'msvcrt', SimpleNamespace(LK_NBLCK=1, locking=lambda *_: None))
        monkeypatch.setattr(pull.subprocess, 'CREATE_NO_WINDOW', 0, raising=False)
    monkeypatch.setattr(ops_windows_job, 'contain_children', lambda: None)
    monkeypatch.setattr(pull, 'read_inventory', lambda *a, **k: [{
        'name': '20261012T220000Z', 'runtime': {'selfhost': True}}])
    monkeypatch.setattr(pull, 'transfer', lambda *a, **k: 'verified_existing')
    monkeypatch.setattr(pull, 'preserve_transition_history', lambda *a: False)
    monkeypatch.setattr(windows, 'plan', lambda *a: {'before': 15, 'delete': []})
    monkeypatch.setattr(windows, 'apply', lambda *a, **k: pytest.fail('must refuse before rotation'))
    assert pull.main() == 1
    assert (protected.read_bytes(), protected.stat().st_mtime_ns) == before
    events = [json.loads(line) for line in (legacy.root / 'pull.log').read_text().splitlines()]
    assert events[-1]['reason'] == 'multiply_linked_writable_destination'
