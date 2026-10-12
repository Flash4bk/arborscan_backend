"""Real private files exercise the additive reader/transfer contract, no Docker.

Legacy cloud fixtures remain separate; these tests do not claim PostgreSQL
restore, a Windows scheduled job, or a production systemd execution.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'deploy-vps'))
import ops_backup_policy as policy
import ops_backup_dispatch as dispatch
import ops_pull_backups as pull
import ops_runtime_contract as contract
import ops_selfhost_backup as backup
import ops_windows_retention as windows

from test_ops_selfhost_backup import environment, FakeBackend, IMAGE
import test_ops_windows_runtime_retention as fixtures


@pytest.fixture
def local_set(environment):
    result = backup.backup(environment.root, environment.output, environment.index,
        layout='version-directory', stamp='20261012T170000Z', factory=FakeBackend,
        free=lambda _: SimpleNamespace(free=10 * 1024**3))
    return environment, environment.output / result['name']


def relocate_source_fixture(folder, environment):
    runtime = json.loads((folder / 'RUNTIME_DEPENDENCIES.json').read_bytes())
    source = '/home/arborscan/ops-runtime/synthetic.archive'
    asset = {'path': source, 'sha256': policy.digest(environment.asset)}
    runtime['images'] = {IMAGE: [asset]}
    policy.atomic_json(folder / 'RUNTIME_DEPENDENCIES.json', runtime)
    policy.write_manifest(folder)
    files = {name: (folder / name).read_bytes() for name in policy.manifest(folder)}
    return {'name': folder.name, 'manifest': (folder / 'SHA256SUMS').read_text(),
        'bytes': sum(len(raw) for raw in files.values()), 'runtime': runtime,
        'runtime_assets': [{**asset, 'bytes': environment.asset.stat().st_size}]}, files, source


def test_seven_runtime_linux_reader_and_legacy_names_are_preserved(local_set):
    _, folder = local_set
    checked = policy.verify_set(folder, require_native=True, require_runtime=True)
    assert checked['runtime_services'] == sorted('/selfhost-' + s for s in contract.SELFHOST_ROLES)
    assert checked['runtime_archives_verified']
    assert policy.runtime_images(folder) == {IMAGE}
    assert policy.runtime_services(folder, include_auth=True) == set(checked['runtime_services'])
    assert contract.LEGACY_V4 == {'/arborscan-api-v4', '/arborscan-quality-worker'}


def test_remote_inventory_handles_unlabelled_isolated_restore_container(local_set, monkeypatch, capsys):
    environment, folder = local_set
    cmds = []
    def run(args, **kwargs):
        cmds.append(args)
        value = 'restore-fixture\n' if args[1] == 'ps' else json.dumps([{
            'Config': {'Labels': None}, 'Mounts': [{
                'Source': '/home/arborscan/ops-backups/' + folder.name + '/compose'}]}])
        return subprocess.CompletedProcess(args, 0, value, '')
    monkeypatch.setattr(subprocess, 'run', run)
    code = pull.INVENTORY.replace("pathlib.Path('/home/arborscan/ops-backups')",
                                  'pathlib.Path(' + repr(str(environment.output)) + ')')
    exec(code, {})
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]['protected_sets'] == [folder.name]
    assert cmds[0] == ['docker', 'ps', '-aq']


def test_real_files_transfer_seven_runtime_and_idempotent_windows_readback(local_set, tmp_path):
    environment, folder = local_set
    record, files, source = relocate_source_fixture(folder, environment)
    offsite = tmp_path / 'offsite'; offsite.mkdir()
    transferred = []
    def download(name, filename, target):
        transferred.append((name, filename)); target.write_bytes(files[filename])
    def many(name, requests):
        for filename, target in requests:
            transferred.append((name, filename))
            target.write_bytes(environment.asset.read_bytes() if name is None else files[filename])
    assert pull.transfer(offsite, record, download, download_many=many,
        free=lambda _: SimpleNamespace(free=10 * 1024**3)) == 'downloaded_verified'
    restored = offsite / record['name']
    result = windows.verify_replacement(offsite, restored.name)
    assert result['selfhost'] is True
    assert len(result['runtime_services']) == 7
    assert result['unique_runtime_archives'] == 1
    assert sum(n is None and f == source for n, f in transferred) == 1
    before = {p.relative_to(offsite).as_posix(): p.read_bytes()
              for p in offsite.rglob('*') if p.is_file()}
    assert pull.transfer(offsite, record, lambda *_: pytest.fail('repeat downloaded data'),
        download_many=lambda *_: pytest.fail('repeat downloaded images')) == 'verified_existing'
    assert {p.relative_to(offsite).as_posix(): p.read_bytes()
            for p in offsite.rglob('*') if p.is_file() and p.name != 'OFFSITE_VERIFIED.json'} == {
                name: raw for name, raw in before.items() if not name.endswith('/OFFSITE_VERIFIED.json')}


@pytest.mark.parametrize('defect', ['missing_role', 'different_project', 'duplicate_name',
                                    'wrong_image', 'missing_marker', 'uncovered_marker', 'no_selfhost_flag'])
def test_seven_runtime_contract_rejects_mislabelled_or_uncovered_data(local_set, defect):
    _, folder = local_set
    runtime = json.loads((folder / 'RUNTIME_DEPENDENCIES.json').read_bytes())
    rows = json.loads((folder / 'containers.private.json').read_bytes())
    covered = policy.manifest(folder)
    if defect == 'missing_role':
        rows.pop()
    elif defect == 'different_project':
        rows[0]['Config']['Labels']['com.docker.compose.project'] = 'another-private-stack'
    elif defect == 'duplicate_name':
        rows[0]['Name'] = rows[1]['Name']
    elif defect == 'wrong_image':
        rows[0]['Image'] = 'sha256:' + 'f' * 64
    elif defect == 'missing_marker':
        (folder / 'SELFHOST_BACKUP.json').unlink()
    elif defect == 'uncovered_marker':
        covered.pop('SELFHOST_BACKUP.json')
    else:
        runtime.pop('selfhost')
    (folder / 'containers.private.json').write_text(json.dumps(rows))
    with pytest.raises(ValueError):
        contract.selected_runtime(folder, runtime, covered)


def test_corrupt_raw_storage_cannot_authorize_linux_reader(local_set):
    _, folder = local_set
    (folder / 'storage-files.tar').write_bytes(b'damaged')
    with pytest.raises((ValueError, policy.PolicyError)):
        policy.verify_set(folder, require_native=True, require_runtime=True)


def test_marker_only_cannot_replace_missing_raw_volume(local_set):
    _, folder = local_set
    (folder / 'runtime-files.tar').unlink()
    policy.write_manifest(folder)
    with pytest.raises(policy.PolicyError, match='incomplete_selfhost_set'):
        policy.verify_set(folder, require_native=True, require_runtime=True)


def test_transition_keeps_original14_outside_new_windows_pool_and_rotates_new14_only():
    fixture = fixtures.WindowsRuntimeRetentionTest(); fixture.setUp()
    try:
        originals = [fixture.fixture('20261001T%06dZ' % i)[0] for i in range(14)]
        before = {p.relative_to(fixture.root).as_posix(): p.read_bytes()
                  for folder in originals for p in folder.rglob('*') if p.is_file()}
        protection = {'format': 1, 'kind': 'selfhost_backup_transition', 'sets': [
            {'name': p.name, 'manifest_sha256': policy.digest(p / 'SHA256SUMS')} for p in originals]}
        (fixture.root / contract.TRANSITION_FILE).write_text(json.dumps(protection))
        fresh = [fixture.fixture('20261012T%06dZ' % i)[0] for i in range(14)]
        latest, record = fixture.fixture('20261013T010000Z'); fixture.copy_runtime(latest, record)
        proposal = windows.plan(fixture.root, latest.name)
        assert proposal['transition_protected_sets'] == 14
        assert proposal['retention_pool_sets'] == 15
        assert proposal['delete'] == [fresh[0].name]
        windows.apply(fixture.root, proposal)
        assert sum((p / 'COMPLETE').is_file() for p in fixture.root.iterdir()) == 28
        for name, raw in before.items():
            assert (fixture.root / name).read_bytes() == raw
    finally:
        fixture.tearDown()


def test_trusted_transition_registry_installs_after_exact_local_originals(local_set):
    environment, folder = local_set
    value = {'format': 1, 'kind': 'selfhost_backup_transition', 'sets': [
        {'name': folder.name, 'manifest_sha256': policy.digest(folder / 'SHA256SUMS')}]}
    record = {'transition_protection': value}
    assert pull.preserve_transition_history(environment.output, [record])
    before = (environment.output / contract.TRANSITION_FILE).read_bytes()
    assert pull.preserve_transition_history(environment.output, [record])
    assert (environment.output / contract.TRANSITION_FILE).read_bytes() == before
    value['sets'][0]['manifest_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='not_verified_locally'):
        pull.preserve_transition_history(environment.output, [record])
    assert (environment.output / contract.TRANSITION_FILE).read_bytes() == before


def test_scheduled_selfhost_requires_protection_then_replays_without_new_stage(local_set):
    environment, folder = local_set
    with pytest.raises(backup.BackupError, match='must_be_protected'):
        backup.scheduled_backup(environment.root, environment.output, environment.index,
            layout='version-directory', factory=FakeBackend, live=())
    (environment.output / contract.TRANSITION_FILE).write_text(json.dumps({
        'format': 1, 'kind': 'selfhost_backup_transition', 'sets': [{
            'name': folder.name, 'manifest_sha256': policy.digest(folder / 'SHA256SUMS')}]}))
    FakeBackend.events = []
    result = backup.scheduled_backup(environment.root, environment.output, environment.index,
        layout='version-directory', operation_id='selfhost-test', factory=FakeBackend, live=(),
        free=lambda _: SimpleNamespace(free=10 * 1024**3))
    assert result['event'] == 'completed_verified'
    assert result['deleted'] == []
    assert len(FakeBackend.events) == 3
    repeat = backup.scheduled_backup(environment.root, environment.output, environment.index,
        layout='version-directory', operation_id='selfhost-test', factory=FakeBackend, live=())
    assert repeat['event'] == 'verified_existing' and repeat['name'] == result['name']
    assert len(FakeBackend.events) == 3
    marker = json.loads((environment.output / result['name'] / 'AUTO_BACKUP.json').read_bytes())
    assert marker['operation_id'] == 'selfhost-test'


def test_unfinished_cloud_transaction_is_not_silently_reused(local_set):
    environment, folder = local_set
    (environment.output / contract.TRANSITION_FILE).write_text(json.dumps({
        'format': 1, 'kind': 'selfhost_backup_transition', 'sets': [{
            'name': folder.name, 'manifest_sha256': policy.digest(folder / 'SHA256SUMS')}]}))
    (environment.output / 'active-backup.json').write_text(json.dumps({
        'format': 1, 'name': '20261012T020000Z', 'operation_id': 'daily-2026-10-12', 'attempt': 1}))
    with pytest.raises(backup.BackupError, match='must_be_archived'):
        backup.scheduled_backup(environment.root, environment.output, environment.index,
            layout='version-directory', factory=FakeBackend, live=())


def test_dispatch_keeps_cloud_source_until_private_config_installed(local_set, tmp_path):
    environment, _ = local_set
    config = tmp_path / 'SELFHOST_BACKUP_CONFIG.private.json'
    args = SimpleNamespace(root=environment.output, operation_id='isolated-test', runtime_index=None)
    commands = []
    assert dispatch.dispatch(args, config=config, legacy=lambda cmd: (
        commands.append(cmd) or SimpleNamespace(returncode=0))) == 0
    assert commands[0][-2:] == ['--operation-id', 'isolated-test']
    assert 'run' in commands[0] and commands[0][1].endswith('ops_backup_policy.py')
    (environment.root / 'PREPARED').touch()
    config.write_text(json.dumps({'format': 1, 'root': str(environment.root),
        'runtime_index': str(environment.index), 'storage_layout': 'version-directory'}))
    config.chmod(0o600)
    calls = []
    assert dispatch.dispatch(args, config=config, legacy=lambda _: pytest.fail('selfhost config used cloud'),
        scheduled=lambda *a, **k: (calls.append((a, k)) or {'event': 'isolated-no-backup'})) == 0
    assert calls[0][0] == (environment.root, environment.output, environment.index)
    assert calls[0][1]['operation_id'] == 'isolated-test'


def test_dispatch_refuses_unprepared_or_overridden_source(local_set, tmp_path):
    environment, _ = local_set
    config = tmp_path / 'SELFHOST_BACKUP_CONFIG.private.json'
    config.write_text(json.dumps({'format': 1, 'root': str(environment.root),
        'runtime_index': str(environment.index), 'storage_layout': 'version-directory'}))
    config.chmod(0o600)
    args = SimpleNamespace(root=environment.output, operation_id=None, runtime_index=None)
    with pytest.raises(ValueError, match='prepared_selfhost_root_required'):
        dispatch.dispatch(args, config=config)
    (environment.root / 'PREPARED').touch(); args.runtime_index = tmp_path / 'another-index'
    with pytest.raises(ValueError, match='configured_runtime_index_required'):
        dispatch.dispatch(args, config=config)


def test_transition_cannot_protect_unknown_or_rewritten_manifest(local_set):
    environment, folder = local_set
    marker = environment.output / contract.TRANSITION_FILE
    marker.write_text(json.dumps({'format': 1, 'kind': 'selfhost_backup_transition',
        'sets': [{'name': folder.name, 'manifest_sha256': '0' * 64}]}))
    with pytest.raises(policy.PolicyError, match='transition_protection_manifest_mismatch'):
        policy.plan(environment.output)


def test_linux_transition_pool_excludes_exact_protected_manifests(local_set):
    environment, folder = local_set
    marker = environment.output / contract.TRANSITION_FILE
    marker.write_text(json.dumps({'format': 1, 'kind': 'selfhost_backup_transition',
        'sets': [{'name': folder.name, 'manifest_sha256': policy.digest(folder / 'SHA256SUMS')}]}))
    proposal = policy.plan(environment.output)
    assert proposal['transition_protected_sets'] == 1
    assert proposal['retention_pool_sets'] == 0
    assert proposal['candidates'] == []
