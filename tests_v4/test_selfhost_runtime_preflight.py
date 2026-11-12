"""Regression for a new deployed image missing from the immutable backup index.

Only temporary fixtures and fake read-only Docker replies. No VPS, DB or real
backup directory is accessed. Archive fixtures test bytes, not PG recovery.
"""
import copy
import ast
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'deploy-vps'))
import ops_backup_dispatch as dispatcher
import ops_backup_policy as policy
import ops_runtime_closure_repair as repair
import ops_selfhost_backup as backup
import ops_selfhost_runtime_preflight as preflight

OLD = 'sha256:' + 'a' * 64
HISTORIC = 'sha256:' + 'c' * 64
CONFIG = backup.canonical({'architecture': 'amd64', 'os': 'linux', 'rootfs': {
    'type': 'layers', 'diff_ids': ['sha256:' + hashlib.sha256(b'synthetic-layer').hexdigest()]}})
NEW = 'sha256:' + hashlib.sha256(CONFIG).hexdigest()
SECRET = 'synthetic-private-not-for-public-logs'
ROLES = ['api', 'db', 'gateway', 'rest', 'storage', 'v3', 'worker']


def test_systemd_entry_uses_selfhost_dispatcher_and_retains_same_root():
    text = (Path(__file__).parents[1] / 'deploy-vps' / 'ops_backup.sh').read_text()
    assert 'exec python3 "$script_dir/ops_backup_dispatch.py"' in text
    assert '--root "${ARBORSCAN_BACKUP_DIR:-/home/arborscan/ops-backups}" "$@"' in text
    assert 'ops_backup_policy.py" run' not in text


@pytest.mark.parametrize('mutation', [
    'tampered', 'truncated', 'wrong_os', 'wrong_arch', 'rootfs_missing',
    'rootfs_type', 'diff_id_count', 'diff_id_duplicate', 'layer_duplicate',
    'layer_symlink', 'duplicate_tar_name',
])
def test_classic_saved_image_requires_platform_and_exact_uncompressed_layers(tmp_path, mutation):
    layer = b'original-uncompressed-layer'
    config = {'architecture': 'amd64', 'os': 'linux',
              'rootfs': {'type': 'layers', 'diff_ids': ['sha256:' + hashlib.sha256(layer).hexdigest()]}}
    layers = ['layer.tar']
    actual = layer
    if mutation == 'tampered':
        actual = b'changed-with-same-config'
    elif mutation == 'truncated':
        actual = layer[:-2]
    elif mutation == 'wrong_os':
        config['os'] = 'windows'
    elif mutation == 'wrong_arch':
        config['architecture'] = 'arm64'
    elif mutation == 'rootfs_missing':
        config.pop('rootfs')
    elif mutation == 'rootfs_type':
        config['rootfs']['type'] = 'other'
    elif mutation == 'diff_id_count':
        config['rootfs']['diff_ids'] = []
    elif mutation == 'diff_id_duplicate':
        config['rootfs']['diff_ids'] *= 2
        layers = ['layer.tar', 'second.tar']
    elif mutation == 'layer_duplicate':
        layers *= 2
        config['rootfs']['diff_ids'] *= 2
    raw = backup.canonical(config)
    image = 'sha256:' + hashlib.sha256(raw).hexdigest()
    path = tmp_path / 'classic.tar'
    with tarfile.open(path, 'w') as target:
        values = {'config.json': raw, 'layer.tar': actual,
                  'manifest.json': backup.canonical([{'Config': 'config.json', 'Layers': layers}])}
        if mutation == 'diff_id_duplicate':
            values['second.tar'] = layer
        for name, content in values.items():
            item = tarfile.TarInfo(name)
            item.size = len(content)
            if name == 'layer.tar' and mutation == 'layer_symlink':
                item.type = tarfile.SYMTYPE; item.linkname = 'config.json'; item.size = 0
                target.addfile(item)
            else:
                target.addfile(item, io.BytesIO(content))
                if name == 'layer.tar' and mutation == 'duplicate_tar_name':
                    target.addfile(item, io.BytesIO(content))
    with pytest.raises(backup.BackupError, match='saved_image_'):
        backup.verify_saved_image(path, image)


def test_classic_saved_image_valid_layer_stream(tmp_path):
    layer = b'valid-uncompressed-layer'
    config = backup.canonical({'architecture': 'amd64', 'os': 'linux',
                              'rootfs': {'type': 'layers', 'diff_ids': ['sha256:' + hashlib.sha256(layer).hexdigest()]}})
    path = tmp_path / 'valid-classic.tar'
    with tarfile.open(path, 'w') as target:
        for name, raw in {'config.json': config, 'layer.tar': layer,
                          'manifest.json': backup.canonical([{'Config': 'config.json', 'Layers': ['layer.tar']}])}.items():
            item = tarfile.TarInfo(name); item.size = len(raw); target.addfile(item, io.BytesIO(raw))
    backup.verify_saved_image(path, 'sha256:' + hashlib.sha256(config).hexdigest())


@pytest.mark.parametrize('case', ['no_budget', 'write_budget', 'concurrent_space', 'post_space', 'success'])
def test_runtime_export_retains_margin_and_preserves_partial(tmp_path, case):
    reserve = 4 * 1024**3
    budget = 1024
    values = [reserve + budget] * 4
    if case == 'no_budget':
        values[0] -= 1
    elif case == 'concurrent_space':
        values[1] = reserve
    elif case == 'post_space':
        values[2] = reserve - 1
    calls = []
    stream = io.BytesIO(b'x' * (budget + 1 if case == 'write_budget' else 128))
    process = SimpleNamespace(stdout=stream, poll=lambda: 0,
                              wait=lambda timeout: 0, terminate=lambda: calls.append('terminated'))
    def popen(command, **kwargs):
        calls.append(command)
        return process
    def free(_):
        return SimpleNamespace(free=values.pop(0) if values else reserve + budget)
    path = tmp_path / 'archive.partial'
    if case == 'success':
        assert backup.stream_saved_image(NEW, path, budget=budget, reserve=reserve, free=free, popen=popen) == 128
        assert path.read_bytes() == b'x' * 128
    else:
        with pytest.raises(backup.BackupError, match='runtime_space|runtime_export_budget'):
            backup.stream_saved_image(NEW, path, budget=budget, reserve=reserve, free=free, popen=popen)
        if case == 'no_budget':
            assert not path.exists() and not calls
        else:
            assert path.exists()
            assert path.stat().st_size == (128 if case == 'post_space' else 0)
    assert not path.with_suffix('').exists()


def image_archive(path, config=CONFIG):
    values = {'config.json': config, 'layer.tar': b'synthetic-layer',
              'manifest.json': json.dumps([{'Config': 'config.json', 'Layers': ['layer.tar']}]).encode()}
    with tarfile.open(path, 'w') as target:
        for name, raw in values.items():
            item = tarfile.TarInfo(name)
            item.size = len(raw)
            target.addfile(item, io.BytesIO(raw))


@pytest.fixture
def env(tmp_path):
    home = tmp_path / 'private'
    home.mkdir()
    root = home / 'selfhost-20261012T140855Z'
    root.mkdir()
    (root / 'compose.private.json').write_text(json.dumps({
        'services': {role: {'image': OLD} for role in ROLES}}))
    assets = home / 'ops-runtime'
    assets.mkdir()
    old = assets / 'old.archive'
    old.write_bytes(b'previous-verified-runtime')
    historic = assets / 'historic.archive'
    historic.write_bytes(b'other-historical-runtime')
    runtime = {'format': 1, 'images': {
        OLD: [{'path': str(old), 'sha256': repair.file_sha(old)}],
        HISTORIC: [{'path': str(historic), 'sha256': repair.file_sha(historic)}]}}
    index = assets / 'RUNTIME_INDEX.json'
    index.write_text(json.dumps(runtime))
    rows = [{'Name': '/selfhost-' + role, 'Image': NEW if role == 'api' else OLD,
             'Config': {'Labels': {'com.docker.compose.service': role,
                                  'com.docker.compose.project': 'selfhost-fixture'}},
             'State': {'Running': role != 'worker'}} for role in ROLES]
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if command[:2] == ['docker', 'compose']:
            return SimpleNamespace(returncode=0, stdout=('\n'.join(f'{i:064x}' for i in range(1, 8))).encode())
        if command[:2] == ['docker', 'inspect']:
            return SimpleNamespace(returncode=0, stdout=json.dumps(rows).encode())
        raise AssertionError('Mutating or unexpected command')
    return SimpleNamespace(home=home, root=root, assets=assets, index=index,
                           runtime=runtime, rows=rows, run=run, calls=calls,
                           old=old, historic=historic)


def additions(env):
    archive = env.assets / 'new-api.tar'
    image_archive(archive)
    return {NEW: {'path': str(archive), 'sha256': repair.file_sha(archive)}}


@pytest.mark.parametrize('size', [None, True, -1, 3 * 1024**3])
def test_runtime_index_refuses_unknown_or_unaffordable_export_before_start(env, monkeypatch, size):
    monkeypatch.setattr(backup, 'PRIVATE_HOME', env.home)
    compose = json.loads((env.root / 'compose.private.json').read_bytes())
    compose['services']['api']['image'] = NEW
    (env.root / 'compose.private.json').write_text(json.dumps(compose))
    calls = []
    def run(command, **kwargs):
        assert command[:3] == ['docker', 'image', 'inspect']
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=(json.dumps(size) if command[-1] == '{{json .Size}}' else command[3]).encode())
    def export(*args, **kwargs):
        raise AssertionError('Refusal must precede Docker save')
    target = env.assets / env.root.name
    with pytest.raises(backup.BackupError, match='runtime_image_size_unknown|insufficient_runtime_space'):
        backup.create_runtime_index(env.root, target, env.index, run=run, export=export,
                                    free=lambda _: SimpleNamespace(free=5 * 1024**3))
    assert not (target / 'RUNTIME_INDEX.json').exists()
    assert not list(target.glob('*.partial'))
    assert calls


def test_live_images_override_stale_base_compose_and_fail_before_copy(env):
    before = env.index.read_bytes()
    with pytest.raises(preflight.RuntimePreflightError, match='runtime_archive_index_missing'):
        preflight.preflight(env.root, env.index, private_home=env.home, run=env.run)
    assert env.index.read_bytes() == before
    assert len(env.calls) == 2
    assert all('start' not in c and 'up' not in c and 'save' not in c and 'pull' not in c for c in env.calls)


def test_presence_check_keeps_full_integrity_verification_separate(env):
    runtime = copy.deepcopy(env.runtime)
    runtime['images'][NEW] = runtime['images'][OLD]
    result = preflight.coverage(env.rows, runtime, private_home=env.home)
    assert result['services'] == 7 and result['integrity_verified'] is False
    assert env.rows[-1]['State']['Running'] is False  # Deliberately stopped worker is valid.


def test_dotdot_cannot_escape_private_home(env):
    with pytest.raises(preflight.RuntimePreflightError, match='unsafe_runtime_preflight_path'):
        preflight.private_path(env.assets / '..' / '..' / 'outside.json', env.home)


def test_snapshot_identity_handles_windows_birthtime_without_weakening_linux_ctime():
    node = next(n for n in ast.parse(backup.FILE_HELPER).body
                if isinstance(n, ast.FunctionDef) and n.name == 'identity')
    namespace = {'os': SimpleNamespace(name='nt')}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<snapshot-identity>', 'exec'), namespace)
    before = SimpleNamespace(st_dev=1, st_ino=2, st_size=3, st_mtime_ns=20,
                             st_ctime_ns=10, st_birthtime_ns=10)
    handle = SimpleNamespace(**{**vars(before), 'st_ctime_ns': 20})
    assert namespace['identity'](before) == namespace['identity'](handle)
    changed = SimpleNamespace(**{**vars(handle), 'st_mtime_ns': 21})
    assert namespace['identity'](before) != namespace['identity'](changed)
    namespace['os'] = SimpleNamespace(name='posix')
    assert namespace['identity'](before) != namespace['identity'](handle)


@pytest.mark.parametrize('change,code', [
    ('missing_worker', 'invalid_selfhost_runtime_inventory'),
    ('duplicate_role', 'invalid_selfhost_runtime_inventory'),
    ('other_project', 'invalid_selfhost_runtime_inventory'),
    ('empty_assets', 'runtime_archive_missing'),
    ('bad_sha', 'invalid_runtime_archive_digest'),
    ('missing_file', 'runtime_preflight_file_missing'),
    ('outside_root', 'unsafe_runtime_preflight_path'),
])
def test_invalid_or_missing_runtime_fails_closed(env, change, code):
    runtime = copy.deepcopy(env.runtime)
    runtime['images'][NEW] = copy.deepcopy(runtime['images'][OLD])
    rows = copy.deepcopy(env.rows)
    if change == 'missing_worker':
        rows.pop()
    elif change == 'duplicate_role':
        rows[0]['Config']['Labels']['com.docker.compose.service'] = 'db'
    elif change == 'other_project':
        rows[0]['Config']['Labels']['com.docker.compose.project'] = 'other'
    elif change == 'empty_assets':
        runtime['images'][NEW] = []
    elif change == 'bad_sha':
        runtime['images'][NEW][0]['sha256'] = SECRET
    elif change == 'missing_file':
        runtime['images'][NEW][0]['path'] = str(env.assets / 'absent')
    elif change == 'outside_root':
        runtime['images'][NEW][0]['path'] = str(env.home.parent / 'outside')
    with pytest.raises(preflight.RuntimePreflightError, match=code):
        preflight.coverage(rows, runtime, private_home=env.home)


def test_preflight_failure_precedes_inventory_attempt_increment_and_snapshot(tmp_path, monkeypatch):
    root = tmp_path / 'backups'
    root.mkdir()
    active = {'format': 1, 'name': '20261105T033653Z',
              'operation_id': 'selfhost-daily-2026-11-05', 'attempt': 8}
    (root / 'active-backup.json').write_text(json.dumps(active))
    before = (root / 'active-backup.json').read_bytes()
    calls = []
    def refuse():
        raise preflight.RuntimePreflightError('runtime_archive_index_missing')
    monkeypatch.setattr(policy, 'inventory', lambda *a, **k: calls.append('inventory'))
    with pytest.raises(preflight.RuntimePreflightError, match='runtime_archive_index_missing'):
        policy.run_backup(root, lambda stage: calls.append('snapshot'), root / 'index',
                          operation_id='selfhost-daily-2026-11-12', before_create=refuse)
    assert calls == []
    assert (root / 'active-backup.json').read_bytes() == before
    assert not (root / '.staging').exists()


def test_completed_replay_does_not_require_current_live_image_archive(tmp_path, monkeypatch):
    root = tmp_path / 'backups'
    root.mkdir()
    folder = root / '20261102T033109Z'
    folder.mkdir()
    (folder / 'AUTO_BACKUP.json').write_text(json.dumps({'operation_id': 'selfhost-existing'}))
    monkeypatch.setattr(policy, 'inventory', lambda *a, **k: [{'name': folder.name, 'valid': True}])
    monkeypatch.setattr(policy, 'verify_set', lambda *a, **k: {'manifest_sha256': 'a' * 64})
    monkeypatch.setattr(policy, 'plan', lambda *a, **k: {})
    monkeypatch.setattr(policy, 'rotate', lambda *a, **k: [])
    monkeypatch.setattr(policy, 'record_success', lambda *a, **k: None)
    def must_not_run():
        raise AssertionError('Historical replay must use its own frozen closure')
    result = policy.run_backup(root, must_not_run, root / 'index',
                               operation_id='selfhost-existing', before_create=must_not_run)
    assert result['event'] == 'verified_existing'


def test_fake_replay_marker_cannot_bypass_new_create_preflight(tmp_path, monkeypatch):
    root = tmp_path / 'backups'
    root.mkdir()
    folder = root / '20261102T033109Z'
    folder.mkdir()
    (folder / 'AUTO_BACKUP.json').write_text(json.dumps({'operation_id': 'selfhost-existing'}))
    monkeypatch.setattr(policy, 'inventory', lambda *a, **k: [])
    def refuse():
        raise preflight.RuntimePreflightError('runtime_archive_index_missing')
    with pytest.raises(preflight.RuntimePreflightError, match='runtime_archive_index_missing'):
        policy.run_backup(root, lambda stage: pytest.fail('snapshot must not start'), root / 'index',
                          operation_id='selfhost-existing', before_create=refuse)
    assert not (root / '.staging').exists()


def test_actual_backend_capture_is_rechecked_before_database_snapshot(env, monkeypatch):
    events = []
    monkeypatch.setattr(backup, 'PRIVATE_HOME', env.home)
    class Backend:
        def __init__(self, root, stage):
            self.rows = env.rows
        def snapshot(self):
            events.append('snapshot')
            raise AssertionError('No database/files copy for missing image')
    with pytest.raises(backup.BackupError, match='runtime_archive_index_missing'):
        backup.build_stage(env.root, env.assets, env.index, layout='version-directory', factory=Backend)
    assert events == []


@pytest.mark.parametrize('message,expected', [
    ('runtime_archive_index_missing', 'runtime_archive_index_missing'),
    (SECRET + '?password=private', 'private_validation_or_runtime_error'),
])
def test_dispatch_logs_only_known_safe_failure_codes(monkeypatch, capsys, message, expected):
    monkeypatch.setattr(sys, 'argv', ['backup'])
    def fail(*args, **kwargs):
        raise ValueError(message)
    monkeypatch.setattr(dispatcher, 'dispatch', fail)
    assert dispatcher.main() == 1
    text = capsys.readouterr().err
    assert json.loads(text)['reason'] == expected
    assert SECRET not in text


def test_repair_uses_real_image_identity_and_preserves_all_old_mappings(env):
    before = env.index.read_bytes()
    raw, plan = repair.build_candidate(env.index, env.rows, additions(env), private_home=env.home)
    candidate = json.loads(raw)
    assert plan['missing_image_ids'] == [NEW] and plan['integrity_verified'] is True
    assert candidate['images'][OLD] == env.runtime['images'][OLD]
    assert candidate['images'][HISTORIC] == env.runtime['images'][HISTORIC]
    assert candidate['service_images']['api'] == NEW
    assert env.index.read_bytes() == before
    assert plan['configuration_changed'] is False and plan['backup_started'] is False


@pytest.mark.parametrize('change,code', [
    ('outer_sha', 'replacement_archive_checksum_mismatch'),
    ('wrong_image', 'replacement_archive_image_mismatch'),
    ('replace_old_mapping', 'exact_missing_image_archives_required'),
    ('historic_corruption', 'runtime_archive_checksum_mismatch'),
])
def test_bad_repair_never_changes_source_index(env, change, code):
    values = additions(env)
    before = env.index.read_bytes()
    if change == 'outer_sha':
        values[NEW]['sha256'] = '0' * 64
    elif change == 'wrong_image':
        image_archive(Path(values[NEW]['path']), b'wrong-image')
        values[NEW]['sha256'] = repair.file_sha(values[NEW]['path'])
    elif change == 'replace_old_mapping':
        values[OLD] = values[NEW]
    elif change == 'historic_corruption':
        env.historic.write_bytes(b'changed')
    with pytest.raises(repair.RepairError, match=code):
        repair.build_candidate(env.index, env.rows, values, private_home=env.home)
    assert env.index.read_bytes() == before


def test_candidate_is_new_private_artifact_not_config_switch(env):
    before = env.index.read_bytes()
    destination = env.assets / 'repair-fixture'
    result = repair.write_candidate(env.index, env.rows, additions(env), destination, private_home=env.home)
    assert repair.file_sha(destination / 'RUNTIME_INDEX.json') == result['candidate_index_sha256']
    assert (destination / 'REPAIR_PLAN.json').is_file()
    assert env.index.read_bytes() == before
    published = (destination / 'RUNTIME_INDEX.json').read_bytes()
    with pytest.raises(repair.RepairError, match='new_private_candidate_directory_required'):
        repair.write_candidate(env.index, env.rows, {}, destination, private_home=env.home)
    assert (destination / 'RUNTIME_INDEX.json').read_bytes() == published


def test_index_change_during_archive_verification_refuses_candidate(env, monkeypatch):
    values = additions(env)
    original = repair.file_sha
    triggered = False
    def changed(path):
        nonlocal triggered
        value = original(path)
        if Path(path) == Path(values[NEW]['path']) and not triggered:
            triggered = True
            env.index.write_bytes(env.index.read_bytes() + b' ')
        return value
    monkeypatch.setattr(repair, 'file_sha', changed)
    destination = env.assets / 'repair-race'
    with pytest.raises(repair.RepairError, match='source_runtime_index_changed'):
        repair.write_candidate(env.index, env.rows, values, destination, private_home=env.home)
    assert not destination.exists()
