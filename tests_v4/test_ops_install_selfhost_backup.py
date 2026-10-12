"""Private real-file install/rollback fixtures; no real service or Docker calls."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'deploy-vps'))
import ops_install_selfhost_backup as install
from test_ops_selfhost_backup import environment, FakeBackend


@pytest.fixture
def fixture(environment, monkeypatch):
    env = environment; private = env.root.parent
    monkeypatch.setattr(install, 'PRIVATE_HOME', private)
    monkeypatch.setattr(install, 'BACKUP_ROOT', env.output)
    old = env.output / '20260901T030000Z'
    (old / 'SHA256SUMS').write_text('synthetic prior manifest')
    (env.output / 'SELFHOST_TRANSITION_PROTECTION.private.json').write_text(json.dumps({
        'format': 1, 'kind': 'selfhost_backup_transition', 'sets': [{
            'name': old.name, 'manifest_sha256': hashlib.sha256((old / 'SHA256SUMS').read_bytes()).hexdigest()}]}))
    for name in ('PREPARED', 'PROMOTED'):
        (env.root / name).touch()
    (env.root / 'promotion.private.json').write_text(json.dumps({'root': str(env.root), 'phase': 'promoted'}))
    tools = private / 'ops-tools'; tools.mkdir()
    for name in install.FILES:
        (tools / name).write_bytes(('synthetic previous ' + name).encode())
    source = env.root / 'install-source'; source.mkdir()
    actual = Path(__file__).parents[1] / 'deploy-vps'
    for name in install.FILES:
        (source / name).write_bytes((actual / name).read_bytes())
    (source / 'INSTALL_SOURCE_SHA256.json').write_text(json.dumps({'format': 1,
        'files': {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in install.FILES}}))
    rows = FakeBackend(env.root, env.root).rows
    for row in rows:
        row['State'] = {'Running': row['Config']['Labels']['com.docker.compose.service'] != 'worker'}
    commands = []
    def run(args, **kwargs):
        commands.append(args)
        value = (b'inactive\n' if 'arborscan-backup.timer' in args else b'failed\n') if args[0] == 'systemctl' else b'fixture\n' if args[1] == 'compose' else json.dumps(rows).encode()
        return subprocess.CompletedProcess(args, 0, value, b'')
    return env, tools, source, rows, commands, run


def test_hash_bound_install_retains_prior_files_and_only_switches_entry(fixture):
    env, tools, source, _, commands, run = fixture
    before = {name: (tools / name).read_bytes() for name in install.FILES}
    old_backups = {p.relative_to(env.output).as_posix(): p.read_bytes() for p in env.output.rglob('*') if p.is_file()}
    result = install.install(env.root, source, env.index, run=run)
    retained = Path(result['rollback_directory'])
    assert all((retained / name).read_bytes() == raw for name, raw in before.items())
    assert all((tools / name).read_bytes() == (source / name).read_bytes() for name in install.FILES)
    config = json.loads((tools / install.CONFIG).read_bytes())
    assert config['root'] == str(env.root) and config['runtime_index'] == str(env.index)
    assert result['service_started'] is result['systemd_changed'] is result['backup_sets_changed'] is False
    assert not any('start' in command or 'stop' in command or 'up' in command for command in commands)
    assert old_backups == {p.relative_to(env.output).as_posix(): p.read_bytes()
                           for p in env.output.rglob('*') if p.is_file() and p.name != 'backup.lock'}
    undone = install.rollback(retained, run=run)
    assert undone['backup_sets_changed'] is False
    assert not (tools / install.CONFIG).exists()
    assert before == {p.name: p.read_bytes() for p in tools.iterdir()}


@pytest.mark.parametrize('guard', ['promotion', 'manifest', 'active_attempt', 'worker', 'service'])
def test_unsafe_install_preserves_existing_tools(fixture, guard):
    env, tools, source, rows, _, run = fixture
    before = {p.name: p.read_bytes() for p in tools.iterdir()}
    if guard == 'promotion':
        (env.root / 'PROMOTED').unlink()
    elif guard == 'manifest':
        (source / 'ops_backup_policy.py').write_bytes(b'changed after review')
    elif guard == 'active_attempt':
        (env.output / 'active-backup.json').write_text('{"operation_id":"cloud-old"}')
    elif guard == 'worker':
        next(row for row in rows if row['Config']['Labels']['com.docker.compose.service'] == 'worker')['State']['Running'] = True
    elif guard == 'service':
        original = run
        def run(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, b'active\n', b'') if args[0] == 'systemctl' else original(args, **kwargs)
    with pytest.raises(ValueError):
        install.install(env.root, source, env.index, run=run)
    assert before == {p.name: p.read_bytes() for p in tools.iterdir()}


def test_partial_write_failure_pauses_entry_and_preserves_exact_previous_tools(fixture, monkeypatch):
    env, tools, source, _, _, run = fixture
    before = {p.name: p.read_bytes() for p in tools.iterdir()}
    original = install.write
    def write(path, raw, mode=0o600):
        if path == tools / 'ops_backup_policy.py':
            raise OSError('synthetic write failure')
        original(path, raw, mode)
    monkeypatch.setattr(install, 'write', write)
    with pytest.raises(OSError):
        install.install(env.root, source, env.index, run=run)
    retained, = env.root.glob('backup-tools-rollback-*')
    assert (tools / install.ENTRY).read_bytes() == install.MAINTENANCE
    assert before == {name: (retained / name).read_bytes() for name in before}
    assert install.receipt(retained)['phase'] == 'paused'
    monkeypatch.setattr(install, 'write', original)
    result = install.install(env.root, source, env.index, run=run, resume_directory=retained)
    assert Path(result['rollback_directory']) == retained
    assert install.receipt(retained)['phase'] == 'installed'
    assert all((tools / name).read_bytes() == (source / name).read_bytes() for name in install.FILES)


def test_rollback_does_not_overwrite_a_later_tool_change(fixture):
    env, tools, source, _, _, run = fixture
    result = install.install(env.root, source, env.index, run=run)
    (tools / 'ops_backup_policy.py').write_bytes(b'newer unrelated operator change')
    before = {p.name: p.read_bytes() for p in tools.iterdir()}
    with pytest.raises(ValueError, match='newer_installed_tool_preserved'):
        install.rollback(result['rollback_directory'], run=run)
    assert before == {p.name: p.read_bytes() for p in tools.iterdir()}


def test_rollback_refuses_an_active_timer_without_changing_tools(fixture):
    env, tools, source, _, _, run = fixture
    result = install.install(env.root, source, env.index, run=run)
    before = {p.name: p.read_bytes() for p in tools.iterdir()}
    original = run
    def run(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, b'active\n', b'') if 'arborscan-backup.timer' in args else original(args, **kwargs)
    with pytest.raises(ValueError, match='timer_must_be_stopped_before_legacy_tool_rollback'):
        install.rollback(result['rollback_directory'], run=run)
    assert before == {p.name: p.read_bytes() for p in tools.iterdir()}


@pytest.mark.parametrize('target', ['maintenance', *install.FILES, install.CONFIG])
def test_abrupt_install_interruption_after_atomic_effect_can_resume(fixture, monkeypatch, target):
    env, tools, source, _, commands, run = fixture
    old = {name: (tools / name).read_bytes() for name in install.FILES}
    original = install.write
    def interrupted(path, raw, mode=0o600):
        original(path, raw, mode)
        if path.parent == tools and ((target == 'maintenance' and raw == install.MAINTENANCE)
            or (target != 'maintenance' and path.name == target and raw != install.MAINTENANCE)):
            raise KeyboardInterrupt('synthetic process killed after atomic effect')
    monkeypatch.setattr(install, 'write', interrupted)
    with pytest.raises(KeyboardInterrupt): install.install(env.root, source, env.index, run=run)
    retained, = env.root.glob('backup-tools-rollback-*')
    assert old == {name: (retained / name).read_bytes() for name in old}
    state = install.receipt(retained)
    assert state['phase'] == 'installing'
    if target != install.ENTRY:
        assert (tools / install.ENTRY).read_bytes() == install.MAINTENANCE
    before = {p.name: p.read_bytes() for p in tools.iterdir()}
    with pytest.raises(ValueError, match='pending_tool_install_requires_resume_or_rollback'):
        install.install(env.root, source, env.index, run=run)
    assert before == {p.name: p.read_bytes() for p in tools.iterdir()}
    monkeypatch.setattr(install, 'write', original)
    result = install.install(env.root, source, env.index, run=run, resume_directory=retained)
    assert Path(result['rollback_directory']) == retained
    assert install.receipt(retained)['phase'] == 'installed'
    assert all((tools / name).read_bytes() == (source / name).read_bytes() for name in install.FILES)
    assert not any('start' in cmd or 'stop' in cmd for cmd in commands)


@pytest.mark.parametrize('target', ['maintenance', *install.FILES])
def test_abrupt_rollback_interruption_is_resumable_with_old_new_or_paused_entry(fixture, monkeypatch, target):
    env, tools, source, _, _, run = fixture
    before = {p.name: p.read_bytes() for p in tools.iterdir()}
    result = install.install(env.root, source, env.index, run=run)
    retained = Path(result['rollback_directory'])
    original = install.write
    def interrupted(path, raw, mode=0o600):
        original(path, raw, mode)
        if path.parent == tools and ((target == 'maintenance' and raw == install.MAINTENANCE)
            or (target != 'maintenance' and path.name == target and raw != install.MAINTENANCE)):
            raise KeyboardInterrupt('synthetic rollback killed after atomic effect')
    monkeypatch.setattr(install, 'write', interrupted)
    with pytest.raises(KeyboardInterrupt): install.rollback(retained, run=run)
    assert install.receipt(retained)['phase'] == 'rolling_back'
    monkeypatch.setattr(install, 'write', original)
    install.rollback(retained, run=run)
    assert install.receipt(retained)['phase'] == 'rolled_back'
    assert before == {p.name: p.read_bytes() for p in tools.iterdir()}
    install.rollback(retained, run=run)
    assert before == {p.name: p.read_bytes() for p in tools.iterdir()}


def test_rollback_after_new_config_unlink_can_resume(fixture, monkeypatch):
    env, tools, source, _, _, run = fixture
    before = {p.name: p.read_bytes() for p in tools.iterdir()}
    result = install.install(env.root, source, env.index, run=run)
    retained = Path(result['rollback_directory'])
    original = Path.unlink
    def interrupted(path, *args, **kwargs):
        original(path, *args, **kwargs)
        if path == tools / install.CONFIG:
            raise KeyboardInterrupt('synthetic process killed after unlink effect')
    monkeypatch.setattr(Path, 'unlink', interrupted)
    with pytest.raises(KeyboardInterrupt): install.rollback(retained, run=run)
    assert not (tools / install.CONFIG).exists()
    monkeypatch.setattr(Path, 'unlink', original)
    install.rollback(retained, run=run)
    assert before == {p.name: p.read_bytes() for p in tools.iterdir()}


def test_paused_install_can_rollback_without_assuming_all_files_are_new(fixture, monkeypatch):
    env, tools, source, _, _, run = fixture
    before = {p.name: p.read_bytes() for p in tools.iterdir()}
    original = install.write
    def interrupted(path, raw, mode=0o600):
        original(path, raw, mode)
        if path == tools / 'ops_selfhost_backup.py': raise KeyboardInterrupt('after third replacement')
    monkeypatch.setattr(install, 'write', interrupted)
    with pytest.raises(KeyboardInterrupt): install.install(env.root, source, env.index, run=run)
    retained, = env.root.glob('backup-tools-rollback-*')
    monkeypatch.setattr(install, 'write', original)
    install.rollback(retained, run=run)
    assert before == {p.name: p.read_bytes() for p in tools.iterdir()}


def test_paused_install_refuses_unrelated_changes_before_any_resume_write(fixture, monkeypatch):
    env, tools, source, _, _, run = fixture
    original = install.write
    def interrupted(path, raw, mode=0o600):
        original(path, raw, mode)
        if path == tools / 'ops_selfhost_backup.py': raise KeyboardInterrupt('after third replacement')
    monkeypatch.setattr(install, 'write', interrupted)
    with pytest.raises(KeyboardInterrupt): install.install(env.root, source, env.index, run=run)
    retained, = env.root.glob('backup-tools-rollback-*')
    (tools / 'ops_backup_policy.py').write_bytes(b'newer unrelated operator work')
    before = {p.name: p.read_bytes() for p in tools.iterdir()}
    monkeypatch.setattr(install, 'write', original)
    with pytest.raises(ValueError, match='newer_installed_tool_preserved'):
        install.install(env.root, source, env.index, run=run, resume_directory=retained)
    assert before == {p.name: p.read_bytes() for p in tools.iterdir()}


def test_entry_never_exposes_mixed_bundle_or_config_during_install(fixture, monkeypatch):
    env, tools, source, _, _, run = fixture
    original = install.write
    entry_events = []
    def monitored(path, raw, mode=0o600):
        if path.parent == tools and path.name != install.ENTRY:
            assert (tools / install.ENTRY).read_bytes() == install.MAINTENANCE
        if path == tools / install.ENTRY:
            entry_events.append(raw)
            if raw != install.MAINTENANCE:
                assert json.loads((tools / install.CONFIG).read_bytes())['root'] == str(env.root)
                assert all((tools / name).read_bytes() == (source / name).read_bytes()
                           for name in install.FILES if name != install.ENTRY)
        original(path, raw, mode)
    monkeypatch.setattr(install, 'write', monitored)
    install.install(env.root, source, env.index, run=run)
    assert entry_events == [install.MAINTENANCE, (source / install.ENTRY).read_bytes()]


def test_orphan_partial_before_atomic_replace_does_not_block_reviewed_resume(fixture, monkeypatch):
    env, tools, source, _, _, run = fixture
    original = install.os.replace
    def interrupted(partial, target):
        if target == tools / 'ops_backup_policy.py':
            raise KeyboardInterrupt('synthetic kill before replace')
        original(partial, target)
    monkeypatch.setattr(install.os, 'replace', interrupted)
    with pytest.raises(KeyboardInterrupt): install.install(env.root, source, env.index, run=run)
    retained, = env.root.glob('backup-tools-rollback-*')
    partial, = tools.glob('ops_backup_policy.py.install-partial-*')
    partial_bytes = partial.read_bytes()
    monkeypatch.setattr(install.os, 'replace', original)
    install.install(env.root, source, env.index, run=run, resume_directory=retained)
    assert partial.read_bytes() == partial_bytes
    assert all((tools / name).read_bytes() == (source / name).read_bytes() for name in install.FILES)


@pytest.mark.skipif(install.os.name != 'posix', reason='Exact Unix modes require a Unix filesystem')
def test_rollback_restores_modes_when_old_and_new_bytes_are_identical(fixture):
    env, tools, source, _, _, run = fixture
    target = tools / 'ops_backup_dispatch.py'
    target.write_bytes((source / target.name).read_bytes()); target.chmod(0o700)
    result = install.install(env.root, source, env.index, run=run)
    assert target.stat().st_mode & 0o777 == 0o600
    install.rollback(result['rollback_directory'], run=run)
    assert target.stat().st_mode & 0o777 == 0o700
