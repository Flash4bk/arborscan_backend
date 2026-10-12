"""Isolated restore orchestration fixtures; real SQL/Linux restore is separate.

Commands are captured, not executed. The fixtures verify that equal row counts
cannot conceal changed sessions, that native restore targets are freshly named,
and that failures never start an API/worker or delete copied jobs/sessions.
"""
import contextlib
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'deploy-vps'))
import ops_selfhost_backup as backup
import ops_selfhost_restore_check as check
from test_ops_selfhost_backup import environment, FakeBackend, inventory


TABLES = {'auth_sessions': [{'id': 'synthetic-session', 'token_hash': 'synthetic-only'}],
          'ml_jobs': [{'id': 'synthetic-job', 'state': 'queued'}], 'report_versions': []}


class Backend(FakeBackend):
    @contextlib.contextmanager
    def snapshot(self):
        yield SimpleNamespace(identifier='00000003-0000001B-1', storage=inventory,
                              tables=lambda: TABLES)


@pytest.fixture
def fixture(environment, monkeypatch):
    result = backup.backup(environment.root, environment.output, environment.index,
        layout='version-directory', stamp='20261012T170000Z', factory=Backend,
        free=lambda _: SimpleNamespace(free=10 * 1024**3))
    monkeypatch.setattr(check, 'PRIVATE_HOME', environment.root.parent)
    monkeypatch.setattr(check, 'BACKUP_ROOT', environment.output)
    # Native roles in this synthetic fixture contain no actual account values.
    monkeypatch.setattr(check, 'role_restore_sql', lambda _: '/* synthetic roles only */')
    return environment, environment.output / result['name']


class Runner:
    def __init__(self, *, mismatch=None, volume_exists=False, native_failure=False):
        self.commands = []; self.mismatch = mismatch
        self.volume_exists = volume_exists; self.native_failure = native_failure
        self.dump_bytes = None

    def __call__(self, args, *, input, stdin, **kw):
        self.commands.append((args, input))
        raw = b''; code = 0
        if args[1:3] == ['volume', 'inspect']:
            code = 0 if self.volume_exists else 1
        elif 'pg_restore' in args:
            self.dump_bytes = stdin.read(); code = int(self.native_failure)
        elif 'psql' in args and input:
            sql = input.decode()
            if 'FROM public.' in sql:
                name = re.search(r'public\."([A-Za-z_]+)"', sql).group(1)
                rows = TABLES[name]
                if self.mismatch == name:
                    rows = [{**rows[0], 'token_hash': 'changed'}]
                raw = backup.canonical(rows)
            elif sql == backup.STORAGE_SQL:
                data = inventory(version='changed' if self.mismatch == 'storage' else 'generation-a')
                raw = backup.canonical(data)
            elif 'json_build_array' in sql:
                raw = backup.canonical([1, 1, 1])
        elif args[1] == 'run' and '--rm' in args:
            raw = backup.canonical({'files_restored': 1, 'directories_restored': 1,
                'ownership_restored': True, 'xattrs_restored': True})
        return subprocess.CompletedProcess(args, code, raw)


def test_native_and_raw_commands_preserve_sessions_jobs_without_starting_worker(fixture, capsys):
    env, folder = fixture; run = Runner()
    result = check.check(folder, env.root / 'backup-restore-checks', run=run,
                         sleep=lambda _: pytest.fail('unexpected wait'))
    assert result['native_sql_restore'] is True
    assert result['restored_backup'] == str(folder) and result['restore_verified'] is True
    assert result['backup_manifest_sha256'] == hashlib.sha256((folder / 'SHA256SUMS').read_bytes()).hexdigest()
    assert result['auth_sessions_rows_matched'] == result['quality_jobs_rows_matched'] == 1
    assert result['migration_versions'] == [1, 1, 1]
    assert run.dump_bytes == (folder / 'postgres/database.dump').read_bytes()
    native = next(cmd for cmd, _ in run.commands if 'pg_restore' in cmd)
    assert '--single-transaction' in native and '--exit-on-error' in native
    starts = [cmd for cmd, _ in run.commands if cmd[1] == 'run']
    assert len(starts) == 3 and all(cmd[cmd.index('--network') + 1] == 'none' for cmd in starts)
    assert not any('--env-file' in cmd or '-e' in cmd for cmd in starts)
    assert run.commands[-1][0][:2] == ['docker', 'stop']
    assert len([cmd for cmd, _ in run.commands if cmd[1:3] == ['volume', 'create']]) == 2
    assert not any('rm' in cmd[1:3] or 'start' in cmd or 'up' in cmd for cmd, _ in run.commands)
    all_sql = b'\n'.join(raw for cmd, raw in run.commands if 'psql' in cmd and raw)
    assert b'DELETE' not in all_sql and b'UPDATE' not in all_sql
    assert 'synthetic-session' not in capsys.readouterr().out
    assert len(list((env.root / 'backup-restore-checks').glob('*/result.json'))) == 1


@pytest.mark.parametrize('failure', ['auth_sessions', 'storage', 'native'])
def test_failed_native_or_same_count_data_mismatch_stops_only_new_db(fixture, failure):
    env, folder = fixture
    run = Runner(mismatch=failure, native_failure=failure == 'native')
    with pytest.raises(RuntimeError):
        check.check(folder, env.root / 'backup-restore-checks', run=run, sleep=lambda _: None)
    assert run.commands[-1][0][:2] == ['docker', 'stop']
    started = next(cmd for cmd, _ in run.commands if cmd[1:3] == ['run', '-d'])
    assert run.commands[-1][0][-1] == started[started.index('--name') + 1]
    assert not any(cmd[1:3] == ['volume', 'create'] for cmd, _ in run.commands)
    assert not list((env.root / 'backup-restore-checks').glob('*/result.json'))


def test_restore_refuses_existing_volume_before_extracting(fixture):
    env, folder = fixture; run = Runner(volume_exists=True)
    with pytest.raises(RuntimeError, match='existing_restore_volume_preserved'):
        check.check(folder, env.root / 'backup-restore-checks', run=run, sleep=lambda _: None)
    assert not any(cmd[1:3] == ['volume', 'create'] for cmd, _ in run.commands)
    assert run.commands[-1][0][:2] == ['docker', 'stop']


def test_unpublished_or_production_path_is_refused_without_docker(fixture):
    env, folder = fixture; run = Runner()
    with pytest.raises(ValueError, match='isolated_evidence_root_required'):
        check.check(folder, env.root.parent / 'production/backup-restore-checks', run=run)
    (folder / 'COMPLETE').unlink()
    with pytest.raises(ValueError, match='verified_local_backup_required'):
        check.check(folder, env.root / 'backup-restore-checks', run=run)
    assert not run.commands


def test_invalid_table_rows_and_order_independent_equality():
    assert check.normalized_rows([{'id': 2}, {'id': 1}]) == check.normalized_rows([{'id': 1}, {'id': 2}])
    for value in ({}, [None], ['invalid']):
        with pytest.raises(ValueError):
            check.normalized_rows(value)
