"""Guarded ops-tools-only install after the separate, verified API cutover.

Does not start/stop services, change systemd, clear active attempts or rotate data.
It retains exact previous tools/config privately. A fail-closed entry prevents a
timer from running mixed modules during installation. Each intent is durable;
interrupted installation can resume, and legacy rollback requires a stopped
timer. Installing this installer file alone has no production effect.
"""
import argparse
import ast
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys

from ops_backup_policy import locked
from ops_runtime_contract import transition_protection, selfhost_inventory
from ops_selfhost_backup import safe_path, PRIVATE_HOME, BACKUP_ROOT

FILES = ('ops_runtime_contract.py', 'ops_verify_restore.py', 'ops_selfhost_backup.py',
         'ops_backup_policy.py', 'ops_backup_dispatch.py', 'ops_backup.sh')
CONFIG = 'SELFHOST_BACKUP_CONFIG.private.json'
ENTRY = 'ops_backup.sh'
MAINTENANCE = b'#!/bin/sh\n# Retained original entry and durable installer receipt are private.\nprintf \'%s\\n\' \'{"event":"backup_tools_install_pending","backup_sets_changed":false}\' >&2\nexit 1\n'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def sync(directory):
    if os.name == 'posix':
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def write(path, raw, mode=0o600):
    path = safe_path(path)
    # A killed process can leave its own partial behind. A unique new temporary
    # file preserves that evidence without blocking a later reviewed retry.
    partial = path.with_name(path.name + '.install-partial-' + secrets.token_hex(8))
    with partial.open('xb') as stream:
        partial.chmod(mode)
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    os.replace(partial, path)
    sync(path.parent)


def receipt(retained):
    retained = safe_path(retained)
    if (retained.parent.parent != PRIVATE_HOME
            or not re.fullmatch(r'selfhost-\d{8}T\d{6}Z', retained.parent.name)
            or not re.fullmatch(r'backup-tools-rollback-\d{8}T\d{6}Z', retained.name)):
        raise ValueError('private_tool_rollback_required')
    state = json.loads(safe_path(retained / 'INSTALL_RECEIPT.private.json').read_bytes())
    if (state.get('format') != 2 or state.get('kind') != 'selfhost_backup_tools_install'
            or state.get('root') != str(retained.parent)
            or state.get('phase') not in ('prepared', 'installing', 'paused', 'installed', 'rolling_back', 'rolled_back')
            or safe_path(state['tools']) != PRIVATE_HOME / 'ops-tools'
            or set(state.get('installed_files_sha256', {})) != set(FILES) | {CONFIG}
            or set(state.get('previous_files', {})) | set(state.get('new_files', [])) != set(FILES) | {CONFIG}
            or set(state.get('previous_files', {})) & set(state.get('new_files', []))
            or ENTRY not in state.get('previous_files', {})
            or state.get('maintenance_entry_sha256') != digest(MAINTENANCE)):
        raise ValueError('exact_rollback_receipt_required')
    return state


def journal(retained, state, phase, step):
    state.update(phase=phase, step=step)
    write(retained / 'INSTALL_RECEIPT.private.json', json.dumps(state, sort_keys=True).encode())


def saved_files(retained, state):
    saved = {}
    for name, expected in state['previous_files'].items():
        raw = safe_path(retained / name).read_bytes()
        if (digest(raw) != expected.get('sha256') or type(expected.get('mode')) is not int
                or not 0 <= expected['mode'] <= 0o777):
            raise ValueError('retained_tool_digest_mismatch')
        saved[name] = (raw, expected['mode'])
    return saved


def reviewed_targets(tools, state):
    """Only exact old/new bytes or this operation's fixed paused entry are safe."""
    current = {}
    for name, expected in state['installed_files_sha256'].items():
        target = safe_path(tools / name)
        if not target.exists():
            if name not in state['new_files']:
                raise ValueError('previous_installed_tool_missing')
            current[name] = None
            continue
        if not target.is_file():
            raise ValueError('private_tool_target_required')
        actual = digest(target.read_bytes())
        old = state['previous_files'].get(name, {}).get('sha256')
        allowed = {expected, old}
        if name == ENTRY:
            allowed.add(state['maintenance_entry_sha256'])
        if actual not in allowed:
            raise ValueError('newer_installed_tool_preserved')
        current[name] = actual
    return current


def mode_matches(path, mode):
    return os.name != 'posix' or path.stat().st_mode & 0o777 == mode


def resume_locked(retained, state, replacements):
    tools = safe_path(state['tools'])
    saved_files(retained, state)
    current = reviewed_targets(tools, state)
    if state['phase'] not in ('prepared', 'installing', 'paused', 'installed'):
        raise ValueError('installation_phase_requires_rollback_review')
    if state['phase'] == 'installed':
        if (current != state['installed_files_sha256']
                or any(not mode_matches(tools / name, 0o700 if name == ENTRY else 0o600) for name in replacements)):
            raise ValueError('installed_bundle_changed')
        return
    # Pause the only systemd entry before touching any import or configuration.
    journal(retained, state, 'installing', 'entry_pause_pending')
    write(tools / ENTRY, MAINTENANCE, 0o700)
    journal(retained, state, 'installing', 'entry_paused')
    try:
        for name, raw in replacements.items():
            if name == ENTRY:
                continue
            journal(retained, state, 'installing', 'write_pending:' + name)
            if current[name] != state['installed_files_sha256'][name] or not mode_matches(tools / name, 0o600):
                write(tools / name, raw)
            journal(retained, state, 'installing', 'written:' + name)
        actual = reviewed_targets(tools, state)
        if any(actual[name] != state['installed_files_sha256'][name] for name in replacements if name != ENTRY):
            raise ValueError('complete_bundle_required_before_entry_switch')
        journal(retained, state, 'installing', 'entry_publish_pending')
        write(tools / ENTRY, replacements[ENTRY], 0o700)
        journal(retained, state, 'installed', 'complete')
    except Exception:
        # Retain a paused entry and all old/new bytes. Automatic legacy rollback
        # with an active timer could re-enable a policy without protection.
        journal(retained, state, 'paused', state.get('step', 'install_failure'))
        raise


def install(root, source, index, *, run=subprocess.run, tools=None, backup_root=None, resume_directory=None):
    root, source, index = safe_path(root), safe_path(source), safe_path(index)
    tools = safe_path(tools or PRIVATE_HOME / 'ops-tools')
    backup_root = safe_path(backup_root or BACKUP_ROOT)
    if (root.parent != PRIVATE_HOME or not re.fullmatch(r'selfhost-\d{8}T\d{6}Z', root.name)
            or not source.is_relative_to(root) or not index.is_relative_to(PRIVATE_HOME)
            or tools != PRIVATE_HOME / 'ops-tools' or backup_root != BACKUP_ROOT):
        raise ValueError('unexpected_private_install_path')
    promotion = json.loads(safe_path(root / 'promotion.private.json').read_bytes())
    if (not (root / 'PREPARED').is_file() or not (root / 'PROMOTED').is_file()
            or promotion.get('root') != str(root) or promotion.get('phase') != 'promoted'):
        raise ValueError('actual_promoted_stack_required')
    protection = transition_protection(backup_root)
    if not protection:
        raise ValueError('original_backups_must_be_protected')
    manifest = json.loads(safe_path(source / 'INSTALL_SOURCE_SHA256.json').read_bytes())
    if manifest.get('format') != 1 or set(manifest.get('files', {})) != set(FILES):
        raise ValueError('exact_install_source_manifest_required')
    replacements = {}
    for name in FILES:
        raw = safe_path(source / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest['files'][name]:
            raise ValueError('install_source_digest_mismatch')
        if name.endswith('.py'):
            ast.parse(raw)
        replacements[name] = raw
    def command(args):
        reply = run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        if reply.returncode:
            raise RuntimeError('install_guard_command_failed')
        return reply.stdout
    status = command(['systemctl', 'show', 'arborscan-backup.service',
                      '--property=ActiveState', '--value']).decode().strip()
    if status not in ('inactive', 'failed'):
        raise ValueError('backup_service_must_be_stopped')
    ids = command(['docker', 'compose', '-f', str(root / 'compose.private.json'),
                   '--profile', 'manual-worker', 'ps', '-aq']).decode().split()
    rows = json.loads(command(['docker', 'inspect', *ids]))
    _, images = selfhost_inventory(rows)
    if any(row['Config']['Labels']['com.docker.compose.service'] == 'worker'
           and row['State']['Running'] for row in rows):
        raise ValueError('worker_must_stay_stopped')
    runtime = json.loads(index.read_bytes())
    if runtime.get('format') != 1 or not images <= runtime.get('images', {}).keys():
        raise ValueError('actual_runtime_index_required')
    config = {'format': 1, 'root': str(root), 'runtime_index': str(index),
              'storage_layout': 'version-directory'}
    replacements[CONFIG] = json.dumps(config, sort_keys=True).encode()
    os.umask(0o077)
    with locked(backup_root):
        active = backup_root / 'active-backup.json'
        if active.exists() and json.loads(safe_path(active).read_bytes()):
            raise ValueError('active_backup_attempt_must_be_retained_separately')
        if resume_directory is not None:
            retained = safe_path(resume_directory); state = receipt(retained)
            if (retained.parent != root or state['source'] != str(source)
                    or state['source_files'] != manifest['files']
                    or state['installed_files_sha256'] != {name: digest(raw) for name, raw in replacements.items()}):
                raise ValueError('reviewed_install_source_differs')
        else:
            for pending in root.glob('backup-tools-rollback-*/INSTALL_RECEIPT.private.json'):
                old = receipt(pending.parent)
                if old['phase'] not in ('installed', 'rolled_back'):
                    raise ValueError('pending_tool_install_requires_resume_or_rollback')
            stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            retained = root / ('backup-tools-rollback-' + stamp)
            retained.mkdir(mode=0o700)
            previous = {}
            for name in replacements:
                target = safe_path(tools / name)
                if target.is_file():
                    raw = target.read_bytes(); previous[name] = (raw, target.stat().st_mode & 0o777)
                    write(retained / name, raw)
            if ENTRY not in previous:
                raise ValueError('existing_backup_entry_required')
            state = {'format': 2, 'kind': 'selfhost_backup_tools_install', 'root': str(root),
                 'tools': str(tools), 'source': str(source), 'previous_files': {
                     name: {'sha256': hashlib.sha256(raw).hexdigest(), 'mode': mode}
                     for name, (raw, mode) in previous.items()},
                 'new_files': sorted(set(replacements) - previous.keys()),
                 'installed_files_sha256': {name: hashlib.sha256(raw).hexdigest()
                     for name, raw in replacements.items()},
                 'maintenance_entry_sha256': digest(MAINTENANCE), 'phase': 'prepared', 'step': 'snapshot_saved',
                 'source_files': manifest['files'], 'systemd_changed': False,
                 'service_started': False, 'backup_sets_changed': False}
            journal(retained, state, 'prepared', 'snapshot_saved')
            sync(root)
        resume_locked(retained, state, replacements)
    return {'event': 'selfhost_backup_tools_installed', 'rollback_directory': str(retained),
            'source_files_sha256': manifest['files'], 'systemd_changed': False,
            'service_started': False, 'backup_sets_changed': False}


def rollback(retained, *, run=subprocess.run):
    retained = safe_path(retained)
    if (retained.parent.parent != PRIVATE_HOME
            or not re.fullmatch(r'selfhost-\d{8}T\d{6}Z', retained.parent.name)
            or not re.fullmatch(r'backup-tools-rollback-\d{8}T\d{6}Z', retained.name)):
        raise ValueError('private_tool_rollback_required')
    state = receipt(retained)
    tools = safe_path(state['tools'])
    status = run(['systemctl', 'show', 'arborscan-backup.service', '--property=ActiveState', '--value'],
                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    if status.returncode or status.stdout.decode().strip() not in ('inactive', 'failed'):
        raise ValueError('backup_service_must_be_stopped')
    timer = run(['systemctl', 'show', 'arborscan-backup.timer', '--property=ActiveState', '--value'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    # The historical policy may not understand transition protection. Never
    # let it resume automatically after merely restoring its source files.
    if timer.returncode or timer.stdout.decode().strip() not in ('inactive', 'failed'):
        raise ValueError('timer_must_be_stopped_before_legacy_tool_rollback')
    with locked(BACKUP_ROOT):
        current = reviewed_targets(tools, state)
        saved = saved_files(retained, state)
        if state['phase'] == 'rolled_back':
            if (any(current[name] != digest(raw) or not mode_matches(tools / name, mode)
                    for name, (raw, mode) in saved.items())
                    or any(current[name] is not None for name in state['new_files'])):
                raise ValueError('rolled_back_bundle_changed')
        else:
            journal(retained, state, 'rolling_back', 'entry_pause_pending')
            write(tools / ENTRY, MAINTENANCE, 0o700)
            for name, (raw, mode) in saved.items():
                if name == ENTRY:
                    continue
                journal(retained, state, 'rolling_back', 'restore_pending:' + name)
                if current[name] != digest(raw) or not mode_matches(tools / name, mode):
                    write(tools / name, raw, mode)
            for name in state['new_files']:
                journal(retained, state, 'rolling_back', 'remove_new_pending:' + name)
                target = safe_path(tools / name)
                if target.exists():
                    target.unlink(); sync(tools)
            journal(retained, state, 'rolling_back', 'legacy_entry_restore_pending')
            raw, mode = saved[ENTRY]
            write(tools / ENTRY, raw, mode)
            journal(retained, state, 'rolled_back', 'complete')
    return {'event': 'backup_tools_rollback_verified', 'rollback_directory': str(retained),
            'service_started': False, 'systemd_changed': False, 'backup_sets_changed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rollback-directory')
    parser.add_argument('--root')
    parser.add_argument('--source')
    parser.add_argument('--runtime-index')
    parser.add_argument('--resume-directory', help='Retained receipt of an interrupted install; original source must match')
    args = parser.parse_args()
    try:
        if args.rollback_directory:
            if args.root or args.source or args.runtime_index or args.resume_directory:
                raise ValueError('rollback_arguments_must_be_separate')
            result = rollback(args.rollback_directory)
        else:
            if not args.root or not args.source or not args.runtime_index:
                raise ValueError('install_arguments_required')
            result = install(args.root, args.source, args.runtime_index, resume_directory=args.resume_directory)
        print(json.dumps(result))
    except Exception:
        print(json.dumps({'event': 'selfhost_backup_tools_install_refused_or_failed',
                          'service_started': False, 'systemd_changed': False}), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
