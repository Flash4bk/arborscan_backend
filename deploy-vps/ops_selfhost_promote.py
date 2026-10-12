"""Promote a verified private recovery without changing HTTPS or cloud data.

The cloud containers are parked, never removed. Original images/configuration
remain available. Returning to cloud is refused once the new database has writes;
an application rollback must then retain the local database and files.

activate is resumable from parking_originals/originals_parked/local_api_starting:
the original Compose is retained before any stop and container IDs are journaled
before each Docker effect. rollback-before-writes can also resume partial parking
or rolling_back. Neither command restores or overwrites a database. The private
promotion journal phase, rather than a historical marker alone, is authoritative.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

import ops_selfhost_stack as stack
import ops_selfhost_audit as audit
import ops_selfhost_backup as backup
import ops_selfhost_release_gates as gates


class PromotionFailure(Exception):
    pass


def atomic_private(path, value):
    audit.write_private(path, value)


def plan(config, source):
    result = copy.deepcopy(config)
    if set(result['services']) != {'db', 'rest', 'storage', 'gateway', 'v3', 'api', 'worker'}:
        raise PromotionFailure('Unexpected service inventory')
    if not result['networks']['default']['internal']:
        raise PromotionFailure('Closed database network required')
    for service, port, name in [('v3', 8000, 'arborscan-api'), ('api', 8001, 'arborscan-api-v4')]:
        item = result['services'][service]
        if item['image'] != source[service]['Image'] or item['environment']['SUPABASE_URL'] != 'http://gateway:8000':
            raise PromotionFailure('Verified application transport required')
        item['ports'] = [f'127.0.0.1:{port}:{port}']
        item['container_name'] = name
        # The bounded staging inference is repeated before promotion. Two CPU
        # cores per API leave capacity for PostgreSQL and the host on this VPS.
        item['cpus'] = 2
    if result['services']['worker'].get('profiles') != ['manual-worker'] or result['services']['worker']['restart'] != 'no':
        raise PromotionFailure('Worker must remain manual and stopped')
    return result


def continuity(root):
    public = audit.parse_fingerprints(audit.run_sql(root))
    storage = audit.parse_storage_fingerprints(audit.run_sql(root, audit.STORAGE_FULL_FINGERPRINT_SQL))
    return hashlib.sha256(audit.canonical({'public': public, 'storage': storage})).hexdigest()


def inspect_originals(runner, source, receipt=None):
    if receipt is not None:
        expected_ids = {row['Id'] for row in source.values()}
        if (len(receipt.get('originals', [])) != len(expected_ids)
                or {item['id'] for item in receipt['originals']} != expected_ids):
            raise PromotionFailure('Original journal inventory differs')
    rows = json.loads(runner.run(['docker', 'inspect', *[row['Id'] for row in source.values()]]).stdout)
    actual = {row['Id']: row for row in rows}
    for expected in source.values():
        row = actual.get(expected['Id'])
        allowed_names = {expected['Name']}
        if receipt is not None:
            original = next((item for item in receipt['originals'] if item['id'] == expected['Id']), None)
            if (original is None or original['name'] != expected['Name'].lstrip('/')
                    or original['restart'] != expected['HostConfig']['RestartPolicy']
                    or original['running'] != expected['State']['Running']):
                raise PromotionFailure('Original journal identity differs')
            expected_parked = original['name'] + '-cloud-rollback-' + receipt['root'].split('selfhost-', 1)[1]
            if original['parked'] != expected_parked:
                raise PromotionFailure('Original journal name differs')
            allowed_names.add('/' + original['parked'])
            restart = row['HostConfig']['RestartPolicy'] if row else None
            if restart not in (original['restart'], {'Name': 'no', 'MaximumRetryCount': 0}):
                raise PromotionFailure('Original restart policy changed outside journal')
        if (not row or row['Name'] not in allowed_names or row['Image'] != expected['Image']
                or row['Config']['Env'] != expected['Config']['Env'] or row['Mounts'] != expected['Mounts']
                or row['HostConfig']['PortBindings'] != expected['HostConfig']['PortBindings']):
            raise PromotionFailure('Original deployment changed; preserve and review')
    return actual


def journal(root, receipt, phase=None, original=None, step=None):
    if phase is not None:
        receipt['phase'] = phase
    if original is not None:
        original['step'] = step
    atomic_private(root / 'promotion.private.json', receipt)


def reviewed_configs(root, receipt, source):
    """Preserve the staging configuration before the first container mutation."""
    candidate = root / 'compose.production.private.json'
    if hashlib.sha256(candidate.read_bytes()).hexdigest() != receipt['compose_sha256']:
        raise PromotionFailure('Candidate configuration changed')
    backup_path = root / 'compose.staging.private.json'
    current = audit.private_json(root / 'compose.private.json')
    if not backup_path.exists():
        if receipt['phase'] != 'prepared':
            raise PromotionFailure('Original staging configuration unavailable')
        stack.private_write(backup_path, json.dumps(current, indent=2))
    original = audit.private_json(backup_path)
    expected = plan(original, source)
    if audit.private_json(candidate) != expected or current not in (original, expected):
        raise PromotionFailure('Reviewed staging or candidate configuration differs')
    return original, expected


def park_originals(root, receipt, source, runner):
    journal(root, receipt, phase='parking_originals')
    for original in receipt['originals']:
        actual = inspect_originals(runner, source, receipt)[original['id']]
        if actual['State']['Running']:
            journal(root, receipt, original=original, step='stop_pending')
            runner.run(['docker', 'stop', '--time', '30', original['id']])
        journal(root, receipt, original=original, step='stopped')
        actual = inspect_originals(runner, source, receipt)[original['id']]
        if actual['HostConfig']['RestartPolicy']['Name'] != 'no':
            journal(root, receipt, original=original, step='restart_disable_pending')
            runner.run(['docker', 'update', '--restart=no', original['id']])
        journal(root, receipt, original=original, step='restart_disabled')
        actual = inspect_originals(runner, source, receipt)[original['id']]
        if actual['Name'] != '/' + original['parked']:
            journal(root, receipt, original=original, step='rename_pending')
            runner.run(['docker', 'rename', original['id'], original['parked']])
        journal(root, receipt, original=original, step='parked')
    journal(root, receipt, phase='originals_parked')


def inspect_named(runner, name):
    ids = runner.run(['docker', 'ps', '-aq', '--filter', 'name=^/' + name + '$']).stdout.decode().split()
    if not ids:
        return None
    if len(ids) != 1:
        raise PromotionFailure('Container name inventory differs')
    rows = json.loads(runner.run(['docker', 'inspect', ids[0]]).stdout)
    if len(rows) != 1:
        raise PromotionFailure('Container name inspection differs')
    return rows[0]


def park_local_candidate(root, receipt, original, source, runner, config):
    if original['name'] == 'arborscan-quality-worker':
        return
    target = inspect_named(runner, original['name'])
    if target is None or target['Id'] == original['id']:
        return
    service = 'v3' if original['name'] == 'arborscan-api' else 'api'
    labels = target['Config'].get('Labels') or {}
    if (target['Image'] != source[service]['Image'] or target['State']['Running']
            or labels.get('com.docker.compose.project') != config['name']
            or labels.get('com.docker.compose.service') != service):
        raise PromotionFailure('Unexpected container occupies original name')
    parked = original['name'] + '-local-rollback-' + root.name[9:]
    original['local_candidate_id'] = target['Id']
    original['local_candidate_parked'] = parked
    journal(root, receipt, original=original, step='local_rename_pending')
    runner.run(['docker', 'rename', target['Id'], parked])
    journal(root, receipt, original=original, step='local_parked')


def prepare(root, checked_backup, restore_receipt, source_proof):
    root = stack.checked_resume_root(root)
    existing = root / 'promotion.private.json'
    if existing.exists() and audit.private_json(existing).get('phase') != 'prepared':
        raise PromotionFailure('Existing promotion retained; cannot replace active journal')
    config = audit.private_json(root / 'compose.private.json')
    source = stack.live_rows(root / 'live-inspect.private.json')
    if not (root / 'PREPARED').is_file():
        raise PromotionFailure('Prepared infrastructure required')
    seeded = audit.private_json(root / 'storage-seed.safe.json')
    audited = audit.private_json(root / 'audit-before-fixtures.json')
    repeated = audit.private_json(root / 'audit-restart.json')
    inference = audit.private_json(root / 'inference-result.json')
    if (seeded.get('sha256_verified') is not True or audited.get('api_health') != 'passed'
            or audited.get('exact_images_policy_models') != 'identical'
            or repeated.get('smoke', {}).get('restart_password_login_versions_contours') != 'passed'
            or inference.get('analysis_saved') is not True):
        raise PromotionFailure('Actual continuity, restart and analysis proofs required')
    try:
        release_gates = gates.verify(root, checked_backup, restore_receipt, source_proof, config)
    except gates.GateFailure as error:
        raise PromotionFailure(str(error)) from None
    runner = stack.Runner(root)
    inspect_originals(runner, source)
    audit.running_stack(root, config)
    candidate = plan(config, source)
    filename = root / 'compose.production.private.json'
    stack.private_write(filename, json.dumps(candidate, indent=2))
    receipt = {'format': 1, 'root': str(root), 'backup': str(checked_backup),
               'compose_sha256': hashlib.sha256(filename.read_bytes()).hexdigest(),
               'before_write_fingerprint': continuity(root),
               'phase': 'prepared', 'originals': [], 'release_gates': release_gates}
    for row in source.values():
        receipt['originals'].append({'id': row['Id'], 'name': row['Name'].lstrip('/'),
            'parked': row['Name'].lstrip('/') + '-cloud-rollback-' + root.name[9:],
            'restart': row['HostConfig']['RestartPolicy'], 'running': row['State']['Running']})
    stack.private_write(root / 'promotion.private.json', json.dumps(receipt, indent=2))
    return {'promotion_prepared': True, 'api_images_unchanged': True,
            'local_db_ports_public': False, 'https_config_changed': False,
            'worker_will_start': False, 'backup_restore_verified': True}


def health():
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for port in (8000, 8001):
        for _ in range(90):
            try:
                with opener.open(f'http://127.0.0.1:{port}/health', timeout=5) as response:
                    if response.status == 200 and json.load(response).get('status') == 'ok':
                        break
            except (OSError, ValueError):
                pass
            time.sleep(2)
        else:
            raise PromotionFailure('Public-port API readiness failed')


def activate(root):
    root = stack.checked_resume_root(root)
    receipt = audit.private_json(root / 'promotion.private.json')
    if receipt.get('root') != str(root):
        raise PromotionFailure('Promotion journal root differs')
    phase = receipt.get('phase')
    if phase not in ('prepared', 'parking_originals', 'originals_parked', 'local_api_starting', 'promoted'):
        raise PromotionFailure('Existing promotion retained; inspect phase before resuming')
    if phase == 'prepared':
        # Recheck only the retained read-only source proof before the very first
        # container effect. The expensive isolated restore was verified during
        # prepare. Partial parking remains resumable without a cloud round-trip.
        saved = receipt.get('release_gates', {})
        if not isinstance(saved.get('source_proof'), str):
            raise PromotionFailure('Prepared source evidence required')
        try:
            observed = gates.source_gate(root, Path(saved['source_proof']))
        except gates.GateFailure as error:
            raise PromotionFailure(str(error)) from None
        if any(saved.get(key) != value for key, value in observed.items()):
            raise PromotionFailure('Prepared source evidence changed')
    # Once the candidate has served writes, only the same local database may be
    # restarted. A pre-start divergence still refuses all deployment mutations.
    if phase in ('prepared', 'parking_originals', 'originals_parked') and continuity(root) != receipt['before_write_fingerprint']:
        raise PromotionFailure('Prepared local data changed; repeat review')
    source = stack.live_rows(root / 'live-inspect.private.json')
    runner = stack.Runner(root)
    _, candidate = reviewed_configs(root, receipt, source)
    inspect_originals(runner, source, receipt)
    if phase in ('prepared', 'parking_originals', 'originals_parked'):
        park_originals(root, receipt, source, runner)
    config = root / 'compose.private.json'
    # Both intent and the original configuration survive a break before or
    # after the atomic candidate replacement or Docker Compose invocation.
    journal(root, receipt, phase='local_api_starting')
    if audit.private_json(config) != candidate:
        atomic_private(config, candidate)
    runner.run(['docker', 'compose', '-f', str(config), 'up', '-d', '--pull', 'never', '--no-deps', 'v3', 'api'])
    health()
    journal(root, receipt, phase='promoted')
    if not (root / 'PROMOTED').exists():
        stack.private_write(root / 'PROMOTED', b'local database and Storage are authoritative\n')
    return {'promotion': 'passed', 'api_ports': [8000, 8001],
            'old_containers': 'parked_restart_disabled', 'worker_started': False,
            'cloud_database_changed': False, 'https_config_changed': False}


def rollback_before_writes(root):
    """Refuse cloud rollback after writes, restart local API when refusing."""
    root = stack.checked_resume_root(root)
    receipt = audit.private_json(root / 'promotion.private.json')
    if receipt.get('root') != str(root):
        raise PromotionFailure('Promotion journal root differs')
    phase = receipt.get('phase')
    if phase == 'cloud_rollback_no_writes':
        return {'rollback': 'original_cloud_api_restored', 'local_data_preserved': True,
                'warning': 'Supabase quota remains; this restores deployment, not cloud availability'}
    if phase not in ('parking_originals', 'originals_parked', 'local_api_starting', 'promoted', 'rolling_back'):
        raise PromotionFailure('Promotion state does not permit cloud rollback')
    runner = stack.Runner(root)
    source = stack.live_rows(root / 'live-inspect.private.json')
    original_config, _ = reviewed_configs(root, receipt, source)
    inspect_originals(runner, source, receipt)
    command = ['docker', 'compose', '-f', str(root / 'compose.private.json')]
    runner.run(command + ['stop', 'v3', 'api'])
    if continuity(root) != receipt['before_write_fingerprint']:
        # Never return to the stale cloud database. If a rollback was already
        # resuming, retain that phase for operator review; no cloud API starts.
        if phase != 'rolling_back':
            runner.run(command + ['up', '-d', '--pull', 'never', '--no-deps', 'v3', 'api'])
            raise PromotionFailure('Local writes exist; cloud rollback refused; local API retained')
        raise PromotionFailure('Local writes exist; rollback resume refused; review retained deployment')
    journal(root, receipt, phase='rolling_back')
    for original in receipt['originals']:
        # A break during the earlier API parking may have left the old worker
        # running. Rollback never resumes training and explicitly stops it.
        if original['name'] == 'arborscan-quality-worker':
            actual = inspect_originals(runner, source, receipt)[original['id']]
            if actual['State']['Running']:
                journal(root, receipt, original=original, step='worker_stop_pending')
                runner.run(['docker', 'stop', '--time', '30', original['id']])
        park_local_candidate(root, receipt, original, source, runner, original_config)
        actual = inspect_originals(runner, source, receipt)[original['id']]
        if actual['Name'] != '/' + original['name']:
            journal(root, receipt, original=original, step='original_rename_pending')
            runner.run(['docker', 'rename', original['id'], original['name']])
        journal(root, receipt, original=original, step='original_name_restored')
        # A merely stopped worker with restart=always could resume training
        # when Docker restarts. API policies are restored; this worker stays
        # explicitly disabled until a separate authorized training operation.
        policy = 'no' if original['name'] == 'arborscan-quality-worker' else original['restart']['Name']
        if policy == 'on-failure' and original['restart'].get('MaximumRetryCount'):
            policy += ':' + str(original['restart']['MaximumRetryCount'])
        journal(root, receipt, original=original, step='original_restart_restore_pending')
        runner.run(['docker', 'update', '--restart=' + policy, original['id']])
        # The training worker remains stopped even if originally running.
        if original['running'] and original['name'] != 'arborscan-quality-worker':
            actual = inspect_originals(runner, source, receipt)[original['id']]
            if not actual['State']['Running']:
                journal(root, receipt, original=original, step='original_start_pending')
                runner.run(['docker', 'start', original['id']])
        journal(root, receipt, original=original, step='original_restored')
    # The phase is authoritative. Preserve the historical marker under a new
    # name instead of leaving a current-authority marker after cloud rollback.
    marker = root / 'PROMOTED'
    if marker.exists():
        retained = root / 'PROMOTED.before-cloud-rollback'
        if retained.exists():
            raise PromotionFailure('Historical promotion marker already retained')
        marker.rename(retained)
    journal(root, receipt, phase='cloud_rollback_no_writes')
    return {'rollback': 'original_cloud_api_restored', 'local_data_preserved': True,
            'warning': 'Supabase quota remains; this restores deployment, not cloud availability'}


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'activate', 'rollback-before-writes'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--backup', type=Path)
    parser.add_argument('--restore-receipt', type=Path)
    parser.add_argument('--source-proof', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        if not args.backup or not args.restore_receipt or not args.source_proof:
            parser.error('--backup, --restore-receipt and --source-proof required')
        result = prepare(args.root, args.backup, args.restore_receipt, args.source_proof)
    elif args.command == 'activate':
        result = activate(args.root)
    else:
        result = rollback_before_writes(args.root)
    print(json.dumps(result))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        # Subprocess errors, URLs and source records may contain credentials.
        category = str(exc) if isinstance(exc, PromotionFailure) else type(exc).__name__
        print(json.dumps({'promotion': 'incomplete', 'category': category}), file=sys.stderr)
        raise SystemExit(1)
