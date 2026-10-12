"""Quarantine only this recovery's newly created DEMO contours, never real data.

The existing exporter purpose marker is ancillary. Original PNG/image/editor,
immutable revision identity and SQL decisions must remain identical. No dataset,
training job or model is created. This command is limited to a private stage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.parse import quote
import uuid

import ops_selfhost_audit as audit
import ops_selfhost_stack as stack

PURPOSE = 'export_smoke_test_only_not_model_evaluation'
KEY = re.compile(r'[0-9a-f-]{36}_[0-9a-f]{64}\.json')


class QuarantineFailure(Exception):
    pass


def fixture_identity(state):
    audit.validate_fixture(state)
    if state.get('complete') is not True:
        raise QuarantineFailure('completed_fixture_required')
    for account in state['accounts']:
        if str(uuid.UUID(account['id'])) != account['id']:
            raise QuarantineFailure('canonical_fixture_uuid_required')
    keys = [row[0] for row in state.get('contours', []) if isinstance(row, list) and len(row) == 2]
    if (len(keys) != 3 or len(set(keys)) != 3 or any(not KEY.fullmatch(k) for k in keys)
            or {row[1] for row in state['contours']} != {'rejected', 'accepted', 'draft'}
            or not audit.SHA256.fullmatch(state.get('image_sha256', ''))):
        raise QuarantineFailure('exact_recovery_contour_fixture_required')
    return state['accounts'][0], set(keys)


def source_accounts_absent(state, execute=subprocess.run):
    # UUID validation happens before interpolation, not after a SQL request.
    fixture_identity(state)
    ids = [account['id'] for account in state['accounts']]
    sql = ('BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY; '
           "SELECT count(*) FROM public.users WHERE id IN (" +
           ','.join("'" + value + "'::uuid" for value in ids) + '); COMMIT;')
    result = execute(['/home/arborscan/.local/bin/psql', '-w', 'service=arborscan_backup',
                      '-v', 'ON_ERROR_STOP=1', '-qAt'], input=sql.encode(),
                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    if result.returncode or result.stdout.strip() != b'0':
        raise QuarantineFailure('fixture_account_exists_or_source_proof_unavailable')
    return {'source_user_matches': 0, 'account_ids_sha256':
            audit.digest_bytes(audit.canonical(sorted(ids)))}


def metadata(http, owner):
    response = http.request('GET', '/rest/v1/contour_revisions', db=True,
                            params={'owner_id': 'eq.' + owner, 'order': 'correction_id'})
    result = response.json()
    if not isinstance(result, list):
        raise QuarantineFailure('fixture_sql_metadata_required')
    return result


def object_path(bucket, owner, key):
    if not KEY.fullmatch(key) or str(uuid.UUID(key[:36])) != key[:36]:
        raise QuarantineFailure('invalid_fixture_contour_key')
    return '/storage/v1/object/' + quote(bucket, safe='') + '/' + quote('v4-corrections/' + owner + '/' + key, safe='/')


def collect(http, bucket, state, baseline):
    owner, expected = fixture_identity(state)
    prefix = 'v4-corrections/' + owner['id'] + '/'
    # The source manifest was captured before any DEMO writes. Also guard both
    # new accounts' namespaces, even if the second account owns no contours.
    for account in state['accounts']:
        original_prefix = 'v4-corrections/' + account['id'] + '/'
        if any(row['bucket'] == bucket and row['name'].startswith(original_prefix) for row in baseline['storage']):
            raise QuarantineFailure('fixture_namespace_existed_in_source')
    listing = http.request('POST', '/storage/v1/object/list/' + quote(bucket, safe=''), db=True,
                           json={'prefix': prefix, 'limit': 6, 'offset': 0,
                                 'sortBy': {'column': 'name', 'order': 'asc'}}).json()
    if not isinstance(listing, list) or len(listing) != 4:
        raise QuarantineFailure('exact_four_owned_fixture_objects_required')
    keys = {row.get('name') for row in listing if isinstance(row, dict)}
    if len(keys) != 4 or not expected < keys:
        raise QuarantineFailure('unexpected_fixture_object_set')
    records = {}
    for key in sorted(keys):
        path = object_path(bucket, owner['id'], key)
        record = http.request('GET', path, db=True).json()
        if (not isinstance(record, dict) or record.get('owner_id') != owner['id']
                or record.get('image_sha256') != state['image_sha256']
                or record.get('purpose') not in (None, PURPOSE)):
            raise QuarantineFailure('fixture_record_identity_differs')
        if key not in expected and record.get('schema_version') != 1:
            raise QuarantineFailure('untracked_fixture_must_be_legacy_png')
        records[key] = record
    sql_rows = metadata(http, owner['id'])
    expected_status = dict(state['contours'])
    if (len(sql_rows) != 3 or {row.get('correction_id') for row in sql_rows} != expected
            or any(row.get('owner_id') != owner['id'] or row.get('status') != expected_status[row['correction_id']]
                   or row.get('image_sha256') != state['image_sha256'] for row in sql_rows)):
        raise QuarantineFailure('fixture_decisions_or_sql_identity_differs')
    return owner, records, sql_rows


def unmarked(value):
    return {key: data for key, data in value.items() if key != 'purpose'}


def snapshot(root, records, sql_rows, proof, state):
    path = root / 'quarantine.originals.private.json'
    identity = {'tag': state['tag'], 'owner': state['accounts'][0]['id'],
                'image_sha256': state['image_sha256'], 'proof': proof}
    if path.exists():
        old = audit.private_json(path)
        if (old['identity'] != identity or old['sql_metadata'] != sql_rows
                or set(old['records']) != set(records)
                or any(unmarked(old['records'][key]) != unmarked(record) for key, record in records.items())):
            raise QuarantineFailure('existing_private_snapshot_differs')
        return old
    result = {'format': 1, 'identity': identity, 'records': records, 'sql_metadata': sql_rows}
    stack.private_write(path, audit.canonical(result))
    return result


def put_marker(http, path, record):
    response = http.session.request('POST', http.base + path,
        headers={'Authorization': 'Bearer ' + http.service_key, 'apikey': http.service_key,
                 'Content-Type': 'application/json', 'x-upsert': 'true'},
        data=audit.canonical({**record, 'purpose': PURPOSE}), timeout=120, allow_redirects=False)
    if response.status_code not in (200, 201):
        raise QuarantineFailure('fixture_marker_write_unconfirmed')


def apply_markers(root, http, bucket, state, baseline, proof, put=put_marker, pause=time.sleep):
    owner, records, sql_rows = collect(http, bucket, state, baseline)
    original = snapshot(root, records, sql_rows, proof, state)
    for key, record in records.items():
        path = object_path(bucket, owner['id'], key)
        if record.get('purpose') != PURPOSE:
            put(http, path, record)
        for attempt in range(5):
            observed = http.request('GET', path, db=True).json()
            if unmarked(observed) != unmarked(original['records'][key]):
                raise QuarantineFailure('fixture_content_changed_beyond_purpose')
            if observed.get('purpose') == PURPOSE:
                break
            if attempt == 4:
                raise QuarantineFailure('fixture_marker_readback_unconfirmed')
            pause(.5)
        # Confirm the author-visible route still reports the same immutable
        # fields and moderation status. This is an API read, not a UI proof.
        public = http.api('GET', '/api/v4/v4/corrections/' + key, owner['token'])
        if (public.get('purpose') != PURPOSE or public.get('image_sha256') != record['image_sha256']
                or public.get('mask_sha256') != record['mask_sha256']
                or public.get('review_status') != dict(state['contours']).get(key, record['review_status'])):
            raise QuarantineFailure('author_route_quarantine_mismatch')
    if metadata(http, owner['id']) != original['sql_metadata']:
        raise QuarantineFailure('fixture_sql_decisions_changed')
    return {'objects_marked': len(records), 'immutable_content_preserved': True,
            'sql_decisions_preserved': True, 'source_accounts_absent': True,
            'real_user_records_changed': False, 'training_started': False,
            'snapshot_sha256': audit.digest_bytes(audit.canonical(original))}


EXPORTER_PROBE = r"""
import json,sys
from arborscan_v4.quality_dataset import sample
records=json.load(sys.stdin)
refused=0
for record in records:
    for kind in ('segmentation','classification'):
        try: sample(record,{},model_type=kind)
        except ValueError as error:
            if str(error)!='synthetic_export_smoke_not_evaluation_data':raise
            refused+=1
        else:raise RuntimeError('synthetic sample admitted')
print(json.dumps({'refusals':refused,'dataset_created':False,'training_started':False}))
"""


def quarantine(root):
    root = stack.checked_resume_root(root)
    if not (root / 'PREPARED').is_file() or (root / 'PROMOTED').exists():
        raise QuarantineFailure('private_preproduction_stage_required')
    config = audit.private_json(root / 'compose.private.json')
    manifest = audit.private_json(root / 'rootmanifest.safe.json')
    base = audit.guarded_stack(root, config, manifest['assets_root'])
    audit.running_stack(root, config)
    state = audit.private_json(root / 'fixture.private.json')
    fixture_identity(state)
    restart = audit.private_json(root / 'audit-restart.json')
    if restart.get('smoke', {}).get('restart_password_login_versions_contours') != 'passed':
        raise QuarantineFailure('actual_restart_gate_required')
    proof = source_accounts_absent(state)
    baseline = audit.private_json(root / 'audit.expected.private.json')
    key = audit.compose_literal(config['services']['api']['environment']['SUPABASE_SERVICE_KEY'])
    http = audit.StageHTTP(base, key, audit.api_bases(config))
    bucket = config['services']['api']['environment'].get('V4_CORRECTIONS_BUCKET', 'arborscan-corrections')
    result = apply_markers(root, http, bucket, state, baseline, proof)
    owner, records, _ = collect(http, bucket, state, baseline)
    runner = stack.Runner(root)
    response = runner.run(['docker', 'compose', '-f', str(root / 'compose.private.json'),
                           'exec', '-T', 'api', 'python', '-c', EXPORTER_PROBE],
                          audit.canonical(list(records.values())))
    actual = json.loads(response.stdout)
    if actual != {'refusals': 8, 'dataset_created': False, 'training_started': False}:
        raise QuarantineFailure('actual_exporter_refusal_unconfirmed')
    result.update(actual_exporter_segmentation_classification_refusals=8, dataset_created=False)
    audit.write_private(root / 'quarantine-result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(quarantine(args.root)))
    except Exception as error:
        category = str(error) if isinstance(error, QuarantineFailure) else type(error).__name__
        print(json.dumps({'quarantine': 'incomplete', 'category': category}), file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
