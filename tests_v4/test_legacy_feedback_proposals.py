"""Immutable legacy proposals use the existing private v4 moderation queue.

Storage and SQL transports are in-memory fixtures, not a PostgreSQL migration
or a production-write test. Real proposal validation, image decoding, hashing,
commit visibility and exporter consent rules execute.
"""
import base64
import copy
from concurrent.futures import ThreadPoolExecutor
import importlib
import io
import json
import threading
import subprocess
import sys
import os

import numpy as np
from PIL import Image
from fastapi import HTTPException
import pytest

from arborscan_v4 import corrections_api as api
from arborscan_v4.quality_dataset import sample

OWNER = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa'
AID = 'cccccccc-cccc-4ccc-8ccc-cccccccccccc'


def proposal(consent=False):
    return {'version': 1, 'kind': 'legacy_user_feedback', 'training_consent': consent,
            'quality': {'tree_ok': True, 'stick_ok': False, 'params_ok': True, 'species_ok': False},
            'corrections': {'species': 'Pinus sylvestris', 'height_m': 17,
                            'crown_width_m': None, 'trunk_diameter_m': None, 'scale_px_to_m': None},
            'has_user_mask': False}


@pytest.fixture
def storage(monkeypatch):
    blobs, metadata = {}, {}
    state = {'fail_register': False}
    lock = threading.RLock()
    class Store:
        def __init__(self, *args): pass
        def ready(self): pass
        def get(self, owner, key):
            with lock: return copy.deepcopy(metadata.get((owner, key)))
        def request(self, method, path, **kwargs):
            assert method == 'GET' and path == 'contour_revisions'
            with lock: return [{'correction_id': key} for owner, key in metadata]
        def transition(self, action, owner, key, **fields):
            assert action == 'legacy'
            if state['fail_register']: raise HTTPException(503, 'Synthetic registry outage')
            with lock:
                return copy.deepcopy(metadata.setdefault((owner, key),
                    {'status': 'submitted', 'decisions': [], 'correction_id': key,
                     'owner_id': owner, 'analysis_id': fields['analysis'], 'image_sha256': fields['image']}))
    monkeypatch.setattr(api, 'WorkflowStore', Store)
    def put(owner, key, record):
        with lock: return copy.deepcopy(blobs.setdefault((owner, key), copy.deepcopy(record)))
    def get(owner, key):
        with lock: return copy.deepcopy(blobs[(owner, key)])
    monkeypatch.setattr(api, '_put', put)
    monkeypatch.setattr(api, '_get', get)
    monkeypatch.setattr(api, '_config', lambda: ('https://synthetic.invalid', {}, 'private'))
    class Listing:
        status_code = 200
        def json(self): return [{'name': key} for owner, key in blobs]
    monkeypatch.setattr(api.requests, 'post', lambda *a, **k: Listing())
    image = io.BytesIO(); Image.new('RGB', (48, 64), 'green').save(image, 'JPEG')
    mask = np.zeros((64, 48), np.uint8); mask[4:60, 8:40] = 255
    encoded = io.BytesIO(); Image.fromarray(mask).save(encoded, 'PNG')
    return {'blobs': blobs, 'metadata': metadata, 'state': state,
            'image': image.getvalue(), 'mask': encoded.getvalue()}


def save(storage, payload=None):
    module = importlib.import_module('arborscan_v4.legacy_feedback')
    return module.save_feedback_proposal(OWNER, AID, storage['image'], storage['mask'], payload or proposal())


def test_same_proposal_is_immutable_and_concurrent_retries_share_one_queue_entry(storage):
    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(lambda _: save(storage), range(2)))
    assert first['saved'] and first['correction_id'] == second['correction_id']
    assert len(storage['blobs']) == len(storage['metadata']) == 1
    record = next(iter(storage['blobs'].values()))
    assert record['schema_version'] == 1 and 'editor_state' not in record
    assert record['requires_workflow_commit'] is True
    assert base64.b64decode(record['original_image_base64']) == storage['image']
    assert record['feedback_proposal']['corrections']['height_m'] == 17
    assert record['eligible_for_training'] is False
    assert first['review_status'] == 'submitted'


def test_changed_proposal_creates_new_record_without_changing_accepted_original(storage):
    first = save(storage); old_key = (OWNER, first['correction_id'])
    storage['metadata'][old_key].update(status='accepted', decisions=[{'action': 'accepted', 'actor_id': 'admin', 'at': 'test'}])
    original = copy.deepcopy(storage['blobs'][old_key])
    revised = proposal(); revised['corrections']['height_m'] = 18
    second = save(storage, revised)
    assert second['correction_id'] != first['correction_id'] and second['review_status'] == 'submitted'
    assert storage['blobs'][old_key] == original
    assert storage['metadata'][old_key]['status'] == 'accepted'
    assert save(storage)['review_status'] == 'accepted'


def test_registration_failure_leaves_hidden_blob_and_retry_commits_same_content(storage):
    storage['state']['fail_register'] = True
    with pytest.raises(HTTPException): save(storage)
    key = next(iter(storage['blobs']))[1]
    record = api._get(OWNER, key)
    assert api._requires_workflow_commit(record)
    with pytest.raises(HTTPException) as caught:
        api._ensure_metadata(api.WorkflowStore(api._config), OWNER, key, record)
    assert caught.value.status_code == 409
    with pytest.raises(HTTPException) as caught: api.get_correction(key, OWNER)
    assert caught.value.status_code == 409
    assert api.list_corrections(owner=OWNER)['items'] == []
    storage['state']['fail_register'] = False
    saved = save(storage)
    assert saved['correction_id'] == key and len(storage['blobs']) == 1
    assert api.get_correction(key, OWNER)['review_status'] == 'submitted'
    assert len(api.list_corrections(owner=OWNER)['items']) == 1


def test_opt_out_survives_mask_acceptance_and_opt_in_still_requires_admin(storage):
    saved = save(storage)
    key = (OWNER, saved['correction_id']); record = storage['blobs'][key]
    metadata = {**storage['metadata'][key], 'status': 'accepted',
                'decisions': [{'action': 'accepted', 'actor_id': 'admin', 'at': 'test'}]}
    with pytest.raises(ValueError, match='training_consent'):
        sample(record, metadata)
    allowed = save(storage, proposal(True)); key = (OWNER, allowed['correction_id'])
    with pytest.raises(ValueError, match='accepted_revision'):
        sample(storage['blobs'][key], storage['metadata'][key])
    metadata = {**storage['metadata'][key], 'status': 'accepted', 'decisions': metadata['decisions']}
    assert sample(storage['blobs'][key], metadata)['fidelity']['iou'] >= .995


def test_invalid_or_forged_proposal_is_rejected_before_storage(storage):
    payload = proposal(); payload['verified'] = True
    with pytest.raises(HTTPException) as caught: save(storage, payload)
    assert caught.value.status_code == 422 and storage['blobs'] == {}


def test_old_png_without_proposal_preserves_existing_behavior(storage):
    result = api._validate_and_save(OWNER, AID, storage['image'], storage['mask'])
    record = storage['blobs'][OWNER, result['correction_id']]
    assert record['schema_version'] == 1 and 'feedback_proposal' not in record
    assert record['eligible_for_training'] is False
    # A PNG without a proven admin decision remains ineligible for training.
    with pytest.raises(ValueError, match='accepted_revision'):
        sample(record, {'status': 'submitted', 'decisions': []})


def test_owner_http_feedback_is_immutable_discoverable_and_retryable(storage, tmp_path):
    from tests_v4.test_legacy_security_remediation import namespace, feedback_client, SOURCE
    import ast
    ns, state = namespace(tmp_path)
    # Execute the actual lazy bridge too; only external storage/SQL are fake.
    node = next(n for n in ast.parse(SOURCE.read_text(encoding='utf-8')).body
                if isinstance(n, ast.FunctionDef) and n.name == '_persist_legacy_feedback')
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), ns)
    ns['_load_feedback_artifacts'] = lambda aid: {'input_bytes': storage['image'],
        'auto_mask_bytes': storage['mask'], 'meta': {'height_m': 10}, 'source': 'synthetic'}
    client = feedback_client(ns)
    body = {'analysis_id': AID, 'corrected_height_m': 17, 'use_for_training': True}
    headers = {'Authorization': 'Bearer owner-session'}
    storage['state']['fail_register'] = True
    assert client.post('/feedback', json=body, headers=headers).status_code == 503
    assert api.list_corrections(owner=OWNER)['items'] == []
    storage['state']['fail_register'] = False
    first = client.post('/feedback', json=body, headers=headers)
    second = client.post('/feedback', json=body, headers=headers)
    assert first.status_code == second.status_code == 200
    assert first.json()['saved'] is True and first.json()['verified'] is False
    assert first.json()['correction_id'] == second.json()['correction_id']
    assert len(api.list_corrections(owner=OWNER)['items']) == 1
    assert state['uploads'] == [] and state['saved_rows'] == []
    saved = api.get_correction(first.json()['correction_id'], OWNER)
    assert saved['feedback_proposal']['corrections']['height_m'] == 17
    assert 'height_m' not in saved  # Proposed measurement is never report truth.


def test_lazy_adapter_import_does_not_start_api_or_import_ml_runtime():
    script = "import sys; import arborscan_v4.legacy_feedback; assert 'arborscan_v4.api' not in sys.modules; assert 'ultralytics' not in sys.modules; assert 'torch' not in sys.modules"
    env = {k: v for k, v in os.environ.items() if not any(word in k.upper() for word in ['SUPABASE', 'PLANTNET', 'OPENWEATHER'])}
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    result = subprocess.run([sys.executable, '-B', '-c', script], env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
