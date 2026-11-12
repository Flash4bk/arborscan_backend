"""Legacy-v3 regressions using exact handlers and synthetic boundary transports.

Importing server.py would load ML weights/provider configuration. Compile its
actual functions instead; real FastAPI request handling, image headers and
geometry still run. No production credentials, records or model inference.
"""
import ast
import asyncio
import base64
import copy
import inspect
import io
import json
import math
from pathlib import Path
import re
import struct
import time
import types
from typing import Optional
from uuid import uuid4
import zlib

import cv2
from fastapi import Body, Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from fastapi.concurrency import run_in_threadpool
from PIL import Image, ImageOps, UnidentifiedImageError
import numpy as np
import pytest
import requests

SOURCE = Path(__file__).parents[1] / 'server.py'
OWNER = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa'
OTHER = 'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb'
AID = 'cccccccc-cccc-4ccc-8ccc-cccccccccccc'
SECRET = 'synthetic-provider-credential'


def namespace(tmp_path):
    names = {'send_feedback', '_plantnet_health_sync', '_run_classifier_sync',
             '_plantnet_unknown', '_extract_bearer_token', '_resolve_auth_token',
             '_validate_analysis_id', '_json_from_bytes', '_legacy_feedback_user',
             '_decode_legacy_image', 'analyze_tree', 'estimate_beta_kg_s_exact',
             'compute_risk', 'bending_stress_score', 'slenderness_score',
             '_save_analysis_record', 'extract_exif_scale', 'admin_verified_list'}
    names.update({'_strip_data_url', 'decode_base64_bytes', 'ensure_png_mask_bytes'})
    nodes = []
    for node in ast.parse(SOURCE.read_text(encoding='utf-8')).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            node.decorator_list = []
            nodes.append(node)
    state = {'uploads': [], 'proposals': [], 'artifact_reads': 0, 'db_reads': 0, 'existing': {},
             'inferences': 0, 'saved_rows': [], 'db_failure': False}
    users = {'owner-session': {'id': OWNER, 'role': 'user'},
             'other-session': {'id': OTHER, 'role': 'user'},
             'admin-session': {'id': OTHER, 'role': 'admin'}}
    def db(method, path, **kwargs):
        if state['db_failure']:
            raise RuntimeError('private-database-error-' + SECRET)
        if method == 'GET' and path == 'analyses':
            state['db_reads'] += 1
            params = kwargs['params']
            if params.get('id') != 'eq.' + AID or params.get('user_id', 'eq.' + OWNER) != 'eq.' + OWNER:
                return []
            return [{'id': AID, 'user_id': OWNER}]
        if method == 'POST' and path == 'analyses':
            state['saved_rows'].append(copy.deepcopy(kwargs['json_body']))
            return None
        raise AssertionError('Unexpected synthetic database operation')
    def artifacts(aid):
        state['artifact_reads'] += 1
        return {'input_bytes': b'synthetic-image', 'meta': {'analysis_id': aid,
                'user_id': OWNER, 'species': 'Old', 'height_m': 10},
                'source': 'synthetic', 'tmp_dir': tmp_path / 'synthetic'}
    def optional(bucket, path):
        if path.endswith('/meta_verified.json') and state['existing']:
            return json.dumps(state['existing']).encode()
        return None
    def upload_json(bucket, path, value):
        state['uploads'].append({'bucket': bucket, 'path': path, 'json': copy.deepcopy(value)})
    def persist(owner, aid, artifacts, mask, proposal):
        state['proposals'].append(copy.deepcopy(proposal))
        return {'saved': True, 'correction_id': aid + '_' + 'a' * 64 + '.json',
                'review_status': 'submitted', 'workflow_version': 1}
    ns = {'Body': Body, 'Header': Header, 'Form': Form, 'File': File, 'Depends': Depends,
          'require_admin': lambda: None,
          'UploadFile': UploadFile, 'Optional': Optional, 'HTTPException': HTTPException,
          'JSONResponse': JSONResponse, 'run_in_threadpool': run_in_threadpool,
          'Image': Image, 'ImageOps': ImageOps, 'UnidentifiedImageError': UnidentifiedImageError,
          'io': io, 'math': math, 'cv2': cv2, 'np': np, 're': re, 'json': json,
          'base64': base64, 'time': time, 'uuid4': uuid4,
          'Path': lambda p: tmp_path if p == '/tmp' else Path(p),
          'shutil': types.SimpleNamespace(rmtree=lambda *a, **k: None),
          'SUPABASE_URL': 'https://synthetic.invalid', 'SUPABASE_SERVICE_KEY': 'synthetic-only',
          'SUPABASE_DB_BASE': 'https://synthetic.invalid/rest/v1', 'SUPABASE_ENABLE_QUEUE': False,
          'SUPABASE_BUCKET_RAW': 'raw', 'SUPABASE_BUCKET_INPUTS': 'inputs',
          'SUPABASE_BUCKET_META': 'meta', 'SUPABASE_BUCKET_VERIFIED': 'verified',
          'VERIFIED_TRUST_THRESHOLD': .7, 'MAX_LEGACY_IMAGE_BYTES': 15 * 1024 * 1024,
          'MAX_LEGACY_PIXELS': 25_000_000, '_now_iso': lambda: '2026-11-12T00:00:00Z',
          '_get_user_by_token': lambda token: copy.deepcopy(users.get(token)),
          '_db_json': db, '_load_feedback_artifacts': artifacts,
          '_download_optional': optional, 'supabase_upload_json': upload_json,
          '_persist_legacy_feedback': persist,
          'supabase_upload_bytes': lambda b, p, raw: state['uploads'].append({'bucket': b, 'path': p}),
          'PLANTNET_API_KEY': SECRET, 'PLANTNET_PROJECT': 'all', 'PLANTNET_LANG': 'ru',
          'PLANTNET_BASE_URL': 'https://synthetic.invalid/v2', 'PLANTNET_TIMEOUT_SEC': 10,
          'PLANTNET_TOP_K': 3, 'PLANTNET_MIN_SCORE': .05,
          'SPECIES_AERO_FACTOR': {}, 'SPECIES_STRENGTH_MPA': {},
          'SPECIES_REF_HEIGHT': {}, 'REAL_STICK_M': 1,
          'ENABLE_ENV_ANALYSIS': False, 'MODEL_VERSIONS': {}, 'BUILD_INFO': {},
          'SCHEMA_VERSION': 'legacy-test', 'API_VERSION': 'legacy-test'}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), ns)
    return ns, state


def feedback_client(ns):
    app = FastAPI()
    app.post('/feedback')(ns['send_feedback'])
    return TestClient(app)


@pytest.mark.parametrize('token,status', [(None, 401), ('invalid-session', 401),
                                         ('other-session', 404), ('admin-session', 404)])
def test_feedback_denies_before_reading_or_writing_another_analysis(tmp_path, token, status):
    ns, state = namespace(tmp_path)
    response = feedback_client(ns).post('/feedback', json={'analysis_id': AID},
        headers={'Authorization': 'Bearer ' + token} if token else {})
    assert response.status_code == status
    assert state['artifact_reads'] == 0 and state['uploads'] == []


def test_owner_feedback_remains_pending_and_never_creates_verified_sample(tmp_path):
    ns, state = namespace(tmp_path)
    response = feedback_client(ns).post('/feedback', headers={'Authorization': 'Bearer owner-session'},
        json={'analysis_id': AID, 'use_for_training': True, 'corrected_height_m': 17})
    assert response.status_code == 200
    assert response.json()['verified'] is False
    assert response.json()['status'] == 'saved_pending_review'
    assert not any(w['bucket'] == 'verified' for w in state['uploads'])
    assert state['uploads'] == []
    assert state['proposals'][0]['training_consent'] is True
    assert state['proposals'][0]['corrections']['height_m'] == 17
    assert response.json()['proposal_only'] is True and response.json()['corrections_applied'] is False


def test_legacy_owner_body_token_still_works_but_cannot_override_bad_header(tmp_path):
    ns, state = namespace(tmp_path)
    client = feedback_client(ns)
    body = {'analysis_id': AID, 'auth_token': 'owner-session'}
    assert client.post('/feedback', json=body).status_code == 200
    assert client.post('/feedback', json=body,
        headers={'Authorization': 'Bearer invalid-session'}).status_code == 401


def test_feedback_does_not_replace_existing_verified_content_or_decision(tmp_path):
    ns, state = namespace(tmp_path)
    state['existing'] = {'verified': True, 'species': 'Original approved species',
                         'height_m': 10, 'reviewer_id': OTHER, 'used_for_training': False}
    response = feedback_client(ns).post('/feedback', headers={'Authorization': 'Bearer owner-session'},
        json={'analysis_id': AID, 'corrected_height_m': 17, 'use_for_training': True})
    assert response.status_code == 200 and response.json()['verified'] is False
    verified = [w for w in state['uploads'] if w['bucket'] == 'verified']
    assert verified == [] and state['uploads'] == []


def test_absent_training_consent_is_not_silently_opted_in(tmp_path):
    ns, state = namespace(tmp_path)
    response = feedback_client(ns).post('/feedback', headers={'Authorization': 'Bearer owner-session'},
        json={'analysis_id': AID})
    assert response.status_code == 200 and response.json()['use_for_training'] is False
    assert state['proposals'][0]['training_consent'] is False


def test_feedback_owner_lookup_outage_is_safe_and_does_not_touch_artifacts(tmp_path):
    ns, state = namespace(tmp_path); state['db_failure'] = True
    response = feedback_client(ns).post('/feedback', headers={'Authorization': 'Bearer owner-session'},
        json={'analysis_id': AID})
    assert response.status_code == 503 and SECRET not in response.text
    assert state['artifact_reads'] == 0 and state['uploads'] == []


def test_legacy_provider_error_never_exposes_credential_in_health_or_identification(tmp_path, capsys):
    ns, _ = namespace(tmp_path)
    def fail(*args, **kwargs):
        raise requests.ConnectionError('transport URL?api-key=' + SECRET)
    ns['requests'] = types.SimpleNamespace(get=fail, post=fail,
        Timeout=requests.Timeout, RequestException=requests.RequestException)
    for result in [ns['_plantnet_health_sync'](),
                   ns['_run_classifier_sync'](np.zeros((32, 32, 3), np.uint8))]:
        assert SECRET not in json.dumps(result)
    assert SECRET not in capsys.readouterr().out


class Tensor:
    def __init__(self, value): self.value = np.asarray(value)
    def cpu(self): return self
    def numpy(self): return self.value
    def item(self): return self.value.item()
    def __getitem__(self, index): return Tensor(self.value[index])


class Boxes(list):
    def __init__(self, rows=()):
        super().__init__(types.SimpleNamespace(xyxy=[Tensor(row)], cls=Tensor([0])) for row in rows)
        self.xyxy = [Tensor(row) for row in rows]


def analysis_setup(tmp_path, detected=True):
    ns, state = namespace(tmp_path)
    pixels = np.zeros((128, 96), np.uint8); pixels[10:118, 36:59] = 1
    tree = types.SimpleNamespace(masks=types.SimpleNamespace(data=[Tensor(pixels)]) if detected else None,
                                 boxes=Boxes([[36, 10, 59, 118]]) if detected else Boxes())
    def infer(image, conf):
        state['inferences'] += 1
        return tree, types.SimpleNamespace(boxes=Boxes()), types.SimpleNamespace(boxes=Boxes([[5, 10, 20, 110]]))
    ns['_run_yolo_sync'] = infer
    ns['_run_classifier_sync'] = lambda image: {'display_name': 'Неизвестно', 'species_group': None}
    ns['extract_exif_scale'] = lambda *a: (None, None)
    ns['prepare_feedback_assets'] = lambda image, mask: {'annotated_base64': 'fixture',
        'image_base64': 'fixture', 'mask_base64': 'fixture', 'image_bytes': b'image',
        'annotated_bytes': b'overlay', 'mask_bytes': b'mask', 'width': 96, 'height': 128,
        'source_width': 96, 'source_height': 128}
    return ns, state


def call_analysis(ns, raw=None, **changes):
    if raw is None:
        b = io.BytesIO(); Image.new('RGB', (96, 128)).save(b, 'PNG'); raw = b.getvalue()
    signature = inspect.signature(ns['analyze_tree'])
    kwargs = {key: getattr(p.default, 'default', p.default)
              for key, p in signature.parameters.items() if key != 'file'}
    kwargs.update(changes)
    file = UploadFile(file=io.BytesIO(raw), filename='fixture.png')
    return asyncio.run(ns['analyze_tree'](file=file, **kwargs))


def test_legacy_upload_bytes_rejected_before_inference(tmp_path):
    ns, state = analysis_setup(tmp_path)
    with pytest.raises(HTTPException) as caught:
        call_analysis(ns, b'x' * (15 * 1024 * 1024 + 1))
    assert caught.value.status_code == 413 and state['inferences'] == 0


def test_legacy_pixel_bomb_header_rejected_before_decode(tmp_path, monkeypatch):
    ns, state = analysis_setup(tmp_path)
    b = io.BytesIO(); Image.new('RGB', (1, 1)).save(b, 'PNG'); raw = bytearray(b.getvalue())
    raw[16:24] = struct.pack('>II', 6000, 6000)
    raw[29:33] = struct.pack('>I', zlib.crc32(raw[12:29]))
    monkeypatch.setattr(cv2, 'imdecode', lambda *a, **k: pytest.fail('unbounded decode reached'))
    with pytest.raises(HTTPException) as caught: call_analysis(ns, bytes(raw))
    assert caught.value.status_code == 413 and state['inferences'] == 0


def test_no_tree_does_not_create_rectangular_prediction(tmp_path):
    ns, _ = analysis_setup(tmp_path, detected=False)
    with pytest.raises(HTTPException) as caught: call_analysis(ns)
    assert caught.value.status_code == 422


def test_unknown_scale_stays_unknown_despite_person_detection_and_species_defaults(tmp_path):
    ns, state = analysis_setup(tmp_path)
    result = json.loads(call_analysis(ns).body)
    assert all(result[k] is None for k in ['height_m', 'crown_width_m', 'trunk_diameter_m', 'scale_px_to_m'])
    assert result['beta']['available'] is False and result['beta']['beta_kg_s'] is None
    assert result['risk']['available'] is False and result['risk']['category'] is None
    assert result['analytic_wind_model']['available'] is False
    assert state['uploads'] == [] and state['saved_rows'] == []


def test_explicit_scale_retains_geometry_but_does_not_invent_beta_or_risk(tmp_path):
    ns, _ = analysis_setup(tmp_path)
    result = json.loads(call_analysis(ns, manual_scale=.1, manual_beta_kg_s=42).body)
    assert result['height_m'] > 0 and result['crown_width_m'] > 0
    assert result['beta']['beta_kg_s'] is None and result['risk']['index'] is None
    assert all(value is None for value in result['analytic_wind_model']['outputs'].values())


def test_legacy_ar_direct_input_does_not_set_scale_or_infer_other_sizes(tmp_path):
    ns, _ = analysis_setup(tmp_path)
    result = json.loads(call_analysis(ns, ar_height_m=8).body)
    assert result['height_m'] == 8
    assert result['scale_px_to_m'] is None and result['crown_width_m'] is None
    assert result['measurement_sources']['height_m'] == 'manual_legacy_ar'


def test_failed_legacy_report_write_is_not_reported_as_saved(tmp_path, capsys):
    ns, state = namespace(tmp_path); state['db_failure'] = True
    assert ns['_save_analysis_record']({'analysis_id': AID}, {'id': OWNER}) is False
    assert SECRET not in capsys.readouterr().out


def test_legacy_exif_failure_does_not_log_exception_contents(tmp_path, capsys):
    ns, _ = namespace(tmp_path)
    def fail(*a, **k):
        raise OSError('synthetic-private-exif-body-' + SECRET)
    ns['Image'] = types.SimpleNamespace(open=fail)
    assert ns['extract_exif_scale'](b'synthetic', 100) == (None, None)
    assert SECRET not in capsys.readouterr().out


@pytest.mark.parametrize('boundary', ['raw_upload', 'temporary_cache'])
def test_legacy_optional_storage_failure_does_not_log_exception_contents(tmp_path, capsys, boundary):
    ns, _ = analysis_setup(tmp_path)
    def fail(*a, **k):
        raise OSError('synthetic-private-storage-url?token=' + SECRET)
    if boundary == 'raw_upload':
        ns['supabase_upload_bytes'] = fail
    else:
        def path(value):
            if value == '/tmp':
                raise OSError('synthetic-private-cache-path-' + SECRET)
            return Path(value)
        ns['Path'] = path
    response = json.loads(call_analysis(ns, authorization='Bearer owner-session').body)
    assert response['analysis_id']
    assert SECRET not in capsys.readouterr().out


def test_legacy_admin_storage_list_failure_is_generic_even_for_admin(tmp_path, capsys):
    ns, _ = namespace(tmp_path)
    def fail(*a, **k):
        raise requests.ConnectionError('synthetic-private-service-key=' + SECRET)
    ns['supabase_list_objects'] = fail
    with pytest.raises(HTTPException) as error:
        ns['admin_verified_list'](admin={'id': OTHER, 'role': 'admin'})
    assert error.value.status_code == 503
    assert SECRET not in str(error.value.detail)
    assert SECRET not in capsys.readouterr().out


def test_confirmed_owner_record_has_consistent_saved_flag_and_guest_has_no_record(tmp_path):
    ns, state = namespace(tmp_path)
    assert ns['_save_analysis_record']({'analysis_id': AID, 'server_saved': False}, {'id': OWNER}) is True
    assert state['saved_rows'][0]['response_json']['server_saved'] is True
    assert ns['_save_analysis_record']({'analysis_id': AID}, None) is False
    assert len(state['saved_rows']) == 1


def test_legacy_kernel_size_is_bounded_before_model_work(tmp_path):
    ns, state = analysis_setup(tmp_path)
    with pytest.raises(HTTPException) as caught:
        call_analysis(ns, ai_smoothness=1_000_000)
    assert caught.value.status_code == 422 and state['inferences'] == 0


def test_mask_header_bomb_is_rejected_before_any_image_decode(tmp_path, monkeypatch):
    ns, state = namespace(tmp_path)
    b = io.BytesIO(); Image.new('L', (1, 1)).save(b, 'PNG'); raw = bytearray(b.getvalue())
    raw[16:24] = struct.pack('>II', 6000, 6000)
    raw[29:33] = struct.pack('>I', zlib.crc32(raw[12:29]))
    monkeypatch.setattr(cv2, 'imdecode', lambda *a, **k: pytest.fail('unbounded mask decode reached'))
    response = feedback_client(ns).post('/feedback', headers={'Authorization': 'Bearer owner-session'},
        json={'analysis_id': AID, 'user_mask_base64': base64.b64encode(raw).decode()})
    assert response.status_code == 413 and state['proposals'] == [] and state['uploads'] == []


@pytest.mark.parametrize('scale', [float('nan'), float('inf'), 1e308])
def test_nonfinite_or_overflowing_legacy_geometry_is_not_serialized(tmp_path, scale):
    ns, _ = analysis_setup(tmp_path)
    with pytest.raises(HTTPException) as caught: call_analysis(ns, manual_scale=scale)
    assert caught.value.status_code == 422


def test_legacy_decoder_preserves_oriented_coordinate_dimensions(tmp_path):
    ns, _ = namespace(tmp_path)
    raw = (SOURCE.parent / 'arborscan_app/test/fixtures/reference_exif6.jpg').read_bytes()
    assert ns['_decode_legacy_image'](raw).shape[:2] == (200, 100)


def test_client_role_or_acceptance_fields_cannot_approve_a_proposal(tmp_path):
    ns, state = namespace(tmp_path)
    response = feedback_client(ns).post('/feedback', headers={'Authorization': 'Bearer owner-session'},
        json={'analysis_id': AID, 'owner_id': OTHER, 'role': 'admin', 'verified': True,
              'eligible_for_training': True, 'review_status': 'accepted'})
    assert response.status_code == 200 and response.json()['verified'] is False
    assert response.json()['review_status'] == 'submitted'
    assert not any(key in state['proposals'][0] for key in ['role', 'owner_id', 'verified', 'review_status'])


def test_legacy_analysis_authorization_header_persists_only_the_confirmed_owner(tmp_path):
    ns, state = analysis_setup(tmp_path)
    app = FastAPI(); app.post('/analyze-tree')(ns['analyze_tree'])
    image = io.BytesIO(); Image.new('RGB', (96, 128)).save(image, 'PNG')
    response = TestClient(app).post('/analyze-tree',
        files={'file': ('fixture.png', image.getvalue(), 'image/png')},
        headers={'Authorization': 'Bearer owner-session'})
    assert response.status_code == 200 and response.json()['server_saved'] is True
    assert len(state['saved_rows']) == 1 and state['saved_rows'][0]['user_id'] == OWNER
    assert state['saved_rows'][0]['response_json']['server_saved'] is True
