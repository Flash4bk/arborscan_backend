import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from arborscan_v4 import model_comparison_api as comparison
from arborscan_v4.corrections_api import current_user


def photo(size=(30, 20), exif=False):
    image = Image.new('RGB', size, (10, 90, 180))
    stream = io.BytesIO()
    if exif:
        metadata = Image.Exif()
        metadata[274] = 6
        image.save(stream, format='JPEG', exif=metadata)
    else:
        image.save(stream, format='PNG')
    return stream.getvalue()


@pytest.fixture
def models(tmp_path, monkeypatch):
    folder = tmp_path / 'models'
    folder.mkdir()
    trusted = {}
    for version in [1, 2, 4]:
        p = folder / f'model_v{version}.pt'
        p.write_bytes(f'trusted-test-weight-{version}'.encode())
        trusted[version] = hashlib.sha256(p.read_bytes()).hexdigest()
    monkeypatch.setattr(comparison, 'TRUSTED_LEGACY_WEIGHTS', trusted)
    monkeypatch.setattr(comparison, 'settings', SimpleNamespace(model_dir=folder))
    monkeypatch.setattr(comparison, '_default_version', lambda: 4)
    return folder, trusted


@pytest.fixture
def client(models):
    app = FastAPI()
    app.include_router(comparison.router)
    app.dependency_overrides[comparison.current_admin] = lambda: 'test-admin'
    return TestClient(app)


def request(models):
    _, hashes = models
    return comparison._validated_request(2, 4, hashes[2], hashes[4], None, None)


def form(models, **extra):
    _, hashes = models
    return dict(baseline_version='2', candidate_version='4',
                baseline_sha256=hashes[2], candidate_sha256=hashes[4], **extra)


def job(models, raw):
    value = request(models)
    value.update(source_model_dir=str(models[0]), upload_sha256=hashlib.sha256(raw).hexdigest())
    return value


class FakeRuntime:
    def __init__(self, version, calls, clock=None):
        self.version, self.calls, self.clock = version, calls, clock
        self.model_names = {0: 'stick', 1: 'tree'} if version == 2 else {0: 'tree'}

    def infer(self, image, x, y):
        self.calls.append((self.version, image.shape, x, y))
        if self.clock is not None:
            self.clock[0] += 0.25
        mask = np.zeros(image.shape[:2], np.uint8)
        mask[1:4, 2:5] = 255
        return SimpleNamespace(detected=True, mask=mask, class_id=1 if self.version == 2 else 0,
                               confidence=0.8, warnings=[])


def test_every_route_has_server_admin_dependency():
    for route in comparison.router.routes:
        assert any(d.call is comparison.current_admin for d in route.dependant.dependencies)


def test_missing_auth_rejected_before_inventory_or_model_loading(monkeypatch):
    app = FastAPI()
    app.include_router(comparison.router)
    monkeypatch.setattr(comparison, '_inventory', lambda: pytest.fail('inventory before auth'))
    monkeypatch.setattr(comparison, '_compare_with_slot', lambda *a: pytest.fail('load before auth'))
    c = TestClient(app)
    assert c.get('/v4/model-comparison/models').status_code == 401
    response = c.post('/v4/model-comparison', data={
        'baseline_version': 2, 'candidate_version': 4,
        'baseline_sha256': 'a' * 64, 'candidate_sha256': 'b' * 64},
        files={'file': ('photo.png', photo(), 'image/png')})
    assert response.status_code == 401


def test_ordinary_user_cannot_compare_or_list(monkeypatch):
    from arborscan_v4 import corrections_api
    app = FastAPI()
    app.include_router(comparison.router)
    app.dependency_overrides[current_user] = lambda: 'ordinary-user'
    monkeypatch.setattr(corrections_api, '_config', lambda: ('https://selfhost.invalid', {}, 'private'))
    monkeypatch.setattr(corrections_api.requests, 'get', lambda *a, **k:
                        SimpleNamespace(status_code=200, json=lambda: [{'role': 'user'}]))
    monkeypatch.setattr(comparison, '_inventory', lambda: pytest.fail('ordinary user loaded inventory'))
    c = TestClient(app)
    assert c.get('/v4/model-comparison/models').status_code == 403
    assert c.post('/v4/model-comparison', data={
        'baseline_version': 2, 'candidate_version': 4,
        'baseline_sha256': 'a' * 64, 'candidate_sha256': 'b' * 64},
        files={'file': ('photo.png', photo(), 'image/png')}).status_code == 403


def test_list_only_curated_real_files_without_loading(client, models, monkeypatch):
    monkeypatch.setattr(comparison, '_new_runtime', lambda *a: pytest.fail('list loads runtime'))
    response = client.get('/v4/model-comparison/models')
    assert response.status_code == 200
    body = response.json()
    assert body['supported'] and body['default_version'] == 4
    assert [m['version'] for m in body['available']] == [1, 2, 4]
    assert all(m['source'] == 'local_saved_weights' for m in body['available'])
    (models[0] / 'model_v1.pt').write_bytes(b'changed')
    assert [m['version'] for m in client.get('/v4/model-comparison/models').json()['available']] == [2, 4]


def test_tampered_pin_and_changed_weight_rejected_before_load(client, models, monkeypatch):
    monkeypatch.setattr(comparison, '_worker_command', lambda *a: pytest.fail('untrusted model executed'))
    data = form(models)
    data['baseline_sha256'] = 'a' * 64
    assert client.post('/v4/model-comparison', data=data,
        files={'file': ('x.png', photo(), 'image/png')}).status_code == 409
    (models[0] / 'model_v2.pt').write_bytes(b'changed')
    assert client.post('/v4/model-comparison', data=form(models),
        files={'file': ('x.png', photo(), 'image/png')}).status_code == 409


def test_symlink_is_not_a_saved_model(models):
    folder, _ = models
    path = folder / 'model_v2.pt'
    original = folder / 'elsewhere.pt'
    path.rename(original)
    try:
        path.symlink_to(original)
    except OSError:
        pytest.skip('OS does not permit test symlinks')
    with pytest.raises(comparison.ComparisonError, match='path') as exc:
        comparison._verified_file(folder, 2)
    assert exc.value.status == 409


@pytest.mark.parametrize('extra', [dict(tap_x='0.4'), dict(tap_x='nan', tap_y='0.2'),
    dict(tap_x='inf', tap_y='0.3'), dict(tap_x='-0.1', tap_y='0.4'), dict(tap_x='0.2', tap_y='1.1')])
def test_bad_tap_rejected_without_worker(client, models, monkeypatch, extra):
    monkeypatch.setattr(comparison, '_supervise', lambda *a: pytest.fail('bad tap reached worker'))
    assert client.post('/v4/model-comparison', data=form(models, **extra),
        files={'file': ('x.png', photo(), 'image/png')}).status_code == 422


def test_distinct_and_known_versions_only(client, models):
    data = form(models)
    data['candidate_version'] = 2
    assert client.post('/v4/model-comparison', data=data,
        files={'file': ('x.png', photo(), 'image/png')}).status_code == 422
    data['candidate_version'] = 3
    assert client.post('/v4/model-comparison', data=data,
        files={'file': ('x.png', photo(), 'image/png')}).status_code == 404


def test_upload_limits_before_worker(client, models, monkeypatch):
    monkeypatch.setattr(comparison, 'MAX_UPLOAD_BYTES', 10)
    monkeypatch.setattr(comparison, '_supervise', lambda *a: pytest.fail('oversized input reached worker'))
    assert client.post('/v4/model-comparison', data=form(models),
        files={'file': ('x.png', b'x' * 11, 'image/png')}).status_code == 413
    assert client.post('/v4/model-comparison', data=form(models),
        files={'file': ('x.png', b'', 'image/png')}).status_code == 422


def test_busy_and_failure_release_slot(client, models, monkeypatch):
    comparison._comparison_slot.acquire()
    try:
        response = client.post('/v4/model-comparison', data=form(models),
            files={'file': ('x.png', photo(), 'image/png')})
        assert response.status_code == 429 and response.headers['retry-after'] == '5'
    finally:
        comparison._comparison_slot.release()
    def fail(*a):
        raise comparison.ComparisonError(409, 'Saved model changed')
    monkeypatch.setattr(comparison, '_supervise', fail)
    assert client.post('/v4/model-comparison', data=form(models),
        files={'file': ('x.png', photo(), 'image/png')}).status_code == 409
    assert comparison._comparison_slot.acquire(blocking=False)
    comparison._comparison_slot.release()


def test_exif_processing_transform_masks_tree_mapping_and_warmed_timing(models, monkeypatch):
    raw = photo((30, 20), exif=True)
    monkeypatch.setattr(comparison, 'PROCESSING_MAX_SIDE', 15)
    calls, clock = [], [0.0]
    monkeypatch.setattr(comparison.time, 'perf_counter', lambda: clock[0])
    def factory(path, version):
        clock[0] += 100  # Loading time is deliberately excluded from inference samples.
        return FakeRuntime(version, calls, clock)
    result = comparison._run_models(raw, job(models, raw), models[0], factory=factory)
    assert result['image']['original_width'] == 20 and result['image']['original_height'] == 30
    assert result['image']['width'] == 10 and result['image']['height'] == 15
    assert result['image']['processing_scale_x'] == result['image']['processing_scale_y'] == 0.5
    assert result['persisted'] is False and result['quality_metrics'] == comparison.QUALITY
    assert [r['tree_class_id'] for r in result['models']] == [1, 0]
    assert len(calls) == 8
    for row in result['models']:
        assert row['inference_timing']['samples_seconds'] == [0.25] * 3
        mask = Image.open(io.BytesIO(base64.b64decode(row['mask_png_base64'])))
        assert mask.mode == 'L' and mask.size == (10, 15)
        assert set(np.unique(np.asarray(mask))) == {0, 255}


def test_source_swap_during_load_is_409(models):
    raw = photo()
    def factory(path, version):
        runtime = FakeRuntime(version, [])
        path.write_bytes(b'changed during load')
        return runtime
    with pytest.raises(comparison.ComparisonError) as exc:
        comparison._run_models(raw, job(models, raw), models[0], factory=factory)
    assert exc.value.status == 409


@pytest.mark.parametrize('names', [{0: 'stick'}, {0: 'tree', 1: 'trees'}, {0: 'potted plant'}])
def test_no_guessed_tree_class(models, names):
    raw = photo()
    def factory(path, version):
        runtime = FakeRuntime(version, [])
        runtime.model_names = names
        return runtime
    with pytest.raises(comparison.ComparisonError) as exc:
        comparison._run_models(raw, job(models, raw), models[0], factory=factory)
    assert exc.value.status == 409


@pytest.mark.parametrize('bad', ['nan-confidence', 'nonbinary', 'shape', 'wrong-class'])
def test_invalid_prediction_is_rejected(models, bad):
    raw = photo()
    class InvalidRuntime(FakeRuntime):
        def infer(self, image, x, y):
            r = super().infer(image, x, y)
            if bad == 'nan-confidence': r.confidence = float('nan')
            if bad == 'nonbinary': r.mask[0, 0] = 127
            if bad == 'shape': r.mask = r.mask[1:]
            if bad == 'wrong-class': r.class_id = 0
            return r
    with pytest.raises(comparison.ComparisonError) as exc:
        comparison._run_models(raw, job(models, raw), models[0],
                               factory=lambda p, v: InvalidRuntime(v, []))
    assert exc.value.status == 502


def test_no_detection_is_empty_mask_without_fake_quality(models):
    raw = photo()
    class NoRuntime(FakeRuntime):
        def infer(self, *a):
            return SimpleNamespace(detected=False, mask=None, confidence=None, warnings=['no_tree_class_candidate'])
    result = comparison._run_models(raw, job(models, raw), models[0], factory=lambda p, v: NoRuntime(v, []))
    for row in result['models']:
        mask = np.asarray(Image.open(io.BytesIO(base64.b64decode(row['mask_png_base64']))))
        assert not row['detected'] and not mask.any() and row['confidence'] is None
    assert all(result['quality_metrics'][k] is None for k in ['precision', 'recall', 'map50', 'map50_95', 'mean_iou'])


def test_private_predict_wrapper_bounds_detections_without_default_model_mutation():
    received = {}
    def model(*args, **kwargs):
        received.update(kwargs)
        return 'result'
    proxy = comparison._BoundedModel(model)
    assert proxy('pixels', max_det=300, device='cpu') == 'result'
    assert received == {'max_det': 16, 'device': 'cpu'}
    assert proxy.model is model


def test_default_api_runtime_is_never_loaded_or_selected(models, monkeypatch):
    # The comparison core has no reference to the application-global vision;
    # tests use two independently created runtimes and preserve the source files.
    raw, calls, created = photo(), [], []
    default = SimpleNamespace(model_version=4, untouched='active-runtime-sentinel')
    monkeypatch.setitem(sys.modules, 'arborscan_v4.api', SimpleNamespace(vision=default))
    before = {p.name: p.read_bytes() for p in models[0].iterdir()}
    def factory(path, version):
        r = FakeRuntime(version, calls)
        created.append(r)
        return r
    comparison._run_models(raw, job(models, raw), models[0], factory=factory)
    assert len(created) == 2 and created[0] is not created[1]
    assert default.__dict__ == {'model_version': 4, 'untouched': 'active-runtime-sentinel'}
    assert before == {p.name: p.read_bytes() for p in models[0].iterdir()}


def test_worker_environment_excludes_server_credentials(models, tmp_path, monkeypatch):
    monkeypatch.setenv('SUPABASE_SERVICE_KEY', 'never-forward-this')
    monkeypatch.setenv('OPENWEATHER_API_KEY', 'another-private-key')
    env = comparison._worker_environment(tmp_path, models[0])
    assert 'SUPABASE_SERVICE_KEY' not in env and 'OPENWEATHER_API_KEY' not in env
    assert env['OMP_NUM_THREADS'] == '1' and env['YOLO_AUTOINSTALL'] == 'false'
    assert env['MODEL_DIR'] == str(models[0])


def test_ultralytics_cpu_device_selection_keeps_thread_limit(monkeypatch):
    torch = pytest.importorskip('torch')
    pytest.importorskip('ultralytics')
    import cv2
    from ultralytics import utils
    from ultralytics.utils import torch_utils
    previous_torch, previous_cv = torch.get_num_threads(), cv2.getNumThreads()
    monkeypatch.setattr(utils, 'NUM_THREADS', 8)
    monkeypatch.setattr(torch_utils, 'NUM_THREADS', 8)
    try:
        torch.set_num_threads(1)
        torch_utils.select_device('cpu', verbose=False)
        assert torch.get_num_threads() == 8  # Reproduce the library's reset.
        comparison._pin_runtime_threads()
        torch_utils.select_device('cpu', verbose=False)
        assert torch.get_num_threads() == cv2.getNumThreads() == 1
    finally:
        torch.set_num_threads(previous_torch)
        cv2.setNumThreads(previous_cv)


def test_real_child_timeout_killed_and_temp_removed(models, tmp_path, monkeypatch):
    observed = []
    def command(job_path, result_path):
        observed.append(job_path.parent)
        return [sys.executable, '-c', 'import time; time.sleep(30)']
    monkeypatch.setattr(comparison, '_worker_command', command)
    monkeypatch.setattr(comparison, 'WALL_TIMEOUT_SECONDS', 0.1)
    started = time.monotonic()
    with pytest.raises(comparison.ComparisonError) as exc:
        comparison._compare_with_slot(photo(), request(models))
    assert exc.value.status == 504 and time.monotonic() - started < 10
    assert observed and not observed[0].exists()
    assert comparison._comparison_slot.acquire(blocking=False)
    comparison._comparison_slot.release()


def test_bad_image_in_actual_child_rejected_before_tensor_load(client, models):
    response = client.post('/v4/model-comparison', data=form(models),
        files={'file': ('bad.jpg', b'not a valid image', 'image/jpeg')})
    assert response.status_code == 422


def test_image_pixel_limit(monkeypatch):
    monkeypatch.setattr(comparison, 'MAX_PIXELS', 100)
    with pytest.raises(comparison.ComparisonError) as exc:
        comparison._decode_frame(photo((11, 10)))
    assert exc.value.status == 422


def test_final_cap_keeps_original_1920_scene_pixels():
    assert comparison.PROCESSING_MAX_SIDE == 2048
    image, frame = comparison._decode_frame(photo((1920, 1080)))
    assert image.shape[:2] == (1080, 1920)
    assert frame['processing_scale_x'] == frame['processing_scale_y'] == 1
    assert frame['width'] == frame['original_width'] == 1920


def test_large_header_is_rejected_before_rgb_allocation(monkeypatch):
    import struct
    import zlib
    raw = bytearray(photo((1, 1)))
    raw[16:24] = struct.pack('>II', 5001, 5000)
    raw[29:33] = struct.pack('>I', zlib.crc32(raw[12:29]))
    converted = []
    def convert(*args, **kwargs):
        converted.append(True)
        raise AssertionError('over-limit pixels must not be allocated')
    monkeypatch.setattr(Image.Image, 'convert', convert)
    with pytest.raises(comparison.ComparisonError) as exc:
        comparison._decode_frame(bytes(raw))
    assert exc.value.status == 422 and not converted


def test_raw_child_error_is_never_exposed(models, monkeypatch):
    def command(job_path, result_path):
        payload = json.dumps({'error': {'status': 409, 'detail': 'private-path-and-secret'}})
        code = 'import pathlib;pathlib.Path(' + repr(str(result_path)) + ').write_text(' + repr(payload) + ')'
        return [sys.executable, '-c', code]
    monkeypatch.setattr(comparison, '_worker_command', command)
    with pytest.raises(comparison.ComparisonError) as exc:
        comparison._compare_with_slot(photo(), request(models))
    assert exc.value.status == 409 and 'private' not in exc.value.detail
