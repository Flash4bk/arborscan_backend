"""Admin-only, disposable comparisons of curated legacy weights; no activation."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import statistics
import subprocess
import sys
import tempfile
import threading
import time

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from config import settings
from .corrections_api import current_admin

SCHEMA = 'arborscan.model-comparison.v1'
TRUSTED_LEGACY_WEIGHTS = {
    1: '535e5295095325db3add8167f8f7bacc24643581bb15d15ff0c8ac809c03788f',
    2: 'c1fed324ff1621eaf3df3bb465f79d917fae09127ccabbced245badeea80375f',
    4: '626f62b97b05fd65275dc3f909c3aec090f6c9957c4ef9f12c4e743a1f75a2bb',
}
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_PIXELS = 25_000_000
MAX_WEIGHT_BYTES = 64 * 1024 * 1024
MAX_RESPONSE_BYTES = 24 * 1024 * 1024
PROCESSING_MAX_SIDE = 2048
MAX_DETECTIONS = 16
WARMUP_RUNS, TIMING_REPEATS = 1, 3
WALL_TIMEOUT_SECONDS, CPU_BUDGET_SECONDS = 120, 90
QUALITY = dict(precision=None, recall=None, map50=None, map50_95=None,
               mean_iou=None, reason='verified_reference_masks_required')
_comparison_slot = threading.BoundedSemaphore(1)
router = APIRouter(prefix='/v4/model-comparison', tags=['admin model comparison'])


class ComparisonError(Exception):
    def __init__(self, status, detail):
        self.status, self.detail = status, detail
        super().__init__(detail)


def _regular(path):
    try:
        meta = path.lstat()
    except OSError:
        raise ComparisonError(404, 'Saved model is unavailable') from None
    if (not stat.S_ISREG(meta.st_mode) or path.is_symlink()
            or getattr(meta, 'st_file_attributes', 0) & 1024):
        raise ComparisonError(409, 'Saved model path is not supported')
    if not 0 < meta.st_size <= MAX_WEIGHT_BYTES:
        raise ComparisonError(409, 'Saved model size is not supported')
    return meta


def _model_path(root, version):
    if version not in TRUSTED_LEGACY_WEIGHTS:
        raise ComparisonError(404, 'Saved model version is unavailable')
    root = Path(root).absolute()
    # Do not accept a linked model directory, a linked file or a path supplied
    # by the client. Docker read-only bind mounts are ordinary directories.
    if root.is_symlink() or (root.exists() and
            getattr(root.lstat(), 'st_file_attributes', 0) & 1024):
        raise ComparisonError(409, 'Saved model directory is not supported')
    path = root / f'model_v{version}.pt'
    _regular(path)
    if path.resolve().parent != root.resolve():
        raise ComparisonError(409, 'Saved model path is not supported')
    return path


def _verified_file(root, version, expected_sha=None, copy_to=None):
    path = _model_path(root, version)
    curated = TRUSTED_LEGACY_WEIGHTS[version]
    if expected_sha is not None and expected_sha != curated:
        raise ComparisonError(409, 'Model weights changed; refresh available models')
    digest = hashlib.sha256()
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    except OSError:
        raise ComparisonError(409, 'Saved model changed or became unavailable') from None
    destination = None
    try:
        with os.fdopen(fd, 'rb') as source:
            before = os.fstat(source.fileno())
            if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= MAX_WEIGHT_BYTES:
                raise ComparisonError(409, 'Saved model size is not supported')
            if copy_to is not None:
                destination = Path(copy_to).open('xb')
            total = 0
            while chunk := source.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_WEIGHT_BYTES:
                    raise ComparisonError(409, 'Saved model size changed')
                digest.update(chunk)
                if destination is not None:
                    destination.write(chunk)
            after = os.fstat(source.fileno())
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (
                    after.st_size, after.st_mtime_ns, after.st_ino):
                raise ComparisonError(409, 'Saved model changed during comparison')
    finally:
        if destination is not None:
            destination.close()
    if digest.hexdigest() != curated:
        raise ComparisonError(409, 'Saved model checksum is not trusted')
    return {'version': version, 'weights_sha256': curated,
            'source': 'local_saved_weights'}


def _default_version():
    # Import lazily: listing weights must not instantiate another runtime.
    from .api import vision
    return int(vision.model_version or settings.active_model_version or 4)


def _inventory():
    available = []
    for version in TRUSTED_LEGACY_WEIGHTS:
        try:
            available.append(_verified_file(settings.model_dir, version))
        except (ComparisonError, OSError):
            continue
    return {'schema': SCHEMA, 'supported': True,
            'default_version': _default_version(), 'available': available}


def _validated_request(baseline_version, candidate_version, baseline_sha,
                       candidate_sha, tap_x, tap_y):
    versions = [baseline_version, candidate_version]
    if any(v not in TRUSTED_LEGACY_WEIGHTS for v in versions):
        raise ComparisonError(404, 'Saved model version is unavailable')
    if baseline_version == candidate_version:
        raise ComparisonError(422, 'Choose two different saved models')
    if any(not re.fullmatch('[a-f0-9]{64}', h) for h in [baseline_sha, candidate_sha]):
        raise ComparisonError(422, 'Expected model checksums are required')
    if (tap_x is None) != (tap_y is None):
        raise ComparisonError(422, 'Supply both tree selection coordinates')
    if tap_x is not None and any(not math.isfinite(v) or not 0 <= v <= 1
                                 for v in [tap_x, tap_y]):
        raise ComparisonError(422, 'Tree selection coordinates must be between zero and one')
    return {'models': [{'version': v, 'weights_sha256': h} for v, h in
                       zip(versions, [baseline_sha, candidate_sha])],
            'tap_x': tap_x, 'tap_y': tap_y}


def _worker_command(job, result):
    return [sys.executable, '-m', 'arborscan_v4.model_comparison_worker',
            str(job), str(result)]


def _worker_environment(folder, model_root):
    # Keep runtime paths, never forward API tokens/passwords or provider keys.
    names = ['PATH', 'SYSTEMROOT', 'WINDIR', 'LD_LIBRARY_PATH', 'DYLD_LIBRARY_PATH',
             'PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV']
    env = {n: os.environ[n] for n in names if n in os.environ}
    project = str(Path(__file__).resolve().parent.parent)
    env['PYTHONPATH'] = project + os.pathsep + env.get('PYTHONPATH', '')
    env.update(MODEL_DIR=str(model_root), MODEL_CACHE_DIR=str(folder / 'runtime-cache'),
               YOLO_CONFIG_DIR=str(folder / 'runtime-config'),
               YOLO_AUTOINSTALL='false', CUDA_VISIBLE_DEVICES='-1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', TEMP=str(folder), TMP=str(folder),
               MODEL_QUALITY_ENABLED='0')
    return env


def _stop_worker(process):
    if process.poll() is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        finally:
            # Retain the semaphore/temp files until the actual child exits,
            # even when it exits between poll and kill.
            process.wait()


def _supervise(raw, request):
    with tempfile.TemporaryDirectory(prefix='arborscan-model-comparison-') as temp:
        folder = Path(temp)
        for name in ['weights', 'runtime-config', 'runtime-cache']:
            (folder / name).mkdir(mode=0o700)
        original_root = Path(settings.model_dir).absolute()
        for selected in request['models']:
            _verified_file(original_root, selected['version'], selected['weights_sha256'],
                           folder / 'weights' / f"model_v{selected['version']}.pt")
        (folder / 'image.bin').write_bytes(raw)
        job = {**request, 'upload_sha256': hashlib.sha256(raw).hexdigest(),
               'source_model_dir': str(original_root)}
        job_path, result_path = folder / 'job.json', folder / 'result.json'
        job_path.write_text(json.dumps(job, allow_nan=False), encoding='utf-8')
        process = None
        try:
            process = subprocess.Popen(_worker_command(job_path, result_path), cwd=folder,
                env=_worker_environment(folder, folder / 'weights'),
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                **({'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}))
            try:
                process.wait(timeout=WALL_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                _stop_worker(process)
                raise ComparisonError(504, 'Model comparison exceeded its time limit') from None
            if not result_path.is_file() or result_path.stat().st_size > MAX_RESPONSE_BYTES:
                raise ComparisonError(503, 'Model comparison could not finish')
            try:
                value = json.loads(result_path.read_text(encoding='utf-8'))
            except (ValueError, UnicodeError):
                raise ComparisonError(502, 'Model comparison returned an invalid result') from None
            if isinstance(value, dict) and 'error' in value:
                error = value['error']
                messages = {
                    404: 'Saved model version is unavailable',
                    409: 'Saved model or image changed; refresh and retry',
                    413: 'Comparison image or preview exceeds its size limit',
                    422: 'Choose a valid image within the 25 megapixel limit',
                    504: 'Model comparison exceeded its time limit',
                }
                if isinstance(error, dict) and error.get('status') in messages:
                    # Never echo a raw child exception or filesystem path.
                    raise ComparisonError(error['status'], messages[error['status']])
                raise ComparisonError(503, 'Model comparison could not finish')
            if (process.returncode or not isinstance(value, dict) or value.get('schema') != SCHEMA
                    or value.get('persisted') is not False or value.get('quality_metrics') != QUALITY
                    or value.get('image', {}).get('upload_sha256') != job['upload_sha256']
                    or [(m.get('version'), m.get('weights_sha256')) for m in value.get('models', [])]
                    != [(m['version'], m['weights_sha256']) for m in request['models']]):
                raise ComparisonError(502, 'Model comparison returned an invalid result')
            return value
        finally:
            if process is not None:
                _stop_worker(process)


def _compare_with_slot(raw, request):
    if not _comparison_slot.acquire(blocking=False):
        raise ComparisonError(429, 'A model comparison is already running; retry later')
    try:
        return _supervise(raw, request)
    finally:
        _comparison_slot.release()


@router.get('/models')
async def list_models(admin: str = Depends(current_admin)):
    return await run_in_threadpool(_inventory)


@router.post('')
async def compare_models(file: UploadFile = File(...), baseline_version: int = Form(...),
        candidate_version: int = Form(...), baseline_sha256: str = Form(...),
        candidate_sha256: str = Form(...), tap_x: float | None = Form(None),
        tap_y: float | None = Form(None), admin: str = Depends(current_admin)):
    try:
        request = _validated_request(baseline_version, candidate_version,
            baseline_sha256, candidate_sha256, tap_x, tap_y)
        raw = await file.read(MAX_UPLOAD_BYTES + 1)
        if not raw:
            raise ComparisonError(422, 'Choose a valid image')
        if len(raw) > MAX_UPLOAD_BYTES:
            raise ComparisonError(413, 'Image exceeds the 15 MiB upload limit')
        return await run_in_threadpool(_compare_with_slot, raw, request)
    except ComparisonError as exc:
        raise HTTPException(exc.status, exc.detail,
                            headers={'Retry-After': '5'} if exc.status == 429 else None) from None
    except (OSError, ValueError, TypeError, KeyError):
        raise HTTPException(503, 'Model comparison is temporarily unavailable') from None


def _decode_frame(raw):
    import cv2
    from .measurement_image import decode_oriented_rgb
    try:
        rgb = decode_oriented_rgb(raw, MAX_PIXELS)
    except Exception:
        raise ComparisonError(422, 'Choose a valid image within the 25 megapixel limit') from None
    original_h, original_w = rgb.shape[:2]
    scale = min(1.0, PROCESSING_MAX_SIDE / max(original_h, original_w))
    width, height = max(1, round(original_w * scale)), max(1, round(original_h * scale))
    if (width, height) != (original_w, original_h):
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), {
        'width': width, 'height': height, 'original_width': original_w,
        'original_height': original_h, 'upload_sha256': hashlib.sha256(raw).hexdigest(),
        'preview_transform': 'resize_only', 'processing_scale_x': width / original_w,
        'processing_scale_y': height / original_h}


def _tree_class(names):
    targets = [int(k) for k, v in names.items()
               if str(v).strip().casefold() in {'tree', 'trees', 'дерево', 'деревья'}]
    if len(targets) != 1:
        raise ComparisonError(409, 'Saved model has no unambiguous tree class')
    return targets[0]


class _BoundedModel:
    def __init__(self, model):
        self.model = model

    def __call__(self, *args, **kwargs):
        return self.model(*args, **{**kwargs, 'max_det': MAX_DETECTIONS})


def _pin_runtime_threads():
    import cv2
    import torch
    from ultralytics import utils
    from ultralytics.utils import torch_utils
    # select_device('cpu') resets Torch to the library's NUM_THREADS on first
    # predict, so setting Torch alone before importing YOLO is insufficient.
    # This runs only in our disposable child, not the API's default runtime.
    utils.NUM_THREADS = torch_utils.NUM_THREADS = 1
    torch.set_num_threads(1)
    cv2.setNumThreads(1)


def _native_thread_counts():
    import cv2
    import torch
    counts = {'torch_threads': torch.get_num_threads(),
              'torch_interop_threads': torch.get_num_interop_threads(),
              'opencv_threads': cv2.getNumThreads()}
    if any(v != 1 for v in counts.values()):
        raise ComparisonError(503, 'Comparison CPU thread control is unavailable')
    return counts


def _new_runtime(path, version):
    import cv2
    from .vision_engine import TreeVisionRuntime
    runtime = TreeVisionRuntime()
    runtime.conf, runtime.imgsz = 0.25, 1024
    runtime._select_model_path = lambda: (path, version)
    runtime.load()
    # Ultralytics can initialise its own OpenCV defaults during import/load.
    # This function is used only in the disposable child, never global vision.
    _pin_runtime_threads()
    runtime.model = _BoundedModel(runtime.model)
    return runtime


def _run_models(raw, job, staged_root, cpu_limit_enforced=False, factory=None):
    import cv2
    import numpy as np
    from PIL import Image
    image, frame = _decode_frame(raw)
    if frame['upload_sha256'] != job['upload_sha256']:
        raise ComparisonError(409, 'Comparison image changed')
    real_runtime = factory is None
    factory = factory or _new_runtime
    rows = []
    for selected in job['models']:
        version, pinned = selected['version'], selected['weights_sha256']
        _verified_file(job['source_model_dir'], version, pinned)
        _verified_file(staged_root, version, pinned)
        path = _model_path(staged_root, version)
        runtime = factory(path, version)
        _verified_file(job['source_model_dir'], version, pinned)
        _verified_file(staged_root, version, pinned)
        tree_id = _tree_class(runtime.model_names)
        for _ in range(WARMUP_RUNS):
            runtime.infer(image, job['tap_x'], job['tap_y'])
            if real_runtime:
                _native_thread_counts()
        samples = []
        for _ in range(TIMING_REPEATS):
            started = time.perf_counter()
            prediction = runtime.infer(image, job['tap_x'], job['tap_y'])
            samples.append(time.perf_counter() - started)
            if real_runtime:
                _native_thread_counts()
        if any(not math.isfinite(v) or v < 0 for v in samples):
            raise ComparisonError(502, 'Saved model returned an invalid timing')
        if prediction.detected:
            mask = np.asarray(prediction.mask)
            confidence = float(prediction.confidence)
            if (mask.shape != image.shape[:2] or not np.isin(mask, [0, 255]).all()
                    or not np.any(mask)
                    or not math.isfinite(confidence) or not 0 <= confidence <= 1
                    or prediction.class_id != tree_id):
                raise ComparisonError(502, 'Saved model returned an invalid mask')
            mask = mask.astype(np.uint8)
        else:
            mask, confidence = np.zeros(image.shape[:2], np.uint8), None
        png = io.BytesIO()
        Image.fromarray(mask).save(png, format='PNG')
        rows.append({'version': version, 'weights_sha256': pinned, 'tree_class_id': tree_id,
            'detected': bool(prediction.detected), 'confidence': confidence,
            'mask_png_base64': base64.b64encode(png.getvalue()).decode('ascii'),
            'inference_timing': {'mean_seconds': statistics.mean(samples),
                'median_seconds': statistics.median(samples), 'samples_seconds': samples},
            'warnings': [str(w)[:160] for w in prediction.warnings[:20]]})
        del runtime
    result = {'schema': SCHEMA, 'persisted': False, 'image': frame,
        'settings': {'imgsz': 1024, 'confidence': 0.25, 'device': 'cpu', 'threads': 1,
            **(_native_thread_counts() if real_runtime else {'opencv_threads': cv2.getNumThreads()}),
            'warmup_runs': WARMUP_RUNS, 'timing_repeats': TIMING_REPEATS,
            'processing_max_side': PROCESSING_MAX_SIDE, 'max_detections': MAX_DETECTIONS,
            'wall_timeout_seconds': WALL_TIMEOUT_SECONDS,
            'cpu_budget_seconds': CPU_BUDGET_SECONDS if cpu_limit_enforced else None,
            'cpu_limit_enforced': cpu_limit_enforced,
            'timing_scope': 'warmed_runtime_infer_including_mask_selection_and_resize'},
        'models': rows, 'quality_metrics': dict(QUALITY)}
    if len(json.dumps(result, allow_nan=False).encode()) > MAX_RESPONSE_BYTES:
        raise ComparisonError(413, 'Comparison preview exceeds the response limit')
    return result
