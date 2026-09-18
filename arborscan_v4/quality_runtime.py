"""Per-request immutable runtime selection, shared SQL activation pointer."""
import hashlib
import os
import threading
from pathlib import Path
from uuid import UUID
import numpy as np
from fastapi import HTTPException

_cache={}
_lock=threading.RLock()


def model_path(id):
    name=str(UUID(str(id)))
    return Path(os.getenv('MODEL_QUALITY_DIR','/app/model-quality'))/'jobs'/name/'candidate.pt'


def checked_runtime(row):
    from .vision_engine import TreeVisionRuntime
    meta=row['metadata'];path=model_path(row['id'])
    if row['model_type']!='segmentation' or meta.get('class_names')!=['tree'] or meta.get('preprocessing')!='exif_rgb_to_bgr_v1':
        raise HTTPException(409,'Model contract incompatible')
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=meta.get('weights_sha256'):
        raise HTTPException(409,'Model missing or checksum mismatch')
    with _lock:
        if row['id'] not in _cache:
            r=TreeVisionRuntime();r._select_model_path=lambda:(path,4)
            r.load()
            if r.model_names!={0:'tree'}:raise HTTPException(409,'Unexpected model classes')
            r.infer(np.zeros((96,64,3),dtype=np.uint8))
            r.quality_model_id=row['id'];r.quality_model_sha=meta['weights_sha256']
            if len(_cache)>1:_cache.clear()
            _cache[row['id']]=r
        return _cache[row['id']]


def select_runtime(default):
    if os.getenv('MODEL_QUALITY_ENABLED','0')!='1':return default
    from .model_quality_api import QualityStore
    store=QualityStore();rows=store.rows('ml_active_models',model_type='eq.segmentation')
    if not rows or rows[0]['model_id'] is None:return default
    # Fail closed if the shared registry is unavailable; never silently use
    # a stale model in one process while another uses the selected model.
    return checked_runtime(store.one('ml_models',rows[0]['model_id']))
