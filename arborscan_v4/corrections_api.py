"""Private, immutable contour submissions. No training or verification writes."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from uuid import UUID
from urllib.parse import quote

import cv2
import numpy as np
import requests
from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
try:
    from .correction_workflow import WorkflowStore, validate_editor, overlay
except ImportError:
    from correction_workflow import WorkflowStore, validate_editor, overlay

router = APIRouter(prefix="/v4/corrections", tags=["contour corrections"])
MAX_IMAGE = 15 * 1024 * 1024
MAX_MASK = 8 * 1024 * 1024
MAX_PIXELS = 25_000_000
KEY_RE = re.compile(r"^[0-9a-f-]{36}_[0-9a-f]{64}\.json$")


def _config():
    from config import settings
    if not settings.supabase_url or not settings.supabase_service_key:
        raise HTTPException(503, "Correction storage is not configured")
    return settings.supabase_url.rstrip("/"), {
        "apikey": settings.supabase_service_key,
        "Authorization": f"Bearer {settings.supabase_service_key}",
    }, os.getenv("V4_CORRECTIONS_BUCKET", "arborscan-corrections")


def _uuid(value):
    try:
        return str(UUID(str(value)))
    except (ValueError, TypeError, AttributeError):
        raise HTTPException(422, "Invalid analysis or user identifier") from None


def current_user(authorization: str | None = Header(default=None)):
    parts = (authorization or "").split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(401, "Sign in before saving a correction")
    token = parts[1]
    if len(token) > 512 or not re.fullmatch(r"[A-Za-z0-9_-]+", token):
        raise HTTPException(401, "Invalid session")
    url, headers, _ = _config()
    try:
        response = requests.get(url + "/rest/v1/auth_sessions", headers=headers, params={
            "token": "eq." + token,
            "expires_at": "gt." + datetime.now(timezone.utc).isoformat(),
            "select": "user_id,users(id)", "limit": "1",
        }, timeout=15)
        if response.status_code != 200:
            raise HTTPException(503, "Session service is unavailable")
        rows = response.json()
    except (requests.RequestException, ValueError):
        raise HTTPException(503, "Session service is unavailable") from None
    if not isinstance(rows, list) or not rows:
        raise HTTPException(401, "Session expired; sign in again")
    user = rows[0].get("users")
    if isinstance(user, list):
        user = user[0] if user else None
    if not isinstance(user, dict) or str(user.get("id")) != str(rows[0].get("user_id")):
        raise HTTPException(401, "Session user does not exist")
    _require_private_bucket()
    return _uuid(user["id"])


def _require_private_bucket():
    url, headers, bucket = _config()
    try:
        response = requests.get(url + "/storage/v1/bucket/" + quote(bucket, safe=""),
                                headers=headers, timeout=15)
        if response.status_code != 200:
            raise HTTPException(503, "Private correction bucket is not ready")
        info = response.json()
        if not isinstance(info, dict) or info.get("public") is not False:
            raise HTTPException(503, "Correction bucket must be private")
    except (requests.RequestException, ValueError):
        raise HTTPException(503, "Cannot verify correction storage privacy") from None


def _object_url(owner, key):
    if not KEY_RE.fullmatch(key):
        raise HTTPException(422, "Invalid correction identifier")
    url, headers, bucket = _config()
    path = f"v4-corrections/{_uuid(owner)}/{key}"
    return url + "/storage/v1/object/" + quote(bucket, safe="") + "/" + quote(path, safe="/"), headers


def _get(owner, key):
    url, headers = _object_url(owner, key)
    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code in (400, 404):
            raise HTTPException(404, "Correction not found")
        if response.status_code != 200:
            raise HTTPException(503, "Correction storage is unavailable")
        record = response.json()
    except (requests.RequestException, ValueError):
        raise HTTPException(503, "Correction storage is unavailable") from None
    if not isinstance(record, dict) or record.get("owner_id") != owner:
        raise HTTPException(503, "Invalid stored correction")
    return record


def _put(owner, key, record):
    url, headers = _object_url(owner, key)
    payload = json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode()
    try:
        response = requests.post(url, headers={**headers,
            "Content-Type": "application/json", "x-upsert": "false"}, data=payload, timeout=60)
        if response.status_code not in (200, 201, 400, 409):
            raise HTTPException(503, "Correction was not confirmed as saved; retry")
        # Confirm durable storage even when this was a retry of the same content.
        stored = _get(owner, key)
        for field in ("analysis_id", "image_sha256", "mask_sha256", "revision"):
            if stored.get(field) != record[field]:
                raise HTTPException(503, "Stored correction does not match submission")
        return stored
    except requests.RequestException:
        raise HTTPException(503, "Correction was not confirmed as saved; retry") from None


def _summary(record):
    return {k: record[k] for k in ("analysis_id", "revision", "created_at", "review_status",
        "image_sha256", "mask_sha256", "width", "height")}


def _validate_and_save(owner, analysis_id, image_bytes, mask_bytes, editor_state=None, parent_id=None):
    analysis_id = _uuid(analysis_id)
    # Inspect headers before allocating decoded image arrays.
    from PIL import Image, UnidentifiedImageError
    from io import BytesIO
    try:
        for raw in (image_bytes, mask_bytes):
            with Image.open(BytesIO(raw)) as im:
                if im.width * im.height > MAX_PIXELS:
                    raise HTTPException(413, "Image resolution exceeds 25 megapixels")
        with Image.open(BytesIO(mask_bytes)) as im:
            if im.format != "PNG":
                raise HTTPException(422, "Mask must be PNG")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(422, "Image or mask cannot be decoded") from None
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    mask = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        raise HTTPException(422, "Image or mask cannot be decoded")
    if image.shape[:2] != mask.shape:
        raise HTTPException(422, "Mask dimensions must match the oriented photo")
    foreground = int(np.count_nonzero(mask > 127))
    if foreground == 0 or foreground == mask.size:
        raise HTTPException(422, "Mask is empty or covers the whole image")
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    mask_hash = hashlib.sha256(mask_bytes).hexdigest()
    state = validate_editor(editor_state, int(image.shape[1]), int(image.shape[0])) if editor_state is not None else None
    revision = hashlib.sha256((analysis_id + ":" + image_hash + ":" + mask_hash).encode()).hexdigest()
    if state is not None:
        revision = hashlib.sha256(json.dumps([analysis_id, image_hash, mask_hash, state, parent_id],
            sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    key = f"{analysis_id}_{revision}.json"
    record = {
        "schema_version": 1, "owner_id": owner, "analysis_id": analysis_id,
        "revision": revision, "created_at": datetime.now(timezone.utc).isoformat(),
        "review_status": "pending_review", "eligible_for_training": False,
        "provenance": "user_submission_analysis_link_not_server_verified",
        "image_sha256": image_hash, "mask_sha256": mask_hash,
        "width": int(image.shape[1]), "height": int(image.shape[0]),
        "original_image_base64": base64.b64encode(image_bytes).decode(),
        "mask_png_base64": base64.b64encode(mask_bytes).decode(),
    }
    if state is not None:
        store = WorkflowStore(_config)
        store.ready()
        if parent_id:
            parent = _get(owner, parent_id)
            if parent['analysis_id'] != analysis_id or parent['image_sha256'] != image_hash:
                raise HTTPException(422, 'Parent must refer to the same original image and analysis')
            _ensure_metadata(store, owner, parent_id, parent)
        record.update(schema_version=2, editor_state=state, parent_id=parent_id,
                      original_ref={'image_sha256':image_hash}, review_status='draft')
        record = _put(owner, key, record)
        metadata = store.transition('register', owner, key, analysis=analysis_id,
                                    image=image_hash, parent=parent_id)
        return {'saved':True, 'correction_id':key, **_summary(overlay(record, metadata)),
                'workflow_version':1, 'parent_id':parent_id}
    return {"saved": True, "correction_id": key, **_summary(_put(owner, key, record))}


def _ensure_metadata(store, owner, key, record):
    metadata = store.get(owner, key)
    if metadata: return metadata
    if record.get('schema_version') == 2:
        # An unregistered blob may be left after a conflicting or timed-out save.
        raise HTTPException(409, 'Revision not committed; retry its original save request')
    return store.transition('legacy', owner, key, analysis=record['analysis_id'], image=record['image_sha256'])


def current_admin(owner: str = Depends(current_user)):
    url, headers, _ = _config()
    try:
        res = requests.get(url+'/rest/v1/users', headers=headers,
                           params={'id':'eq.'+owner,'select':'role','limit':'1'}, timeout=15)
        if res.status_code != 200: raise HTTPException(503, 'Role service unavailable')
        rows = res.json()
        if not rows or str(rows[0].get('role','')).strip().lower() != 'admin':
            raise HTTPException(403, 'Admin required')
    except (requests.RequestException, ValueError):
        raise HTTPException(503, 'Role service unavailable') from None
    return owner


@router.get('/workflow/capabilities')
def capabilities(owner: str = Depends(current_user)):
    WorkflowStore(_config).ready()
    return {'workflow_version':1, 'editor_version':1, 'owner_id':owner}


@router.post('/workflow')
async def save_workflow(analysis_id: str = Form(...), image: UploadFile = File(...),
                        mask: UploadFile = File(...), editor_state: str = Form(...),
                        parent_id: str | None = Form(default=None), owner: str = Depends(current_user)):
    raw_image = await image.read(MAX_IMAGE+1)
    raw_mask = await mask.read(MAX_MASK+1)
    if len(raw_image)>MAX_IMAGE or len(raw_mask)>MAX_MASK:
        raise HTTPException(413, 'Photo or mask exceeds upload size limit')
    if not raw_image or not raw_mask: raise HTTPException(422, 'Photo and mask required')
    if parent_id and not KEY_RE.fullmatch(parent_id): raise HTTPException(422, 'Invalid parent')
    return await run_in_threadpool(_validate_and_save, owner, analysis_id, raw_image, raw_mask, editor_state, parent_id)


@router.post('/{correction_id}/submit')
def submit(correction_id: str, owner: str = Depends(current_user)):
    record = _get(owner, correction_id)
    store = WorkflowStore(_config)
    _ensure_metadata(store, owner, correction_id, record)
    return overlay(record, store.transition('submit', owner, correction_id)) | {'correction_id':correction_id}


@router.get('/workflow/queue')
def moderation_queue(offset: int = 0, admin: str = Depends(current_admin)):
    if offset < 0: raise HTTPException(422, 'Invalid offset')
    return WorkflowStore(_config).queue(offset)


@router.get('/workflow/review/{owner_id}/{correction_id}')
def review_detail(owner_id: str, correction_id: str, admin: str = Depends(current_admin)):
    owner_id = _uuid(owner_id)
    record = _get(owner_id, correction_id)
    metadata = WorkflowStore(_config).get(owner_id, correction_id)
    if not metadata: raise HTTPException(404, 'Revision not submitted')
    return overlay(record, metadata) | {'correction_id':correction_id}


class Decision(BaseModel):
    decision: str = Field(max_length=20)
    reason: str = Field(default='', max_length=2000)


@router.post('/workflow/review/{owner_id}/{correction_id}')
def decide(owner_id: str, correction_id: str, decision: Decision, admin: str = Depends(current_admin)):
    owner_id = _uuid(owner_id)
    if not KEY_RE.fullmatch(correction_id): raise HTTPException(422, 'Invalid correction')
    if decision.decision not in ('accepted','rejected') or (decision.decision=='rejected' and not decision.reason.strip()):
        raise HTTPException(422, 'Rejection requires a reason')
    return WorkflowStore(_config).transition('decide', owner_id, correction_id,
        actor=admin, decision=decision.decision, reason=decision.reason.strip())


@router.post("")
async def save_correction(analysis_id: str = Form(...),
                          image: UploadFile = File(...), mask: UploadFile = File(...),
                          owner: str = Depends(current_user)):
    image_bytes = await image.read(MAX_IMAGE + 1)
    mask_bytes = await mask.read(MAX_MASK + 1)
    if not image_bytes or not mask_bytes:
        raise HTTPException(422, "Photo and mask are required")
    if len(image_bytes) > MAX_IMAGE or len(mask_bytes) > MAX_MASK:
        raise HTTPException(413, "Photo or mask exceeds upload size limit")
    return await run_in_threadpool(_validate_and_save, owner, analysis_id, image_bytes, mask_bytes)


@router.get("")
def list_corrections(offset: int = 0, owner: str = Depends(current_user)):
    if offset < 0:
        raise HTTPException(422, "Offset must not be negative")
    url, headers, bucket = _config()
    try:
        response = requests.post(url + "/storage/v1/object/list/" + quote(bucket, safe=""),
            headers=headers, json={"prefix": f"v4-corrections/{owner}/", "limit": 50,
                                  "offset": offset, "sortBy": {"column": "name", "order": "asc"}}, timeout=30)
        if response.status_code != 200:
            raise HTTPException(503, "Correction storage is unavailable")
        rows = response.json()
        if not isinstance(rows, list):
            raise HTTPException(503, "Invalid correction listing")
    except (requests.RequestException, ValueError):
        raise HTTPException(503, "Correction storage is unavailable") from None
    return {"items": [{"correction_id": row["name"], "created_at": row.get("created_at")}
             for row in rows if isinstance(row, dict) and KEY_RE.fullmatch(str(row.get("name", "")))],
            "next_offset": offset + len(rows) if len(rows) == 50 else None}


@router.get("/{correction_id}")
def get_correction(correction_id: str, owner: str = Depends(current_user)):
    record = _get(owner, correction_id)
    try:
        metadata = WorkflowStore(_config).get(owner, correction_id)
        if metadata: record = overlay(record, metadata)
        elif record.get('schema_version') == 2: raise HTTPException(409, 'Revision save not committed')
    except HTTPException as exc:
        if exc.status_code != 503: raise
        if record.get('schema_version') == 2: raise
        record = {**record, 'workflow_available':False}
    return {"correction_id": correction_id, **record}
