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


def _validate_and_save(owner, analysis_id, image_bytes, mask_bytes):
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
    revision = hashlib.sha256((analysis_id + ":" + image_hash + ":" + mask_hash).encode()).hexdigest()
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
    return {"saved": True, "correction_id": key, **_summary(_put(owner, key, record))}


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
    return {"correction_id": correction_id, **record}
