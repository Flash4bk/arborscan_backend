import os
import io
import base64
import json
import re
import shutil
import time
import threading
import hashlib
import secrets
import math
import cv2
import numpy as np
import requests
import torch
from ultralytics import YOLO
from PIL import Image, ExifTags
from fastapi import FastAPI, File, UploadFile, HTTPException, Body, Form, Header, Depends
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
from uuid import uuid4
from pathlib import Path
from pydantic import BaseModel
from datetime import datetime, timedelta
from collections import deque
from typing import Optional, Dict, Any, List, Tuple

from rembg import remove, new_session

from config import settings

try:
    from google.oauth2 import id_token as google_id_token
    from google.auth.transport import requests as google_requests
except Exception:
    google_id_token = None
    google_requests = None

# -------------------------------------
# CONFIG
# -------------------------------------

settings.ensure_runtime_dirs()

PROJECT_ROOT = settings.project_root
MODEL_DIR = settings.model_dir
MODEL_CACHE_DIR = settings.model_cache_dir

SUPABASE_URL = settings.supabase_url
SUPABASE_SERVICE_KEY = settings.supabase_service_key

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    print("[!] Warning: SUPABASE_URL or Supabase service key is not set.")

SUPABASE_BUCKET_INPUTS = settings.supabase_bucket_inputs
SUPABASE_BUCKET_PRED = settings.supabase_bucket_predictions
SUPABASE_BUCKET_META = settings.supabase_bucket_meta
SUPABASE_BUCKET_VERIFIED = settings.supabase_bucket_verified
SUPABASE_BUCKET_RAW = settings.supabase_bucket_raw
SUPABASE_BUCKET_MODELS = settings.supabase_bucket_models

SUPABASE_DB_BASE = (
    SUPABASE_URL.rstrip("/") + "/rest/v1" if SUPABASE_URL else None
)
SUPABASE_QUEUE_TABLE = "arborscan_feedback_queue"
SUPABASE_ENABLE_QUEUE = settings.supabase_enable_queue

PLANTNET_API_KEY = settings.plantnet_api_key

NOMINATIM_URL = settings.nominatim_base_url
NOMINATIM_USER_AGENT = settings.nominatim_user_agent
ENABLE_ENV_ANALYSIS = settings.enable_environmental_analysis


def _sb_headers(json_ct: bool = True) -> dict:
    headers = {
        "apikey": SUPABASE_SERVICE_KEY or "",
        "Authorization": (
            f"Bearer {SUPABASE_SERVICE_KEY}" if SUPABASE_SERVICE_KEY else ""
        ),
    }
    if json_ct:
        headers["Content-Type"] = "application/json"
    return headers


def _supabase_is_configured() -> bool:
    return bool(SUPABASE_URL and SUPABASE_SERVICE_KEY and SUPABASE_DB_BASE)


def training_state_get() -> dict:
    if not _supabase_is_configured():
        raise RuntimeError("Supabase DB is not configured")

    response = requests.get(
        f"{SUPABASE_DB_BASE}/training_state?id=eq.1&select=*",
        headers=_sb_headers(json_ct=False),
        timeout=30,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"training_state_get error {response.status_code}: {response.text}"
        )

    rows = response.json()
    return rows[0] if rows else {}


def training_state_ensure_row(default_active_version: int = 0) -> dict:
    if not _supabase_is_configured():
        return {}

    state = training_state_get()
    if state:
        return state

    payload = {
        "id": 1,
        "retrain_requested": False,
        "training_in_progress": False,
        "last_model_version": int(default_active_version),
        "active_model_version": int(default_active_version),
    }
    response = requests.post(
        f"{SUPABASE_DB_BASE}/training_state",
        headers={**_sb_headers(), "Prefer": "return=representation"},
        data=json.dumps(payload),
        timeout=30,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"training_state_ensure_row error {response.status_code}: "
            f"{response.text}"
        )

    rows = response.json()
    return rows[0] if rows else payload


def training_state_update(fields: dict) -> dict:
    if not _supabase_is_configured():
        raise RuntimeError("Supabase DB is not configured")

    response = requests.patch(
        f"{SUPABASE_DB_BASE}/training_state?id=eq.1",
        headers={**_sb_headers(), "Prefer": "return=representation"},
        data=json.dumps(fields),
        timeout=30,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"training_state_update error {response.status_code}: "
            f"{response.text}"
        )

    rows = response.json()
    return rows[0] if rows else fields


MODEL_VERSIONS = {
    "tree_yolo": "tree_yolov8_seg_versioned",
    "stick_yolo": "stick_yolov8_det_v1.0.3",
    "classifier": "plantnet_api_v2",
    "mask_refiner": "u2net_rembg_solid_v2",
}
BUILD_INFO = {
    "git_commit": os.getenv("GIT_COMMIT", "unknown"),
    "build_time": os.getenv("BUILD_TIME"),
}
SCHEMA_VERSION = "1.0.0"
API_VERSION = "3.0.0"
VERIFIED_TRUST_THRESHOLD = float(os.getenv("VERIFIED_TRUST_THRESHOLD", "0.70"))

REAL_STICK_M = 1.0
CLASS_NAMES_RU = ["Береза", "Дуб", "Ель", "Сосна", "Тополь"]


# =============================================
# SUPABASE STORAGE HELPERS
# =============================================

def supabase_upload_bytes(bucket: str, path: str, data: bytes):
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise RuntimeError("Supabase Storage is not configured")

    response = requests.post(
        SUPABASE_URL.rstrip("/") + f"/storage/v1/object/{bucket}/{path}",
        headers={
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            "apikey": SUPABASE_SERVICE_KEY,
            "Content-Type": "application/octet-stream",
            "x-upsert": "true",
        },
        data=data,
        timeout=60,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"Supabase upload error {response.status_code}: {response.text}"
        )


def supabase_upload_json(bucket: str, path: str, obj: dict):
    supabase_upload_bytes(
        bucket,
        path,
        json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8"),
    )


def supabase_list_objects(
    bucket: str,
    prefix: str = "",
    *,
    page_size: int = 1000,
    max_items: Optional[int] = None,
) -> list[dict]:
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise RuntimeError("Supabase Storage is not configured")

    page_size = max(1, min(int(page_size), 1000))
    offset = 0
    items: list[dict] = []

    while True:
        response = requests.post(
            SUPABASE_URL.rstrip("/") + f"/storage/v1/object/list/{bucket}",
            headers=_sb_headers(),
            json={
                "prefix": prefix,
                "limit": page_size,
                "offset": offset,
                "sortBy": {"column": "name", "order": "desc"},
            },
            timeout=30,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Supabase list error {response.status_code}: {response.text}"
            )

        page = response.json()
        if not isinstance(page, list):
            raise RuntimeError("Supabase list returned a non-list response")

        items.extend(page)
        if max_items is not None and len(items) >= max_items:
            return items[:max_items]
        if len(page) < page_size:
            return items
        offset += page_size


def supabase_download_bytes(bucket: str, path: str) -> bytes:
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise RuntimeError("Supabase Storage is not configured")

    response = requests.get(
        SUPABASE_URL.rstrip("/")
        + f"/storage/v1/object/authenticated/{bucket}/{path}",
        headers=_sb_headers(False),
        timeout=settings.model_download_timeout_sec,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"Supabase download error {response.status_code}: {response.text}"
        )
    if not response.content:
        raise RuntimeError(f"Supabase returned an empty object: {bucket}/{path}")
    return response.content


# =============================================
# MODEL MANAGER
# =============================================

TREE_MODEL: Optional[YOLO] = None
TREE_MODEL_VERSION: Optional[int] = None
TREE_MODEL_PATH: Optional[str] = None
TREE_MODEL_SOURCE: Optional[str] = None
TREE_MODEL_LOADED_AT: Optional[str] = None

stick_model: Optional[YOLO] = None
STICK_MODEL_PATH: Optional[str] = None
STICK_MODEL_SOURCE: Optional[str] = None
STICK_MODEL_LOADED_AT: Optional[str] = None

MODEL_LOCK = threading.RLock()
REMBG_LOCK = threading.Lock()
REMBG_SESSION = None

_MODEL_LAST_CHECK_TS = 0.0
_MODEL_CHECK_INTERVAL_SEC = settings.model_check_interval_sec
_MODEL_LAST_ERROR: Optional[str] = None
_STICK_MODEL_LAST_ERROR: Optional[str] = None


def _utc_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_model_file(
    path: Path,
    *,
    expected_sha256: Optional[str] = None,
    label: str = "model",
) -> dict:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} file does not exist: {path}")
    if not path.is_file():
        raise RuntimeError(f"{label} path is not a file: {path}")

    size = path.stat().st_size
    if size < settings.model_min_size_bytes:
        raise RuntimeError(
            f"{label} is too small ({size} bytes): {path}. "
            "The file may be incomplete or contain an HTTP error response."
        )

    actual_sha256 = None
    if expected_sha256:
        actual_sha256 = _sha256_file(path)
        if actual_sha256.lower() != expected_sha256.lower():
            raise RuntimeError(
                f"{label} SHA-256 mismatch for {path}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )

    return {
        "path": str(path),
        "size_bytes": size,
        "sha256": actual_sha256,
    }


def _load_yolo_checked(
    path: Path,
    *,
    expected_task: Optional[str],
    expected_sha256: Optional[str],
    label: str,
) -> tuple[YOLO, dict]:
    file_info = _validate_model_file(
        path,
        expected_sha256=expected_sha256,
        label=label,
    )
    model = YOLO(str(path))
    task = getattr(model, "task", None)
    if expected_task and task and task != expected_task:
        raise RuntimeError(
            f"{label} has task '{task}', expected '{expected_task}': {path}"
        )
    file_info["task"] = task
    return model, file_info


def _tree_model_filename(version: int) -> str:
    return f"model_v{int(version)}.pt"


def _bundled_tree_candidates(version: int) -> list[Path]:
    candidates = [MODEL_DIR / _tree_model_filename(version)]
    if version == 0:
        candidates.extend(
            [
                MODEL_DIR / "tree_model.pt",
                MODEL_DIR / "base.pt",
            ]
        )
    return candidates


def _cache_tree_path(version: int) -> Path:
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return MODEL_CACHE_DIR / _tree_model_filename(version)


def _basic_model_file_is_valid(path: Path) -> bool:
    try:
        _validate_model_file(path, label="model")
        return True
    except Exception:
        return False


def _download_to_cache(
    *,
    bucket: str,
    object_name: str,
    destination: Path,
    expected_sha256: Optional[str],
    label: str,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = supabase_download_bytes(bucket, object_name)
    if len(payload) < settings.model_min_size_bytes:
        raise RuntimeError(
            f"Downloaded {label} is too small ({len(payload)} bytes): "
            f"{bucket}/{object_name}"
        )

    temporary = destination.with_name(
        f".{destination.name}.{uuid4().hex}.download"
    )
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

        _validate_model_file(
            temporary,
            expected_sha256=expected_sha256,
            label=label,
        )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)

    return destination


def _resolve_tree_model_path(
    version: int,
    *,
    allow_download: bool = True,
) -> tuple[Path, str]:
    version = int(version)
    expected_sha256 = settings.tree_model_sha256(version)

    for candidate in _bundled_tree_candidates(version):
        if candidate.exists():
            _validate_model_file(
                candidate,
                expected_sha256=expected_sha256,
                label=f"tree model v{version}",
            )
            return candidate.resolve(), "bundled"

    cache_path = _cache_tree_path(version)
    if cache_path.exists():
        try:
            _validate_model_file(
                cache_path,
                expected_sha256=expected_sha256,
                label=f"cached tree model v{version}",
            )
            return cache_path.resolve(), "cache"
        except Exception:
            cache_path.unlink(missing_ok=True)

    if not allow_download:
        raise FileNotFoundError(
            f"Tree model v{version} is not present in {MODEL_DIR} "
            f"or {MODEL_CACHE_DIR}"
        )

    remote_candidates = [_tree_model_filename(version)]
    if version == 0:
        remote_candidates.extend(["tree_model.pt", "base.pt"])

    errors: list[str] = []
    for object_name in remote_candidates:
        try:
            path = _download_to_cache(
                bucket=SUPABASE_BUCKET_MODELS,
                object_name=object_name,
                destination=cache_path,
                expected_sha256=expected_sha256,
                label=f"tree model v{version}",
            )
            return path.resolve(), "supabase"
        except Exception as exc:
            errors.append(f"{object_name}: {exc}")

    raise RuntimeError(
        f"Tree model v{version} is unavailable. " + " | ".join(errors)
    )


def _resolve_stick_model_path(
    *,
    allow_download: bool = True,
) -> tuple[Path, str]:
    candidates = [settings.stick_model_path]
    explicit = os.getenv("STICK_MODEL_PATH")
    if explicit:
        candidates.insert(0, Path(explicit).expanduser().resolve())

    for candidate in candidates:
        if candidate.exists():
            _validate_model_file(
                candidate,
                expected_sha256=settings.stick_model_sha256,
                label="stick model",
            )
            return candidate.resolve(), "bundled"

    cache_path = MODEL_CACHE_DIR / settings.stick_model_filename
    if cache_path.exists():
        try:
            _validate_model_file(
                cache_path,
                expected_sha256=settings.stick_model_sha256,
                label="cached stick model",
            )
            return cache_path.resolve(), "cache"
        except Exception:
            cache_path.unlink(missing_ok=True)

    if not allow_download:
        raise FileNotFoundError(
            f"Stick model is not present in {MODEL_DIR} or {MODEL_CACHE_DIR}"
        )

    path = _download_to_cache(
        bucket=SUPABASE_BUCKET_MODELS,
        object_name=settings.stick_model_object,
        destination=cache_path,
        expected_sha256=settings.stick_model_sha256,
        label="stick model",
    )
    return path.resolve(), "supabase"


def _discover_local_tree_versions() -> set[int]:
    versions: set[int] = set()
    for directory in (MODEL_DIR, MODEL_CACHE_DIR):
        if not directory.exists():
            continue
        for path in directory.glob("model_v*.pt"):
            match = re.fullmatch(r"model_v(\d+)\.pt", path.name)
            if match and _basic_model_file_is_valid(path):
                versions.add(int(match.group(1)))

    if any(
        _basic_model_file_is_valid(path)
        for path in (MODEL_DIR / "tree_model.pt", MODEL_DIR / "base.pt")
        if path.exists()
    ):
        versions.add(0)
    return versions


def _discover_remote_tree_versions() -> set[int]:
    versions: set[int] = set()
    if not _supabase_is_configured():
        return versions
    for item in supabase_list_objects(SUPABASE_BUCKET_MODELS):
        name = (item.get("name") or "").split("/")[-1]
        match = re.fullmatch(r"model_v(\d+)\.pt", name)
        if match:
            versions.add(int(match.group(1)))
    return versions


def _fallback_active_model_version() -> int:
    configured = int(settings.active_model_version)
    local_versions = _discover_local_tree_versions()

    if configured in local_versions or configured > 0:
        return configured
    if settings.auto_select_latest_local_model and local_versions:
        return max(local_versions)
    if 0 in local_versions:
        return 0
    return configured


def _get_active_model_version() -> int:
    try:
        state = training_state_get()
        raw = state.get("active_model_version")
        if raw is not None:
            version = int(raw)
            if version != 0:
                return version
            if 0 in _discover_local_tree_versions():
                return 0
    except Exception:
        pass
    return _fallback_active_model_version()


def _bootstrap_training_state() -> int:
    fallback_version = _fallback_active_model_version()
    if not _supabase_is_configured():
        return fallback_version

    state = training_state_ensure_row(fallback_version)
    current = int(state.get("active_model_version") or 0)

    if current == 0 and 0 not in _discover_local_tree_versions():
        if fallback_version > 0:
            state = training_state_update(
                {"active_model_version": int(fallback_version)}
            )
            current = int(state.get("active_model_version") or fallback_version)
    return current if current >= 0 else fallback_version


def _load_tree_model_candidate(version: int) -> tuple[YOLO, Path, str, dict]:
    path, source = _resolve_tree_model_path(version, allow_download=True)
    model, info = _load_yolo_checked(
        path,
        expected_task=settings.tree_model_task,
        expected_sha256=settings.tree_model_sha256(version),
        label=f"tree model v{version}",
    )
    return model, path, source, info


def reload_tree_model(
    force: bool = False,
    *,
    requested_version: Optional[int] = None,
) -> dict:
    global TREE_MODEL, TREE_MODEL_VERSION, TREE_MODEL_PATH
    global TREE_MODEL_SOURCE, TREE_MODEL_LOADED_AT
    global _MODEL_LAST_CHECK_TS, _MODEL_LAST_ERROR

    now = time.time()
    if (
        requested_version is None
        and not force
        and (now - _MODEL_LAST_CHECK_TS) < _MODEL_CHECK_INTERVAL_SEC
    ):
        return _tree_model_runtime_info()
    _MODEL_LAST_CHECK_TS = now

    version = (
        int(requested_version)
        if requested_version is not None
        else _get_active_model_version()
    )
    if (
        not force
        and TREE_MODEL is not None
        and TREE_MODEL_VERSION == version
    ):
        return _tree_model_runtime_info()

    try:
        candidate, path, source, info = _load_tree_model_candidate(version)
    except Exception as exc:
        _MODEL_LAST_ERROR = str(exc)
        if TREE_MODEL is not None:
            print(
                f"[!] Could not reload requested tree model v{version}; "
                f"continuing with loaded v{TREE_MODEL_VERSION}: {exc}"
            )
            return _tree_model_runtime_info()
        raise

    # The currently working model is replaced only after the new model
    # has been fully read and validated by Ultralytics.
    TREE_MODEL = candidate
    TREE_MODEL_VERSION = version
    TREE_MODEL_PATH = str(path)
    TREE_MODEL_SOURCE = source
    TREE_MODEL_LOADED_AT = _utc_iso()
    _MODEL_LAST_ERROR = None

    print(
        f"[*] Tree model loaded: v{version}, source={source}, "
        f"path={path}, size={info['size_bytes']}"
    )
    return _tree_model_runtime_info()


def load_stick_model(force: bool = False) -> dict:
    global stick_model, STICK_MODEL_PATH, STICK_MODEL_SOURCE
    global STICK_MODEL_LOADED_AT, _STICK_MODEL_LAST_ERROR

    if stick_model is not None and not force:
        return _stick_model_runtime_info()

    try:
        path, source = _resolve_stick_model_path(allow_download=True)
        candidate, info = _load_yolo_checked(
            path,
            expected_task=settings.stick_model_task,
            expected_sha256=settings.stick_model_sha256,
            label="stick model",
        )
    except Exception as exc:
        _STICK_MODEL_LAST_ERROR = str(exc)
        raise

    stick_model = candidate
    STICK_MODEL_PATH = str(path)
    STICK_MODEL_SOURCE = source
    STICK_MODEL_LOADED_AT = _utc_iso()
    _STICK_MODEL_LAST_ERROR = None

    print(
        f"[*] Stick model loaded: source={source}, path={path}, "
        f"size={info['size_bytes']}"
    )
    return _stick_model_runtime_info()


def get_tree_model() -> YOLO:
    with MODEL_LOCK:
        reload_tree_model(force=False)
        if TREE_MODEL is None:
            reload_tree_model(force=True)
        if TREE_MODEL is None:
            raise RuntimeError("Tree model is not loaded")
        return TREE_MODEL


def get_stick_model() -> YOLO:
    with MODEL_LOCK:
        if stick_model is None:
            load_stick_model(force=True)
        if stick_model is None:
            raise RuntimeError("Stick model is not loaded")
        return stick_model


def activate_tree_model(version: int) -> dict:
    version = int(version)
    with MODEL_LOCK:
        candidate, path, source, info = _load_tree_model_candidate(version)

        # Persist the version only after the file has been validated and
        # Ultralytics has successfully created a model object from it.
        if _supabase_is_configured():
            training_state_ensure_row(version)
            training_state_update({"active_model_version": version})

        global TREE_MODEL, TREE_MODEL_VERSION, TREE_MODEL_PATH
        global TREE_MODEL_SOURCE, TREE_MODEL_LOADED_AT, _MODEL_LAST_ERROR
        TREE_MODEL = candidate
        TREE_MODEL_VERSION = version
        TREE_MODEL_PATH = str(path)
        TREE_MODEL_SOURCE = source
        TREE_MODEL_LOADED_AT = _utc_iso()
        _MODEL_LAST_ERROR = None

        print(
            f"[*] Tree model activated: v{version}, source={source}, "
            f"path={path}, size={info['size_bytes']}"
        )
        return _tree_model_runtime_info()


def list_available_model_versions() -> list[dict]:
    sources: dict[int, set[str]] = {}

    for version in _discover_local_tree_versions():
        sources.setdefault(version, set()).add("local")

    try:
        for version in _discover_remote_tree_versions():
            sources.setdefault(version, set()).add("supabase")
    except Exception:
        pass

    active = _get_active_model_version()
    if active not in sources:
        sources.setdefault(active, set()).add("configured")

    result = []
    for version in sorted(sources):
        local_path = None
        local_size = None
        try:
            path, source = _resolve_tree_model_path(
                version,
                allow_download=False,
            )
            local_path = str(path)
            local_size = path.stat().st_size
            sources[version].add(source)
        except Exception:
            pass

        result.append(
            {
                "version": version,
                "is_active": version == active,
                "is_loaded": version == TREE_MODEL_VERSION,
                "sources": sorted(sources[version]),
                "local_path": local_path,
                "size_bytes": local_size,
            }
        )
    return result


def _tree_model_runtime_info() -> dict:
    size = None
    if TREE_MODEL_PATH:
        try:
            size = Path(TREE_MODEL_PATH).stat().st_size
        except Exception:
            pass
    return {
        "loaded": TREE_MODEL is not None,
        "version": TREE_MODEL_VERSION,
        "path": TREE_MODEL_PATH,
        "source": TREE_MODEL_SOURCE,
        "size_bytes": size,
        "loaded_at": TREE_MODEL_LOADED_AT,
        "last_error": _MODEL_LAST_ERROR,
    }


def _stick_model_runtime_info() -> dict:
    size = None
    if STICK_MODEL_PATH:
        try:
            size = Path(STICK_MODEL_PATH).stat().st_size
        except Exception:
            pass
    return {
        "loaded": stick_model is not None,
        "path": STICK_MODEL_PATH,
        "source": STICK_MODEL_SOURCE,
        "size_bytes": size,
        "loaded_at": STICK_MODEL_LOADED_AT,
        "last_error": _STICK_MODEL_LAST_ERROR,
    }


def get_rembg_session():
    global REMBG_SESSION
    if REMBG_SESSION is not None:
        return REMBG_SESSION
    with REMBG_LOCK:
        if REMBG_SESSION is None:
            print("[*] Loading rembg U-2-Net model...")
            REMBG_SESSION = new_session("u2net")
            remove(
                np.zeros((10, 10, 3), dtype=np.uint8),
                session=REMBG_SESSION,
                only_mask=True,
            )
            print("[*] rembg loaded and warmed up.")
    return REMBG_SESSION


def _strip_data_url(b64: str) -> str:
    if not b64: return b64
    b64 = b64.strip()
    if b64.startswith("data:") and "base64," in b64: b64 = b64.split("base64,", 1)[1]
    return "".join(b64.split()) 

def decode_base64_bytes(b64: str) -> bytes:
    if b64 is None: return b""
    b64_clean = _strip_data_url(str(b64)).strip().replace("-", "+").replace("_", "/")
    pad = len(b64_clean) % 4
    if pad: b64_clean += "=" * (4 - pad)
    raw = base64.b64decode(b64_clean, validate=False)
    try:
        as_text = raw.decode("utf-8").strip()
        if len(as_text) > 16 and all(c.isalnum() or c in "+/=_-\n\r" for c in as_text):
            as_text = _strip_data_url(as_text).strip().replace("-", "+").replace("_", "/")
            pad2 = len(as_text) % 4
            if pad2: as_text += "=" * (4 - pad2)
            raw2 = base64.b64decode(as_text, validate=False)
            if raw2.startswith(b"\x89PNG\r\n\x1a\n") or raw2[:3] == b"\xff\xd8\xff": return raw2
    except Exception: pass
    return raw

def ensure_png_mask_bytes(mask_b64: str) -> bytes:
    raw = decode_base64_bytes(mask_b64)
    try:
        if raw[:1] in (b"{", b"["):
            obj = json.loads(raw.decode("utf-8"))
            if isinstance(obj, dict) and obj.get("mask_png_base64"):
                raw = decode_base64_bytes(str(obj["mask_png_base64"]))
    except Exception: pass
    np_buf = np.frombuffer(raw, np.uint8)
    mask = cv2.imdecode(np_buf, cv2.IMREAD_GRAYSCALE)
    if mask is None: raise ValueError("user_mask_base64 is not a valid PNG/JPEG")
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    ok, out = cv2.imencode(".png", mask_bin)
    return out.tobytes()

def encode_jpeg_base64(img_bgr, max_side=1280, quality=74):
    h, w = img_bgr.shape[:2]
    longest = max(h, w)
    if longest > max_side:
        scale = max_side / float(longest)
        img_bgr = cv2.resize(
            img_bgr,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    ok, out = cv2.imencode(
        ".jpg",
        img_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not ok:
        raise ValueError("Failed to encode JPEG")
    return base64.b64encode(out.tobytes()).decode("ascii")


def draw_mask(img_bgr, mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        approx = cv2.approxPolyDP(
            cnt,
            0.003 * cv2.arcLength(cnt, True),
            True,
        )
        cv2.drawContours(img_bgr, [approx], -1, (0, 255, 0), 3)
    return encode_jpeg_base64(img_bgr, max_side=1280, quality=74)


def prepare_feedback_assets(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    max_side: int = 1600,
    jpeg_quality: int = 82,
) -> dict:
    """Create image/mask files with identical dimensions for user correction.

    The original uploaded photo may be several thousand pixels wide. Flutter
    receives a reduced copy so the editor remains responsive. The automatic
    mask is resized with nearest-neighbour interpolation to exactly the same
    dimensions. The verified training image is later taken from this reduced
    copy, so a user mask can never have a different shape from its image.
    """
    if img_bgr is None or mask is None:
        raise ValueError("Image and mask are required")

    height, width = img_bgr.shape[:2]
    longest = max(height, width)
    scale = 1.0
    if longest > max_side:
        scale = max_side / float(longest)

    target_w = max(1, int(round(width * scale)))
    target_h = max(1, int(round(height * scale)))

    if (target_w, target_h) != (width, height):
        feedback_img = cv2.resize(
            img_bgr,
            (target_w, target_h),
            interpolation=cv2.INTER_AREA,
        )
        feedback_mask = cv2.resize(
            mask,
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        feedback_img = img_bgr.copy()
        feedback_mask = mask.copy()

    _, feedback_mask = cv2.threshold(
        feedback_mask.astype(np.uint8),
        127,
        255,
        cv2.THRESH_BINARY,
    )

    annotated = feedback_img.copy()
    contours, _ = cv2.findContours(
        feedback_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(annotated, contours, -1, (0, 255, 0), 3)

    ok_img, img_buf = cv2.imencode(
        ".jpg",
        feedback_img,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    ok_ann, ann_buf = cv2.imencode(
        ".jpg",
        annotated,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    ok_mask, mask_buf = cv2.imencode(".png", feedback_mask)

    # Visual overlay for Flutter: pixels outside the mask are transparent.
    mask_overlay = np.zeros((target_h, target_w, 4), dtype=np.uint8)
    mask_overlay[:, :, 1] = 255
    mask_overlay[:, :, 3] = feedback_mask
    ok_overlay, overlay_buf = cv2.imencode(".png", mask_overlay)

    if not (ok_img and ok_ann and ok_mask and ok_overlay):
        raise ValueError("Failed to encode feedback assets")

    image_bytes = img_buf.tobytes()
    annotated_bytes = ann_buf.tobytes()
    mask_bytes = mask_buf.tobytes()
    overlay_bytes = overlay_buf.tobytes()
    return {
        "image_bytes": image_bytes,
        "annotated_bytes": annotated_bytes,
        "mask_bytes": mask_bytes,
        "mask_overlay_bytes": overlay_bytes,
        "image_base64": base64.b64encode(image_bytes).decode("ascii"),
        "annotated_base64": base64.b64encode(annotated_bytes).decode("ascii"),
        "mask_base64": base64.b64encode(overlay_bytes).decode("ascii"),
        "width": target_w,
        "height": target_h,
        "source_width": width,
        "source_height": height,
    }


def normalize_mask_to_image(mask_png_bytes: bytes, image_bytes: bytes) -> bytes:
    """Binarize and resize a submitted mask to the exact training image size."""
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    mask = cv2.imdecode(np.frombuffer(mask_png_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Training image cannot be decoded")
    if mask is None:
        raise ValueError("Submitted mask cannot be decoded")

    target_h, target_w = image.shape[:2]
    if mask.shape[:2] != (target_h, target_w):
        mask = cv2.resize(
            mask,
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST,
        )
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    ok, out = cv2.imencode(".png", mask)
    if not ok:
        raise ValueError("Failed to encode normalized mask")
    return out.tobytes()

def _deg(v):
    d = v[0][0] / v[0][1]; m = v[1][0] / v[1][1]; s = v[2][0] / v[2][1]
    return d + m / 60 + s / 3600

def extract_gps(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        exif = img._getexif()
        if not exif: return None
        gps_info = next((v for k, v in exif.items() if ExifTags.TAGS.get(k) == "GPSInfo"), None)
        if not gps_info: return None
        lat = _deg(gps_info[2]); lon = _deg(gps_info[4])
        if gps_info[1] == "S": lat = -lat
        if gps_info[3] == "W": lon = -lon
        return {"lat": lat, "lon": lon}
    except Exception: return None

def reverse_geocode(lat, lon):
    try:
        r = requests.get(NOMINATIM_URL, params={"lat": lat, "lon": lon, "format": "jsonv2", "accept-language": "ru"}, headers={"User-Agent": NOMINATIM_USER_AGENT}, timeout=5)
        return r.json().get("display_name")
    except Exception: return None

SPECIES_STRENGTH_MPA = {
    "Береза": 45.0, "Дуб": 60.0, "Ель": 40.0, "Сосна": 42.0, "Тополь": 30.0,
    "Клен": 50.0, "Ясень": 55.0, "Липа": 35.0,
}

BETA_EMPIRICAL_STATS = {
    "Сосна": {"mean": 47.7, "min": 25.5, "max": 90.0, "ref_height": 20.0},
    "Ель": {"mean": 60.0, "min": 30.0, "max": 100.0, "ref_height": 20.0},
    "Береза": {"mean": 52.0, "min": 25.0, "max": 90.0, "ref_height": 20.0},
    "Дуб": {"mean": 65.0, "min": 35.0, "max": 110.0, "ref_height": 20.0},
    "Тополь": {"mean": 58.0, "min": 30.0, "max": 100.0, "ref_height": 25.0},
    "Клен": {"mean": 55.0, "min": 30.0, "max": 100.0, "ref_height": 20.0},
    "Ясень": {"mean": 55.0, "min": 30.0, "max": 100.0, "ref_height": 20.0},
    "Липа": {"mean": 50.0, "min": 25.0, "max": 90.0, "ref_height": 20.0},
}

def _clamp(x: float, lo: float, hi: float) -> float: return max(lo, min(hi, x))

def estimate_beta_kg_s(species: str, height_m, manual_beta_kg_s=None, crown_density_factor=1.0) -> dict:
    stats = BETA_EMPIRICAL_STATS.get(species, BETA_EMPIRICAL_STATS["Сосна"])
    if manual_beta_kg_s is not None and manual_beta_kg_s > 0:
        value = round(_clamp(float(manual_beta_kg_s), 5.0, 200.0), 2)
        return {"beta_kg_s": value, "beta_max_scenario": value, "method": "manual", "source": "Введено вручную", "input": {"manual_beta_kg_s": float(manual_beta_kg_s)}}
    h, density = float(height_m or 0), float(crown_density_factor or 1.0)
    if h <= 0:
        return {"beta_kg_s": stats["mean"], "beta_max_scenario": stats["max"], "method": "species_default", "source": "Статистическое среднее", "input": {"species": species}}
    height_ratio = h / stats["ref_height"]
    beta_expected = _clamp(stats["mean"] * (height_ratio ** 1.5) * density, 5.0, stats["max"] * 1.5)
    beta_max_scenario = _clamp(beta_expected * 1.88, beta_expected, stats["max"] * 1.5)
    return {"beta_kg_s": round(beta_expected, 2), "beta_max_scenario": round(beta_max_scenario, 2), "method": "empirical_borisevich_2021", "source": "Полевая статистика (Borisevich 2021)", "input": {"species": species, "height_m": h, "crown_density_factor": density}}

def slenderness_score(height_m, diameter_m):
    if not diameter_m or diameter_m <= 0: return 0.0, 0.0
    S = height_m / diameter_m
    if S >= 100: score = 1.0
    elif S >= 80: score = 0.8
    elif S >= 60: score = 0.5
    elif S >= 40: score = 0.2
    else: score = 0.0
    return S, score

def bending_stress_score(species, height_m, trunk_diameter_m, beta_kg_s, wind_speed):
    if not all([height_m, trunk_diameter_m, beta_kg_s]) or trunk_diameter_m <= 0: return 99.0, 0.0, [], 0.0, 0.0, 0.0
    expl = []
    design_wind_speed = float(wind_speed) if wind_speed and wind_speed > 0 else 25.0
    force_n = beta_kg_s * design_wind_speed
    lever_arm_m = height_m * 0.5 
    bending_moment_nm = force_n * lever_arm_m
    w_m3 = (3.14159 * (trunk_diameter_m ** 3)) / 32.0
    stress_mpa = (bending_moment_nm / w_m3) / 1_000_000.0
    limit_mpa = SPECIES_STRENGTH_MPA.get(species, 40.0) 
    safety_factor = limit_mpa / stress_mpa if stress_mpa > 0 else 99.0

    expl.append(f"Расчетный ветер: {design_wind_speed:.1f} м/с. Момент у основания: {bending_moment_nm/1000:.1f} кН·м")
    expl.append(f"Напряжение в стволе: {stress_mpa:.1f} МПа (Предел для '{species}': {limit_mpa} МПа)")
    if safety_factor < 1.0: expl.append(f"КРИТИЧЕСКИЙ РИСК ИЗЛОМА! Запас прочности: {safety_factor:.2f}"); score = 1.0
    elif safety_factor < 1.5: expl.append(f"Низкий запас прочности ({safety_factor:.2f}). Риск поломки ствола."); score = 0.7
    else: expl.append(f"Ствол выдержит ветер. Запас прочности: {safety_factor:.2f}"); score = 0.1
    return safety_factor, score, expl, force_n, lever_arm_m, bending_moment_nm

def compute_risk(species, height, diameter, lean_angle_deg, beta_info, wind_speed):
    expl = []
    lean_score = 0.0
    if lean_angle_deg > 15.0:
        lean_score = 1.0; expl.append(f"ОПАСНОСТЬ: Аномальный наклон ствола ({lean_angle_deg}°). Риск выкорчевывания.")
    elif lean_angle_deg > 7.0:
        lean_score = 0.5; expl.append(f"Внимание: Наклон ствола {lean_angle_deg}°.")
    else: expl.append(f"Наклон ствола в пределах нормы ({lean_angle_deg}°).")

    S, s_score = slenderness_score(height, diameter)
    if diameter and diameter > 0:
        state = "Опасно (Тонкое)" if S > 80 else "Норма"
        expl.append(f"Стройность H/D: {S:.1f} ({state})")

    beta = beta_info.get("beta_max_scenario") or beta_info.get("beta_kg_s") or 0
    safety_factor, wind_score_val, wind_expl, f_n, l_m, m_nm = bending_stress_score(species, height, diameter, beta, wind_speed)
    expl.extend(wind_expl)

    index = max(lean_score, s_score, wind_score_val) + 0.1 * (lean_score + s_score + wind_score_val)
    index = max(0.0, min(index, 1.0))
    cat = "низкий" if index < 0.4 else "средний" if index < 0.7 else "высокий"
    return {"index": index, "category": cat, "explanation": expl}, f_n, l_m, m_nm, safety_factor


class FeedbackRequest(BaseModel):
    analysis_id: str; use_for_training: bool; tree_ok: bool; stick_ok: bool; params_ok: bool; species_ok: bool
    correct_species: str | None = None; correct_height_m: float | None = None; correct_crown_width_m: float | None = None
    correct_trunk_diameter_m: float | None = None; correct_scale_px_to_m: float | None = None; user_mask_base64: str | None = None

class AdminSetTrainingRequest(BaseModel):
    use_for_training: bool | None = None; enabled: bool | None = None; include: bool | None = None; value: bool | None = None

class AuthRegisterRequest(BaseModel):
    name: str
    email: str
    password: str


class AuthLoginRequest(BaseModel):
    email: str
    password: str


class AuthGoogleRequest(BaseModel):
    id_token: str
    email: str | None = None
    name: str | None = None
    photo_url: str | None = None


app = FastAPI(title="ArborScan API v2.9 (Protected Admin API)")

AUTH_TOKEN_TTL_DAYS = int(os.getenv("ARBORSCAN_AUTH_TOKEN_TTL_DAYS", "30"))
GOOGLE_CLIENT_ID = os.getenv(
    "GOOGLE_CLIENT_ID",
    "946297507051-33c4msb91harv7rqppf2f31qn10n1m2m.apps.googleusercontent.com",
)


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _hash_password(
    password: str,
    salt_hex: str | None = None,
) -> tuple[str, str]:
    if salt_hex is None:
        salt_hex = secrets.token_bytes(16).hex()
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt_hex),
        120_000,
    ).hex()
    return digest, salt_hex


def _user_public(row: dict) -> dict:
    data = {
        "id": row.get("id"),
        "name": row.get("name"),
        "email": row.get("email"),
        "role": row.get("role") or "user",
        "created_at": row.get("created_at"),
    }
    if "provider" in row:
        data["provider"] = row.get("provider")
    if "avatar_url" in row:
        data["avatar_url"] = row.get("avatar_url")
    return data


def _db_json(
    method: str,
    path: str,
    *,
    params: dict | None = None,
    json_body: dict | list | None = None,
    prefer: str | None = None,
    timeout: int = 15,
):
    if not SUPABASE_DB_BASE:
        raise RuntimeError("Supabase database is not configured")

    headers = _sb_headers()
    if prefer:
        headers["Prefer"] = prefer

    response = requests.request(
        method=method,
        url=f"{SUPABASE_DB_BASE}/{path.lstrip('/')}",
        headers=headers,
        params=params,
        json=json_body,
        timeout=timeout,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"Supabase DB error {response.status_code} for {path}: "
            f"{response.text}"
        )
    if response.status_code == 204 or not response.content:
        return None
    return response.json()


def _create_session(user_id: str) -> dict:
    token = secrets.token_urlsafe(48)
    created_at = _now_iso()
    expires_at = (
        datetime.utcnow() + timedelta(days=AUTH_TOKEN_TTL_DAYS)
    ).isoformat(timespec="seconds") + "Z"

    _db_json(
        "POST",
        "auth_sessions",
        json_body={
            "token": token,
            "user_id": user_id,
            "created_at": created_at,
            "expires_at": expires_at,
        },
        prefer="return=minimal",
    )
    return {"token": token, "expires_at": expires_at}


def _get_user_by_token(token: str) -> dict | None:
    token = (token or "").strip()
    if not token or not SUPABASE_DB_BASE:
        return None

    try:
        rows = _db_json(
            "GET",
            "auth_sessions",
            params={
                "token": f"eq.{token}",
                "expires_at": f"gt.{_now_iso()}",
                "select": "token,user_id,expires_at,users(*)",
                "limit": "1",
            },
            timeout=10,
        )
    except Exception as exc:
        print(f"[!] Session lookup failed: {exc}")
        return None

    if not rows:
        return None

    user = rows[0].get("users")
    if isinstance(user, list):
        user = user[0] if user else None
    return user if isinstance(user, dict) else None


def _extract_bearer_token(authorization: str | None) -> str | None:
    value = (authorization or "").strip()
    if not value:
        return None
    parts = value.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    token = parts[1].strip()
    return token or None


def _resolve_auth_token(
    authorization: str | None,
    query_token: str | None = None,
) -> str | None:
    return _extract_bearer_token(authorization) or (query_token or "").strip() or None


async def require_authenticated_user(
    authorization: str | None = Header(default=None),
) -> dict:
    token = _extract_bearer_token(authorization)
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Требуется авторизация.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = _get_user_by_token(token)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Сессия отсутствует или истекла.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


async def require_admin(
    user: dict = Depends(require_authenticated_user),
) -> dict:
    if (user.get("role") or "user").strip().lower() != "admin":
        raise HTTPException(
            status_code=403,
            detail="Доступ разрешён только администратору.",
        )
    return user


def _email_norm(email: str) -> str:
    return (email or "").strip().lower()


@app.post("/auth/register")
async def auth_register(payload: AuthRegisterRequest):
    if not SUPABASE_DB_BASE:
        raise HTTPException(status_code=500, detail="Database disabled")

    name = payload.name.strip()
    email = _email_norm(payload.email)
    password = payload.password

    if len(name) < 2:
        raise HTTPException(status_code=422, detail="Имя слишком короткое.")
    if "@" not in email or len(email) > 254:
        raise HTTPException(status_code=422, detail="Некорректный email.")
    if len(password) < 4:
        raise HTTPException(status_code=422, detail="Пароль слишком короткий.")

    existing = _db_json(
        "GET",
        "users",
        params={"email": f"eq.{email}", "select": "id", "limit": "1"},
    )
    if existing:
        raise HTTPException(
            status_code=409,
            detail="Пользователь с такой почтой уже существует.",
        )

    password_hash, salt = _hash_password(password)
    user_id = str(uuid4())
    now = _now_iso()
    created = _db_json(
        "POST",
        "users",
        json_body={
            "id": user_id,
            "name": name,
            "email": email,
            "password_hash": password_hash,
            "salt": salt,
            "role": "user",
            "created_at": now,
            "updated_at": now,
        },
        prefer="return=representation",
    )
    user = created[0] if created else None
    if not user:
        raise HTTPException(status_code=500, detail="Не удалось создать профиль.")

    session = _create_session(user_id)
    return {
        "ok": True,
        "user": _user_public(user),
        "token": session["token"],
        "expires_at": session["expires_at"],
    }


@app.post("/auth/login")
async def auth_login(payload: AuthLoginRequest):
    email = _email_norm(payload.email)
    rows = _db_json(
        "GET",
        "users",
        params={"email": f"eq.{email}", "select": "*", "limit": "1"},
    )
    if not rows:
        raise HTTPException(status_code=401, detail="Неверная почта или пароль.")

    user = rows[0]
    try:
        expected, _ = _hash_password(payload.password, user["salt"])
    except Exception:
        raise HTTPException(status_code=401, detail="Неверная почта или пароль.")

    if not secrets.compare_digest(expected, user.get("password_hash") or ""):
        raise HTTPException(status_code=401, detail="Неверная почта или пароль.")

    session = _create_session(user["id"])
    return {
        "ok": True,
        "user": _user_public(user),
        "token": session["token"],
        "expires_at": session["expires_at"],
    }


@app.post("/auth/google")
async def auth_google(payload: AuthGoogleRequest):
    if not SUPABASE_DB_BASE:
        raise HTTPException(status_code=500, detail="Database disabled")
    if not google_id_token or not GOOGLE_CLIENT_ID:
        raise HTTPException(
            status_code=500,
            detail="Google Auth is not configured",
        )

    try:
        info = google_id_token.verify_oauth2_token(
            payload.id_token,
            google_requests.Request(),
            GOOGLE_CLIENT_ID,
        )
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"Google error: {exc}")

    sub = str(info.get("sub") or "")
    email = _email_norm(str(info.get("email") or payload.email or ""))
    if not sub or not email:
        raise HTTPException(status_code=401, detail="Google не вернул данные аккаунта.")

    name = str(
        info.get("name")
        or payload.name
        or email.split("@")[0]
        or "Google user"
    ).strip()
    avatar_url = str(info.get("picture") or payload.photo_url or "")
    now = _now_iso()

    rows = _db_json(
        "GET",
        "users",
        params={
            "or": f"(google_sub.eq.{sub},email.eq.{email})",
            "select": "*",
            "limit": "1",
        },
    )

    if not rows:
        user_id = str(uuid4())
        password_hash, salt = _hash_password(secrets.token_urlsafe(24))
        created = _db_json(
            "POST",
            "users",
            json_body={
                "id": user_id,
                "name": name,
                "email": email,
                "password_hash": password_hash,
                "salt": salt,
                "role": "user",
                "created_at": now,
                "updated_at": now,
                "provider": "google",
                "google_sub": sub,
                "avatar_url": avatar_url,
            },
            prefer="return=representation",
        )
        user = created[0]
    else:
        user = rows[0]
        updated = _db_json(
            "PATCH",
            "users",
            params={"id": f"eq.{user['id']}"},
            json_body={
                "name": name or user.get("name"),
                "provider": "google",
                "google_sub": sub or user.get("google_sub"),
                "avatar_url": avatar_url or user.get("avatar_url"),
                "updated_at": now,
            },
            prefer="return=representation",
        )
        user = updated[0] if updated else user

    session = _create_session(user["id"])
    return {
        "ok": True,
        "user": _user_public(user),
        "token": session["token"],
        "expires_at": session["expires_at"],
    }


@app.get("/auth/me")
async def auth_me(
    token: str | None = None,
    authorization: str | None = Header(default=None),
):
    resolved = _resolve_auth_token(authorization, token)
    user = _get_user_by_token(resolved or "")
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Сессия отсутствует или истекла.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {"ok": True, "user": _user_public(user)}


@app.post("/auth/logout")
async def auth_logout(
    authorization: str | None = Header(default=None),
    user: dict = Depends(require_authenticated_user),
):
    del user
    token = _extract_bearer_token(authorization)
    if token:
        _db_json(
            "DELETE",
            "auth_sessions",
            params={"token": f"eq.{token}"},
            prefer="return=minimal",
        )
    return {"ok": True}


@app.post("/auth/set-role")
async def auth_set_role_disabled():
    raise HTTPException(
        status_code=410,
        detail=(
            "Самостоятельная смена роли отключена. "
            "Роль администратора назначается в таблице users доверенным оператором."
        ),
    )


def _save_analysis_record(response: dict, user: dict | None):
    if not SUPABASE_DB_BASE:
        return
    risk = response.get("risk") or {}
    beta = response.get("beta") or {}
    analytic_out = (response.get("analytic_wind_model") or {}).get("outputs") or {}
    gps = response.get("gps") or {}
    payload = {
        "id": response.get("analysis_id"),
        "user_id": user.get("id") if user else None,
        "created_at": _now_iso(),
        "species": response.get("species"),
        "risk_index": risk.get("index"),
        "risk_category": risk.get("category"),
        "height_m": response.get("height_m"),
        "crown_width_m": response.get("crown_width_m"),
        "trunk_diameter_m": response.get("trunk_diameter_m"),
        "beta_kg_s": beta.get("beta_kg_s"),
        "base_moment_nm": analytic_out.get("base_moment_nm"),
        "center_of_load_m": analytic_out.get("center_of_load_m"),
        "lat": gps.get("lat"),
        "lon": gps.get("lon"),
        "address": response.get("address"),
        "response_json": response,
    }
    try:
        _db_json(
            "POST",
            "analyses",
            json_body=payload,
            prefer="resolution=merge-duplicates,return=minimal",
        )
    except Exception as exc:
        print(f"[!] Failed to save analysis row: {exc}")


def _analysis_summary(row: dict) -> dict:
    return {
        "analysis_id": row.get("id"),
        "created_at": row.get("created_at"),
        "species": row.get("species"),
        "risk_index": row.get("risk_index"),
        "risk_category": row.get("risk_category"),
        "height_m": row.get("height_m"),
        "crown_width_m": row.get("crown_width_m"),
        "trunk_diameter_m": row.get("trunk_diameter_m"),
        "lat": row.get("lat"),
        "lon": row.get("lon"),
        "address": row.get("address"),
    }


@app.get("/analyses/my")
async def analyses_my(
    token: str | None = None,
    limit: int = 100,
    authorization: str | None = Header(default=None),
):
    resolved = _resolve_auth_token(authorization, token)
    user = _get_user_by_token(resolved or "")
    if not user:
        raise HTTPException(status_code=401, detail="Сессия отсутствует или истекла.")

    rows = _db_json(
        "GET",
        "analyses",
        params={
            "user_id": f"eq.{user['id']}",
            "order": "created_at.desc",
            "limit": str(max(1, min(int(limit), 500))),
            "select": "*",
        },
    )
    return {"ok": True, "items": [_analysis_summary(row) for row in rows or []]}


@app.get("/analyses/{analysis_id}")
async def analyses_get(
    analysis_id: str,
    token: str | None = None,
    authorization: str | None = Header(default=None),
):
    resolved = _resolve_auth_token(authorization, token)
    user = _get_user_by_token(resolved or "")
    if not user:
        raise HTTPException(status_code=401, detail="Сессия отсутствует или истекла.")

    rows = _db_json(
        "GET",
        "analyses",
        params={"id": f"eq.{analysis_id}", "select": "*", "limit": "1"},
    )
    if not rows:
        raise HTTPException(status_code=404, detail="Анализ не найден.")

    row = rows[0]
    if row.get("user_id") != user["id"] and user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Нет доступа к этому анализу.")
    return {"ok": True, "analysis": row.get("response_json")}


@app.get("/admin/me")
async def admin_me(admin: dict = Depends(require_admin)):
    return {"ok": True, "user": _user_public(admin)}


@app.get("/admin/analyses")
async def admin_analyses(
    limit: int = 200,
    admin: dict = Depends(require_admin),
):
    del admin
    rows = _db_json(
        "GET",
        "analyses",
        params={
            "order": "created_at.desc",
            "limit": str(max(1, min(int(limit), 1000))),
            "select": "*",
        },
    )
    return {"ok": True, "items": [_analysis_summary(row) for row in rows or []]}


@app.get("/profile/stats")
async def profile_stats(
    token: str | None = None,
    authorization: str | None = Header(default=None),
):
    resolved = _resolve_auth_token(authorization, token)
    user = _get_user_by_token(resolved or "")
    if not user:
        raise HTTPException(status_code=401, detail="Сессия отсутствует или истекла.")

    rows = _db_json(
        "GET",
        "analyses",
        params={
            "user_id": f"eq.{user['id']}",
            "order": "created_at.desc",
            "select": "*",
        },
    ) or []
    risks = [
        row.get("risk_index")
        for row in rows
        if isinstance(row.get("risk_index"), (int, float))
    ]
    return {
        "ok": True,
        "user": _user_public(user),
        "stats": {
            "total_analyses": len(rows),
            "with_geo": sum(1 for row in rows if row.get("lat") is not None),
            "high_risk_count": sum(
                1 for row in rows if row.get("risk_category") == "высокий"
            ),
            "avg_risk": sum(risks) / len(risks) if risks else None,
            "last_analysis": _analysis_summary(rows[0]) if rows else None,
        },
    }


TRAINING_EVENTS = deque(maxlen=int(os.getenv("TRAINING_EVENTS_MAXLEN", "200")))


def log_training_event(
    level: str,
    message: str,
    data: dict | None = None,
) -> None:
    event = {
        "ts": _now_iso(),
        "level": (level or "INFO").upper(),
        "message": message,
        "data": data or {},
    }
    TRAINING_EVENTS.append(event)
    print(
        f"[TRAINING_EVENT] {event['ts']} {event['level']}: "
        f"{event['message']} {event['data']}"
    )



@app.on_event("startup")
def _startup_load_models():
    print(f"[*] ArborScan API {API_VERSION} startup")
    print(f"[*] Project root: {PROJECT_ROOT}")
    print(f"[*] Model directory: {MODEL_DIR}")
    print(f"[*] Model cache directory: {MODEL_CACHE_DIR}")

    try:
        selected_version = _bootstrap_training_state()
        print(f"[*] Selected active tree model version: v{selected_version}")
    except Exception as exc:
        selected_version = _fallback_active_model_version()
        print(
            "[!] Could not initialize training_state; "
            f"using local/env version v{selected_version}: {exc}"
        )

    try:
        with MODEL_LOCK:
            load_stick_model(force=True)
            reload_tree_model(
                force=True,
                requested_version=selected_version,
            )
    except Exception as exc:
        # The process remains alive so /health can report the exact cause.
        # /analyze-tree will return an error until the model issue is fixed.
        print(f"[!] Model startup failed: {exc}")

    if settings.preload_rembg:
        try:
            get_rembg_session()
        except Exception as exc:
            print(f"[!] rembg preload failed: {exc}")

@app.get("/health")
def health(deep: bool = False):
    supabase_info = {
        "configured": _supabase_is_configured(),
        "reachable": None,
        "error": None,
    }
    if deep and _supabase_is_configured():
        started = time.perf_counter()
        try:
            state = training_state_get()
            supabase_info.update(
                {
                    "reachable": True,
                    "latency_ms": round(
                        (time.perf_counter() - started) * 1000,
                        1,
                    ),
                    "active_model_version": state.get(
                        "active_model_version"
                    ),
                }
            )
        except Exception as exc:
            supabase_info.update(
                {
                    "reachable": False,
                    "error": str(exc),
                }
            )

    tree_info = _tree_model_runtime_info()
    stick_info = _stick_model_runtime_info()
    models_ready = bool(tree_info["loaded"] and stick_info["loaded"])

    return {
        "status": "ok" if models_ready else "degraded",
        "api_version": API_VERSION,
        "schema_version": SCHEMA_VERSION,
        "build": BUILD_INFO,
        "models_ready": models_ready,
        "tree_model": tree_info,
        "stick_model": stick_info,
        "available_tree_models": list_available_model_versions(),
        "paths": {
            "project_root": str(PROJECT_ROOT),
            "model_dir": str(MODEL_DIR),
            "model_cache_dir": str(MODEL_CACHE_DIR),
        },
        "supabase": supabase_info,
        "plantnet": {"configured": bool(PLANTNET_API_KEY)},
        "rembg": {
            "loaded": REMBG_SESSION is not None,
            "preload_enabled": settings.preload_rembg,
        },
        "checked_at": _utc_iso(),
    }


@app.get("/health/models")
def health_models():
    return {
        "tree_model": _tree_model_runtime_info(),
        "stick_model": _stick_model_runtime_info(),
        "available_tree_models": list_available_model_versions(),
        "configured_active_model_version": _get_active_model_version(),
        "loaded_active_model_version": TREE_MODEL_VERSION,
    }


def normalize_address_ru(address: str | None) -> str | None:
    if not address: return address
    replacements = {"Інтэрнат": "Интернат", "вуліца": "улица", "вул.": "ул.", "Машынабудаўнікоў": "Машиностроителей"}
    for src, dst in replacements.items(): address = address.replace(src, dst)
    return address

def _run_yolo_sync(img_array, conf=0.25):
    tree = get_tree_model()
    stick = get_stick_model()
    return (
        tree(img_array, imgsz=1024, retina_masks=True, conf=conf)[0],
        stick(img_array)[0],
    )

def map_plantnet_name(raw_name: str) -> str:
    name_lower = raw_name.lower()
    for n in ["сосна", "ель", "дуб", "береза", "тополь", "клен", "ясень", "липа"]:
        if n in name_lower or n.replace('е','ё') in name_lower: return n.capitalize()
    return raw_name.capitalize()

def _run_classifier_sync(crop_bgr):
    if not PLANTNET_API_KEY:
        return "Неизвестно"
    ok, encoded_img = cv2.imencode(".jpg", crop_bgr)
    if not ok: return "Неизвестно"
    try:
        r = requests.post(f"https://my-api.plantnet.org/v2/identify/all?api-key={PLANTNET_API_KEY}&lang=ru", files=[('images', ('crop.jpg', encoded_img.tobytes(), 'image/jpeg'))], data={'organs': ['habit']}, timeout=12)
        if r.status_code == 200 and r.json().get('results'):
            return map_plantnet_name(r.json().get('results')[0].get('species', {}).get('commonNames', [r.json().get('results')[0].get('species', {}).get('scientificNameWithoutAuthor', 'Неизвестно')])[0])
    except Exception: pass
    return "Неизвестно"

@app.post("/analyze-tree")
async def analyze_tree(
    file: UploadFile = File(...),
    tap_x: Optional[float] = Form(None), 
    tap_y: Optional[float] = Form(None), 
    ar_height_m: Optional[float] = Form(None),
    ar_crown_width_m: Optional[float] = Form(None),
    ar_trunk_diameter_m: Optional[float] = Form(None),
    manual_scale: Optional[float] = Form(None),
    camera_distance_m: Optional[float] = Form(None), # <--- ДОБАВЛЕНА ДИСТАНЦИЯ (ОПТИКА)
    manual_beta_kg_s: Optional[float] = Form(None),
    crown_density_factor: Optional[float] = Form(None),
    manual_wind_speed_m_s: Optional[float] = Form(None),
    ai_conf: Optional[float] = Form(0.25),
    ai_use_rembg: Optional[str] = Form("false"),
    ai_smoothness: Optional[int] = Form(5),
    auth_token: Optional[str] = Form(None),
    lat: Optional[float] = Form(None),
    lon: Optional[float] = Form(None),
):
    analysis_user = _get_user_by_token(auth_token) if auth_token else None

    image_bytes = await file.read()
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None: raise HTTPException(status_code=400, detail="Не удалось прочитать изображение")
    H, W = img.shape[:2]

    conf_val = max(0.05, min(0.95, float(ai_conf))) if ai_conf else 0.25
    tree_res, stick_res = await run_in_threadpool(_run_yolo_sync, img, conf_val)
    
    masks = []
    distances = []
    target_x = (float(tap_x) * W) if tap_x is not None else (W / 2.0)
    target_y = (float(tap_y) * H) if tap_y is not None else (H / 2.0)
    
    smooth_k = max(1, int(ai_smoothness)) if ai_smoothness else 5
    kernel = np.ones((smooth_k, smooth_k), np.uint8)

    if tree_res.masks is None or len(tree_res.masks.data) == 0:
        fallback_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.rectangle(fallback_mask, (int(W*0.3), int(H*0.1)), (int(W*0.7), int(H*0.9)), 255, -1)
        masks.append(fallback_mask)
        idx = 0
        x1, y1, x2, y2 = int(W*0.3), int(H*0.1), int(W*0.7), int(H*0.9)
    else:
        idx = -1
        for i, m in enumerate(tree_res.masks.data):
            tmp_mask = cv2.resize((m.cpu().numpy() > 0.5).astype(np.uint8) * 255, (W, H), interpolation=cv2.INTER_NEAREST)
            if smooth_k > 1:
                tmp_mask = cv2.morphologyEx(tmp_mask, cv2.MORPH_CLOSE, kernel)
            masks.append(tmp_mask)
            
            if tap_x is not None and tap_y is not None:
                check_y = min(max(int(target_y), 0), H - 1)
                check_x = min(max(int(target_x), 0), W - 1)
                if tmp_mask[check_y, check_x] > 0:
                    idx = i 
                    
            x1_t, y1_t, x2_t, y2_t = tree_res.boxes.xyxy[i].cpu().numpy()
            dist = (((x1_t+x2_t)/2.0) - target_x)**2 + (((y1_t+y2_t)/2.0) - target_y)**2
            distances.append(dist)

        if idx == -1 and distances:
            idx = int(np.argmin(distances))
            
        yolo_mask = masks[idx]
        try:
            x1, y1, x2, y2 = map(int, tree_res.boxes.xyxy[idx].cpu().numpy())
        except Exception:
            ys_tmp, xs_tmp = np.where(yolo_mask > 0)
            x1, y1, x2, y2 = xs_tmp.min(), ys_tmp.min(), xs_tmp.max(), ys_tmp.max()

    margin = 30
    x1_c, y1_c = max(0, x1 - margin), max(0, y1 - margin)
    x2_c, y2_c = min(W, x2 + margin), min(H, y2 + margin)

    use_rembg = str(ai_use_rembg).lower() in ("true", "1", "yes")
    
    if use_rembg:
        try:
            crop_bgr = img[y1_c:y2_c, x1_c:x2_c]
            def _refine_mask_sync(crop):
                return remove(
                    crop,
                    session=get_rembg_session(),
                    only_mask=True,
                )
            refined_crop_mask = await run_in_threadpool(_refine_mask_sync, crop_bgr)
            
            if len(refined_crop_mask.shape) == 3: refined_crop_mask = refined_crop_mask[:, :, 0]
            _, mask_bin = cv2.threshold(refined_crop_mask, 127, 255, cv2.THRESH_BINARY)
            
            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            solid_crop_mask = np.zeros_like(mask_bin)
            cv2.drawContours(solid_crop_mask, contours, -1, 255, thickness=cv2.FILLED)
            
            final_mask = np.zeros((H, W), dtype=np.uint8)
            final_mask[y1_c:y2_c, x1_c:x2_c] = solid_crop_mask
            if smooth_k > 1:
                final_mask = cv2.morphologyEx(cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)
            mask = final_mask
        except Exception: mask = yolo_mask
    else:
        try: mask = yolo_mask
        except Exception: mask = fallback_mask

    ys, xs = np.where(mask > 0)
    if len(ys) == 0: return JSONResponse({"error": "Ошибка контура"}, status_code=400)
    y_min, y_max = ys.min(), ys.max()
    height_px = y_max - y_min

    crown_width_px = 0
    for y in range(y_min, y_min + int(0.7 * height_px)):
        row = np.where(mask[y] > 0)[0]
        if len(row) > 0: crown_width_px = max(crown_width_px, row.max() - row.min())

    trunk_vals = [np.where(mask[y] > 0)[0].max() - np.where(mask[y] > 0)[0].min() for y in range(y_max - int(0.2 * height_px), y_max) if len(np.where(mask[y] > 0)[0]) > 0]
    trunk_px = np.mean(trunk_vals) if trunk_vals else 0

    species_name = await run_in_threadpool(_run_classifier_sync, img[y1_c:y2_c, x1_c:x2_c])

    # 6. МАСШТАБ И РАЗМЕРЫ (СВЯТОЙ ГРААЛЬ)
    scale = None
    dimensions_source = "Неизвестно"

    if manual_scale and float(manual_scale) > 0:
        scale = float(manual_scale); dimensions_source = "Пользовательский маркер"
    
    # --- НОВЫЙ БЛОК: ОПТИЧЕСКОЕ ВЫЧИСЛЕНИЕ МАСШТАБА ---
    if not scale and camera_distance_m and float(camera_distance_m) > 0:
        # Угол обзора современных смартфонов ~ 60-65 градусов по вертикали
        # Для гарантии берем 60 градусов (1.047 рад)
        fov_v_rad = 1.047
        dist = float(camera_distance_m)
        # Реальная высота всего кадра в метрах на этом расстоянии
        visible_world_height_m = 2.0 * dist * math.tan(fov_v_rad / 2.0)
        # Масштаб: сколько метров в 1 пикселе
        scale = visible_world_height_m / H
        dimensions_source = f"Оптика (Дистанция {dist}м)"
    # ----------------------------------------------------

    if not scale and len(stick_res.boxes) > 0:
        best = max(stick_res.boxes, key=lambda b: b.xyxy[0][3] - b.xyxy[0][1])
        stick_h = best.xyxy[0][3].cpu().item() - best.xyxy[0][1].cpu().item()
        if stick_h > 10: scale = REAL_STICK_M / stick_h; dimensions_source = "Авто-маркер (AI)"
    
    if not scale:
        if ar_height_m and height_px > 0: scale = float(ar_height_m) / height_px; dimensions_source = "Пропорционально (по AR Высоте)"
        elif ar_crown_width_m and crown_width_px > 0: scale = float(ar_crown_width_m) / crown_width_px; dimensions_source = "Пропорционально (по AR Кроне)"
        elif ar_trunk_diameter_m and trunk_px > 0: scale = float(ar_trunk_diameter_m) / trunk_px; dimensions_source = "Пропорционально (по AR Стволу)"

    if not scale:
        ref_h = BETA_EMPIRICAL_STATS.get(species_name, BETA_EMPIRICAL_STATS["Сосна"])["ref_height"]
        if height_px > 0: scale = ref_h / height_px; dimensions_source = f"Био. статистика ({species_name})"

    height_m = round(height_px * scale, 2) if scale else None
    crown_m = round(crown_width_px * scale, 2) if scale else None
    trunk_m = round(trunk_px * scale, 2) if scale and trunk_px else None

    height_m_ai = height_m
    crown_m_ai = crown_m
    trunk_m_ai = trunk_m

    if ar_height_m: height_m = round(float(ar_height_m), 2)
    if ar_crown_width_m: crown_m = round(float(ar_crown_width_m), 2)
    if ar_trunk_diameter_m: trunk_m = round(float(ar_trunk_diameter_m), 2)

    if ar_trunk_diameter_m and not ar_height_m and trunk_px > 0:
        if (height_px / trunk_px) < 35:  
            typical_s = {"Сосна": 65, "Ель": 70, "Береза": 60, "Дуб": 45, "Тополь": 55, "Клен": 50, "Ясень": 55, "Липа": 50}.get(species_name, 55)
            height_m = round(float(ar_trunk_diameter_m) * typical_s, 2)
            if not ar_crown_width_m: crown_m = round(float(ar_trunk_diameter_m) * (12 if species_name in ["Ель", "Сосна"] else 18), 2)
            dimensions_source = "Аллометрия (Коррекция перспективы)"

    if ar_height_m or ar_crown_width_m or ar_trunk_diameter_m:
        if not manual_scale and "Аллометрия" not in dimensions_source: dimensions_source = "Введено пользователем"

    ar_measurements = {"height_m": height_m if ar_height_m else None, "crown_width_m": crown_m if ar_crown_width_m else None, "trunk_diameter_m": trunk_m if ar_trunk_diameter_m else None}
    measurement_sources = {"height_m": "ar" if ar_height_m else "image", "crown_width_m": "ar" if ar_crown_width_m else "image", "trunk_diameter_m": "ar" if ar_trunk_diameter_m else "image"}

    crown_density_ai = 1.0
    try:
        crown_mask = mask[y_min : y_min + int(0.7 * height_px), :]
        contours, _ = cv2.findContours(crown_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            hull = cv2.convexHull(np.vstack(contours))
            if cv2.contourArea(hull) > 0: crown_density_ai = max(0.1, min(round(float(np.sum(crown_mask > 0) / cv2.contourArea(hull)), 3), 1.0))
    except Exception: pass

    lean_angle_deg = 0.0
    try:
        trunk_mask = mask[y_max - int(0.3 * height_px) : y_max, :]
        ys_trunk, xs_trunk = np.where(trunk_mask > 0)
        if len(xs_trunk) > 50:
            vx, vy, _, _ = cv2.fitLine(np.column_stack((xs_trunk, ys_trunk)), cv2.DIST_L2, 0, 0.01, 0.01)
            dev = abs(90.0 - abs(np.degrees(np.arctan2(vy, vx))[0]))
            lean_angle_deg = round(float(180 - dev if dev > 90 else dev), 1)
    except Exception: pass

    feedback_assets = prepare_feedback_assets(img, mask)
    annotated_b64 = feedback_assets["annotated_base64"]

    gps, address = None, None
    if ENABLE_ENV_ANALYSIS:
        gps = {"lat": float(lat), "lon": float(lon)} if (lat and lon) else extract_gps(image_bytes)
        if gps: address = normalize_address_ru(reverse_geocode(gps["lat"], gps["lon"]))

    final_crown_density = crown_density_factor if crown_density_factor else crown_density_ai
    beta_info = estimate_beta_kg_s(species_name, height_m, manual_beta_kg_s=manual_beta_kg_s, crown_density_factor=final_crown_density)
    
    wind_design = float(manual_wind_speed_m_s or 25.0)
    risk_data, f_n, l_m, m_nm, s_f = compute_risk(species_name, height_m, trunk_m, lean_angle_deg, beta_info, wind_design)

    analytic_wind_model = {
        "available": True,
        "outputs": {"total_force_n": round(f_n, 1), "center_of_load_m": round(l_m, 1), "base_moment_nm": round(m_nm, 1), "analytical_score": round(s_f, 2)},
        "inputs": {"crown_start_height_m": round((height_m or 0) * 0.5, 1), "n_elements": 1}
    }

    analysis_id = str(uuid4())
    meta = {
        "analysis_id": analysis_id, "species": species_name, "height_m": height_m, "crown_width_m": crown_m, "trunk_diameter_m": trunk_m,
        "height_m_ai": height_m_ai, "crown_width_m_ai": crown_m_ai, "trunk_diameter_m_ai": trunk_m_ai, "crown_density_ai": crown_density_ai,
        "lean_angle_deg": lean_angle_deg, "ar_measurements": ar_measurements, "measurement_sources": measurement_sources,
        "dimensions_source": dimensions_source, "beta": beta_info, "analytic_wind_model": analytic_wind_model, "scale_px_to_m": scale,
        "gps": gps, "address": address, "risk": risk_data, "model_versions": MODEL_VERSIONS, "build": BUILD_INFO, "schema_version": SCHEMA_VERSION, "api_version": API_VERSION,
        "ai_settings": {"conf": conf_val, "smoothness": smooth_k, "use_rembg": use_rembg},
        "feedback_image": {
            "width": feedback_assets["width"],
            "height": feedback_assets["height"],
            "source_width": feedback_assets["source_width"],
            "source_height": feedback_assets["source_height"],
        },
    }

    try:
        # Keep the original upload for audit, and a resized image that exactly
        # matches both the automatic and future user masks.
        supabase_upload_bytes(
            SUPABASE_BUCKET_RAW,
            f"{analysis_id}/input_original.jpg",
            image_bytes,
        )
        # Backward-compatible name for older tools.
        supabase_upload_bytes(
            SUPABASE_BUCKET_RAW,
            f"{analysis_id}/input.jpg",
            feedback_assets["image_bytes"],
        )
        supabase_upload_bytes(
            SUPABASE_BUCKET_RAW,
            f"{analysis_id}/feedback_input.jpg",
            feedback_assets["image_bytes"],
        )
        supabase_upload_bytes(
            SUPABASE_BUCKET_RAW,
            f"{analysis_id}/annotated.jpg",
            feedback_assets["annotated_bytes"],
        )
        supabase_upload_bytes(
            SUPABASE_BUCKET_RAW,
            f"{analysis_id}/mask_auto.png",
            feedback_assets["mask_bytes"],
        )
        supabase_upload_json(
            SUPABASE_BUCKET_RAW,
            f"{analysis_id}/meta_auto.json",
            meta,
        )
    except Exception as e:
        print(f"[!] Failed to upload raw sample: {e}")

    try:
        tmp_dir = Path("/tmp") / analysis_id
        tmp_dir.mkdir(parents=True, exist_ok=True)
        (tmp_dir / "input.jpg").write_bytes(feedback_assets["image_bytes"])
        (tmp_dir / "feedback_input.jpg").write_bytes(
            feedback_assets["image_bytes"]
        )
        (tmp_dir / "annotated.jpg").write_bytes(
            feedback_assets["annotated_bytes"]
        )
        (tmp_dir / "mask_auto.png").write_bytes(
            feedback_assets["mask_bytes"]
        )
        (tmp_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        print(f"[!] Failed to cache in /tmp: {e}")

    response = {
        "analysis_id": analysis_id, "species": species_name, "height_m": height_m, "crown_width_m": crown_m, "trunk_diameter_m": trunk_m,
        "height_m_ai": height_m_ai, "crown_width_m_ai": crown_m_ai, "trunk_diameter_m_ai": trunk_m_ai, "ar_measurements": ar_measurements,
        "measurement_sources": measurement_sources, "dimensions_source": dimensions_source, "beta": beta_info, "analytic_wind_model": analytic_wind_model,
        "scale_px_to_m": scale,
        "original_image_base64": feedback_assets["image_base64"],
        "annotated_image_base64": feedback_assets["annotated_base64"],
        "mask_image_base64": feedback_assets["mask_base64"],
        "feedback_image_width": feedback_assets["width"],
        "feedback_image_height": feedback_assets["height"],
        "gps": gps, "address": address, "risk": risk_data,
    }

    response["server_saved"] = analysis_user is not None
    try: _save_analysis_record(response, analysis_user)
    except Exception as e: print(f"[!] Failed to save DB record: {e}")

    return JSONResponse(response)


def _validate_analysis_id(analysis_id: str) -> str:
    value = (analysis_id or "").strip()
    if not re.fullmatch(
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
        value,
    ):
        raise HTTPException(status_code=422, detail="Некорректный analysis_id")
    return value.lower()


def _download_optional(bucket: str, path: str) -> bytes | None:
    try:
        return supabase_download_bytes(bucket, path)
    except Exception:
        return None


def _json_from_bytes(raw: bytes | None) -> dict:
    if not raw:
        return {}
    try:
        value = json.loads(raw.decode("utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _load_feedback_artifacts(analysis_id: str) -> dict:
    """Load feedback source files from local cache or durable RAW storage."""
    tmp_dir = Path("/tmp") / analysis_id
    if tmp_dir.exists():
        input_path = (
            tmp_dir / "feedback_input.jpg"
            if (tmp_dir / "feedback_input.jpg").exists()
            else tmp_dir / "input.jpg"
        )
        meta_path = tmp_dir / "meta.json"
        if input_path.exists() and meta_path.exists():
            try:
                return {
                    "source": "tmp",
                    "tmp_dir": tmp_dir,
                    "input_bytes": input_path.read_bytes(),
                    "annotated_bytes": (
                        (tmp_dir / "annotated.jpg").read_bytes()
                        if (tmp_dir / "annotated.jpg").exists()
                        else None
                    ),
                    "auto_mask_bytes": (
                        (tmp_dir / "mask_auto.png").read_bytes()
                        if (tmp_dir / "mask_auto.png").exists()
                        else None
                    ),
                    "meta": json.loads(meta_path.read_text(encoding="utf-8")),
                }
            except Exception as exc:
                print(f"[!] Failed to read feedback cache {analysis_id}: {exc}")

    input_bytes = (
        _download_optional(
            SUPABASE_BUCKET_RAW,
            f"{analysis_id}/feedback_input.jpg",
        )
        or _download_optional(
            SUPABASE_BUCKET_RAW,
            f"{analysis_id}/input.jpg",
        )
        or _download_optional(
            SUPABASE_BUCKET_RAW,
            f"{analysis_id}/input_original.jpg",
        )
    )
    meta = _json_from_bytes(
        _download_optional(
            SUPABASE_BUCKET_RAW,
            f"{analysis_id}/meta_auto.json",
        )
    )
    if not input_bytes or not meta:
        raise HTTPException(
            status_code=404,
            detail=(
                "Анализ не найден. Исходные данные отсутствуют в кеше и "
                "в хранилище arborscan-raw."
            ),
        )

    return {
        "source": "supabase_raw",
        "tmp_dir": tmp_dir,
        "input_bytes": input_bytes,
        "annotated_bytes": _download_optional(
            SUPABASE_BUCKET_RAW,
            f"{analysis_id}/annotated.jpg",
        ),
        "auto_mask_bytes": _download_optional(
            SUPABASE_BUCKET_RAW,
            f"{analysis_id}/mask_auto.png",
        ),
        "meta": meta,
    }


@app.post("/feedback")
@app.post("/api/feedback")
def send_feedback(
    payload: dict = Body(...),
    authorization: str | None = Header(default=None),
):
    analysis_id = _validate_analysis_id(
        payload.get("analysis_id") or payload.get("analysisId") or ""
    )

    def _as_bool(value, default: bool = True) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "y", "ok"}:
                return True
            if normalized in {"0", "false", "no", "n"}:
                return False
        return default

    def _as_float(*values):
        for value in values:
            if value is None:
                continue
            if isinstance(value, (int, float)):
                result = float(value)
            elif isinstance(value, str):
                normalized = value.strip().replace(",", ".")
                if not normalized or normalized.lower() in {
                    "null",
                    "none",
                    "nan",
                }:
                    continue
                try:
                    result = float(normalized)
                except ValueError:
                    continue
            else:
                continue
            if math.isfinite(result):
                return result
        return None

    use_for_training = _as_bool(
        payload.get("use_for_training", payload.get("useForTraining")),
        default=True,
    )
    tree_ok = _as_bool(
        payload.get("tree_ok", payload.get("treeOk")),
        default=True,
    )
    stick_ok = _as_bool(
        payload.get("stick_ok", payload.get("stickOk")),
        default=True,
    )
    params_ok = _as_bool(
        payload.get("params_ok", payload.get("paramsOk")),
        default=True,
    )
    species_ok = _as_bool(
        payload.get("species_ok", payload.get("speciesOk")),
        default=True,
    )

    correct_species_raw = (
        payload.get("correct_species") or payload.get("correctSpecies")
    )
    correct_species = (
        str(correct_species_raw).strip() if correct_species_raw is not None else None
    )
    if correct_species == "":
        correct_species = None

    corrected_height_m = _as_float(
        payload.get("corrected_height_m"),
        payload.get("correctedHeightM"),
        payload.get("height_m_corrected"),
    )
    corrected_crown_width_m = _as_float(
        payload.get("corrected_crown_width_m"),
        payload.get("correctedCrownWidthM"),
        payload.get("crown_width_m_corrected"),
    )
    corrected_trunk_diameter_m = _as_float(
        payload.get("corrected_trunk_diameter_m"),
        payload.get("correctedTrunkDiameterM"),
        payload.get("trunk_diameter_m_corrected"),
    )
    corrected_scale_px_to_m = _as_float(
        payload.get("corrected_scale_px_to_m"),
        payload.get("correctedScalePxToM"),
        payload.get("scale_px_to_m_corrected"),
        payload.get("scalePxToMCorrected"),
    )
    user_mask_base64 = (
        payload.get("user_mask_base64")
        or payload.get("userMaskBase64")
        or payload.get("mask_base64")
        or payload.get("maskBase64")
    )

    if corrected_height_m is not None and not 0.5 <= corrected_height_m <= 100:
        raise HTTPException(status_code=422, detail="Высота должна быть от 0,5 до 100 м")
    if (
        corrected_crown_width_m is not None
        and not 0.1 <= corrected_crown_width_m <= 100
    ):
        raise HTTPException(
            status_code=422,
            detail="Ширина кроны должна быть от 0,1 до 100 м",
        )
    if (
        corrected_trunk_diameter_m is not None
        and not 0.01 <= corrected_trunk_diameter_m <= 10
    ):
        raise HTTPException(
            status_code=422,
            detail="Диаметр ствола должен быть от 0,01 до 10 м",
        )
    if corrected_scale_px_to_m is not None and corrected_scale_px_to_m <= 0:
        raise HTTPException(status_code=422, detail="Масштаб должен быть больше нуля")

    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise HTTPException(status_code=500, detail="Supabase не настроен")

    resolved_token = _resolve_auth_token(
        authorization,
        payload.get("auth_token") or payload.get("authToken"),
    )
    feedback_user = _get_user_by_token(resolved_token or "")
    verifier_role = (
        str(feedback_user.get("role") or "user").strip().lower()
        if feedback_user
        else "anonymous"
    )
    if verifier_role not in {"admin", "user"}:
        verifier_role = "user"

    artifacts = _load_feedback_artifacts(analysis_id)
    input_bytes: bytes = artifacts["input_bytes"]
    annotated_bytes: bytes | None = artifacts.get("annotated_bytes")
    meta = dict(artifacts["meta"])

    existing_verified_meta = _json_from_bytes(
        _download_optional(
            SUPABASE_BUCKET_VERIFIED,
            f"{analysis_id}/meta_verified.json",
        )
    )
    existing_mask_bytes = (
        _download_optional(
            SUPABASE_BUCKET_VERIFIED,
            f"{analysis_id}/user_mask.png",
        )
        or _download_optional(
            SUPABASE_BUCKET_INPUTS,
            f"{analysis_id}/user_mask.png",
        )
    )

    submitted_mask_bytes = None
    mask_text = str(user_mask_base64 or "").strip()
    if mask_text and mask_text.lower() not in {"null", "undefined"}:
        try:
            submitted_mask_bytes = normalize_mask_to_image(
                ensure_png_mask_bytes(mask_text),
                input_bytes,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Не удалось обработать пользовательскую маску: {exc}",
            )

    final_mask_bytes = submitted_mask_bytes or existing_mask_bytes
    if final_mask_bytes is not None:
        final_mask_bytes = normalize_mask_to_image(final_mask_bytes, input_bytes)
    has_user_mask = final_mask_bytes is not None

    # A corrected mask is positive evidence even when the automatic mask was
    # marked as incorrect. The same applies to an explicitly corrected species.
    tree_component_ok = tree_ok or has_user_mask
    species_component_ok = species_ok or bool(correct_species)
    trust = round(
        (0.30 if tree_component_ok else 0.0)
        + (0.20 if stick_ok else 0.0)
        + (0.20 if params_ok else 0.0)
        + (0.30 if species_component_ok else 0.0),
        3,
    )

    now_iso = _now_iso()
    meta.update(
        {
            "analysis_id": analysis_id,
            "tree_ok": tree_ok,
            "stick_ok": stick_ok,
            "params_ok": params_ok,
            "species_ok": species_ok,
            "correct_species": correct_species,
            "has_user_mask": has_user_mask,
            "use_for_training": use_for_training,
            "exclude_from_training": not use_for_training,
            "trust_score": trust,
            "feedback_received_at": now_iso,
            "feedback_source": artifacts["source"],
            "verifier_role": verifier_role,
            "verifier_user_id": (
                feedback_user.get("id") if feedback_user else None
            ),
            "feedback_revision": int(
                existing_verified_meta.get("feedback_revision", 0) or 0
            )
            + 1,
        }
    )

    if not species_ok and correct_species:
        meta["species"] = correct_species
    if corrected_height_m is not None:
        meta["height_m"] = corrected_height_m
    if corrected_crown_width_m is not None:
        meta["crown_width_m"] = corrected_crown_width_m
    if corrected_trunk_diameter_m is not None:
        meta["trunk_diameter_m"] = corrected_trunk_diameter_m
    if corrected_scale_px_to_m is not None:
        meta["scale_px_to_m"] = corrected_scale_px_to_m

    is_verified = bool(
        use_for_training and trust >= VERIFIED_TRUST_THRESHOLD
    )
    meta["verified"] = is_verified
    if is_verified:
        meta["verified_at"] = now_iso

    feedback_audit = {
        "analysis_id": analysis_id,
        "received_at": now_iso,
        "verifier_role": verifier_role,
        "verifier_user_id": feedback_user.get("id") if feedback_user else None,
        "use_for_training": use_for_training,
        "quality": {
            "tree_ok": tree_ok,
            "stick_ok": stick_ok,
            "params_ok": params_ok,
            "species_ok": species_ok,
        },
        "corrections": {
            "species": correct_species,
            "height_m": corrected_height_m,
            "crown_width_m": corrected_crown_width_m,
            "trunk_diameter_m": corrected_trunk_diameter_m,
            "scale_px_to_m": corrected_scale_px_to_m,
        },
        "has_user_mask": has_user_mask,
        "trust_score": trust,
        "verified": is_verified,
    }

    try:
        # Durable audit is always stored, even when the user opts out of model
        # training. Opt-out therefore means "do not train", not "discard".
        supabase_upload_json(
            SUPABASE_BUCKET_RAW,
            f"{analysis_id}/feedback.json",
            feedback_audit,
        )
        supabase_upload_json(
            SUPABASE_BUCKET_RAW,
            f"{analysis_id}/meta_feedback.json",
            meta,
        )
        if submitted_mask_bytes is not None:
            supabase_upload_bytes(
                SUPABASE_BUCKET_RAW,
                f"{analysis_id}/user_mask.png",
                submitted_mask_bytes,
            )

        supabase_upload_bytes(
            SUPABASE_BUCKET_INPUTS,
            f"{analysis_id}/input.jpg",
            input_bytes,
        )
        if annotated_bytes:
            supabase_upload_bytes(
                SUPABASE_BUCKET_INPUTS,
                f"{analysis_id}/annotated.jpg",
                annotated_bytes,
            )
        if final_mask_bytes:
            supabase_upload_bytes(
                SUPABASE_BUCKET_INPUTS,
                f"{analysis_id}/user_mask.png",
                final_mask_bytes,
            )
        supabase_upload_json(
            SUPABASE_BUCKET_META,
            f"{analysis_id}.json",
            meta,
        )

        if is_verified:
            supabase_upload_bytes(
                SUPABASE_BUCKET_VERIFIED,
                f"{analysis_id}/input.jpg",
                input_bytes,
            )
            if annotated_bytes:
                supabase_upload_bytes(
                    SUPABASE_BUCKET_VERIFIED,
                    f"{analysis_id}/annotated.jpg",
                    annotated_bytes,
                )
            if final_mask_bytes:
                supabase_upload_bytes(
                    SUPABASE_BUCKET_VERIFIED,
                    f"{analysis_id}/user_mask.png",
                    final_mask_bytes,
                )
            # Preserve worker/admin fields that may already exist.
            meta_verified = dict(existing_verified_meta)
            meta_verified.update(meta)
            supabase_upload_json(
                SUPABASE_BUCKET_VERIFIED,
                f"{analysis_id}/meta_verified.json",
                meta_verified,
            )
        elif existing_verified_meta:
            # A later opt-out must immediately exclude an already verified
            # sample without deleting its audit history.
            updated_existing = dict(existing_verified_meta)
            updated_existing.update(
                {
                    "use_for_training": use_for_training,
                    "exclude_from_training": not use_for_training,
                    "feedback_received_at": now_iso,
                    "feedback_revision": meta["feedback_revision"],
                }
            )
            supabase_upload_json(
                SUPABASE_BUCKET_VERIFIED,
                f"{analysis_id}/meta_verified.json",
                updated_existing,
            )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка сохранения обратной связи в Supabase: {exc}",
        )

    if SUPABASE_ENABLE_QUEUE:
        try:
            supabase_db_insert(
                SUPABASE_QUEUE_TABLE,
                {
                    "analysis_id": analysis_id,
                    "trust_score": trust,
                    "species": meta.get("species"),
                    "has_user_mask": has_user_mask,
                    "tree_ok": tree_ok,
                    "stick_ok": stick_ok,
                    "params_ok": params_ok,
                    "species_ok": species_ok,
                },
            )
        except Exception as exc:
            print(f"[!] Queue error: {exc}")

    tmp_dir: Path = artifacts["tmp_dir"]
    try:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass

    if not use_for_training:
        status = "saved_not_for_training"
    elif is_verified:
        status = "verified"
    else:
        status = "saved_pending_review"

    return {
        "status": status,
        "analysis_id": analysis_id,
        "trust_score": trust,
        "verified": is_verified,
        "use_for_training": use_for_training,
        "has_user_mask": has_user_mask,
        "recovered_from": artifacts["source"],
        "corrected": {
            "species": meta.get("species"),
            "height_m": meta.get("height_m"),
            "crown_width_m": meta.get("crown_width_m"),
            "trunk_diameter_m": meta.get("trunk_diameter_m"),
            "scale_px_to_m": meta.get("scale_px_to_m"),
        },
    }


@app.get("/admin/verified-list")
def admin_verified_list(
    include_used: bool = False,
    admin: dict = Depends(require_admin),
):
    del admin
    try:
        objects = supabase_list_objects(SUPABASE_BUCKET_VERIFIED)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    analysis_ids = sorted(
        {
            obj["name"].split("/")[0]
            for obj in objects
            if obj.get("name")
        }
    )
    results = []
    for analysis_id in analysis_ids:
        try:
            meta = json.loads(
                supabase_download_bytes(
                    SUPABASE_BUCKET_VERIFIED,
                    f"{analysis_id}/meta_verified.json",
                )
            )
            if not include_used and meta.get("used_for_training") is True:
                continue
            results.append(
                {
                    "analysis_id": analysis_id,
                    "species": meta.get("species"),
                    "risk_category": (meta.get("risk") or {}).get("category"),
                    "trust_score": meta.get("trust_score"),
                    "verified": meta.get("verified", True),
                    "verified_at": meta.get("verified_at"),
                    "exclude_from_training": (
                        meta.get("exclude_from_training", False) is True
                    ),
                    "has_user_mask": meta.get("has_user_mask", False) is True,
                    "used_for_training": (
                        meta.get("used_for_training", False) is True
                    ),
                }
            )
        except Exception:
            continue
    return {"count": len(results), "items": results}


@app.post("/admin/verified/{analysis_id}/set-training")
def admin_set_training_flag(
    analysis_id: str,
    req: AdminSetTrainingRequest,
    admin: dict = Depends(require_admin),
):
    del admin
    flag = next(
        (
            value
            for value in (
                req.use_for_training,
                req.enabled,
                req.include,
                req.value,
            )
            if value is not None
        ),
        None,
    )
    if flag is None:
        raise HTTPException(status_code=400, detail="Missing boolean flag")

    def _load_json(path: str) -> dict:
        try:
            return json.loads(
                supabase_download_bytes(
                    SUPABASE_BUCKET_VERIFIED,
                    path,
                ).decode("utf-8")
            )
        except Exception:
            return {}

    for path in (
        f"{analysis_id}/meta.json",
        f"{analysis_id}/meta_verified.json",
    ):
        data = _load_json(path)
        data.update(
            {
                "analysis_id": analysis_id,
                "use_for_training": bool(flag),
                "exclude_from_training": not bool(flag),
                "training_flag_updated_at": _now_iso(),
            }
        )
        supabase_upload_json(SUPABASE_BUCKET_VERIFIED, path, data)

    log_training_event(
        "INFO",
        "Администратор изменил включение примера в датасет",
        {"analysis_id": analysis_id, "include": bool(flag)},
    )
    return {
        "analysis_id": analysis_id,
        "use_for_training": bool(flag),
        "exclude_from_training": not bool(flag),
    }


@app.get("/admin/analysis/{analysis_id}")
def admin_get_analysis(
    analysis_id: str,
    admin: dict = Depends(require_admin),
):
    del admin
    try:
        input_img = supabase_download_bytes(
            SUPABASE_BUCKET_VERIFIED,
            f"{analysis_id}/input.jpg",
        )
        annotated_img = supabase_download_bytes(
            SUPABASE_BUCKET_VERIFIED,
            f"{analysis_id}/annotated.jpg",
        )
        try:
            user_mask_img = supabase_download_bytes(
                SUPABASE_BUCKET_VERIFIED,
                f"{analysis_id}/user_mask.png",
            )
        except Exception:
            user_mask_img = None

        try:
            tree_pred = json.loads(
                supabase_download_bytes(
                    SUPABASE_BUCKET_VERIFIED,
                    f"{analysis_id}/tree_pred.json",
                )
            )
        except Exception:
            tree_pred = {}

        try:
            stick_pred = json.loads(
                supabase_download_bytes(
                    SUPABASE_BUCKET_VERIFIED,
                    f"{analysis_id}/stick_pred.json",
                )
            )
        except Exception:
            stick_pred = {}

        meta = json.loads(
            supabase_download_bytes(
                SUPABASE_BUCKET_VERIFIED,
                f"{analysis_id}/meta_verified.json",
            )
        )
    except Exception as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Analysis not found: {exc}",
        )

    return {
        "analysis_id": analysis_id,
        "images": {
            "input_base64": base64.b64encode(input_img).decode("utf-8"),
            "annotated_base64": base64.b64encode(annotated_img).decode("utf-8"),
            "user_mask_base64": (
                base64.b64encode(user_mask_img).decode("utf-8")
                if user_mask_img
                else None
            ),
        },
        "tree_pred": tree_pred,
        "stick_pred": stick_pred,
        "meta": meta,
    }


@app.get("/admin/training-status")
def admin_training_status(admin: dict = Depends(require_admin)):
    del admin
    training_state_ensure_row()
    state = training_state_get()
    return {
        "active_model_version": state.get("active_model_version", 0),
        "last_model_version": state.get("last_model_version", 0),
        "training_in_progress": state.get("training_in_progress", False),
        "retrain_requested": state.get("retrain_requested", False),
        "last_error": state.get("last_error"),
        "training_started_at": state.get("training_started_at"),
        "training_completed_at": state.get("training_completed_at"),
    }


@app.get("/admin/training-events")
def admin_training_events(
    limit: int = 15,
    admin: dict = Depends(require_admin),
):
    del admin
    normalized_limit = max(1, min(int(limit), 200))
    events = list(TRAINING_EVENTS)[-normalized_limit:]
    return {"events": list(reversed(events))}


@app.post("/admin/set-active-model")
async def admin_set_active_model(
    payload: dict = Body(...),
    admin: dict = Depends(require_admin),
):
    del admin
    raw_version = (
        payload.get("version")
        if payload.get("version") is not None
        else payload.get("model_version")
    )
    if raw_version is None:
        raw_version = payload.get("active_model_version")
    if raw_version is None:
        raise HTTPException(status_code=422, detail="Missing model version")

    try:
        version = int(raw_version)
        model_info = await run_in_threadpool(activate_tree_model, version)
    except Exception as exc:
        log_training_event(
            "ERROR",
            "Не удалось активировать модель",
            {"version": raw_version, "error": str(exc)},
        )
        raise HTTPException(
            status_code=400,
            detail=f"Model v{raw_version} could not be activated: {exc}",
        )

    log_training_event(
        "INFO",
        "Активная модель изменена",
        {"version": version},
    )
    return {
        "status": "ok",
        "active_model_version": version,
        "model": model_info,
    }


@app.post("/admin/request-retrain")
def admin_request_retrain(admin: dict = Depends(require_admin)):
    del admin
    training_state_ensure_row()
    training_state_update({"retrain_requested": True})
    log_training_event("INFO", "Администратор запросил переобучение")
    return {"status": "ok", "retrain_requested": True}


@app.get("/admin/models")
def admin_models(admin: dict = Depends(require_admin)):
    del admin
    return {
        "models": list_available_model_versions(),
        "active_model_version": _get_active_model_version(),
    }

