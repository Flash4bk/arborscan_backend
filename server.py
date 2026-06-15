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
import cv2
import numpy as np
import requests
import torch
from ultralytics import YOLO
from PIL import Image, ExifTags
from fastapi import FastAPI, File, UploadFile, HTTPException, Body, Form
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
from uuid import uuid4
from pathlib import Path
from pydantic import BaseModel
from datetime import datetime, timedelta
from collections import deque
from typing import Optional, Dict, Any, List, Tuple

from rembg import remove, new_session

try:
    from google.oauth2 import id_token as google_id_token
    from google.auth.transport import requests as google_requests
except Exception:
    google_id_token = None
    google_requests = None

# -------------------------------------
# CONFIG
# -------------------------------------

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    print("[!] Warning: SUPABASE_URL or SUPABASE_SERVICE_KEY not set. /feedback will not upload to Supabase.")

SUPABASE_BUCKET_INPUTS = "arborscan-inputs"
SUPABASE_BUCKET_PRED = "arborscan-predictions"
SUPABASE_BUCKET_META = "arborscan-meta"
SUPABASE_BUCKET_VERIFIED = "arborscan-verified"
SUPABASE_BUCKET_RAW = "arborscan-raw"
SUPABASE_BUCKET_MODELS = os.getenv("SUPABASE_BUCKET_MODELS", "arborscan-models")

# Supabase Postgres REST API
SUPABASE_DB_BASE = SUPABASE_URL.rstrip("/") + "/rest/v1" if SUPABASE_URL else None
SUPABASE_QUEUE_TABLE = "arborscan_feedback_queue"
SUPABASE_ENABLE_QUEUE = os.getenv("SUPABASE_ENABLE_QUEUE", "false").lower() == "true"

# Pl@ntNet API Key
PLANTNET_API_KEY = os.getenv("PLANTNET_API_KEY", "2b10s2QVGCWyalEeU1xv2nOKO")

NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
NOMINATIM_USER_AGENT = os.getenv("NOMINATIM_USER_AGENT", "arborscan-backend/1.0")

ENABLE_ENV_ANALYSIS = os.getenv("ENABLE_ENV_ANALYSIS", "true").lower() == "true"

# ---------------------------------------------------------
# Supabase PostgREST helpers 
# ---------------------------------------------------------

def _sb_headers(json_ct: bool = True) -> dict:
    h = {
        "apikey": SUPABASE_SERVICE_KEY or "",
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}" if SUPABASE_SERVICE_KEY else "",
    }
    if json_ct:
        h["Content-Type"] = "application/json"
    return h


def training_state_get() -> dict:
    if not SUPABASE_DB_BASE:
        raise RuntimeError("Supabase DB is not configured")
    url = f"{SUPABASE_DB_BASE}/training_state?id=eq.1&select=*"
    resp = requests.get(url, headers=_sb_headers(json_ct=False), timeout=30)
    if resp.status_code >= 400:
        raise RuntimeError(f"training_state_get error {resp.status_code}: {resp.text}")
    rows = resp.json()
    if not rows:
        return {}
    return rows[0]


def training_state_ensure_row():
    state = training_state_get()
    if state:
        return
    url = f"{SUPABASE_DB_BASE}/training_state"
    payload = {
        "id": 1,
        "retrain_requested": False,
        "training_in_progress": False,
        "last_model_version": 0,
        "active_model_version": 0,
    }
    requests.post(url, headers={**_sb_headers(), "Prefer": "return=representation"}, data=json.dumps(payload), timeout=30)


def training_state_update(fields: dict) -> dict:
    if not SUPABASE_DB_BASE:
        raise RuntimeError("Supabase DB is not configured")
    url = f"{SUPABASE_DB_BASE}/training_state?id=eq.1"
    resp = requests.patch(url, headers={**_sb_headers(), "Prefer": "return=representation"}, data=json.dumps(fields), timeout=30)
    rows = resp.json()
    return rows[0] if rows else fields


# -------------------------------------
# MODEL VERSIONS
# -------------------------------------

MODEL_VERSIONS = {
    "tree_yolo": "tree_yolov8_seg_v1.2.0",
    "stick_yolo": "stick_yolov8_det_v1.0.3",
    "classifier": "plantnet_api_v2", 
    "mask_refiner": "u2net_rembg_v1.0"
}
BUILD_INFO = {
    "git_commit": os.getenv("GIT_COMMIT", "unknown"),
    "build_time": os.getenv("BUILD_TIME")
}
SCHEMA_VERSION = "1.0.0"
API_VERSION = "2.2.0" # Added Proportional & Statistical Scaling
VERIFIED_TRUST_THRESHOLD = 0.0

REAL_STICK_M = 1.0
CLASS_NAMES_RU = ["Береза", "Дуб", "Ель", "Сосна", "Тополь"]

# -------------------------------------
# LOADING AI MODELS
# -------------------------------------

print("[*] Loading YOLO models...")
tree_model = None  
stick_model = YOLO("models/stick_model.pt")
print("[*] YOLO Models loaded.")

REMBG_SESSION = None

# =============================================
# SUPABASE UTILS (Storage + DB)
# =============================================

def supabase_upload_bytes(bucket: str, path: str, data: bytes):
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise RuntimeError("Supabase is not configured")
    url = SUPABASE_URL.rstrip("/") + f"/storage/v1/object/{bucket}/{path}"
    headers = {
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/octet-stream",
        "x-upsert": "true",
    }
    requests.post(url, headers=headers, data=data, timeout=30)

def supabase_upload_json(bucket: str, path: str, obj: dict):
    data = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
    supabase_upload_bytes(bucket, path, data)

def supabase_list_objects(bucket: str, prefix: str = ""):
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise RuntimeError("Supabase is not configured")
    url = SUPABASE_URL.rstrip("/") + f"/storage/v1/object/list/{bucket}"
    payload = {
        "prefix": prefix,
        "limit": 200,
        "offset": 0,
        "sortBy": {"column": "name", "order": "desc"},
    }
    resp = requests.post(url, headers=_sb_headers(), json=payload, timeout=15)
    return resp.json()

def supabase_download_bytes(bucket: str, path: str) -> bytes:
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise RuntimeError("Supabase is not configured")
    url = SUPABASE_URL.rstrip("/") + f"/storage/v1/object/authenticated/{bucket}/{path}"
    resp = requests.get(url, headers=_sb_headers(False), timeout=60)
    return resp.content


# ---------------------------------------------------------
# Model hot-swap (tree model) 
# ---------------------------------------------------------

TREE_MODEL: Optional[YOLO] = None
TREE_MODEL_VERSION: Optional[int] = None
MODEL_LOCK = threading.Lock()
_MODEL_LAST_CHECK_TS = 0.0
_MODEL_CHECK_INTERVAL_SEC = float(os.getenv("MODEL_CHECK_INTERVAL_SEC", "2.0"))

def _local_model_path(version: int) -> str:
    cache_dir = os.getenv("MODEL_CACHE_DIR", "/tmp/models")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    return str(Path(cache_dir) / f"model_v{version}.pt")

def _download_model_if_needed(version: int) -> str:
    filename = f"model_v{version}.pt"
    local_path = _local_model_path(version)
    if os.path.exists(local_path):
        return local_path
    data = supabase_download_bytes(SUPABASE_BUCKET_MODELS, filename)
    with open(local_path, "wb") as f:
        f.write(data)
    return local_path

def _get_active_model_version() -> int:
    state = training_state_get()
    v = state.get("active_model_version")
    return int(v) if v is not None else 0

def list_available_model_versions() -> list[dict]:
    versions: set[int] = set()
    try:
        objects = supabase_list_objects(SUPABASE_BUCKET_MODELS)
        for obj in objects:
            name = obj.get("name") or ""
            base = name.split("/")[-1]
            mm = re.search(r"model_v(\d+)\.pt$", base)
            if mm:
                versions.add(int(mm.group(1)))
    except Exception:
        pass

    env_hint = os.getenv("AVAILABLE_MODEL_VERSIONS", "").strip()
    if env_hint:
        for part in env_hint.split(","):
            part = part.strip()
            if part:
                try: versions.add(int(part))
                except ValueError: pass

    for p in Path("/tmp/models").glob("model_v*.pt"):
        mm = re.search(r"model_v(\d+)\.pt$", p.name)
        if mm: versions.add(int(mm.group(1)))

    if Path("models").exists():
        for p in Path("models").glob("model_v*.pt"):
            mm = re.search(r"model_v(\d+)\.pt$", p.name)
            if mm: versions.add(int(mm.group(1)))

    active = _get_active_model_version()
    versions.add(active)
    return [{"version": v, "is_active": v == active} for v in sorted(versions)]

def reload_tree_model(force: bool = False):
    global TREE_MODEL, TREE_MODEL_VERSION, _MODEL_LAST_CHECK_TS
    now = time.time()
    if not force and (now - _MODEL_LAST_CHECK_TS) < _MODEL_CHECK_INTERVAL_SEC:
        return
    _MODEL_LAST_CHECK_TS = now
    v = _get_active_model_version()
    if not force and TREE_MODEL is not None and TREE_MODEL_VERSION == v:
        return

    if v == 0:
        local_fallback = "models/tree_model.pt"
        if os.path.exists(local_fallback):
            print(f"[*] Using bundled tree model: {local_fallback}")
            TREE_MODEL = YOLO(local_fallback)
            TREE_MODEL_VERSION = 0
            return
        try:
            path = _download_model_if_needed(0)
            print(f"[*] Using downloaded tree model v0: {path}")
            TREE_MODEL = YOLO(path)
            TREE_MODEL_VERSION = 0
            return
        except Exception as e:
            raise RuntimeError(f"No tree model available (v0). {e}")

    path = _download_model_if_needed(v)
    print(f"[*] Switching tree model to v{v}: {path}")
    TREE_MODEL = YOLO(path)
    TREE_MODEL_VERSION = v

def get_tree_model() -> YOLO:
    with MODEL_LOCK:
        reload_tree_model(force=False)
        if TREE_MODEL is None:
            reload_tree_model(force=True)
        return TREE_MODEL


# =============================================
# BASE64 / IMAGE UTILS
# =============================================

def _strip_data_url(b64: str) -> str:
    if not b64: return b64
    b64 = b64.strip()
    if b64.startswith("data:") and "base64," in b64:
        b64 = b64.split("base64,", 1)[1]
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
            if raw2.startswith(b"\x89PNG\r\n\x1a\n") or raw2[:3] == b"\xff\xd8\xff":
                return raw2
    except Exception:
        pass
    return raw

def ensure_png_mask_bytes(mask_b64: str) -> bytes:
    raw = decode_base64_bytes(mask_b64)
    try:
        if raw[:1] in (b"{", b"["):
            obj = json.loads(raw.decode("utf-8"))
            if isinstance(obj, dict) and obj.get("mask_png_base64"):
                raw = decode_base64_bytes(str(obj["mask_png_base64"]))
    except Exception:
        pass

    np_buf = np.frombuffer(raw, np.uint8)
    mask = cv2.imdecode(np_buf, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError("user_mask_base64 is not a valid PNG/JPEG image payload")
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    ok, out = cv2.imencode(".png", mask_bin)
    if not ok:
        raise ValueError("Failed to encode mask as PNG")
    return out.tobytes()

def encode_jpeg_base64(img_bgr, max_side=1280, quality=74):
    h, w = img_bgr.shape[:2]
    longest = max(h, w)
    if longest > max_side:
        scale = max_side / float(longest)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    ok, out = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return base64.b64encode(out.tobytes()).decode("ascii")

def draw_mask(img_bgr, mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        approx = cv2.approxPolyDP(cnt, 0.003 * cv2.arcLength(cnt, True), True)
        cv2.drawContours(img_bgr, [approx], -1, (0, 255, 0), 3)
    return encode_jpeg_base64(img_bgr, max_side=1280, quality=74)

# =============================================
# EXIF → GPS
# =============================================

def _deg(v):
    d = v[0][0] / v[0][1]
    m = v[1][0] / v[1][1]
    s = v[2][0] / v[2][1]
    return d + m / 60 + s / 3600

def extract_gps(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        exif = img._getexif()
        if not exif: return None
        gps_info = None
        for k, v in exif.items():
            if ExifTags.TAGS.get(k) == "GPSInfo":
                gps_info = v
                break
        if not gps_info: return None
        lat = _deg(gps_info[2])
        lon = _deg(gps_info[4])
        if gps_info[1] == "S": lat = -lat
        if gps_info[3] == "W": lon = -lon
        return {"lat": lat, "lon": lon}
    except Exception:
        return None

def reverse_geocode(lat, lon):
    try:
        r = requests.get(NOMINATIM_URL, params={"lat": lat, "lon": lon, "format": "jsonv2"}, headers={"User-Agent": NOMINATIM_USER_AGENT}, timeout=5)
        return r.json().get("display_name")
    except Exception:
        return None


# =============================================
# ПРОФЕССИОНАЛЬНАЯ ОЦЕНКА РИСКА (SIA METHOD)
# =============================================

SPECIES_STRENGTH_MPA = {
    "Береза": 45.0,
    "Дуб": 60.0,
    "Ель": 40.0,
    "Сосна": 42.0,
    "Тополь": 30.0,
    "Клен": 50.0,
    "Ясень": 55.0,
    "Липа": 35.0,
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

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def estimate_beta_kg_s(species: str, height_m, manual_beta_kg_s=None, crown_density_factor=1.0) -> dict:
    stats = BETA_EMPIRICAL_STATS.get(species, BETA_EMPIRICAL_STATS["Сосна"])

    if manual_beta_kg_s is not None and manual_beta_kg_s > 0:
        value = round(_clamp(float(manual_beta_kg_s), 5.0, 200.0), 2)
        return {
            "beta_kg_s": value,
            "beta_max_scenario": value,
            "method": "manual",
            "source": "Введено вручную",
            "input": {"manual_beta_kg_s": float(manual_beta_kg_s)},
            "notes": ["Использован пользовательский коэффициент."],
        }

    h = float(height_m or 0)
    density = float(crown_density_factor or 1.0)

    if h <= 0:
        return {
            "beta_kg_s": stats["mean"],
            "beta_max_scenario": stats["max"],
            "method": "species_default",
            "source": "Статистическое среднее",
            "input": {"species": species},
            "notes": ["Высота не измерена. Взято среднее значение."],
        }

    height_ratio = h / stats["ref_height"]
    beta_expected = stats["mean"] * (height_ratio ** 1.5) * density
    beta_expected = _clamp(beta_expected, 5.0, stats["max"] * 1.5)

    beta_max_scenario = _clamp(beta_expected * 1.88, beta_expected, stats["max"] * 1.5)

    return {
        "beta_kg_s": round(beta_expected, 2),
        "beta_max_scenario": round(beta_max_scenario, 2),
        "method": "empirical_borisevich_2021",
        "source": "Полевая статистика (Borisevich et al. 2021)",
        "input": {"species": species, "height_m": h, "crown_density_factor": density},
    }


def slenderness_score(height_m, diameter_m):
    if not diameter_m or diameter_m <= 0: 
        return 0.0, 0.0
    S = height_m / diameter_m
    if S >= 100: score = 1.0
    elif S >= 80: score = 0.8
    elif S >= 60: score = 0.5
    elif S >= 40: score = 0.2
    else: score = 0.0
    return S, score


def bending_stress_score(species, height_m, trunk_diameter_m, beta_kg_s, wind_speed):
    if not all([height_m, trunk_diameter_m, beta_kg_s]) or trunk_diameter_m <= 0:
        return 99.0, 0.0, [], 0.0, 0.0, 0.0

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
    
    if safety_factor < 1.0:
        expl.append(f"КРИТИЧЕСКИЙ РИСК ИЗЛОМА! Запас прочности: {safety_factor:.2f}")
        score = 1.0
    elif safety_factor < 1.5:
        expl.append(f"Низкий запас прочности ({safety_factor:.2f}). Риск поломки ствола.")
        score = 0.7
    else:
        expl.append(f"Ствол выдержит ветер. Запас прочности: {safety_factor:.2f}")
        score = 0.1

    return safety_factor, score, expl, force_n, lever_arm_m, bending_moment_nm


def compute_risk(species, height, diameter, lean_angle_deg, beta_info, wind_speed):
    expl = []
    
    lean_score = 0.0
    if lean_angle_deg > 15.0:
        lean_score = 1.0
        expl.append(f"ОПАСНОСТЬ: Аномальный наклон ствола ({lean_angle_deg}°). Риск выкорчевывания.")
    elif lean_angle_deg > 7.0:
        lean_score = 0.5
        expl.append(f"Внимание: Наклон ствола {lean_angle_deg}°.")
    else:
        expl.append(f"Наклон ствола в пределах нормы ({lean_angle_deg}°).")

    S, s_score = slenderness_score(height, diameter)
    if diameter and diameter > 0:
        state = "Опасно (Тонкое)" if S > 80 else "Норма"
        expl.append(f"Стройность H/D: {S:.1f} ({state})")

    beta = beta_info.get("beta_max_scenario") or beta_info.get("beta_kg_s") or 0
    safety_factor, wind_score_val, wind_expl, f_n, l_m, m_nm = bending_stress_score(species, height, diameter, beta, wind_speed)
    expl.extend(wind_expl)

    index = max(lean_score, s_score, wind_score_val) + 0.1 * (lean_score + s_score + wind_score_val)
    index = max(0.0, min(index, 1.0))

    if index < 0.4: cat = "низкий"
    elif index < 0.7: cat = "средний"
    else: cat = "высокий"

    return {
        "index": index,
        "category": cat,
        "explanation": expl,
    }, f_n, l_m, m_nm, safety_factor


# =============================================
# FASTAPI + MODELS
# =============================================

class FeedbackRequest(BaseModel):
    analysis_id: str
    use_for_training: bool
    tree_ok: bool
    stick_ok: bool
    params_ok: bool
    species_ok: bool
    correct_species: str | None = None
    correct_height_m: float | None = None
    correct_crown_width_m: float | None = None
    correct_trunk_diameter_m: float | None = None
    correct_scale_px_to_m: float | None = None
    user_mask_base64: str | None = None

class AdminSetTrainingRequest(BaseModel):
    use_for_training: bool | None = None
    enabled: bool | None = None
    include: bool | None = None
    value: bool | None = None

class AuthRegisterRequest(BaseModel):
    name: str
    email: str
    password: str

class AuthLoginRequest(BaseModel):
    email: str
    password: str

class AuthRoleRequest(BaseModel):
    token: str
    role: str
    admin_code: str | None = None

class AuthGoogleRequest(BaseModel):
    id_token: str
    email: str | None = None
    name: str | None = None
    photo_url: str | None = None


app = FastAPI(title="ArborScan API v2.2")

# =============================================
# SUPABASE AUTH & ANALYSES
# =============================================

AUTH_TOKEN_TTL_DAYS = int(os.getenv("ARBORSCAN_AUTH_TOKEN_TTL_DAYS", "30"))
AUTH_ADMIN_CODE = os.getenv("ARBORSCAN_ADMIN_CODE", "8426")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "946297507051-33c4msb91harv7rqppf2f31qn10n1m2m.apps.googleusercontent.com")

def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def _hash_password(password: str, salt_hex: str | None = None) -> tuple[str, str]:
    if salt_hex is None:
        salt = secrets.token_bytes(16)
        salt_hex = salt.hex()
    else:
        salt = bytes.fromhex(salt_hex)

    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000).hex()
    return digest, salt_hex

def _user_public(row: dict) -> dict:
    data = {"id": row.get("id"), "name": row.get("name"), "email": row.get("email"), "role": row.get("role"), "created_at": row.get("created_at")}
    if "provider" in row: data["provider"] = row["provider"]
    if "avatar_url" in row: data["avatar_url"] = row["avatar_url"]
    return data

def _create_session(user_id: str) -> dict:
    if not SUPABASE_DB_BASE: raise RuntimeError("Supabase is not configured.")
    token = secrets.token_urlsafe(32)
    created_at = _now_iso()
    expires_at = (datetime.utcnow() + timedelta(days=AUTH_TOKEN_TTL_DAYS)).isoformat(timespec="seconds") + "Z"
    requests.post(f"{SUPABASE_DB_BASE}/auth_sessions", headers=_sb_headers(), json={"token": token, "user_id": user_id, "created_at": created_at, "expires_at": expires_at}, timeout=10)
    return {"token": token, "expires_at": expires_at}

def _get_user_by_token(token: str) -> dict | None:
    if not token or not SUPABASE_DB_BASE: return None
    rows = requests.get(f"{SUPABASE_DB_BASE}/auth_sessions?token=eq.{token}&expires_at=gt.{_now_iso()}&select=*,users(*)", headers=_sb_headers(), timeout=10).json()
    return rows[0].get("users") if rows else None

def _email_norm(email: str) -> str:
    return (email or "").strip().lower()

def _validate_auth_payload(name: str | None, email: str, password: str, need_name: bool = False):
    if need_name and (not name or len(name.strip()) < 2): raise HTTPException(status_code=400, detail="Имя должно быть не короче 2 символов")
    if "@" not in email or "." not in email: raise HTTPException(status_code=400, detail="Введите корректную почту")
    if not password or len(password) < 4: raise HTTPException(status_code=400, detail="Пароль должен быть не короче 4 символов")

@app.post("/auth/register")
async def auth_register(payload: AuthRegisterRequest):
    if not SUPABASE_DB_BASE: raise HTTPException(status_code=500, detail="Database disabled")
    name, email, password = payload.name.strip(), _email_norm(payload.email), payload.password
    _validate_auth_payload(name, email, password, need_name=True)

    if requests.get(f"{SUPABASE_DB_BASE}/users?email=eq.{email}", headers=_sb_headers(), timeout=10).json():
        raise HTTPException(status_code=409, detail="Пользователь с такой почтой уже существует")

    password_hash, salt = _hash_password(password)
    user_id, now = str(uuid4()), _now_iso()

    requests.post(f"{SUPABASE_DB_BASE}/users", headers=_sb_headers(), json={"id": user_id, "name": name, "email": email, "password_hash": password_hash, "salt": salt, "role": "user", "created_at": now, "updated_at": now}, timeout=10)
    session = _create_session(user_id)
    user_rows = requests.get(f"{SUPABASE_DB_BASE}/users?id=eq.{user_id}", headers=_sb_headers(), timeout=10).json()
    return {"ok": True, "user": _user_public(user_rows[0]), "token": session["token"], "expires_at": session["expires_at"]}

@app.post("/auth/login")
async def auth_login(payload: AuthLoginRequest):
    if not SUPABASE_DB_BASE: raise HTTPException(status_code=500, detail="Database disabled")
    email, password = _email_norm(payload.email), payload.password
    _validate_auth_payload(None, email, password, need_name=False)

    rows = requests.get(f"{SUPABASE_DB_BASE}/users?email=eq.{email}", headers=_sb_headers(), timeout=10).json()
    if not rows: raise HTTPException(status_code=401, detail="Неверная почта или пароль")

    user = rows[0]
    expected, _ = _hash_password(password, user["salt"])
    if not secrets.compare_digest(expected, user["password_hash"]): raise HTTPException(status_code=401, detail="Неверная почта или пароль")

    session = _create_session(user["id"])
    return {"ok": True, "user": _user_public(user), "token": session["token"], "expires_at": session["expires_at"]}

@app.post("/auth/google")
async def auth_google(payload: AuthGoogleRequest):
    if not SUPABASE_DB_BASE: raise HTTPException(status_code=500, detail="Database disabled")
    if not google_id_token or not GOOGLE_CLIENT_ID: raise HTTPException(status_code=500, detail="Google Auth is not configured")

    try: info = google_id_token.verify_oauth2_token(payload.id_token, google_requests.Request(), GOOGLE_CLIENT_ID)
    except Exception as e: raise HTTPException(status_code=401, detail=f"Google token error: {e}")

    sub, email = str(info.get("sub") or ""), _email_norm(str(info.get("email") or payload.email or ""))
    name = str(info.get("name") or payload.name or email.split("@")[0] or "Google user").strip()
    avatar_url = str(info.get("picture") or payload.photo_url or "")

    if not sub or not email: raise HTTPException(status_code=401, detail="Google didn't return sub/email")

    now = _now_iso()
    rows = requests.get(f"{SUPABASE_DB_BASE}/users?or=(google_sub.eq.{sub},email.eq.{email})", headers=_sb_headers(), timeout=10).json()

    if not rows:
        user_id = str(uuid4())
        password_hash, salt = _hash_password(secrets.token_urlsafe(24))
        requests.post(f"{SUPABASE_DB_BASE}/users", headers=_sb_headers(), json={"id": user_id, "name": name, "email": email, "password_hash": password_hash, "salt": salt, "role": "user", "created_at": now, "updated_at": now, "provider": "google", "google_sub": sub, "avatar_url": avatar_url}, timeout=10)
        user = requests.get(f"{SUPABASE_DB_BASE}/users?id=eq.{user_id}", headers=_sb_headers(), timeout=10).json()[0]
    else:
        user = rows[0]
        resp = requests.patch(f"{SUPABASE_DB_BASE}/users?id=eq.{user['id']}", headers={**_sb_headers(), "Prefer": "return=representation"}, json={"name": name or user.get("name"), "provider": "google", "google_sub": sub or user.get("google_sub"), "avatar_url": avatar_url or user.get("avatar_url"), "updated_at": now}, timeout=10)
        user = resp.json()[0]

    session = _create_session(user["id"])
    return {"ok": True, "user": _user_public(user), "token": session["token"], "expires_at": session["expires_at"]}

@app.get("/auth/me")
async def auth_me(token: str):
    user = _get_user_by_token(token)
    if not user: raise HTTPException(status_code=401, detail="Сессия не найдена или истекла")
    return {"ok": True, "user": _user_public(user)}

@app.post("/auth/set-role")
async def auth_set_role(payload: AuthRoleRequest):
    user = _get_user_by_token(payload.token)
    if not user: raise HTTPException(status_code=401, detail="Сессия не найдена")

    role = (payload.role or "").strip().lower()
    if role not in ("user", "admin"): raise HTTPException(status_code=400, detail="Недопустимая роль")
    if role == "admin" and payload.admin_code != AUTH_ADMIN_CODE: raise HTTPException(status_code=403, detail="Неверный код")

    resp = requests.patch(f"{SUPABASE_DB_BASE}/users?id=eq.{user['id']}", headers={**_sb_headers(), "Prefer": "return=representation"}, json={"role": role, "updated_at": _now_iso()}, timeout=10)
    return {"ok": True, "user": _user_public(resp.json()[0])}


def _save_analysis_record(response: dict, user: dict | None):
    if not SUPABASE_DB_BASE: return
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
        "response_json": response
    }
    requests.post(f"{SUPABASE_DB_BASE}/analyses", headers={**_sb_headers(), "Prefer": "resolution=merge-duplicates"}, json=payload, timeout=10)


def _analysis_summary(row: dict) -> dict:
    return {
        "analysis_id": row.get("id"), "created_at": row.get("created_at"), "species": row.get("species"),
        "risk_index": row.get("risk_index"), "risk_category": row.get("risk_category"),
        "height_m": row.get("height_m"), "crown_width_m": row.get("crown_width_m"), "trunk_diameter_m": row.get("trunk_diameter_m"),
        "beta_kg_s": row.get("beta_kg_s"), "base_moment_nm": row.get("base_moment_nm"), "center_of_load_m": row.get("center_of_load_m"),
        "lat": row.get("lat"), "lon": row.get("lon"), "address": row.get("address"),
    }

@app.get("/analyses/my")
async def analyses_my(token: str, limit: int = 100):
    user = _get_user_by_token(token)
    if not user: raise HTTPException(status_code=401, detail="Сессия не найдена")
    limit = max(1, min(int(limit), 500))
    resp = requests.get(f"{SUPABASE_DB_BASE}/analyses?user_id=eq.{user['id']}&order=created_at.desc&limit={limit}", headers=_sb_headers(), timeout=10)
    return {"ok": True, "items": [_analysis_summary(r) for r in resp.json()]}

@app.get("/analyses/{analysis_id}")
async def analyses_get(analysis_id: str, token: str):
    user = _get_user_by_token(token)
    if not user: raise HTTPException(status_code=401, detail="Сессия не найдена")
    rows = requests.get(f"{SUPABASE_DB_BASE}/analyses?id=eq.{analysis_id}", headers=_sb_headers(), timeout=10).json()
    if not rows: raise HTTPException(status_code=404, detail="Анализ не найден")
    row = rows[0]
    if row.get("user_id") != user["id"] and user.get("role") != "admin": raise HTTPException(status_code=403, detail="Нет доступа")
    return {"ok": True, "analysis": row.get("response_json")}

@app.get("/admin/analyses")
async def admin_analyses(token: str, limit: int = 200):
    user = _get_user_by_token(token)
    if not user or user.get("role") != "admin": raise HTTPException(status_code=403, detail="Нужна роль администратора")
    limit = max(1, min(int(limit), 1000))
    resp = requests.get(f"{SUPABASE_DB_BASE}/analyses?order=created_at.desc&limit={limit}", headers=_sb_headers(), timeout=10)
    return {"ok": True, "items": [_analysis_summary(r) for r in resp.json()]}

@app.get("/profile/stats")
async def profile_stats(token: str):
    user = _get_user_by_token(token)
    if not user: raise HTTPException(status_code=401, detail="Сессия не найдена")
    url = f"{SUPABASE_DB_BASE}/analyses?user_id=eq.{user['id']}&order=created_at.desc"
    rows = requests.get(url, headers=_sb_headers(), timeout=10).json()

    total = len(rows)
    with_geo = sum(1 for r in rows if r.get("lat") is not None and r.get("lon") is not None)
    high_risk = sum(1 for r in rows if r.get("risk_category") == "высокий")
    risks = [r.get("risk_index") for r in rows if isinstance(r.get("risk_index"), (int, float))]
    
    return {
        "ok": True, "user": _user_public(user),
        "stats": {
            "total_analyses": total, "with_geo": with_geo, "high_risk_count": high_risk,
            "avg_risk": sum(risks) / len(risks) if risks else None,
            "last_analysis": _analysis_summary(rows[0]) if rows else None
        }
    }


TRAINING_EVENTS = deque(maxlen=int(os.getenv("TRAINING_EVENTS_MAXLEN", "200")))

def log_training_event(level: str, message: str, data: dict | None = None) -> None:
    try:
        evt = {"ts": datetime.utcnow().replace(microsecond=0).isoformat() + "Z", "level": level.upper(), "message": message, "data": data or {}}
        TRAINING_EVENTS.append(evt)
        print(f"[TRAINING_EVENT] {evt['ts']} {evt['level']}: {evt['message']} {evt['data']}")
    except Exception: pass


@app.on_event("startup")
def _startup_load_models():
    try:
        training_state_ensure_row()
        with MODEL_LOCK: reload_tree_model(force=True)
        
        global REMBG_SESSION
        print("[*] Loading rembg (U-2-Net) model for ultra-sharp masks...")
        REMBG_SESSION = new_session("u2net")
        try:
            dummy = np.zeros((10, 10, 3), dtype=np.uint8)
            remove(dummy, session=REMBG_SESSION, only_mask=True)
            print("[*] rembg loaded and warmed up.")
        except Exception as e:
            print(f"[!] rembg warmup failed: {e}")

    except Exception as e:
        print(f"[!] Startup models load failed: {e}")


def normalize_address_ru(address: str | None) -> str | None:
    if not address: return address
    replacements = {"Інтэрнат": "Интернат", "вуліца": "улица", "вул.": "ул.", "Машынабудаўнікоў": "Машиностроителей", "Аўтазаводскі": "Автозаводский", "раён": "район", "Пасёлак": "Посёлок", "Заводскі": "Заводской", "Мінск": "Минск", "Беларусь": "Беларусь"}
    out = address
    for src, dst in replacements.items(): out = out.replace(src, dst)
    return out


# === СИНХРОННЫЕ ФУНКЦИИ ДЛЯ РАБОТЫ В ПУЛЕ ПОТОКОВ ===
def _run_yolo_sync(img_array):
    tree_model_local = get_tree_model()
    t_res = tree_model_local(img_array, imgsz=1024, retina_masks=True, conf=0.25)[0]
    s_res = stick_model(img_array)[0]
    return t_res, s_res

def map_plantnet_name(raw_name: str) -> str:
    name_lower = raw_name.lower()
    if "сосна" in name_lower: return "Сосна"
    if "ель" in name_lower: return "Ель"
    if "дуб" in name_lower: return "Дуб"
    if "береза" in name_lower or "берёза" in name_lower: return "Береза"
    if "тополь" in name_lower: return "Тополь"
    if "клен" in name_lower or "клён" in name_lower: return "Клен"
    if "ясень" in name_lower: return "Ясень"
    if "липа" in name_lower: return "Липа"
    return raw_name.capitalize()

def _run_classifier_sync(crop_bgr):
    ok, encoded_img = cv2.imencode(".jpg", crop_bgr)
    if not ok: return "Неизвестно"
    
    url = f"https://my-api.plantnet.org/v2/identify/all?api-key={PLANTNET_API_KEY}&lang=ru"
    files = [('images', ('crop.jpg', encoded_img.tobytes(), 'image/jpeg'))]
    data = {'organs': ['habit']}
    
    try:
        r = requests.post(url, files=files, data=data, timeout=12)
        if r.status_code == 200:
            results = r.json().get('results', [])
            if results:
                species = results[0].get('species', {})
                common = species.get('commonNames', [])
                if common: return map_plantnet_name(common[0])
                return map_plantnet_name(species.get('scientificNameWithoutAuthor', 'Неизвестно'))
    except Exception as e: print(f"[!] Pl@ntNet API error: {e}")
    return "Неизвестно"


@app.post("/analyze-tree")
async def analyze_tree(
    file: UploadFile = File(...),
    ar_height_m: Optional[float] = Form(None),
    ar_crown_width_m: Optional[float] = Form(None),
    ar_trunk_diameter_m: Optional[float] = Form(None),
    manual_scale: Optional[float] = Form(None), # НОВОЕ ПОЛЕ ДЛЯ ПРЕДМЕТОВ В КАДРЕ
    manual_beta_kg_s: Optional[float] = Form(None),
    crown_density_factor: Optional[float] = Form(None),
    manual_wind_speed_m_s: Optional[float] = Form(None),
    manual_wind_gust_m_s: Optional[float] = Form(None),
    auth_token: Optional[str] = Form(None),
    lat: Optional[float] = Form(None),
    lon: Optional[float] = Form(None),
):
    analysis_user = _get_user_by_token(auth_token) if auth_token else None

    image_bytes = await file.read()
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None: raise HTTPException(status_code=400, detail="Не удалось прочитать изображение")
    H, W = img.shape[:2]

    # 1. YOLO
    tree_res, stick_res = await run_in_threadpool(_run_yolo_sync, img)
    if tree_res.masks is None: return JSONResponse({"error": "Дерево не найдено"}, status_code=400)

    # 2. СНАЙПЕРСКИЙ ФОКУС
    masks, distances_to_center = [], []
    img_center_x, img_center_y = W / 2.0, H / 2.0

    for i, m in enumerate(tree_res.masks.data):
        mask = cv2.resize((m.cpu().numpy() > 0.5).astype(np.uint8) * 255, (W, H), interpolation=cv2.INTER_NEAREST)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        x1, y1, x2, y2 = tree_res.boxes.xyxy[i].cpu().numpy()
        distances_to_center.append(((x1 + x2)/2.0 - img_center_x)**2 + ((y1 + y2)/2.0 - img_center_y)**2)
        masks.append(mask)

    idx = int(np.argmin(distances_to_center))
    yolo_mask = masks[idx]

    # 3. ИДЕАЛЬНАЯ МАСКА (REMBG)
    try:
        x1, y1, x2, y2 = map(int, tree_res.boxes.xyxy[idx].cpu().numpy())
        margin = 30
        x1_c, y1_c = max(0, x1 - margin), max(0, y1 - margin)
        x2_c, y2_c = min(W, x2 + margin), min(H, y2 + margin)
        
        crop_bgr = img[y1_c:y2_c, x1_c:x2_c]
        
        def _refine_mask_sync(crop):
            return remove(crop, session=REMBG_SESSION, only_mask=True)
            
        refined_crop_mask = await run_in_threadpool(_refine_mask_sync, crop_bgr)
        if len(refined_crop_mask.shape) == 3: refined_crop_mask = refined_crop_mask[:, :, 0]
        _, refined_crop_mask = cv2.threshold(refined_crop_mask, 127, 255, cv2.THRESH_BINARY)
        
        final_mask = np.zeros((H, W), dtype=np.uint8)
        final_mask[y1_c:y2_c, x1_c:x2_c] = refined_crop_mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)
    except Exception:
        mask = yolo_mask

    # 4. ИЗВЛЕЧЕНИЕ ПИКСЕЛЬНЫХ РАЗМЕРОВ
    ys, xs = np.where(mask > 0)
    if len(ys) == 0: return JSONResponse({"error": "Ошибка обработки контура дерева"}, status_code=400)
    y_min, y_max = ys.min(), ys.max()
    height_px = y_max - y_min

    crown_width_px = 0
    for y in range(y_min, y_min + int(0.7 * height_px)):
        row = np.where(mask[y] > 0)[0]
        if len(row) > 0: crown_width_px = max(crown_width_px, row.max() - row.min())

    trunk_vals = [np.where(mask[y] > 0)[0].max() - np.where(mask[y] > 0)[0].min() for y in range(y_max - int(0.2 * height_px), y_max) if len(np.where(mask[y] > 0)[0]) > 0]
    trunk_px = np.mean(trunk_vals) if trunk_vals else 0

    # 5. КЛАССИФИКАЦИЯ (PLANTNET)
    species_name = await run_in_threadpool(_run_classifier_sync, img[y1:y2, x1:x2])

    # 6. МАСШТАБ И РАЗМЕРЫ (СВЯТОЙ ГРААЛЬ: Каскадный поиск масштаба)
    scale = None
    dimensions_source = "Неизвестно"

    if manual_scale and float(manual_scale) > 0:
        scale = float(manual_scale)
        dimensions_source = "Пользовательский маркер (Линия на фото)"
    
    if not scale and len(stick_res.boxes) > 0:
        best = max(stick_res.boxes, key=lambda b: b.xyxy[0][3] - b.xyxy[0][1])
        stick_h = best.xyxy[0][3].cpu().item() - best.xyxy[0][1].cpu().item()
        if stick_h > 10:
            scale = REAL_STICK_M / stick_h
            dimensions_source = "Авто-маркер (AI)"
    
    if not scale:
        if ar_height_m and height_px > 0:
            scale = float(ar_height_m) / height_px
            dimensions_source = "Пропорционально (по AR Высоте)"
        elif ar_crown_width_m and crown_width_px > 0:
            scale = float(ar_crown_width_m) / crown_width_px
            dimensions_source = "Пропорционально (по AR Кроне)"
        elif ar_trunk_diameter_m and trunk_px > 0:
            scale = float(ar_trunk_diameter_m) / trunk_px
            dimensions_source = "Пропорционально (по AR Стволу)"

    if not scale:
        ref_h = BETA_EMPIRICAL_STATS.get(species_name, BETA_EMPIRICAL_STATS["Сосна"])["ref_height"]
        if height_px > 0:
            scale = ref_h / height_px
            dimensions_source = f"Статистика ({species_name} ≈ {ref_h}м)"

    height_m = round(height_px * scale, 2) if scale else None
    crown_m = round(crown_width_px * scale, 2) if scale else None
    trunk_m = round(trunk_px * scale, 2) if scale and trunk_px else None

    # ПЕРЕЗАПИСЬ (Если мы знаем точные AR-значения, они приоритетнее расчетов ИИ)
    if ar_height_m: height_m = round(float(ar_height_m), 2)
    if ar_crown_width_m: crown_m = round(float(ar_crown_width_m), 2)
    if ar_trunk_diameter_m: trunk_m = round(float(ar_trunk_diameter_m), 2)

    if ar_height_m or ar_crown_width_m or ar_trunk_diameter_m:
        if not manual_scale: 
            dimensions_source = "AR-клинометр"

    ar_measurements = {"height_m": height_m if ar_height_m else None, "crown_width_m": crown_m if ar_crown_width_m else None, "trunk_diameter_m": trunk_m if ar_trunk_diameter_m else None}
    measurement_sources = {"height_m": "ar" if ar_height_m else "image", "crown_width_m": "ar" if ar_crown_width_m else "image", "trunk_diameter_m": "ar" if ar_trunk_diameter_m else "image"}

    # 7. AI: ПОРИСТОСТЬ КРОНЫ И УГОЛ НАКЛОНА
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

    annotated_b64 = draw_mask(img.copy(), mask)

    # 8. РАСЧЕТ РИСКОВ (SIA)
    gps, address = None, None
    if ENABLE_ENV_ANALYSIS:
        gps = {"lat": float(lat), "lon": float(lon)} if (lat and lon) else extract_gps(image_bytes)
        if gps: address = normalize_address_ru(reverse_geocode(gps["lat"], gps["lon"]))

    final_crown_density = crown_density_factor if crown_density_factor else crown_density_ai
    beta_info = estimate_beta_kg_s(species_name, height_m, manual_beta_kg_s=manual_beta_kg_s, crown_density_factor=final_crown_density)
    
    wind_design = float(manual_wind_gust_m_s or manual_wind_speed_m_s or 25.0)
    risk, f_n, l_m, m_nm, s_f = compute_risk(species_name, height_m, trunk_m, lean_angle_deg, beta_info, wind_design)

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
        "gps": gps, "address": address, "risk": risk, "model_versions": MODEL_VERSIONS, "build": BUILD_INFO, "schema_version": SCHEMA_VERSION, "api_version": API_VERSION,
    }

    tree_pred = {"box_xyxy": tree_res.boxes.xyxy[idx].cpu().numpy().tolist(), "confidence": float(tree_res.boxes.conf[idx].cpu().item()) if len(tree_res.boxes.conf)>idx else None, "class_id": int(tree_res.boxes.cls[idx].cpu().item()) if len(tree_res.boxes.cls)>idx else None}
    stick_pred = {"box_xyxy": None, "scale_px_to_m": scale}
    if len(stick_res.boxes) > 0:
        best = max(stick_res.boxes, key=lambda b: b.xyxy[0][3] - b.xyxy[0][1])
        stick_pred.update({"box_xyxy": [int(x) for x in best.xyxy[0].cpu().numpy()], "confidence": float(best.conf[0].cpu().item())})

    try:
        supabase_upload_bytes(SUPABASE_BUCKET_RAW, f"{analysis_id}/input.jpg", image_bytes)
        supabase_upload_json(SUPABASE_BUCKET_RAW, f"{analysis_id}/meta_auto.json", meta)
        try: supabase_upload_bytes(SUPABASE_BUCKET_RAW, f"{analysis_id}/annotated.jpg", base64.b64decode(annotated_b64))
        except Exception: pass
        try:
            supabase_upload_json(SUPABASE_BUCKET_RAW, f"{analysis_id}/tree_pred.json", tree_pred)
            supabase_upload_json(SUPABASE_BUCKET_RAW, f"{analysis_id}/stick_pred.json", stick_pred)
        except Exception: pass
    except Exception as e: print(f"[!] Failed to upload raw sample: {e}")

    try:
        tmp_dir = Path("/tmp") / analysis_id
        tmp_dir.mkdir(parents=True, exist_ok=True)
        (tmp_dir / "input.jpg").write_bytes(image_bytes)
        try: (tmp_dir / "annotated.jpg").write_bytes(base64.b64decode(annotated_b64))
        except Exception: pass
        (tmp_dir / "tree_pred.json").write_text(json.dumps(tree_pred, ensure_ascii=False))
        (tmp_dir / "stick_pred.json").write_text(json.dumps(stick_pred, ensure_ascii=False))
        (tmp_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False))
    except Exception as e: print(f"[!] Failed to cache in /tmp: {e}")

    response = {
        "analysis_id": analysis_id, "species": species_name, "height_m": height_m, "crown_width_m": crown_m, "trunk_diameter_m": trunk_m,
        "height_m_ai": height_m_ai, "crown_width_m_ai": crown_m_ai, "trunk_diameter_m_ai": trunk_m_ai, "ar_measurements": ar_measurements,
        "measurement_sources": measurement_sources, "dimensions_source": dimensions_source, "beta": beta_info, "analytic_wind_model": analytic_wind_model,
        "scale_px_to_m": scale, "annotated_image_base64": annotated_b64, "gps": gps, "address": address, "risk": risk,
    }
    try: response["original_image_base64"] = encode_jpeg_base64(img.copy(), max_side=1280, quality=72)
    except Exception: response["original_image_base64"] = None

    response["server_saved"] = analysis_user is not None
    try: _save_analysis_record(response, analysis_user)
    except Exception as e: print(f"[!] Failed to save DB record: {e}")

    return JSONResponse(response)


@app.post("/feedback")
@app.post("/api/feedback")
def send_feedback(payload: dict = Body(...)):
    analysis_id = payload.get('analysis_id') or payload.get('analysisId')
    if not analysis_id: raise HTTPException(status_code=422, detail='analysis_id is required')

    def _b(val, default=True):
        if val is None: return default
        if isinstance(val, bool): return val
        if isinstance(val, (int, float)): return bool(val)
        if isinstance(val, str):
            v = val.strip().lower()
            if v in ('1','true','yes','y','ok'): return True
            if v in ('0','false','no','n'): return False
        return default

    use_for_training = _b(payload.get('use_for_training', payload.get('useForTraining')), default=True)
    tree_ok = _b(payload.get('tree_ok', payload.get('treeOk')), default=True)
    stick_ok = _b(payload.get('stick_ok', payload.get('stickOk')), default=True)
    params_ok = _b(payload.get('params_ok', payload.get('paramsOk')), default=True)
    species_ok = _b(payload.get('species_ok', payload.get('speciesOk')), default=True)
    correct_species = payload.get('correct_species') or payload.get('correctSpecies')

    def _f(val):
        if val is None: return None
        if isinstance(val, (int, float)): return float(val)
        if isinstance(val, str):
            s = val.strip().replace(',', '.')
            if not s or s.lower() in ('null', 'none', 'nan'): return None
            try: return float(s)
            except Exception: return None
        return None

    corrected_height_m = _f(payload.get('corrected_height_m') or payload.get('correctedHeightM'))
    corrected_crown_width_m = _f(payload.get('corrected_crown_width_m') or payload.get('correctedCrownWidthM'))
    corrected_trunk_diameter_m = _f(payload.get('corrected_trunk_diameter_m') or payload.get('correctedTrunkDiameterM'))
    corrected_scale_px_to_m = _f(payload.get('corrected_scale_px_to_m') or payload.get('scale_px_to_m_corrected'))
    user_mask_base64 = payload.get('user_mask_base64') or payload.get('mask_base64')

    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY: raise HTTPException(status_code=500, detail="Supabase не настроен")
    tmp_dir = Path("/tmp") / analysis_id
    if not tmp_dir.exists(): raise HTTPException(status_code=404, detail="analysis_id не найден")

    if not use_for_training:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return {"status": "ignored", "reason": "user_disabled_training"}

    try: meta = json.loads((tmp_dir / "meta.json").read_text(encoding="utf-8"))
    except Exception as e: raise HTTPException(status_code=500, detail=f"Ошибка чтения meta.json: {e}")

    meta.update({"tree_ok": tree_ok, "stick_ok": stick_ok, "params_ok": params_ok, "species_ok": species_ok, "correct_species": correct_species})
    if not species_ok and correct_species: meta["species"] = correct_species
    if corrected_height_m is not None: meta["height_m"] = corrected_height_m
    if corrected_crown_width_m is not None: meta["crown_width_m"] = corrected_crown_width_m
    if corrected_trunk_diameter_m is not None: meta["trunk_diameter_m"] = corrected_trunk_diameter_m
    if corrected_scale_px_to_m is not None: meta["scale_px_to_m"] = corrected_scale_px_to_m

    trust = sum([0.3 if tree_ok else 0, 0.2 if stick_ok else 0, 0.2 if params_ok else 0, 0.3 if (species_ok or correct_species) else 0])
    meta["trust_score"] = trust
    is_verified = (use_for_training and trust >= VERIFIED_TRUST_THRESHOLD)

    try:
        if (tmp_dir / "input.jpg").exists(): supabase_upload_bytes(SUPABASE_BUCKET_INPUTS, f"{analysis_id}/input.jpg", (tmp_dir / "input.jpg").read_bytes())
        if (tmp_dir / "annotated.jpg").exists(): supabase_upload_bytes(SUPABASE_BUCKET_INPUTS, f"{analysis_id}/annotated.jpg", (tmp_dir / "annotated.jpg").read_bytes())

        existing_has_user_mask = False
        try: existing_has_user_mask = bool(json.loads(supabase_download_bytes(SUPABASE_BUCKET_VERIFIED, f"{analysis_id}/meta_verified.json")).get("has_user_mask", False))
        except Exception: pass
        meta["has_user_mask"] = existing_has_user_mask

        if user_mask_base64 and str(user_mask_base64).strip().lower() not in ("null", "undefined"):
            try:
                supabase_upload_bytes(SUPABASE_BUCKET_INPUTS, f"{analysis_id}/user_mask.png", ensure_png_mask_bytes(str(user_mask_base64)))
                meta["has_user_mask"] = True
            except Exception as e: print(f"[!] User mask error: {e}")

        if (tmp_dir / "tree_pred.json").exists(): supabase_upload_bytes(SUPABASE_BUCKET_PRED, f"{analysis_id}/tree_pred.json", (tmp_dir / "tree_pred.json").read_bytes())
        if (tmp_dir / "stick_pred.json").exists(): supabase_upload_bytes(SUPABASE_BUCKET_PRED, f"{analysis_id}/stick_pred.json", (tmp_dir / "stick_pred.json").read_bytes())
        supabase_upload_json(SUPABASE_BUCKET_META, f"{analysis_id}.json", meta)

        if is_verified:
            try:
                supabase_upload_bytes(SUPABASE_BUCKET_VERIFIED, f"{analysis_id}/input.jpg", (tmp_dir / "input.jpg").read_bytes())
                if (tmp_dir / "annotated.jpg").exists(): supabase_upload_bytes(SUPABASE_BUCKET_VERIFIED, f"{analysis_id}/annotated.jpg", (tmp_dir / "annotated.jpg").read_bytes())
                if user_mask_base64:
                    try: supabase_upload_bytes(SUPABASE_BUCKET_VERIFIED, f"{analysis_id}/user_mask.png", ensure_png_mask_bytes(user_mask_base64))
                    except Exception as e: print(f"[!] Verified mask error: {e}")
                supabase_upload_bytes(SUPABASE_BUCKET_VERIFIED, f"{analysis_id}/tree_pred.json", (tmp_dir / "tree_pred.json").read_bytes())
                supabase_upload_bytes(SUPABASE_BUCKET_VERIFIED, f"{analysis_id}/stick_pred.json", (tmp_dir / "stick_pred.json").read_bytes())
                
                meta_verified = meta.copy()
                meta_verified["verified"] = True
                meta_verified["verified_at"] = datetime.utcnow().isoformat()
                meta_verified["verifier_role"] = "admin" if not use_for_training else "user"
                supabase_upload_json(SUPABASE_BUCKET_VERIFIED, f"{analysis_id}/meta_verified.json", meta_verified)
            except Exception as e: print(f"[!] Verified upload error: {e}")

    except Exception as e: raise HTTPException(status_code=500, detail=f"Ошибка Supabase: {e}")

    if SUPABASE_ENABLE_QUEUE:
        try: supabase_db_insert(SUPABASE_QUEUE_TABLE, {"analysis_id": analysis_id, "trust_score": trust, "species": meta.get("species"), "has_user_mask": meta.get("has_user_mask", False), "tree_ok": meta.get("tree_ok"), "stick_ok": meta.get("stick_ok"), "params_ok": meta.get("params_ok"), "species_ok": meta.get("species_ok")})
        except Exception as e: print(f"[!] Queue error: {e}")

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return {"status": "ok", "analysis_id": analysis_id, "trust_score": trust}


@app.get("/admin/verified-list")
def admin_verified_list(include_used: bool = False):
    try: objects = supabase_list_objects(SUPABASE_BUCKET_VERIFIED)
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

    analysis_ids = sorted({obj["name"].split("/")[0] for obj in objects})
    results = []
    for aid in analysis_ids:
        try:
            meta = json.loads(supabase_download_bytes(SUPABASE_BUCKET_VERIFIED, f"{aid}/meta_verified.json"))
            if (not include_used) and meta.get("used_for_training") is True: continue
            results.append({"analysis_id": aid, "species": meta.get("species"), "risk_category": meta.get("risk", {}).get("category"), "trust_score": meta.get("trust_score"), "verified": meta.get("verified", True), "verified_at": meta.get("verified_at"), "exclude_from_training": meta.get("exclude_from_training", False) == True, "has_user_mask": meta.get("has_user_mask", False) == True, "used_for_training": meta.get("used_for_training", False) == True})
        except Exception: continue
    return {"count": len(results), "items": results}


@app.post("/admin/verified/{analysis_id}/set-training")
def admin_set_training_flag(analysis_id: str, req: AdminSetTrainingRequest):
    flag = next((v for v in (req.use_for_training, req.enabled, req.include, req.value) if v is not None), None)
    if flag is None: raise HTTPException(status_code=400, detail="Missing boolean flag")

    def _load_json(path):
        try: return json.loads(supabase_download_bytes(SUPABASE_BUCKET_VERIFIED, path).decode("utf-8"))
        except Exception: return {}

    for path in [f"{analysis_id}/meta.json", f"{analysis_id}/meta_verified.json"]:
        data = _load_json(path)
        data.update({"analysis_id": analysis_id, "use_for_training": flag, "exclude_from_training": not flag})
        supabase_upload_json(SUPABASE_BUCKET_VERIFIED, path, data)
    return {"analysis_id": analysis_id, "use_for_training": flag, "exclude_from_training": not flag}


@app.get("/admin/analysis/{analysis_id}")
def admin_get_analysis(analysis_id: str):
    try:
        input_img = supabase_download_bytes(SUPABASE_BUCKET_VERIFIED, f"{analysis_id}/input.jpg")
        annotated_img = supabase_download_bytes(SUPABASE_BUCKET_VERIFIED, f"{analysis_id}/annotated.jpg")
        try: user_mask_img = supabase_download_bytes(SUPABASE_BUCKET_VERIFIED, f"{analysis_id}/user_mask.png")
        except Exception: user_mask_img = None
        tree_pred = json.loads(supabase_download_bytes(SUPABASE_BUCKET_VERIFIED, f"{analysis_id}/tree_pred.json"))
        stick_pred = json.loads(supabase_download_bytes(SUPABASE_BUCKET_VERIFIED, f"{analysis_id}/stick_pred.json"))
        meta = json.loads(supabase_download_bytes(SUPABASE_BUCKET_VERIFIED, f"{analysis_id}/meta_verified.json"))
    except Exception as e: raise HTTPException(status_code=404, detail=f"Analysis not found: {e}")

    return {
        "analysis_id": analysis_id,
        "images": {"input_base64": base64.b64encode(input_img).decode("utf-8"), "annotated_base64": base64.b64encode(annotated_img).decode("utf-8"), "user_mask_base64": base64.b64encode(user_mask_img).decode("utf-8") if user_mask_img else None},
        "tree_pred": tree_pred, "stick_pred": stick_pred, "meta": meta,
    }


@app.get("/admin/training-status")
def admin_training_status():
    training_state_ensure_row()
    state = training_state_get()
    return {"active_model_version": state.get("active_model_version", 0), "last_model_version": state.get("last_model_version", 0), "training_in_progress": state.get("training_in_progress", False), "retrain_requested": state.get("retrain_requested", False)}

@app.post("/admin/set-active-model")
async def admin_set_active_model(payload: dict = Body(...)):
    training_state_ensure_row()
    v = int(payload.get('version') or payload.get('model_version') or payload.get('active_model_version') or 0)
    try: supabase_download_bytes(SUPABASE_BUCKET_MODELS, f"model_v{v}.pt")
    except Exception as e: raise HTTPException(status_code=400, detail=f"Model not found: {e}")
    training_state_update({"active_model_version": v})
    with MODEL_LOCK: reload_tree_model(force=True)
    return {"status": "ok", "active_model_version": v}

@app.post("/admin/request-retrain")
def admin_request_retrain():
    training_state_ensure_row()
    training_state_update({"retrain_requested": True})
    return {"status": "ok", "retrain_requested": True}

@app.get("/admin/models")
def admin_models():
    return {"models": list_available_model_versions(), "active_model_version": _get_active_model_version()}

@app.get("/admin/training-events")
def admin_training_events(limit: int = 15):
    return {"events": list(reversed(list(TRAINING_EVENTS)[-max(1, min(int(limit), 200)):]))}