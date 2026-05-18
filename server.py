import os
import io
import base64
import json
import re
import shutil
import time
import threading
import cv2
import numpy as np
import requests
from ultralytics import YOLO
from PIL import Image, ExifTags
import torch
from torchvision import models, transforms
from fastapi import FastAPI, File, UploadFile, HTTPException, Body, Form, Form
from fastapi.responses import JSONResponse
from uuid import uuid4
from pathlib import Path
from pydantic import BaseModel
from datetime import datetime
from collections import deque
from typing import Optional, Dict, Any, List, Tuple
from tree_dynamics import compute_analytic_wind_model

# -------------------------------------
# CONFIG
# -------------------------------------

# Supabase config: URL и SERVICE KEY задаём через переменные окружения на Railway
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    print("[!] Warning: SUPABASE_URL or SUPABASE_SERVICE_KEY not set. /feedback will not upload to Supabase.")

# Buckets в Supabase Storage
SUPABASE_BUCKET_INPUTS = "arborscan-inputs"
SUPABASE_BUCKET_PRED = "arborscan-predictions"
SUPABASE_BUCKET_META = "arborscan-meta"
SUPABASE_BUCKET_VERIFIED = "arborscan-verified"

# NEW: bucket для сохранения всех загрузок (raw dataset)
SUPABASE_BUCKET_RAW = "arborscan-raw"

# Bucket for versioned segmentation models (model_vN.pt)
SUPABASE_BUCKET_MODELS = os.getenv("SUPABASE_BUCKET_MODELS", "arborscan-models")

# Таблица в Supabase Postgres для очереди доверенных примеров
# (создаёшь её сам в Supabase SQL, напр. arborscan_feedback_queue)
SUPABASE_DB_BASE = SUPABASE_URL.rstrip("/") + "/rest/v1" if SUPABASE_URL else None
SUPABASE_QUEUE_TABLE = "arborscan_feedback_queue"
# Включать ли вставку в Postgres-очередь. По умолчанию выключено,
# чтобы отсутствие таблицы не ломало пайплайн обучения.
SUPABASE_ENABLE_QUEUE = os.getenv("SUPABASE_ENABLE_QUEUE", "false").lower() == "true"

# ---------------------------------------------------------
# Supabase PostgREST helpers (training_state)
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
        raise RuntimeError("Supabase DB is not configured (SUPABASE_URL missing)")
    url = f"{SUPABASE_DB_BASE}/training_state?id=eq.1&select=*"
    resp = requests.get(url, headers=_sb_headers(json_ct=False), timeout=30)
    if resp.status_code >= 400:
        raise RuntimeError(f"training_state_get error {resp.status_code}: {resp.text}")
    rows = resp.json()
    if not rows:
        return {}
    return rows[0]


def training_state_ensure_row():
    # Ensure row id=1 exists
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
    resp = requests.post(
        url,
        headers={**_sb_headers(), "Prefer": "return=representation"},
        data=json.dumps(payload),
        timeout=30,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"training_state_ensure_row error {resp.status_code}: {resp.text}")


def training_state_update(fields: dict) -> dict:
    if not SUPABASE_DB_BASE:
        raise RuntimeError("Supabase DB is not configured (SUPABASE_URL missing)")
    url = f"{SUPABASE_DB_BASE}/training_state?id=eq.1"
    resp = requests.patch(
        url,
        headers={**_sb_headers(), "Prefer": "return=representation"},
        data=json.dumps(fields),
        timeout=30,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"training_state_update error {resp.status_code}: {resp.text}")
    rows = resp.json()
    return rows[0] if rows else fields


# Старые настройки окружения (оставляю как есть, чтобы ничего не сломать)
WEATHER_API_KEY = (os.getenv("WEATHER_API_KEY")
                 or os.getenv("OPENWEATHER_API_KEY")
                 or os.getenv("OPENWEATHERMAP_API_KEY")
                 or os.getenv("dc825ffd002731568ec7766eafb54bc9")
                 or None)
WEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

SOILGRIDS_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"

NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
NOMINATIM_USER_AGENT = os.getenv(
    "NOMINATIM_USER_AGENT",
    "arborscan-backend/1.0 (contact: example@mail.com)"
)

ENABLE_ENV_ANALYSIS = os.getenv("ENABLE_ENV_ANALYSIS", "true").lower() == "true"


# -------------------------------------
# MODEL VERSIONS
# -------------------------------------

MODEL_VERSIONS = {
    "tree_yolo": "tree_yolov8_seg_v1.2.0",
    "stick_yolo": "stick_yolov8_det_v1.0.3",
    "classifier": "resnet18_species_v0.9.1",
}
BUILD_INFO = {
    # желательно прокидывать из CI / Railway
    "git_commit": os.getenv("GIT_COMMIT", "unknown"),
    "build_time": os.getenv("BUILD_TIME")
}
SCHEMA_VERSION = "1.0.0"
API_VERSION = "2.0.0"
VERIFIED_TRUST_THRESHOLD = 0.0

# -------------------------------------
# CLASSES / CONSTANTS
# -------------------------------------

CLASS_NAMES_RU = ["Береза", "Дуб", "Ель", "Сосна", "Тополь"]
REAL_STICK_M = 1.0

# -------------------------------------
# LOADING MODELS
# -------------------------------------

print("[*] Loading YOLO models...")
tree_model = None  # loaded dynamically from Supabase models bucket via active_model_version
stick_model = YOLO("models/stick_model.pt")

print("[*] Loading classifier...")
classifier = models.resnet18(weights=None)
classifier.fc = torch.nn.Linear(classifier.fc.in_features, 5)
classifier.load_state_dict(torch.load("models/classifier.pth", map_location="cpu"))
classifier.eval()

transformer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

print("[*] Models loaded.")

# =============================================
# SUPABASE UTILS (Storage + DB)
# =============================================

def supabase_upload_bytes(bucket: str, path: str, data: bytes):
    """
    Загрузка бинарных данных в Supabase Storage через REST API.
    """
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise RuntimeError("Supabase is not configured (no URL or SERVICE_KEY)")

    url = SUPABASE_URL.rstrip("/") + f"/storage/v1/object/{bucket}/{path}"
    headers = {
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/octet-stream",
        "x-upsert": "true",
    }
    resp = requests.post(url, headers=headers, data=data, timeout=30)
    if resp.status_code >= 400:
        raise RuntimeError(f"Supabase upload error {resp.status_code}: {resp.text}")


def supabase_upload_json(bucket: str, path: str, obj: dict):
    data = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
    supabase_upload_bytes(bucket, path, data)

def supabase_list_objects(bucket: str, prefix: str = ""):
    """
    Вернуть список объектов в Supabase Storage (метаданные).
    """
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise RuntimeError("Supabase is not configured (no URL or SERVICE_KEY)")

    url = SUPABASE_URL.rstrip("/") + f"/storage/v1/object/list/{bucket}"
    headers = {
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "prefix": prefix,
        "limit": 200,
        "offset": 0,
        "sortBy": {"column": "name", "order": "desc"},
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=15)
    if resp.status_code >= 400:
        raise RuntimeError(f"Supabase list error {resp.status_code}: {resp.text}")
    return resp.json()


def supabase_download_bytes(bucket: str, path: str) -> bytes:
    """
    Скачать файл из Supabase Storage.

    Для приватных bucket'ов используем authenticated endpoint.
    """
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise RuntimeError("Supabase is not configured (no URL or SERVICE_KEY)")

    url = SUPABASE_URL.rstrip("/") + f"/storage/v1/object/authenticated/{bucket}/{path}"
    headers = {
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "apikey": SUPABASE_SERVICE_KEY,
    }
    resp = requests.get(url, headers=headers, timeout=60)
    if resp.status_code >= 400:
        raise RuntimeError(f"Supabase download error {resp.status_code}: {resp.text}")
    return resp.content



# ---------------------------------------------------------
# Model hot-swap (tree model) using training_state.active_model_version
# Models are stored in Supabase Storage bucket: SUPABASE_BUCKET_MODELS as model_v{N}.pt
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
    if v is None:
        # default to 0 if not set
        return 0
    return int(v)

def list_available_model_versions() -> list[dict]:
    """Return model versions for the Admin Panel dropdown.

    Источник версий:
      1) Supabase Storage bucket SUPABASE_BUCKET_MODELS (model_v*.pt) — основной источник
      2) AVAILABLE_MODEL_VERSIONS env var (comma-separated ints) — опционально
      3) локальный кеш /tmp/models (model_v*.pt)
      4) bundled ./models (model_v*.pt)
    Всегда включает текущую active_model_version.
    """
    versions: set[int] = set()

    # 1) Storage listing
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

    # 2) Optional env hint
    env_hint = os.getenv("AVAILABLE_MODEL_VERSIONS", "").strip()
    if env_hint:
        for part in env_hint.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                versions.add(int(part))
            except ValueError:
                pass

    # 3) Local cache
    for p in Path("/tmp/models").glob("model_v*.pt"):
        mm = re.search(r"model_v(\d+)\.pt$", p.name)
        if mm:
            versions.add(int(mm.group(1)))

    # 4) Bundled models dir
    if Path("models").exists():
        for p in Path("models").glob("model_v*.pt"):
            mm = re.search(r"model_v(\d+)\.pt$", p.name)
            if mm:
                versions.add(int(mm.group(1)))

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

    # version 0: fallback to bundled local model if exists
    if v == 0:
        local_fallback = "models/tree_model.pt"
        if os.path.exists(local_fallback):
            print(f"[*] Using bundled tree model: {local_fallback}")
            TREE_MODEL = YOLO(local_fallback)
            TREE_MODEL_VERSION = 0
            return
        # else try download v0 if present
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
        if TREE_MODEL is None:
            raise RuntimeError("TREE_MODEL is not loaded")
        return TREE_MODEL

def supabase_db_insert(table: str, row: dict):
    """
    Вставка записи в Supabase Postgres через REST (PostgREST).
    Используется для очереди доверенных примеров.
    """
    if not SUPABASE_DB_BASE or not SUPABASE_SERVICE_KEY:
        raise RuntimeError("Supabase DB is not configured")

    url = f"{SUPABASE_DB_BASE}/{table}"
    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }
    resp = requests.post(url, headers=headers, json=row, timeout=10)
    if resp.status_code >= 400:
        raise RuntimeError(
            f"Supabase DB insert error {resp.status_code}: {resp.text}"
        )

# =============================================
# BASE64 / IMAGE UTILS
# =============================================

def _strip_data_url(b64: str) -> str:
    # Accept 'data:image/png;base64,...' or plain base64
    if not b64:
        return b64
    b64 = b64.strip()
    if b64.startswith("data:") and "base64," in b64:
        b64 = b64.split("base64,", 1)[1]
    return "".join(b64.split())  # remove whitespace/newlines

def decode_base64_bytes(b64: str) -> bytes:
    """Decode base64 string into bytes.

    Supports:
      - data-URL prefix (data:image/png;base64,...)
      - urlsafe base64 ('-' and '_' instead of '+' and '/')
      - missing padding '='
      - "double base64" (base64 of a base64 string) that sometimes happens in clients
    """
    if b64 is None:
        return b""

    b64_clean = _strip_data_url(str(b64)).strip()

    # Normalize urlsafe alphabet and padding
    b64_clean = b64_clean.replace("-", "+").replace("_", "/")
    pad = len(b64_clean) % 4
    if pad:
        b64_clean += "=" * (4 - pad)

    # First pass
    raw = base64.b64decode(b64_clean, validate=False)

    # If it looks like ASCII base64 text (double-encoded), try decode again
    try:
        as_text = raw.decode("utf-8").strip()
        if len(as_text) > 16 and all(c.isalnum() or c in "+/=_-\n\r" for c in as_text):
            as_text = _strip_data_url(as_text).strip().replace("-", "+").replace("_", "/")
            pad2 = len(as_text) % 4
            if pad2:
                as_text += "=" * (4 - pad2)
            raw2 = base64.b64decode(as_text, validate=False)
            # If second pass yields a PNG/JPEG signature, prefer it
            if raw2.startswith(b"\x89PNG\r\n\x1a\n") or raw2[:3] == b"\xff\xd8\xff":
                return raw2
    except Exception:
        pass

        return raw

    except Exception:
        pass

    return raw

def ensure_png_mask_bytes(mask_b64: str) -> bytes:
    """Return VALID PNG bytes for a mask.

    Supported inputs for `mask_b64`:
    1) base64(PNG/JPEG bytes)
    2) base64(JSON) where JSON contains `mask_png_base64` (base64 PNG bytes)

    Output is binarized (0/255) grayscale PNG.
    """
    raw = decode_base64_bytes(mask_b64)

    # If the client sent base64(JSON), extract embedded PNG base64.
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

    # Binarize for segmentation ground truth
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    ok, out = cv2.imencode(".png", mask_bin)
    if not ok:
        raise ValueError("Failed to encode mask as PNG")
    return out.tobytes()



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
        if not exif:
            return None

        gps_info = None
        for k, v in exif.items():
            tag = ExifTags.TAGS.get(k)
            if tag == "GPSInfo":
                gps_info = v
                break

        if not gps_info:
            return None

        lat = _deg(gps_info[2])
        lon = _deg(gps_info[4])

        if gps_info[1] == "S":
            lat = -lat
        if gps_info[3] == "W":
            lon = -lon

        return {"lat": lat, "lon": lon}
    except Exception:
        return None


# =============================================
# Reverse geocode (OSM)
# =============================================

def reverse_geocode(lat, lon):
    try:
        params = {
            "lat": lat,
            "lon": lon,
            "format": "jsonv2"
        }
        headers = {"User-Agent": NOMINATIM_USER_AGENT}
        r = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=5)
        r.raise_for_status()
        data = r.json()
        return data.get("display_name")
    except Exception:
        return None


# =============================================
# Weather API (OpenWeatherMap)
# =============================================

def get_weather(lat, lon):
    if not WEATHER_API_KEY:
        return None
    try:
        params = {
            "lat": lat,
            "lon": lon,
            "appid": WEATHER_API_KEY,
            "units": "metric",
            "lang": "ru",
        }
        r = requests.get(WEATHER_BASE_URL, params=params, timeout=5)
        r.raise_for_status()
        data = r.json()
        wind = data.get("wind", {})
        main = data.get("main", {})

        return {
            "temperature": main.get("temp"),
            "wind_speed": wind.get("speed"),
            "wind_gust": wind.get("gust"),
            "pressure": main.get("pressure"),
            "humidity": main.get("humidity"),
        }
    except Exception:
        return None


# =============================================
# SoilGrids (почва)
# =============================================

def get_soil(lat, lon):
    try:
        params = {
            "lon": lon,
            "lat": lat,
            "property": "clay,sand,silt,soc,phh2o",
            "depth": "0-5cm",
        }
        r = requests.get(SOILGRIDS_URL, params=params, timeout=7)
        r.raise_for_status()
        data = r.json()

        result = {}
        layers = data.get("properties", {}).get("layers", [])
        for layer in layers:
            name = layer.get("name")
            first_depth = layer.get("depths", [])
            if first_depth:
                mean = first_depth[0].get("values", {}).get("mean")
                result[name] = mean
        return result
    except Exception:
        return None


# =============================================
# Risk Calculation
# =============================================

SPECIES_BASE = {
    "Береза": 0.7,
    "Дуб": 0.5,
    "Ель": 1.0,
    "Сосна": 0.75,
    "Тополь": 0.95,
}

def slenderness_score(height, diameter):
    if not diameter or diameter <= 0:
        return 1.0
    S = height / diameter
    if S >= 80:
        return 1.0
    if S >= 60:
        return 0.7
    if S >= 40:
        return 0.4
    return 0.2

def soil_score(soil):
    if not soil:
        return 0.5
    clay = soil.get("clay") or 0
    sand = soil.get("sand") or 0
    org = soil.get("soc") or 0
    if org > 80:
        return 1.0
    if clay > 40:
        return 0.9
    if sand > 60:
        return 0.7
    return 0.5

def wind_score(weather):
    if not weather:
        return 0.5
    gust = weather.get("wind_gust") or weather.get("wind_speed") or 0
    if gust <= 5:
        return 0.2
    if gust <= 10:
        return 0.4
    if gust <= 15:
        return 0.6
    if gust <= 25:
        return 0.8
    return 1.0

def compute_risk(species, height, crown, diameter, weather, soil):
    expl = []

    base = SPECIES_BASE.get(species, 0.7)
    expl.append(f"Порода ({species}) базовый риск: {base:.2f}")

    if diameter and diameter > 0:
        S = height / diameter
    else:
        S = 0.0
    s_score = slenderness_score(height, diameter)
    expl.append(f"Коэфф. стройности H/D: {S:.1f} → {s_score:.2f}")

    w_score = wind_score(weather)
    expl.append(f"Ветровая нагрузка: {w_score:.2f}")

    soil_s = soil_score(soil)
    expl.append(f"Почвенный фактор: {soil_s:.2f}")

    index = 0.3 * base + 0.3 * s_score + 0.25 * w_score + 0.15 * soil_s
    index = max(0, min(index, 1))

    if index < 0.4:
        cat = "низкий"
    elif index < 0.7:
        cat = "средний"
    else:
        cat = "высокий"

    expl.append(f"Итоговый риск {index:.2f} ({cat})")

    return {
        "index": index,
        "category": cat,
        "explanation": expl,
    }



# =============================================
# Beta coefficient estimation (wind drag scaling)
# =============================================

# β values for full-scale Scots pine from Borisevich/Kamluk/Rebko paper:
# 25.5–90.0 kg/s. For other species, values are provisional expert ranges.
BETA_SPECIES_PARAMS = {
    "Сосна": {"k_area": 3.4, "min": 25.5, "max": 90.0, "base": 47.7},
    "Ель": {"k_area": 3.8, "min": 30.0, "max": 100.0, "base": 60.0},
    "Тополь": {"k_area": 3.6, "min": 30.0, "max": 100.0, "base": 58.0},
    "Береза": {"k_area": 3.3, "min": 25.0, "max": 90.0, "base": 52.0},
    "Дуб": {"k_area": 3.7, "min": 35.0, "max": 110.0, "base": 65.0},
}

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def estimate_beta_kg_s(species: str, height_m, crown_width_m, trunk_diameter_m=None, manual_beta_kg_s=None) -> dict:
    """Estimate β in kg/s for ArborScan practical wind-risk calculation.

    Method:
      1) If user provides manual β > 0, use it and only clamp to a broad safety range.
      2) Otherwise estimate crown frontal area:
            H_crown ≈ 0.45 * H
            A_crown ≈ 0.65 * H_crown * W_crown
         Then:
            β_raw = k_species * A_crown
         For pine, k≈3.4 and clamp range 25.5–90 kg/s follows the paper.
    """
    params = BETA_SPECIES_PARAMS.get(species, BETA_SPECIES_PARAMS["Сосна"])

    if manual_beta_kg_s is not None and manual_beta_kg_s > 0:
        value = round(_clamp(float(manual_beta_kg_s), 5.0, 200.0), 2)
        return {
            "beta_kg_s": value,
            "method": "manual",
            "source": "Введено вручную пользователем",
            "formula": "F = β · v",
            "input": {
                "manual_beta_kg_s": float(manual_beta_kg_s),
                "species": species,
            },
            "notes": [
                "β использован как заданный пользователем коэффициент аэродинамической интенсивности.",
                "Единица измерения: кг/с.",
            ],
        }

    h = float(height_m or 0)
    w = float(crown_width_m or 0)

    if h <= 0 or w <= 0:
        value = round(float(params["base"]), 2)
        return {
            "beta_kg_s": value,
            "method": "species_default",
            "source": "Среднее/экспертное значение по породе",
            "formula": "F = β · v",
            "input": {
                "species": species,
                "height_m": height_m,
                "crown_width_m": crown_width_m,
            },
            "notes": [
                "Недостаточно геометрических параметров для расчёта по кроне.",
                "Использовано базовое значение β для породы.",
            ],
        }

    crown_height_m = 0.45 * h
    crown_frontal_area_m2 = 0.65 * crown_height_m * w
    beta_raw = float(params["k_area"]) * crown_frontal_area_m2
    beta = round(_clamp(beta_raw, float(params["min"]), float(params["max"])), 2)

    return {
        "beta_kg_s": beta,
        "method": "estimated_from_geometry",
        "source": "Оценка по AR/геометрии кроны",
        "formula": "Hкроны≈0.45H; A≈0.65·Hкроны·Wкроны; β=clamp(k·A)",
        "input": {
            "species": species,
            "height_m": h,
            "crown_width_m": w,
            "trunk_diameter_m": trunk_diameter_m,
            "estimated_crown_height_m": round(crown_height_m, 2),
            "estimated_crown_frontal_area_m2": round(crown_frontal_area_m2, 2),
            "k_area": params["k_area"],
            "beta_min": params["min"],
            "beta_max": params["max"],
            "beta_raw": round(beta_raw, 2),
        },
        "notes": [
            "Это приближённый β для практического риска, а не лабораторно подобранный коэффициент.",
            "Для сосны диапазон ограничен опубликованными значениями 25.5–90.0 кг/с.",
        ],
    }

def beta_wind_force_score(beta_kg_s, weather) -> tuple[float, float | None]:
    """Return normalized score and force F=β*v using wind gust/speed."""
    if not beta_kg_s or beta_kg_s <= 0 or not weather:
        return 0.5, None
    v = weather.get("wind_gust") or weather.get("wind_speed") or 0
    try:
        v = float(v)
    except Exception:
        v = 0.0
    force_n = float(beta_kg_s) * v

    # Conservative practical normalization:
    # <150 N low, 150–500 moderate, 500–1200 high, >1200 very high.
    if force_n <= 150:
        score = 0.25
    elif force_n <= 500:
        score = 0.45
    elif force_n <= 1200:
        score = 0.75
    else:
        score = 1.0
    return score, round(force_n, 2)


# =============================================
# IMAGE ENCODING / DRAW MASK
# =============================================

def encode_jpeg_base64(img_bgr, max_side=1280, quality=74):
    """Resize and encode BGR image to JPEG base64 for mobile response.

    The original file is still saved to Supabase RAW storage. For API response
    we send a compressed preview to avoid Android OutOfMemoryError.
    """
    h, w = img_bgr.shape[:2]
    longest = max(h, w)
    if longest > max_side:
        scale = max_side / float(longest)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, out = cv2.imencode(".jpg", img_bgr, encode_params)
    if not ok:
        raise ValueError("Failed to encode image as JPEG")
    return base64.b64encode(out.tobytes()).decode("ascii")


def draw_mask(img_bgr, mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        approx = cv2.approxPolyDP(cnt, 0.003 * cv2.arcLength(cnt, True), True)
        cv2.drawContours(img_bgr, [approx], -1, (0, 255, 0), 3)

    return encode_jpeg_base64(img_bgr, max_side=1280, quality=74)


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

    # новые поля для исправленных параметров и масштаба
    correct_height_m: float | None = None
    correct_crown_width_m: float | None = None
    correct_trunk_diameter_m: float | None = None
    correct_scale_px_to_m: float | None = None

    # PNG маска, закодированная в base64
    user_mask_base64: str | None = None

class TrustedExample(BaseModel):
    analysis_id: str
    species: str | None = None
    trust_score: float | None = None
    created_at: str | None = None

    tree_ok: bool | None = None
    stick_ok: bool | None = None
    params_ok: bool | None = None
    species_ok: bool | None = None

    has_user_mask: bool | None = None
    use_for_training: bool | None = None
    needs_manual_review: bool | None = None




class AdminSetTrainingRequest(BaseModel):
    # accept several keys for backward/forward compatibility with the Flutter admin UI
    use_for_training: bool | None = None
    enabled: bool | None = None
    include: bool | None = None
    value: bool | None = None


app = FastAPI(title="ArborScan API v2.0")

@app.on_event("startup")
async def _log_routes_on_startup():
    try:
        fb = [r.path for r in app.router.routes if getattr(r, "path", "").endswith("feedback") or "feedback" in getattr(r, "path", "")]
        print(f"[*] Registered feedback routes: {sorted(set(fb))}")
    except Exception as e:
        print(f"[!] Failed to list routes: {e}")


# --- Training events (in-memory) ---
# Stored in a small ring buffer so the Admin Panel can show a live-ish log.
TRAINING_EVENTS = deque(maxlen=int(os.getenv("TRAINING_EVENTS_MAXLEN", "200")))

def log_training_event(level: str, message: str, data: dict | None = None) -> None:
    try:
        evt = {
            "ts": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "level": level.upper(),
            "message": message,
            "data": data or {},
        }
        TRAINING_EVENTS.append(evt)
        # keep a visible server log line too
        print(f"[TRAINING_EVENT] {evt['ts']} {evt['level']}: {evt['message']} {evt['data']}")
    except Exception:
        # Never break app because of logging
        pass


@app.on_event("startup")
def _startup_load_models():
    try:
        training_state_ensure_row()
        with MODEL_LOCK:
            reload_tree_model(force=True)
    except Exception as e:
        print(f"[!] Startup model load failed: {e}")





def normalize_address_ru(address: str | None) -> str | None:
    """Normalize common Belarusian OSM address fragments to Russian display text."""
    if not address:
        return address
    replacements = {
        "Інтэрнат": "Интернат",
        "вуліца": "улица",
        "вул.": "ул.",
        "Машынабудаўнікоў": "Машиностроителей",
        "Аўтазаводскі": "Автозаводский",
        "раён": "район",
        "Пасёлак": "Посёлок",
        "Заводскі": "Заводской",
        "Мінск": "Минск",
        "Беларусь": "Беларусь",
    }
    out = address
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    return out

@app.post("/analyze-tree")
async def analyze_tree(
    file: UploadFile = File(...),
    ar_height_m: Optional[float] = Form(None),
    ar_crown_width_m: Optional[float] = Form(None),
    ar_trunk_diameter_m: Optional[float] = Form(None),
    manual_beta_kg_s: Optional[float] = Form(None),
    crown_start_height_m: Optional[float] = Form(None),
    crown_density_factor: Optional[float] = Form(None),
    crown_shape_factor: Optional[float] = Form(None),
    manual_wind_speed_m_s: Optional[float] = Form(None),
    manual_wind_gust_m_s: Optional[float] = Form(None),
    lat: Optional[float] = Form(None),
    lon: Optional[float] = Form(None),
):
    image_bytes = await file.read()
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Не удалось прочитать изображение")

    H, W = img.shape[:2]

    # -----------------------------
    # YOLO TREE
    # -----------------------------
    tree_model_local = get_tree_model()
    tree_res = tree_model_local(img)[0]
    if tree_res.masks is None:
        return JSONResponse({"error": "Дерево не найдено"}, status_code=400)

    masks = []
    areas = []
    for m in tree_res.masks.data:
        mask = (m.cpu().numpy() > 0.5).astype(np.uint8) * 255
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        areas.append(mask.sum())
        masks.append(mask)

    idx = int(np.argmax(areas))
    mask = masks[idx]

    # -----------------------------
    # YOLO STICK
    # -----------------------------
    stick_res = stick_model(img)[0]
    scale = None
    if len(stick_res.boxes) > 0:
        best = max(stick_res.boxes, key=lambda b: b.xyxy[0][3] - b.xyxy[0][1])
        x1s, y1s, x2s, y2s = best.xyxy[0].cpu().numpy().astype(int)
        stick_h = y2s - y1s
        if stick_h > 10:
            scale = REAL_STICK_M / stick_h

    # -----------------------------
    # MEASUREMENTS
    # -----------------------------
    ys, xs = np.where(mask > 0)
    y_min, y_max = ys.min(), ys.max()
    height_px = y_max - y_min

    height_m = round(height_px * scale, 2) if scale else None

    crown_width_px = 0
    for y in range(y_min, y_min + int(0.7 * height_px)):
        row = np.where(mask[y] > 0)[0]
        if len(row) > 0:
            crown_width_px = max(crown_width_px, row.max() - row.min())
    crown_m = round(crown_width_px * scale, 2) if scale else None

    trunk_vals = []
    trunk_top = y_max - int(0.2 * height_px)
    for y in range(trunk_top, y_max):
        row = np.where(mask[y] > 0)[0]
        if len(row) > 0:
            trunk_vals.append(row.max() - row.min())
    trunk_px = np.mean(trunk_vals) if trunk_vals else None
    trunk_m = round(trunk_px * scale, 2) if scale and trunk_px else None

    # -----------------------------
    # AI + AR MEASUREMENTS MERGE
    # -----------------------------
    height_m_ai = height_m
    crown_m_ai = crown_m
    trunk_m_ai = trunk_m

    has_ar_height = ar_height_m is not None and ar_height_m > 0
    has_ar_crown = ar_crown_width_m is not None and ar_crown_width_m > 0
    has_ar_trunk = ar_trunk_diameter_m is not None and ar_trunk_diameter_m > 0

    if has_ar_height:
        height_m = round(float(ar_height_m), 2)
    if has_ar_crown:
        crown_m = round(float(ar_crown_width_m), 2)
    if has_ar_trunk:
        trunk_m = round(float(ar_trunk_diameter_m), 2)

    ar_measurements = {
        "height_m": round(float(ar_height_m), 2) if has_ar_height else None,
        "crown_width_m": round(float(ar_crown_width_m), 2) if has_ar_crown else None,
        "trunk_diameter_m": round(float(ar_trunk_diameter_m), 2) if has_ar_trunk else None,
    }

    measurement_sources = {
        "height_m": "ar" if has_ar_height else "image",
        "crown_width_m": "ar" if has_ar_crown else "image",
        "trunk_diameter_m": "ar" if has_ar_trunk else "image",
    }

    dimensions_source = "ИИ + AR" if any([has_ar_height, has_ar_crown, has_ar_trunk]) else "ИИ / фото"

    # -----------------------------
    # CLASSIFIER
    # -----------------------------
    x1, y1, x2, y2 = tree_res.boxes.xyxy[idx].cpu().numpy().astype(int)
    crop = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    pil_crop = Image.fromarray(crop)
    tens = transformer(pil_crop).unsqueeze(0)
    with torch.no_grad():
        pred = classifier(tens)
        cls_id = int(torch.argmax(pred))
    species_name = CLASS_NAMES_RU[cls_id]

    # -----------------------------
    # ANNOTATED IMAGE
    # -----------------------------
    annotated_b64 = draw_mask(img.copy(), mask)

    # -----------------------------
    # ENVIRONMENT
    # -----------------------------
    gps = None
    address = None
    weather = None
    soil = None
    risk = None

    if ENABLE_ENV_ANALYSIS:
        if lat is not None and lon is not None:
            gps = {"lat": float(lat), "lon": float(lon)}
        else:
            gps = extract_gps(image_bytes)
        if gps:
            address = normalize_address_ru(reverse_geocode(gps["lat"], gps["lon"]))
            weather = get_weather(gps["lat"], gps["lon"])
            soil = get_soil(gps["lat"], gps["lon"])

        # Manual wind values are used when GPS/weather is unavailable
        # or when the expert wants to test a scenario.
        if manual_wind_speed_m_s is not None or manual_wind_gust_m_s is not None:
            if weather is None:
                weather = {}
            if manual_wind_speed_m_s is not None and manual_wind_speed_m_s > 0:
                weather["wind_speed"] = float(manual_wind_speed_m_s)
            if manual_wind_gust_m_s is not None and manual_wind_gust_m_s > 0:
                weather["wind_gust"] = float(manual_wind_gust_m_s)
            weather["source"] = "manual"

        risk = compute_risk(
            species_name,
            height_m or 0,
            crown_m or 0,
            trunk_m or 0,
            weather,
            soil,
        )

    # -----------------------------
    # BETA COEFFICIENT / WIND LOAD
    # -----------------------------
    beta_info = estimate_beta_kg_s(
        species_name,
        height_m,
        crown_m,
        trunk_m,
        manual_beta_kg_s=manual_beta_kg_s,
    )
    wind_force_score_value, wind_force_n = beta_wind_force_score(
        beta_info.get("beta_kg_s"),
        weather,
    )
    beta_info["wind_force_n"] = wind_force_n
    beta_info["wind_force_score"] = wind_force_score_value

    # -----------------------------
    # ANALYTICAL WIND-LOAD MODEL
    # -----------------------------
    wind_speed_for_model = None
    wind_gust_for_model = None
    if weather:
        wind_speed_for_model = weather.get("wind_speed")
        wind_gust_for_model = weather.get("wind_gust")

    analytic_wind_model = compute_analytic_wind_model(
        species=species_name,
        height_m=height_m,
        crown_width_m=crown_m,
        trunk_diameter_m=trunk_m,
        beta_kg_s=beta_info.get("beta_kg_s"),
        wind_speed_m_s=wind_speed_for_model,
        wind_gust_m_s=wind_gust_for_model,
        crown_start_height_m=crown_start_height_m,
        crown_density_factor=crown_density_factor or 1.0,
        crown_shape_factor=crown_shape_factor or 1.0,
        n_elements=20,
    )

    # Add β explanation to risk without breaking the existing risk model yet.
    if risk is not None:
        risk.setdefault("explanation", [])
        beta_value = beta_info.get("beta_kg_s")
        risk["explanation"].append(
            f"Коэффициент β: {beta_value:.2f} кг/с ({beta_info.get('source')})"
            if isinstance(beta_value, (int, float))
            else f"Коэффициент β: {beta_info.get('source')}"
        )
        if wind_force_n is not None:
            risk["explanation"].append(
                f"Оценка ветровой силы по F=β·v: {wind_force_n:.1f} Н"
            )
        if analytic_wind_model.get("available"):
            out = analytic_wind_model.get("outputs", {})
            risk["explanation"].append(
                f"Аналитический момент у основания: {out.get('base_moment_nm')} Н·м"
            )
            risk["explanation"].append(
                f"Центр ветровой нагрузки: {out.get('center_of_load_m')} м"
            )

    # -----------------------------
    # TEMP CACHE FOR FEEDBACK
    # -----------------------------
    analysis_id = str(uuid4())

    meta = {
        "analysis_id": analysis_id,
        "species": species_name,
        "height_m": height_m,
        "crown_width_m": crown_m,
        "trunk_diameter_m": trunk_m,
        "height_m_ai": height_m_ai,
        "crown_width_m_ai": crown_m_ai,
        "trunk_diameter_m_ai": trunk_m_ai,
        "ar_measurements": ar_measurements,
        "measurement_sources": measurement_sources,
        "dimensions_source": dimensions_source,
        "beta": beta_info,
        "analytic_wind_model": analytic_wind_model,
        "scale_px_to_m": scale,
        "gps": gps,
        "address": address,
        "weather": weather,
        "soil": soil,
        "risk": risk,
        "beta": beta_info,
        "analytic_wind_model": analytic_wind_model,
        "model_versions": MODEL_VERSIONS,
        "model_versions": MODEL_VERSIONS,
        "build": BUILD_INFO,
        "schema_version": SCHEMA_VERSION,
        "api_version": API_VERSION,

    }

    # -----------------------------
    # PREPARE PRED OBJECTS (also for RAW storage)
    # -----------------------------
    # tree_pred
    tree_box_xyxy = tree_res.boxes.xyxy[idx].cpu().numpy().tolist()
    tree_conf = None
    tree_cls_id = None
    try:
        tree_conf = float(tree_res.boxes.conf[idx].cpu().item())
    except Exception:
        pass
    try:
        tree_cls_id = int(tree_res.boxes.cls[idx].cpu().item())
    except Exception:
        pass

    tree_pred = {
        "box_xyxy": tree_box_xyxy,
        "confidence": tree_conf,
        "class_id": tree_cls_id,
    }

    # stick_pred
    stick_pred = {
        "box_xyxy": None,
        "scale_px_to_m": scale,
    }
    try:
        if len(stick_res.boxes) > 0:
            best = max(stick_res.boxes, key=lambda b: b.xyxy[0][3] - b.xyxy[0][1])
            x1b, y1b, x2b, y2b = best.xyxy[0].cpu().numpy().astype(int)
            stick_pred["box_xyxy"] = [int(x1b), int(y1b), int(x2b), int(y2b)]
            try:
                stick_pred["confidence"] = float(best.conf[0].cpu().item())
            except Exception:
                pass
    except Exception:
        pass

    # -----------------------------
    # NEW: SAVE RAW SAMPLE (ALWAYS) → Supabase Storage
    # -----------------------------
    # Сохраняем все загрузки обычных пользователей независимо от feedback.
    # Если Supabase не настроен/временно недоступен — анализ НЕ ломаем.
    try:
        # input
        supabase_upload_bytes(
            SUPABASE_BUCKET_RAW,
            f"{analysis_id}/input.jpg",
            image_bytes,
        )

        # meta
        supabase_upload_json(
            SUPABASE_BUCKET_RAW,
            f"{analysis_id}/meta_auto.json",
            meta,
        )

        # annotated image (jpg)
        try:
            annotated_bytes_for_raw = base64.b64decode(annotated_b64)
            supabase_upload_bytes(
                SUPABASE_BUCKET_RAW,
                f"{analysis_id}/annotated.jpg",
                annotated_bytes_for_raw,
            )
        except Exception as e:
            print(f"[!] Failed to decode/upload annotated.jpg to RAW for {analysis_id}: {e}")

        # predictions (json)
        try:
            supabase_upload_json(
                SUPABASE_BUCKET_RAW,
                f"{analysis_id}/tree_pred.json",
                tree_pred,
            )
        except Exception as e:
            print(f"[!] Failed to upload tree_pred.json to RAW for {analysis_id}: {e}")

        try:
            supabase_upload_json(
                SUPABASE_BUCKET_RAW,
                f"{analysis_id}/stick_pred.json",
                stick_pred,
            )
        except Exception as e:
            print(f"[!] Failed to upload stick_pred.json to RAW for {analysis_id}: {e}")

    except Exception as e:
        print(f"[!] Failed to upload raw sample {analysis_id} to Supabase: {e}")

    # -----------------------------
    # CACHE IN /tmp FOR FEEDBACK
    # -----------------------------
    try:
        tmp_dir = Path("/tmp") / analysis_id
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # Оригинальное изображение
        with open(tmp_dir / "input.jpg", "wb") as f:
            f.write(image_bytes)

        # Аннотированное изображение (для контроля/обучения)
        try:
            annotated_bytes = base64.b64decode(annotated_b64)
            with open(tmp_dir / "annotated.jpg", "wb") as f:
                f.write(annotated_bytes)
        except Exception as e:
            print(f"[!] Failed to save annotated for {analysis_id}: {e}")

        # Предсказание дерева
        with open(tmp_dir / "tree_pred.json", "w", encoding="utf-8") as f:
            json.dump(tree_pred, f, ensure_ascii=False, indent=2)

        # Предсказание палки
        with open(tmp_dir / "stick_pred.json", "w", encoding="utf-8") as f:
            json.dump(stick_pred, f, ensure_ascii=False, indent=2)

        # Метаданные
        with open(tmp_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    except Exception as e:
        print(f"[!] Failed to cache analysis {analysis_id} in /tmp: {e}")

    # -----------------------------
    # RESPONSE
    # -----------------------------
    response = {
        "analysis_id": analysis_id,
        "species": species_name,
        "height_m": height_m,
        "crown_width_m": crown_m,
        "trunk_diameter_m": trunk_m,
        "height_m_ai": height_m_ai,
        "crown_width_m_ai": crown_m_ai,
        "trunk_diameter_m_ai": trunk_m_ai,
        "ar_measurements": ar_measurements,
        "measurement_sources": measurement_sources,
        "dimensions_source": dimensions_source,
        "beta": beta_info,
        "analytic_wind_model": analytic_wind_model,
        "scale_px_to_m": scale,
        "annotated_image_base64": annotated_b64,
    }
    # Добавляем сжатый preview оригинального изображения для экрана feedback.
    # Полный оригинал уже сохранён в Supabase RAW; в ответе держим лёгкую версию,
    # чтобы Flutter не падал с OutOfMemoryError на больших фото.
    try:
        response["original_image_base64"] = encode_jpeg_base64(img.copy(), max_side=1280, quality=72)
    except Exception:
        response["original_image_base64"] = None


    if gps:
        response["gps"] = gps
    if address:
        response["address"] = address
    if weather:
        response["weather"] = weather
    if soil:
        response["soil"] = soil
    if risk:
        response["risk"] = risk

    return JSONResponse(response)


@app.post("/feedback")
@app.post("/api/feedback")
def send_feedback(payload: dict = Body(...)):
    """
    Получаем подтверждение/исправление от пользователя и,
    если всё ок, сохраняем пример в Supabase для будущего обучения моделей
    + кладём запись в очередь доверенных примеров (Supabase DB).
    """
    # Accept both snake_case and camelCase from Flutter; avoid 422 on minor schema drift.
    analysis_id = payload.get('analysis_id') or payload.get('analysisId')
    if not analysis_id:
        raise HTTPException(status_code=422, detail='analysis_id is required')

    def _b(val, default=True):
        if val is None:
            return default
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return bool(val)
        if isinstance(val, str):
            v = val.strip().lower()
            if v in ('1','true','yes','y','ok'):
                return True
            if v in ('0','false','no','n'):
                return False
        return default

    use_for_training = _b(payload.get('use_for_training', payload.get('useForTraining')), default=True)
    tree_ok = _b(payload.get('tree_ok', payload.get('treeOk')), default=True)
    stick_ok = _b(payload.get('stick_ok', payload.get('stickOk')), default=True)
    params_ok = _b(payload.get('params_ok', payload.get('paramsOk')), default=True)
    species_ok = _b(payload.get('species_ok', payload.get('speciesOk')), default=True)

    correct_species = payload.get('correct_species') or payload.get('correctSpecies')
    correct_tree_mask = payload.get('correct_tree_mask') or payload.get('correctTreeMask')
    correct_stick_mask = payload.get('correct_stick_mask') or payload.get('correctStickMask')

    def _f(val):
        """Best-effort float parsing for user-corrected numeric fields."""
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            s = val.strip().replace(',', '.')
            if not s or s.lower() in ('null', 'none', 'nan'):
                return None
            try:
                return float(s)
            except Exception:
                return None
        return None

    corrected_height_m = _f(payload.get('corrected_height_m') or payload.get('correctedHeightM'))
    corrected_crown_width_m = _f(payload.get('corrected_crown_width_m') or payload.get('correctedCrownWidthM'))
    corrected_trunk_diameter_m = _f(payload.get('corrected_trunk_diameter_m') or payload.get('correctedTrunkDiameterM'))
    corrected_scale_px_to_m = _f(
        payload.get('corrected_scale_px_to_m') or payload.get('correctedScalePxToM') or
        payload.get('scale_px_to_m_corrected') or payload.get('scalePxToMCorrected') or
        payload.get('scale_px_to_m') or payload.get('scalePxToM') or
        payload.get('scale') or payload.get('corrected_scale') or payload.get('correctedScale')
    )
    user_mask_base64 = payload.get('user_mask_base64') or payload.get('userMaskBase64') or payload.get('mask_base64') or payload.get('maskBase64')

    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise HTTPException(status_code=500, detail="Supabase не настроен на сервере")

    tmp_dir = Path("/tmp") / analysis_id
    if not tmp_dir.exists():
        raise HTTPException(status_code=404, detail="analysis_id не найден или истёк срок хранения")

    # Если пользователь не хочет использовать пример в обучении
    if not use_for_training:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass
        return {"status": "ignored", "reason": "user_disabled_training"}

    meta_path = tmp_dir / "meta.json"
    if not meta_path.exists():
        raise HTTPException(status_code=500, detail="meta.json не найден для указанного analysis_id")

    # Загружаем исходное meta
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка чтения meta.json: {e}")

    # Обновляем meta фидбеком
    meta["tree_ok"] = tree_ok
    meta["stick_ok"] = stick_ok
    meta["params_ok"] = params_ok
    meta["species_ok"] = species_ok
    meta["correct_species"] = correct_species

    # Исправленный вид дерева
    if (not species_ok) and correct_species:
        meta["species"] = correct_species

    # Исправленные численные параметры (если пришли от клиента)
    if corrected_height_m is not None:
        meta["height_m"] = corrected_height_m
    if corrected_crown_width_m is not None:
        meta["crown_width_m"] = corrected_crown_width_m
    if corrected_trunk_diameter_m is not None:
        meta["trunk_diameter_m"] = corrected_trunk_diameter_m
    if corrected_scale_px_to_m is not None:
        meta["scale_px_to_m"] = corrected_scale_px_to_m

    # Trust score
    trust = 0.0
    if tree_ok:
        trust += 0.3
    if stick_ok:
        trust += 0.2
    if params_ok:
        trust += 0.2
    if species_ok or correct_species:
        trust += 0.3
    meta["trust_score"] = trust

    # ---------------------------------------------
    # VERIFIED PIPELINE
    # ---------------------------------------------

    is_verified = (
    use_for_training and
    trust >= VERIFIED_TRUST_THRESHOLD
    )


    analysis_id = analysis_id

    # -----------------------------
    # UPLOAD TO SUPABASE STORAGE
    # -----------------------------
    try:
        # input.jpg
        input_path = tmp_dir / "input.jpg"
        if input_path.exists():
            supabase_upload_bytes(
                SUPABASE_BUCKET_INPUTS,
                f"{analysis_id}/input.jpg",
                input_path.read_bytes(),
            )

        # annotated.jpg
        annotated_path = tmp_dir / "annotated.jpg"
        if annotated_path.exists():
            supabase_upload_bytes(
                SUPABASE_BUCKET_INPUTS,
                f"{analysis_id}/annotated.jpg",
                annotated_path.read_bytes(),
            )

        # user_mask.png (segmentation ground truth)
        # ВАЖНО: сохраняем/загружаем ТОЛЬКО валидный PNG (0/255), иначе OpenCV/YOLO dataset builder не сможет читать маску.
                # Маска пользователя (обводка) — опционально.
        # Если маски нет, это НЕ ошибка (просто этот пример не пойдёт в сегментационный датасет).
                # Маска пользователя (обводка) — опционально.
        # Правило: если has_user_mask когда-либо стал True — НЕ затираем его последующими /feedback без маски.
        existing_has_user_mask = False
        try:
            mv_raw = supabase_download_bytes(SUPABASE_BUCKET_VERIFIED, f"{analysis_id}/meta_verified.json")
            mv = json.loads(mv_raw)
            existing_has_user_mask = bool(mv.get("has_user_mask", False))
        except Exception:
            existing_has_user_mask = False

        meta["has_user_mask"] = existing_has_user_mask

        mask_b64 = user_mask_base64
        if mask_b64 is not None:
            mask_b64_str = str(mask_b64).strip().lower()
        else:
            mask_b64_str = ""

        if mask_b64_str and mask_b64_str not in ("null", "undefined"):
            try:
                mask_png_bytes = ensure_png_mask_bytes(str(mask_b64))
                supabase_upload_bytes(
                    SUPABASE_BUCKET_INPUTS,
                    f"{analysis_id}/user_mask.png",
                    mask_png_bytes,
                )
                meta["has_user_mask"] = True
            except Exception as e:
                # Не валим feedback целиком; просто предупреждаем, что маску не удалось распарсить.
                print(f"[!] User mask provided but could not be decoded for {analysis_id}: {e}")
        else:
            # Маски нет — нормальный кейс.
            pass


        # tree_pred.json
        tree_pred_path = tmp_dir / "tree_pred.json"
        if tree_pred_path.exists():
            supabase_upload_bytes(
                SUPABASE_BUCKET_PRED,
                f"{analysis_id}/tree_pred.json",
                tree_pred_path.read_bytes(),
            )

        # stick_pred.json
        stick_pred_path = tmp_dir / "stick_pred.json"
        if stick_pred_path.exists():
            supabase_upload_bytes(
                SUPABASE_BUCKET_PRED,
                f"{analysis_id}/stick_pred.json",
                stick_pred_path.read_bytes(),
            )

        # meta.json (обновлённый)
        supabase_upload_json(
            SUPABASE_BUCKET_META,
            f"{analysis_id}.json",
            meta,
        )

        if is_verified:
            try:
                # input
                supabase_upload_bytes(
                    SUPABASE_BUCKET_VERIFIED,
                    f"{analysis_id}/input.jpg",
                    (tmp_dir / "input.jpg").read_bytes(),
                )

                # annotated
                annotated_path = tmp_dir / "annotated.jpg"
                if annotated_path.exists():
                    supabase_upload_bytes(
                        SUPABASE_BUCKET_VERIFIED,
                        f"{analysis_id}/annotated.jpg",
                        annotated_path.read_bytes(),
                    )

                
                # user mask (если есть) — нормализуем в валидный PNG
                if user_mask_base64:
                    try:
                        mask_png_bytes = ensure_png_mask_bytes(user_mask_base64)
                        supabase_upload_bytes(
                            SUPABASE_BUCKET_VERIFIED,
                            f"{analysis_id}/user_mask.png",
                            mask_png_bytes,
                        )
                    except Exception as e:
                        print(f"[!] Failed to upload VERIFIED user mask for {analysis_id}: {e}")

                # predictions
                supabase_upload_bytes(
                    SUPABASE_BUCKET_VERIFIED,
                    f"{analysis_id}/tree_pred.json",
                    (tmp_dir / "tree_pred.json").read_bytes(),
                )

                supabase_upload_bytes(
                    SUPABASE_BUCKET_VERIFIED,
                    f"{analysis_id}/stick_pred.json",
                    (tmp_dir / "stick_pred.json").read_bytes(),
                )

                meta_verified = meta.copy()
                meta_verified["verified"] = True
                meta_verified["verified_at"] = datetime.utcnow().isoformat()
                meta_verified["verifier_role"] = "admin" if not use_for_training else "user"

                supabase_upload_json(
                    SUPABASE_BUCKET_VERIFIED,
                    f"{analysis_id}/meta_verified.json",
                    meta_verified,
                )
               
                

            except Exception as e:
                print(f"[!] Failed to upload VERIFIED sample {analysis_id}: {e}")


    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при загрузке в Supabase: {e}")

def request_retrain_if_needed():
    # считаем сколько масок ещё не использовано
    count = count_untrained_masks()

    if count >= 10:
        training_state_update({"retrain_requested": True})

    # -----------------------------
    # Запись в очередь доверенных примеров (Supabase DB)
    # -----------------------------
    # -----------------------------
    # 7) (Опционально) очередь доверенных примеров (Supabase Postgres)
    # -----------------------------
    if SUPABASE_ENABLE_QUEUE:
        try:
            queue_row = {
                "analysis_id": analysis_id,
                "trust_score": trust,
                "species": meta.get("species"),
                "has_user_mask": meta.get("has_user_mask", False),
                "tree_ok": meta.get("tree_ok"),
                "stick_ok": meta.get("stick_ok"),
                "params_ok": meta.get("params_ok"),
                "species_ok": meta.get("species_ok"),
            }
            supabase_db_insert(SUPABASE_QUEUE_TABLE, queue_row)
        except Exception as e:
            # Не падаем для пользователя; отсутствие таблицы / ошибки очереди не должны блокировать обучение.
            print(f"[!] Queue insert skipped for {analysis_id}: {e}")
    else:
        # Очередь выключена — нормально для пайплайна обучения через Storage.
        pass
    # Чистим /tmp
    try:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception as e:
        print(f"[!] Failed to remove tmp dir for {analysis_id}: {e}")

    return {
        "status": "ok",
        "analysis_id": analysis_id,
        "trust_score": trust,
    }
@app.get("/admin/verified-list")
def admin_verified_list(include_used: bool = False):
    """
    Возвращает список analysis_id из arborscan-verified + краткую информацию из meta_verified.json.

    По умолчанию скрывает примеры, которые уже использовались для обучения (used_for_training=true),
    чтобы вкладка «Датасет для обучения» в приложении показывала только "актуальные" примеры.
    Если нужно увидеть всё — вызови с include_used=true.
    """
    try:
        objects = supabase_list_objects(SUPABASE_BUCKET_VERIFIED)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    analysis_ids = sorted({obj["name"].split("/")[0] for obj in objects})

    results = []
    for aid in analysis_ids:
        try:
            meta_bytes = supabase_download_bytes(
                SUPABASE_BUCKET_VERIFIED,
                f"{aid}/meta_verified.json",
            )
            meta = json.loads(meta_bytes)

            # Hide already-consumed samples by default (keep them in Supabase, just not shown in UI)
            if (not include_used) and meta.get("used_for_training") is True:
                continue

            results.append({
                "analysis_id": aid,
                "species": meta.get("species"),
                "risk_category": meta.get("risk", {}).get("category"),
                "trust_score": meta.get("trust_score"),
                "verified": meta.get("verified", True),
                "verified_at": meta.get("verified_at"),
                # important for TrainingDatasetPage
                "exclude_from_training": meta.get("exclude_from_training", False) == True,
                # optional debug fields
                "has_user_mask": meta.get("has_user_mask", False) == True,
                "used_for_training": meta.get("used_for_training", False) == True,
            })
        except Exception:
            continue

    return {
        "count": len(results),
        "items": results,
    }

@app.get("/admin/training-candidates")
def admin_training_candidates(limit: int = 50):
    """
    Диагностический эндпоинт: показывает, почему воркер видит 0 new samples.

    Возвращает:
      - eligible_new: которые попадут как NEW (has_user_mask && !exclude && !used_for_training)
      - eligible_replay_pool: которые могут быть выбраны как REPLAY (has_user_mask && !exclude && used_for_training)
      - skipped_counts: статистика причин отсева
      - examples: первые N примеров с краткой причиной
    """
    try:
        objects = supabase_list_objects(SUPABASE_BUCKET_VERIFIED)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"list_objects failed: {e}")

    analysis_ids = sorted({obj["name"].split("/")[0] for obj in objects})

    eligible_new = []
    eligible_replay_pool = []
    skipped_counts = {
        "no_meta_verified": 0,
        "no_user_mask": 0,
        "excluded": 0,
        "already_used": 0,
        "other": 0,
    }
    examples = []

    def _reason(meta):
        if meta is None:
            return "no_meta_verified"
        if not meta.get("has_user_mask", False):
            return "no_user_mask"
        if meta.get("exclude_from_training", False):
            return "excluded"
        if meta.get("used_for_training", False):
            return "already_used"
        return "eligible_new"

    for aid in analysis_ids:
        meta = None
        try:
            meta_bytes = supabase_download_bytes(SUPABASE_BUCKET_VERIFIED, f"{aid}/meta_verified.json")
            meta = json.loads(meta_bytes)
        except Exception:
            meta = None

        r = _reason(meta)
        if r == "eligible_new":
            eligible_new.append(aid)
        elif r == "already_used":
            if meta and meta.get("has_user_mask", False) and (not meta.get("exclude_from_training", False)):
                eligible_replay_pool.append(aid)
            skipped_counts["already_used"] += 1
        elif r in skipped_counts:
            skipped_counts[r] += 1
        else:
            skipped_counts["other"] += 1

        if len(examples) < max(0, int(limit)):
            examples.append({
                "analysis_id": aid,
                "reason": r,
                "has_user_mask": bool(meta.get("has_user_mask")) if meta else None,
                "exclude_from_training": bool(meta.get("exclude_from_training")) if meta else None,
                "used_for_training": bool(meta.get("used_for_training")) if meta else None,
                "verified_at": meta.get("verified_at") if meta else None,
            })

    return {
        "bucket_verified": SUPABASE_BUCKET_VERIFIED,
        "eligible_new_count": len(eligible_new),
        "eligible_new": eligible_new[:limit],
        "eligible_replay_pool_count": len(eligible_replay_pool),
        "eligible_replay_pool": eligible_replay_pool[:limit],
        "skipped_counts": skipped_counts,
        "examples": examples,
        "notes": {
            "new_criteria": "has_user_mask=true AND exclude_from_training!=true AND used_for_training!=true",
            "replay_pool_criteria": "has_user_mask=true AND exclude_from_training!=true AND used_for_training=true",
        }
    }
@app.post("/admin/verified/{analysis_id}/set-training")
@app.post("/admin/verified/{analysis_id}/set-training/")
def admin_set_training_flag(analysis_id: str, req: AdminSetTrainingRequest):
    """
    Toggle whether a verified sample should be used for training.
    The admin UI uses this to include/exclude items from the training dataset export.

    Важно: обновляем И meta.json, И meta_verified.json, т.к. export/worker берёт поля из meta_verified.json.
    """
    # pick the first provided flag from a set of accepted keys
    flag = None
    for v in (req.use_for_training, req.enabled, req.include, req.value):
        if v is not None:
            flag = bool(v)
            break
    if flag is None:
        raise HTTPException(
            status_code=400,
            detail="Missing boolean flag. Send JSON with one of: use_for_training / enabled / include / value.",
        )

    # Helper to load json safely
    def _load_json(path: str) -> dict:
        try:
            raw = supabase_download_bytes(SUPABASE_BUCKET_VERIFIED, path)
            if not raw:
                return {}
            if isinstance(raw, (bytes, bytearray)):
                return json.loads(raw.decode("utf-8"))
            return json.loads(raw)
        except Exception:
            return {}

    # Update meta.json (legacy)
    meta_path = f"{analysis_id}/meta.json"
    meta = _load_json(meta_path)
    meta["analysis_id"] = analysis_id
    meta["use_for_training"] = flag
    meta["exclude_from_training"] = (not flag)
    supabase_upload_json(SUPABASE_BUCKET_VERIFIED, meta_path, meta)

    # Update meta_verified.json (source of truth for training pipeline)
    mv_path = f"{analysis_id}/meta_verified.json"
    mv = _load_json(mv_path)
    mv["analysis_id"] = analysis_id
    mv["use_for_training"] = flag
    mv["exclude_from_training"] = (not flag)
    supabase_upload_json(SUPABASE_BUCKET_VERIFIED, mv_path, mv)

    return {"analysis_id": analysis_id, "use_for_training": flag, "exclude_from_training": (not flag)}
@app.get("/admin/analysis/{analysis_id}")
def admin_get_analysis(analysis_id: str):
    """
    Детали одного verified анализа для админки
    """
    try:
        input_img = supabase_download_bytes(
            SUPABASE_BUCKET_VERIFIED,
            f"{analysis_id}/input.jpg",
        )
        annotated_img = supabase_download_bytes(
            SUPABASE_BUCKET_VERIFIED,
            f"{analysis_id}/annotated.jpg",
        )

        # Optional: user-corrected mask (may not exist for older verified items)
        user_mask_img = None
        try:
            user_mask_img = supabase_download_bytes(
                SUPABASE_BUCKET_VERIFIED,
                f"{analysis_id}/user_mask.png",
            )
        except Exception:
            user_mask_img = None

        tree_pred = json.loads(
            supabase_download_bytes(
                SUPABASE_BUCKET_VERIFIED,
                f"{analysis_id}/tree_pred.json",
            )
        )
        stick_pred = json.loads(
            supabase_download_bytes(
                SUPABASE_BUCKET_VERIFIED,
                f"{analysis_id}/stick_pred.json",
            )
        )
        meta = json.loads(
            supabase_download_bytes(
                SUPABASE_BUCKET_VERIFIED,
                f"{analysis_id}/meta_verified.json",
            )
        )

    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Analysis {analysis_id} not found or incomplete: {e}",
        )

    return {
        "analysis_id": analysis_id,
        "images": {
            "input_base64": base64.b64encode(input_img).decode("utf-8"),
            "annotated_base64": base64.b64encode(annotated_img).decode("utf-8"),
            "user_mask_base64": base64.b64encode(user_mask_img).decode("utf-8")
            if user_mask_img
            else None,
        },
        "tree_pred": tree_pred,
        "stick_pred": stick_pred,
        "meta": meta,
    }

# =============================================
# DATASET COLLECTION ENDPOINT (for training from app)
# =============================================

DATASET_ROOT = "datasets/trees_segmentation"
IMAGES_DIR = os.path.join(DATASET_ROOT, "images")
MASKS_DIR = os.path.join(DATASET_ROOT, "masks")
META_DIR = os.path.join(DATASET_ROOT, "meta")

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(MASKS_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)


class UserMaskPayload(BaseModel):
    analysis_id: str
    image_base64: str
    mask_base64: str
    meta: dict | None = None



# ---------------------------------------------------------
# Admin training control + model version switch (used by app)
# ---------------------------------------------------------

@app.get("/admin/training-status")
def admin_training_status():
    training_state_ensure_row()
    state = training_state_get()
    # ensure defaults
    if "active_model_version" not in state or state["active_model_version"] is None:
        state["active_model_version"] = 0
    if "last_model_version" not in state or state["last_model_version"] is None:
        state["last_model_version"] = 0
    if "training_in_progress" not in state or state["training_in_progress"] is None:
        state["training_in_progress"] = False
    if "retrain_requested" not in state or state["retrain_requested"] is None:
        state["retrain_requested"] = False
    return state

class _SetActiveModelBody(BaseModel):
    version: int

@app.post("/admin/set-active-model")
async def admin_set_active_model(payload: dict = Body(...)):
    training_state_ensure_row()
    raw_v = payload.get('version') or payload.get('model_version') or payload.get('active_model_version')
    if raw_v is None:
        raise HTTPException(status_code=422, detail="Missing 'version' in request body")
    v = int(raw_v)

    # verify model exists in Supabase Storage bucket SUPABASE_BUCKET_MODELS
    filename = f"model_v{v}.pt"
    try:
        _ = supabase_download_bytes(SUPABASE_BUCKET_MODELS, filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Model file not found in Supabase Storage: {filename}. {e}")

    training_state_update({"active_model_version": v})

    # hot reload immediately (no server restart)
    with MODEL_LOCK:
        reload_tree_model(force=True)

    return {"status": "ok", "active_model_version": v}

@app.post("/admin/request-retrain")
def admin_request_retrain():
    log_training_event("INFO", "Admin requested retraining")
    training_state_ensure_row()
    training_state_update({"retrain_requested": True})
    return {"status": "ok", "retrain_requested": True}

@app.post("/dataset/user-mask")
def save_user_mask(payload: UserMaskPayload):
    """Сохраняет пару (оригинал + маска) в локальный датасет.
    Маска нормализуется в валидный PNG (0/255). Поддерживает data-URL prefix."""
    analysis_id = payload.analysis_id

    try:
        image_bytes = decode_base64_bytes(payload.image_base64)
        mask_png_bytes = ensure_png_mask_bytes(payload.mask_base64)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 data")

    image_path = os.path.join(IMAGES_DIR, f"{analysis_id}.jpg")
    mask_path = os.path.join(MASKS_DIR, f"{analysis_id}.png")
    meta_path = os.path.join(META_DIR, f"{analysis_id}.json")

    with open(image_path, "wb") as f:
        f.write(image_bytes)

    with open(mask_path, "wb") as f:
        f.write(mask_png_bytes)

    meta = {
        "analysis_id": analysis_id,
        "saved_at": datetime.utcnow().isoformat(),
        **(payload.meta or {}),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {
        "status": "ok",
        "analysis_id": analysis_id,
        "files": {
            "image": image_path,
            "mask": mask_path,
            "meta": meta_path
        }
    }


@app.get("/admin/models")
def admin_models():
    """List available model versions and which one is active."""
    return {"models": list_available_model_versions(), "active_model_version": _get_active_model_version()}

@app.get("/admin/training-events")
def admin_training_events(limit: int = 15):
    """Return last training/admin events for UI."""
    limit = max(1, min(int(limit), 200))
    items = list(TRAINING_EVENTS)[-limit:]
    # newest first for UI convenience
    return {"events": list(reversed(items))}



def get_latest_model_path():
    state = training_state_get()
    v = int(state.get("last_model_version", 0))
    if v == 0:
        return "models/base.pt"
    return f"models/model_v{v}.pt"
