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
from ultralytics import YOLO
from PIL import Image, ExifTags
import torch
from torchvision import models, transforms
from fastapi import FastAPI, File, UploadFile, HTTPException, Body, Form
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
from uuid import uuid4
from pathlib import Path
from pydantic import BaseModel
from datetime import datetime, timedelta
from collections import deque
from typing import Optional, Dict, Any, List, Tuple

try:
    from google.oauth2 import id_token as google_id_token
    from google.auth.transport import requests as google_requests
except Exception:
    google_id_token = None
    google_requests = None
from tree_dynamics import compute_analytic_wind_model

# -------------------------------------
# CONFIG
# -------------------------------------

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    print("[!] Warning: SUPABASE_URL or SUPABASE_SERVICE_KEY not set. /feedback will not upload to Supabase.")

# Buckets в Supabase Storage
SUPABASE_BUCKET_INPUTS = "arborscan-inputs"
SUPABASE_BUCKET_PRED = "arborscan-predictions"
SUPABASE_BUCKET_META = "arborscan-meta"
SUPABASE_BUCKET_VERIFIED = "arborscan-verified"
SUPABASE_BUCKET_RAW = "arborscan-raw"
SUPABASE_BUCKET_MODELS = os.getenv("SUPABASE_BUCKET_MODELS", "arborscan-models")

# Supabase Postgres REST API (Вместо локального SQLite)
SUPABASE_DB_BASE = SUPABASE_URL.rstrip("/") + "/rest/v1" if SUPABASE_URL else None
SUPABASE_QUEUE_TABLE = "arborscan_feedback_queue"
SUPABASE_ENABLE_QUEUE = os.getenv("SUPABASE_ENABLE_QUEUE", "false").lower() == "true"

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


# Настройки окружения
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
tree_model = None  
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
    if v is None:
        return 0
    return int(v)

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
            if not part:
                continue
            try:
                versions.add(int(part))
            except ValueError:
                pass

    for p in Path("/tmp/models").glob("model_v*.pt"):
        mm = re.search(r"model_v(\d+)\.pt$", p.name)
        if mm:
            versions.add(int(mm.group(1)))

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
        if TREE_MODEL is None:
            raise RuntimeError("TREE_MODEL is not loaded")
        return TREE_MODEL

def supabase_db_insert(table: str, row: dict):
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
    if not b64:
        return b64
    b64 = b64.strip()
    if b64.startswith("data:") and "base64," in b64:
        b64 = b64.split("base64,", 1)[1]
    return "".join(b64.split()) 

def decode_base64_bytes(b64: str) -> bytes:
    if b64 is None:
        return b""

    b64_clean = _strip_data_url(str(b64)).strip()
    b64_clean = b64_clean.replace("-", "+").replace("_", "/")
    pad = len(b64_clean) % 4
    if pad:
        b64_clean += "=" * (4 - pad)

    raw = base64.b64decode(b64_clean, validate=False)

    try:
        as_text = raw.decode("utf-8").strip()
        if len(as_text) > 16 and all(c.isalnum() or c in "+/=_-\n\r" for c in as_text):
            as_text = _strip_data_url(as_text).strip().replace("-", "+").replace("_", "/")
            pad2 = len(as_text) % 4
            if pad2:
                as_text += "=" * (4 - pad2)
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

# =============================================
# Beta coefficient estimation (На базе Borisevich et al. 2021)
# =============================================

# Эмпирические данные (на базе полевых испытаний полноразмерных деревьев)
BETA_EMPIRICAL_STATS = {
    "Сосна": {"mean": 47.7, "min": 25.5, "max": 90.0, "ref_height": 20.0},
    "Ель": {"mean": 60.0, "min": 30.0, "max": 100.0, "ref_height": 20.0},
    "Береза": {"mean": 52.0, "min": 25.0, "max": 90.0, "ref_height": 20.0},
    "Дуб": {"mean": 65.0, "min": 35.0, "max": 110.0, "ref_height": 20.0},
    "Тополь": {"mean": 58.0, "min": 30.0, "max": 100.0, "ref_height": 25.0},
}

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def estimate_beta_kg_s(species: str, height_m, crown_width_m, trunk_diameter_m=None, manual_beta_kg_s=None, crown_density_factor=1.0) -> dict:
    stats = BETA_EMPIRICAL_STATS.get(species, BETA_EMPIRICAL_STATS["Сосна"])

    # Если администратор/пользователь ввел бету вручную
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

    # Если высота неизвестна, берем среднее по породе
    if h <= 0:
        return {
            "beta_kg_s": stats["mean"],
            "beta_max_scenario": stats["max"],
            "method": "species_default",
            "source": "Статистическое среднее по породе",
            "input": {"species": species},
            "notes": ["Высота не измерена. Использовано среднее значение из исследований."],
        }

    # 1. Аллометрическое масштабирование. 
    # Так как лобовая площадь при ветре сильно деформируется, зависимость от высоты 
    # берется не квадратичная, а с коэффициентом ~1.5 (эмпирическая поправка на гибкость).
    height_ratio = h / stats["ref_height"]
    
    # 2. Ожидаемый коэффициент бета
    beta_expected = stats["mean"] * (height_ratio ** 1.5) * density
    beta_expected = _clamp(beta_expected, 5.0, stats["max"] * 1.5)

    # 3. Худший сценарий (Worst-case)
    # Как показало Дерево №3 в статье, бета может быть почти в 1.9 раз больше средней (90 / 47.7 = 1.88)
    # из-за аномально жестких веток.
    beta_max_scenario = _clamp(beta_expected * 1.88, beta_expected, stats["max"] * 1.5)

    return {
        "beta_kg_s": round(beta_expected, 2),
        "beta_max_scenario": round(beta_max_scenario, 2),
        "method": "empirical_borisevich_2021",
        "source": "Полевая статистика (Borisevich et al. 2021)",
        "formula": f"β = {stats['mean']} * (H/{stats['ref_height']})^1.5",
        "input": {
            "species": species,
            "height_m": h,
            "crown_density_factor": density
        },
        "notes": [
            "Расчет базируется на полевых испытаниях реальных деревьев.",
            "Площадь кроны не используется напрямую, так как доказана её слабая связь с β из-за аэродинамической реконфигурации (streamlining).",
            f"Внимание: при жесткой архитектуре ветвей возможен худший сценарий до {round(beta_max_scenario, 2)} кг/с."
        ],
    }

def beta_wind_force_score(beta_kg_s, weather) -> tuple[float, float | None]:
    if not beta_kg_s or beta_kg_s <= 0 or not weather:
        return 0.5, None
    v = weather.get("wind_gust") or weather.get("wind_speed") or 0
    try:
        v = float(v)
    except Exception:
        v = 0.0
    
    force_n = float(beta_kg_s) * v

    # Шкала силы ветра адаптирована под реальные ньютоны
    if force_n <= 200:
        score = 0.25
    elif force_n <= 600:
        score = 0.45
    elif force_n <= 1500:
        score = 0.75
    else:
        score = 1.0
    return score, round(force_n, 2)


# =============================================
# IMAGE ENCODING / DRAW MASK
# =============================================

def encode_jpeg_base64(img_bgr, max_side=1280, quality=74):
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
    correct_height_m: float | None = None
    correct_crown_width_m: float | None = None
    correct_trunk_diameter_m: float | None = None
    correct_scale_px_to_m: float | None = None
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
    use_for_training: bool | None = None
    enabled: bool | None = None
    include: bool | None = None
    value: bool | None = None


# =============================================
# AUTH MODELS
# =============================================

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


app = FastAPI(title="ArborScan API v2.0")

# =============================================
# SUPABASE AUTH & ANALYSES (РЕШЕНИЕ ПРОБЛЕМЫ SQLITE)
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

    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        120_000,
    ).hex()
    return digest, salt_hex

def _user_public(row: dict) -> dict:
    data = {
        "id": row.get("id"),
        "name": row.get("name"),
        "email": row.get("email"),
        "role": row.get("role"),
        "created_at": row.get("created_at"),
    }
    if "provider" in row:
        data["provider"] = row["provider"]
    if "avatar_url" in row:
        data["avatar_url"] = row["avatar_url"]
    return data

def _create_session(user_id: str) -> dict:
    if not SUPABASE_DB_BASE:
        raise RuntimeError("Supabase is not configured.")
    token = secrets.token_urlsafe(32)
    created_at = _now_iso()
    expires_at = (datetime.utcnow() + timedelta(days=AUTH_TOKEN_TTL_DAYS)).isoformat(timespec="seconds") + "Z"
    
    url = f"{SUPABASE_DB_BASE}/auth_sessions"
    payload = {
        "token": token,
        "user_id": user_id,
        "created_at": created_at,
        "expires_at": expires_at
    }
    requests.post(url, headers=_sb_headers(), json=payload, timeout=10)
    return {"token": token, "expires_at": expires_at}

def _get_user_by_token(token: str) -> dict | None:
    if not token or not SUPABASE_DB_BASE:
        return None
    now = _now_iso()
    url = f"{SUPABASE_DB_BASE}/auth_sessions?token=eq.{token}&expires_at=gt.{now}&select=*,users(*)"
    resp = requests.get(url, headers=_sb_headers(), timeout=10)
    if resp.status_code >= 400:
        return None
    rows = resp.json()
    if not rows:
        return None
    return rows[0].get("users")

def _email_norm(email: str) -> str:
    return (email or "").strip().lower()

def _validate_auth_payload(name: str | None, email: str, password: str, need_name: bool = False):
    if need_name and (not name or len(name.strip()) < 2):
        raise HTTPException(status_code=400, detail="Имя должно быть не короче 2 символов")
    if "@" not in email or "." not in email:
        raise HTTPException(status_code=400, detail="Введите корректную почту")
    if not password or len(password) < 4:
        raise HTTPException(status_code=400, detail="Пароль должен быть не короче 4 символов")

@app.post("/auth/register")
async def auth_register(payload: AuthRegisterRequest):
    if not SUPABASE_DB_BASE: raise HTTPException(status_code=500, detail="Database disabled")
    name = payload.name.strip()
    email = _email_norm(payload.email)
    password = payload.password
    _validate_auth_payload(name, email, password, need_name=True)

    url_check = f"{SUPABASE_DB_BASE}/users?email=eq.{email}"
    if requests.get(url_check, headers=_sb_headers(), timeout=10).json():
        raise HTTPException(status_code=409, detail="Пользователь с такой почтой уже существует")

    password_hash, salt = _hash_password(password)
    user_id = str(uuid4())
    now = _now_iso()

    url_insert = f"{SUPABASE_DB_BASE}/users"
    payload_user = {
        "id": user_id, "name": name, "email": email,
        "password_hash": password_hash, "salt": salt, "role": "user",
        "created_at": now, "updated_at": now
    }
    requests.post(url_insert, headers=_sb_headers(), json=payload_user, timeout=10)

    session = _create_session(user_id)
    user_rows = requests.get(f"{SUPABASE_DB_BASE}/users?id=eq.{user_id}", headers=_sb_headers(), timeout=10).json()

    return {
        "ok": True,
        "user": _user_public(user_rows[0]),
        "token": session["token"],
        "expires_at": session["expires_at"],
    }

@app.post("/auth/login")
async def auth_login(payload: AuthLoginRequest):
    if not SUPABASE_DB_BASE: raise HTTPException(status_code=500, detail="Database disabled")
    email = _email_norm(payload.email)
    password = payload.password
    _validate_auth_payload(None, email, password, need_name=False)

    url = f"{SUPABASE_DB_BASE}/users?email=eq.{email}"
    rows = requests.get(url, headers=_sb_headers(), timeout=10).json()

    if not rows:
        raise HTTPException(status_code=401, detail="Неверная почта или пароль")

    user = rows[0]
    expected, _ = _hash_password(password, user["salt"])
    if not secrets.compare_digest(expected, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Неверная почта или пароль")

    session = _create_session(user["id"])
    return {
        "ok": True,
        "user": _user_public(user),
        "token": session["token"],
        "expires_at": session["expires_at"],
    }

@app.post("/auth/google")
async def auth_google(payload: AuthGoogleRequest):
    if not SUPABASE_DB_BASE: raise HTTPException(status_code=500, detail="Database disabled")

    if google_id_token is None or google_requests is None:
        raise HTTPException(status_code=500, detail="google-auth не установлен")

    if not GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=500, detail="GOOGLE_CLIENT_ID не задан")

    try:
        info = google_id_token.verify_oauth2_token(
            payload.id_token,
            google_requests.Request(),
            GOOGLE_CLIENT_ID,
        )
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Google token не прошёл проверку: {e}")

    sub = str(info.get("sub") or "")
    email = _email_norm(str(info.get("email") or payload.email or ""))
    name = str(info.get("name") or payload.name or email.split("@")[0] or "Google user").strip()
    avatar_url = str(info.get("picture") or payload.photo_url or "")

    if not sub or not email:
        raise HTTPException(status_code=401, detail="Google не вернул sub/email")

    now = _now_iso()

    url_check = f"{SUPABASE_DB_BASE}/users?or=(google_sub.eq.{sub},email.eq.{email})"
    rows = requests.get(url_check, headers=_sb_headers(), timeout=10).json()

    if not rows:
        user_id = str(uuid4())
        random_password = secrets.token_urlsafe(24)
        password_hash, salt = _hash_password(random_password)
        
        payload_user = {
            "id": user_id, "name": name, "email": email, "password_hash": password_hash, 
            "salt": salt, "role": "user", "created_at": now, "updated_at": now, 
            "provider": "google", "google_sub": sub, "avatar_url": avatar_url
        }
        requests.post(f"{SUPABASE_DB_BASE}/users", headers=_sb_headers(), json=payload_user, timeout=10)
        user = requests.get(f"{SUPABASE_DB_BASE}/users?id=eq.{user_id}", headers=_sb_headers(), timeout=10).json()[0]
    else:
        user = rows[0]
        update_payload = {
            "name": name or user.get("name"),
            "provider": "google",
            "google_sub": sub or user.get("google_sub"),
            "avatar_url": avatar_url or user.get("avatar_url"),
            "updated_at": now
        }
        url_patch = f"{SUPABASE_DB_BASE}/users?id=eq.{user['id']}"
        resp = requests.patch(url_patch, headers={**_sb_headers(), "Prefer": "return=representation"}, json=update_payload, timeout=10)
        user = resp.json()[0]

    session = _create_session(user["id"])
    return {
        "ok": True,
        "user": _user_public(user),
        "token": session["token"],
        "expires_at": session["expires_at"],
    }

@app.get("/auth/me")
async def auth_me(token: str):
    user = _get_user_by_token(token)
    if user is None:
        raise HTTPException(status_code=401, detail="Сессия не найдена или истекла")
    return {"ok": True, "user": _user_public(user)}

@app.post("/auth/set-role")
async def auth_set_role(payload: AuthRoleRequest):
    user = _get_user_by_token(payload.token)
    if user is None:
        raise HTTPException(status_code=401, detail="Сессия не найдена или истекла")

    role = (payload.role or "").strip().lower()
    if role not in ("user", "admin"):
        raise HTTPException(status_code=400, detail="Недопустимая роль")

    if role == "admin" and payload.admin_code != AUTH_ADMIN_CODE:
        raise HTTPException(status_code=403, detail="Неверный код администратора")

    url_patch = f"{SUPABASE_DB_BASE}/users?id=eq.{user['id']}"
    resp = requests.patch(url_patch, headers={**_sb_headers(), "Prefer": "return=representation"}, json={"role": role, "updated_at": _now_iso()}, timeout=10)
    updated = resp.json()[0]

    return {"ok": True, "user": _user_public(updated)}


def _save_analysis_record(response: dict, user: dict | None):
    if not SUPABASE_DB_BASE: return
    risk = response.get("risk") or {}
    beta = response.get("beta") or {}
    analytic = response.get("analytic_wind_model") or {}
    analytic_out = analytic.get("outputs") or {}
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
    
    url = f"{SUPABASE_DB_BASE}/analyses"
    requests.post(url, headers={**_sb_headers(), "Prefer": "resolution=merge-duplicates"}, json=payload, timeout=10)


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
        "beta_kg_s": row.get("beta_kg_s"),
        "base_moment_nm": row.get("base_moment_nm"),
        "center_of_load_m": row.get("center_of_load_m"),
        "lat": row.get("lat"),
        "lon": row.get("lon"),
        "address": row.get("address"),
    }

@app.get("/analyses/my")
async def analyses_my(token: str, limit: int = 100):
    user = _get_user_by_token(token)
    if user is None:
        raise HTTPException(status_code=401, detail="Сессия не найдена или истекла")

    limit = max(1, min(int(limit), 500))
    url = f"{SUPABASE_DB_BASE}/analyses?user_id=eq.{user['id']}&order=created_at.desc&limit={limit}"
    resp = requests.get(url, headers=_sb_headers(), timeout=10)
    
    return {"ok": True, "items": [_analysis_summary(r) for r in resp.json()]}

@app.get("/analyses/{analysis_id}")
async def analyses_get(analysis_id: str, token: str):
    user = _get_user_by_token(token)
    if user is None:
        raise HTTPException(status_code=401, detail="Сессия не найдена или истекла")

    url = f"{SUPABASE_DB_BASE}/analyses?id=eq.{analysis_id}"
    rows = requests.get(url, headers=_sb_headers(), timeout=10).json()

    if not rows:
        raise HTTPException(status_code=404, detail="Анализ не найден")
    
    row = rows[0]
    if row.get("user_id") != user.get("id") and user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Нет доступа к этому анализу")

    return {"ok": True, "analysis": row.get("response_json")}

@app.get("/admin/analyses")
async def admin_analyses(token: str, limit: int = 200):
    user = _get_user_by_token(token)
    if user is None or user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Нужна роль администратора")

    limit = max(1, min(int(limit), 1000))
    url = f"{SUPABASE_DB_BASE}/analyses?order=created_at.desc&limit={limit}"
    resp = requests.get(url, headers=_sb_headers(), timeout=10)

    return {"ok": True, "items": [_analysis_summary(r) for r in resp.json()]}

@app.get("/profile/stats")
async def profile_stats(token: str):
    user = _get_user_by_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Сессия не найдена или истекла")

    url = f"{SUPABASE_DB_BASE}/analyses?user_id=eq.{user['id']}&order=created_at.desc"
    rows = requests.get(url, headers=_sb_headers(), timeout=10).json()

    total = len(rows)
    with_geo = sum(1 for r in rows if r.get("lat") is not None and r.get("lon") is not None)
    high_risk = sum(1 for r in rows if r.get("risk_category") == "высокий")
    
    risks = [r.get("risk_index") for r in rows if isinstance(r.get("risk_index"), (int, float))]
    avg_risk = sum(risks) / len(risks) if risks else None
    
    last_analysis = _analysis_summary(rows[0]) if rows else None

    return {
        "ok": True,
        "user": _user_public(user),
        "stats": {
            "total_analyses": total,
            "with_geo": with_geo,
            "high_risk_count": high_risk,
            "avg_risk": avg_risk,
            "last_analysis": last_analysis
        }
    }


# --- Training events (in-memory) ---
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
        print(f"[TRAINING_EVENT] {evt['ts']} {evt['level']}: {evt['message']} {evt['data']}")
    except Exception:
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


# === СИНХРОННЫЕ ФУНКЦИИ ДЛЯ РАБОТЫ В ПУЛЕ ПОТОКОВ (РЕШЕНИЕ ЗАВИСАНИЙ) ===
def _run_yolo_sync(img_array):
    tree_model_local = get_tree_model()
    t_res = tree_model_local(img_array)[0]
    s_res = stick_model(img_array)[0]
    return t_res, s_res

def _run_classifier_sync(crop_rgb):
    pil_crop = Image.fromarray(crop_rgb)
    tens = transformer(pil_crop).unsqueeze(0)
    with torch.no_grad():
        pred = classifier(tens)
        cls_id = int(torch.argmax(pred))
    return CLASS_NAMES_RU[cls_id]
# ========================================================================


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
    auth_token: Optional[str] = Form(None),
    lat: Optional[float] = Form(None),
    lon: Optional[float] = Form(None),
):
    analysis_user = _get_user_by_token(auth_token) if auth_token else None

    image_bytes = await file.read()
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Не удалось прочитать изображение")

    H, W = img.shape[:2]

    # -----------------------------
    # АСИНХРОННЫЙ ЗАПУСК YOLO (Фоновый поток)
    # -----------------------------
    tree_res, stick_res = await run_in_threadpool(_run_yolo_sync, img)
    
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
    # CLASSIFIER (Запуск в фоне)
    # -----------------------------
    x1, y1, x2, y2 = tree_res.boxes.xyxy[idx].cpu().numpy().astype(int)
    crop = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    
    species_name = await run_in_threadpool(_run_classifier_sync, crop)

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

    wind_speed_for_model = weather.get("wind_speed") if weather else None
    wind_gust_for_model = weather.get("wind_gust") if weather else None

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

    if risk is not None:
        risk.setdefault("explanation", [])
        beta_value = beta_info.get("beta_kg_s")
        beta_max = beta_info.get("beta_max_scenario")
        
        risk["explanation"].append(
            f"Коэффициент β (ожидаемый): {beta_value:.2f} кг/с ({beta_info.get('source')})"
        )
        if beta_max and beta_max > beta_value:
            risk["explanation"].append(
                f"Коэффициент β (худший сценарий жесткой кроны): до {beta_max:.2f} кг/с"
            )

        if wind_force_n is not None:
            risk["explanation"].append(
                f"Ветровая сила (ожидаемая): {wind_force_n:.1f} Н"
            )
            if beta_max:
                v = weather.get("wind_gust") or weather.get("wind_speed") or 0
                max_force = beta_max * float(v)
                risk["explanation"].append(
                    f"Ветровая сила (порыв + худший β): до {max_force:.1f} Н"
                )

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
        "model_versions": MODEL_VERSIONS,
        "build": BUILD_INFO,
        "schema_version": SCHEMA_VERSION,
        "api_version": API_VERSION,
    }

    # PREPARE PRED OBJECTS
    tree_box_xyxy = tree_res.boxes.xyxy[idx].cpu().numpy().tolist()
    tree_conf = None
    tree_cls_id = None
    try:
        tree_conf = float(tree_res.boxes.conf[idx].cpu().item())
        tree_cls_id = int(tree_res.boxes.cls[idx].cpu().item())
    except Exception:
        pass

    tree_pred = {
        "box_xyxy": tree_box_xyxy,
        "confidence": tree_conf,
        "class_id": tree_cls_id,
    }

    stick_pred = {
        "box_xyxy": None,
        "scale_px_to_m": scale,
    }
    try:
        if len(stick_res.boxes) > 0:
            best = max(stick_res.boxes, key=lambda b: b.xyxy[0][3] - b.xyxy[0][1])
            x1b, y1b, x2b, y2b = best.xyxy[0].cpu().numpy().astype(int)
            stick_pred["box_xyxy"] = [int(x1b), int(y1b), int(x2b), int(y2b)]
            stick_pred["confidence"] = float(best.conf[0].cpu().item())
    except Exception:
        pass

    # SAVE RAW SAMPLE TO SUPABASE
    try:
        supabase_upload_bytes(SUPABASE_BUCKET_RAW, f"{analysis_id}/input.jpg", image_bytes)
        supabase_upload_json(SUPABASE_BUCKET_RAW, f"{analysis_id}/meta_auto.json", meta)
        
        try:
            annotated_bytes_for_raw = base64.b64decode(annotated_b64)
            supabase_upload_bytes(SUPABASE_BUCKET_RAW, f"{analysis_id}/annotated.jpg", annotated_bytes_for_raw)
        except Exception:
            pass

        try:
            supabase_upload_json(SUPABASE_BUCKET_RAW, f"{analysis_id}/tree_pred.json", tree_pred)
            supabase_upload_json(SUPABASE_BUCKET_RAW, f"{analysis_id}/stick_pred.json", stick_pred)
        except Exception:
            pass

    except Exception as e:
        print(f"[!] Failed to upload raw sample {analysis_id} to Supabase: {e}")

    # CACHE IN /tmp FOR FEEDBACK
    try:
        tmp_dir = Path("/tmp") / analysis_id
        tmp_dir.mkdir(parents=True, exist_ok=True)
        with open(tmp_dir / "input.jpg", "wb") as f:
            f.write(image_bytes)
        try:
            annotated_bytes = base64.b64decode(annotated_b64)
            with open(tmp_dir / "annotated.jpg", "wb") as f:
                f.write(annotated_bytes)
        except Exception:
            pass
        with open(tmp_dir / "tree_pred.json", "w", encoding="utf-8") as f:
            json.dump(tree_pred, f, ensure_ascii=False, indent=2)
        with open(tmp_dir / "stick_pred.json", "w", encoding="utf-8") as f:
            json.dump(stick_pred, f, ensure_ascii=False, indent=2)
        with open(tmp_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    except Exception as e:
        print(f"[!] Failed to cache analysis {analysis_id} in /tmp: {e}")

    # RESPONSE
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
    
    try:
        response["original_image_base64"] = encode_jpeg_base64(img.copy(), max_side=1280, quality=72)
    except Exception:
        response["original_image_base64"] = None

    if gps: response["gps"] = gps
    if address: response["address"] = address
    if weather: response["weather"] = weather
    if soil: response["soil"] = soil
    if risk: response["risk"] = risk

    if analysis_user is not None:
        response["user"] = _user_public(analysis_user)
        response["server_saved"] = True
    else:
        response["server_saved"] = False

    try:
        _save_analysis_record(response, analysis_user)
    except Exception as e:
        print(f"[!] Failed to save analysis {analysis_id} to DB: {e}")

    return JSONResponse(response)


@app.post("/feedback")
@app.post("/api/feedback")
def send_feedback(payload: dict = Body(...)):
    analysis_id = payload.get('analysis_id') or payload.get('analysisId')
    if not analysis_id:
        raise HTTPException(status_code=422, detail='analysis_id is required')

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

    if not use_for_training:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass
        return {"status": "ignored", "reason": "user_disabled_training"}

    meta_path = tmp_dir / "meta.json"
    if not meta_path.exists():
        raise HTTPException(status_code=500, detail="meta.json не найден для указанного analysis_id")

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка чтения meta.json: {e}")

    meta["tree_ok"] = tree_ok
    meta["stick_ok"] = stick_ok
    meta["params_ok"] = params_ok
    meta["species_ok"] = species_ok
    meta["correct_species"] = correct_species

    if (not species_ok) and correct_species:
        meta["species"] = correct_species

    if corrected_height_m is not None: meta["height_m"] = corrected_height_m
    if corrected_crown_width_m is not None: meta["crown_width_m"] = corrected_crown_width_m
    if corrected_trunk_diameter_m is not None: meta["trunk_diameter_m"] = corrected_trunk_diameter_m
    if corrected_scale_px_to_m is not None: meta["scale_px_to_m"] = corrected_scale_px_to_m

    trust = 0.0
    if tree_ok: trust += 0.3
    if stick_ok: trust += 0.2
    if params_ok: trust += 0.2
    if species_ok or correct_species: trust += 0.3
    meta["trust_score"] = trust

    is_verified = (use_for_training and trust >= VERIFIED_TRUST_THRESHOLD)

    try:
        input_path = tmp_dir / "input.jpg"
        if input_path.exists():
            supabase_upload_bytes(SUPABASE_BUCKET_INPUTS, f"{analysis_id}/input.jpg", input_path.read_bytes())

        annotated_path = tmp_dir / "annotated.jpg"
        if annotated_path.exists():
            supabase_upload_bytes(SUPABASE_BUCKET_INPUTS, f"{analysis_id}/annotated.jpg", annotated_path.read_bytes())

        existing_has_user_mask = False
        try:
            mv_raw = supabase_download_bytes(SUPABASE_BUCKET_VERIFIED, f"{analysis_id}/meta_verified.json")
            mv = json.loads(mv_raw)
            existing_has_user_mask = bool(mv.get("has_user_mask", False))
        except Exception:
            existing_has_user_mask = False

        meta["has_user_mask"] = existing_has_user_mask

        mask_b64_str = str(user_mask_base64).strip().lower() if user_mask_base64 else ""

        if mask_b64_str and mask_b64_str not in ("null", "undefined"):
            try:
                mask_png_bytes = ensure_png_mask_bytes(str(user_mask_base64))
                supabase_upload_bytes(SUPABASE_BUCKET_INPUTS, f"{analysis_id}/user_mask.png", mask_png_bytes)
                meta["has_user_mask"] = True
            except Exception as e:
                print(f"[!] User mask provided but could not be decoded for {analysis_id}: {e}")

        tree_pred_path = tmp_dir / "tree_pred.json"
        if tree_pred_path.exists():
            supabase_upload_bytes(SUPABASE_BUCKET_PRED, f"{analysis_id}/tree_pred.json", tree_pred_path.read_bytes())

        stick_pred_path = tmp_dir / "stick_pred.json"
        if stick_pred_path.exists():
            supabase_upload_bytes(SUPABASE_BUCKET_PRED, f"{analysis_id}/stick_pred.json", stick_pred_path.read_bytes())

        supabase_upload_json(SUPABASE_BUCKET_META, f"{analysis_id}.json", meta)

        if is_verified:
            try:
                supabase_upload_bytes(SUPABASE_BUCKET_VERIFIED, f"{analysis_id}/input.jpg", (tmp_dir / "input.jpg").read_bytes())
                
                if annotated_path.exists():
                    supabase_upload_bytes(SUPABASE_BUCKET_VERIFIED, f"{analysis_id}/annotated.jpg", annotated_path.read_bytes())

                if user_mask_base64:
                    try:
                        mask_png_bytes = ensure_png_mask_bytes(user_mask_base64)
                        supabase_upload_bytes(SUPABASE_BUCKET_VERIFIED, f"{analysis_id}/user_mask.png", mask_png_bytes)
                    except Exception as e:
                        print(f"[!] Failed to upload VERIFIED user mask for {analysis_id}: {e}")

                supabase_upload_bytes(SUPABASE_BUCKET_VERIFIED, f"{analysis_id}/tree_pred.json", (tmp_dir / "tree_pred.json").read_bytes())
                supabase_upload_bytes(SUPABASE_BUCKET_VERIFIED, f"{analysis_id}/stick_pred.json", (tmp_dir / "stick_pred.json").read_bytes())

                meta_verified = meta.copy()
                meta_verified["verified"] = True
                meta_verified["verified_at"] = datetime.utcnow().isoformat()
                meta_verified["verifier_role"] = "admin" if not use_for_training else "user"

                supabase_upload_json(SUPABASE_BUCKET_VERIFIED, f"{analysis_id}/meta_verified.json", meta_verified)

            except Exception as e:
                print(f"[!] Failed to upload VERIFIED sample {analysis_id}: {e}")

    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при загрузке в Supabase: {e}")

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
            print(f"[!] Queue insert skipped for {analysis_id}: {e}")

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

            if (not include_used) and meta.get("used_for_training") is True:
                continue

            results.append({
                "analysis_id": aid,
                "species": meta.get("species"),
                "risk_category": meta.get("risk", {}).get("category"),
                "trust_score": meta.get("trust_score"),
                "verified": meta.get("verified", True),
                "verified_at": meta.get("verified_at"),
                "exclude_from_training": meta.get("exclude_from_training", False) == True,
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

    meta_path = f"{analysis_id}/meta.json"
    meta = _load_json(meta_path)
    meta["analysis_id"] = analysis_id
    meta["use_for_training"] = flag
    meta["exclude_from_training"] = (not flag)
    supabase_upload_json(SUPABASE_BUCKET_VERIFIED, meta_path, meta)

    mv_path = f"{analysis_id}/meta_verified.json"
    mv = _load_json(mv_path)
    mv["analysis_id"] = analysis_id
    mv["use_for_training"] = flag
    mv["exclude_from_training"] = (not flag)
    supabase_upload_json(SUPABASE_BUCKET_VERIFIED, mv_path, mv)

    return {"analysis_id": analysis_id, "use_for_training": flag, "exclude_from_training": (not flag)}


@app.get("/admin/analysis/{analysis_id}")
def admin_get_analysis(analysis_id: str):
    try:
        input_img = supabase_download_bytes(
            SUPABASE_BUCKET_VERIFIED,
            f"{analysis_id}/input.jpg",
        )
        annotated_img = supabase_download_bytes(
            SUPABASE_BUCKET_VERIFIED,
            f"{analysis_id}/annotated.jpg",
        )

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

@app.get("/admin/training-status")
def admin_training_status():
    training_state_ensure_row()
    state = training_state_get()
    if "active_model_version" not in state or state["active_model_version"] is None:
        state["active_model_version"] = 0
    if "last_model_version" not in state or state["last_model_version"] is None:
        state["last_model_version"] = 0
    if "training_in_progress" not in state or state["training_in_progress"] is None:
        state["training_in_progress"] = False
    if "retrain_requested" not in state or state["retrain_requested"] is None:
        state["retrain_requested"] = False
    return state

@app.post("/admin/set-active-model")
async def admin_set_active_model(payload: dict = Body(...)):
    training_state_ensure_row()
    raw_v = payload.get('version') or payload.get('model_version') or payload.get('active_model_version')
    if raw_v is None:
        raise HTTPException(status_code=422, detail="Missing 'version' in request body")
    v = int(raw_v)

    filename = f"model_v{v}.pt"
    try:
        _ = supabase_download_bytes(SUPABASE_BUCKET_MODELS, filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Model file not found in Supabase Storage: {filename}. {e}")

    training_state_update({"active_model_version": v})

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
    return {"models": list_available_model_versions(), "active_model_version": _get_active_model_version()}

@app.get("/admin/training-events")
def admin_training_events(limit: int = 15):
    limit = max(1, min(int(limit), 200))
    items = list(TRAINING_EVENTS)[-limit:]
    return {"events": list(reversed(items))}

def get_latest_model_path():
    state = training_state_get()
    v = int(state.get("last_model_version", 0))
    if v == 0:
        return "models/base.pt"
    return f"models/model_v{v}.pt"