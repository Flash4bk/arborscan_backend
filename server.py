import os
import io
import base64
import json
import shutil
import cv2
import numpy as np
import requests
from ultralytics import YOLO
from PIL import Image, ExifTags
import torch
from torchvision import models, transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from uuid import uuid4
from pathlib import Path
from pydantic import BaseModel

# -------------------------------------
# CONFIG
# -------------------------------------

# Supabase config: URL и SERVICE KEY задаём через переменные окружения на Railway
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    print("[!] Warning: SUPABASE_URL or SUPABASE_SERVICE_KEY not set. Raw upload and /feedback will not upload to Supabase.")

# Buckets в Supabase Storage
SUPABASE_BUCKET_INPUTS = "arborscan-inputs"
SUPABASE_BUCKET_PRED = "arborscan-predictions"
SUPABASE_BUCKET_META = "arborscan-meta"

# Таблица в Supabase Postgres для очереди доверенных примеров
# (создаёшь её сам в Supabase SQL, напр. arborscan_feedback_queue)
SUPABASE_DB_BASE = SUPABASE_URL.rstrip("/") + "/rest/v1" if SUPABASE_URL else None
SUPABASE_QUEUE_TABLE = "arborscan_feedback_queue"

# Погода / Почва / Геокодирование
# ВАЖНО: сохраняем обратную совместимость: сначала нормальная переменная WEATHER_API_KEY,
# а если её нет — пробуем старую "dc825..." (как у тебя было), чтобы не сломать деплой.
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY") or os.getenv("dc825ffd002731568ec7766eafb54bc9", None)
WEATHER_BASE_URL = os.getenv("WEATHER_BASE_URL", "https://api.openweathermap.org/data/2.5/weather")

SOILGRIDS_URL = os.getenv("SOILGRIDS_URL", "https://rest.isric.org/soilgrids/v2.0/properties/query")

NOMINATIM_URL = os.getenv("NOMINATIM_URL", "https://nominatim.openstreetmap.org/reverse")
NOMINATIM_USER_AGENT = os.getenv(
    "NOMINATIM_USER_AGENT",
    "arborscan-backend/1.0 (contact: example@mail.com)"
)

ENABLE_ENV_ANALYSIS = os.getenv("ENABLE_ENV_ANALYSIS", "true").lower() == "true"

# Версии моделей (для датасета и отката)
MODEL_VERSION = {
    "tree": os.getenv("TREE_MODEL_VERSION", "v1.0"),
    "stick": os.getenv("STICK_MODEL_VERSION", "v1.0"),
    "classifier": os.getenv("CLASSIFIER_MODEL_VERSION", "v1.0"),
}

# Куда складывать RAW-данные в Supabase (без новых bucket’ов)
# Все анализы будут сохраняться в подпапку raw/{analysis_id}/...
RAW_PREFIX = os.getenv("RAW_PREFIX", "raw")

# -------------------------------------
# CLASSES / CONSTANTS
# -------------------------------------

CLASS_NAMES_RU = ["Береза", "Дуб", "Ель", "Сосна", "Тополь"]
REAL_STICK_M = 1.0

# -------------------------------------
# LOADING MODELS
# -------------------------------------

print("[*] Loading YOLO models...")
tree_model = YOLO("models/tree_model.pt")
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
        return

    url = f"{SUPABASE_URL.rstrip('/')}/storage/v1/object/{bucket}/{path}"

    headers = {
        # 🔴 КРИТИЧЕСКИ ВАЖНО
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "apikey": SUPABASE_SERVICE_KEY,
        "Content-Type": "application/octet-stream",
        "x-upsert": "true",
    }

    r = requests.post(url, headers=headers, data=data, timeout=30)

    if r.status_code >= 400:
        raise RuntimeError(f"Supabase upload error {r.status_code}: {r.text}")


def supabase_upload_json(bucket: str, path: str, obj: dict):
    supabase_upload_bytes(bucket, path, json.dumps(obj).encode("utf-8"))


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

def load_model_from_supabase(bucket: str, path: str) -> bytes:
    url = f"{SUPABASE_URL.rstrip('/')}/storage/v1/object/{bucket}/{path}"
    headers = {
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "apikey": SUPABASE_SERVICE_KEY,
    }
    r = requests.get(url, headers=headers, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Failed to load model {bucket}/{path}: {r.text}")
    return r.content

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
# DRAW MASK ONLY (для аннотации)
# =============================================

def draw_mask(img_bgr, mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        approx = cv2.approxPolyDP(cnt, 0.003 * cv2.arcLength(cnt, True), True)
        cv2.drawContours(img_bgr, [approx], -1, (0, 255, 0), 3)

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


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


app = FastAPI(title="ArborScan API v2.1 (raw dataset + model versions)")

def get_active_model_versions():
    """
    Получить активные версии моделей из Supabase DB
    """
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return MODEL_VERSION  # fallback на env / константы

    url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/model_versions"
    headers = {
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "apikey": SUPABASE_SERVICE_KEY,
    }
    params = {
        "is_active": "eq.true",
        "select": "model_type,version,storage_bucket,storage_path"
    }

    r = requests.get(url, headers=headers, params=params, timeout=10)
    if r.status_code != 200:
        print("[!] Failed to fetch model versions:", r.text)
        return MODEL_VERSION

    data = r.json()
    versions = {}
    for row in data:
        versions[row["model_type"]] = {
            "version": row["version"],
            "bucket": row["storage_bucket"],
            "path": row["storage_path"],
        }

    return versions

def load_active_models():
    versions = get_active_model_versions()

    # TREE
    tb = load_model_from_supabase(versions["tree"]["bucket"], versions["tree"]["path"])
    tpath = "/tmp/tree_model.pt"
    with open(tpath, "wb") as f: f.write(tb)
    tree = YOLO(tpath)

    # STICK
    sb = load_model_from_supabase(versions["stick"]["bucket"], versions["stick"]["path"])
    spath = "/tmp/stick_model.pt"
    with open(spath, "wb") as f: f.write(sb)
    stick = YOLO(spath)

    # CLASSIFIER
    cb = load_model_from_supabase(versions["classifier"]["bucket"], versions["classifier"]["path"])
    cpath = "/tmp/classifier.pth"
    with open(cpath, "wb") as f: f.write(cb)

    clf = models.resnet18(weights=None)
    clf.fc = torch.nn.Linear(clf.fc.in_features, 5)
    clf.load_state_dict(torch.load(cpath, map_location="cpu"))
    clf.eval()

    return tree, stick, clf


@app.post("/analyze-tree")
async def analyze_tree(file: UploadFile = File(...)):
    image_bytes = await file.read()
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Не удалось прочитать изображение")

    H, W = img.shape[:2]

    # -----------------------------
    # YOLO TREE
    # -----------------------------
    tree_res = tree_model(img)[0]
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
        gps = extract_gps(image_bytes)
        if gps:
            address = reverse_geocode(gps["lat"], gps["lon"])
            weather = get_weather(gps["lat"], gps["lon"])
            soil = get_soil(gps["lat"], gps["lon"])

        risk = compute_risk(
            species_name,
            height_m or 0,
            crown_m or 0,
            trunk_m or 0,
            weather,
            soil,
        )

    # -----------------------------
    # ID + META
    # -----------------------------
    analysis_id = str(uuid4())

    meta = {
        "analysis_id": analysis_id,
        "species": species_name,
        "height_m": height_m,
        "crown_width_m": crown_m,
        "trunk_diameter_m": trunk_m,
        "scale_px_to_m": scale,
        "gps": gps,
        "address": address,
        "weather": weather,
        "soil": soil,
        "risk": risk,
        "model_version": get_active_model_versions(),  # ← ВАЖНО
        "raw_prefix": RAW_PREFIX,
            }


    # -----------------------------
    # TEMP CACHE FOR FEEDBACK (локально, как и было)
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
        with open(tmp_dir / "tree_pred.json", "w", encoding="utf-8") as f:
            json.dump(tree_pred, f, ensure_ascii=False, indent=2)

        # Предсказание палки
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

        with open(tmp_dir / "stick_pred.json", "w", encoding="utf-8") as f:
            json.dump(stick_pred, f, ensure_ascii=False, indent=2)

        # Метаданные
        with open(tmp_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    except Exception as e:
        print(f"[!] Failed to cache analysis {analysis_id} in /tmp: {e}")

    # -----------------------------
    # RAW DATASET UPLOAD (НОВОЕ)
    # Сохраняем ВСЕ анализы в Supabase Storage, даже без feedback.
    # Делаем best-effort: если Supabase не настроен — анализ всё равно вернётся.
    # -----------------------------
    if SUPABASE_URL and SUPABASE_SERVICE_KEY:
        try:
            raw_base = f"{RAW_PREFIX}/{analysis_id}"

            # input.jpg
            supabase_upload_bytes(
                SUPABASE_BUCKET_INPUTS,
                f"{raw_base}/input.jpg",
                image_bytes,
            )

            # annotated.jpg
            try:
                annotated_bytes = base64.b64decode(annotated_b64)
                supabase_upload_bytes(
                    SUPABASE_BUCKET_INPUTS,
                    f"{raw_base}/annotated.jpg",
                    annotated_bytes,
                )
            except Exception as e:
                print(f"[!] RAW upload annotated failed for {analysis_id}: {e}")

            # tree_pred.json + stick_pred.json из /tmp (если есть)
            tmp_dir = Path("/tmp") / analysis_id

            tp = tmp_dir / "tree_pred.json"
            if tp.exists():
                supabase_upload_bytes(
                    SUPABASE_BUCKET_PRED,
                    f"{raw_base}/tree_pred.json",
                    tp.read_bytes(),
                )

            sp = tmp_dir / "stick_pred.json"
            if sp.exists():
                supabase_upload_bytes(
                    SUPABASE_BUCKET_PRED,
                    f"{raw_base}/stick_pred.json",
                    sp.read_bytes(),
                )

            # meta.json (в meta bucket — привычный формат)
            supabase_upload_json(
                SUPABASE_BUCKET_META,
                f"{raw_base}/meta.json",
                meta,
            )

        except Exception as e:
            print(f"[!] RAW upload failed for {analysis_id}: {e}")

    # -----------------------------
    # RESPONSE
    # -----------------------------
    response = {
        "analysis_id": analysis_id,
        "species": species_name,
        "height_m": height_m,
        "crown_width_m": crown_m,
        "trunk_diameter_m": trunk_m,
        "scale_px_to_m": scale,
        "annotated_image_base64": annotated_b64,
        "model_version": MODEL_VERSION,
    }

    # добавляем оригинальное изображение
    try:
        response["original_image_base64"] = base64.b64encode(image_bytes).decode("utf-8")
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
def send_feedback(feedback: FeedbackRequest):
    """
    Получаем подтверждение/исправление от пользователя и,
    если всё ок, сохраняем пример в Supabase для будущего обучения моделей
    + кладём запись в очередь доверенных примеров (Supabase DB).

    ВАЖНО: raw-пример уже сохранён в raw/{analysis_id}/...
    Здесь мы сохраняем именно "исправленную/подтверждённую" часть.
    """
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise HTTPException(status_code=500, detail="Supabase не настроен на сервере")

    tmp_dir = Path("/tmp") / feedback.analysis_id
    if not tmp_dir.exists():
        raise HTTPException(status_code=404, detail="analysis_id не найден или истёк срок хранения")

    # Если пользователь не хочет использовать пример в обучении
    if not feedback.use_for_training:
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
    meta["tree_ok"] = feedback.tree_ok
    meta["stick_ok"] = feedback.stick_ok
    meta["params_ok"] = feedback.params_ok
    meta["species_ok"] = feedback.species_ok
    meta["correct_species"] = feedback.correct_species
    meta["use_for_training"] = feedback.use_for_training

    # Исправленный вид дерева
    if (not feedback.species_ok) and feedback.correct_species:
        meta["species"] = feedback.correct_species

    # Исправленные численные параметры (если пришли от клиента)
    if feedback.correct_height_m is not None:
        meta["height_m"] = feedback.correct_height_m
    if feedback.correct_crown_width_m is not None:
        meta["crown_width_m"] = feedback.correct_crown_width_m
    if feedback.correct_trunk_diameter_m is not None:
        meta["trunk_diameter_m"] = feedback.correct_trunk_diameter_m
    if feedback.correct_scale_px_to_m is not None:
        meta["scale_px_to_m"] = feedback.correct_scale_px_to_m

    # Trust score
    trust = 0.0
    if feedback.tree_ok:
        trust += 0.3
    if feedback.stick_ok:
        trust += 0.2
    if feedback.params_ok:
        trust += 0.2
    if feedback.species_ok or feedback.correct_species:
        trust += 0.3
    meta["trust_score"] = trust

    analysis_id = feedback.analysis_id

    # -----------------------------
    # UPLOAD "CORRECTED/TRUSTED" TO SUPABASE STORAGE (как и было)
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

        # user_mask.png
        meta["has_user_mask"] = False
        if feedback.user_mask_base64:
            try:
                mask_bytes = base64.b64decode(feedback.user_mask_base64)
                supabase_upload_bytes(
                    SUPABASE_BUCKET_INPUTS,
                    f"{analysis_id}/user_mask.png",
                    mask_bytes,
                )
                meta["has_user_mask"] = True
            except Exception as e:
                print(f"[!] Failed to decode/upload user mask for {analysis_id}: {e}")

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
        # Оставляем совместимость: в meta bucket как раньше {analysis_id}.json
        supabase_upload_json(
            SUPABASE_BUCKET_META,
            f"{analysis_id}.json",
            meta,
        )

    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при загрузке в Supabase: {e}")

    # -----------------------------
    # Запись в очередь доверенных примеров (Supabase DB)
    # -----------------------------
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
        # Не падаем для пользователя, просто логируем
        print(f"[!] Failed to insert feedback into DB queue for {analysis_id}: {e}")

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



@app.get("/admin/model-versions")
def list_model_versions():
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise HTTPException(status_code=500, detail="Supabase not configured")

    url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/model_versions"
    headers = {
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "apikey": SUPABASE_SERVICE_KEY,
    }
    params = {
        "select": "*",
        "order": "created_at.desc"
    }

    r = requests.get(url, headers=headers, params=params, timeout=10)
    if r.status_code != 200:
        raise HTTPException(status_code=500, detail=r.text)

    return r.json()

class ActivateModelRequest(BaseModel):
    model_type: str   # tree | stick | classifier
    version: str      # v1.1, v2.0, ...

@app.post("/admin/activate-model")
def activate_model(req: ActivateModelRequest):
    headers = {
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "apikey": SUPABASE_SERVICE_KEY,
        "Content-Type": "application/json",
    }

    # деактивируем текущую
    requests.patch(
        f"{SUPABASE_URL.rstrip('/')}/rest/v1/model_versions"
        f"?model_type=eq.{req.model_type}",
        headers=headers,
        json={"is_active": False},
        timeout=10,
    )

    # активируем выбранную
    r = requests.patch(
        f"{SUPABASE_URL.rstrip('/')}/rest/v1/model_versions"
        f"?model_type=eq.{req.model_type}&version=eq.{req.version}",
        headers=headers,
        json={"is_active": True},
        timeout=10,
    )
    if r.status_code not in (200, 204):
        raise HTTPException(status_code=500, detail=r.text)

    # hot-reload
    global tree_model, stick_model, classifier
    tree_model, stick_model, classifier = load_active_models()

    return {"status": "ok", "model_type": req.model_type, "version": req.version}
