
import os
import io
import base64
import cv2
import numpy as np
import requests
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont, ExifTags
import torch
from torchvision import models, transforms
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse



# -------------------------------------
# CONFIG
# -------------------------------------



WEATHER_API_KEY = os.getenv("dc825ffd002731568ec7766eafb54bc9", None)
WEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

SOILGRIDS_URL = (
    "https://rest.isric.org/soilgrids/v2.0/properties/query"
)

NOMINATIM_URL = (
    "https://nominatim.openstreetmap.org/reverse"
)
NOMINATIM_USER_AGENT = os.getenv(
    "NOMINATIM_USER_AGENT",
    "arborscan-backend/1.0 (contact: example@mail.com)"
)

ENABLE_ENV_ANALYSIS = os.getenv("ENABLE_ENV_ANALYSIS", "true").lower() == "true"

# -------------------------------------
# CLASSES
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
# EXIF → GPS
# =============================================

def _deg(v):
    d = v[0][0] / v[0][1]
    m = v[1][0] / v[1][1]
    s = v[2][0] / v[2][1]
    return d + m/60 + s/3600

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

        return {
            "lat": lat,
            "lon": lon
        }

    except:
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
    except:
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
            "humidity": main.get("humidity")
        }

    except:
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
            "depth": "0-5cm"
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

    except:
        return None


# =============================================
# Risk Calculation (based on PDFs you uploaded)
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

    s_score = slenderness_score(height, diameter)
    expl.append(f"Коэфф. стройности H/D: {height/diameter if diameter else 0:.1f} → {s_score:.2f}")

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
        "explanation": expl
    }


# =============================================
# DRAW MASK ONLY (NO BBOX / NO TEXT)
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
# MAIN APP
# =============================================

app = FastAPI(title="ArborScan API v2.0")

@app.post("/analyze-tree")
async def analyze_tree(file: UploadFile = File(...)):
    image_bytes = await file.read()
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    H, W = img.shape[:2]

    # ---------------------------------------------
    # YOLO TREE
    # ---------------------------------------------
    tree_res = tree_model(img)[0]
    if tree_res.masks is None:
        return JSONResponse({"error": "Дерево не найдено"}, status_code=400)

    # выбираем самый большой mask
    masks = []
    areas = []
    for i, m in enumerate(tree_res.masks.data):
        mask = (m.cpu().numpy() > 0.5).astype(np.uint8) * 255
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        areas.append(mask.sum())
        masks.append(mask)

    idx = int(np.argmax(areas))
    mask = masks[idx]

    # ---------------------------------------------
    # YOLO STICK
    # ---------------------------------------------
    stick_res = stick_model(img)[0]
    scale = None
    if len(stick_res.boxes) > 0:
        best = max(stick_res.boxes, key=lambda b: b.xyxy[0][3] - b.xyxy[0][1])
        x1, y1, x2, y2 = best.xyxy[0].cpu().numpy().astype(int)
        stick_h = y2 - y1
        if stick_h > 10:
            scale = REAL_STICK_M / stick_h

    # ---------------------------------------------
    # MEASUREMENTS
    # ---------------------------------------------

    ys, xs = np.where(mask > 0)
    y_min, y_max = ys.min(), ys.max()
    height_px = y_max - y_min

    if scale:
        height_m = round(height_px * scale, 2)
    else:
        height_m = None

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

    # ---------------------------------------------
    # CLASSIFIER (RESNET)
    # ---------------------------------------------
    x1, y1, x2, y2 = tree_res.boxes.xyxy[idx].cpu().numpy().astype(int)
    crop = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    pil_crop = Image.fromarray(crop)
    tens = transformer(pil_crop).unsqueeze(0)
    with torch.no_grad():
        pred = classifier(tens)
        cls_id = int(torch.argmax(pred))
    species_name = CLASS_NAMES_RU[cls_id]

    # ---------------------------------------------
    # Annotated image
    # ---------------------------------------------
    annotated_b64 = draw_mask(img.copy(), mask)

    # ---------------------------------------------
    # ENVIRONMENT ANALYSIS
    # ---------------------------------------------

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
            soil
        )

    # ---------------------------------------------
    # RESPONSE
    # ---------------------------------------------

    response = {
        "species": species_name,
        "height_m": height_m,
        "crown_width_m": crown_m,
        "trunk_diameter_m": trunk_m,
        "scale_px_to_m": scale,
        "annotated_image_base64": annotated_b64,
    }

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
