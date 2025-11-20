import io
import os
import math
import time
import base64
import requests
import numpy as np
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from PIL import Image, ImageDraw, ImageFont, ExifTags
import cv2
import torch
from ultralytics import YOLO


# -------------------------------
#   Модели
# -------------------------------

CLASSIFIER_PATH = "D:\arborscan_backend\models\classifier.pth"
STICK_MODEL_PATH = "D:\arborscan_backend\models\stick_model.pt"
TREE_MODEL_PATH = "D:\arborscan_backend\models\tree_model.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# 🔹 Классификатор (ResNet18)
TREE_CLASSES = ["Береза", "Дуб", "Ель", "Сосна", "Тополь"]
classifier_model = None

if os.path.exists(CLASSIFIER_PATH):
    try:
        classifier_model = torch.load(CLASSIFIER_PATH, map_location=device)
        classifier_model.eval()
        print("[OK] classifier.pth loaded")
    except Exception as e:
        print("[ERR] failed to load classifier:", e)
else:
    print("[ERR] classifier.pth not found")


# 🔹 YOLO — stick
stick_model = None
if os.path.exists(STICK_MODEL_PATH):
    try:
        stick_model = YOLO(STICK_MODEL_PATH)
        print("[OK] stick_model.pt loaded")
    except Exception as e:
        print("[ERR] stick model:", e)
else:
    print("[ERR] stick_model.pt not found")

# 🔹 YOLO — tree seg
tree_model = None
if os.path.exists(TREE_MODEL_PATH):
    try:
        tree_model = YOLO(TREE_MODEL_PATH)
        print("[OK] tree_model.pt loaded")
    except Exception as e:
        print("[ERR] tree model:", e)
else:
    print("[ERR] tree_model.pt not found")


# -------------------------------
#   Утилиты
# -------------------------------

def pil_to_cv2(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def cv2_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def get_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf"
    ]
    for c in candidates:
        if os.path.exists(c):
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()


# -------------------------------
#   GPS из EXIF
# -------------------------------

def extract_gps_from_exif(img: Image.Image) -> Optional[Dict[str, float]]:
    try:
        exif = img._getexif()
        if not exif:
            return None

        exif_data = {
            ExifTags.TAGS.get(k, k): v
            for k, v in exif.items()
            if k in ExifTags.TAGS
        }

        gps_info = exif_data.get("GPSInfo")
        if not gps_info:
            return None

        def to_deg(value):
            d = value[0][0] / value[0][1]
            m = value[1][0] / value[1][1]
            s = value[2][0] / value[2][1]
            return d + m/60 + s/3600

        lat = lon = None

        if 2 in gps_info and 1 in gps_info:
            lat = to_deg(gps_info[2])
            if gps_info[1] == "S":
                lat = -lat

        if 4 in gps_info and 3 in gps_info:
            lon = to_deg(gps_info[4])
            if gps_info[3] == "W":
                lon = -lon

        if lat is not None and lon is not None:
            return {"lat": lat, "lon": lon}
        return None

    except Exception as e:
        print("GPS error:", e)
        return None


# -------------------------------
#   Reverse Geocode
# -------------------------------

def reverse_geocode(lat, lon):
    try:
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {
            "format": "json",
            "lat": lat,
            "lon": lon,
            "zoom": 18,
            "addressdetails": 1
        }
        headers = {"User-Agent": "ArborScan"}
        r = requests.get(url, params=params, headers=headers, timeout=10)

        if r.status_code != 200:
            return None

        return r.json().get("display_name")

    except:
        return None


# -------------------------------
#   Weather
# -------------------------------

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")

def get_weather(lat, lon):
    if not WEATHER_API_KEY:
        return None
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": WEATHER_API_KEY,
            "units": "metric",
            "lang": "ru"
        }
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None

        d = r.json()
        return {
            "temperature_c": d["main"].get("temp"),
            "humidity_pct": d["main"].get("humidity"),
            "pressure_hpa": d["main"].get("pressure"),
            "wind_speed_ms": d["wind"].get("speed"),
            "wind_gust_ms": d["wind"].get("gust"),
            "description": d["weather"][0].get("description") if d.get("weather") else None
        }

    except:
        return None


# -------------------------------
#   SoilGrids
# -------------------------------

SOILGRID_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"

def get_soil(lat, lon):
    try:
        params = {
            "lat": lat,
            "lon": lon,
            "property": ["clay", "sand", "silt", "soc", "phh2o"],
            "depth": "0-5cm",
            "value": "mean"
        }
        r = requests.get(SOILGRID_URL, params=params, timeout=15)
        if r.status_code != 200:
            return None

        data = r.json()
        props = {p["name"]: p for p in data.get("properties", [])}

        def extract(name):
            p = props.get(name)
            if not p:
                return None
            return p["layers"][0]["values"].get("mean")

        return {
            "clay_pct": extract("clay"),
            "sand_pct": extract("sand"),
            "silt_pct": extract("silt"),
            "soc": extract("soc"),
            "phh2o": extract("phh2o")
        }
    except:
        return None


# -------------------------------
#   Детекция палки (масштаб)
# -------------------------------

def detect_stick_scale(img_bgr):
    if stick_model is None:
        return None
    res = stick_model(img_bgr)[0]

    if not res.boxes or len(res.boxes) == 0:
        return None

    b = res.boxes
    scores = b.conf.cpu().numpy()
    xyxy = b.xyxy.cpu().numpy()

    idx = int(np.argmax(scores))
    x1, y1, x2, y2 = xyxy[idx]

    length = max(x2-x1, y2-y1)
    if length <= 0:
        return None

    return 1.0 / float(length)


# -------------------------------
#   Сегментация дерева
# -------------------------------

def segment_measure(img_bgr, scale):
    res = tree_model(img_bgr)[0]
    if res.masks is None or len(res.masks) == 0:
        raise RuntimeError("Дерево не найдено")

    masks = res.masks.data.cpu().numpy()
    boxes = res.boxes.xyxy.cpu().numpy()

    areas = [np.sum(m > 0.5) for m in masks]
    idx = int(np.argmax(areas))
    mask = masks[idx] > 0.5
    x1, y1, x2, y2 = boxes[idx]

    ys, xs = np.where(mask)
    min_y, max_y = ys.min(), ys.max()
    height_px = max_y - min_y

    min_x, max_x = xs.min(), xs.max()
    crown_px = max_x - min_x

    # Trunk
    h = height_px
    tb = max_y - int(h * 0.15)
    trunk_x = xs[(ys >= tb)]
    trunk_px = 0
    if len(trunk_x) > 0:
        trunk_px = trunk_x.max() - trunk_x.min()

    return {
        "height_m": height_px * scale,
        "crown_width_m": crown_px * scale,
        "trunk_diameter_m": trunk_px * scale if trunk_px else None,
        "bbox": [float(x1), float(y1), float(x2), float(y2)],
        "mask": mask
    }


# -------------------------------
#   Классификация вида
# -------------------------------

def classify_tree(img: Image.Image):
    if classifier_model is None:
        return "Неизвестный вид"

    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])

    t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = classifier_model(t)
        idx = int(torch.argmax(out).item())
        if 0 <= idx < len(TREE_CLASSES):
            return TREE_CLASSES[idx]
        return "Неизвестный вид"


# -------------------------------
#   Риск падения
# -------------------------------

def compute_risk(species, height, crown, trunk, weather, soil):
    index = 0.5
    cat = "средний"

    explanations = []

    if weather:
        gust = weather.get("wind_gust_ms") or weather.get("wind_speed_ms", 0)
        if gust is not None:
            index += min(0.5, gust/30)
            explanations.append(f"Порывы ветра: {gust} м/с")

    if soil:
        clay = soil.get("clay_pct")
        sand = soil.get("sand_pct")
        if clay and clay > 40:
            index += 0.1
            explanations.append("Глинистая почва (медленный дренаж)")
        if sand and sand > 60:
            index += 0.1
            explanations.append("Песчаная почва (низкая устойчивость корней)")

    if index < 0.33:
        cat = "низкий"
    elif index > 0.66:
        cat = "высокий"

    return {
        "index": index,
        "category": cat,
        "explanation": explanations
    }


# -------------------------------
#   Рисование результатов
# -------------------------------

def draw_results(img_bgr, species, h, c, t, bbox, mask):
    img_pil = cv2_to_pil(img_bgr).convert("RGBA")
    ov = Image.new("RGBA", img_pil.size, (0,0,0,0))
    draw = ImageDraw.Draw(ov)

    w, h_img = img_pil.size
    font = get_font(22)

    # Mask
    mh, mw = mask.shape
    if (mw, mh) != (w, h_img):
        m = Image.fromarray((mask*255).astype(np.uint8)).resize((w,h_img))
        mask2 = np.array(m) > 128
    else:
        mask2 = mask

    edges = cv2.Canny(mask2.astype(np.uint8)*255, 50,150)
    ys, xs = np.where(edges>0)
    for x,y in zip(xs,ys):
        draw.point((x,y), fill=(0,255,0,255))

    # BBox
    x1,y1,x2,y2 = bbox
    draw.rectangle([(x1,y1),(x2,y2)], outline=(255,215,0,255), width=3)

    # Text
    txt = [
        f"Вид: {species}",
        f"Высота: {h:.2f} м",
        f"Крона: {c:.2f} м",
        f"Ствол: {t:.2f} м" if t else "Ствол: -"
    ]

    y0 = 20
    for line in txt:
        draw.text((20,y0), line, fill="white", font=font)
        y0 += 28

    out = Image.alpha_composite(img_pil, ov).convert("RGB")
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# -------------------------------
#   FastAPI
# -------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.post("/analyze-tree")
async def analyze(file: UploadFile = File(...)):
    try:
        data = await file.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        img_bgr = pil_to_cv2(img)
    except:
        return JSONResponse({"error": "Не удалось прочитать файл"}, status_code=400)

    # Масштаб
    scale = detect_stick_scale(img_bgr)
    if not scale:
        return JSONResponse({"error": "Эталонная палка не найдена"}, status_code=400)

    # Измерения дерева
    try:
        m = segment_measure(img_bgr, scale)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    # Вид
    species = classify_tree(img)

    # GPS
    gps = extract_gps_from_exif(img)
    address = None
    weather = None
    soil = None

    if gps:
        address = reverse_geocode(gps["lat"], gps["lon"])
        weather = get_weather(gps["lat"], gps["lon"])
        soil = get_soil(gps["lat"], gps["lon"])

    # Риск
    risk = compute_risk(
        species,
        m["height_m"],
        m["crown_width_m"],
        m["trunk_diameter_m"],
        weather,
        soil
    )

    # Картинка
    annotated = draw_results(
        img_bgr,
        species,
        m["height_m"],
        m["crown_width_m"],
        m["trunk_diameter_m"],
        m["bbox"],
        m["mask"]
    )

    return {
        "species": species,
        "height_m": m["height_m"],
        "crown_width_m": m["crown_width_m"],
        "trunk_diameter_m": m["trunk_diameter_m"],
        "scale_m_per_px": scale,
        "gps": gps,
        "address": address,
        "weather": weather,
        "soil": soil,
        "risk": risk,
        "annotated_image_base64": annotated
    }


@app.get("/")
def root():
    return {"status": "running"}
