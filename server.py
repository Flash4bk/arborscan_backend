import os
import io
import math
import time
import base64
from typing import Optional, Dict, Any, List, Tuple

import cv2
import numpy as np
import requests
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw, ImageFont, ExifTags

import torch
from ultralytics import YOLO

from supabase import create_client, Client

# -----------------------------
#  Конфиг
# -----------------------------

# Модели
CLASSIFIER_PATH = "classifier.pth"
STICK_MODEL_PATH = "stick_model.pt"
TREE_MODEL_PATH = "tree_model.pt"

# API ключи
WEATHER_API_KEY = os.getenv("dc825ffd002731568ec7766eafb54bc9", "")
SOILGRID_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"

# Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://unzwklebqizjqtipnxot.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
SUPABASE_TABLE = "trees_analysis"

supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print(f"[Supabase] init error: {e}")
        supabase = None

# Кэш погоды и почвы
WEATHER_CACHE: Dict[Tuple[float, float], Tuple[float, Dict[str, Any]]] = {}
SOIL_CACHE: Dict[Tuple[float, float], Tuple[float, Dict[str, Any]]] = {}
CACHE_TTL_SEC = 600  # 10 минут

# -----------------------------
#  Инициализация моделей
# -----------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Init] Using device: {device}")

# Классификатор породы дерева (ResNet-18)
TREE_CLASSES = ["Береза", "Дуб", "Ель", "Сосна", "Тополь"]

classifier_model = None
if os.path.exists(CLASSIFIER_PATH):
    try:
        classifier_model = torch.load(CLASSIFIER_PATH, map_location=device)
        classifier_model.eval()
        print("[Init] Classifier loaded")
    except Exception as e:
        print(f"[Init] Failed to load classifier: {e}")
        classifier_model = None
else:
    print("[Init] classifier.pth not found")

# YOLOv8 для палки (детекция)
stick_model = None
if os.path.exists(STICK_MODEL_PATH):
    try:
        stick_model = YOLO(STICK_MODEL_PATH)
        print("[Init] Stick model loaded")
    except Exception as e:
        print(f"[Init] Failed to load stick_model: {e}")
        stick_model = None
else:
    print("[Init] stick_model.pt not found")

# YOLOv8-seg для дерева (сегментация)
tree_model = None
if os.path.exists(TREE_MODEL_PATH):
    try:
        tree_model = YOLO(TREE_MODEL_PATH)
        print("[Init] Tree model loaded")
    except Exception as e:
        print(f"[Init] Failed to load tree_model: {e}")
        tree_model = None
else:
    print("[Init] tree_model.pt not found")

# -----------------------------
#  FastAPI
# -----------------------------

app = FastAPI(title="ArborScan Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


# -----------------------------
#  Вспомогательные функции
# -----------------------------

def pil_to_cv2(img: Image.Image) -> np.ndarray:
    """PIL -> OpenCV BGR"""
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def cv2_to_pil(img: np.ndarray) -> Image.Image:
    """OpenCV BGR -> PIL RGB"""
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def get_font(size: int) -> ImageFont.FreeTypeFont:
    """
    Подбираем шрифт с поддержкой кириллицы.
    На Linux/Railway чаще всего есть DejaVuSans.
    """
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "DejaVuSans.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    # Фолбэк: базовый PIL-шрифт (может не поддерживать кириллицу)
    return ImageFont.load_default()


def extract_gps_from_exif(img: Image.Image) -> Optional[Dict[str, float]]:
    """Извлечение GPS из EXIF, если есть."""
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

        def _convert_to_degrees(value):
            # value: ((num, den), (num, den), (num, den))
            d = value[0][0] / value[0][1]
            m = value[1][0] / value[1][1]
            s = value[2][0] / value[2][1]
            return d + (m / 60.0) + (s / 3600.0)

        lat = None
        lon = None

        gps_lat = gps_info.get(2)
        gps_lat_ref = gps_info.get(1)
        gps_lon = gps_info.get(4)
        gps_lon_ref = gps_info.get(3)

        if gps_lat and gps_lat_ref and gps_lon and gps_lon_ref:
            lat = _convert_to_degrees(gps_lat)
            if gps_lat_ref in ["S", b"S"]:
                lat = -lat
            lon = _convert_to_degrees(gps_lon)
            if gps_lon_ref in ["W", b"W"]:
                lon = -lon

        if lat is not None and lon is not None:
            return {"lat": lat, "lon": lon}
        return None
    except Exception as e:
        print(f"[EXIF] GPS extract error: {e}")
        return None


def reverse_geocode(lat: float, lon: float) -> Optional[str]:
    """Получаем адрес через Nominatim (OSM)."""
    try:
        params = {
            "format": "json",
            "lat": lat,
            "lon": lon,
            "zoom": 18,
            "addressdetails": 1,
        }
        headers = {"User-Agent": "ArborScan/1.0"}
        resp = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        return data.get("display_name")
    except Exception as e:
        print(f"[Reverse geocode] error: {e}")
        return None


def get_weather(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """Погода с кэшем."""
    if not WEATHER_API_KEY:
        return None

    key = (round(lat, 3), round(lon, 3))
    now = time.time()

    if key in WEATHER_CACHE:
        ts, cached = WEATHER_CACHE[key]
        if now - ts < CACHE_TTL_SEC:
            return cached

    try:
        params = {
            "lat": lat,
            "lon": lon,
            "units": "metric",
            "lang": "ru",
            "appid": WEATHER_API_KEY,
        }
        resp = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params=params,
            timeout=10,
        )
        if resp.status_code != 200:
            print(f"[Weather] HTTP {resp.status_code}: {resp.text}")
            return None

        data = resp.json()
        weather = {
            "temperature_c": float(data["main"]["temp"]),
            "humidity_pct": int(data["main"]["humidity"]),
            "pressure_hpa": int(data["main"]["pressure"]),
            "wind_speed_ms": float(data["wind"]["speed"])
            if data.get("wind", {}).get("speed") is not None
            else None,
            "wind_gust_ms": float(data["wind"]["gust"])
            if data.get("wind", {}).get("gust") is not None
            else None,
            "description": data["weather"][0]["description"]
            if data.get("weather")
            else None,
        }
        WEATHER_CACHE[key] = (now, weather)
        return weather
    except Exception as e:
        print(f"[Weather] error: {e}")
        return None


def get_soil(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """Почва с кэшем (SoilGrids)."""
    key = (round(lat, 3), round(lon, 3))
    now = time.time()

    if key in SOIL_CACHE:
        ts, cached = SOIL_CACHE[key]
        if now - ts < CACHE_TTL_SEC:
            return cached

    try:
        params = {
            "lat": lat,
            "lon": lon,
            "property": ["clay", "sand", "silt", "soc", "phh2o"],
            "depth": "0-5cm",
            "value": "mean",
        }
        resp = requests.get(SOILGRID_URL, params=params, timeout=15)
        if resp.status_code != 200:
            print(f"[Soil] HTTP {resp.status_code}: {resp.text}")
            return None

        data = resp.json()
        props = {p["name"]: p for p in data.get("properties", [])}

        def _extract(prop_name: str) -> Optional[float]:
            p = props.get(prop_name)
            if not p:
                return None
            vals = p.get("layers", [])[0].get("values", {})
            mean_val = vals.get("mean")
            if mean_val is None:
                return None
            return float(mean_val)

        clay = _extract("clay")  # %
        sand = _extract("sand")  # %
        silt = _extract("silt")  # %
        soc = _extract("soc")    # g/kg
        ph = _extract("phh2o")   # pH in H2O

        soil = {
            "clay_pct": clay,
            "sand_pct": sand,
            "silt_pct": silt,
            "soc": soc,
            "phh2o": ph,
        }

        SOIL_CACHE[key] = (now, soil)
        return soil
    except Exception as e:
        print(f"[Soil] error: {e}")
        return None


def detect_stick_scale_bymodel(img_bgr: np.ndarray) -> Optional[float]:
    """
    Определяем масштаб по палке длиной 1 метр.
    Возвращаем scale_m_per_px (м/пиксель).
    """
    if stick_model is None:
        print("[Stick] model not loaded")
        return None

    try:
        res = stick_model(img_bgr)[0]
    except Exception as e:
        print(f"[Stick] inference error: {e}")
        return None

    if res.boxes is None or len(res.boxes) == 0:
        print("[Stick] no boxes")
        return None

    # Берём детекцию с наибольшей уверенностью
    boxes = res.boxes
    scores = boxes.conf.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy()
    idx = int(np.argmax(scores))
    x1, y1, x2, y2 = xyxy[idx]

    w = x2 - x1
    h = y2 - y1
    length_px = float(max(w, h))  # длина палки в пикселях по длинной стороне

    if length_px <= 0:
        return None

    scale_m_per_px = 1.0 / length_px  # 1 метр / длину палки
    return scale_m_per_px


def segment_tree_and_measure(img_bgr: np.ndarray, scale_m_per_px: float) -> Dict[str, float]:
    """
    Сегментация дерева YOLOv8-seg и вычисление:
    - высота (м)
    - ширина кроны (м)
    - диаметр ствола (м)
    """
    if tree_model is None:
        raise RuntimeError("Tree model not loaded")

    res = tree_model(img_bgr)[0]
    if res.masks is None or res.boxes is None or len(res.masks) == 0:
        raise RuntimeError("Дерево не найдено")

    masks = res.masks.data.cpu().numpy()  # [N, H, W]
    boxes = res.boxes.xyxy.cpu().numpy()  # [N, 4]

    # Берем маску с максимальной площадью
    areas = [np.sum(m > 0.5) for m in masks]
    idx = int(np.argmax(areas))
    mask = masks[idx] > 0.5
    x1, y1, x2, y2 = boxes[idx]

    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        raise RuntimeError("Маска дерева пустая")

    # Высота = диапазон по Y
    min_y, max_y = ys.min(), ys.max()
    height_px = max_y - min_y

    # Ширина кроны = максимальный горизонтальный разброс маски
    min_x, max_x = xs.min(), xs.max()
    crown_px = max_x - min_x

    # Диаметр ствола: измеряем ширину маски в нижней части (10-15% высоты)
    h = height_px
    if h <= 0:
        raise RuntimeError("Некорректная высота дерева")

    trunk_band_top = max_y - int(h * 0.15)
    trunk_band_bottom = max_y

    band_mask = (ys >= trunk_band_top) & (ys <= trunk_band_bottom)
    band_xs = xs[band_mask]
    band_ys = ys[band_mask]

    trunk_px = 0.0
    if len(band_xs) > 0:
        # Для каждого Y в полосе считаем ширину, берём максимум
        unique_rows = np.unique(band_ys)
        for row in unique_rows:
            row_mask = band_ys == row
            row_xs = band_xs[row_mask]
            width_row = row_xs.max() - row_xs.min()
            if width_row > trunk_px:
                trunk_px = float(width_row)

    # Переводим в метры
    height_m = float(height_px * scale_m_per_px)
    crown_width_m = float(crown_px * scale_m_per_px)
    trunk_diameter_m = float(trunk_px * scale_m_per_px) if trunk_px > 0 else None

    return {
        "height_m": height_m,
        "crown_width_m": crown_width_m,
        "trunk_diameter_m": trunk_diameter_m,
        # Дополнительно возвращаем bbox и маску для рисования
        "bbox": [float(x1), float(y1), float(x2), float(y2)],
        "mask": mask,
    }


def classify_tree_species(img_pil: Image.Image) -> str:
    """Классификация вида дерева (ResNet-18)."""
    if classifier_model is None:
        return "Неизвестный вид"

    try:
        from torchvision import transforms
    except ImportError:
        print("[Classifier] torchvision not installed")
        return "Неизвестный вид"

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    img_t = transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = classifier_model(img_t)
        probs = torch.softmax(logits, dim=1)[0]
        idx = int(torch.argmax(probs).item())
    if 0 <= idx < len(TREE_CLASSES):
        return TREE_CLASSES[idx]
    return "Неизвестный вид"


def compute_risk(
    species: str,
    height_m: Optional[float],
    crown_width_m: Optional[float],
    trunk_diameter_m: Optional[float],
    weather: Optional[Dict[str, Any]],
    soil: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Оценка риска падения дерева.
    Возвращает индекс [0..1], категорию и список текстовых факторов.
    """
    explanations: List[str] = []
    score = 0.0
    weight_sum = 0.0

    # 1) Вид дерева
    species_vuln = {
        "Ель": 0.8,
        "Сосна": 0.7,
        "Тополь": 0.9,
        "Береза": 0.6,
        "Дуб": 0.4,
    }
    base_v = 0.5
    for name, v in species_vuln.items():
        if name in species:
            base_v = v
            break
    score += base_v * 1.0
    weight_sum += 1.0
    explanations.append(f"Порода дерева: {species} (базовая чувствительность {base_v:.2f})")

    # 2) Высота и стройность (H/D)
    if height_m is not None:
        # нормируем: 0м ->0, 30м->1
        h_norm = max(0.0, min(1.0, height_m / 30.0))
        score += h_norm * 1.0
        weight_sum += 1.0
        explanations.append(f"Высота дерева: {height_m:.1f} м")

        if trunk_diameter_m and trunk_diameter_m > 0:
            slenderness = height_m / trunk_diameter_m  # H/D
            # считаем, что 50 = высочайшая стройность, 20 = безопасно
            s_norm = max(0.0, min(1.0, (slenderness - 20.0) / 30.0))
            score += s_norm * 1.0
            weight_sum += 1.0
            explanations.append(
                f"Отношение высоты к диаметру (H/D={slenderness:.1f}) "
                "увеличивает парусность."
            )

    # 3) Погода: скорость ветра и порывы
    if weather:
        wind = weather.get("wind_speed_ms") or 0.0
        gust = weather.get("wind_gust_ms") or wind

        # 0..25 м/с
        wind_norm = max(0.0, min(1.0, gust / 25.0))
        score += wind_norm * 1.2
        weight_sum += 1.2
        explanations.append(
            f"Порывы ветра: {gust:.1f} м/с (усиление нагрузки на крону)."
        )

    # 4) Почва
    if soil:
        clay = soil.get("clay_pct")
        sand = soil.get("sand_pct")
        ph = soil.get("phh2o")
        soc = soil.get("soc")

        # Тип почвы
        soil_type = "неопределённый тип"
        soil_risk = 0.5
        if clay is not None and sand is not None:
            c = clay
            s = sand
            if c > 40:
                soil_type = "тяжёлая глинистая почва"
                soil_risk = 0.7
            elif s > 60:
                soil_type = "лёгкая песчаная почва"
                soil_risk = 0.6
            elif c > 25 and s > 25:
                soil_type = "суглинок"
                soil_risk = 0.5
            else:
                soil_type = "смешанный тип"
                soil_risk = 0.55

        score += soil_risk * 0.8
        weight_sum += 0.8
        explanations.append(f"Тип почвы: {soil_type} (влияние на устойчивость корней).")

        if ph is not None:
            if ph < 5.5:
                explanations.append(
                    f"Кислая почва (pH={ph:.1f}) может ослаблять корневую систему."
                )
            elif ph > 7.5:
                explanations.append(
                    f"Щелочная почва (pH={ph:.1f}) влияет на доступность питательных веществ."
                )
        if soc is not None:
            explanations.append(
                f"Органическое вещество (SOC={soc:.0f}) влияет на структуру и влажность почвы."
            )

    # Нормируем итог
    if weight_sum <= 0:
        index = 0.5
    else:
        index = score / weight_sum
        index = max(0.0, min(1.0, index))

    if index < 0.33:
        category = "низкий"
    elif index < 0.66:
        category = "средний"
    else:
        category = "высокий"

    explanations.append(f"Итоговая оценка риска: {index:.2f} ({category}).")

    return {
        "index": index,
        "category": category,
        "explanation": explanations,
    }


def draw_results_image(
    img_bgr: np.ndarray,
    species: str,
    height_m: Optional[float],
    crown_width_m: Optional[float],
    trunk_diameter_m: Optional[float],
    bbox: Optional[List[float]] = None,
    mask: Optional[np.ndarray] = None,
) -> str:
    """
    Рисуем контуры, bbox и подписи, возвращаем base64 JPEG.
    """
    img_pil = cv2_to_pil(img_bgr).convert("RGBA")
    overlay = Image.new("RGBA", img_pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    w, h = img_pil.size
    font_title = get_font(26)
    font_body = get_font(20)

    # Сегментация дерева (обводка)
    if mask is not None:
        # Маска может иметь размер, отличный от картинки (если модель ресайзила)
        mh, mw = mask.shape
        if (mw, mh) != (w, h):
            mask_resized = Image.fromarray((mask * 255).astype(np.uint8)).resize(
                (w, h), resample=Image.NEAREST
            )
            mask_arr = np.array(mask_resized) > 128
        else:
            mask_arr = mask

        # Рёбра контура через Canny
        mask_uint8 = (mask_arr.astype(np.uint8) * 255)
        edges = cv2.Canny(mask_uint8, 50, 150)
        ys, xs = np.where(edges > 0)
        for x, y0 in zip(xs, ys):
            draw.point((int(x), int(y0)), fill=(0, 255, 0, 255))

    # BBox
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        draw.rectangle(
            [(x1, y1), (x2, y2)],
            outline=(255, 215, 0, 255),
            width=3,
        )

    # Текст слева сверху
    lines = [f"Вид: {species}"]
    if height_m is not None:
        lines.append(f"Высота: {height_m:.2f} м")
    if crown_width_m is not None:
        lines.append(f"Ширина кроны: {crown_width_m:.2f} м")
    if trunk_diameter_m is not None:
        lines.append(f"Диаметр ствола: {trunk_diameter_m:.2f} м")

    # Подложка под текст
    padding = 8
    x_text, y_text = 16, 16

    # Вычислим высоту блока
    text_width = 0
    text_height = 0
    for line in lines:
        bbox_text = font_body.getbbox(line)
        w_line = bbox_text[2] - bbox_text[0]
        h_line = bbox_text[3] - bbox_text[1]
        text_width = max(text_width, w_line)
        text_height += h_line + 4

    rect_coords = [
        x_text - padding,
        y_text - padding,
        x_text + text_width + padding,
        y_text + text_height + padding,
    ]
    draw.rectangle(rect_coords, fill=(0, 0, 0, 140))

    # Рисуем текст
    y_cursor = y_text
    for line in lines:
        draw.text((x_text, y_cursor), line, font=font_body, fill=(255, 255, 255, 255))
        bbox_text = font_body.getbbox(line)
        line_height = bbox_text[3] - bbox_text[1]
        y_cursor += line_height + 4

    result_img = Image.alpha_composite(img_pil, overlay).convert("RGB")

    # В base64
    buffer = io.BytesIO()
    result_img.save(buffer, format="JPEG", quality=90)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return b64


def save_analysis_to_supabase(result: Dict[str, Any], annotated_b64: Optional[str]) -> None:
    """Сохранение анализа в таблицу Supabase."""
    if supabase is None:
        print("[Supabase] client not initialized, skip save")
        return

    try:
        gps = result.get("gps") or {}
        weather = result.get("weather")
        soil = result.get("soil")
        risk = result.get("risk")

        payload = {
            "species": result.get("species"),
            "height": result.get("height_m"),
            "crown": result.get("crown_width_m"),
            "trunk": result.get("trunk_diameter_m"),
            "scale": result.get("scale_m_per_px") or result.get("scale_px_to_m"),
            # Здесь в image_url кладём data URI base64,
            # чтобы потом в приложении можно было легко декодировать.
            "image_url": f"data:image/jpeg;base64,{annotated_b64}"
            if annotated_b64
            else None,
            "gps_lat": gps.get("lat"),
            "gps_lon": gps.get("lon"),
            "address": result.get("address"),
            "weather": weather,
            "soil": soil,
            "risk": risk,
        }

        supabase.table(SUPABASE_TABLE).insert(payload).execute()
        print("[Supabase] analysis saved")
    except Exception as e:
        print(f"[Supabase] save error: {e}")


# -----------------------------
#  FastAPI endpoints
# -----------------------------

@app.get("/")
def root():
    return {"status": "ok", "message": "ArborScan backend running"}


@app.post("/analyze-tree")
async def analyze_tree(file: UploadFile = File(...)):
    # Чтение файла
    try:
        contents = await file.read()
        img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        return JSONResponse(
            {"error": "Не удалось прочитать изображение"}, status_code=400
        )

    img_bgr = pil_to_cv2(img_pil)

    # Масштаб
    scale_m_per_px = None
    if stick_model is not None:
        scale_m_per_px = detect_stick_scale_bymodel(img_bgr)

    if scale_m_per_px is None:
        return JSONResponse(
            {"error": "Эталонная палка не найдена или модель недоступна"},
            status_code=400,
        )

    # Сегментация и измерения
    try:
        measure = segment_tree_and_measure(img_bgr, scale_m_per_px)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    height_m = measure["height_m"]
    crown_width_m = measure["crown_width_m"]
    trunk_diameter_m = measure["trunk_diameter_m"]
    tree_bbox = measure["bbox"]
    tree_mask = measure["mask"]

    # Классификация вида
    species = classify_tree_species(img_pil)

    # GPS
    gps = extract_gps_from_exif(img_pil)
    address = None
    weather = None
    soil = None

    if gps:
        # Адрес
        address = reverse_geocode(gps["lat"], gps["lon"])
        # Погода
        weather = get_weather(gps["lat"], gps["lon"])
        # Почва
        soil = get_soil(gps["lat"], gps["lon"])

    # Риск
    risk = compute_risk(
        species=species,
        height_m=height_m,
        crown_width_m=crown_width_m,
        trunk_diameter_m=trunk_diameter_m,
        weather=weather,
        soil=soil,
    )

    # Аннотированное изображение
    annotated_b64 = draw_results_image(
        img_bgr=img_bgr,
        species=species,
        height_m=height_m,
        crown_width_m=crown_width_m,
        trunk_diameter_m=trunk_diameter_m,
        bbox=tree_bbox,
        mask=tree_mask,
    )

    # Формируем ответ
    result: Dict[str, Any] = {
        "species": species,
        "height_m": height_m,
        "crown_width_m": crown_width_m,
        "trunk_diameter_m": trunk_diameter_m,
        "scale_m_per_px": scale_m_per_px,
        "annotated_image_base64": annotated_b64,
        "gps": gps,
        "address": address,
        "weather": weather,
        "soil": soil,
        "risk": risk,
    }

    # Сохраняем в Supabase (не блокируем ответ, но здесь синхронно; на ранней стадии это ок)
    try:
        save_analysis_to_supabase(result, annotated_b64)
    except Exception as e:
        print(f"[Supabase] error in /analyze-tree: {e}")

    return JSONResponse(result)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
