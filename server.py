import os
import io
import base64
import cv2
import numpy as np
import torch

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont

# =========================
# НАСТРОЙКИ
# =========================
TREE_MODEL_PATH = os.getenv("TREE_MODEL_PATH", "models/tree_model.pt")
STICK_MODEL_PATH = os.getenv("STICK_MODEL_PATH", "models/stick_model.pt")
CLASSIFIER_PATH = os.getenv("CLASSIFIER_PATH", "models/classifier.pth")

CLASS_NAMES_RU = ["Берёза", "Дуб", "Ель", "Сосна", "Тополь"]
REAL_STICK_LENGTH_M = 1.0  # длина палки в метрах

FONT_PATH = "C:/Windows/Fonts/arial.ttf"
# Если шрифт не найдётся, будем использовать по умолчанию

app = FastAPI(title="ArborScan API")


# =========================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =========================
def postprocess_mask(mask: np.ndarray) -> np.ndarray:
    if mask is None or mask.size == 0:
        return None
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255
    if mask.max() == 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return m
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest).astype(np.uint8) * 255


def measure_tree(mask: np.ndarray, meters_per_px: float):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or meters_per_px is None:
        return None, None, None
    y_min, y_max = ys.min(), ys.max()
    height_px = y_max - y_min
    height_m = height_px * meters_per_px

    crown_top = int(y_min)
    crown_bot = int(y_min + 0.7 * height_px)
    crown_w = 0
    for y in range(crown_top, crown_bot):
        row = np.where(mask[y] > 0)[0]
        if len(row) > 0:
            crown_w = max(crown_w, row.max() - row.min())
    crown_m = crown_w * meters_per_px

    trunk_top = int(y_max - 0.2 * height_px)
    trunk_w = []
    for y in range(trunk_top, y_max):
        row = np.where(mask[y] > 0)[0]
        if len(row) > 0:
            width = row.max() - row.min()
            if width > 10:
                trunk_w.append(width)

    trunk_m = (np.mean(trunk_w) * meters_per_px) if trunk_w else None

    height_m = round(height_m, 2)
    crown_m = round(crown_m, 2)
    trunk_m = round(trunk_m, 2) if trunk_m else None

    return height_m, crown_m, trunk_m


def load_classifier(model_path: str, class_names_ru):
    if not os.path.exists(model_path):
        print(f"[!] Классификатор не найден: {model_path}")
        return None, None
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names_ru))
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return model, tfm


def classify_tree(model, tfm, img_bgr, bbox):
    if model is None:
        return None
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = img_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = cv2.cvtColor(img_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    crop = Image.fromarray(crop)
    tens = tfm(crop).unsqueeze(0)
    with torch.no_grad():
        logits = model(tens)
        cls_id = int(torch.argmax(logits, dim=1).item())
    return CLASS_NAMES_RU[cls_id]


def draw_results_image(img_bgr, mask, bbox, species, h_m, cw_m, dbh_m, scale):
    # контуры
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        epsilon = 0.003 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        cv2.drawContours(img_bgr, [approx], -1, (0, 255, 0), 2)

    # панель с текстом
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    font_size = max(18, pil.width // 60)
    try:
        if os.path.exists(FONT_PATH):
            font = ImageFont.truetype(FONT_PATH, font_size)
        else:
            font = ImageFont.load_default()
    except OSError:
        font = ImageFont.load_default()

    lines = [f"Вид: {species or 'не определён'}"]
    if scale:
        lines += [
            f"Высота: {h_m} м" if h_m is not None else "Высота: -",
            f"Ширина кроны: {cw_m} м" if cw_m is not None else "Ширина кроны: -",
            f"Диаметр ствола: {dbh_m} м" if dbh_m is not None else "Диаметр ствола: -",
            f"1 px = {scale:.4f} м",
        ]
    else:
        lines += [
            "⚠ Масштаб по палке не найден",
        ]

    max_text_width = 0
    for line in lines:
        bbox = font.getbbox(line)
        w = bbox[2] - bbox[0]
        max_text_width = max(max_text_width, w)
    panel_width = max(260, max_text_width + 40)
    new_w = pil.width + panel_width
    new_h = pil.height

    new_img = Image.new("RGB", (new_w, new_h), (240, 240, 240))
    new_img.paste(pil, (0, 0))

    draw = ImageDraw.Draw(new_img)
    y_offset = 20
    for line in lines:
        draw.text((pil.width + 20, y_offset), line, fill=(0, 0, 0), font=font)
        y_offset += font_size + 8

    # возвращаем в base64
    buf = io.BytesIO()
    new_img.save(buf, format="JPEG", quality=90)
    img_bytes = buf.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    return img_b64


# =========================
# ЗАГРУЗКА МОДЕЛЕЙ ПРИ СТАРТЕ
# =========================
print("[*] Загрузка YOLO моделей...")
yolo_tree = YOLO(TREE_MODEL_PATH)
yolo_stick = YOLO(STICK_MODEL_PATH)
print("[*] Загрузка классификатора...")
clf_model, clf_tfm = load_classifier(CLASSIFIER_PATH, CLASS_NAMES_RU)
print("[*] Модели загружены.")


# =========================
# API
# =========================
@app.post("/analyze-tree")
async def analyze_tree(file: UploadFile = File(...)):
    # читаем изображение
    img_bytes = await file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Не удалось прочитать изображение"},
        )

    H, W = img_bgr.shape[:2]

    # ---------- дерево ----------
    res_tree = yolo_tree(img_bgr)[0]
    if res_tree.masks is None:
        return {"error": "Дерево не найдено"}

    areas, valid_masks, valid_boxes = [], [], []
    for i, mask_data in enumerate(res_tree.masks.data):
        mask = (mask_data.cpu().numpy() > 0.5).astype(np.uint8) * 255
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        mask = postprocess_mask(mask)
        if mask is not None and mask.max() > 0:
            area = cv2.countNonZero(mask)
            if area < 500:
                continue
            areas.append(area)
            valid_masks.append(mask)
            valid_boxes.append(res_tree.boxes.xyxy[i].cpu().numpy().astype(int))

    if not valid_masks:
        return {"error": "Дерево не найдено"}

    idx = int(np.argmax(areas))
    mask = valid_masks[idx]
    xyxy = valid_boxes[idx]

    # ---------- палка ----------
    scale = None
    stick_res = yolo_stick(img_bgr, conf=0.3)[0]

    if len(stick_res.boxes) > 0:
        best_box = max(
            stick_res.boxes,
            key=lambda b: (b.xyxy[0][3] - b.xyxy[0][1]),
        )
        x1s, y1s, x2s, y2s = best_box.xyxy[0].cpu().numpy().astype(int)
        stick_h = y2s - y1s
        if stick_h > 20:
            scale_tmp = REAL_STICK_LENGTH_M / stick_h
            if 0.001 < scale_tmp < 0.05:
                scale = scale_tmp
                cv2.rectangle(img_bgr, (x1s, y1s), (x2s, y2s), (0, 128, 255), 2)

    meters_per_px = scale if scale else None
    if meters_per_px is not None:
        h_m, cw_m, dbh_m = measure_tree(mask, meters_per_px)
    else:
        h_m = cw_m = dbh_m = None

    # ---------- классификация ----------
    species = classify_tree(clf_model, clf_tfm, img_bgr, xyxy) or "не определён"

    # ---------- картинка с аннотацией ----------
    annotated_b64 = draw_results_image(
        img_bgr.copy(),
        mask,
        xyxy,
        species,
        h_m,
        cw_m,
        dbh_m,
        scale,
    )

    return {
        "species": species,
        "height_m": h_m,
        "crown_width_m": cw_m,
        "trunk_diameter_m": dbh_m,
        "scale_m_per_px": scale,
        "annotated_image_base64": annotated_b64,
    }
