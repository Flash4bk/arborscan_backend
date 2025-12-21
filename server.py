import io
import os
import json
import base64
import shutil
from uuid import uuid4
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ExifTags

import torch
from torchvision import models, transforms
from ultralytics import YOLO

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

from config import (
    GOOGLE_DRIVE_CREDENTIALS,
    GOOGLE_DRIVE_ROOT_FOLDER,
    GOOGLE_DRIVE_UPLOADS_FOLDER,
    GOOGLE_DRIVE_MODELS_FOLDER,
    ENABLE_ENV_ANALYSIS,
)

# =========================================================
# GOOGLE DRIVE INIT
# =========================================================

SCOPES = ["https://www.googleapis.com/auth/drive"]

credentials = service_account.Credentials.from_service_account_file(
    GOOGLE_DRIVE_CREDENTIALS,
    scopes=SCOPES,
)

drive = build("drive", "v3", credentials=credentials)


def get_or_create_folder(name: str, parent_id: str | None = None) -> str:
    query = f"name='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    if parent_id:
        query += f" and '{parent_id}' in parents"

    res = drive.files().list(q=query, fields="files(id)").execute()
    files = res.get("files", [])

    if files:
        return files[0]["id"]

    metadata = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
    }
    if parent_id:
        metadata["parents"] = [parent_id]

    folder = drive.files().create(body=metadata, fields="id").execute()
    return folder["id"]


ROOT_FOLDER_ID = get_or_create_folder(GOOGLE_DRIVE_ROOT_FOLDER)
UPLOADS_FOLDER_ID = get_or_create_folder(GOOGLE_DRIVE_UPLOADS_FOLDER, ROOT_FOLDER_ID)
MODELS_FOLDER_ID = get_or_create_folder(GOOGLE_DRIVE_MODELS_FOLDER, ROOT_FOLDER_ID)


def upload_bytes(data: bytes, name: str, parent_id: str, mime: str):
    media = MediaIoBaseUpload(io.BytesIO(data), mimetype=mime, resumable=False)
    file = drive.files().create(
        body={"name": name, "parents": [parent_id]},
        media_body=media,
        fields="id",
    ).execute()
    return file["id"]


# =========================================================
# MODELS
# =========================================================

tree_model = YOLO("models/tree_model.pt")
stick_model = YOLO("models/stick_model.pt")

classifier = models.resnet18(weights=None)
classifier.fc = torch.nn.Linear(classifier.fc.in_features, 5)
classifier.load_state_dict(torch.load("models/classifier.pth", map_location="cpu"))
classifier.eval()

transformer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

CLASS_NAMES_RU = ["Береза", "Дуб", "Ель", "Сосна", "Тополь"]
REAL_STICK_M = 1.0

# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(title="ArborScan API")

# ---------------------------------------------------------
# REQUEST MODELS
# ---------------------------------------------------------

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


# =========================================================
# ANALYZE
# =========================================================

@app.post("/analyze-tree")
async def analyze_tree(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Invalid image")

    H, W = img.shape[:2]

    tree_res = tree_model(img)[0]
    if tree_res.masks is None:
        raise HTTPException(400, "Tree not found")

    mask = (tree_res.masks.data[0].cpu().numpy() > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

    stick_res = stick_model(img)[0]
    scale = None
    if stick_res.boxes:
        box = max(stick_res.boxes, key=lambda b: b.xyxy[0][3] - b.xyxy[0][1])
        h = box.xyxy[0][3] - box.xyxy[0][1]
        if h > 10:
            scale = REAL_STICK_M / float(h)

    ys, xs = np.where(mask > 0)
    height_px = ys.max() - ys.min()
    height_m = round(height_px * scale, 2) if scale else None

    x1, y1, x2, y2 = tree_res.boxes.xyxy[0].cpu().numpy().astype(int)
    crop = Image.fromarray(cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
    tens = transformer(crop).unsqueeze(0)

    with torch.no_grad():
        cls = torch.argmax(classifier(tens)).item()

    analysis_id = str(uuid4())

    annotated = img.copy()
    cv2.drawContours(annotated, cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0, 255, 0), 3)

    buf = io.BytesIO()
    Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)).save(buf, "JPEG")

    upload_bytes(image_bytes, f"{analysis_id}_input.jpg", UPLOADS_FOLDER_ID, "image/jpeg")
    upload_bytes(buf.getvalue(), f"{analysis_id}_annotated.jpg", UPLOADS_FOLDER_ID, "image/jpeg")

    return {
        "analysis_id": analysis_id,
        "species": CLASS_NAMES_RU[cls],
        "height_m": height_m,
        "scale_px_to_m": scale,
        "original_image_base64": base64.b64encode(image_bytes).decode(),
        "annotated_image_base64": base64.b64encode(buf.getvalue()).decode(),
    }


# =========================================================
# FEEDBACK
# =========================================================

@app.post("/feedback")
def feedback(data: FeedbackRequest):
    if not data.use_for_training:
        return {"status": "ignored"}

    meta = data.dict()
    upload_bytes(
        json.dumps(meta, ensure_ascii=False, indent=2).encode(),
        f"{data.analysis_id}_meta.json",
        UPLOADS_FOLDER_ID,
        "application/json",
    )

    if data.user_mask_base64:
        upload_bytes(
            base64.b64decode(data.user_mask_base64),
            f"{data.analysis_id}_mask.png",
            UPLOADS_FOLDER_ID,
            "image/png",
        )

    return {"status": "ok"}
