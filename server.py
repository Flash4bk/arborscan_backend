# server.py — FULL VERSION (deploy-ready)
# - FastAPI backend
# - Supabase Storage ingestion (raw/{analysis_id}/...)
# - Feedback with mask -> YOLO bbox label (yolo.txt)
# - Admin queue / dataset build / train (Ultralytics YOLO)
#
# ENV REQUIRED:
#   SUPABASE_URL=https://<project>.supabase.co
#   SUPABASE_SERVICE_KEY=sb_secret_...
# Optional:
#   SUPABASE_BUCKET_INPUTS=arborscan-inputs
#   SUPABASE_BUCKET_META=arborscan-meta
#   SUPABASE_BUCKET_PRED=arborscan-pred
#   SUPABASE_QUEUE_TABLE=training_queue
#   RAW_PREFIX=raw
#   AUTO_UPLOAD_RAW=1

import os
import io
import json
import base64
import shutil
import time
from uuid import uuid4
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import requests
import numpy as np
from PIL import Image, ImageDraw

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Optional ML libs (already in your environment per logs)
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None  # type: ignore


# =========================
# ENV
# =========================

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "").strip()

SUPABASE_BUCKET_INPUTS = os.getenv("SUPABASE_BUCKET_INPUTS", "arborscan-inputs").strip()
SUPABASE_BUCKET_META = os.getenv("SUPABASE_BUCKET_META", "arborscan-meta").strip()
SUPABASE_BUCKET_PRED = os.getenv("SUPABASE_BUCKET_PRED", "arborscan-pred").strip()

SUPABASE_QUEUE_TABLE = os.getenv("SUPABASE_QUEUE_TABLE", "training_queue").strip()
RAW_PREFIX = os.getenv("RAW_PREFIX", "raw").strip()

AUTO_UPLOAD_RAW = os.getenv("AUTO_UPLOAD_RAW", "1").strip() not in ("0", "false", "False", "no", "NO")

# Derived endpoints
SUPABASE_STORAGE_OBJECT_BASE = f"{SUPABASE_URL.rstrip('/')}/storage/v1/object"
SUPABASE_REST_BASE = f"{SUPABASE_URL.rstrip('/')}/rest/v1"

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    # Do not crash import with a stacktrace that hides FastAPI, but fail loudly.
    # Railway will show this immediately if env is missing.
    raise RuntimeError("SUPABASE_URL and/or SUPABASE_SERVICE_KEY are not set")


# =========================
# APP
# =========================

app = FastAPI(title="ArborScan Backend", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# SUPABASE HELPERS
# =========================

def sb_headers_json() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "apikey": SUPABASE_SERVICE_KEY,
        "Content-Type": "application/json",
    }


def sb_headers_storage(content_type: Optional[str] = None, upsert: bool = True) -> Dict[str, str]:
    h = {
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "apikey": SUPABASE_SERVICE_KEY,
    }
    if content_type:
        h["Content-Type"] = content_type
    # Supabase Storage supports upsert via x-upsert
    if upsert:
        h["x-upsert"] = "true"
    return h


def sb_storage_object_url(bucket: str, path: str) -> str:
    # path must be url-safe; we assume simple ascii paths
    return f"{SUPABASE_STORAGE_OBJECT_BASE}/{bucket}/{path}"


def sb_upload_bytes(bucket: str, path: str, data: bytes, content_type: str) -> None:
    url = sb_storage_object_url(bucket, path)
    headers = sb_headers_storage(content_type=content_type, upsert=True)
    r = requests.post(url, headers=headers, data=data, timeout=60)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"Supabase upload error {r.status_code}: {url} -> {r.text}")


def sb_upload_json(bucket: str, path: str, obj: Dict[str, Any]) -> None:
    payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    sb_upload_bytes(bucket, path, payload, "application/json")


def sb_download(bucket: str, path: str) -> bytes:
    url = sb_storage_object_url(bucket, path)
    headers = sb_headers_storage(content_type=None, upsert=False)
    r = requests.get(url, headers=headers, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Supabase download error {r.status_code}: {bucket}/{path} -> {r.text}")
    return r.content


def sb_db_select(table: str, params: Dict[str, str]) -> List[Dict[str, Any]]:
    url = f"{SUPABASE_REST_BASE}/{table}"
    r = requests.get(url, headers=sb_headers_json(), params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Supabase DB select error {r.status_code}: {r.text}")
    return r.json()


def sb_db_insert(table: str, row: Dict[str, Any]) -> None:
    url = f"{SUPABASE_REST_BASE}/{table}"
    headers = sb_headers_json()
    headers["Prefer"] = "return=minimal"
    r = requests.post(url, headers=headers, json=row, timeout=30)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"Supabase DB insert error {r.status_code}: {r.text}")


def sb_db_patch(table: str, match: Dict[str, str], patch: Dict[str, Any]) -> int:
    # match uses PostgREST filters, e.g. {"analysis_id": "eq.<id>"}
    url = f"{SUPABASE_REST_BASE}/{table}"
    headers = sb_headers_json()
    headers["Prefer"] = "return=minimal"
    r = requests.patch(url, headers=headers, params=match, json=patch, timeout=30)
    if r.status_code not in (200, 204):
        raise RuntimeError(f"Supabase DB patch error {r.status_code}: {r.text}")
    # PostgREST does not return affected count by default; return 1 if OK
    return 1


# =========================
# ML LOADING (YOLO + classifier placeholder)
# =========================

YOLO_MODEL = None
CLASSIFIER = None

def load_models() -> None:
    global YOLO_MODEL, CLASSIFIER

    print("[*] Loading YOLO models...")
    if YOLO is not None:
        # Use a lightweight default; ultralytics will download if missing (as in your logs)
        YOLO_MODEL = YOLO("yolov8n.pt")
    else:
        YOLO_MODEL = None
    print("[*] Loading classifier...")
    # Keep classifier as placeholder (you can plug-in your species model later)
    CLASSIFIER = object()
    print("[*] Models loaded.")


# Load at import-time so Railway logs show it once
try:
    load_models()
except Exception as e:
    # Do not crash whole app if model download fails; app can still serve non-ML endpoints
    print(f"[!] Model load failed: {e}")


# =========================
# DATA MODELS (Pydantic)
# =========================

class AnalyzeResult(BaseModel):
    analysis_id: str
    status: str
    meta_path: Optional[str] = None
    raw_upload: Optional[Dict[str, Any]] = None
    model_versions: Optional[Dict[str, str]] = None


class FeedbackRequest(BaseModel):
    analysis_id: str

    use_for_training: bool = True

    tree_ok: bool = True
    stick_ok: bool = True
    params_ok: bool = True
    species_ok: bool = True

    correct_species: Optional[str] = None

    correct_height_m: Optional[float] = None
    correct_crown_width_m: Optional[float] = None
    correct_trunk_diameter_m: Optional[float] = None
    correct_scale_px_to_m: Optional[float] = None

    # base64 WITHOUT data: prefix (client should strip it)
    user_mask_base64: Optional[str] = None


class QueueStatusRequest(BaseModel):
    analysis_id: str
    status: str = Field(..., pattern="^(queued|accepted|rejected)$")


class DatasetBuildRequest(BaseModel):
    dataset_type: str = "yolo_tree"
    limit: int = 100
    note: Optional[str] = None


class TrainRequest(BaseModel):
    dataset_id: str
    train_yolo: bool = True
    train_classifier: bool = False
    epochs: int = 10
    imgsz: int = 416
    batch: int = 2
    note: Optional[str] = None


# =========================
# UTIL: mask -> YOLO bbox
# =========================

def _strip_data_prefix(b64: str) -> str:
    # allow data:image/png;base64,....
    if "," in b64 and b64.lower().startswith("data:"):
        return b64.split(",", 1)[1]
    return b64


def mask_png_bytes_to_yolo_bbox(mask_bytes: bytes) -> Optional[Tuple[float, float, float, float]]:
    """
    Convert user mask PNG to YOLO bbox normalized (xc, yc, w, h).
    Returns None if mask is empty.
    """
    im = Image.open(io.BytesIO(mask_bytes)).convert("RGBA")
    arr = np.array(im)
    # use alpha if present else luminance
    alpha = arr[:, :, 3]
    if alpha is not None:
        mask = alpha > 0
    else:
        mask = arr[:, :, :3].mean(axis=2) > 0

    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())

    w = im.width
    h = im.height

    # bbox in pixel coords
    bw = max(1, x1 - x0 + 1)
    bh = max(1, y1 - y0 + 1)
    xc = x0 + bw / 2.0
    yc = y0 + bh / 2.0

    # normalize
    return (xc / w, yc / h, bw / w, bh / h)


def yolo_txt_from_bbox(class_id: int, bbox: Tuple[float, float, float, float]) -> str:
    xc, yc, w, h = bbox
    # clamp
    xc = float(np.clip(xc, 0.0, 1.0))
    yc = float(np.clip(yc, 0.0, 1.0))
    w = float(np.clip(w, 0.0, 1.0))
    h = float(np.clip(h, 0.0, 1.0))
    return f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n"


# =========================
# ANALYZE PIPELINE (placeholder inference)
# =========================

def infer_tree_and_stick(image_bytes: bytes) -> Dict[str, Any]:
    """
    Placeholder inference:
    - If YOLO_MODEL exists, run a quick predict and return one bbox.
    - Otherwise return stub.
    """
    result: Dict[str, Any] = {
        "tree_pred": {},
        "stick_pred": {},
        "annotated_jpg": None,
    }

    try:
        im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return result

    # Simple annotation placeholder
    annotated = im.copy()
    draw = ImageDraw.Draw(annotated)
    draw.rectangle([10, 10, min(200, im.width - 10), min(200, im.height - 10)], outline="red", width=3)

    buf = io.BytesIO()
    annotated.save(buf, format="JPEG", quality=85)
    result["annotated_jpg"] = buf.getvalue()

    # If YOLO is available, run predict
    if YOLO_MODEL is not None:
        try:
            # ultralytics accepts PIL or np
            yres = YOLO_MODEL.predict(source=np.array(im), imgsz=416, conf=0.25, verbose=False)
            # pick first box if exists
            boxes = []
            if yres and hasattr(yres[0], "boxes") and yres[0].boxes is not None:
                b = yres[0].boxes
                if hasattr(b, "xyxy") and b.xyxy is not None and len(b.xyxy) > 0:
                    for i in range(min(3, len(b.xyxy))):
                        xyxy = b.xyxy[i].cpu().numpy().tolist()
                        boxes.append({"xyxy": xyxy})
            result["tree_pred"] = {"boxes": boxes, "source": "yolo"}
        except Exception as e:
            result["tree_pred"] = {"boxes": [], "source": f"yolo_error:{e}"}
    else:
        result["tree_pred"] = {"boxes": [], "source": "stub_no_yolo"}

    # stick pred placeholder
    result["stick_pred"] = {"boxes": [], "source": "stub"}

    return result


def get_model_versions() -> Dict[str, str]:
    # basic version visibility in meta
    return {
        "yolo": "yolov8n.pt" if YOLO_MODEL is not None else "none",
        "classifier": "stub",
    }


# =========================
# ENDPOINTS
# =========================

@app.post("/analyze-tree", response_model=AnalyzeResult)
async def analyze_tree(file: UploadFile = File(...)) -> AnalyzeResult:
    """
    Receives image, runs inference, stores temp artifacts in /tmp/{analysis_id},
    and uploads RAW artifacts to Supabase Storage under raw/{analysis_id}/...
    """
    analysis_id = str(uuid4())
    tmp_dir = Path("/tmp") / analysis_id
    tmp_dir.mkdir(parents=True, exist_ok=True)

    image_bytes = await file.read()
    input_path = tmp_dir / "input.jpg"
    input_path.write_bytes(image_bytes)

    inf = infer_tree_and_stick(image_bytes)

    annotated_bytes = inf.get("annotated_jpg")
    if annotated_bytes:
        (tmp_dir / "annotated.jpg").write_bytes(annotated_bytes)

    # Save preds locally
    tree_pred_path = tmp_dir / "tree_pred.json"
    stick_pred_path = tmp_dir / "stick_pred.json"
    tree_pred_path.write_text(json.dumps(inf.get("tree_pred", {}), ensure_ascii=False, indent=2), encoding="utf-8")
    stick_pred_path.write_text(json.dumps(inf.get("stick_pred", {}), ensure_ascii=False, indent=2), encoding="utf-8")

    meta: Dict[str, Any] = {
        "analysis_id": analysis_id,
        "created_at": datetime.utcnow().isoformat(),
        "filename": file.filename,
        "content_type": file.content_type,
        "model_versions": get_model_versions(),
        "raw_prefix": RAW_PREFIX,
        "buckets": {
            "inputs": SUPABASE_BUCKET_INPUTS,
            "meta": SUPABASE_BUCKET_META,
            "pred": SUPABASE_BUCKET_PRED,
        },
    }
    (tmp_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    raw_upload_report: Dict[str, Any] = {"ok": True, "uploaded": [], "errors": []}

    if AUTO_UPLOAD_RAW:
        try:
            # Upload input.jpg to inputs bucket
            sb_upload_bytes(SUPABASE_BUCKET_INPUTS, f"{RAW_PREFIX}/{analysis_id}/input.jpg", image_bytes, "image/jpeg")
            raw_upload_report["uploaded"].append(f"{SUPABASE_BUCKET_INPUTS}:{RAW_PREFIX}/{analysis_id}/input.jpg")

            # annotated.jpg
            ann_path = tmp_dir / "annotated.jpg"
            if ann_path.exists():
                sb_upload_bytes(SUPABASE_BUCKET_INPUTS, f"{RAW_PREFIX}/{analysis_id}/annotated.jpg", ann_path.read_bytes(), "image/jpeg")
                raw_upload_report["uploaded"].append(f"{SUPABASE_BUCKET_INPUTS}:{RAW_PREFIX}/{analysis_id}/annotated.jpg")

            # preds to pred bucket under raw/
            sb_upload_json(SUPABASE_BUCKET_PRED, f"{RAW_PREFIX}/{analysis_id}/tree_pred.json", json.loads(tree_pred_path.read_text(encoding="utf-8")))
            raw_upload_report["uploaded"].append(f"{SUPABASE_BUCKET_PRED}:{RAW_PREFIX}/{analysis_id}/tree_pred.json")

            sb_upload_json(SUPABASE_BUCKET_PRED, f"{RAW_PREFIX}/{analysis_id}/stick_pred.json", json.loads(stick_pred_path.read_text(encoding="utf-8")))
            raw_upload_report["uploaded"].append(f"{SUPABASE_BUCKET_PRED}:{RAW_PREFIX}/{analysis_id}/stick_pred.json")

            # meta to meta bucket raw/ and legacy
            sb_upload_json(SUPABASE_BUCKET_META, f"{RAW_PREFIX}/{analysis_id}/meta.json", meta)
            raw_upload_report["uploaded"].append(f"{SUPABASE_BUCKET_META}:{RAW_PREFIX}/{analysis_id}/meta.json")

            sb_upload_json(SUPABASE_BUCKET_META, f"{analysis_id}.json", meta)  # legacy compatibility
            raw_upload_report["uploaded"].append(f"{SUPABASE_BUCKET_META}:{analysis_id}.json")

        except Exception as e:
            raw_upload_report["ok"] = False
            raw_upload_report["errors"].append(str(e))
            print(f"[!] Failed to upload RAW artifacts: {e}")

    return AnalyzeResult(
        analysis_id=analysis_id,
        status="ok",
        meta_path=str(tmp_dir / "meta.json"),
        raw_upload=raw_upload_report,
        model_versions=get_model_versions(),
    )


@app.post("/feedback")
def send_feedback(feedback: FeedbackRequest) -> Dict[str, Any]:
    """
    Saves corrected/confirmed example. RAW example already in raw/{analysis_id}/...
    Here we upload corrected meta + optional user mask + YOLO label derived from mask.
    Also inserts row into Supabase DB queue.
    """
    analysis_id = feedback.analysis_id
    tmp_dir = Path("/tmp") / analysis_id
    if not tmp_dir.exists():
        raise HTTPException(status_code=404, detail="analysis_id not found (expired tmp)")

    meta_path = tmp_dir / "meta.json"
    if not meta_path.exists():
        raise HTTPException(status_code=500, detail="meta.json missing for analysis_id")

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read meta.json: {e}")

    # if user opted out
    if not feedback.use_for_training:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return {"status": "ignored", "reason": "user_disabled_training", "analysis_id": analysis_id}

    # update meta with feedback fields
    meta["tree_ok"] = feedback.tree_ok
    meta["stick_ok"] = feedback.stick_ok
    meta["params_ok"] = feedback.params_ok
    meta["species_ok"] = feedback.species_ok
    meta["correct_species"] = feedback.correct_species
    meta["use_for_training"] = feedback.use_for_training

    if (not feedback.species_ok) and feedback.correct_species:
        meta["species"] = feedback.correct_species

    if feedback.correct_height_m is not None:
        meta["height_m"] = feedback.correct_height_m
    if feedback.correct_crown_width_m is not None:
        meta["crown_width_m"] = feedback.correct_crown_width_m
    if feedback.correct_trunk_diameter_m is not None:
        meta["trunk_diameter_m"] = feedback.correct_trunk_diameter_m
    if feedback.correct_scale_px_to_m is not None:
        meta["scale_px_to_m"] = feedback.correct_scale_px_to_m

    # trust score
    trust = 0.0
    trust += 0.3 if feedback.tree_ok else 0.0
    trust += 0.2 if feedback.stick_ok else 0.0
    trust += 0.2 if feedback.params_ok else 0.0
    trust += 0.3 if (feedback.species_ok or feedback.correct_species) else 0.0
    meta["trust_score"] = trust

    # upload corrected artifacts
    uploaded: List[str] = []
    errors: List[str] = []

    # upload user mask & yolo label
    meta["has_user_mask"] = False

    yolo_txt: Optional[str] = None
    try:
        if feedback.user_mask_base64:
            b64 = _strip_data_prefix(feedback.user_mask_base64)
            mask_bytes = base64.b64decode(b64)

            sb_upload_bytes(SUPABASE_BUCKET_INPUTS, f"{RAW_PREFIX}/{analysis_id}/user_mask.png", mask_bytes, "image/png")
            uploaded.append(f"{SUPABASE_BUCKET_INPUTS}:{RAW_PREFIX}/{analysis_id}/user_mask.png")
            meta["has_user_mask"] = True

            bbox = mask_png_bytes_to_yolo_bbox(mask_bytes)
            if bbox is not None:
                # class_id 0 = tree (keep consistent with your training logs: nc=2 etc.)
                yolo_txt = yolo_txt_from_bbox(0, bbox)
            else:
                # empty mask -> fallback bbox
                yolo_txt = "0 0.5 0.5 0.8 0.8\n"
        else:
            # no mask -> fallback bbox
            yolo_txt = "0 0.5 0.5 0.8 0.8\n"

        if yolo_txt is not None:
            sb_upload_bytes(SUPABASE_BUCKET_INPUTS, f"{RAW_PREFIX}/{analysis_id}/yolo.txt", yolo_txt.encode("utf-8"), "text/plain")
            uploaded.append(f"{SUPABASE_BUCKET_INPUTS}:{RAW_PREFIX}/{analysis_id}/yolo.txt")

    except Exception as e:
        errors.append(f"mask/yolo upload failed: {e}")

    # upload updated meta (raw path + legacy)
    try:
        sb_upload_json(SUPABASE_BUCKET_META, f"{RAW_PREFIX}/{analysis_id}/meta.json", meta)
        uploaded.append(f"{SUPABASE_BUCKET_META}:{RAW_PREFIX}/{analysis_id}/meta.json")
        sb_upload_json(SUPABASE_BUCKET_META, f"{analysis_id}.json", meta)
        uploaded.append(f"{SUPABASE_BUCKET_META}:{analysis_id}.json")
    except Exception as e:
        errors.append(f"meta upload failed: {e}")

    # queue insert (do not fail user if DB insert fails)
    try:
        row = {
            "analysis_id": analysis_id,
            "trust_score": trust,
            "species": meta.get("species"),
            "has_user_mask": meta.get("has_user_mask", False),
            "tree_ok": meta.get("tree_ok"),
            "stick_ok": meta.get("stick_ok"),
            "params_ok": meta.get("params_ok"),
            "species_ok": meta.get("species_ok"),
            "status": "queued",
            "created_at": datetime.utcnow().isoformat(),
        }
        sb_db_insert(SUPABASE_QUEUE_TABLE, row)
    except Exception as e:
        print(f"[!] Failed to insert into queue: {e}")
        errors.append(f"queue insert failed: {e}")

    # cleanup tmp
    try:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass

    return {
        "status": "ok" if not errors else "partial",
        "analysis_id": analysis_id,
        "trust_score": trust,
        "uploaded": uploaded,
        "errors": errors,
    }


# =========================
# ADMIN: training queue
# =========================

@app.get("/admin/training-queue")
def admin_get_training_queue(status: str = "queued", limit: int = 50) -> Dict[str, Any]:
    """
    Returns queue rows from Supabase DB table.
    """
    try:
        rows = sb_db_select(
            SUPABASE_QUEUE_TABLE,
            {
                "status": f"eq.{status}",
                "order": "created_at.desc",
                "limit": str(limit),
                "select": "*",
            },
        )
        return {"status": "ok", "count": len(rows), "rows": rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/training-queue/status")
def admin_set_training_queue_status(req: QueueStatusRequest) -> Dict[str, Any]:
    """
    Update status for analysis_id in training_queue.
    """
    try:
        sb_db_patch(
            SUPABASE_QUEUE_TABLE,
            match={"analysis_id": f"eq.{req.analysis_id}"},
            patch={"status": req.status, "updated_at": datetime.utcnow().isoformat()},
        )
        return {"status": "ok", "analysis_id": req.analysis_id, "new_status": req.status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# ADMIN: dataset build
# =========================

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


@app.post("/admin/dataset/build")
def build_dataset(req: DatasetBuildRequest) -> Dict[str, Any]:
    """
    Build a local dataset in /tmp/datasets/{dataset_id}:
      images/train, labels/train, meta/
      data.yaml
      manifest.json

    Pulls rows from DB where status=accepted (recommended), but if none, uses queued.
    """
    try:
        rows = sb_db_select(
            SUPABASE_QUEUE_TABLE,
            {
                "status": "eq.accepted",
                "order": "created_at.asc",
                "limit": str(req.limit),
                "select": "*",
            },
        )
        if not rows:
            # fallback: queued
            rows = sb_db_select(
                SUPABASE_QUEUE_TABLE,
                {
                    "status": "eq.queued",
                    "order": "created_at.asc",
                    "limit": str(req.limit),
                    "select": "*",
                },
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Queue read failed: {e}")

    if not rows:
        raise HTTPException(status_code=400, detail="No samples in queue")

    dataset_id = str(uuid4())
    base_dir = Path("/tmp/datasets") / dataset_id
    images_dir = base_dir / "images" / "train"
    labels_dir = base_dir / "labels" / "train"
    meta_dir = base_dir / "meta"

    _ensure_dir(images_dir)
    _ensure_dir(labels_dir)
    _ensure_dir(meta_dir)

    manifest: Dict[str, Any] = {
        "dataset_id": dataset_id,
        "dataset_type": req.dataset_type,
        "created_at": datetime.utcnow().isoformat(),
        "total_samples": 0,
        "samples": [],
    }

    usable = 0
    missing: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows, start=1):
        aid = row.get("analysis_id")
        if not aid:
            continue
        fname = f"{idx:06d}"

        # required files:
        img_path = f"{RAW_PREFIX}/{aid}/input.jpg"
        yolo_path = f"{RAW_PREFIX}/{aid}/yolo.txt"

        try:
            img_bytes = sb_download(SUPABASE_BUCKET_INPUTS, img_path)
            lab_bytes = sb_download(SUPABASE_BUCKET_INPUTS, yolo_path)
        except Exception as e:
            missing.append({"analysis_id": aid, "error": str(e)})
            continue

        (images_dir / f"{fname}.jpg").write_bytes(img_bytes)
        (labels_dir / f"{fname}.txt").write_bytes(lab_bytes)

        # meta: try raw then legacy
        meta_bytes = None
        try:
            meta_bytes = sb_download(SUPABASE_BUCKET_META, f"{RAW_PREFIX}/{aid}/meta.json")
        except Exception:
            try:
                meta_bytes = sb_download(SUPABASE_BUCKET_META, f"{aid}.json")
            except Exception:
                meta_bytes = json.dumps({"analysis_id": aid}, ensure_ascii=False).encode("utf-8")

        (meta_dir / f"{fname}.json").write_bytes(meta_bytes)

        manifest["samples"].append({"analysis_id": aid, "file": fname})
        usable += 1

    if usable == 0:
        raise HTTPException(status_code=400, detail={"message": "No usable samples (missing files)", "missing": missing})

    manifest["total_samples"] = usable
    (base_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # YOLO data.yaml
    # class mapping: 0 tree, 1 stick (expand later)
    data_yaml = (
        f"path: {str(base_dir)}\n"
        f"train: images/train\n"
        f"val: images/train\n"
        f"names:\n"
        f"  0: tree\n"
        f"  1: stick\n"
    )
    (base_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")

    return {
        "status": "ok",
        "dataset_id": dataset_id,
        "total_samples": usable,
        "storage": {"type": "local", "path": str(base_dir)},
        "missing": missing,
    }


# =========================
# ADMIN: TRAIN
# =========================

@app.post("/admin/train")
def admin_train(req: TrainRequest) -> Dict[str, Any]:
    """
    Train YOLO on a built dataset. This runs synchronously (simple for now).
    Requires ultralytics installed.
    """
    if not req.train_yolo and not req.train_classifier:
        raise HTTPException(status_code=400, detail="Nothing to train")

    base_dir = Path("/tmp/datasets") / req.dataset_id
    data_yaml = base_dir / "data.yaml"
    if not base_dir.exists() or not data_yaml.exists():
        raise HTTPException(status_code=404, detail="Dataset not found. Build dataset first.")

    result: Dict[str, Any] = {
        "status": "ok",
        "dataset_id": req.dataset_id,
        "train_yolo": req.train_yolo,
        "train_classifier": req.train_classifier,
        "epochs": req.epochs,
        "artifacts": {},
    }

    if req.train_yolo:
        if YOLO is None:
            raise HTTPException(status_code=500, detail="Ultralytics YOLO is not available in this environment")

        try:
            print("[YOLO] Starting training...")
            model = YOLO("yolov8n.pt")
            train_res = model.train(
                data=str(data_yaml),
                epochs=int(req.epochs),
                imgsz=int(req.imgsz),
                batch=int(req.batch),
                device="cpu",
                verbose=True,
            )
            # Save artifacts (local)
            runs_dir = Path("runs") / "detect" / "train"
            best_pt = runs_dir / "weights" / "best.pt"
            last_pt = runs_dir / "weights" / "last.pt"

            result["artifacts"]["yolo_runs_dir"] = str(runs_dir)
            result["artifacts"]["best_pt"] = str(best_pt) if best_pt.exists() else None
            result["artifacts"]["last_pt"] = str(last_pt) if last_pt.exists() else None

            print("[YOLO] Training finished")

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"YOLO training failed: {e}")

    if req.train_classifier:
        # Placeholder for your species classifier pipeline
        # You can later load dataset meta and train your classifier here.
        result["artifacts"]["classifier"] = "stub_not_implemented"

    return result


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "time": datetime.utcnow().isoformat(),
        "env": {
            "SUPABASE_URL": SUPABASE_URL,
            "INPUTS_BUCKET": SUPABASE_BUCKET_INPUTS,
            "META_BUCKET": SUPABASE_BUCKET_META,
            "PRED_BUCKET": SUPABASE_BUCKET_PRED,
            "QUEUE_TABLE": SUPABASE_QUEUE_TABLE,
            "RAW_PREFIX": RAW_PREFIX,
            "AUTO_UPLOAD_RAW": AUTO_UPLOAD_RAW,
        },
        "models": get_model_versions(),
    }
