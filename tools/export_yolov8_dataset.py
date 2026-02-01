import os
import json
import random
import shutil
from pathlib import Path

import requests
import cv2
import numpy as np

"""
Exporter for YOLOv8 SEGMENTATION dataset.

Key rules for Ultralytics YOLOv8-seg:
- Each label line MUST contain: cls x y w h (bbox) + polygon points (x1 y1 x2 y2 ...), all normalized.
- Dataset MUST be pure segmentation (no bbox-only lines), otherwise Ultralytics drops segments and training crashes.
- We therefore:
  - include ONLY samples that have a valid mask we can convert to a polygon
  - output ONLY the 'tree' class (cls=0)
"""

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

BUCKET = os.getenv("SUPABASE_BUCKET_VERIFIED", "arborscan-verified")

TOOLS_DIR = Path(__file__).resolve().parent
OUT_DIR = TOOLS_DIR / "dataset_yolov8"

TRAIN_SPLIT = float(os.getenv("YOLO_TRAIN_SPLIT", "0.8"))

# We train only tree segmentation here
NAMES = {0: "tree"}


def _headers():
    return {"Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}


def list_objects(prefix=""):
    # Storage REST list
    url = f"{SUPABASE_URL}/storage/v1/object/list/{BUCKET}"
    r = requests.post(url, headers=_headers(), json={"prefix": prefix}, timeout=60)
    r.raise_for_status()
    return r.json()


def download(path: str) -> bytes:
    # Storage REST download (requires service key)
    url = f"{SUPABASE_URL}/storage/v1/object/{BUCKET}/{path}"
    r = requests.get(url, headers=_headers(), timeout=60)
    r.raise_for_status()
    return r.content


def _largest_contour_from_mask(mask_gray: np.ndarray):
    if mask_gray is None:
        return None
    _, mask_bin = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _contour_to_poly(cnt) -> np.ndarray:
    # simplify a bit to avoid huge polygons
    peri = cv2.arcLength(cnt, True)
    eps = 0.002 * peri
    approx = cv2.approxPolyDP(cnt, eps, True)
    pts = approx.squeeze()
    if pts.ndim != 2 or len(pts) < 3:
        return None
    return pts


def _norm(v, denom):
    return float(v) / float(denom) if denom else 0.0


def _yolo_seg_line(cls_id: int, pts: np.ndarray, w: int, h: int) -> str:
    xs = pts[:, 0]
    ys = pts[:, 1]
    x1, x2 = float(xs.min()), float(xs.max())
    y1, y2 = float(ys.min()), float(ys.max())

    # bbox in YOLO format (normalized)
    xc = _norm((x1 + x2) / 2.0, w)
    yc = _norm((y1 + y2) / 2.0, h)
    bw = _norm((x2 - x1), w)
    bh = _norm((y2 - y1), h)

    # polygon normalized
    poly = []
    for x, y in pts:
        poly.append(_norm(x, w))
        poly.append(_norm(y, h))

    parts = [str(cls_id), f"{xc:.6f}", f"{yc:.6f}", f"{bw:.6f}", f"{bh:.6f}"] + [f"{p:.6f}" for p in poly]
    return " ".join(parts)


def main():
    print("SUPABASE_URL =", SUPABASE_URL)
    print("SUPABASE_SERVICE_KEY exists =", bool(SUPABASE_SERVICE_KEY))
    print("BUCKET =", BUCKET)

    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_KEY")

    print("[*] Listing verified samples...")
    objects = list_objects()
    # objects contain names like "<analysis_id>/file"
    ids = sorted({o["name"].split("/")[0] for o in objects if o.get("name")})

    print(f"[*] Found {len(ids)} verified samples (folders)")

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    for split in ["train", "val"]:
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Build list of VALID samples (must have decodable mask -> polygon)
    valid = []
    for aid in ids:
        try:
            mask_bytes = download(f"{aid}/user_mask.png")
            mask_np = np.frombuffer(mask_bytes, np.uint8)
            mask = cv2.imdecode(mask_np, cv2.IMREAD_GRAYSCALE)
            cnt = _largest_contour_from_mask(mask)
            if cnt is None:
                continue
            pts = _contour_to_poly(cnt)
            if pts is None:
                continue
            valid.append(aid)
        except Exception as e:
            # Most common: 400/404 due to invalid URL or missing file
            print(f"[!] No valid user mask for {aid}: {e}")

    print(f"[*] Valid segmentation samples: {len(valid)}")
    if len(valid) < 2:
        raise RuntimeError("Not enough valid segmentation samples to train (need at least 2).")

    random.shuffle(valid)
    if len(valid) < 5:
        train_ids = set(valid)
    else:
        split_idx = max(1, int(len(valid) * TRAIN_SPLIT))
        # ensure at least 1 val sample if possible
        split_idx = min(split_idx, len(valid) - 1)
        train_ids = set(valid[:split_idx])

    # Export images + labels for valid only
    for aid in valid:
        split = "train" if aid in train_ids else "val"

        img_bytes = download(f"{aid}/input.jpg")
        img_path = OUT_DIR / "images" / split / f"{aid}.jpg"
        img_path.write_bytes(img_bytes)

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # mask -> polygon
        mask_bytes = download(f"{aid}/user_mask.png")
        mask_np = np.frombuffer(mask_bytes, np.uint8)
        mask = cv2.imdecode(mask_np, cv2.IMREAD_GRAYSCALE)
        cnt = _largest_contour_from_mask(mask)
        pts = _contour_to_poly(cnt)
        if pts is None:
            # shouldn't happen since we filtered valid, but be safe
            continue

        label_line = _yolo_seg_line(0, pts, w, h)
        (OUT_DIR / "labels" / split / f"{aid}.txt").write_text(label_line + "\n", encoding="utf-8")

    # data.yaml - ABSOLUTE path (most robust in containers)
    out_abs = OUT_DIR.as_posix()
    yaml = f"""path: {out_abs}
train: images/train
val: images/val

names:
  0: tree
"""
    (OUT_DIR / "data.yaml").write_text(yaml, encoding="utf-8")

    print("[✓] YOLOv8 segmentation dataset exported:", OUT_DIR)


if __name__ == "__main__":
    main()
