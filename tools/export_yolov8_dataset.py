import os
import json
import random
import shutil
from pathlib import Path
import requests
import cv2
import numpy as np

# ==============================
# CONFIG
# ==============================

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

BUCKET = "arborscan-verified"

OUT_DIR = Path("dataset_yolov8")
IMG_SIZE = None            # None = оригинальный размер
TRAIN_SPLIT = 0.8

CLASSES = {
    0: "tree",
    1: "stick",
}

# ==============================
# SUPABASE HELPERS
# ==============================

def list_objects(prefix=""):
    url = f"{SUPABASE_URL}/storage/v1/object/list/{BUCKET}"
    headers = {"Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
    payload = {"prefix": prefix}
    r = requests.post(url, headers=headers, json=payload)
    r.raise_for_status()
    return r.json()

def download(path):
    url = f"{SUPABASE_URL}/storage/v1/object/{BUCKET}/{path}"
    headers = {"Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    return r.content

# ==============================
# YOLO HELPERS
# ==============================

def norm_xy(x, y, w, h, img_w, img_h):
    return x / img_w, y / img_h, w / img_w, h / img_h

def norm_poly(points, img_w, img_h):
    out = []
    for x, y in points:
        out.append(x / img_w)
        out.append(y / img_h)
    return out

# ==============================
# MAIN
# ==============================

def main():
    print("[*] Listing verified samples...")
    objects = list_objects()
    ids = sorted({o["name"].split("/")[0] for o in objects})

    print(f"[*] Found {len(ids)} verified samples")

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    for split in ["train", "val"]:
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    random.shuffle(ids)
    split_idx = int(len(ids) * TRAIN_SPLIT)
    train_ids = set(ids[:split_idx])

    for aid in ids:
        split = "train" if aid in train_ids else "val"

        # ---- image
        img_bytes = download(f"{aid}/input.jpg")
        img_path = OUT_DIR / "images" / split / f"{aid}.jpg"
        img_path.write_bytes(img_bytes)

        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        labels = []

        # ---- tree (segmentation)
        tree_pred = json.loads(download(f"{aid}/tree_pred.json"))
        if "mask" in tree_pred:
            poly = tree_pred["mask"]
            poly_norm = norm_poly(poly, w, h)
            labels.append("0 " + " ".join(map(str, poly_norm)))

        # ---- stick (bbox)
        stick_pred = json.loads(download(f"{aid}/stick_pred.json"))
        if stick_pred.get("box_xyxy"):
            x1, y1, x2, y2 = stick_pred["box_xyxy"]
            xc = (x1 + x2) / 2
            yc = (y1 + y2) / 2
            bw = x2 - x1
            bh = y2 - y1
            xc, yc, bw, bh = norm_xy(xc, yc, bw, bh, w, h)
            labels.append(f"1 {xc} {yc} {bw} {bh}")

        label_path = OUT_DIR / "labels" / split / f"{aid}.txt"
        label_path.write_text("\n".join(labels))

    # ---- data.yaml
    yaml = f"""
path: {OUT_DIR.resolve()}
train: images/train
val: images/val

names:
"""
    for i, name in CLASSES.items():
        yaml += f"  {i}: {name}\n"

    (OUT_DIR / "data.yaml").write_text(yaml.strip())

    print("[✓] YOLOv8 dataset exported:", OUT_DIR)

if __name__ == "__main__":
    main()
