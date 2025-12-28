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

print("SUPABASE_URL =", SUPABASE_URL)
print("SUPABASE_SERVICE_KEY exists =", bool(SUPABASE_SERVICE_KEY))


BUCKET = "arborscan-verified"

OUT_DIR = Path("dataset_yolov8")
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

    if len(ids) < 5:
        train_ids = set(ids)
    else:
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

        # =====================================================
        # TREE — FROM USER MASK (SEGMENTATION)
        # =====================================================

        try:
            mask_bytes = download(f"{aid}/user_mask.png")
            mask_np = np.frombuffer(mask_bytes, np.uint8)
            mask = cv2.imdecode(mask_np, cv2.IMREAD_GRAYSCALE)

            _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            contours, _ = cv2.findContours(
                mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                # берём самый большой контур (дерево)
                cnt = max(contours, key=cv2.contourArea)
                cnt = cnt.squeeze()

                if len(cnt.shape) == 2 and len(cnt) >= 3:
                    poly_norm = norm_poly(cnt, w, h)
                    labels.append("0 " + " ".join(map(str, poly_norm)))

        except Exception as e:
            print(f"[!] No valid user mask for {aid}: {e}")

        # =====================================================
        # STICK — BBOX (как было)
        # =====================================================

        try:
            stick_pred = json.loads(download(f"{aid}/stick_pred.json"))
            if stick_pred.get("box_xyxy"):
                x1, y1, x2, y2 = stick_pred["box_xyxy"]
                xc = (x1 + x2) / 2 / w
                yc = (y1 + y2) / 2 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                labels.append(f"1 {xc} {yc} {bw} {bh}")
        except Exception:
            pass

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
