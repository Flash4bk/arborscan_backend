# build_yolo_dataset.py
import cv2
import numpy as np
import os
from tqdm import tqdm

RAW = "raw_data"
OUT = "dataset"
IMG_OUT = f"{OUT}/images/train"
LBL_OUT = f"{OUT}/labels/train"

os.makedirs(IMG_OUT, exist_ok=True)
os.makedirs(LBL_OUT, exist_ok=True)

def mask_to_polygon(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    return c.reshape(-1, 2)

for aid in tqdm(os.listdir(RAW)):
    src = os.path.join(RAW, aid)
    img_path = f"{src}/input.jpg"
    mask_path = f"{src}/user_mask.png"

    if not os.path.exists(mask_path):
        continue

    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    h, w = mask.shape

    poly = mask_to_polygon(mask)
    if poly is None or len(poly) < 6:
        continue

    poly = poly.astype(float)
    poly[:, 0] /= w
    poly[:, 1] /= h

    label = "0 " + " ".join([f"{x:.6f} {y:.6f}" for x, y in poly])

    cv2.imwrite(f"{IMG_OUT}/{aid}.jpg", img)
    with open(f"{LBL_OUT}/{aid}.txt", "w") as f:
        f.write(label)
