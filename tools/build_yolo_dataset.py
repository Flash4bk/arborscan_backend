import cv2
import numpy as np
from pathlib import Path

RAW = Path("raw_data")
OUT = Path("yolo_dataset")

OUT_IMG = OUT / "images/train"
OUT_LBL = OUT / "labels/train"

OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_LBL.mkdir(parents=True, exist_ok=True)

for sample in RAW.iterdir():
    img = sample / "input.jpg"
    mask = sample / "user_mask.png"

    if not img.exists() or not mask.exists():
        continue

    image = cv2.imread(str(img))
    mask_img = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)

    if mask_img is None:
        print(f"❌ bad mask: {sample.name}")
        continue

    h, w = mask_img.shape

    # contours → YOLO-seg
    contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        continue

    label_lines = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 100:
            continue

        cnt = cnt.squeeze()
        norm = cnt / np.array([[w, h]])

        line = "0 " + " ".join(f"{x:.6f} {y:.6f}" for x, y in norm)
        label_lines.append(line)

    if not label_lines:
        continue

    cv2.imwrite(str(OUT_IMG / f"{sample.name}.jpg"), image)

    with open(OUT_LBL / f"{sample.name}.txt", "w") as f:
        f.write("\n".join(label_lines))

print("✅ Dataset built")
