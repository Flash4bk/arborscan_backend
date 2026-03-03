import cv2
import numpy as np
import json
import csv
from pathlib import Path

RAW = Path("raw_data")
OUT = Path("yolo_dataset")

OUT_IMG = OUT / "images/train"
OUT_LBL = OUT / "labels/train"

OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_LBL.mkdir(parents=True, exist_ok=True)

OUT_META = OUT / "meta/train"
OUT_META.mkdir(parents=True, exist_ok=True)

AR_CSV = OUT / "ar_targets.csv"
_ar_rows = []

for sample in RAW.iterdir():
    img = sample / "input.jpg"
    mask = sample / "user_mask.png"

    if not img.exists() or not mask.exists():
        continue

    # --- optional meta (AR participates via metadata/filters) ---
    meta_path = None
    for cand in [sample / "meta_verified.json", sample / "meta.json", sample / "meta_auto.json", sample / "pred.json"]:
        if cand.exists():
            meta_path = cand
            break

    meta = None
    if meta_path:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            # copy meta for later multi-task / filtering
            (OUT_META / f"{sample.name}.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        except Exception:
            meta = None

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


    # collect AR regression targets (optional)
    if isinstance(meta, dict):
        ar = meta.get("ar") if isinstance(meta.get("ar"), dict) else None
        if ar:
            row = {
                "analysis_id": meta.get("analysis_id") or sample.name,
                "sample": sample.name,
                "scale_source": meta.get("scale_source"),
                "ar_points_count": ar.get("points_count"),
                "ar_required_points": ar.get("required_points"),
                "ar_height_m": ar.get("height_m"),
                "ar_trunk_diameter_m": ar.get("trunk_diameter_m"),
                "ar_crown_width_m": ar.get("crown_width_m"),
                "species": meta.get("species"),
            }
            _ar_rows.append(row)


# Write AR targets CSV (only rows with usable AR)
if _ar_rows:
    with open(AR_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(_ar_rows[0].keys()))
        w.writeheader()
        w.writerows(_ar_rows)

print("✅ Dataset built")
