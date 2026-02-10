import os
import json
import random
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import requests
import cv2
import numpy as np

"""
Exporter for YOLOv8 SEGMENTATION dataset (tree only) with REPRODUCIBLE selection.

This script exports a dataset strictly from an explicit manifest (list of analysis_id),
instead of scanning the bucket and making implicit choices.

Ultralytics YOLOv8-seg label format (per line):
  cls xc yc w h x1 y1 x2 y2 ... (all normalized)

Key rules (kept from your original script):
- include ONLY samples with a valid user mask that can be converted to a polygon
- output ONLY the 'tree' class (cls=0)
"""

# -----------------------------
# Env / defaults
# -----------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

DEFAULT_BUCKET_VERIFIED = os.getenv("SUPABASE_BUCKET_VERIFIED", "arborscan-verified")
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "dataset_yolov8"

# We train only tree segmentation here
NAMES = {0: "tree"}


def _base_url() -> str:
    if not SUPABASE_URL:
        return ""
    return SUPABASE_URL.rstrip("/")


def _headers() -> dict:
    # For Storage REST API, it's safer to send both Authorization and apikey.
    return {
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "apikey": SUPABASE_SERVICE_KEY,
    }


def download(bucket: str, path: str) -> bytes:
    """
    Storage REST download (private buckets): /object/authenticated
    """
    url = f"{_base_url()}/storage/v1/object/authenticated/{bucket}/{path}"
    r = requests.get(url, headers=_headers(), timeout=60)
    r.raise_for_status()
    return r.content


def _largest_contour_from_mask(mask_gray: np.ndarray, *, min_area: float = 100.0):
    if mask_gray is None:
        return None
    _, mask_bin = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < float(min_area):
        return None
    return cnt


def _contour_to_poly(cnt) -> Optional[np.ndarray]:
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


def _ensure_empty_dir(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def _write_data_yaml(out_dir: Path) -> Path:
    # ABSOLUTE path is most robust in containers
    out_abs = out_dir.resolve().as_posix()
    yaml = f"""path: {out_abs}
train: images/train
val: images/val

names:
  0: tree
"""
    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(yaml, encoding="utf-8")
    return data_yaml


def _load_manifest(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if "selection" not in obj:
        obj["selection"] = {}
    sel = obj["selection"]
    # support both top-level train_ids/val_ids and nested selection.*_ids
    train_ids = sel.get("train_ids") or obj.get("train_ids") or []
    val_ids = sel.get("val_ids") or obj.get("val_ids") or []
    if not isinstance(train_ids, list) or not isinstance(val_ids, list):
        raise RuntimeError("Manifest must contain selection.train_ids and selection.val_ids as lists.")
    sel["train_ids"] = train_ids
    sel["val_ids"] = val_ids
    if "policy" not in obj:
        obj["policy"] = {}
    return obj


def export_from_manifest(
    *,
    bucket_verified: str,
    out_dir: Path,
    manifest_in: Path,
    min_mask_area: float,
) -> Tuple[Path, Path]:
    """
    Returns (manifest_out_path, data_yaml_path)
    """
    manifest = _load_manifest(manifest_in)
    sel = manifest["selection"]

    # Prepare dirs
    _ensure_empty_dir(out_dir)
    for split in ["train", "val"]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    dropped: List[Dict[str, Any]] = []
    kept_train: List[str] = []
    kept_val: List[str] = []

    def _export_one(aid: str, split: str) -> bool:
        # download image
        img_bytes = download(bucket_verified, f"{aid}/input.jpg")
        img_path = out_dir / "images" / split / f"{aid}.jpg"
        img_path.write_bytes(img_bytes)

        img = cv2.imread(str(img_path))
        if img is None:
            dropped.append({"id": aid, "reason": "image_decode_failed"})
            return False
        h, w = img.shape[:2]

        # download + decode mask
        try:
            mask_bytes = download(bucket_verified, f"{aid}/user_mask.png")
        except Exception:
            dropped.append({"id": aid, "reason": "mask_missing"})
            return False

        mask_np = np.frombuffer(mask_bytes, np.uint8)
        mask = cv2.imdecode(mask_np, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            dropped.append({"id": aid, "reason": "mask_decode_failed"})
            return False

        cnt = _largest_contour_from_mask(mask, min_area=min_mask_area)
        if cnt is None:
            dropped.append({"id": aid, "reason": f"mask_no_contour_or_small_area(<{min_mask_area})"})
            return False

        pts = _contour_to_poly(cnt)
        if pts is None:
            dropped.append({"id": aid, "reason": "mask_polygon_invalid"})
            return False

        label_line = _yolo_seg_line(0, pts, w, h)
        (out_dir / "labels" / split / f"{aid}.txt").write_text(label_line + "\n", encoding="utf-8")
        return True

    # Export train
    for aid in sel["train_ids"]:
        if _export_one(str(aid), "train"):
            kept_train.append(str(aid))

    # Export val
    for aid in sel["val_ids"]:
        if _export_one(str(aid), "val"):
            kept_val.append(str(aid))

    # Update manifest with what actually got exported
    sel["train_ids"] = kept_train
    sel["val_ids"] = kept_val
    sel["dropped_ids"] = dropped

    manifest["export"] = {
        "out_dir": out_dir.resolve().as_posix(),
        "train_count": len(kept_train),
        "val_count": len(kept_val),
        "dropped_count": len(dropped),
    }
    manifest["policy"]["min_mask_area"] = float(min_mask_area)

    # Write data.yaml
    data_yaml = _write_data_yaml(out_dir)

    # Write manifest_out.json (canonical name: manifest.json)
    manifest_out = out_dir / "manifest.json"
    manifest_out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # Optional quick stats
    stats = {
        "train": len(kept_train),
        "val": len(kept_val),
        "dropped": len(dropped),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    return manifest_out, data_yaml


def main():
    parser = argparse.ArgumentParser(description="Export reproducible YOLOv8-seg dataset from a manifest.json")
    parser.add_argument("--bucket-verified", default=DEFAULT_BUCKET_VERIFIED)
    parser.add_argument("--manifest-in", required=True, help="Path to manifest_in.json containing selection.train_ids/val_ids")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--min-mask-area", type=float, default=float(os.getenv("MIN_MASK_AREA", "100")))

    args = parser.parse_args()

    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_KEY")

    out_dir = Path(args.out_dir).resolve()
    manifest_in = Path(args.manifest_in).resolve()

    print("SUPABASE_URL =", SUPABASE_URL)
    print("SUPABASE_SERVICE_KEY exists =", bool(SUPABASE_SERVICE_KEY))
    print("BUCKET_VERIFIED =", args.bucket_verified)
    print("OUT_DIR =", out_dir)
    print("MANIFEST_IN =", manifest_in)

    if not manifest_in.exists():
        raise RuntimeError(f"manifest_in not found: {manifest_in}")

    manifest_out, data_yaml = export_from_manifest(
        bucket_verified=args.bucket_verified,
        out_dir=out_dir,
        manifest_in=manifest_in,
        min_mask_area=args.min_mask_area,
    )

    print("[✓] Export complete")
    print("  manifest:", manifest_out)
    print("  data.yaml:", data_yaml)


if __name__ == "__main__":
    main()
