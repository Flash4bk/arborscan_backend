"""Build a reproducible Ultralytics YOLO segmentation dataset from ArborScan.

The exporter reads an explicit manifest with ``selection.train_ids`` and
``selection.val_ids`` and downloads only verified ArborScan samples from
Supabase Storage.

IMPORTANT
---------
Ultralytics YOLO segmentation label format is:

    class x1 y1 x2 y2 x3 y3 ...

All polygon coordinates are normalized to [0, 1]. Bounding-box coordinates
MUST NOT be prepended to a segmentation label.

The exporter intentionally fails closed when the resulting dataset is invalid.
This prevents the retraining worker from starting on malformed labels, missing
validation data, mismatched masks, or exact image leakage between train/val.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote

import cv2
import numpy as np
import requests


# ---------------------------------------------------------------------------
# Environment / defaults
# ---------------------------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

DEFAULT_BUCKET_VERIFIED = os.getenv(
    "SUPABASE_BUCKET_VERIFIED",
    "arborscan-verified",
)
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "dataset_yolov8"

# This exporter builds a tree-only segmentation dataset.
CLASS_ID_TREE = 0
CLASS_NAMES = {CLASS_ID_TREE: "tree"}

# UUIDs are expected in practice, but this also permits safe legacy IDs.
_SAFE_ANALYSIS_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,160}$")


# ---------------------------------------------------------------------------
# Supabase Storage helpers
# ---------------------------------------------------------------------------
def _base_url() -> str:
    value = (SUPABASE_URL or "").strip().rstrip("/")
    if not value:
        raise RuntimeError("Missing SUPABASE_URL")
    return value


def _headers() -> Dict[str, str]:
    key = (SUPABASE_SERVICE_KEY or "").strip()
    if not key:
        raise RuntimeError("Missing SUPABASE_SERVICE_KEY")
    return {
        "Authorization": f"Bearer {key}",
        "apikey": key,
    }


def _validate_analysis_id(value: Any) -> str:
    aid = str(value).strip()
    if not aid:
        raise ValueError("analysis_id is empty")
    if aid in {".", ".."} or not _SAFE_ANALYSIS_ID_RE.fullmatch(aid):
        raise ValueError(f"unsafe analysis_id: {aid!r}")
    return aid


def download(bucket: str, object_path: str) -> bytes:
    """Download an object from a private Supabase Storage bucket."""
    bucket = str(bucket).strip()
    if not bucket:
        raise ValueError("bucket is empty")

    encoded_bucket = quote(bucket, safe="-._~")
    encoded_path = "/".join(
        quote(part, safe="-._~") for part in object_path.split("/") if part
    )
    url = (
        f"{_base_url()}/storage/v1/object/authenticated/"
        f"{encoded_bucket}/{encoded_path}"
    )

    response = requests.get(url, headers=_headers(), timeout=(10, 60))
    response.raise_for_status()
    return response.content


# ---------------------------------------------------------------------------
# Image / mask / polygon helpers
# ---------------------------------------------------------------------------
def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _decode_image(data: bytes, flags: int) -> Optional[np.ndarray]:
    if not data:
        return None
    encoded = np.frombuffer(data, dtype=np.uint8)
    if encoded.size == 0:
        return None
    return cv2.imdecode(encoded, flags)


def _largest_contour_from_mask(
    mask_gray: np.ndarray,
    *,
    min_area: float,
) -> Optional[np.ndarray]:
    if mask_gray is None or mask_gray.ndim != 2:
        return None

    # User masks are expected to be binary, but thresholding makes the exporter
    # robust to antialiased PNG edges and non-exact 0/255 values.
    _, mask_bin = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        mask_bin,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if float(cv2.contourArea(contour)) < float(min_area):
        return None
    return contour


def _contour_to_polygon(contour: np.ndarray) -> Optional[np.ndarray]:
    """Convert a contour to a simplified Nx2 float polygon."""
    if contour is None or len(contour) < 3:
        return None

    perimeter = float(cv2.arcLength(contour, True))
    if not math.isfinite(perimeter) or perimeter <= 0:
        return None

    # A small simplification removes redundant contour pixels without changing
    # the geometry materially. If simplification collapses the contour, retry
    # with a smaller epsilon and finally with the original contour points.
    candidates: Sequence[np.ndarray] = (
        cv2.approxPolyDP(contour, 0.002 * perimeter, True),
        cv2.approxPolyDP(contour, 0.001 * perimeter, True),
        contour,
    )

    for candidate in candidates:
        points = np.asarray(candidate).reshape(-1, 2).astype(np.float64)
        if points.shape[0] < 3:
            continue
        if not np.isfinite(points).all():
            continue

        # At least three unique vertices are required for a polygon.
        unique_points = np.unique(points, axis=0)
        if unique_points.shape[0] < 3:
            continue

        area = abs(float(cv2.contourArea(points.astype(np.float32))))
        if not math.isfinite(area) or area <= 0:
            continue
        return points

    return None


# Fidelity is measured against the whole binary source mask, including holes
# and disconnected regions, after serializing and parsing the actual label.
MIN_MASK_POLYGON_IOU = 0.995


def _faithful_polygon(contour, mask, *, width, height):
    source = mask > 127
    perimeter = float(cv2.arcLength(contour, True))
    best_iou = 0.0
    for fraction in (0.002, 0.001, 0.0005, 0.00025, 0.0001, 0.0):
        candidate = (cv2.approxPolyDP(contour, fraction * perimeter, True)
                     if fraction else contour)
        points = np.asarray(candidate).reshape(-1, 2).astype(np.float64)
        if len(points) < 3 or not np.isfinite(points).all():
            continue
        if abs(cv2.contourArea(points.astype(np.float32))) <= 0:
            continue
        line = _yolo_segmentation_line(
            CLASS_ID_TREE, points, width=width, height=height)
        normalized = np.array([float(v) for v in line.split()[1:]]).reshape(-1, 2)
        pixels = np.rint(normalized * [width, height]).astype(np.int32)
        raster = np.zeros(mask.shape, dtype=np.uint8)
        cv2.fillPoly(raster, [pixels], 1)
        result = raster.astype(bool)
        intersection = int(np.count_nonzero(source & result))
        union = int(np.count_nonzero(source | result))
        iou = intersection / union if union else 0.0
        best_iou = max(best_iou, iou)
        if iou >= MIN_MASK_POLYGON_IOU:
            return points, line, {
                "mask_polygon_iou": iou,
                "simplification_epsilon_px": fraction * perimeter,
                "source_mask_foreground_px": int(np.count_nonzero(source)),
                "mask_pixels_lost": int(np.count_nonzero(source & ~result)),
                "mask_pixels_added": int(np.count_nonzero(result & ~source)),
            }
    raise ValueError(
        f"Full source mask cannot be represented faithfully by one polygon: "
        f"best_iou={best_iou:.6f}, required={MIN_MASK_POLYGON_IOU}. "
        "Review holes, disconnected regions or degenerate contours."
    )


def _normalize_polygon(
    points: np.ndarray,
    *,
    width: int,
    height: int,
) -> Optional[np.ndarray]:
    if width <= 0 or height <= 0:
        return None
    if points is None or points.ndim != 2 or points.shape[1] != 2:
        return None
    if points.shape[0] < 3:
        return None

    normalized = points.astype(np.float64).copy()
    normalized[:, 0] /= float(width)
    normalized[:, 1] /= float(height)

    if not np.isfinite(normalized).all():
        return None

    # A contour originating from a mask with exactly the same image dimensions
    # should already be inside [0, 1]. A tiny tolerance is allowed for numeric
    # noise, then values are clipped to the legal YOLO range.
    tolerance = 1e-6
    if (
        float(normalized.min()) < -tolerance
        or float(normalized.max()) > 1.0 + tolerance
    ):
        return None

    normalized = np.clip(normalized, 0.0, 1.0)
    return normalized


def _yolo_segmentation_line(
    class_id: int,
    points: np.ndarray,
    *,
    width: int,
    height: int,
) -> str:
    """Create one valid Ultralytics YOLO segmentation label line.

    Format:
        class x1 y1 x2 y2 x3 y3 ...

    There are deliberately NO bounding-box fields here.
    """
    normalized = _normalize_polygon(points, width=width, height=height)
    if normalized is None:
        raise ValueError("polygon cannot be normalized safely")

    flat = normalized.reshape(-1)
    parts = [str(int(class_id))]
    parts.extend(f"{float(value):.6f}" for value in flat)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Manifest / filesystem helpers
# ---------------------------------------------------------------------------
def _ensure_empty_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _unique_ids(values: Sequence[Any]) -> Tuple[List[str], List[Dict[str, Any]]]:
    output: List[str] = []
    dropped: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for raw in values:
        try:
            aid = _validate_analysis_id(raw)
        except Exception as exc:
            dropped.append(
                {
                    "id": str(raw),
                    "reason": "invalid_analysis_id",
                    "detail": str(exc),
                }
            )
            continue

        if aid in seen:
            dropped.append(
                {
                    "id": aid,
                    "reason": "duplicate_analysis_id_in_split",
                }
            )
            continue

        seen.add(aid)
        output.append(aid)

    return output, dropped


def _load_manifest(path: Path) -> Dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Cannot read manifest JSON: {path}: {exc}") from exc

    if not isinstance(manifest, dict):
        raise RuntimeError("Manifest root must be a JSON object")

    selection = manifest.get("selection")
    if selection is None:
        selection = {}
        manifest["selection"] = selection
    if not isinstance(selection, dict):
        raise RuntimeError("manifest.selection must be an object")

    train_raw = selection.get("train_ids") or manifest.get("train_ids") or []
    val_raw = selection.get("val_ids") or manifest.get("val_ids") or []

    if not isinstance(train_raw, list) or not isinstance(val_raw, list):
        raise RuntimeError(
            "Manifest must contain selection.train_ids and "
            "selection.val_ids as lists"
        )

    train_ids, train_dropped = _unique_ids(train_raw)
    val_ids, val_dropped = _unique_ids(val_raw)

    # The exact same analysis ID must never exist in both splits. Train wins;
    # the duplicated validation ID is removed and documented.
    train_set = set(train_ids)
    clean_val: List[str] = []
    cross_split_dropped: List[Dict[str, Any]] = []
    for aid in val_ids:
        if aid in train_set:
            cross_split_dropped.append(
                {
                    "id": aid,
                    "split": "val",
                    "reason": "analysis_id_already_present_in_train",
                }
            )
        else:
            clean_val.append(aid)

    selection["train_ids"] = train_ids
    selection["val_ids"] = clean_val
    selection["manifest_dropped_ids"] = (
        train_dropped + val_dropped + cross_split_dropped
    )

    policy = manifest.get("policy")
    if policy is None:
        manifest["policy"] = {}
    elif not isinstance(policy, dict):
        raise RuntimeError("manifest.policy must be an object")

    return manifest


def _write_data_yaml(out_dir: Path) -> Path:
    # Absolute path is intentional here: the retraining worker may execute from
    # a different working directory. The manifest records the same path so the
    # generated training artifact remains auditable.
    root = out_dir.resolve().as_posix()
    content = (
        f"path: {root}\n"
        "train: images/train\n"
        "val: images/val\n"
        "\n"
        "names:\n"
        "  0: tree\n"
    )
    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(content, encoding="utf-8")
    return data_yaml


# ---------------------------------------------------------------------------
# Dataset integrity validation
# ---------------------------------------------------------------------------
def _parse_and_validate_label(path: Path) -> None:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise RuntimeError(f"Empty label file: {path}")

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(
            f"Expected exactly one tree instance per label file: {path}; "
            f"got {len(lines)} lines"
        )

    tokens = lines[0].split()
    if len(tokens) < 7:
        raise RuntimeError(
            f"Invalid segmentation label (need class + >=3 XY points): {path}"
        )

    try:
        class_id = int(tokens[0])
    except ValueError as exc:
        raise RuntimeError(f"Invalid class ID in {path}: {tokens[0]!r}") from exc

    if class_id != CLASS_ID_TREE:
        raise RuntimeError(
            f"Unexpected class ID {class_id} in {path}; expected {CLASS_ID_TREE}"
        )

    coordinate_tokens = tokens[1:]
    if len(coordinate_tokens) % 2 != 0:
        raise RuntimeError(
            f"Segmentation coordinates must be XY pairs: {path}"
        )

    try:
        coordinates = [float(value) for value in coordinate_tokens]
    except ValueError as exc:
        raise RuntimeError(f"Non-numeric coordinate in {path}") from exc

    if not all(math.isfinite(value) for value in coordinates):
        raise RuntimeError(f"Non-finite coordinate in {path}")

    if not all(0.0 <= value <= 1.0 for value in coordinates):
        raise RuntimeError(f"Coordinate outside [0, 1] in {path}")


def _dataset_image_hashes(image_dir: Path) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    for path in sorted(image_dir.glob("*.jpg")):
        digest = _sha256_bytes(path.read_bytes())
        hashes[digest] = path.name
    return hashes


def validate_exported_dataset(out_dir: Path) -> Dict[str, Any]:
    """Validate files and ensure there is no exact train/val image leakage."""
    result: Dict[str, Any] = {}

    for split in ("train", "val"):
        image_dir = out_dir / "images" / split
        label_dir = out_dir / "labels" / split

        image_stems = {path.stem for path in image_dir.glob("*.jpg")}
        label_stems = {path.stem for path in label_dir.glob("*.txt")}

        if image_stems != label_stems:
            missing_labels = sorted(image_stems - label_stems)
            missing_images = sorted(label_stems - image_stems)
            raise RuntimeError(
                f"Dataset file mismatch in {split}: "
                f"missing_labels={missing_labels}, "
                f"missing_images={missing_images}"
            )

        if not image_stems:
            raise RuntimeError(
                f"Exported {split} split is empty. Retraining is blocked."
            )

        for label_path in sorted(label_dir.glob("*.txt")):
            _parse_and_validate_label(label_path)

        result[f"{split}_count"] = len(image_stems)

    train_hashes = _dataset_image_hashes(out_dir / "images" / "train")
    val_hashes = _dataset_image_hashes(out_dir / "images" / "val")
    overlap = sorted(set(train_hashes).intersection(val_hashes))

    if overlap:
        details = [
            {
                "sha256": digest,
                "train": train_hashes[digest],
                "val": val_hashes[digest],
            }
            for digest in overlap
        ]
        raise RuntimeError(
            "Exact train/val image leakage detected after export: "
            + json.dumps(details, ensure_ascii=False)
        )

    result["train_val_exact_duplicate_count"] = 0
    result["label_format"] = "ultralytics_yolo_segmentation_polygon_only"
    result["validated"] = True
    return result


# ---------------------------------------------------------------------------
# Main export routine
# ---------------------------------------------------------------------------
def export_from_manifest(
    *,
    bucket_verified: str,
    out_dir: Path,
    manifest_in: Path,
    min_mask_area: float,
) -> Tuple[Path, Path]:
    """Export the dataset and return ``(manifest_out, data_yaml)``."""
    if min_mask_area <= 0:
        raise ValueError("min_mask_area must be > 0")

    manifest = _load_manifest(manifest_in)
    # The old verified directory is not evidence of a revision decision.
    # Production training now consumes immutable snapshots from quality_dataset.
    if manifest.get('purpose') != 'export_smoke_test_only_not_model_evaluation':
        raise ValueError('Legacy verified export is not a reviewed dataset. Use model-quality snapshots; this exporter is smoke-only.')
    selection = manifest["selection"]

    _ensure_empty_dir(out_dir)
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    dropped: List[Dict[str, Any]] = list(
        selection.get("manifest_dropped_ids") or []
    )
    kept: Dict[str, List[str]] = {"train": [], "val": []}
    records: List[Dict[str, Any]] = []

    # Hash -> first exported sample. Train is exported before val, so any exact
    # duplicate appearing later in val is automatically removed rather than
    # leaking into validation.
    seen_image_hashes: Dict[str, Tuple[str, str]] = {}

    def drop(
        aid: str,
        split: str,
        reason: str,
        *,
        detail: Optional[str] = None,
    ) -> bool:
        item: Dict[str, Any] = {
            "id": aid,
            "split": split,
            "reason": reason,
        }
        if detail:
            item["detail"] = detail
        dropped.append(item)
        return False

    def export_one(raw_aid: Any, split: str) -> bool:
        try:
            aid = _validate_analysis_id(raw_aid)
        except Exception as exc:
            return drop(
                str(raw_aid),
                split,
                "invalid_analysis_id",
                detail=str(exc),
            )

        # ------------------------- image -------------------------
        try:
            image_bytes = download(bucket_verified, f"{aid}/input.jpg")
        except Exception as exc:
            return drop(aid, split, "image_download_failed", detail=str(exc))

        image = _decode_image(image_bytes, cv2.IMREAD_COLOR)
        if image is None:
            return drop(aid, split, "image_decode_failed")

        height, width = image.shape[:2]
        if width <= 1 or height <= 1:
            return drop(aid, split, "image_invalid_dimensions")

        image_sha256 = _sha256_bytes(image_bytes)
        duplicate_of = seen_image_hashes.get(image_sha256)
        if duplicate_of is not None:
            first_split, first_aid = duplicate_of
            reason = (
                "exact_image_duplicate_across_splits"
                if first_split != split
                else "exact_image_duplicate_within_split"
            )
            return drop(
                aid,
                split,
                reason,
                detail=f"duplicate_of={first_split}/{first_aid}",
            )

        # -------------------------- mask -------------------------
        try:
            mask_bytes = download(bucket_verified, f"{aid}/user_mask.png")
        except Exception as exc:
            return drop(aid, split, "mask_download_failed", detail=str(exc))

        mask = _decode_image(mask_bytes, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return drop(aid, split, "mask_decode_failed")

        mask_height, mask_width = mask.shape[:2]
        if (mask_width, mask_height) != (width, height):
            # Do not resize silently. A mismatch means that the mask/image
            # alignment is not trustworthy and would corrupt training labels.
            return drop(
                aid,
                split,
                "mask_image_dimension_mismatch",
                detail=(
                    f"image={width}x{height}, "
                    f"mask={mask_width}x{mask_height}"
                ),
            )

        contour = _largest_contour_from_mask(
            mask,
            min_area=min_mask_area,
        )
        if contour is None:
            return drop(
                aid,
                split,
                "mask_no_contour_or_too_small",
                detail=f"min_mask_area={min_mask_area}",
            )

        try:
            polygon, label_line, fidelity = _faithful_polygon(
                contour, mask, width=width, height=height)
        except ValueError as exc:
            return drop(aid, split, "mask_polygon_fidelity_failed", detail=str(exc))

        # ------------------------- persist -----------------------
        image_path = out_dir / "images" / split / f"{aid}.jpg"
        label_path = out_dir / "labels" / split / f"{aid}.txt"

        image_path.write_bytes(image_bytes)
        label_path.write_text(label_line + "\n", encoding="utf-8")

        # Validate immediately so malformed labels never survive export.
        try:
            _parse_and_validate_label(label_path)
        except Exception:
            image_path.unlink(missing_ok=True)
            label_path.unlink(missing_ok=True)
            raise

        mask_sha256 = _sha256_bytes(mask_bytes)
        seen_image_hashes[image_sha256] = (split, aid)
        kept[split].append(aid)
        records.append(
            {
                "analysis_id": aid,
                "split": split,
                "image_sha256": image_sha256,
                "mask_sha256": mask_sha256,
                "width": int(width),
                "height": int(height),
                "polygon_points": int(polygon.shape[0]),
                **fidelity,
                "mask_area_px": float(cv2.contourArea(contour)),
            }
        )
        return True

    # Train is deliberately exported first. Exact duplicates found later in val
    # are rejected, protecting validation from leakage.
    for aid in selection["train_ids"]:
        export_one(aid, "train")

    for aid in selection["val_ids"]:
        export_one(aid, "val")

    selection["train_ids"] = kept["train"]
    selection["val_ids"] = kept["val"]
    selection["dropped_ids"] = dropped

    manifest.setdefault("policy", {})["min_mask_polygon_iou"] = MIN_MASK_POLYGON_IOU
    manifest["policy"]["min_mask_area"] = float(min_mask_area)
    manifest["policy"]["exact_image_dedup"] = "sha256"
    manifest["policy"]["mask_dimension_policy"] = "must_match_image"
    manifest["policy"]["label_format"] = (
        "class x1 y1 x2 y2 ... normalized; no bbox fields"
    )

    data_yaml = _write_data_yaml(out_dir)

    # Keep the rejection report even when validation blocks training, e.g.
    # when every validation image was an exact duplicate of training data.
    # The error is re-raised after reports are written, preserving a non-zero
    # exit status for the worker.
    validation_error: Optional[Exception] = None
    try:
        integrity = validate_exported_dataset(out_dir)
    except Exception as exc:
        validation_error = exc
        integrity = {
            "train_count": len(kept["train"]),
            "val_count": len(kept["val"]),
            "label_format": "ultralytics_yolo_segmentation_polygon_only",
            "validated": False,
            "validation_error": str(exc),
        }

    manifest["export"] = {
        "out_dir": out_dir.resolve().as_posix(),
        "train_count": len(kept["train"]),
        "val_count": len(kept["val"]),
        "dropped_count": len(dropped),
        "integrity": integrity,
        "records": records,
    }

    manifest_out = out_dir / "manifest.json"
    manifest_out.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    stats = {
        "train": len(kept["train"]),
        "val": len(kept["val"]),
        "dropped": len(dropped),
        **integrity,
    }
    (out_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if validation_error is not None:
        print(f"[BLOCKED] Dataset integrity validation failed: {validation_error}")
        print(f"  rejection report: {manifest_out}")
        print(f"  stats: {out_dir / 'stats.json'}")
        raise validation_error

    return manifest_out, data_yaml


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export a reproducible ArborScan tree-segmentation dataset from "
            "an explicit manifest"
        )
    )
    parser.add_argument(
        "--bucket-verified",
        default=DEFAULT_BUCKET_VERIFIED,
    )
    parser.add_argument(
        "--manifest-in",
        required=True,
        help="Path to manifest JSON containing selection.train_ids/val_ids",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
    )
    parser.add_argument(
        "--min-mask-area",
        type=float,
        default=float(os.getenv("MIN_MASK_AREA", "100")),
    )

    args = parser.parse_args()

    # Validate secrets without ever printing their values.
    _base_url()
    _headers()

    out_dir = Path(args.out_dir).expanduser().resolve()
    manifest_in = Path(args.manifest_in).expanduser().resolve()

    if not manifest_in.exists() or not manifest_in.is_file():
        raise RuntimeError(f"manifest_in not found: {manifest_in}")

    print(f"SUPABASE_URL configured = {bool(SUPABASE_URL)}")
    print(f"SUPABASE_SERVICE_KEY configured = {bool(SUPABASE_SERVICE_KEY)}")
    print(f"BUCKET_VERIFIED = {args.bucket_verified}")
    print(f"OUT_DIR = {out_dir}")
    print(f"MANIFEST_IN = {manifest_in}")

    manifest_out, data_yaml = export_from_manifest(
        bucket_verified=args.bucket_verified,
        out_dir=out_dir,
        manifest_in=manifest_in,
        min_mask_area=args.min_mask_area,
    )

    print("[OK] Export complete and dataset integrity validated")
    print(f"  manifest: {manifest_out}")
    print(f"  data.yaml: {data_yaml}")


if __name__ == "__main__":
    main()
