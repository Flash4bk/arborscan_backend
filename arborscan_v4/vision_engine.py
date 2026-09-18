from __future__ import annotations

import base64
import os
import threading
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

import cv2
import numpy as np
from ultralytics import YOLO

from config import settings


@dataclass
class DetectionResult:
    detected: bool
    mask: Optional[np.ndarray] = None
    confidence: Optional[float] = None
    class_id: Optional[int] = None
    class_name: Optional[str] = None
    bbox: Optional[tuple[int, int, int, int]] = None
    warnings: List[str] = field(default_factory=list)


class TreeVisionRuntime:
    def __init__(self) -> None:
        self.model: Optional[YOLO] = None
        self.model_path: Optional[Path] = None
        self.model_version: Optional[int] = None
        self.model_names: Dict[int, str] = {}
        self.conf = max(0.05, min(float(os.getenv("V4_TREE_CONF", "0.25")), 0.95))
        self.imgsz = max(320, min(int(os.getenv("V4_TREE_IMGSZ", "1024")), 1536))
        self._lock = threading.RLock()

    def _select_model_path(self) -> tuple[Path, int]:
        configured = int(settings.active_model_version or 0)
        candidates: List[tuple[int, Path]] = []
        for path in settings.model_dir.glob("model_v*.pt"):
            stem = path.stem
            try:
                version = int(stem.replace("model_v", ""))
            except ValueError:
                continue
            if path.is_file() and path.stat().st_size >= settings.model_min_size_bytes:
                candidates.append((version, path.resolve()))

        if configured > 0:
            configured_path = settings.model_dir / f"model_v{configured}.pt"
            if configured_path.exists():
                return configured_path.resolve(), configured

        if settings.auto_select_latest_local_model and candidates:
            version, path = max(candidates, key=lambda item: item[0])
            return path, version

        raise FileNotFoundError(
            f"No valid tree segmentation model found in {settings.model_dir}"
        )

    def load(self) -> None:
        path, version = self._select_model_path()
        model = YOLO(str(path))
        task = getattr(model, "task", None)
        if task and task != "segment":
            raise RuntimeError(
                f"Unified v4 requires a segmentation model, got task={task!r}: {path}"
            )
        raw_names = getattr(model, "names", {}) or {}
        if isinstance(raw_names, list):
            names = {i: str(name) for i, name in enumerate(raw_names)}
        else:
            names = {int(k): str(v) for k, v in dict(raw_names).items()}

        self.model = model
        self.model_path = path
        self.model_version = version
        self.model_names = names
        self.model_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    def health(self) -> dict:
        return {
            "loaded": self.model is not None,
            "version": self.model_version,
            "path": str(self.model_path) if self.model_path else None,
            "task": getattr(self.model, "task", None) if self.model else None,
            "names": self.model_names,
            "confidence_threshold": self.conf,
            "imgsz": self.imgsz,
        }

    def _tree_class_ids(self) -> Optional[Set[int]]:
        if not self.model_names:
            return None
        recognized: Set[int] = set()
        for class_id, name in self.model_names.items():
            normalized = name.strip().casefold()
            if normalized in {"tree", "trees", "дерево", "деревья"} or "tree" in normalized:
                recognized.add(class_id)
        if recognized:
            return recognized
        if len(self.model_names) == 1:
            return set(self.model_names.keys())
        return None

    def infer(
        self,
        image_bgr: np.ndarray,
        tap_x: Optional[float] = None,
        tap_y: Optional[float] = None,
    ) -> DetectionResult:
        if self.model is None:
            raise RuntimeError("Tree model is not loaded")
        if image_bgr is None or image_bgr.ndim != 3:
            raise ValueError("image_bgr must be a decoded BGR image")

        with self._lock:
            result = self.model(
                image_bgr,
                imgsz=self.imgsz,
                retina_masks=True,
                conf=self.conf,
                verbose=False,
                device="cpu",
            )[0]

        masks_obj = getattr(result, "masks", None)
        boxes_obj = getattr(result, "boxes", None)
        if masks_obj is None or boxes_obj is None or len(masks_obj.data) == 0:
            return DetectionResult(detected=False, warnings=["tree_segmentation_returned_no_masks"])

        n = min(len(masks_obj.data), len(boxes_obj))
        if n <= 0:
            return DetectionResult(detected=False, warnings=["tree_segmentation_returned_no_candidates"])

        h, w = image_bgr.shape[:2]
        allowed_ids = self._tree_class_ids()
        warnings: List[str] = []
        if allowed_ids is None and len(self.model_names) > 1:
            warnings.append("tree_class_name_not_resolved_using_all_segmentation_classes")

        target_px: Optional[tuple[float, float]] = None
        if tap_x is not None and tap_y is not None:
            tx, ty = float(tap_x), float(tap_y)
            if 0.0 <= tx <= 1.0 and 0.0 <= ty <= 1.0:
                target_px = (tx * w, ty * h)
            else:
                warnings.append("tap_coordinates_outside_normalized_range_ignored")

        candidates = []
        for i in range(n):
            try:
                class_id = int(boxes_obj.cls[i].item())
            except Exception:
                class_id = 0
            if allowed_ids is not None and class_id not in allowed_ids:
                continue

            mask_small = (masks_obj.data[i].detach().cpu().numpy() > 0.5).astype(np.uint8)
            mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0).astype(np.uint8) * 255
            ys, xs = np.where(mask > 0)
            if ys.size == 0:
                continue

            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            area = int(ys.size)
            area_ratio = area / float(max(1, h * w))
            try:
                confidence = float(boxes_obj.conf[i].item())
            except Exception:
                confidence = 0.0

            contains_tap = False
            distance = None
            if target_px is not None:
                tx, ty = target_px
                ix = max(0, min(w - 1, int(round(tx))))
                iy = max(0, min(h - 1, int(round(ty))))
                contains_tap = bool(mask[iy, ix] > 0)
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                distance = ((cx - tx) / max(w, 1)) ** 2 + ((cy - ty) / max(h, 1)) ** 2

            score = confidence * (0.55 + 0.45 * min(1.0, area_ratio ** 0.5 * 3.0))
            candidates.append(
                {
                    "mask": mask,
                    "confidence": confidence,
                    "class_id": class_id,
                    "bbox": (x1, y1, x2, y2),
                    "area": area,
                    "contains_tap": contains_tap,
                    "distance": distance,
                    "score": score,
                }
            )

        if not candidates:
            return DetectionResult(detected=False, warnings=warnings + ["no_tree_class_candidate"])

        if target_px is not None:
            containing = [c for c in candidates if c["contains_tap"]]
            if containing:
                selected = max(containing, key=lambda c: (c["confidence"], c["area"]))
            else:
                selected = min(candidates, key=lambda c: c["distance"] if c["distance"] is not None else 999.0)
                warnings.append("tap_did_not_hit_mask_nearest_tree_selected")
        else:
            selected = max(candidates, key=lambda c: c["score"])

        class_id = selected["class_id"]
        return DetectionResult(
            detected=True,
            mask=selected["mask"],
            confidence=selected["confidence"],
            class_id=class_id,
            class_name=self.model_names.get(class_id, str(class_id)),
            bbox=selected["bbox"],
            warnings=warnings,
        )


def crop_tree(image_bgr: np.ndarray, detection: DetectionResult, margin_ratio: float = 0.04):
    if not detection.detected or detection.bbox is None:
        return None
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = detection.bbox
    mx = int(round((x2 - x1 + 1) * margin_ratio))
    my = int(round((y2 - y1 + 1) * margin_ratio))
    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(w - 1, x2 + mx)
    y2 = min(h - 1, y2 + my)
    return image_bgr[y1 : y2 + 1, x1 : x2 + 1].copy()


def _resize_pair(image_bgr: np.ndarray, mask: np.ndarray, max_side: int = 1600):
    h, w = image_bgr.shape[:2]
    longest = max(h, w)
    scale = min(1.0, max_side / float(longest))
    tw, th = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    if (tw, th) == (w, h):
        return image_bgr.copy(), mask.copy()
    image = cv2.resize(image_bgr, (tw, th), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)
    return image, resized_mask


def encode_visuals(image_bgr: np.ndarray, mask: np.ndarray, bbox: Optional[tuple[int, int, int, int]]) -> dict:
    image, mask_small = _resize_pair(image_bgr, mask)
    mask_bool = mask_small > 0
    annotated = image.copy()
    contours, _ = cv2.findContours(mask_small.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(annotated, contours, -1, (0, 255, 0), 3)

    if bbox is not None:
        src_h, src_w = image_bgr.shape[:2]
        dst_h, dst_w = image.shape[:2]
        sx, sy = dst_w / float(src_w), dst_h / float(src_h)
        x1, y1, x2, y2 = bbox
        cv2.rectangle(
            annotated,
            (int(round(x1 * sx)), int(round(y1 * sy))),
            (int(round(x2 * sx)), int(round(y2 * sy))),
            (0, 255, 0),
            2,
        )

    overlay = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    overlay[:, :, 1] = 255
    overlay[:, :, 3] = mask_bool.astype(np.uint8) * 110

    ok_img, img_buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    ok_ann, ann_buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    ok_mask, mask_buf = cv2.imencode(".png", overlay)
    if not (ok_img and ok_ann and ok_mask):
        raise RuntimeError("Could not encode response images")

    return {
        "original_image_base64": base64.b64encode(img_buf.tobytes()).decode("ascii"),
        "annotated_image_base64": base64.b64encode(ann_buf.tobytes()).decode("ascii"),
        "mask_image_base64": base64.b64encode(mask_buf.tobytes()).decode("ascii"),
        "width": image.shape[1],
        "height": image.shape[0],
    }
