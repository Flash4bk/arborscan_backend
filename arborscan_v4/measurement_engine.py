from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


_EPS = 1e-9


@dataclass(frozen=True)
class PixelGeometry:
    image_width_px: int
    image_height_px: int
    tree_height_px: float
    crown_width_px: float
    trunk_width_px_estimate: Optional[float]
    mask_area_px: int
    bbox_x1: int
    bbox_y1: int
    bbox_x2: int
    bbox_y2: int
    touches_image_edge: bool

    @property
    def bbox_width_px(self) -> float:
        return float(self.bbox_x2 - self.bbox_x1 + 1)

    @property
    def bbox_height_px(self) -> float:
        return float(self.bbox_y2 - self.bbox_y1 + 1)

    @property
    def mask_area_ratio(self) -> float:
        total = max(1, self.image_width_px * self.image_height_px)
        return float(self.mask_area_px) / float(total)


@dataclass(frozen=True)
class CalibrationRequest:
    # Direct AR measurements.  They are accepted as physical values, not as
    # ground truth for every other dimension.
    ar_height_m: Optional[float] = None
    ar_crown_width_m: Optional[float] = None
    ar_trunk_diameter_m: Optional[float] = None
    ar_trunk_measurement_height_m: Optional[float] = None
    ar_quality: Optional[float] = None

    # Explicit photo calibration.  ``manual_scale`` is kept for compatibility
    # with the current Flutter app.  New UI should prefer reference_length_*.
    manual_scale_px_to_m: Optional[float] = None
    reference_length_m: Optional[float] = None
    reference_length_px: Optional[float] = None
    reference_same_plane: Optional[bool] = None

    # Optional future/manual DBH line measured on the same photo.
    dbh_width_px: Optional[float] = None
    dbh_measurement_height_m: Optional[float] = None


@dataclass(frozen=True)
class ScaleCandidate:
    source: str
    px_to_m: float
    confidence: float


@dataclass
class CalibrationResult:
    px_to_m: Optional[float]
    source: Optional[str]
    confidence: float
    conflict: bool = False
    candidates: List[ScaleCandidate] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def available(self) -> bool:
        return self.px_to_m is not None and self.px_to_m > 0


@dataclass(frozen=True)
class MeasurementResult:
    value_m: Optional[float]
    value_px: Optional[float]
    source: str
    confidence: float
    measurement_height_m: Optional[float] = None
    standard: Optional[str] = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class FusedMeasurements:
    height: MeasurementResult
    crown_width: MeasurementResult
    trunk_diameter: MeasurementResult
    status: str
    overall_confidence: float
    warnings: tuple[str, ...]


def _positive(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number) or number <= 0:
        return None
    return number


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _robust_row_span(mask_bool: np.ndarray, y: int) -> Optional[float]:
    xs = np.flatnonzero(mask_bool[y])
    if xs.size == 0:
        return None
    return float(xs[-1] - xs[0] + 1)


def compute_pixel_geometry(mask: np.ndarray) -> PixelGeometry:
    """Measure robust 2D geometry from a binary segmentation mask.

    These are pixel measurements only.  No metres are invented here.
    ``trunk_width_px_estimate`` is diagnostic and is *not* automatically called
    DBH because the whole-tree mask does not guarantee a 1.3 m measurement
    plane.
    """

    if mask is None or getattr(mask, "ndim", 0) != 2:
        raise ValueError("mask must be a 2D array")

    mask_bool = np.asarray(mask) > 0
    h, w = mask_bool.shape
    ys, xs = np.where(mask_bool)
    if ys.size == 0:
        raise ValueError("mask contains no foreground pixels")

    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())
    tree_height = float(y2 - y1 + 1)

    # Crown width: 95th percentile of row spans above the lowest 12% of the
    # tree.  The percentile is intentionally used instead of max() to suppress
    # one-pixel branches/noise.
    crown_end = min(y2, y1 + max(1, int(round(tree_height * 0.88))))
    crown_spans = [
        span
        for y in range(y1, crown_end + 1)
        if (span := _robust_row_span(mask_bool, y)) is not None
    ]
    crown_width = (
        float(np.percentile(crown_spans, 95))
        if crown_spans
        else float(x2 - x1 + 1)
    )

    # Diagnostic trunk-width estimate.  We look at a lower-middle band and use
    # the narrow portion of row spans.  It is deliberately not promoted to a
    # metric trunk diameter unless the caller supplies an explicit DBH line.
    trunk_y1 = max(y1, y1 + int(round(tree_height * 0.62)))
    trunk_y2 = min(y2, y1 + int(round(tree_height * 0.90)))
    trunk_spans = [
        span
        for y in range(trunk_y1, trunk_y2 + 1)
        if (span := _robust_row_span(mask_bool, y)) is not None
    ]
    trunk_estimate: Optional[float] = None
    if len(trunk_spans) >= 5:
        upper_limit = max(3.0, crown_width * 0.45)
        filtered = [s for s in trunk_spans if s <= upper_limit]
        source = filtered if len(filtered) >= 3 else trunk_spans
        # Median of the lower half is resistant to low branches and grass.
        ordered = sorted(source)
        lower_half = ordered[: max(1, (len(ordered) + 1) // 2)]
        trunk_estimate = float(np.median(lower_half))

    edge_margin = 1
    touches_edge = (
        x1 <= edge_margin
        or y1 <= edge_margin
        or x2 >= (w - 1 - edge_margin)
        or y2 >= (h - 1 - edge_margin)
    )

    return PixelGeometry(
        image_width_px=w,
        image_height_px=h,
        tree_height_px=tree_height,
        crown_width_px=crown_width,
        trunk_width_px_estimate=trunk_estimate,
        mask_area_px=int(mask_bool.sum()),
        bbox_x1=x1,
        bbox_y1=y1,
        bbox_x2=x2,
        bbox_y2=y2,
        touches_image_edge=touches_edge,
    )


def resolve_calibration(
    geometry: PixelGeometry,
    request: CalibrationRequest,
) -> CalibrationResult:
    """Resolve an absolute photo scale using only explicit physical evidence.

    Deliberately NOT supported:
      * average species height;
      * assumed 1.7 m person / 1.5 m car;
      * assumed 1 m stick;
      * focal-distance heuristics without calibrated camera intrinsics.
    """

    candidates: List[ScaleCandidate] = []
    notes: List[str] = []

    manual = _positive(request.manual_scale_px_to_m)
    if manual is not None:
        if manual > 1.0:
            notes.append("manual_scale_px_to_m_is_implausibly_large")
        else:
            candidates.append(
                ScaleCandidate(
                    source="legacy_manual_scale",
                    px_to_m=manual,
                    confidence=0.75,
                )
            )

    ref_m = _positive(request.reference_length_m)
    ref_px = _positive(request.reference_length_px)
    if ref_m is not None and ref_px is not None:
        ref_scale = ref_m / ref_px
        same_plane = request.reference_same_plane is True
        candidates.append(
            ScaleCandidate(
                source="reference_object",
                px_to_m=ref_scale,
                confidence=0.95 if same_plane else 0.82,
            )
        )
        if not same_plane:
            notes.append("reference_depth_relative_to_tree_not_confirmed")

    ar_quality = _clamp01(request.ar_quality if request.ar_quality is not None else 0.85)
    ar_height = _positive(request.ar_height_m)
    if ar_height is not None and geometry.tree_height_px > _EPS:
        candidates.append(
            ScaleCandidate(
                source="ar_height_calibration",
                px_to_m=ar_height / geometry.tree_height_px,
                confidence=min(0.90, ar_quality),
            )
        )

    ar_crown = _positive(request.ar_crown_width_m)
    if ar_crown is not None and geometry.crown_width_px > _EPS:
        candidates.append(
            ScaleCandidate(
                source="ar_crown_calibration",
                px_to_m=ar_crown / geometry.crown_width_px,
                confidence=min(0.82, ar_quality),
            )
        )

    if not candidates:
        return CalibrationResult(
            px_to_m=None,
            source=None,
            confidence=0.0,
            candidates=[],
            notes=notes,
        )

    explicit = [c for c in candidates if c.source in {"reference_object", "legacy_manual_scale"}]
    ar_candidates = [c for c in candidates if c.source.startswith("ar_")]

    # A verified reference/manual scale wins because it is a direct photo
    # calibration.  AR candidates still cross-check it and can emit warnings.
    if explicit:
        chosen = sorted(explicit, key=lambda c: c.confidence, reverse=True)[0]
        conflicts = []
        for other in ar_candidates:
            rel = abs(other.px_to_m - chosen.px_to_m) / max(chosen.px_to_m, other.px_to_m, _EPS)
            if rel > 0.25:
                conflicts.append(other.source)
        if conflicts:
            notes.append("calibration_crosscheck_conflict:" + ",".join(conflicts))
        return CalibrationResult(
            px_to_m=chosen.px_to_m,
            source=chosen.source,
            confidence=chosen.confidence * (0.85 if conflicts else 1.0),
            conflict=bool(conflicts),
            candidates=candidates,
            notes=notes,
        )

    # If both AR height and crown define a scale, they should agree.  If they
    # disagree strongly, keep the direct AR dimensions but do NOT derive the
    # remaining metric from a contradictory photo scale.
    if len(ar_candidates) >= 2:
        values = [c.px_to_m for c in ar_candidates]
        spread = (max(values) - min(values)) / max(max(values), _EPS)
        if spread > 0.20:
            notes.append(f"ar_scale_conflict:{spread:.3f}")
            return CalibrationResult(
                px_to_m=None,
                source=None,
                confidence=0.0,
                conflict=True,
                candidates=candidates,
                notes=notes,
            )

        weights = np.array([max(c.confidence, 0.01) for c in ar_candidates], dtype=float)
        vals = np.array(values, dtype=float)
        scale = float(np.average(vals, weights=weights))
        confidence = min(c.confidence for c in ar_candidates)
        return CalibrationResult(
            px_to_m=scale,
            source="ar_fused_calibration",
            confidence=confidence,
            candidates=candidates,
            notes=notes,
        )

    chosen = ar_candidates[0]
    return CalibrationResult(
        px_to_m=chosen.px_to_m,
        source=chosen.source,
        confidence=chosen.confidence,
        candidates=candidates,
        notes=notes,
    )


def _derived_source(scale_source: Optional[str]) -> str:
    if scale_source is None:
        return "unavailable"
    if scale_source.startswith("ar_"):
        return "ar+vision"
    if scale_source == "reference_object":
        return "reference+vision"
    if scale_source == "legacy_manual_scale":
        return "manual_scale+vision"
    return "vision"


def fuse_measurements(
    geometry: PixelGeometry,
    calibration_request: CalibrationRequest,
    calibration: CalibrationResult,
    segmentation_confidence: float,
) -> FusedMeasurements:
    seg_conf = _clamp01(segmentation_confidence)
    warnings: List[str] = list(calibration.notes)

    if geometry.touches_image_edge:
        warnings.append("tree_mask_touches_image_edge")

    ar_quality = _clamp01(
        calibration_request.ar_quality
        if calibration_request.ar_quality is not None
        else 0.85
    )

    ar_h = _positive(calibration_request.ar_height_m)
    if ar_h is not None:
        height = MeasurementResult(
            value_m=round(ar_h, 3),
            value_px=geometry.tree_height_px,
            source="ar",
            confidence=min(0.90, ar_quality),
            notes=("direct_ar_measurement",),
        )
    elif calibration.available:
        height = MeasurementResult(
            value_m=round(geometry.tree_height_px * calibration.px_to_m, 3),
            value_px=geometry.tree_height_px,
            source=_derived_source(calibration.source),
            confidence=_clamp01(min(seg_conf, calibration.confidence) * 0.95),
            notes=("derived_from_segmentation_and_absolute_scale",),
        )
    else:
        height = MeasurementResult(
            value_m=None,
            value_px=geometry.tree_height_px,
            source="vision",
            confidence=seg_conf,
            notes=("absolute_scale_required_for_metres",),
        )

    ar_c = _positive(calibration_request.ar_crown_width_m)
    if ar_c is not None:
        crown = MeasurementResult(
            value_m=round(ar_c, 3),
            value_px=geometry.crown_width_px,
            source="ar",
            confidence=min(0.86, ar_quality),
            notes=("direct_ar_measurement",),
        )
    elif calibration.available:
        crown = MeasurementResult(
            value_m=round(geometry.crown_width_px * calibration.px_to_m, 3),
            value_px=geometry.crown_width_px,
            source=_derived_source(calibration.source),
            confidence=_clamp01(min(seg_conf, calibration.confidence) * 0.85),
            notes=("weak_perspective_assumption",),
        )
    else:
        crown = MeasurementResult(
            value_m=None,
            value_px=geometry.crown_width_px,
            source="vision",
            confidence=seg_conf * 0.9,
            notes=("absolute_scale_required_for_metres",),
        )

    ar_t = _positive(calibration_request.ar_trunk_diameter_m)
    ar_t_height = _positive(calibration_request.ar_trunk_measurement_height_m)
    if ar_t is not None:
        standard = "dbh_1_3m" if ar_t_height is not None and abs(ar_t_height - 1.3) <= 0.08 else None
        notes = ["direct_ar_measurement"]
        if standard is None:
            notes.append("measurement_height_is_not_confirmed_as_dbh_1_3m")
        trunk = MeasurementResult(
            value_m=round(ar_t, 4),
            value_px=geometry.trunk_width_px_estimate,
            source="ar",
            confidence=min(0.86, ar_quality),
            measurement_height_m=ar_t_height,
            standard=standard,
            notes=tuple(notes),
        )
    else:
        dbh_px = _positive(calibration_request.dbh_width_px)
        dbh_h = _positive(calibration_request.dbh_measurement_height_m)
        if dbh_px is not None and calibration.available:
            standard = "dbh_1_3m" if dbh_h is not None and abs(dbh_h - 1.3) <= 0.08 else None
            trunk = MeasurementResult(
                value_m=round(dbh_px * calibration.px_to_m, 4),
                value_px=dbh_px,
                source=_derived_source(calibration.source),
                confidence=_clamp01(min(seg_conf, calibration.confidence) * 0.82),
                measurement_height_m=dbh_h,
                standard=standard,
                notes=("explicit_trunk_line_on_photo",),
            )
        else:
            trunk = MeasurementResult(
                value_m=None,
                value_px=geometry.trunk_width_px_estimate,
                source="vision",
                confidence=seg_conf * 0.45 if geometry.trunk_width_px_estimate else 0.0,
                notes=(
                    "whole_tree_mask_trunk_width_is_diagnostic_only",
                    "dbh_requires_ar_or_explicit_1_3m_line",
                ),
            )

    metric_values = [height.value_m, crown.value_m, trunk.value_m]
    count = sum(v is not None for v in metric_values)
    if count == 3:
        status = "measured"
    elif count > 0:
        status = "partial_measurement"
    else:
        status = "absolute_scale_required"

    available_confidences = [
        m.confidence
        for m in (height, crown, trunk)
        if m.value_m is not None
    ]
    overall = (
        float(np.mean(available_confidences))
        if available_confidences
        else seg_conf * 0.6
    )

    return FusedMeasurements(
        height=height,
        crown_width=crown,
        trunk_diameter=trunk,
        status=status,
        overall_confidence=_clamp01(overall),
        warnings=tuple(dict.fromkeys(warnings)),
    )
