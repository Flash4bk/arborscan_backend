from __future__ import annotations

import os
import hashlib
import math
from contextlib import asynccontextmanager
from typing import Optional
from uuid import uuid4

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from config import settings
from . import API_VERSION, SCHEMA_VERSION
from .measurement_engine import (
    CalibrationRequest,
    compute_pixel_geometry,
    fuse_measurements,
    resolve_calibration,
)
from .plantnet_service import PlantNetClient
from .measurement_image import decode_oriented_rgb
from .schema import (
    AnalysisStatus,
    BoundingBox,
    CalibrationInfo,
    ImagePayload,
    MechanicalProfileInfo,
    MeasurementSource,
    MeasurementsInfo,
    MeasurementValue,
    PixelMeasurements,
    QualityInfo,
    RiskInfo,
    ScaleCandidate,
    SpeciesCandidate,
    SpeciesInfo,
    TreeDetectionInfo,
    UnifiedAnalysisResponse,
)
from .species_profiles import get_mechanical_profile
from .vision_engine import TreeVisionRuntime, crop_tree, encode_visuals


MAX_UPLOAD_BYTES = max(1, int(os.getenv("V4_MAX_UPLOAD_MB", "15"))) * 1024 * 1024

vision = TreeVisionRuntime()
plantnet = PlantNetClient()


@asynccontextmanager
async def lifespan(_: FastAPI):
    await run_in_threadpool(vision.load)
    print(f"[*] ArborScan Unified Analysis {API_VERSION} ready")
    print(f"[*] Tree model: {vision.health()}")
    yield


app = FastAPI(
    title="ArborScan Unified Analysis v4",
    version=API_VERSION,
    lifespan=lifespan,
)


def _measurement_model(result) -> MeasurementValue:
    source_map = {
        "ar": MeasurementSource.AR,
        "ar+vision": MeasurementSource.AR_VISION,
        "reference+vision": MeasurementSource.REFERENCE_VISION,
        "manual_scale+vision": MeasurementSource.MANUAL_SCALE_VISION,
        "vision": MeasurementSource.VISION,
        "unavailable": MeasurementSource.UNAVAILABLE,
    }
    return MeasurementValue(
        value_m=result.value_m,
        value_px=result.value_px,
        source=source_map.get(result.source, MeasurementSource.UNAVAILABLE),
        confidence=result.confidence,
        measurement_height_m=result.measurement_height_m,
        standard=result.standard,
        notes=list(result.notes),
    )


def _species_model(raw: dict) -> SpeciesInfo:
    return SpeciesInfo(
        status=str(raw.get("status") or "unknown"),
        display_name=str(raw.get("display_name") or "Неизвестно"),
        scientific_name=raw.get("scientific_name"),
        common_names=list(raw.get("common_names") or []),
        confidence=raw.get("confidence"),
        top_results=[
            SpeciesCandidate(
                display_name=str(item.get("display_name") or "Неизвестно"),
                scientific_name=item.get("scientific_name"),
                common_names=list(item.get("common_names") or []),
                confidence=item.get("confidence"),
            )
            for item in (raw.get("top_results") or [])
            if isinstance(item, dict)
        ],
        predicted_organs=list(raw.get("predicted_organs") or []),
        remaining_requests=raw.get("remaining_requests"),
        latency_ms=raw.get("latency_ms"),
        message=raw.get("message"),
    )


def _empty_measurement(note: str) -> MeasurementValue:
    return MeasurementValue(
        value_m=None,
        value_px=None,
        source=MeasurementSource.UNAVAILABLE,
        confidence=0.0,
        notes=[note],
    )


@app.get("/")
def root():
    return {
        "service": "ArborScan Unified Analysis",
        "api_version": API_VERSION,
        "schema_version": SCHEMA_VERSION,
        "legacy_v3_untouched": True,
    }


@app.get("/health")
def health(deep: bool = False):
    model_info = vision.health()
    return {
        "status": "ok" if model_info.get("loaded") else "degraded",
        "api_version": API_VERSION,
        "schema_version": SCHEMA_VERSION,
        "model": model_info,
        "plantnet": plantnet.health() if deep else {"configured": bool(settings.plantnet_api_key)},
        "accuracy_policy": {
            "fake_mask_fallback": False,
            "species_height_scale_fallback": False,
            "person_car_assumed_scale": False,
            "species_mechanical_fallback": False,
            "risk_enabled": False,
        },
    }


@app.post("/v4/analyze-tree", response_model=UnifiedAnalysisResponse)
async def analyze_tree_v4(
    file: UploadFile = File(...),
    tap_x: Optional[float] = Form(None),
    tap_y: Optional[float] = Form(None),
    ar_height_m: Optional[float] = Form(None),
    ar_crown_width_m: Optional[float] = Form(None),
    ar_trunk_diameter_m: Optional[float] = Form(None),
    ar_trunk_measurement_height_m: Optional[float] = Form(None),
    ar_quality: Optional[float] = Form(None),
    ar_photo_sha256: Optional[str] = Form(None),
    ar_same_tree_confirmed: bool = Form(False),
    crown_width_px: Optional[float] = Form(None),
    manual_scale: Optional[float] = Form(None),
    reference_length_m: Optional[float] = Form(None),
    reference_length_px: Optional[float] = Form(None),
    reference_same_plane: Optional[bool] = Form(None),
    dbh_width_px: Optional[float] = Form(None),
    dbh_measurement_height_m: Optional[float] = Form(None),
    camera_distance_m: Optional[float] = Form(None),
    include_images: bool = Form(True),
):
    analysis_id = str(uuid4())
    image_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image upload")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
        )

    # Bound decoded allocation and normalize EXIF before any pixel coordinates.
    from PIL import Image, UnidentifiedImageError
    try:
        image = cv2.cvtColor(decode_oriented_rgb(image_bytes), cv2.COLOR_RGB2BGR)
    except ValueError:
        raise HTTPException(413, 'Image resolution exceeds 25 megapixels') from None
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError):
        raise HTTPException(400, 'Image cannot be decoded') from None
    for value in (ar_height_m, ar_crown_width_m, ar_trunk_diameter_m,
                  ar_trunk_measurement_height_m, manual_scale, reference_length_m,
                  reference_length_px, dbh_width_px, dbh_measurement_height_m, crown_width_px):
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise HTTPException(422, 'Dimensions must be finite and positive')
    if ar_quality is not None and (not math.isfinite(ar_quality) or not 0 <= ar_quality <= 1):
        raise HTTPException(422, 'Invalid AR diagnostic weight')
    if image is None:
        raise HTTPException(status_code=400, detail="Image cannot be decoded")

    warnings = []
    ar_matches = ar_same_tree_confirmed and ar_photo_sha256 == hashlib.sha256(image_bytes).hexdigest()
    if any(v is not None for v in (ar_height_m, ar_crown_width_m, ar_trunk_diameter_m)) and not ar_matches:
        warnings.append('ar_not_bound_to_this_photo_direct_dimensions_ignored')
    if camera_distance_m is not None:
        warnings.append(
            "camera_distance_m_not_used_without_calibrated_intrinsics_and_pose"
        )

    detection = await run_in_threadpool(vision.infer, image, tap_x, tap_y)
    warnings.extend(detection.warnings)

    if not detection.detected or detection.mask is None:
        h, w = image.shape[:2]
        empty_species = SpeciesInfo(status="not_run", display_name="Неизвестно")
        empty_profile = MechanicalProfileInfo(
            available=False,
            reason="species_not_identified",
        )
        return UnifiedAnalysisResponse(
            segmentation_model_version=vision.health().get('version'),
            analysis_id=analysis_id,
            api_version=API_VERSION,
            schema_version=SCHEMA_VERSION,
            analysis_status=AnalysisStatus.TREE_NOT_DETECTED,
            tree=TreeDetectionInfo(
                detected=False,
                model_version=vision.model_version,
                model_path=str(vision.model_path) if vision.model_path else None,
            ),
            species=empty_species,
            mechanical_profile=empty_profile,
            pixel_measurements=PixelMeasurements(
                image_width_px=w,
                image_height_px=h,
            ),
            calibration=CalibrationInfo(available=False),
            measurements=MeasurementsInfo(
                height=_empty_measurement("tree_not_detected"),
                crown_width=_empty_measurement("tree_not_detected"),
                trunk_diameter=_empty_measurement("tree_not_detected"),
            ),
            quality=QualityInfo(
                overall=0.0,
                status="invalid_for_measurement",
                segmentation=0.0,
                calibration=0.0,
                warnings=list(dict.fromkeys(warnings + ["tree_not_detected"])),
            ),
            risk=RiskInfo(available=False, reason="tree_not_detected"),
            images=ImagePayload(width=w, height=h),
            warnings=list(dict.fromkeys(warnings + ["tree_not_detected"])),
            persisted=False,
        )

    geometry = compute_pixel_geometry(detection.mask)
    crop = crop_tree(image, detection)
    species_raw = await run_in_threadpool(plantnet.identify, crop)
    species = _species_model(species_raw)
    mechanical_raw = get_mechanical_profile(species.scientific_name)
    mechanical = MechanicalProfileInfo(**mechanical_raw)

    calibration_request = CalibrationRequest(
        ar_height_m=ar_height_m,
        ar_crown_width_m=ar_crown_width_m,
        ar_trunk_diameter_m=ar_trunk_diameter_m,
        ar_trunk_measurement_height_m=ar_trunk_measurement_height_m,
        ar_quality=ar_quality,
        ar_photo_matches=ar_matches,
        crown_width_px=crown_width_px,
        manual_scale_px_to_m=manual_scale,
        reference_length_m=reference_length_m,
        reference_length_px=reference_length_px,
        reference_same_plane=reference_same_plane,
        dbh_width_px=dbh_width_px,
        dbh_measurement_height_m=dbh_measurement_height_m,
    )
    calibration_result = resolve_calibration(geometry, calibration_request)
    fused = fuse_measurements(
        geometry,
        calibration_request,
        calibration_result,
        segmentation_confidence=float(detection.confidence or 0.0),
    )
    warnings.extend(fused.warnings)

    if mechanical.available is False:
        warnings.append("mechanical_profile_not_available_risk_not_calculated")

    visual_payload = ImagePayload(width=image.shape[1], height=image.shape[0])
    if include_images:
        visuals = encode_visuals(image, detection.mask, detection.bbox)
        visual_payload = ImagePayload(**visuals)

    segmentation_conf = max(0.0, min(1.0, float(detection.confidence or 0.0)))
    quality_overall = fused.overall_confidence
    if geometry.touches_image_edge:
        quality_overall *= 0.85
    quality_overall = max(0.0, min(1.0, quality_overall))
    quality_status = (
        "good"
        if quality_overall >= 0.80
        else "usable"
        if quality_overall >= 0.60
        else "low_confidence"
    )

    try:
        status = AnalysisStatus(fused.status)
    except ValueError:
        status = AnalysisStatus.PARTIAL

    x1, y1, x2, y2 = detection.bbox or (
        geometry.bbox_x1,
        geometry.bbox_y1,
        geometry.bbox_x2,
        geometry.bbox_y2,
    )

    return UnifiedAnalysisResponse(
        segmentation_model_version=vision.health().get('version'),
        analysis_id=analysis_id,
        api_version=API_VERSION,
        schema_version=SCHEMA_VERSION,
        analysis_status=status,
        tree=TreeDetectionInfo(
            detected=True,
            segmentation_confidence=segmentation_conf,
            class_id=detection.class_id,
            class_name=detection.class_name,
            bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
            mask_area_ratio=geometry.mask_area_ratio,
            touches_image_edge=geometry.touches_image_edge,
            model_version=vision.model_version,
            model_path=str(vision.model_path) if vision.model_path else None,
        ),
        species=species,
        mechanical_profile=mechanical,
        pixel_measurements=PixelMeasurements(
            image_width_px=geometry.image_width_px,
            image_height_px=geometry.image_height_px,
            tree_height_px=round(geometry.tree_height_px, 2),
            crown_width_px=round(geometry.crown_width_px, 2),
            trunk_width_px_estimate=(
                round(geometry.trunk_width_px_estimate, 2)
                if geometry.trunk_width_px_estimate is not None
                else None
            ),
            mask_area_px=geometry.mask_area_px,
            bbox_width_px=round(geometry.bbox_width_px, 2),
            bbox_height_px=round(geometry.bbox_height_px, 2),
        ),
        calibration=CalibrationInfo(
            available=calibration_result.available,
            px_to_m=(round(calibration_result.px_to_m, 9) if calibration_result.px_to_m else None),
            source=calibration_result.source,
            confidence=round(calibration_result.confidence, 3),
            conflict=calibration_result.conflict,
            candidates=[
                ScaleCandidate(
                    source=c.source,
                    px_to_m=round(c.px_to_m, 9),
                    confidence=round(c.confidence, 3),
                )
                for c in calibration_result.candidates
            ],
            notes=calibration_result.notes,
        ),
        measurements=MeasurementsInfo(
            height=_measurement_model(fused.height),
            crown_width=_measurement_model(fused.crown_width),
            trunk_diameter=_measurement_model(fused.trunk_diameter),
        ),
        quality=QualityInfo(
            overall=round(quality_overall, 3),
            status=quality_status,
            segmentation=round(segmentation_conf, 3),
            calibration=round(calibration_result.confidence, 3),
            warnings=list(dict.fromkeys(warnings)),
        ),
        risk=RiskInfo(
            available=False,
            reason=(
                "mechanical_profile_not_validated"
                if not mechanical.available
                else "risk_engine_intentionally_disabled_during_measurement_validation"
            ),
        ),
        images=visual_payload,
        warnings=list(dict.fromkeys(warnings)),
        persisted=False,
    )


# Private contour submissions, isolated from verified training data.
from .corrections_api import router as corrections_router
app.include_router(corrections_router)
from .report_history import router as reports_router
app.include_router(reports_router)
