from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AnalysisStatus(str, Enum):
    MEASURED = "measured"
    PARTIAL = "partial_measurement"
    SCALE_REQUIRED = "absolute_scale_required"
    TREE_NOT_DETECTED = "tree_not_detected"


class MeasurementSource(str, Enum):
    AR = "ar"
    AR_VISION = "ar+vision"
    REFERENCE_VISION = "reference+vision"
    MANUAL_SCALE_VISION = "manual_scale+vision"
    VISION = "vision"
    UNAVAILABLE = "unavailable"


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class TreeDetectionInfo(BaseModel):
    detected: bool
    segmentation_confidence: Optional[float] = None
    class_id: Optional[int] = None
    class_name: Optional[str] = None
    bbox: Optional[BoundingBox] = None
    mask_area_ratio: Optional[float] = None
    touches_image_edge: Optional[bool] = None
    model_version: Optional[int] = None
    model_path: Optional[str] = None


class SpeciesCandidate(BaseModel):
    display_name: str
    scientific_name: Optional[str] = None
    common_names: List[str] = Field(default_factory=list)
    confidence: Optional[float] = None


class SpeciesInfo(BaseModel):
    status: str
    display_name: str = "Неизвестно"
    scientific_name: Optional[str] = None
    common_names: List[str] = Field(default_factory=list)
    confidence: Optional[float] = None
    top_results: List[SpeciesCandidate] = Field(default_factory=list)
    predicted_organs: List[Dict[str, Any]] = Field(default_factory=list)
    remaining_requests: Optional[int] = None
    latency_ms: Optional[float] = None
    message: Optional[str] = None


class MechanicalProfileInfo(BaseModel):
    available: bool
    scientific_name: Optional[str] = None
    source: Optional[str] = None
    properties: Dict[str, float] = Field(default_factory=dict)
    reason: Optional[str] = None


class PixelMeasurements(BaseModel):
    image_width_px: int
    image_height_px: int
    tree_height_px: Optional[float] = None
    crown_width_px: Optional[float] = None
    trunk_width_px_estimate: Optional[float] = None
    mask_area_px: Optional[int] = None
    bbox_width_px: Optional[float] = None
    bbox_height_px: Optional[float] = None


class ScaleCandidate(BaseModel):
    source: str
    px_to_m: float
    confidence: float


class CalibrationInfo(BaseModel):
    available: bool
    px_to_m: Optional[float] = None
    source: Optional[str] = None
    confidence: float = 0.0
    conflict: bool = False
    candidates: List[ScaleCandidate] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class MeasurementValue(BaseModel):
    value_m: Optional[float] = None
    value_px: Optional[float] = None
    source: MeasurementSource = MeasurementSource.UNAVAILABLE
    confidence: float = 0.0
    measurement_height_m: Optional[float] = None
    standard: Optional[str] = None
    notes: List[str] = Field(default_factory=list)


class MeasurementsInfo(BaseModel):
    height: MeasurementValue
    crown_width: MeasurementValue
    trunk_diameter: MeasurementValue


class QualityInfo(BaseModel):
    overall: float
    status: str
    segmentation: float
    calibration: float
    warnings: List[str] = Field(default_factory=list)


class RiskInfo(BaseModel):
    available: bool = False
    reason: str = "disabled_until_measurements_and_species_profiles_are_validated"


class ImagePayload(BaseModel):
    original_image_base64: Optional[str] = None
    annotated_image_base64: Optional[str] = None
    mask_image_base64: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None


class UnifiedAnalysisResponse(BaseModel):
    analysis_id: str
    api_version: str
    schema_version: str
    analysis_status: AnalysisStatus
    tree: TreeDetectionInfo
    species: SpeciesInfo
    mechanical_profile: MechanicalProfileInfo
    pixel_measurements: PixelMeasurements
    calibration: CalibrationInfo
    measurements: MeasurementsInfo
    quality: QualityInfo
    risk: RiskInfo
    images: ImagePayload
    warnings: List[str] = Field(default_factory=list)
    persisted: bool = False
