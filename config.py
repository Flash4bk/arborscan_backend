from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    value = Path(raw).expanduser() if raw and raw.strip() else default
    return value.resolve()


_FILE_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _env_path("PROJECT_ROOT", _FILE_ROOT)
_MODEL_DIR = _env_path("MODEL_DIR", _PROJECT_ROOT / "models")
_MODEL_CACHE_DIR = _env_path("MODEL_CACHE_DIR", Path("/tmp/arborscan-models"))


class Settings(BaseModel):
    # -----------------------------
    # Project / model paths
    # -----------------------------
    project_root: Path = _PROJECT_ROOT
    model_dir: Path = _MODEL_DIR
    model_cache_dir: Path = _MODEL_CACHE_DIR

    active_model_version: int = _env_int("ACTIVE_MODEL_VERSION", 0)
    auto_select_latest_local_model: bool = _env_bool(
        "AUTO_SELECT_LATEST_LOCAL_MODEL", True
    )
    model_check_interval_sec: float = _env_float(
        "MODEL_CHECK_INTERVAL_SEC", 2.0
    )
    model_min_size_bytes: int = _env_int(
        "MODEL_MIN_SIZE_BYTES", 1_000_000
    )
    model_download_timeout_sec: int = _env_int(
        "MODEL_DOWNLOAD_TIMEOUT_SEC", 180
    )

    stick_model_filename: str = os.getenv(
        "STICK_MODEL_FILENAME", "stick_model.pt"
    )
    stick_model_object: str = os.getenv(
        "STICK_MODEL_OBJECT", "stick_model.pt"
    )
    tree_model_task: str = os.getenv("TREE_MODEL_TASK", "segment")
    stick_model_task: str = os.getenv("STICK_MODEL_TASK", "detect")

    # Optional SHA-256 checks. Leave empty when hashes are not configured.
    stick_model_sha256: Optional[str] = (
        os.getenv("STICK_MODEL_SHA256") or None
    )

    # -----------------------------
    # Supabase
    # -----------------------------
    supabase_url: Optional[str] = os.getenv("SUPABASE_URL") or None
    supabase_service_key: Optional[str] = (
        os.getenv("SUPABASE_SERVICE_KEY")
        or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or None
    )

    supabase_bucket_inputs: str = os.getenv(
        "SUPABASE_BUCKET_INPUTS", "arborscan-inputs"
    )
    supabase_bucket_predictions: str = os.getenv(
        "SUPABASE_BUCKET_PRED", "arborscan-predictions"
    )
    supabase_bucket_meta: str = os.getenv(
        "SUPABASE_BUCKET_META", "arborscan-meta"
    )
    supabase_bucket_verified: str = os.getenv(
        "SUPABASE_BUCKET_VERIFIED", "arborscan-verified"
    )
    supabase_bucket_raw: str = os.getenv(
        "SUPABASE_BUCKET_RAW", "arborscan-raw"
    )
    supabase_bucket_models: str = os.getenv(
        "SUPABASE_BUCKET_MODELS", "arborscan-models"
    )
    supabase_bucket_datasets: str = os.getenv(
        "SUPABASE_BUCKET_DATASETS", "arborscan-datasets"
    )
    supabase_enable_queue: bool = _env_bool(
        "SUPABASE_ENABLE_QUEUE", False
    )

    # -----------------------------
    # External services
    # -----------------------------
    plantnet_api_key: Optional[str] = os.getenv("PLANTNET_API_KEY") or None

    weather_api_key: Optional[str] = (
        os.getenv("WEATHER_API_KEY")
        or os.getenv("OPENWEATHER_API_KEY")
        or os.getenv("OPENWEATHERMAP_API_KEY")
        or None
    )
    weather_base_url: str = os.getenv(
        "WEATHER_BASE_URL",
        "https://api.openweathermap.org/data/2.5/weather",
    )

    soil_base_url: str = os.getenv(
        "SOILGRIDS_BASE_URL",
        "https://rest.isric.org/soilgrids/v2.0/properties/query",
    )

    nominatim_base_url: str = os.getenv(
        "NOMINATIM_BASE_URL",
        "https://nominatim.openstreetmap.org/reverse",
    )
    nominatim_user_agent: str = os.getenv(
        "NOMINATIM_USER_AGENT",
        "arborscan-backend/2.8 (contact: admin@example.com)",
    )

    enable_environmental_analysis: bool = _env_bool(
        "ENABLE_ENV_ANALYSIS", True
    )

    # Loading rembg during startup consumes RAM and makes cold starts slower.
    # By default it is loaded only when a request explicitly enables rembg.
    preload_rembg: bool = _env_bool("PRELOAD_REMBG", False)

    @property
    def stick_model_path(self) -> Path:
        return self.model_dir / self.stick_model_filename

    def tree_model_sha256(self, version: int) -> Optional[str]:
        value = os.getenv(f"MODEL_V{version}_SHA256")
        return value.strip().lower() if value and value.strip() else None

    def ensure_runtime_dirs(self) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_runtime_dirs()
