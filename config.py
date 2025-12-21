import os
from pydantic import BaseModel
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# путь к service account ключу
GOOGLE_DRIVE_CREDENTIALS = BASE_DIR / "credentials" / "arborscan-drive-sa.json"

# имя корневой папки на Google Drive
GOOGLE_DRIVE_ROOT_FOLDER = "ArborScan"

# подпапки
GOOGLE_DRIVE_UPLOADS_FOLDER = "uploads"
GOOGLE_DRIVE_MODELS_FOLDER = "models"

class Settings(BaseModel):
    # URL твоего бэкенда остаётся как есть

    # --- Погода (OpenWeatherMap One Call 3.0 / Current Weather) ---
    weather_api_key: str | None = os.getenv("dc825ffd002731568ec7766eafb54bc9")
    weather_base_url: str = os.getenv(
        "WEATHER_BASE_URL",
        "https://api.openweathermap.org/data/2.5/weather"
    )

    # --- Почвы (SoilGrids v2.0) ---
    soil_base_url: str = os.getenv(
        "SOILGRIDS_BASE_URL",
        "https://rest.isric.org/soilgrids/v2.0/properties/query"
    )

    # --- Обратное геокодирование (Nominatim / OSM) ---
    nominatim_base_url: str = os.getenv(
        "NOMINATIM_BASE_URL",
        "https://nominatim.openstreetmap.org/reverse"
    )
    nominatim_user_agent: str = os.getenv(
        "NOMINATIM_USER_AGENT",
        "arborscan-backend/1.0 (contact: your-email@example.com)"
    )

    # Флаг, чтобы можно было одним движением отключить расширенный анализ
    enable_environmental_analysis: bool = os.getenv(
        "ENABLE_ENV_ANALYSIS", "true"
    ).lower() == "true"

settings = Settings()
