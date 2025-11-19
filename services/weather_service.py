from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import requests

from config import settings
from services.geo_service import Location

@dataclass
class Weather:
    temperature: Optional[float] = None  # °C
    wind_speed: Optional[float] = None   # м/с
    wind_gust: Optional[float] = None    # м/с
    wind_deg: Optional[int] = None       # направление, градусы
    pressure: Optional[int] = None       # гПа
    humidity: Optional[int] = None       # %
    provider: str = "openweathermap"

def get_weather(location: Location) -> Optional[Weather]:
    if not settings.weather_api_key:
        return None

    params = {
        "lat": location.lat,
        "lon": location.lon,
        "appid": settings.weather_api_key,
        "units": "metric",
        "lang": "ru",
    }

    try:
        resp = requests.get(
            settings.weather_base_url, params=params, timeout=5
        )
        resp.raise_for_status()
        data = resp.json()

        wind = data.get("wind", {})
        main = data.get("main", {})

        return Weather(
            temperature=main.get("temp"),
            wind_speed=wind.get("speed"),
            wind_gust=wind.get("gust"),
            wind_deg=wind.get("deg"),
            pressure=main.get("pressure"),
            humidity=main.get("humidity"),
        )
    except Exception:
        return None
