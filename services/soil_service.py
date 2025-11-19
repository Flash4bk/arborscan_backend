from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import requests

from config import settings
from services.geo_service import Location

@dataclass
class Soil:
    clay: Optional[float] = None        # %
    sand: Optional[float] = None        # %
    silt: Optional[float] = None        # %
    organic_carbon: Optional[float] = None  # g/kg или т.п.
    ph: Optional[float] = None
    source: str = "soilgrids"

def get_soil(location: Location) -> Optional[Soil]:
    try:
        params = {
            "lon": location.lon,
            "lat": location.lat,
            # Берём верхний слой 0-5 см
            "property": "clay,sand,silt,soc,phh2o",
            "depth": "0-5cm",
        }
        resp = requests.get(
            settings.soil_base_url, params=params, timeout=7
        )
        resp.raise_for_status()
        data = resp.json()

        layers = data.get("properties", {}).get("layers", [])
        values = {}
        for layer in layers:
            name = layer.get("name")
            # Берём среднее значение (mean) по первому depth_interval
            intervals = layer.get("depths", [])
            if not intervals:
                continue
            mean_val = intervals[0].get("values", {}).get("mean")
            values[name] = mean_val

        return Soil(
            clay=values.get("clay"),
            sand=values.get("sand"),
            silt=values.get("silt"),
            organic_carbon=values.get("soc"),
            ph=values.get("phh2o"),
        )
    except Exception:
        # Бывают точки, где SoilGrids возвращает null (города и т.д.) – просто игнорируем
        return None
