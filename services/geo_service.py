from __future__ import annotations
from typing import Optional
from dataclasses import dataclass
from PIL import Image, ExifTags
import io
import requests

from config import settings

# --- Вспомогательные структуры ---

@dataclass
class Location:
    lat: float
    lon: float
    address: Optional[str] = None
    source: str = "exif"  # exif | manual | unknown

# --- EXIF GPS из JPEG/HEIC ---

def _convert_to_degrees(value):
    # value типа ((num, den), (num, den), (num, den))
    d = value[0][0] / value[0][1]
    m = value[1][0] / value[1][1]
    s = value[2][0] / value[2][1]
    return d + (m / 60.0) + (s / 3600.0)

def extract_gps_from_image(image_bytes: bytes) -> Optional[Location]:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        exif_data = image._getexif()
        if not exif_data:
            return None

        exif = {
            ExifTags.TAGS.get(k, k): v
            for k, v in exif_data.items()
        }

        gps_info = exif.get("GPSInfo")
        if not gps_info:
            return None

        gps_tags = {}
        for key in gps_info.keys():
            name = ExifTags.GPSTAGS.get(key, key)
            gps_tags[name] = gps_info[key]

        lat = _convert_to_degrees(gps_tags["GPSLatitude"])
        lon = _convert_to_degrees(gps_tags["GPSLongitude"])

        if gps_tags.get("GPSLatitudeRef") == "S":
            lat = -lat
        if gps_tags.get("GPSLongitudeRef") == "W":
            lon = -lon

        return Location(lat=lat, lon=lon, source="exif")
    except Exception:
        # Ничего не ломаем, просто возвращаем None
        return None

# --- Обратное геокодирование (координаты → человекочитаемый адрес) ---

def reverse_geocode(location: Location) -> Location:
    if not settings.nominatim_base_url:
        return location

    try:
        params = {
            "lat": location.lat,
            "lon": location.lon,
            "format": "jsonv2",
            "addressdetails": 1,
        }
        headers = {
            "User-Agent": settings.nominatim_user_agent
        }
        resp = requests.get(
            settings.nominatim_base_url,
            params=params,
            headers=headers,
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        address = data.get("display_name")
        return Location(
            lat=location.lat,
            lon=location.lon,
            address=address,
            source=location.source,
        )
    except Exception:
        return location
