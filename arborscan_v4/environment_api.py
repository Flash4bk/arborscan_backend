"""Explicit current conditions lookup; independent of analysis and report reads.

Only fixed provider URLs, no secret in a response or an exception. SoilGrids
REST is paused: use the same provider's documented WCS nearest-cell service.
Caches are bounded, process-local and account-scoped; no report is mutated.
"""
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from copy import deepcopy
from datetime import datetime, timezone
from io import BytesIO
import json
from threading import Lock
import time

import requests
from PIL import Image, UnidentifiedImageError
from fastapi import APIRouter, Depends, HTTPException, Query, Response

from .corrections_api import current_user
from .environment_snapshot import finite, point, WEATHER_FIELDS, SOIL_FIELDS

router = APIRouter(prefix='/v4/environment', tags=['survey environment'])
WEATHER_URL = 'https://api.openweathermap.org/data/2.5/weather'
SOIL_URL = 'https://maps.isric.org/mapserv'
POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix='environment-provider')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class ProviderFailure(Exception):
    def __init__(self, reason):
        self.reason = reason


def provider_bytes(url, params, get=requests.get):
    # Do not pass exception text/URLs to callers: OpenWeather's key is a query
    # parameter. No automatic redirects, retries or disabling of TLS checks.
    start = time.monotonic()
    try:
        with get(url, params=params, timeout=(2, 4), allow_redirects=False, stream=True) as response:
            if response.status_code in (401, 403):
                raise ProviderFailure('provider_authentication')
            if response.status_code == 429:
                raise ProviderFailure('provider_rate_limited')
            if response.status_code != 200:
                raise ProviderFailure('provider_unavailable')
            chunks, size = [], 0
            for chunk in response.iter_content(16384):
                size += len(chunk)
                if size > 65536:
                    raise ProviderFailure('provider_invalid_response')
                if time.monotonic() - start > 6:
                    raise ProviderFailure('provider_timeout')
                chunks.append(chunk)
            return b''.join(chunks)
    except requests.Timeout:
        raise ProviderFailure('provider_timeout') from None
    except requests.RequestException:
        raise ProviderFailure('provider_unavailable') from None


def weather_values(raw):
    try:
        data = json.loads(raw)
        measured_at = finite(data['dt'], 0)
        at = datetime.fromtimestamp(measured_at, timezone.utc).isoformat()
        main, wind = data.get('main', {}), data.get('wind', {})
        values = dict(zip(WEATHER_FIELDS, [main.get('temp'), wind.get('speed'),
            wind.get('gust'), wind.get('deg'), main.get('pressure'), main.get('humidity')]))
        for key, value in values.items():
            if value is not None:
                finite(value, *WEATHER_FIELDS[key])
        if not any(v is not None for v in values.values()):
            raise ValueError()
        return values, at
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError, OSError):
        raise ProviderFailure('provider_invalid_response') from None


def soil_value(raw, name):
    try:
        with Image.open(BytesIO(raw)) as image:
            if image.format != 'TIFF' or image.size != (3, 3) or image.mode not in ('I', 'I;16', 'I;16S'):
                raise ValueError()
            value = image.getpixel((1, 1))
            nodata = image.tag_v2.get(42113)
            if value == -32768 or nodata is not None and value == float(nodata):
                return None
            unit, maximum = SOIL_FIELDS[name]
            # ISRIC table: clay/sand/silt g/kg -> %, soc dg/kg -> g/kg,
            # phh2o pH*10 -> pH. All five stored integers divide by ten.
            converted = finite(value, 0) / 10
            finite(converted, 0, maximum)
            return {'name': name, 'value': converted, 'unit': unit,
                    'depth_cm': [0, 5], 'q05': None, 'q95': None}
    except (ValueError, TypeError, KeyError, OSError, UnidentifiedImageError, Image.DecompressionBombError):
        raise ProviderFailure('provider_invalid_response') from None


def fetch_weather(lat, lon, key, get=requests.get):
    raw = provider_bytes(WEATHER_URL, {'lat': lat, 'lon': lon, 'appid': key,
                                     'units': 'metric', 'lang': 'ru'}, get)
    return weather_values(raw)


def fetch_soil(lat, lon, name, get=requests.get):
    # Three output pixels place the centre pixel at exactly the requested
    # position. Reprojection uses NEAREST; no neighbouring-cell imputation.
    params = {'map': f'/map/{name}.map', 'SERVICE': 'WCS', 'VERSION': '1.0.0',
              'REQUEST': 'GetCoverage', 'COVERAGE': f'{name}_0-5cm_mean',
              'CRS': 'EPSG:4326', 'RESPONSE_CRS': 'EPSG:4326',
              'BBOX': f'{lon - .002},{lat - .002},{lon + .002},{lat + .002}',
              'WIDTH': 3, 'HEIGHT': 3, 'FORMAT': 'GEOTIFF_INT16', 'INTERPOLATION': 'NEAREST'}
    return soil_value(provider_bytes(SOIL_URL, params, get), name)


def envelope(kind, lat, lon):
    weather = kind == 'weather'
    result = {'value': None, 'source': 'OpenWeather' if weather else 'SoilGrids',
        'retrieved_at': utc_now(), 'data_at': None,
        'kind': 'current' if weather else 'modelled_grid',
        'request_point': {'lat': lat, 'lon': lon}, 'status': 'unavailable',
        'reason': None, 'cached': False,
        'attribution': ('OpenWeather https://openweathermap.org/' if weather else
            'ISRIC — World Soil Information; SoilGrids 2.0, CC BY 4.0; '
            'Poggio et al. (2021), https://doi.org/10.5194/soil-7-217-2021'),
        'limitations': (['Текущие условия по точке запроса; это не погода на момент съёмки старого фото.',
            'Данные провайдера объединяют модели, наблюдения и спутниковые данные; не измерение у дерева.']
            if weather else ['Сетка 250 м, модельная оценка слоя 0–5 см; не проба почвы у корня.',
                'Интервал неопределённости 90% у источника существует, но в этом запросе не получен.',
                'Дата отбора пробы и дата обновления конкретной ячейки неизвестны.',
                'Не определяет состояние корней, несущую способность или устойчивость дерева.'])}
    if weather:
        result['units'] = 'C_m/s_degrees_hPa_percent'
    else:
        result.update(dataset_version='2.0', resolution_m=250, access_method='WCS_nearest_cell')
    return result


class EnvironmentService:
    def __init__(self, weather=fetch_weather, soil=fetch_soil, clock=time.monotonic,
                 key=None, pool=POOL):
        self.weather, self.soil, self.clock, self.key, self.pool = weather, soil, clock, key, pool
        self.lock = Lock()
        self.cache = OrderedDict()
        self.budgets = {}

    def _allow(self, name, count, limit):
        now = self.clock()
        with self.lock:
            queue = self.budgets.setdefault(name, deque())
            while queue and queue[0] <= now - 60:
                queue.popleft()
            if len(queue) + count > limit:
                return False
            queue.extend([now] * count)
            # Bound idle account entries, never evict an active rate window.
            if len(self.budgets) > 1024:
                self.budgets = {key: q for key, q in self.budgets.items() if q and q[-1] > now - 60}
            return True

    def _cached(self, key):
        with self.lock:
            entry = self.cache.get(key)
            if entry is None:
                return None
            expiry, value = entry
            if expiry <= self.clock():
                del self.cache[key]
                return None
            self.cache.move_to_end(key)
            result = deepcopy(value)
            result['cached'] = True
            return result

    def _store(self, key, value):
        ttl = (600 if key[1] == 'weather' else 86400) if value['status'] == 'ok' else (300 if value['status'] == 'partial' else 30)
        with self.lock:
            current = self.cache.get(key)
            if (current is not None and current[0] > self.clock()
                    and current[1]['status'] == 'ok' and value['status'] != 'ok'):
                # A parallel rate-limited request must not evict data obtained
                # by the already-running request for this exact owner/point.
                return
            self.cache[key] = (self.clock() + ttl, deepcopy(value))
            self.cache.move_to_end(key)
            while len(self.cache) > 256:
                self.cache.popitem(last=False)

    def lookup(self, owner, lat, lon):
        point({'lat': lat, 'lon': lon})
        if not self._allow(('owner', owner), 1, 10):
            raise HTTPException(429, 'Too many environment requests; retry in one minute', headers={'Retry-After': '60'})
        result, jobs = {}, {}
        for kind in ('weather', 'soil'):
            cache_key = (owner, kind, lat, lon)
            cached = self._cached(cache_key)
            if cached is not None:
                result[kind] = cached
                continue
            item = result[kind] = envelope(kind, lat, lon)
            if kind == 'weather':
                if self.key is None:
                    from config import settings
                    api_key = settings.weather_api_key
                else:
                    api_key = self.key
                if not api_key:
                    item['reason'] = 'provider_not_configured'
                elif not self._allow('weather', 1, 30):
                    item['reason'] = 'provider_rate_limited'
                else:
                    jobs[kind] = self.pool.submit(self.weather, lat, lon, api_key)
            elif abs(lat) > 89.998 or abs(lon) > 179.998:
                item['reason'] = 'no_coverage'
            elif not self._allow('soil_wcs', 5, 5):
                item['reason'] = 'provider_rate_limited'
            else:
                jobs[kind] = {name: self.pool.submit(self.soil, lat, lon, name) for name in SOIL_FIELDS}
        deadline = time.monotonic() + 7
        for kind, future in jobs.items():
            item = result[kind]
            if kind == 'weather':
                try:
                    item['value'], item['data_at'] = future.result(timeout=max(0, deadline - time.monotonic()))
                    item.update(status='ok', reason=None)
                except FutureTimeout:
                    future.cancel()
                    item['reason'] = 'provider_timeout'
                except ProviderFailure as error:
                    item['reason'] = error.reason
            else:
                properties, errors = [], []
                for task in future.values():
                    try:
                        value = task.result(timeout=max(0, deadline - time.monotonic()))
                        if value is not None:
                            properties.append(value)
                    except FutureTimeout:
                        task.cancel()
                        errors.append('provider_timeout')
                    except ProviderFailure as error:
                        errors.append(error.reason)
                if properties:
                    item.update(value={'properties': properties}, status='partial' if errors or len(properties) < 5 else 'ok',
                                reason='partial_data' if errors or len(properties) < 5 else None)
                else:
                    item['reason'] = errors[0] if errors else 'no_coverage'
            item['retrieved_at'] = utc_now()
        for kind, item in result.items():
            if not item['cached']:
                self._store((owner, kind, lat, lon), item)
        return {'environment_version': 1, **result}


SERVICE = EnvironmentService()


@router.get('')
def current_environment(response: Response, lat: float = Query(..., ge=-90, le=90),
                        lon: float = Query(..., ge=-180, le=180), owner=Depends(current_user)):
    # In-memory cache is owner scoped; proxies/browser caches must never reuse
    # an authenticated response across sessions.
    response.headers['Cache-Control'] = 'private, no-store'
    response.headers['Vary'] = 'Authorization'
    try:
        point({'lat': lat, 'lon': lon})
    except ValueError:
        raise HTTPException(422, 'Invalid WGS84 coordinates') from None
    return SERVICE.lookup(owner, lat, lon)
