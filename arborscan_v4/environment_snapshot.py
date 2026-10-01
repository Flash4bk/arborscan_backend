"""Optional, versioned AS-09 metadata inside an immutable report snapshot.

Validation does not attest to a client supplied location or provider result.
No lookup is performed while saving, reopening, listing or exporting a report.
"""
import math
from datetime import datetime


def finite(value, minimum=None, maximum=None):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('Non-finite number')
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError('Number out of range')
    return value


def point(value, crs=False):
    if not isinstance(value, dict) or set(value) != ({'lat', 'lon', 'crs'} if crs else {'lat', 'lon'}):
        raise ValueError('Invalid point')
    finite(value['lat'], -90, 90)
    finite(value['lon'], -180, 180)
    if crs and value['crs'] != 'EPSG:4326':
        raise ValueError('Unsupported coordinate system')
    return value


def timestamp(value, nullable=False):
    if value is None and nullable:
        return
    if not isinstance(value, str) or len(value) > 40:
        raise ValueError('Invalid timestamp')
    if datetime.fromisoformat(value.replace('Z', '+00:00')).utcoffset() is None:
        raise ValueError('Timestamp requires offset')


WEATHER_FIELDS = {
    'temperature_c': (-100, 100), 'wind_speed_m_s': (0, 200),
    'wind_gust_m_s': (0, 250), 'wind_direction_deg': (0, 360),
    'pressure_hpa': (0, 1500), 'relative_humidity_pct': (0, 100),
}
SOIL_FIELDS = {'clay': ('%', 100), 'sand': ('%', 100), 'silt': ('%', 100),
               'soc': ('g/kg', 1000), 'phh2o': ('pH', 14)}
REASONS = {'provider_not_configured', 'provider_timeout', 'provider_unavailable',
           'provider_rate_limited', 'provider_invalid_response', 'provider_authentication',
           'provider_busy', 'no_coverage', 'partial_data'}


def validate_environment(environment):
    if environment is None:
        return
    if not isinstance(environment, dict):
        raise ValueError('Invalid environment')
    if 'version' not in environment:
        # Preserve the original history envelope used by existing clients.
        for field in ('gps', 'weather', 'soil'):
            item = environment.get(field)
            if item is not None and (not isinstance(item, dict) or 'value' not in item
                                     or not item.get('source') or not item.get('retrieved_at')):
                raise ValueError('Invalid legacy environment envelope')
        return
    if type(environment['version']) is not int or environment['version'] != 1:
        raise ValueError('Unsupported environment version')
    if set(environment) - {'version', 'gps', 'weather', 'soil'}:
        raise ValueError('Unknown environment field')
    gps = environment.get('gps')
    if gps is not None:
        if not isinstance(gps, dict) or set(gps) - {
            'value', 'source', 'retrieved_at', 'observed_at', 'captured_local',
            'accuracy_m', 'is_last_known', 'is_approximate', 'position_kind',
        }:
            raise ValueError('Invalid location provenance')
        point(gps['value'], crs=True)
        if gps['source'] not in ('exif', 'device', 'manual'):
            raise ValueError('Invalid location source')
        timestamp(gps['retrieved_at'])
        timestamp(gps.get('observed_at'), nullable=True)
        local = gps.get('captured_local')
        if local is not None:
            if (not isinstance(local, str) or len(local) > 40
                    or datetime.fromisoformat(local).utcoffset() is not None):
                raise ValueError('EXIF local time must not invent a timezone')
        if gps.get('accuracy_m') is not None:
            finite(gps['accuracy_m'], 0, 1_000_000)
        if type(gps.get('is_last_known')) is not bool:
            raise ValueError('Last-known status required')
        if gps.get('is_approximate') is not None and type(gps['is_approximate']) is not bool:
            raise ValueError('Invalid approximate permission flag')
        if gps.get('position_kind') not in ('camera_or_device', 'tree'):
            raise ValueError('Location target required')
    for field in ('weather', 'soil'):
        envelope = environment.get(field)
        if envelope is None:
            continue
        _validate_conditions(field, envelope)
        if gps is None or any(envelope['request_point'][k] != gps['value'][k] for k in ('lat', 'lon')):
            raise ValueError('Conditions belong to a different point')


def _validate_conditions(field, item):
    common = {'value', 'source', 'retrieved_at', 'data_at', 'kind', 'request_point',
              'status', 'reason', 'cached', 'attribution', 'limitations', 'units'}
    extra = {'dataset_version', 'resolution_m', 'access_method'} if field == 'soil' else set()
    if not isinstance(item, dict) or set(item) - common - extra:
        raise ValueError('Invalid conditions envelope')
    if item.get('source') != ('OpenWeather' if field == 'weather' else 'SoilGrids'):
        raise ValueError('Unknown conditions source')
    if item.get('kind') != ('current' if field == 'weather' else 'modelled_grid'):
        raise ValueError('Wrong conditions time/model kind')
    point(item['request_point'])
    timestamp(item['retrieved_at'])
    timestamp(item.get('data_at'), nullable=True)
    if type(item.get('cached')) is not bool or item.get('status') not in ('ok', 'partial', 'unavailable'):
        raise ValueError('Invalid conditions status')
    if item.get('reason') is not None and item['reason'] not in REASONS:
        raise ValueError('Unknown provider error')
    if not isinstance(item.get('attribution'), str) or not 1 <= len(item['attribution']) <= 1500:
        raise ValueError('Source attribution required')
    limitations = item.get('limitations', [])
    if not isinstance(limitations, list) or len(limitations) > 12 or any(
            not isinstance(x, str) or len(x) > 1000 for x in limitations):
        raise ValueError('Invalid source limitations')
    if item['status'] == 'unavailable':
        if item.get('value') is not None or not item.get('reason'):
            raise ValueError('Unavailable conditions must not contain invented values')
        return
    values = item.get('value')
    if not isinstance(values, dict):
        raise ValueError('Missing conditions data')
    if field == 'weather':
        if set(values) - WEATHER_FIELDS.keys() or not any(v is not None for v in values.values()):
            raise ValueError('Unknown or empty weather values')
        timestamp(item['data_at'])
        if item.get('units') != 'C_m/s_degrees_hPa_percent':
            raise ValueError('Invalid weather units')
        for key, value in values.items():
            if value is not None:
                finite(value, *WEATHER_FIELDS[key])
    else:
        if item.get('dataset_version') != '2.0' or item.get('resolution_m') != 250:
            raise ValueError('Unsupported soil dataset')
        if item.get('access_method') not in ('WCS_nearest_cell', 'REST'):
            raise ValueError('Unknown soil sampling method')
        if set(values) != {'properties'} or not isinstance(values['properties'], list) or not 1 <= len(values['properties']) <= 5:
            raise ValueError('Invalid soil properties')
        names = set()
        for prop in values['properties']:
            if not isinstance(prop, dict) or set(prop) != {'name', 'value', 'unit', 'depth_cm', 'q05', 'q95'}:
                raise ValueError('Invalid soil property')
            name = prop['name']
            if name not in SOIL_FIELDS or name in names:
                raise ValueError('Unknown or duplicated soil property')
            names.add(name)
            unit, maximum = SOIL_FIELDS[name]
            if prop['unit'] != unit or prop['depth_cm'] != [0, 5]:
                raise ValueError('Invalid soil unit or depth')
            finite(prop['value'], 0, maximum)
            for quantile in ('q05', 'q95'):
                if prop[quantile] is not None:
                    finite(prop[quantile], 0, maximum)
            if prop['q05'] is not None and prop['q95'] is not None and prop['q05'] > prop['q95']:
                raise ValueError('Invalid soil uncertainty interval')
