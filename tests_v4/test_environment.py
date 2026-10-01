"""AS-09: fixed providers, units, timeout/quota, immutable account snapshots."""
from concurrent.futures import Future, TimeoutError as FutureTimeout
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from PIL import Image, TiffImagePlugin
import pytest
import requests

from arborscan_v4 import environment_api as api, report_history as history
from arborscan_v4.environment_snapshot import point, validate_environment, SOIL_FIELDS

OWNER = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa'
OTHER = 'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb'
AT = '2026-10-01T12:00:00Z'
IMAGE = (Path(__file__).parents[1] / 'arborscan_app/test/fixtures/reference_exif6.jpg').read_bytes()


def gps(lat=0, lon=30):
    return {'value': {'lat': lat, 'lon': lon, 'crs': 'EPSG:4326'}, 'source': 'manual',
            'retrieved_at': AT, 'observed_at': None, 'accuracy_m': None,
            'is_last_known': False, 'position_kind': 'tree'}


def tiff(value):
    image = Image.new('I', (3, 3), value)
    image.putpixel((0, 0), 800)  # Centre, not the first cell, must be sampled.
    out = BytesIO()
    tags = TiffImagePlugin.ImageFileDirectory_v2()
    tags[42113] = '-32768'
    image.save(out, format='TIFF', tiffinfo=tags)
    return out.getvalue()


def weather(lat, lon, key):
    return {'temperature_c': 12.5, 'wind_speed_m_s': 0, 'wind_gust_m_s': None}, AT


def soil(lat, lon, name):
    return api.soil_value(tiff(50), name)


class ImmediatePool:
    def submit(self, function, *args):
        result = Future()
        try:
            result.set_result(function(*args))
        except Exception as error:
            result.set_exception(error)
        return result


def service(**kwargs):
    return api.EnvironmentService(weather=kwargs.pop('weather', weather),
        soil=kwargs.pop('soil', soil), key=kwargs.pop('key', 'private-test-key'),
        pool=ImmediatePool(), **kwargs)


def conditions():
    data = service().lookup(OWNER, 0, 30)
    return {'version': 1, 'gps': gps(), 'weather': data['weather'], 'soil': data['soil']}


def snapshot(environment):
    return {'version': 1, 'kind': 'v4', 'change_source': 'manual', 'captured_at': AT,
            'report': {'measurements': {'height': {'value_m': 4.5}}}, 'environment': environment}


@pytest.mark.parametrize('lat,lon', [(0, 0), (0, 45), (-90, 180), (90, -180)])
def test_wgs84_accepts_zero_and_endpoints(lat, lon):
    assert point({'lat': lat, 'lon': lon}) == {'lat': lat, 'lon': lon}


@pytest.mark.parametrize('lat,lon', [(91, 0), (0, 181), (float('nan'), 0),
    (0, float('inf')), (True, 1), ('0', 0), (None, 5)])
def test_wgs84_rejects_nonfinite_missing_wrong_type(lat, lon):
    with pytest.raises(ValueError):
        point({'lat': lat, 'lon': lon})


def test_new_environment_preserves_source_units_and_original_measurements():
    original = conditions()
    result = history.validate_snapshot(json.dumps(snapshot(original)), IMAGE)
    assert result['environment'] == original
    assert result['report']['measurements']['height']['value_m'] == 4.5
    assert result['environment']['weather']['value']['wind_speed_m_s'] == 0
    assert result['environment']['weather']['value']['wind_gust_m_s'] is None
    assert result['environment']['soil']['value']['properties'][0]['depth_cm'] == [0, 5]
    assert result['report']['beta_kg_s'] is None


def test_explicit_environment_revision_preserves_photo_binding_and_old_snapshot():
    before = snapshot(conditions())
    old = history.validate_snapshot(json.dumps(before), IMAGE)
    after = deepcopy(before)
    after.update(change_source='environment_edit')
    after['environment']['gps'] = gps(1, 30)
    after['environment'].update(weather=None, soil=None)
    new = history.validate_snapshot(json.dumps(after), IMAGE)
    assert old['environment']['gps']['value']['lat'] == 0
    assert new['environment']['gps']['value']['lat'] == 1
    assert old['image'] == new['image'] and old['report'] == new['report']


@pytest.mark.parametrize('mutate', [
    lambda e: e['gps']['value'].update(crs='EPSG:3857'),
    lambda e: e['gps'].update(accuracy_m=-1),
    lambda e: e['gps'].update(source='inferred_species'),
    lambda e: e['gps'].update(observed_at='2026-10-01T12:00:00'),
    lambda e: e['gps'].update(is_last_known=1),
    lambda e: e['gps'].update(is_approximate='true'),
    lambda e: e['gps']['value'].update(lon=31),  # stale asynchronous provider response
    lambda e: e.update(gps=None),
    lambda e: e['weather'].update(kind='historical'),
    lambda e: e['weather'].update(units='fahrenheit_mph'),
    lambda e: e['weather'].update(data_at=None),
    lambda e: e['weather']['value'].update(wind_speed_m_s=-2),
    lambda e: e['soil']['value']['properties'][0].update(unit='g/kg'),
    lambda e: e['soil']['value']['properties'][0].update(depth_cm=[5, 15]),
    lambda e: e['soil']['value']['properties'].append(e['soil']['value']['properties'][0]),
    lambda e: e['soil'].update(resolution_m=1),
    lambda e: e['weather'].update(status='unavailable', reason='provider_timeout'),
    lambda e: e.update(version=True),
])
def test_invalid_or_stale_environment_cannot_be_saved(mutate):
    env = conditions()
    mutate(env)
    with pytest.raises(HTTPException) as error:
        history.validate_snapshot(json.dumps(snapshot(env)), IMAGE)
    assert error.value.status_code == 422


def test_exif_without_timezone_keeps_local_time_without_invented_precision():
    env = {'version': 1, 'gps': gps()}
    env['gps'].update(source='exif', position_kind='camera_or_device',
                      captured_local='2025-02-03T04:05:06', observed_at=None)
    validate_environment(env)
    assert env['gps']['accuracy_m'] is None
    env['gps']['captured_local'] += 'Z'
    with pytest.raises(ValueError):
        validate_environment(env)


def test_legacy_envelopes_remain_readable_and_never_refresh_conditions():
    old = {'weather': {'value': {'temperature': 8}, 'source': 'historical_fixture', 'retrieved_at': AT}}
    with patch('requests.get') as provider:
        saved = history.validate_snapshot(json.dumps(snapshot(old)), IMAGE)
        store = object.__new__(history.ReportStore)
        with patch.object(store, 'request', return_value=[{'payload_sha256': 'old'}]), \
             patch.object(store, 'read_blob', return_value=saved):
            assert store.get(OWNER, OWNER)['snapshot']['environment'] == old
        provider.assert_not_called()


def test_saved_index_contains_same_environment_and_owner_bound_snapshot():
    original = conditions()
    store = object.__new__(history.ReportStore)
    with patch.object(store, 'ready'), patch.object(store, 'write_blob', return_value='digest'), \
         patch.object(store, 'request', return_value={}) as rpc:
        payload = history.validate_snapshot(json.dumps(snapshot(original)), IMAGE)
        store.save(OWNER, OWNER, OWNER, None, payload)
        call = deepcopy(rpc.call_args.kwargs['json'])
        original['gps']['value']['lon'] = 32
        assert call['p_owner'] == OWNER
        assert call['p_summary']['environment'] == payload['environment']
        assert call['p_summary']['environment']['gps']['value']['lon'] == 30
        assert 'original_base64' not in json.dumps(call['p_summary'])


@pytest.mark.parametrize('name,raw,expected,unit', [('clay', 295, 29.5, '%'),
    ('sand', 520, 52, '%'), ('silt', 185, 18.5, '%'), ('soc', 187, 18.7, 'g/kg'),
    ('phh2o', 64, 6.4, 'pH')])
def test_soil_actual_raw_units_and_center_pixel(name, raw, expected, unit):
    result = api.soil_value(tiff(raw), name)
    assert result == {'name': name, 'value': expected, 'unit': unit, 'depth_cm': [0, 5], 'q05': None, 'q95': None}


def test_soil_nodata_is_absent_but_actual_zero_is_not():
    assert api.soil_value(tiff(-32768), 'clay') is None
    assert api.soil_value(tiff(0), 'clay')['value'] == 0
    with pytest.raises(api.ProviderFailure):
        api.soil_value(tiff(10001), 'clay')


def test_weather_metric_values_and_timestamp_not_photo_date():
    values, at = api.weather_values(json.dumps({'dt': 1790856000,
        'main': {'temp': 10, 'humidity': 100, 'pressure': 998},
        'wind': {'speed': 0, 'gust': 4.2, 'deg': 0}}))
    assert values['temperature_c'] == 10 and values['wind_gust_m_s'] == 4.2
    assert values['wind_direction_deg'] == 0
    assert at.endswith('+00:00')
    values, _ = api.weather_values(json.dumps({'dt': 1790856000, 'main': {'temp': 0}}))
    assert values['wind_speed_m_s'] is None


@pytest.mark.parametrize('bad', ['{}', '{bad', '{"dt": 5, "main":{"temp":NaN}}',
                                '{"dt": true,"main":{"temp":8}}'])
def test_malformed_provider_weather_is_not_success(bad):
    with pytest.raises(api.ProviderFailure) as error:
        api.weather_values(bad)
    assert error.value.reason == 'provider_invalid_response'


class FakeResponse:
    def __init__(self, status, body):
        self.status_code, self.body = status, body
    def __enter__(self):
        return self
    def __exit__(self, *_):
        pass
    def iter_content(self, size):
        yield self.body


@pytest.mark.parametrize('status,reason', [(401, 'provider_authentication'), (403, 'provider_authentication'),
    (429, 'provider_rate_limited'), (500, 'provider_unavailable'), (302, 'provider_unavailable')])
def test_http_errors_do_not_expose_secrets_or_follow_redirects(status, reason):
    get = Mock(return_value=FakeResponse(status, b'url?appid=private-test-key'))
    with pytest.raises(api.ProviderFailure) as error:
        api.fetch_weather(0, 0, 'private-test-key', get=get)
    assert error.value.reason == reason and 'private-test-key' not in str(error.value)
    assert get.call_args.args == (api.WEATHER_URL,)
    assert get.call_args.kwargs['allow_redirects'] is False
    assert get.call_args.kwargs['timeout'] == (2, 4)
    assert 'verify' not in get.call_args.kwargs


@pytest.mark.parametrize('exception,reason', [(requests.Timeout('secret'), 'provider_timeout'),
    (requests.ConnectionError('secret'), 'provider_unavailable'),
    (requests.exceptions.SSLError('secret'), 'provider_unavailable')])
def test_network_and_tls_failures_preserve_reason_without_secret(exception, reason):
    with pytest.raises(api.ProviderFailure) as error:
        api.fetch_weather(0, 0, 'private-test-key', get=Mock(side_effect=exception))
    assert error.value.reason == reason and 'secret' not in str(error.value)


def test_fixed_soil_request_nearest_center_wgs84_and_max_body():
    get = Mock(return_value=FakeResponse(200, tiff(295)))
    assert api.fetch_soil(50.2, 10.2, 'clay', get=get)['value'] == 29.5
    p = get.call_args.kwargs['params']
    assert get.call_args.args == (api.SOIL_URL,)
    assert p['COVERAGE'] == 'clay_0-5cm_mean' and p['CRS'] == 'EPSG:4326'
    assert p['WIDTH'] == p['HEIGHT'] == 3 and p['INTERPOLATION'] == 'NEAREST'
    assert p['BBOX'].split(',')[:2] == [str(10.2 - .002), str(50.2 - .002)]
    with pytest.raises(api.ProviderFailure):
        api.provider_bytes(api.SOIL_URL, {}, get=Mock(return_value=FakeResponse(200, b'x' * 65537)))


def test_cache_has_dates_is_detached_and_isolated_between_accounts():
    calls = Mock(side_effect=weather)
    clock = [100.0]
    client = service(weather=calls, clock=lambda: clock[0])
    first = client.lookup(OWNER, 0, 30)
    cached = client.lookup(OWNER, 0, 30)
    assert cached['weather']['cached'] is True
    assert cached['weather']['retrieved_at'] == first['weather']['retrieved_at']
    first['weather']['value']['temperature_c'] = 999
    assert client.lookup(OWNER, 0, 30)['weather']['value']['temperature_c'] == 12.5
    second_owner = client.lookup(OTHER, 0, 30)
    assert not second_owner['weather']['cached'] and calls.call_count == 2
    assert second_owner['soil']['reason'] == 'provider_rate_limited'  # process-wide, not owner-only
    clock[0] += 601
    assert not client.lookup(OWNER, 0, 30)['weather']['cached']
    assert client.lookup(OWNER, 0, 30)['soil']['cached']


def test_new_point_never_reuses_previous_weather_or_soil():
    calls = Mock(side_effect=lambda lat, lon, key: ({'temperature_c': lon}, AT))
    client = service(weather=calls)
    a = client.lookup(OWNER, 0, 10)
    b = client.lookup(OWNER, 0, 20)
    assert a['weather']['value']['temperature_c'] == 10
    assert b['weather']['value']['temperature_c'] == 20
    assert b['soil']['value'] is None and b['soil']['request_point']['lon'] == 20


def test_parallel_provider_failure_does_not_evict_same_point_success():
    client = service()
    good = client.lookup(OWNER, 0, 30)['soil']
    late_failure = api.envelope('soil', 0, 30)
    late_failure['reason'] = 'provider_rate_limited'
    client._store((OWNER, 'soil', 0, 30), late_failure)
    assert client.lookup(OWNER, 0, 30)['soil']['value'] == good['value']


def test_partial_failure_keeps_available_soil_and_weather_independent():
    def partial(lat, lon, name):
        if name == 'clay':
            raise api.ProviderFailure('provider_timeout')
        return soil(lat, lon, name)
    result = service(soil=partial).lookup(OWNER, 0, 30)
    assert result['weather']['status'] == 'ok'
    assert result['soil']['status'] == 'partial'
    assert len(result['soil']['value']['properties']) == 4
    validate_environment({'version': 1, 'gps': gps(), 'weather': result['weather'], 'soil': result['soil']})


def test_timed_out_future_cannot_later_mutate_snapshot_or_cached_response():
    class LateFuture:
        completed = False
        def result(self, timeout):
            if not self.completed:
                raise FutureTimeout()
            return {'temperature_c': 99}, AT
        def cancel(self):
            return False  # An already-running HTTP operation may not cancel.
    late = LateFuture()
    class LatePool(ImmediatePool):
        def submit(self, function, *args):
            return late if function is weather else super().submit(function, *args)
    client = api.EnvironmentService(weather=weather, soil=soil, key='test', pool=LatePool())
    response = client.lookup(OWNER, 0, 30)
    saved = history.validate_snapshot(json.dumps(snapshot({'version': 1, 'gps': gps(),
        'weather': response['weather'], 'soil': response['soil']})), IMAGE)
    late.completed = True
    assert saved['environment']['weather']['value'] is None
    assert saved['environment']['weather']['reason'] == 'provider_timeout'
    assert client.lookup(OWNER, 0, 30)['weather']['value'] is None


def test_no_configuration_and_no_land_coverage_are_honest_absences():
    result = service(key='', soil=lambda *_: None).lookup(OWNER, 0, 30)
    assert result['weather']['reason'] == 'provider_not_configured'
    assert result['soil']['reason'] == 'no_coverage'
    assert result['soil']['value'] is None and result['weather']['value'] is None
    validate_environment({'version': 1, 'gps': gps(), 'weather': result['weather'], 'soil': result['soil']})


def test_all_environment_routes_require_login_and_ignore_untrusted_provider_url():
    app = FastAPI()
    app.include_router(api.router)
    client = TestClient(app)
    with patch.object(api.SERVICE, 'lookup') as lookup:
        assert client.get('/v4/environment?lat=0&lon=0').status_code == 401
        lookup.assert_not_called()
        app.dependency_overrides[api.current_user] = lambda: OWNER
        lookup.return_value = {'environment_version': 1}
        response = client.get('/v4/environment?lat=0&lon=0&url=http://169.254.169.254/')
        assert response.status_code == 200
        lookup.assert_called_once_with(OWNER, 0., 0.)
        assert response.headers['cache-control'] == 'private, no-store'
        for query in ('lat=NaN&lon=0', 'lat=91&lon=0', 'lat=0&lon=181', 'lat=0'):
            assert client.get('/v4/environment?' + query).status_code == 422


def test_owner_request_limit_is_not_bypassed_by_changing_points():
    client = service(key='', soil=lambda *_: None)
    for longitude in range(10):
        client.lookup(OWNER, 0, longitude)
    with pytest.raises(HTTPException) as error:
        client.lookup(OWNER, 0, 30)
    assert error.value.status_code == 429
    assert error.value.headers['Retry-After'] == '60'
