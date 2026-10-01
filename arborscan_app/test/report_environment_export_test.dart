import 'dart:convert';
import 'dart:io';

import 'package:arborscan_app/report_export_data.dart';
import 'package:arborscan_app/report_pdf.dart';
import 'package:flutter_test/flutter_test.dart';
import 'report_export_test.dart' as fixtures;

Map<String, dynamic> environmentFixture(
        {num temperature = 16, double latitude = 0}) =>
    {
      'version': 1,
      'gps': {
        'value': {'lat': latitude, 'lon': 31.7, 'crs': 'EPSG:4326'},
        'source': 'manual',
        'retrieved_at': '2026-10-01T10:05:00Z',
        'observed_at': null,
        'accuracy_m': null,
        'position_kind': 'tree',
        'is_last_known': false,
      },
      'weather': {
        'source': 'OpenWeather',
        'kind': 'current',
        'status': 'ok',
        'retrieved_at': '2026-10-01T10:06:00Z',
        'data_at': '2026-10-01T10:00:00Z',
        'cached': true,
        'request_point': {'lat': latitude, 'lon': 31.7},
        'units': 'C_m/s_degrees_hPa_percent',
        'attribution': 'OpenWeather - DEMO, synthetic fixture',
        'value': {
          'temperature_c': temperature,
          'wind_speed_m_s': 2.3,
          'wind_gust_m_s': 4.5,
          'wind_direction_deg': 270,
          'pressure_hpa': 1008,
          'relative_humidity_pct': 68
        },
        'limitations': ['Контрольные данные теста, не полевой замер.'],
      },
      'soil': {
        'source': 'SoilGrids',
        'kind': 'modelled_grid',
        'status': 'partial',
        'reason': 'partial_data',
        'retrieved_at': '2026-10-01T10:06:00Z',
        'data_at': null,
        'cached': false,
        'request_point': {'lat': latitude, 'lon': 31.7},
        'dataset_version': '2.0',
        'resolution_m': 250,
        'access_method': 'WCS_nearest_cell',
        'attribution': 'ISRIC SoilGrids 2.0, CC BY 4.0 - DEMO',
        'value': {
          'properties': [
            {
              'name': 'clay',
              'value': 22,
              'unit': '%',
              'depth_cm': [0, 5],
              'q05': 12,
              'q95': 35
            },
            {
              'name': 'soc',
              'value': 10.5,
              'unit': 'g/kg',
              'depth_cm': [0, 5],
              'q05': null,
              'q95': null
            },
            {
              'name': 'phh2o',
              'value': 6.7,
              'unit': 'pH',
              'depth_cm': [0, 5],
              'q05': 5.9,
              'q95': 7.1
            },
          ]
        },
      },
    };

ReportExportData _newReport(String version,
        {Map<String, dynamic>? environment}) =>
    ReportExportData(local: false, photo: fixtures.fixturePhoto(), record: {
      'analysis_id': 'DEMO-AS15-AS09',
      'version_id': version,
      'created_at': '2026-10-01T10:07:00Z',
    }, snapshot: {
      'version': 1,
      'kind': 'v4',
      'captured_at': '2026-09-24T12:00:00Z',
      'report': {
        'species': {
          'display_name': 'ДЕМОНСТРАЦИОННЫЕ ДАННЫЕ',
          'scientific_name': 'Pinus sylvestris',
          'source': 'synthetic fixture'
        },
        'measurement_method_version': 2,
        'measurements': {
          'height': {'value_m': 13.2, 'source': 'ar'},
          'crown_width': {'value_m': 5.1, 'source': 'reference+vision'},
          'trunk_diameter': {
            'value_m': 0.31,
            'source': 'ar',
            'standard': 'dbh_1_3m'
          },
        },
      },
      if (environment != null) 'environment': environment,
    });

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  test(
      'legacy measurement fields, sources and original survive export without recalculation',
      () {
    final photo = fixtures.fixturePhoto();
    final raw = {
      'height_m': 13.25,
      'crown_width_m': '5.10',
      'trunk_diameter_m': 0.42,
      'measurement_sources': {'height_m': 'ar', 'crown_width_m': 'image'},
      'original_image_base64': base64Encode(photo),
      'risk': {'index': 0.75, 'category': 'high'},
      'beta': {'beta_kg_s': 38.6, 'method': 'species_default'},
    };
    final data = ReportExportData(
        snapshot: {'kind': 'legacy', 'report': raw}, record: {}, local: false);
    expect(data.metrics.take(3).map((m) => m.value), [13.25, 5.1, .42]);
    expect(data.metrics.first.source, 'AR-измерение');
    expect(data.photo, orderedEquals(photo));
    expect(data.historicalValues.join('\n'),
        contains('Прежнее значение β: 38.6 кг/с'));
    expect(data.provenance.join('\n'), contains('Код источника height_m: ar'));
    final unsupported = ReportExportData(snapshot: {
      'report': {'height_m': 'unknown unit 25ft'}
    }, record: {}, local: true);
    expect(unsupported.metrics.first.value, isNull);
    expect(unsupported.metrics.first.limitation,
        contains('присутствует, но его формат не поддерживается'));
  });

  test(
      'conditions retain units, soil depth, zero coordinate, dates and frozen version',
      () {
    final environment = environmentFixture();
    final first = _newReport('environment-v1', environment: environment);
    environment['weather']['value']['temperature_c'] = 24;
    environment['gps']['value']['lat'] = 1;
    final second = _newReport('environment-v2', environment: environment);
    final text = first.environment.join('\n');
    expect(text, contains('Координаты: 0.0, 31.7'));
    expect(text, contains('Температура: 16 °C'));
    expect(text, contains('Порывы: 4.5 м/с'));
    expect(text, contains('Направление ветра: 270 °'));
    expect(text, contains('слой 0-5 см'));
    expect(text, contains('Органический углерод: 10.5 г/кг'));
    expect(text, contains('Квантили'.toLowerCase()));
    expect(text, contains('2026-10-01T10:00:00Z'));
    expect(text, contains('не погода на дату старого снимка'));
    expect(second.environment.join('\n'), contains('Температура: 24 °C'));
    expect(first.filename, isNot(second.filename));
  });

  test(
      'missing, stale, legacy and unavailable conditions are not zero or current weather',
      () {
    final old = ReportExportData(
        record: {},
        local: true,
        snapshot: {
          'report': {},
          'environment': {
            'gps': {
              'value': {'lat': 0, 'lon': 0},
              'source': 'device',
              'is_last_known': true,
              'retrieved_at': 'old',
              'observed_at': 'older'
            },
            'weather': {
              'value': {'temperature_c': 0, 'wind_m_s': 0},
              'source': 'legacy',
              'retrieved_at': 'old'
            },
            'soil': {
              'value': null,
              'source': 'SoilGrids',
              'status': 'unavailable',
              'reason': 'provider_timeout'
            },
          }
        });
    final text = old.environment.join('\n');
    expect(text, contains('Координаты: 0, 0'));
    expect(text, contains('Это не новый GPS-замер'));
    expect(text, contains('Температура: 0 °C'));
    expect(text, contains('не сохранён'));
    expect(text, contains('источник не ответил вовремя'));
    final empty = _newReport('missing');
    expect(empty.environment.join('\n'), contains('Место не сохранено'));
    expect(empty.environment.join('\n'), isNot(contains('Координаты: 0')));
  });

  test(
      'generate five controlled PDFs including two versions of one observation',
      () async {
    final dir = Directory(Platform.environment['ARBORSCAN_AS09_PDF_OUTPUT'] ??
        '../output/pdf/as15-as09')
      ..createSync(recursive: true);
    final data = [
      ReportExportData(local: false, photo: fixtures.fixturePhoto(), record: {
        'analysis_id': 'DEMO-AS15-AS09',
        'version_id': 'legacy'
      }, snapshot: {
        'kind': 'legacy',
        'captured_at': '2026-09-20T09:00:00Z',
        'report': {
          'species': 'DEMO старый отчёт',
          'height_m': 13.25,
          'crown_width_m': 5.1,
          'trunk_diameter_m': .42,
          'measurement_sources': {
            'height_m': 'ar',
            'crown_width_m': 'image',
            'trunk_diameter_m': 'image'
          },
          'scale_px_to_m': .01,
          'gps': {'lat': 0, 'lon': 31.7},
          'risk': {'index': .75, 'category': 'high'},
          'beta': {'beta_kg_s': 38.6, 'method': 'species_default'},
          'analytic_wind_model': {
            'outputs': {'total_force_n': 965, 'base_moment_nm': 4825}
          },
        }
      }),
      _newReport('new'),
      fixtures.fixture('reference-local'),
      _newReport('environment-v1', environment: environmentFixture()),
      _newReport('environment-v2',
          environment: environmentFixture(temperature: 24, latitude: 1)),
    ];
    for (final item in data) {
      final bytes = await buildReportPdf(item,
          generatedAt: DateTime.utc(2026, 10, 1, 11));
      expect(ascii.decode(bytes.take(5).toList()), '%PDF-');
      await File('${dir.path}/${item.filename}').writeAsBytes(bytes);
      await File('${dir.path}/${item.version}.json').writeAsString(jsonEncode({
        'record': item.record,
        'snapshot': item.snapshot,
        'metrics': [
          for (final m in item.metrics)
            {
              'label': m.label,
              'value': m.value,
              'unit': m.unit,
              'source': m.source
            }
        ]
      }));
    }
  });
}
