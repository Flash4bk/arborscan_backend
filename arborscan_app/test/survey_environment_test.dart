import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:image/image.dart' as im;
import 'package:shared_preferences/shared_preferences.dart';
import 'package:arborscan_app/survey_environment.dart';
import 'package:arborscan_app/survey_environment_ui.dart';
import 'package:arborscan_app/contour_drafts.dart';
import 'package:arborscan_app/report_history_service.dart';
import 'package:arborscan_app/corrections_service.dart';
import 'package:arborscan_app/app_theme.dart';

const owner = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa';
const other = 'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb';
const point = SurveyPoint(
    lat: 0,
    lon: 0,
    source: 'manual',
    positionKind: 'tree',
    retrievedAt: '2026-10-01T00:00:00Z');
Map<String, dynamic> response(SurveyPoint p, {double temperature = 12}) => {
      'environment_version': 1,
      'weather': {
        'source': 'OpenWeather',
        'value': {'temperature_c': temperature},
        'request_point': {'lat': p.lat, 'lon': p.lon},
        'status': 'ok'
      },
      'soil': {
        'source': 'SoilGrids',
        'value': null,
        'request_point': {'lat': p.lat, 'lon': p.lon},
        'status': 'unavailable',
        'reason': 'no_coverage'
      }
    };
void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  setUp(() => SharedPreferences.setMockInitialValues(
      {'arborscan_auth_token': 'test', 'arborscan_user_id': owner}));
  test(
      'zero coordinates valid; non-finite, wrong CRS and missing values absent',
      () {
    expect(SurveyPoint.fromJson(point.toJson())?.lat, 0);
    for (final v in [
      null,
      {},
      {
        'value': {'lat': 0}
      },
      {'lat': double.nan, 'lon': 0},
      {'lat': 91, 'lon': 0},
      {'lat': 0, 'lon': 181},
      {
        'value': {'lat': 0, 'lon': 0, 'crs': 'EPSG:3857'}
      }
    ]) {
      expect(SurveyPoint.fromJson(v), isNull);
    }
  });
  test(
      'original JPEG EXIF coordinates precede orientation and never invent timezone or accuracy',
      () {
    final image = im.Image(width: 12, height: 8);
    image.exif.imageIfd.orientation = 6;
    image.exif.gpsIfd[1] = im.IfdValueAscii('S');
    image.exif.gpsIfd[2] = im.IfdValueRational.list([50, 12, 0]
        .map((v) => im.IfdValueRational(v, 1).toRational())
        .toList());
    image.exif.gpsIfd[3] = im.IfdValueAscii('W');
    image.exif.gpsIfd[4] = im.IfdValueRational.list([10, 12, 0]
        .map((v) => im.IfdValueRational(v, 1).toRational())
        .toList());
    image.exif.exifIfd[0x9003] = im.IfdValueAscii('2026:09:30 12:00:00');
    final bytes = Uint8List.fromList(im.encodeJpg(image)),
        original = Uint8List.fromList(im.encodeJpg(image));
    final parsed = surveyPointFromExif(bytes)!;
    expect(parsed.lat, closeTo(-50.2, 1e-9));
    expect(parsed.lon, closeTo(-10.2, 1e-9));
    expect(parsed.observedAt, isNull);
    expect(parsed.accuracyM, isNull);
    expect(parsed.capturedLocal, '2026-09-30T12:00:00');
    expect(bytes, original);
    image.exif.exifIfd[0x9011] = im.IfdValueAscii('+03:00');
    expect(
        surveyPointFromExif(Uint8List.fromList(im.encodeJpg(image)))!
            .observedAt,
        '2026-09-30T09:00:00.000Z');
    expect(
        surveyPointFromExif(Uint8List.fromList(im.encodePng(image))), isNull);
  });
  test(
      'changing point and account reject late conditions, saved copy immutable',
      () async {
    final pending = Completer<http.Response>(), sent = Completer<void>();
    final c =
        SurveyEnvironmentController({'version': 1, 'gps': point.toJson()});
    final service = EnvironmentService(
        clientFactory: () => MockClient((r) {
              sent.complete();
              return pending.future;
            }));
    final refresh = c.refresh('test', service: service);
    await sent.future;
    const second = SurveyPoint(
        lat: 1, lon: 2, source: 'manual', retrievedAt: '2026-10-01T00:00:00Z');
    c.setPoint(second);
    pending.complete(http.Response(jsonEncode(response(point)), 200));
    await refresh;
    expect(c.point!.lat, 1);
    expect(c.snapshot['weather'], isNull);
    await c.refresh('test',
        service: EnvironmentService(
            clientFactory: () => MockClient((_) async =>
                http.Response(jsonEncode(response(second)), 200))));
    final saved = c.snapshot;
    c.setPoint(point);
    expect(saved['gps']['value']['lat'], 1);
    expect(saved['weather']['value']['temperature_c'], 12);
    final pending2 = Completer<http.Response>(), sent2 = Completer<void>();
    final old = c.refresh('test',
        service: EnvironmentService(
            clientFactory: () => MockClient((_) {
                  sent2.complete();
                  return pending2.future;
                })));
    await sent2.future;
    CorrectionsService.authChanges.value++;
    pending2.complete(http.Response(jsonEncode(response(point)), 200));
    await old;
    expect(c.snapshot, isEmpty);
    c.dispose();
  });
  test(
      'auth, incompatible API, wrong response point and network failures retain chosen location',
      () async {
    for (final code in [401, 403, 404, 405, 429, 503]) {
      final c =
          SurveyEnvironmentController({'version': 1, 'gps': point.toJson()});
      await c.refresh('test',
          service: EnvironmentService(
              clientFactory: () => MockClient((r) async {
                    expect(r.headers['Authorization'], 'Bearer test');
                    expect(r.url.path, '/api/v4/v4/environment');
                    return http.Response('{}', code);
                  })));
      expect(c.point!.lat, 0);
      expect(c.error, isNotEmpty);
      expect(c.busy, isFalse);
      c.dispose();
    }
    final c =
        SurveyEnvironmentController({'version': 1, 'gps': point.toJson()});
    await c.refresh('test',
        service: EnvironmentService(
            clientFactory: () => MockClient(
                (_) async => throw const SocketException('offline'))));
    expect(c.error, contains('Нет связи'));
    await c.refresh('test',
        service: EnvironmentService(
            clientFactory: () => MockClient((_) async => http.Response(
                jsonEncode(response(const SurveyPoint(
                    lat: 1, lon: 2, source: 'manual', retrievedAt: ''))),
                200))));
    expect(c.error, contains('другой точке'));
    c.dispose();
  });
  test(
      'server report cache restores exact saved version after service recreation and isolates owners',
      () async {
    final dir = await Directory.systemTemp.createTemp('as09-cache-');
    try {
      final store = ContourDrafts(directory: () async => dir);
      final data = {
        'record': {'version_id': 'version-one', 'analysis_id': owner},
        'snapshot': {
          'version': 1,
          'environment': {'version': 1, 'gps': point.toJson()},
          'report': {'height_m': 4.8},
          'image': {
            'original_base64': base64Encode([1, 2, 3])
          }
        }
      };
      final online = ReportHistoryService(
          cache: store,
          clientFactory: () =>
              MockClient((_) async => http.Response(jsonEncode(data), 200)));
      await online.record('test', 'version-one');
      final offline = ReportHistoryService(
          cache: ContourDrafts(directory: () async => dir),
          clientFactory: () =>
              MockClient((_) async => throw const SocketException('offline')));
      final restored = await offline.record('test', 'version-one');
      expect(restored['cached'], true);
      expect(restored['snapshot'], data['snapshot']);
      final forbidden = ReportHistoryService(
          cache: store,
          clientFactory: () =>
              MockClient((_) async => http.Response('{}', 403)));
      await expectLater(forbidden.record('test', 'version-one'),
          throwsA(isA<CorrectionException>()));
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString('arborscan_user_id', other);
      await expectLater(offline.record('test', 'version-one'),
          throwsA(isA<CorrectionException>()));
    } finally {
      await dir.delete(recursive: true);
    }
  });
  test('old API cannot silently discard new environment snapshot on upload',
      () async {
    final dir = await Directory.systemTemp.createTemp('as09-old-');
    try {
      var posts = 0;
      final store = ContourDrafts(directory: () async => dir);
      final service = ReportHistoryService(
          journal: store,
          clientFactory: () => MockClient((r) async {
                if (r.method == 'POST') posts++;
                return http.Response('{"history_version":1}', 200);
              }));
      await service.stage(
          token: 'test',
          localId: 'draft',
          analysisId: owner,
          image: Uint8List.fromList([1]),
          snapshot: {
            'version': 1,
            'environment': {'version': 1, 'gps': point.toJson()}
          });
      await expectLater(
          service.upload('test', 'draft'),
          throwsA(isA<CorrectionException>()
              .having((e) => e.message, 'message', contains('снимок места'))));
      expect(posts, 0);
      expect(
          (await store.load(owner, 'draft'))!['snapshot']['environment']['gps']
              ['value']['lat'],
          0);
    } finally {
      await dir.delete(recursive: true);
    }
  });
  testWidgets(
      'conditions summary renders actual units and missing coverage with large font',
      (t) async {
    await t.pumpWidget(MaterialApp(
        theme: AppTheme.light(),
        home: MediaQuery(
            data: const MediaQueryData(textScaler: TextScaler.linear(2)),
            child: Scaffold(
                body: SingleChildScrollView(
                    child: EnvironmentSummary(snapshot: {
              'version': 1,
              'gps': point.toJson(),
              ...response(point)..remove('environment_version')
            }))))));
    await t.tap(find.text('Погода'));
    await t.pumpAndSettle();
    expect(find.text('Температура: 12.0 °C'), findsOneWidget);
    expect(find.textContaining('не архив погоды'), findsOneWidget);
    expect(find.textContaining('Для этой точки данных нет'), findsOneWidget);
    expect(t.takeException(), isNull);
  });
}
