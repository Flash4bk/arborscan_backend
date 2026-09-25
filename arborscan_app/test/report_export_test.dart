import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as im;
import 'package:shared_preferences/shared_preferences.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:arborscan_app/report_export_loader.dart';
import 'package:arborscan_app/report_export_data.dart';
import 'package:arborscan_app/report_pdf.dart';
import 'package:arborscan_app/report_export_button.dart';
import 'package:arborscan_app/corrections_service.dart';

Uint8List fixturePhoto() {
  final image = im.Image(width: 400, height: 600);
  im.fill(image, color: im.ColorRgb8(225, 235, 243));
  im.fillRect(image,
      x1: 0, y1: 550, x2: 399, y2: 599, color: im.ColorRgb8(140, 155, 100));
  im.fillRect(image,
      x1: 190, y1: 70, x2: 210, y2: 550, color: im.ColorRgb8(110, 70, 40));
  im.fillCircle(image,
      x: 200, y: 175, radius: 105, color: im.ColorRgb8(30, 100, 55));
  im.fillRect(image,
      x1: 70, y1: 450, x2: 78, y2: 550, color: im.ColorRgb8(30, 70, 180));
  im.drawString(image, 'DEMO AS-12',
      font: im.arial24, x: 15, y: 15, color: im.ColorRgb8(0, 0, 0));
  return Uint8List.fromList(im.encodePng(image));
}

ReportExportData fixture(String name,
        {num height = 4.8, bool old = false, bool long = false}) =>
    ReportExportData(local: true, photo: fixturePhoto(), record: {
      'analysis_id': 'DEMO-AS12',
      'version_id': name,
      'created_at': '2026-09-25T09:00:00Z'
    }, snapshot: {
      'version': 1,
      'kind': old ? 'legacy' : 'reference',
      'captured_at': old ? null : '2026-09-24T12:00:00Z',
      'report': {
        'species': long
            ? List.filled(
                    35, 'Сосна обыкновенная Pinus sylvestris subsp. lapponica')
                .join(' ')
            : 'ДЕМОНСТРАЦИОННЫЕ ДАННЫЕ',
        'height_m': old ? null : height,
        'crown_width_m': old ? null : 2.1,
        'method': 'known_object_segment_v2',
        'geometry': {
          'crown_height': {
            'value': 2.1,
            'unit': 'm',
            'source': 'reference',
            'definition': long
                ? List.filled(45,
                        'Проекционная величина, перспектива ограничивает метод.')
                    .join(' ')
                : 'Проекция живой кроны'
          }
        }
      },
      if (!old)
        'reference': {
          'version': 2,
          'method': 'known_object_segment_v2',
          'coordinates': 'normalized_oriented_image',
          'width': 400,
          'height': 600,
          'length_m': 1,
          'reference': [
            {'x': 0.185, 'y': 0.9167},
            {'x': 0.185, 'y': 0.75}
          ],
          'tree': [
            {'x': 0.5, 'y': 0.9167},
            {'x': 0.5, 'y': 0.1167}
          ],
          'crown': [
            {'x': 0.2375, 'y': 0.2917},
            {'x': 0.7625, 'y': 0.2917}
          ],
          'crown_height': [
            {'x': 0.5, 'y': 0.4667},
            {'x': 0.5, 'y': 0.1167}
          ],
          'trunk': [
            {'x': 0.475, 'y': 0.7},
            {'x': 0.525, 'y': 0.7}
          ],
          'trunk_axis': [
            {'x': 0.5, 'y': 0.6},
            {'x': 0.5, 'y': 0.8}
          ],
          'outline': [
            {'x': 0.17, 'y': 0.92},
            {'x': 0.2, 'y': 0.92},
            {'x': 0.2, 'y': 0.74},
            {'x': 0.17, 'y': 0.74}
          ]
        }
    });

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  test(
      'pinned contour GET, verified original, offline cache and owner isolation',
      () async {
    SharedPreferences.setMockInitialValues({
      'arborscan_auth_token': 'test-a',
      'arborscan_user_id': '00000000-0000-4000-8000-000000000001'
    });
    final root = await Directory.systemTemp.createTemp('as12-cache');
    var calls = 0;
    final photo = fixturePhoto();
    final auth = CorrectionsService(
        clientFactory: () => MockClient((r) async {
              calls++;
              expect(r.method, 'GET');
              expect(r.url.path, endsWith('/v4/corrections/pinned-revision'));
              expect(r.headers['Authorization'], 'Bearer test-a');
              return http.Response(
                  jsonEncode({
                    'correction_id': 'pinned-revision',
                    'original_image_base64': base64Encode(photo),
                    'mask_png_base64': base64Encode(photo)
                  }),
                  200);
            }));
    Future<ReportExportData> load(CorrectionsService service) =>
        loadReportExport(
            snapshot: {
              'correction_id': 'pinned-revision',
              'report': {'height_m': 12}
            },
            record: {
              'version_id': 'v1'
            },
            photo: photo,
            local: false,
            token: 'test-a',
            service: service,
            cacheDirectory: () async => root);
    expect((await load(auth)).mask, isNotNull);
    final offline = CorrectionsService(
        clientFactory: () =>
            MockClient((_) async => throw const SocketException('offline')));
    expect((await load(offline)).mask, isNotNull);
    expect(calls, 1);
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(
        'arborscan_user_id', '00000000-0000-4000-8000-000000000002');
    await expectLater(load(offline), throwsA(isA<CorrectionException>()));
    await root.delete(recursive: true);
  });
  test(
      'late response after account change and wrong correction original rejected',
      () async {
    SharedPreferences.setMockInitialValues({
      'arborscan_auth_token': 'test-a',
      'arborscan_user_id': '00000000-0000-4000-8000-000000000001'
    });
    final root = await Directory.systemTemp.createTemp('as12-isolation');
    final auth = CorrectionsService(
        clientFactory: () => MockClient((_) async {
              await (await SharedPreferences.getInstance())
                  .setString('arborscan_auth_token', 'test-b');
              return http.Response('{}', 200);
            }));
    await expectLater(
        loadReportExport(
            snapshot: {'correction_id': 'revision'},
            record: {},
            photo: fixturePhoto(),
            local: false,
            token: 'test-a',
            service: auth,
            cacheDirectory: () async => root),
        throwsA(isA<CorrectionException>()));
    expect(await root.list().length, 0);
    await root.delete(recursive: true);
  });
  test(
      'immutable snapshot, units, missing data, version separation and redaction',
      () {
    final raw = {'height_m': 12.345, 'species': 'A'};
    final d = ReportExportData(
        snapshot: {'report': raw},
        record: {'analysis_id': 'a', 'version_id': 'b'},
        local: true);
    raw['height_m'] = 999;
    expect(d.metrics.first.value, 12.345);
    expect(d.metrics.first.display, '12.35 м');
    expect(d.metrics.last.value, isNull);
    expect(fixture('v1').filename, isNot(fixture('v2', height: 5.4).filename));
    expect(exportText('https://example.com/?token=SECRET C:\\private\\x'),
        isNot(contains('SECRET')));
    expect(exportNumber(double.nan), 'нет данных');
    expect(exportNumber(0), '0');
  });
  test('EXIF orientation and reference coordinate bounds', () {
    final original = im.Image(width: 600, height: 400);
    original.exif.imageIfd.orientation = 6;
    final data = fixture('exif');
    final d = ReportExportData(
        snapshot: data.snapshot,
        record: data.record,
        photo: Uint8List.fromList(im.encodeJpg(original)),
        local: true);
    final images = prepareReportImages(d);
    final decoded = im.decodeImage(images.original!)!;
    expect([decoded.width, decoded.height], [400, 600]);
    final bad = data.snapshot;
    bad['reference']['width'] = 333;
    expect(
        () => prepareReportImages(ReportExportData(
            snapshot: bad, record: {}, photo: fixturePhoto(), local: true)),
        throwsFormatException);
  });
  test('generate real PDFs, frozen fixtures and selected versions', () async {
    final dir = Directory('../output/pdf/as12')..createSync(recursive: true);
    for (final data in [
      fixture('full-v1'),
      fixture('full-v2', height: 5.4),
      fixture('old', old: true),
      fixture('long', long: true)
    ]) {
      final bytes = await buildReportPdf(data,
          generatedAt: DateTime.utc(2026, 9, 25, 10));
      expect(ascii.decode(bytes.take(5).toList()), '%PDF-');
      await File('${dir.path}/${data.filename}').writeAsBytes(bytes);
      await File('${dir.path}/${data.version}.json').writeAsString(
          jsonEncode({'record': data.record, 'snapshot': data.snapshot}));
      await File('${dir.path}/demo.photo').writeAsBytes(data.photo!);
    }
    final missing = ReportExportData(snapshot: {
      'report': {'species': 'ДЕМОНСТРАЦИОННЫЙ неполный отчёт'}
    }, record: {
      'analysis_id': 'DEMO-AS12',
      'version_id': 'missing'
    }, local: true);
    expect(() => prepareReportImages(missing), throwsFormatException);
    await File('${dir.path}/${missing.filename}')
        .writeAsBytes(await buildReportPdf(missing, partial: true));
  });
  testWidgets('session change invalidates export before loading another owner',
      (tester) async {
    SharedPreferences.setMockInitialValues({'arborscan_auth_token': 'test-a'});
    var loads = 0;
    await tester.pumpWidget(
        MaterialApp(home: Scaffold(body: ReportExportButton(load: () async {
      loads++;
      return fixture('test');
    }))));
    CorrectionsService.authChanges.value++;
    await tester.pump();
    await tester.tap(find.byKey(const ValueKey('export-pdf')));
    expect(loads, 0);
    expect(find.textContaining('Аккаунт изменился'), findsOneWidget);
  });
  testWidgets(
      'double tap single operation; cancel partial export produces no save',
      (tester) async {
    SharedPreferences.setMockInitialValues({'arborscan_auth_token': 'test-a'});
    var loads = 0, nativeCalls = 0;
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(
            const MethodChannel('arborscan/report_export'), (call) async {
      nativeCalls++;
      return 'saved';
    });
    await tester.pumpWidget(
        MaterialApp(home: Scaffold(body: ReportExportButton(load: () async {
      loads++;
      return ReportExportData(snapshot: {}, record: {}, local: true);
    }))));
    await tester.tap(find.byKey(const ValueKey('export-pdf')));
    await tester.tap(find.byKey(const ValueKey('export-pdf')));
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 400));
    expect(loads, 1);
    await tester.tap(find.text('Отмена'));
    await tester.pumpAndSettle();
    expect(find.text('Экспорт отменён'), findsOneWidget);
    expect(nativeCalls, 0);
  });
}
