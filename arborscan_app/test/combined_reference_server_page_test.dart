import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:crypto/crypto.dart';
import 'package:image/image.dart' as im;
import 'package:arborscan_app/app_theme.dart';
import 'package:arborscan_app/geometry_report.dart';
import 'package:arborscan_app/reference_measurement_page.dart';
import 'package:arborscan_app/report_history_service.dart';
import 'package:arborscan_app/server_report_page.dart';
import 'package:arborscan_app/unified_analysis_report_page.dart';
import 'package:arborscan_app/contour_workspace_page.dart';
import 'package:arborscan_app/corrections_service.dart';
import 'combined_reference_service_test.dart' show InspectClient, response;
import 'combined_reference_workflow_test.dart' as fixture show measurement;

void main() {
  const owner = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa';
  const aid = 'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb';
  const cid = 'eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee';
  const executionId = 'dddddddd-dddd-4ddd-8ddd-dddddddddddd';
  late Directory folder;
  late Uint8List original;
  late Map<String, dynamic> snapshot;
  setUp(() async {
    SharedPreferences.setMockInitialValues(
        {'arborscan_auth_token': 'test', 'arborscan_user_id': owner});
    folder = await Directory.systemTemp.createTemp('reference-server-ui-');
    original = await File('test/fixtures/reference_exif6.jpg').readAsBytes();
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(
            const MethodChannel('plugins.flutter.io/path_provider'),
            (_) async => folder.path);
    final mask = im.Image(width: 100, height: 200, numChannels: 4);
    for (final p in mask) {
      final inside = p.x > 40 && p.x < 60;
      p.setRgba(inside ? 255 : 0, inside ? 255 : 0, inside ? 255 : 0, 255);
    }
    snapshot = {
      'version': 1,
      'kind': 'reference',
      'captured_at': '2026-10-13T00:00:00Z',
      'reference': fixture.measurement().toJson(),
      'correction_id': cid,
      'report': {
        'analysis_id': executionId,
        ...fixture.measurement().report,
        'tree': {'detected': true},
        'contour_preview': {
          'correction_id': cid,
          'photo_sha256': sha256.convert(original).toString(),
          'width': 100,
          'height': 200,
          'mask_png_base64': base64Encode(im.encodePng(mask))
        }
      }
    };
  });
  tearDown(() async {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(
            const MethodChannel('plugins.flutter.io/path_provider'), null);
    await folder.delete(recursive: true);
  });
  Future<void> idle(WidgetTester t) async {
    for (var i = 0; i < 250; i++) {
      await t.runAsync(() async {
        await t.pump(const Duration(milliseconds: 100));
        await Future<void>.delayed(const Duration(milliseconds: 20));
      });
      if (find.byType(LinearProgressIndicator).evaluate().isEmpty && i > 5) {
        return;
      }
    }
    fail('History screen did not finish its local operation.');
  }

  Future<Map<String, dynamic>> open(WidgetTester t,
      {bool saved = false, ReportHistoryService? service}) async {
    final history = service ?? ReportHistoryService();
    final row = (await t.runAsync(() => history.stage(
        token: 'test',
        localId: 'report',
        analysisId: aid,
        image: original,
        snapshot: snapshot)))!;
    if (saved) {
      await t.runAsync(() => history.journal
          .save(owner, 'report', {...row, 'saved': true}, original));
    }
    await t.runAsync(() => t.pumpWidget(MaterialApp(
        theme: AppTheme.light(),
        home: ServerReportPage(localId: 'report', service: service))));
    await idle(t);
    return row;
  }

  for (final valid in [false, true]) {
    testWidgets('cold history opens exact pinned original only, valid=$valid',
        (t) async {
      final auth = CorrectionsService(
          clientFactory: () => InspectClient((req) async {
                if (req.url.path.endsWith('/$cid')) {
                  return response({
                    'correction_id': cid,
                    'analysis_id': valid ? aid : executionId,
                    'original_image_base64': base64Encode(original),
                    'mask_png_base64': snapshot['report']['contour_preview']
                        ['mask_png_base64'],
                    'review_status': 'draft'
                  });
                }
                return response({'workflow_version': 1});
              }));
      await open(t, service: ReportHistoryService(auth: auth));
      expect(find.text('Создать / исправить контур дерева'), findsNothing);
      await t.scrollUntilVisible(find.text('Связанный контур и модерация'), 150,
          scrollable: find.byType(Scrollable).first);
      await t.runAsync(() => t.tap(find.text('Связанный контур и модерация')));
      await idle(t);
      if (valid) {
        final workspace =
            t.widget<ContourWorkspacePage>(find.byType(ContourWorkspacePage));
        expect(workspace.analysisId, aid);
        expect(workspace.record!['correction_id'], cid);
        expect(sha256.convert(workspace.image!), sha256.convert(original));
        expect(find.text('Оригинальное фото недоступно.'), findsNothing);
        expect(find.textContaining('Старая запись содержит только PNG'),
            findsOneWidget);
      } else {
        expect(find.byType(ContourWorkspacePage), findsNothing);
        await t.scrollUntilVisible(
            find.textContaining('Контур относится к другому фото или отчёту.'),
            -150,
            scrollable: find.byType(Scrollable).first);
        expect(
            find.textContaining('Контур относится к другому фото или отчёту.'),
            findsOneWidget);
      }
      await t.pumpWidget(const SizedBox());
    });
  }

  for (final detected in [false, true]) {
    testWidgets(
        'history labels pinned correction as user revision, AI=$detected',
        (t) async {
      t.view.physicalSize = const Size(800, 1600);
      t.view.devicePixelRatio = 1;
      addTearDown(t.view.resetPhysicalSize);
      addTearDown(t.view.resetDevicePixelRatio);
      snapshot['report']['tree'] = {'detected': detected};
      await open(t);
      await t.scrollUntilVisible(
          find.textContaining('Зелёная маска - закреплённая исправленная'), 150,
          scrollable: find.byType(Scrollable).first);
      expect(find.textContaining('Зелёная маска - закреплённая исправленная'),
          findsOneWidget);
      expect(find.textContaining('сохранённое выделение дерева моделью'),
          findsNothing);
      if (!detected) {
        expect(
            find.text(
                'Модель не нашла дерево; исправленный контур создан вручную.'),
            findsOneWidget);
      }
      await t.pumpWidget(const SizedBox());
    });
  }

  for (final saved in [false, true]) {
    testWidgets(
        'reference revision parent uses only confirmed server row $saved',
        (t) async {
      final row = await open(t, saved: saved);
      await t.scrollUntilVisible(
          find.text('Изменить эталон / новая версия'), 150,
          scrollable: find.byType(Scrollable).first);
      await t
          .runAsync(() => t.tap(find.text('Изменить эталон / новая версия')));
      await idle(t);
      final page = t.widget<ReferenceMeasurementPage>(
          find.byType(ReferenceMeasurementPage));
      final draft = (await t
          .runAsync(() => referenceStore().load(owner, page.draftId!)))!;
      expect(draft['server_parent_id'], saved ? row['version_id'] : null);
      expect(draft['server_snapshot']['correction_id'], cid);
      expect(draft['server_snapshot']['reference'], snapshot['reference']);
      await t.pumpWidget(const SizedBox());
    });
  }

  for (final hasOriginal in [false, true]) {
    testWidgets('model source UI shows only original geometry $hasOriginal',
        (t) async {
      final geometry = {
        'crown_height': {
          'value': null,
          'unit': 'm',
          'reason': 'whole_tree_mask_has_no_live_crown_base'
        }
      };
      if (hasOriginal) snapshot['report']['model_geometry'] = geometry;
      await open(t);
      await t.scrollUntilVisible(
          find.text('Исходный результат модели / источники'), 150,
          scrollable: find.byType(Scrollable).first);
      await t.ensureVisible(find.text('Исходный результат модели / источники'));
      await t.pumpAndSettle();
      await t.runAsync(
          () => t.tap(find.text('Исходный результат модели / источники')));
      await idle(t);
      final page = t.widget<UnifiedAnalysisReportPage>(
          find.byType(UnifiedAnalysisReportPage));
      expect(page.historyAnalysisId, aid);
      expect(page.result.analysisId, executionId);
      expect(page.result.raw['geometry'], hasOriginal ? geometry : null);
      expect(page.result.raw.containsKey('height_m'), false);
      expect(page.result.raw.containsKey('method'), false);
      expect(page.result.raw.containsKey('contour_preview'), false);
      final advanced = find.byKey(const PageStorageKey('analysis-advanced'));
      final title = find.descendant(
          of: advanced, matching: find.text('Расширенный анализ'));
      await t.scrollUntilVisible(title, 150,
          scrollable: find.byType(Scrollable).first);
      await t.ensureVisible(title);
      await t.pumpAndSettle();
      await t.tap(title);
      await t.pumpAndSettle();
      final shownGeometry = t.widget<GeometryReport>(
          find.descendant(of: advanced, matching: find.byType(GeometryReport)));
      expect(shownGeometry.data, hasOriginal ? geometry : null);
      expect(shownGeometry.compact, false);
      expect(
          find.descendant(
              of: advanced,
              matching: find.textContaining(
                  'Высота отмеченной живой кроны (проекция): нет данных')),
          findsOneWidget);
      expect(find.text('Высота отмеченной живой кроны (проекция): 2.00 м'),
          findsNothing);
      await t.pumpWidget(const SizedBox());
    });
  }
}
