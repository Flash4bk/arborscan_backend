import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:crypto/crypto.dart';
import 'package:http/http.dart' as http;
import 'package:image/image.dart' as im;
import 'package:arborscan_app/app_theme.dart';
import 'package:arborscan_app/contour_drafts.dart';
import 'package:arborscan_app/corrections_service.dart';
import 'package:arborscan_app/reference_measurement.dart';
import 'package:arborscan_app/reference_measurement_page.dart';
import 'package:arborscan_app/reference_preview.dart';
import 'package:arborscan_app/photo_analysis_service.dart';
import 'package:arborscan_app/report_history_service.dart';
import 'package:arborscan_app/contour_editor_state.dart';
import 'package:arborscan_app/contour_workspace_page.dart';
import 'package:arborscan_app/combined_reference.dart';
import 'combined_reference_service_test.dart' show InspectClient, response;

const owner = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa';
const analysisId = 'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb';
const parentId = 'cccccccc-cccc-4ccc-8ccc-cccccccccccc';

ReferenceMeasurement measurement() => ReferenceMeasurement(
        version: 2,
        width: 100,
        height: 200,
        lengthM: 1,
        samePlane: true,
        reference: const [
          Offset(.2, .8),
          Offset(.2, .6)
        ],
        tree: const [
          Offset(.5, .9),
          Offset(.5, .1)
        ],
        crown: const [
          Offset(.2, .3),
          Offset(.8, .3)
        ],
        outline: const [
          Offset(.1, .8),
          Offset(.3, .8),
          Offset(.3, .6),
          Offset(.1, .6)
        ]);

void main() {
  late Directory folder;
  late Uint8List original;
  late ContourDrafts store;
  late Map<String, dynamic> seed;
  setUp(() async {
    SharedPreferences.setMockInitialValues(
        {'arborscan_auth_token': 'test', 'arborscan_user_id': owner});
    folder = await Directory.systemTemp.createTemp('combined-reference-');
    original = await File('test/fixtures/reference_exif6.jpg').readAsBytes();
    store = ContourDrafts(directory: () async => folder);
    final mask = im.Image(width: 100, height: 200, numChannels: 4);
    for (final p in mask) {
      p.setRgba(0, 255, 0, p.x > 40 && p.x < 60 ? 110 : 0);
    }
    seed = {
      'version': 1,
      'kind': 'v4',
      'captured_at': '2026-10-13T00:00:00Z',
      'image': {'sha256': sha256.convert(original).toString()},
      'reference': measurement().toJson(),
      'ar': {
        'photo_sha256': sha256.convert(original).toString(),
        'association': 'user_confirmed_same_tree'
      },
      'report': {
        'analysis_id': analysisId,
        'analysis_status': 'absolute_scale_required',
        'tree': {'detected': true},
        'species': {'display_name': 'Тестовая сосна'},
        'measurements': {
          'height': {'value_px': 160, 'value_m': null, 'source': 'vision'}
        },
        'images': {
          'width': 100,
          'height': 200,
          'mask_image_base64': base64Encode(im.encodePng(mask))
        }
      }
    };
  });
  tearDown(() async {
    await folder.delete(recursive: true);
  });
  Future<void> settle(WidgetTester t) async {
    for (var i = 0; i < 10; i++) {
      await t.runAsync(() async {
        await t.pump(const Duration(milliseconds: 100));
        await Future<void>.delayed(const Duration(milliseconds: 20));
      });
    }
  }

  Future<void> waitUntil(WidgetTester t, bool Function() ready) async {
    for (var i = 0; i < 250; i++) {
      if (ready()) return;
      await t.runAsync(() async {
        await t.pump(const Duration(milliseconds: 50));
        await Future<void>.delayed(const Duration(milliseconds: 20));
      });
    }
    fail('Async screen operation did not finish before test teardown.');
  }

  bool actionFinished(String label) {
    final elements = find.text(label).evaluate().toList();
    if (elements.isEmpty) return true; // The action navigated to another page.
    ButtonStyleButton? button;
    elements.single.visitAncestorElements((element) {
      if (element.widget is ButtonStyleButton) {
        button = element.widget as ButtonStyleButton;
        return false;
      }
      return true;
    });
    return button == null || button!.onPressed != null;
  }

  Future<void> mount(WidgetTester t, Widget page) async {
    await t.runAsync(
        () => t.pumpWidget(MaterialApp(theme: AppTheme.light(), home: page)));
    await settle(t);
    await waitUntil(
        t, () => find.byType(LinearProgressIndicator).evaluate().isEmpty);
  }

  Future<void> tapText(WidgetTester t, String label) async {
    await t.scrollUntilVisible(find.text(label), 150,
        scrollable: find.byType(Scrollable).first);
    await Scrollable.ensureVisible(t.element(find.text(label)), alignment: .5);
    await t.pumpAndSettle();
    await t.runAsync(() => t.tap(find.text(label)));
    await settle(t);
    await waitUntil(t, () => actionFinished(label));
  }

  testWidgets(
      'same original + model + reference survive cold reopening and 100cm recalculation',
      (t) async {
    var pickerCalls = 0, requests = 0;
    final model = PhotoAnalysisService(
        clientFactory: () => InspectClient((_) async {
              requests++;
              return response({
                'analysis_id': analysisId,
                'tree': {'detected': false},
                'measurements': {
                  'height': {
                    'value_px': null,
                    'value_m': null,
                    'source': 'unavailable'
                  }
                }
              });
            }));
    await mount(
        t,
        ReferenceMeasurementPage(
            initialImage: original,
            initialSnapshot: seed,
            analysisId: analysisId,
            parentId: parentId,
            drafts: store,
            pickPhoto: () async {
              pickerCalls++;
              return null;
            },
            analysisService: model));
    final preview = t.widget<ReferencePreview>(find.byType(ReferencePreview));
    expect(sha256.convert(preview.image), sha256.convert(original));
    expect(preview.width, 100);
    expect(preview.height, 200);
    final first = (await t.runAsync(() => store.list(owner)))!.single;
    expect(first['report']['height_m'], closeTo(4, 1e-10));
    expect(first['server_snapshot']['report']['species']['display_name'],
        'Тестовая сосна');
    expect(
        first['server_snapshot']['report']['measurements']['height']['value_m'],
        null);
    expect(first['server_parent_id'], parentId);
    await tapText(t, 'Выделить дерево на этом фото');
    expect(requests, 1);
    expect(pickerCalls, 0);
    expect(find.textContaining('Модель не нашла дерево.'), findsOneWidget);
    final after =
        (await t.runAsync(() => store.load(owner, first['draft_id'])))!;
    expect(after['report']['height_m'], closeTo(4, 1e-10));
    expect(after['server_snapshot']['report']['tree']['detected'], false);
    expect(after['server_snapshot']['ar'], seed['ar']);
    await t.pumpWidget(const SizedBox());
    await mount(
        t,
        ReferenceMeasurementPage(
            draftId: first['draft_id'],
            drafts: ContourDrafts(directory: () async => folder),
            analysisService: model));
    await t.scrollUntilVisible(find.byType(TextField), 150,
        scrollable: find.byType(Scrollable).first);
    await t.runAsync(() => t.enterText(find.byType(TextField), '100'));
    await settle(t);
    await t.ensureVisible(find.byType(DropdownButton<String>));
    await t.tap(find.byType(DropdownButton<String>));
    await t.pumpAndSettle();
    await t.runAsync(() => t.tap(find.text('Сантиметры').last));
    await settle(t);
    await tapText(t, 'Сохранить отчёт на устройстве');
    final reopened =
        (await t.runAsync(() => store.load(owner, first['draft_id'])))!;
    expect(reopened['report']['height_m'], closeTo(4, 1e-10));
    expect(reopened['reference'], first['reference']);
    expect(reopened['server_snapshot']['report']['tree']['detected'], false);
    expect(reopened['server_parent_id'], parentId);
    expect(requests, 1);
    await t.pumpWidget(const SizedBox());
  });
  testWidgets(
      'network failure keeps manual draft and prior valid model response; repeat works',
      (t) async {
    final originalGeometry = {
      'crown_height': {'value': null, 'unit': 'm', 'reason': 'fresh_model_v1'}
    };
    seed['kind'] = 'reference';
    seed['report']['geometry'] = measurement().geometry;
    seed['report']['model_geometry'] = {'old_model_geometry': true};
    var calls = 0;
    final model = PhotoAnalysisService(
        clientFactory: () => InspectClient((_) async {
              calls++;
              if (calls == 1) throw const SocketException('offline');
              return response({
                'tree': {'detected': true},
                'species': {'display_name': 'повтор'},
                'measurement_method_version': 1,
                'geometry': originalGeometry,
                'images': {'width': 100, 'height': 200}
              });
            }));
    await mount(
        t,
        ReferenceMeasurementPage(
            initialImage: original,
            initialSnapshot: seed,
            analysisId: analysisId,
            drafts: store,
            analysisService: model));
    await tapText(t, 'Выделить дерево на этом фото');
    final rows = (await t.runAsync(() => store.list(owner)))!;
    expect(rows.single['server_snapshot']['report']['species']['display_name'],
        'Тестовая сосна');
    expect(rows.single['report']['height_m'], closeTo(4, 1e-10));
    expect(rows.single['server_snapshot']['report']['model_geometry'],
        {'old_model_geometry': true});
    await tapText(t, 'Выделить дерево на этом фото');
    expect(calls, 2);
    final retry = (await t.runAsync(() => store.list(owner)))!.single;
    expect(retry['server_snapshot']['report']['species']['display_name'],
        'повтор');
    expect(retry['report']['height_m'], closeTo(4, 1e-10));
    expect(
        retry['server_snapshot']['report']['model_geometry'], originalGeometry);
    final merged = combinedReferenceSnapshot(
        base: Map<String, dynamic>.from(retry['server_snapshot']),
        reference: measurement().toJson(),
        manualReport: measurement().report,
        capturedAt: '2026-10-13T00:00:00Z',
        environment: null);
    final sources = modelReportForSources(merged);
    expect(sources['geometry'], originalGeometry);
    expect(sources['measurement_method_version'], 1);
    expect(merged['report']['geometry'], measurement().geometry);
    await t.pumpWidget(const SizedBox());
    await mount(
        t, ReferenceMeasurementPage(draftId: retry['draft_id'], drafts: store));
    final reopened = (await t.runAsync(() => store.list(owner)))!.single;
    expect(reopened['server_snapshot']['report']['model_geometry'],
        originalGeometry);
    await t.pumpWidget(const SizedBox());
  });
  testWidgets(
      'save sends combined child, original model measurements and same original',
      (t) async {
    const executionId = '99999999-9999-4999-8999-999999999999';
    seed['report']['analysis_id'] = executionId;
    Map<String, dynamic>? sentSnapshot;
    String? sentParent;
    final history = ReportHistoryService(
        journal: ContourDrafts(
            directory: () async => Directory('${folder.path}/history')),
        clientFactory: () => InspectClient((request) async {
              if (request.method == 'GET') {
                return response({
                  'reference_versions': [1, 2],
                  'environment_version': 1
                });
              }
              final multi = request as http.MultipartRequest;
              expect(multi.fields['analysis_id'], analysisId);
              sentSnapshot = jsonDecode(multi.fields['snapshot']!);
              sentParent = multi.fields['parent_id'];
              expect(await multi.files.single.finalize().toBytes(), original);
              return response({
                'saved': true,
                'persisted': true,
                'record': {'version_id': multi.fields['version_id']}
              });
            }));
    await mount(
        t,
        ReferenceMeasurementPage(
            initialImage: original,
            initialSnapshot: seed,
            analysisId: analysisId,
            parentId: parentId,
            drafts: store,
            history: history));
    await tapText(t, 'Сохранить в аккаунте / новую версию');
    expect(sentParent, parentId);
    expect(sentSnapshot!['kind'], 'reference');
    expect(sentSnapshot!['report']['analysis_id'], executionId);
    expect(sentSnapshot!['reference'], measurement().toJson());
    expect(sentSnapshot!['report']['measurements'],
        seed['report']['measurements']);
    expect(sentSnapshot!['report']['images']['mask_image_base64'],
        seed['report']['images']['mask_image_base64']);
    expect(sentSnapshot!['report']['height_m'], closeTo(4, 1e-10));
    expect(sentSnapshot!['ar'], seed['ar']);
    await t.pumpWidget(const SizedBox());
  });
  testWidgets('photo binding mismatch never creates a reference draft',
      (t) async {
    seed['image'] = {'sha256': 'other-original'};
    await mount(
        t,
        ReferenceMeasurementPage(
            initialImage: original,
            initialSnapshot: seed,
            analysisId: analysisId,
            drafts: store));
    expect(find.text('FormatException: Отчёт относится к другому фото.'),
        findsOneWidget);
    expect(find.byType(ReferencePreview), findsNothing);
    expect((await t.runAsync(() => store.list(owner)))!, isEmpty);
    await t.pumpWidget(const SizedBox());
  });
  testWidgets(
      'manual tree draft saves exact revision then pins it with reference, no moderation',
      (t) async {
    t.view.physicalSize = const Size(800, 1600);
    t.view.devicePixelRatio = 1;
    addTearDown(t.view.resetPhysicalSize);
    addTearDown(t.view.resetDevicePixelRatio);
    const correctionId = 'eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee';
    final contours = ContourDrafts(
        directory: () async => Directory('${folder.path}/contours'));
    final binary = im.Image(width: 100, height: 200, numChannels: 4);
    for (final p in binary) {
      p.setRgba(p.x > 40 && p.x < 60 ? 255 : 0, p.x > 40 && p.x < 60 ? 255 : 0,
          p.x > 40 && p.x < 60 ? 255 : 0, 255);
    }
    final mask = Uint8List.fromList(im.encodePng(binary));
    final editor = const ContourEditorState(
        width: 100,
        height: 200,
        closed: true,
        points: [
          Offset(.4, .1),
          Offset(.6, .1),
          Offset(.6, .9),
          Offset(.4, .9)
        ]).toJson();
    await t.runAsync(() => contours.save(
        owner,
        '${analysisId}_root',
        {
          'analysis_id': analysisId,
          'editor_state': editor,
          'mask_png_base64': base64Encode(mask),
          'status': 'draft'
        },
        original));
    var posts = 0;
    final auth = CorrectionsService(
        clientFactory: () => InspectClient((request) async {
              if (request.method == 'GET') {
                return response({'workflow_version': 1});
              }
              expect(request.url.path.endsWith('/workflow'), true);
              posts++;
              return response({
                'saved': true,
                'workflow_version': 1,
                'correction_id': correctionId,
                'review_status': 'draft'
              });
            }));
    seed['report']['tree'] = {'detected': false};
    await mount(
        t,
        ReferenceMeasurementPage(
            initialImage: original,
            initialSnapshot: seed,
            analysisId: analysisId,
            parentId: parentId,
            drafts: store,
            contourDrafts: contours,
            service: auth));
    await tapText(t, 'Создать / исправить контур дерева');
    expect(find.byType(ContourWorkspacePage), findsOneWidget);
    await tapText(t, 'Сохранить контур');
    await tapText(t, 'Использовать этот контур в отчёте');
    expect(posts, 1);
    final pinned = (await t.runAsync(() => store.list(owner)))!.single;
    expect(pinned['server_snapshot']['correction_id'], correctionId);
    expect(pinned['server_snapshot']['report']['contour_preview'], {
      'correction_id': correctionId,
      'photo_sha256': sha256.convert(original).toString(),
      'mask_png_base64': base64Encode(mask),
      'width': 100,
      'height': 200
    });
    expect(pinned['report']['height_m'], closeTo(4, 1e-10));
    expect(pinned['server_snapshot']['report']['tree']['detected'], false);
    expect(pinned['server_snapshot']['report']['measurements'],
        seed['report']['measurements']);
    await t.pumpWidget(const SizedBox());
    await mount(t,
        ReferenceMeasurementPage(draftId: pinned['draft_id'], drafts: store));
    await t.scrollUntilVisible(
        find.textContaining('Зелёным показана закреплённая'), 150,
        scrollable: find.byType(Scrollable).first);
    expect(
        find.textContaining('Зелёным показана закреплённая'), findsOneWidget);
    await t.scrollUntilVisible(
        find.textContaining('В сохранённом результате модель не нашла дерево.'),
        150,
        scrollable: find.byType(Scrollable).first);
    expect(
        find.textContaining('В сохранённом результате модель не нашла дерево.'),
        findsOneWidget);
    await t.scrollUntilVisible(find.byType(ReferencePreview), -150,
        scrollable: find.byType(Scrollable).first);
    final preview = t.widget<ReferencePreview>(find.byType(ReferencePreview));
    expect(preview.correctionId, correctionId);
    await t.pumpWidget(const SizedBox());
  });
  testWidgets(
      'account switch discards pending inference UI, owner draft stays isolated',
      (t) async {
    final sent = Completer<void>(), result = Completer<http.StreamedResponse>();
    final model = PhotoAnalysisService(
        clientFactory: () => InspectClient((_) {
              sent.complete();
              return result.future;
            }));
    await mount(
        t,
        ReferenceMeasurementPage(
            initialImage: original,
            initialSnapshot: seed,
            analysisId: analysisId,
            drafts: store,
            analysisService: model));
    await t.scrollUntilVisible(find.text('Выделить дерево на этом фото'), 150,
        scrollable: find.byType(Scrollable).first);
    await t.runAsync(() => t.tap(find.text('Выделить дерево на этом фото')));
    await settle(t);
    await sent.future;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('arborscan_auth_token', 'other');
    await prefs.setString(
        'arborscan_user_id', 'dddddddd-dddd-4ddd-8ddd-dddddddddddd');
    CorrectionsService.authChanges.value++;
    result.complete(response({
      'tree': {'detected': true},
      'species': {'display_name': 'late-account-result'}
    }));
    await settle(t);
    expect(find.byType(ReferencePreview), findsNothing);
    expect(find.textContaining('Аккаунт изменился'), findsOneWidget);
    final rows = (await t.runAsync(() => store.list(owner)))!;
    expect(rows.single['server_snapshot']['report']['species']['display_name'],
        'Тестовая сосна');
    expect(
        (await t.runAsync(
            () => store.list('dddddddd-dddd-4ddd-8ddd-dddddddddddd')))!,
        isEmpty);
    await t.pumpWidget(const SizedBox());
  });
}
