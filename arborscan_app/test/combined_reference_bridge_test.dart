import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:crypto/crypto.dart';
import 'package:arborscan_app/analyze_page.dart';
import 'package:arborscan_app/app_theme.dart';
import 'package:arborscan_app/reference_measurement_page.dart';
import 'package:arborscan_app/reference_preview.dart';
import 'package:arborscan_app/unified_analysis_models.dart';
import 'package:arborscan_app/unified_analysis_report_page.dart';

void main() {
  const owner = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa';
  const aid = 'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb';
  late Directory folder;
  late Uint8List original;
  setUp(() async {
    SharedPreferences.setMockInitialValues(
        {'arborscan_auth_token': 'test', 'arborscan_user_id': owner});
    folder = await Directory.systemTemp.createTemp('reference-ui-bridge-');
    original = await File('test/fixtures/reference_exif6.jpg').readAsBytes();
    const path = MethodChannel('plugins.flutter.io/path_provider');
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(path, (_) async => folder.path);
  });
  tearDown(() async {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(
            const MethodChannel('plugins.flutter.io/path_provider'), null);
    await folder.delete(recursive: true);
  });
  Future<void> settle(WidgetTester t) async {
    for (var i = 0; i < 10; i++) {
      await t.runAsync(() async {
        await t.pump(const Duration(milliseconds: 100));
        await Future<void>.delayed(const Duration(milliseconds: 20));
      });
    }
    for (var i = 0;
        i < 250 && find.byType(LinearProgressIndicator).evaluate().isNotEmpty;
        i++) {
      await t.runAsync(() async {
        await t.pump(const Duration(milliseconds: 50));
        await Future<void>.delayed(const Duration(milliseconds: 20));
      });
    }
    expect(find.byType(LinearProgressIndicator), findsNothing);
  }

  testWidgets(
      'main selected photo enters reference without another picker call',
      (t) async {
    const picker = MethodChannel('plugins.flutter.io/image_picker');
    const available = MethodChannel('arborscan/image_selection');
    final messenger = t.binding.defaultBinaryMessenger;
    var pickerCalls = 0;
    messenger.setMockMethodCallHandler(picker, (_) async {
      pickerCalls++;
      return File('test/fixtures/reference_exif6.jpg').absolute.path;
    });
    messenger.setMockMethodCallHandler(available, (_) async => true);
    addTearDown(() {
      messenger.setMockMethodCallHandler(picker, null);
      messenger.setMockMethodCallHandler(available, null);
    });
    await t.runAsync(() => t.pumpWidget(MaterialApp(
        theme: AppTheme.light(), home: const Scaffold(body: ArborScanPage()))));
    await settle(t);
    await t.ensureVisible(find.text('Галерея'));
    await t.pumpAndSettle();
    await t.runAsync(() => t.tap(find.text('Галерея')));
    await settle(t);
    await t.scrollUntilVisible(find.text('По эталону'), 150,
        scrollable: find.byType(Scrollable).first);
    await t.runAsync(() => t.tap(find.text('По эталону')));
    await settle(t);
    final page = t.widget<ReferenceMeasurementPage>(
        find.byType(ReferenceMeasurementPage));
    expect(sha256.convert(page.initialImage!), sha256.convert(original));
    expect(pickerCalls, 1);
    await t.scrollUntilVisible(find.byType(ReferencePreview), 150,
        scrollable: find.byType(Scrollable).first);
    final preview = t.widget<ReferencePreview>(find.byType(ReferencePreview));
    expect(preview.width, 100);
    expect(preview.height, 200);
    expect(sha256.convert(preview.image), sha256.convert(original));
    await t.pumpWidget(const SizedBox());
  });
  testWidgets(
      'live analysis report bridges exact photo, model result and parent into reference',
      (t) async {
    const parent = 'cccccccc-cccc-4ccc-8ccc-cccccccccccc';
    final raw = <String, dynamic>{
      'analysis_id': aid,
      'analysis_status': 'absolute_scale_required',
      'tree': {'detected': true},
      'species': {'display_name': 'Тестовая сосна'},
      'measurements': {
        'height': {'value_px': 160, 'value_m': null, 'source': 'vision'}
      },
      'images': {'width': 100, 'height': 200},
      'captured_at': '2026-10-13T00:00:00Z'
    };
    final snapshot = {
      'report': raw,
      'captured_at': raw['captured_at'],
      'image': {'sha256': sha256.convert(original).toString()},
      'ar': {'association': 'user_confirmed_same_tree'}
    };
    await t.runAsync(() => t.pumpWidget(MaterialApp(
        theme: AppTheme.light(),
        home: UnifiedAnalysisReportPage(
            result: UnifiedAnalysisResult.fromJson(raw),
            fallbackImageBytes: original,
            allowServerSave: false,
            referenceParentId: parent,
            referenceSnapshot: snapshot))));
    await settle(t);
    await t.scrollUntilVisible(find.text('Добавить эталон к этому фото'), 150,
        scrollable: find.byType(Scrollable).first);
    await t.runAsync(() => t.tap(find.text('Добавить эталон к этому фото')));
    await settle(t);
    final page = t.widget<ReferenceMeasurementPage>(
        find.byType(ReferenceMeasurementPage));
    expect(sha256.convert(page.initialImage!), sha256.convert(original));
    expect(page.analysisId, aid);
    expect(page.parentId, parent);
    expect(
        page.initialSnapshot!['report']['measurements'], raw['measurements']);
    expect(page.initialSnapshot!['report']['species'], raw['species']);
    expect(page.initialSnapshot!['ar'], snapshot['ar']);
    expect(
        find.textContaining('Отчёт относится к другому фото.'), findsNothing);
    await t.pumpWidget(const SizedBox());
  });

  testWidgets('tree target remains in oriented original coordinates after zoom',
      (t) async {
    Offset? selected;
    await t.pumpWidget(MaterialApp(
        home: Builder(
            builder: (context) => Scaffold(
                body: FilledButton(
                    onPressed: () async {
                      selected = await Navigator.push<Offset>(
                          context,
                          MaterialPageRoute(
                              builder: (_) => SelectTreePointPage(
                                  image: original,
                                  width: 100,
                                  height: 200,
                                  sessionToken: 'test')));
                    },
                    child: const Text('select'))))));
    await t.tap(find.text('select'));
    await t.pumpAndSettle();
    final viewer = t.widget<InteractiveViewer>(find.byType(InteractiveViewer));
    viewer.transformationController!.value = Matrix4.diagonal3Values(2, 2, 1);
    await t.pump();
    final box = t.renderObject<RenderBox>(
        find.byKey(const ValueKey('tree-selection-canvas')));
    await t.tapAt(
        box.localToGlobal(Offset(box.size.width * .25, box.size.height * .3)));
    await t.pump();
    final preview = t.widget<ReferencePreview>(find.byType(ReferencePreview));
    expect(preview.tap!.dx, closeTo(.25, 1e-9));
    expect(preview.tap!.dy, closeTo(.3, 1e-9));
    await t.runAsync(() => t.tap(find.text('Выбрать это дерево')));
    await settle(t);
    expect(selected!.dx, closeTo(.25, 1e-9));
    expect(selected!.dy, closeTo(.3, 1e-9));
    await t.pumpWidget(const SizedBox());
  });

  testWidgets('tree target cannot return across an unannounced account change',
      (t) async {
    await t.pumpWidget(MaterialApp(
        home: SelectTreePointPage(
            image: original,
            width: 100,
            height: 200,
            initial: const Offset(.25, .3),
            sessionToken: 'test')));
    await t.pumpAndSettle();
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('arborscan_auth_token', 'other-account');
    await t.runAsync(() => t.tap(find.text('Выбрать это дерево')));
    await settle(t);
    expect(find.text('Аккаунт изменился. Закройте выбор.'), findsOneWidget);
    expect(find.byKey(const ValueKey('tree-selection-canvas')), findsNothing);
    expect(t.takeException(), isNull);
    await t.pumpWidget(const SizedBox());
  });
}
