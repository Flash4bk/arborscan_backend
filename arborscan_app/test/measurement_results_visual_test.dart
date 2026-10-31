import 'dart:convert';
import 'dart:io';
import 'dart:ui' as ui;

import 'package:arborscan_app/app_theme.dart';
import 'package:arborscan_app/contour_drafts.dart';
import 'package:arborscan_app/reference_measurement_page.dart';
import 'package:arborscan_app/reference_preview.dart';
import 'package:arborscan_app/species_annotation.dart';
import 'package:crypto/crypto.dart';
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as im;
import 'package:shared_preferences/shared_preferences.dart';

import 'report_export_test.dart' as fixtures;

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  setUpAll(() async {
    for (final entry in {
      'Atlas': 'assets/fonts/DejaVuSans.ttf',
      'Literata': 'assets/fonts/Literata.ttf',
      'MaterialIcons': 'fonts/MaterialIcons-Regular.otf',
    }.entries) {
      final loader = FontLoader(entry.key)
        ..addFont(rootBundle.load(entry.value));
      await loader.load();
    }
  });
  testWidgets('real reference screen renders compact frozen results and layers',
      (tester) async {
    await tester.binding.setSurfaceSize(const Size(411, 891));
    addTearDown(() => tester.binding.setSurfaceSize(null));
    SharedPreferences.setMockInitialValues({
      'arborscan_auth_token': 'visual-test-only',
      'arborscan_user_id': 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa',
    });
    final folder = (await tester
        .runAsync(() => Directory.systemTemp.createTemp('results32-')))!;
    final fixture = fixtures.fixture('visual32');
    final snapshot = fixture.snapshot;
    snapshot['reference']['same_plane'] = true;
    const correction = 'eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee';
    final mask = im.Image(width: 400, height: 600);
    im.fillRect(mask,
        x1: 190, y1: 70, x2: 210, y2: 550, color: im.ColorRgb8(255, 255, 255));
    im.fillCircle(mask,
        x: 200, y: 175, radius: 105, color: im.ColorRgb8(255, 255, 255));
    snapshot['correction_id'] = correction;
    snapshot['report']['species_annotation'] = {
      'version': 1,
      'source': 'user',
      'confirmed': false,
      'display_name': 'Тестовое дерево',
    };
    snapshot['report']['contour_preview'] = {
      'correction_id': correction,
      'photo_sha256': sha256.convert(fixture.photo!).toString(),
      'width': 400,
      'height': 600,
      'mask_png_base64': base64Encode(im.encodePng(mask)),
    };
    final capture = GlobalKey();
    final dir = Directory(Platform.environment['ARBORSCAN_RESULTS_UI_OUTPUT'] ??
        '../output/recognition-results/ui')
      ..createSync(recursive: true);
    Future<void> screenshot(String name) async {
      await tester.pumpAndSettle();
      final boundary =
          capture.currentContext!.findRenderObject() as RenderRepaintBoundary;
      await tester.runAsync(() async {
        final image = await boundary.toImage(pixelRatio: 2);
        final bytes = await image.toByteData(format: ui.ImageByteFormat.png);
        await File('${dir.path}/$name.png')
            .writeAsBytes(bytes!.buffer.asUint8List());
        image.dispose();
      });
    }

    try {
      await tester.pumpWidget(RepaintBoundary(
          key: capture,
          child: MaterialApp(
              theme: AppTheme.light(),
              debugShowCheckedModeBanner: false,
              home: ReferenceMeasurementPage(
                  initialImage: fixture.photo,
                  initialSnapshot: snapshot,
                  drafts: ContourDrafts(directory: () async => folder)))));
      for (var i = 0; i < 80; i++) {
        await tester.runAsync(
            () => Future<void>.delayed(const Duration(milliseconds: 20)));
        await tester.pump();
        if (find.text('Тестовое дерево').evaluate().isNotEmpty) break;
      }
      await tester.pumpAndSettle();
      expect(find.text('Тестовое дерево'), findsOneWidget);
      await tester.runAsync(() async {
        for (final element in find.byType(Image).evaluate()) {
          final widget = element.widget as Image;
          await precacheImage(widget.image, element);
        }
      });
      await tester.pumpAndSettle();
      expect(
          tester
              .widgetList<RawImage>(find.byType(RawImage))
              .where((widget) => widget.image != null)
              .length,
          greaterThanOrEqualTo(2));
      expect(find.text('Экспорт PDF'), findsNothing);
      expect(find.text('Выбрать фото'), findsNothing);
      expect(tester.getRect(find.byType(SpeciesAnnotationCard)).top,
          greaterThan(tester.getRect(find.byType(ReferencePreview)).bottom));
      await screenshot('reference-top33');
      final summary = find.text('Результат измерений и расчётов');
      await tester.scrollUntilVisible(summary, 150,
          scrollable: find.byType(Scrollable).first);
      await tester.ensureVisible(summary);
      await tester.pumpAndSettle();
      expect(
          find.descendant(
              of: find.byKey(const ValueKey('measurement-summary-trunk')),
              matching: find.textContaining('0,200 м')),
          findsOneWidget);
      expect(find.textContaining('Коэффициент β (кг/с): не рассчитан'),
          findsOneWidget);
      expect(find.textContaining('не рассчитана'), findsNWidgets(2));
      expect(find.textContaining('44,4'), findsNothing);
      await screenshot('reference-summary33');
      await tester.scrollUntilVisible(find.text('Фото и экспорт'), 100,
          scrollable: find.byType(Scrollable).first);
      await tester.ensureVisible(find.text('Фото и экспорт'));
      await tester.tap(find.text('Фото и экспорт'));
      await tester.pumpAndSettle();
      expect(find.text('Выбрать фото'), findsOneWidget);
      expect(find.text('Экспорт PDF'), findsOneWidget);
      expect(find.text('Как измерять по эталону'), findsOneWidget);
      expect(tester.takeException(), isNull);
    } finally {
      await tester.pumpWidget(const SizedBox.shrink());
      if (!folder.absolute.path
          .startsWith(Directory.systemTemp.absolute.path)) {
        throw StateError('Unexpected fixture cleanup path');
      }
      await tester.runAsync(() => folder.delete(recursive: true));
    }
  });
}
