import 'dart:convert';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:arborscan_app/image_line_page.dart';
import 'package:arborscan_app/reference_preview.dart';
import 'package:arborscan_app/stick_page.dart';
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;

final _photo = img.encodePng(img.Image(width: 100, height: 100));

Future<void> _settleImage(WidgetTester tester) async {
  await tester.pump();
  await tester.runAsync(
      () async => Future<void>.delayed(const Duration(milliseconds: 100)));
  await tester.pumpAndSettle();
}

Future<ByteData> _render(
    WidgetTester tester, CustomPainter painter, Size size, double scale) async {
  final recorder = ui.PictureRecorder();
  final canvas = Canvas(recorder)
    ..translate(32 - size.width / 2 * scale, 32 - size.height / 2 * scale)
    ..scale(scale);
  painter.paint(canvas, size);
  final picture = recorder.endRecording();
  final result = await tester.runAsync(() async {
    final image = await picture.toImage(64, 64);
    final pixels = await image.toByteData(format: ui.ImageByteFormat.rawRgba);
    image.dispose();
    return pixels!;
  });
  picture.dispose();
  return result!;
}

Rect _bounds(ByteData data) {
  int left = 64, top = 64, right = -1, bottom = -1;
  for (var y = 0; y < 64; y++) {
    for (var x = 0; x < 64; x++) {
      if (data.getUint8((y * 64 + x) * 4 + 3) < 10) continue;
      if (x < left) left = x;
      if (y < top) top = y;
      if (x > right) right = x;
      if (y > bottom) bottom = y;
    }
  }
  return Rect.fromLTRB(left.toDouble(), top.toDouble(), (right + 1).toDouble(),
      (bottom + 1).toDouble());
}

Future<void> _compact(WidgetTester tester, Finder paint,
    TransformationController controller, List<double> scales) async {
  Rect? baseline;
  for (final scale in scales) {
    controller.value = Matrix4.diagonal3Values(scale, scale, 1);
    if (scale > 1) {
      expect(
          tester.renderObject<RenderCustomPaint>(paint).debugNeedsPaint, true);
    }
    await tester.pump();
    final pixels = await _render(
        tester,
        tester.widget<CustomPaint>(paint).painter!,
        tester.getSize(paint),
        scale);
    baseline ??= _bounds(pixels);
    expect(_bounds(pixels), baseline);
    expect(baseline.width, inInclusiveRange(6, 12));
    expect(baseline.height, inInclusiveRange(6, 12));
    expect(pixels.getUint8((32 * 64 + 32) * 4 + 3), 0,
        reason: 'The marker must leave the object at its centre visible.');
  }
}

void main() {
  testWidgets(
      'all reference annotations and tree selector use hollow zoom-safe rings',
      (tester) async {
    final controller = TransformationController();
    addTearDown(controller.dispose);
    for (final field in ReferencePreviewPainter.colors.keys) {
      Rect? baseline;
      for (final scale in [1.0, 2.0, 5.0, 10.0, 12.0]) {
        controller.value = Matrix4.diagonal3Values(scale, scale, 1);
        final reference = {
          field: [
            {'x': .5, 'y': .5}
          ]
        };
        final pixels = await _render(
            tester,
            ReferencePreviewPainter(reference, null, controller: controller),
            const Size(100, 100),
            scale);
        baseline ??= _bounds(pixels);
        expect(_bounds(pixels), baseline, reason: field);
        expect(baseline.width, inInclusiveRange(6, 10));
        expect(pixels.getUint8((32 * 64 + 32) * 4 + 3), 0, reason: field);
        final hidden = await _render(
            tester,
            ReferencePreviewPainter(reference, const Offset(.5, .5),
                controller: controller, showHandles: false),
            const Size(100, 100),
            scale);
        expect(_bounds(hidden).isEmpty, true);
      }
    }
    for (final scale in [1.0, 12.0]) {
      controller.value = Matrix4.diagonal3Values(scale, scale, 1);
      final pixels = await _render(
          tester,
          ReferencePreviewPainter(null, const Offset(.5, .5),
              controller: controller),
          const Size(100, 100),
          scale);
      expect(_bounds(pixels).width, inInclusiveRange(6, 10));
      expect(pixels.getUint8((32 * 64 + 32) * 4 + 3), 0);
    }
  });

  for (final scale in [1.0, 5.0]) {
    testWidgets(
        'line editor adds a close endpoint with unchanged original coordinates at $scale',
        (tester) async {
      await tester.binding.setSurfaceSize(const Size(600, 900));
      addTearDown(() => tester.binding.setSurfaceSize(null));
      List<Offset>? result;
      await tester.pumpWidget(MaterialApp(
          home: Builder(
              builder: (context) => TextButton(
                  onPressed: () async {
                    result = await Navigator.push<List<Offset>>(
                        context,
                        MaterialPageRoute(
                            builder: (_) => ImageLinePage(
                                image: _photo,
                                width: 100,
                                height: 100,
                                title: 'Высота эталона',
                                initial: const [Offset(.5, .5)])));
                  },
                  child: const Text('Open')))));
      await tester.tap(find.text('Open'));
      await _settleImage(tester);
      final canvas = find.byKey(const ValueKey('image-line-canvas'));
      final paint =
          find.descendant(of: canvas, matching: find.byType(CustomPaint));
      final viewer =
          tester.widget<InteractiveViewer>(find.byType(InteractiveViewer));
      final controller = viewer.transformationController!;
      await _compact(tester, paint, controller, [1, 2, 5, 10, 12]);
      controller.value = Matrix4.identity();
      await tester.tap(find.byTooltip('Скрыть точки'));
      await tester.pump();
      expect(
          _bounds(await _render(
                  tester,
                  tester.widget<CustomPaint>(paint).painter!,
                  tester.getSize(paint),
                  1))
              .isEmpty,
          true);
      await tester.tap(find.text('Конец'));
      await tester.pump();
      final size = tester.getSize(canvas);
      controller.value = Matrix4.identity()
        ..translateByDouble(
            size.width / 2 * (1 - scale), size.height / 2 * (1 - scale), 0, 1)
        ..scaleByDouble(scale, scale, 1, 1);
      await tester.pump();
      final position = tester.getTopLeft(find.byType(InteractiveViewer)) +
          size.center(Offset.zero);
      await tester.tapAt(position + const Offset(3, 0));
      await tester.pump();
      await tester.tap(find.text('Применить отрезок'));
      await tester.pumpAndSettle();
      expect(result, hasLength(2));
      expect(result!.first, const Offset(.5, .5));
      expect(result!.last.dx, closeTo(.5 + 3 / scale / size.width, 1e-8));
      expect(result!.last.dy, closeTo(.5, 1e-8));
      expect(tester.takeException(), isNull);
    });
  }

  testWidgets(
      'legacy ruler close points, hide, pinch and cancel preserve the scale',
      (tester) async {
    await tester.binding.setSurfaceSize(const Size(600, 900));
    addTearDown(() => tester.binding.setSurfaceSize(null));
    double? result;
    await tester.pumpWidget(MaterialApp(
        home: Builder(
            builder: (context) => TextButton(
                onPressed: () async {
                  result = await Navigator.push<double>(
                      context,
                      MaterialPageRoute(
                          builder: (_) => StickPage(
                              originalImageBase64: base64Encode(_photo),
                              currentScalePxToM: 0)));
                },
                child: const Text('Open')))));
    await tester.tap(find.text('Open'));
    await _settleImage(tester);
    final canvas = find.byKey(const ValueKey('legacy-reference-canvas'));
    final paint =
        find.descendant(of: canvas, matching: find.byType(CustomPaint));
    final controller = tester
        .widget<InteractiveViewer>(find.byType(InteractiveViewer))
        .transformationController!;
    final size = tester.getSize(canvas);
    final centre = tester.getCenter(canvas);
    await tester.tapAt(centre);
    await tester.pump();
    await _compact(tester, paint, controller, [1, 2, 5, 8]);
    controller.value = Matrix4.identity();
    await tester.pump();
    await tester.tapAt(centre + const Offset(3, 0));
    await tester.pump();
    final drag = await tester.startGesture(centre, pointer: 1);
    await drag.moveBy(const Offset(-20, 0));
    await tester.pump();
    await drag.cancel();
    await tester.pump();
    final first = await tester.startGesture(centre, pointer: 1);
    await first.moveBy(const Offset(-20, 0));
    await tester.pump();
    final second =
        await tester.startGesture(centre + const Offset(60, 0), pointer: 2);
    await tester.pump();
    await first.moveBy(const Offset(-10, 0));
    await second.moveBy(const Offset(10, 0));
    await first.up();
    await second.up();
    await tester.pumpAndSettle();
    await tester.tap(find.byTooltip('Скрыть точки'));
    await tester.pump();
    await tester.tap(find.byTooltip('Применить'));
    await tester.pumpAndSettle();
    await tester.enterText(find.byType(TextField), '1');
    await tester.tap(find.text('ГОТОВО'));
    await tester.pumpAndSettle();
    expect(result, closeTo(1 / (3 / size.width * 100), 1e-8));
    expect(tester.takeException(), isNull);
  });
}
