import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:arborscan_app/reference_preview.dart';
import 'package:arborscan_app/report_export_data.dart';
import 'package:arborscan_app/report_pdf.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as im;
import 'report_export_test.dart' as fixtures;
import 'combined_reference_export_test.dart' as combined;

Future<ByteData> pixels(WidgetTester tester, CustomPainter painter,
    {double scale = 1}) async {
  final recorder = ui.PictureRecorder();
  final canvas = Canvas(recorder)
    ..translate(32 - 50 * scale, 32 - 50 * scale)
    ..scale(scale);
  painter.paint(canvas, const Size(100, 100));
  final picture = recorder.endRecording();
  final result = await tester.runAsync(() async {
    final image = await picture.toImage(64, 64);
    final data = await image.toByteData(format: ui.ImageByteFormat.rawRgba);
    image.dispose();
    return data!;
  });
  picture.dispose();
  return result!;
}

int inkWidth(ByteData pixels) {
  var left = 64, right = -1;
  for (var y = 0; y < 64; y++) {
    for (var x = 0; x < 64; x++) {
      if (pixels.getUint8((y * 64 + x) * 4 + 3) < 10) continue;
      if (x < left) left = x;
      if (x > right) right = x;
    }
  }
  return right < left ? 0 : right - left + 1;
}

void main() {
  testWidgets('each layer can be toggled without changing saved coordinates',
      (tester) async {
    final photo =
        Uint8List.fromList(im.encodePng(im.Image(width: 100, height: 100)));
    final mask = im.Image(width: 100, height: 100, numChannels: 4);
    im.fillRect(mask,
        x1: 40, y1: 10, x2: 60, y2: 90, color: im.ColorRgba8(0, 255, 0, 100));
    final reference = {
      for (final field in ReferencePreviewPainter.labels.keys)
        field: [
          {'x': .5, 'y': .5}
        ],
    };
    final before = jsonEncode(reference);
    Set<String>? lastSelection;
    await tester.pumpWidget(MaterialApp(
        home: Scaffold(
            body: SizedBox(
                width: 300,
                height: 300,
                child: ReferencePreview(
                    image: photo,
                    width: 100,
                    height: 100,
                    reference: reference,
                    tap: const Offset(.5, .5),
                    report: {
                      'images': {
                        'mask_image_base64': base64Encode(im.encodePng(mask))
                      }
                    },
                    onHiddenLayersChanged: (hidden) =>
                        lastSelection = hidden)))));
    await tester.pumpAndSettle();
    final preview = find.byType(ReferencePreview);
    final images = find.descendant(of: preview, matching: find.byType(Image));
    expect(images, findsNWidgets(2));
    await tester.tap(find.byTooltip('Слои фото'));
    await tester.pumpAndSettle();
    final fields = [
      'mask',
      ...ReferencePreviewPainter.labels.keys,
      'selection'
    ];
    for (final field in fields) {
      final tile = find.byKey(ValueKey('photo-layer-$field'));
      await tester.ensureVisible(tile);
      expect(tester.widget<CheckboxListTile>(tile).value, true);
      await tester.tap(tile);
      await tester.pump();
      expect(tester.widget<CheckboxListTile>(tile).value, false);
    }
    expect(lastSelection, fields.toSet());
    await tester.ensureVisible(find.text('Готово'));
    await tester.tap(find.text('Готово'));
    await tester.pumpAndSettle();
    expect(images, findsOneWidget);
    final paint = find.descendant(
        of: preview,
        matching: find.byWidgetPredicate((widget) =>
            widget is CustomPaint &&
            widget.painter is ReferencePreviewPainter));
    expect(
        inkWidth(
            await pixels(tester, tester.widget<CustomPaint>(paint).painter!)),
        0);
    expect(jsonEncode(reference), before);
    await tester.tap(find.byTooltip('Слои фото'));
    await tester.pumpAndSettle();
    final treeTile = find.byKey(const ValueKey('photo-layer-tree'));
    await tester.ensureVisible(treeTile);
    await tester.tap(treeTile);
    await tester.pump();
    await tester.ensureVisible(find.text('Готово'));
    await tester.tap(find.text('Готово'));
    await tester.pumpAndSettle();
    final width = inkWidth(
        await pixels(tester, tester.widget<CustomPaint>(paint).painter!));
    expect(width, inInclusiveRange(4, 8));
    expect(images, findsOneWidget,
        reason: 'Only the tree height was restored, not its mask.');
    expect(tester.takeException(), isNull);
  });

  testWidgets(
      'result handles are smaller and zoom-stable; editor handles unchanged',
      (tester) async {
    final reference = {
      'tree': [
        {'x': .5, 'y': .5}
      ]
    };
    final controller = TransformationController();
    addTearDown(controller.dispose);
    int? reduced;
    for (final scale in [1.0, 5.0, 12.0]) {
      controller.value = Matrix4.diagonal3Values(scale, scale, 1);
      final original = inkWidth(await pixels(tester,
          ReferencePreviewPainter(reference, null, controller: controller),
          scale: scale));
      final small = inkWidth(await pixels(
          tester,
          ReferencePreviewPainter(reference, null,
              controller: controller, handleSizeFactor: .75),
          scale: scale));
      reduced ??= small;
      expect(small, reduced);
      expect(small, lessThan(original));
      expect(
          inkWidth(await pixels(
              tester,
              ReferencePreviewPainter(reference, null,
                  hiddenLayers: const {'tree'}),
              scale: scale)),
          0);
    }
  });

  test(
      'PDF honours frozen layer choices, validates hidden data and retains measurements',
      () {
    final photo = im.Image(width: 100, height: 100);
    im.fill(photo, color: im.ColorRgb8(200, 200, 200));
    final hidden = {'reference', 'outline'};
    final snapshot = <String, dynamic>{
      'report': {'height_m': 4, 'crown_width_m': 2},
      'reference': {
        'coordinates': 'normalized_oriented_image',
        'width': 100,
        'height': 100,
        'length_m': 1,
        'reference': [
          {'x': .2, 'y': .2},
          {'x': .2, 'y': .8}
        ],
        'outline': [
          {'x': .1, 'y': .1},
          {'x': .3, 'y': .1},
          {'x': .3, 'y': .9}
        ],
        'tree': [
          {'x': .6, 'y': .2},
          {'x': .6, 'y': .8}
        ],
      },
    };
    final before = jsonEncode(snapshot);
    final data = ReportExportData(
        snapshot: snapshot,
        record: {},
        photo: Uint8List.fromList(im.encodePng(photo)),
        local: true,
        hiddenLayers: hidden);
    hidden.clear();
    expect(data.hiddenLayers, {'reference', 'outline'});
    final images = prepareReportImages(data);
    final overlay = im.decodePng(images.annotated!)!;
    expect(overlay.getPixel(20, 50).r, 200);
    expect(overlay.getPixel(20, 50).b, 200);
    expect(overlay.getPixel(59, 50).r, 196);
    expect(overlay.getPixel(59, 50).g, 43);
    expect(overlay.getPixel(59, 15).r, 200,
        reason: 'The previous radius-six filled mark extended this far.');
    expect(data.metrics.first.value, 4);
    expect(jsonEncode(data.snapshot), before);
    expect(images.legend.join(' '), contains('Скрыты на фото'));
    snapshot['reference']['reference'] = [
      {'x': 2, 'y': .2}
    ];
    expect(
        () => prepareReportImages(ReportExportData(
            snapshot: snapshot,
            record: {},
            photo: data.photo,
            local: true,
            hiddenLayers: const {'reference'})),
        throwsFormatException);
  });

  test('generate a control PDF with independently hidden reference layers',
      () async {
    final fixture = fixtures.fixture('layers-selected');
    final data = ReportExportData(
        snapshot: fixture.snapshot,
        record: fixture.record,
        photo: fixture.photo,
        local: true,
        hiddenLayers: const {'reference', 'outline', 'trunk_axis'});
    final dir = Directory('../output/combined-reference/layers-31')
      ..createSync(recursive: true);
    final images = prepareReportImages(data);
    await File('${dir.path}/selected-overlay.png')
        .writeAsBytes(images.annotated!);
    await File('${dir.path}/all-overlay.png')
        .writeAsBytes(prepareReportImages(fixture).annotated!);
    await File('${dir.path}/selected.pdf').writeAsBytes(await buildReportPdf(
        data,
        generatedAt: DateTime.utc(2026, 10, 13, 16)));
    expect(data.snapshot, fixture.snapshot);
    expect(data.metrics.first.value, fixture.metrics.first.value);
  });

  test(
      'hiding a mask removes both PDF overlays but does not bypass mask binding',
      () {
    final fixture = combined.combinedFixture();
    ReportExportData selected(ReportExportData source) => ReportExportData(
        snapshot: source.snapshot,
        record: source.record,
        photo: source.photo,
        local: true,
        hiddenLayers: const {'mask'});
    final data = selected(fixture);
    final images = prepareReportImages(data);
    final overlay = im.decodePng(images.annotated!)!;
    expect(overlay.getPixel(808, 450).g, 200,
        reason: 'The mask is hidden outside the measurement line.');
    expect(overlay.getPixel(800, 450).g, 43,
        reason: 'The tree height remains separately visible.');
    expect(overlay.getPixel(700, 180).g, 125,
        reason: 'Crown width remains visible when only the mask is hidden.');
    expect(images.contour, isNull);
    expect(images.legend.join(' '), contains('Скрыты на фото: Контур дерева'));
    expect(data.metrics.first.value, 4);
    expect(
        () => prepareReportImages(
            selected(combined.combinedFixture(wrongMask: true))),
        throwsFormatException);
  });
}
