import 'dart:convert';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:arborscan_app/contour_editor_state.dart';
import 'package:arborscan_app/mask_drawing_page.dart';
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;

final _photo = img.encodePng(img.Image(width: 100, height: 100));

Finder get _canvas => find.byKey(const ValueKey('mask-image-canvas'));
Finder get _paint =>
    find.descendant(of: _canvas, matching: find.byType(CustomPaint));

Future<void> _open(WidgetTester tester, List<Offset> points,
    {ValueChanged<Map<String, dynamic>>? onDraftChanged}) async {
  await tester.binding.setSurfaceSize(const Size(600, 900));
  addTearDown(() => tester.binding.setSurfaceSize(null));
  await tester.pumpWidget(MaterialApp(
    home: MaskDrawingPage(
      originalImageBase64: base64Encode(_photo),
      editorState: ContourEditorState(
          width: 100, height: 100, closed: false, points: points),
      onDraftChanged: onDraftChanged,
    ),
  ));
  await tester.runAsync(() async {
    await Future<void>.delayed(const Duration(milliseconds: 100));
  });
  await tester.pumpAndSettle();
  expect(_paint, findsOneWidget);
}

TransformationController _controller(WidgetTester tester) => tester
    .widget<InteractiveViewer>(find.byType(InteractiveViewer))
    .transformationController!;

void _zoom(WidgetTester tester, double scale) {
  final size = tester.widget<CustomPaint>(_paint).size;
  // Keep the image centre in the viewport while zooming.
  _controller(tester).value = Matrix4.identity()
    ..translateByDouble(
        size.width / 2 * (1 - scale), size.height / 2 * (1 - scale), 0, 1)
    ..scaleByDouble(scale, scale, 1, 1);
}

Future<ByteData> _renderPoint(WidgetTester tester, double scale) async {
  final paint = tester.widget<CustomPaint>(_paint);
  final recorder = ui.PictureRecorder();
  final canvas = Canvas(recorder)
    ..translate(
        32 - paint.size.width / 2 * scale, 32 - paint.size.height / 2 * scale)
    ..scale(scale);
  paint.painter!.paint(canvas, paint.size);
  final picture = recorder.endRecording();
  final data = await tester.runAsync(() async {
    final image = await picture.toImage(64, 64);
    final pixels = await image.toByteData(format: ui.ImageByteFormat.rawRgba);
    image.dispose();
    return pixels!;
  });
  picture.dispose();
  return data!;
}

Rect _colouredBounds(ByteData data) {
  var minX = 64;
  var minY = 64;
  var maxX = -1;
  var maxY = -1;
  for (var y = 0; y < 64; y++) {
    for (var x = 0; x < 64; x++) {
      final offset = (y * 64 + x) * 4;
      // Yellow and cyan handle pixels, excluding the dark background/outline.
      if (data.getUint8(offset + 1) > 90 &&
          (data.getUint8(offset) > 90 || data.getUint8(offset + 2) > 90)) {
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
      }
    }
  }
  return Rect.fromLTRB(minX.toDouble(), minY.toDouble(), (maxX + 1).toDouble(),
      (maxY + 1).toDouble());
}

void main() {
  testWidgets('rings remain compact and hollow at every photo zoom',
      (tester) async {
    await _open(tester, const [Offset(.5, .5)]);
    Rect? firstBounds;
    for (final scale in [1.0, 2.0, 5.0, 10.0]) {
      _zoom(tester, scale);
      if (scale > 1) {
        // A controller update repaints the handle even without rebuilding page.
        expect(tester.renderObject<RenderCustomPaint>(_paint).debugNeedsPaint,
            isTrue);
      }
      await tester.pump();
      final pixels = await _renderPoint(tester, scale);
      final bounds = _colouredBounds(pixels);
      firstBounds ??= bounds;
      expect(bounds, firstBounds);
      expect(bounds.width, lessThanOrEqualTo(10));
      expect(bounds.height, lessThanOrEqualTo(10));
      const centre = (32 * 64 + 32) * 4;
      expect(pixels.getUint8(centre), 0);
      expect(pixels.getUint8(centre + 1), 0);
      expect(pixels.getUint8(centre + 2), 0);
    }
  });

  testWidgets('active ring stays hollow and hiding marks does not change draft',
      (tester) async {
    final drafts = <Map<String, dynamic>>[];
    await _open(tester, const [Offset(.5, .5)], onDraftChanged: drafts.add);
    final centre = tester.getCenter(_canvas);
    final gesture = await tester.startGesture(centre);
    await tester.pump();
    final active = await _renderPoint(tester, 1);
    expect(_colouredBounds(active).width, lessThanOrEqualTo(12));
    expect(active.getUint8((32 * 64 + 32) * 4 + 1), 0);
    await gesture.up();
    await tester.pump();
    final count = drafts.length;
    await tester.tap(find.byTooltip('Скрыть точки'));
    await tester.pump();
    final hidden = await _renderPoint(tester, 1);
    expect(_colouredBounds(hidden).isEmpty, isTrue);
    expect(drafts.length, count);
    await tester.tap(find.byTooltip('Показать точки'));
    await tester.pump();
    expect(_colouredBounds(await _renderPoint(tester, 1)).isEmpty, isFalse);
    expect(drafts.length, count);
  });

  for (final scale in [1.0, 5.0]) {
    testWidgets('small handles keep 14px touch targets at zoom $scale',
        (tester) async {
      Map<String, dynamic>? draft;
      await _open(tester, const [Offset(.45, .5), Offset(.475, .5)],
          onDraftChanged: (value) => draft = value);
      _zoom(tester, scale);
      await tester.pump();
      final size = tester.widget<CustomPaint>(_paint).size;
      final matrix = _controller(tester).value;
      final scene = Offset(.475 * size.width, .5 * size.height);
      final point = tester.getTopLeft(_canvas) +
          MatrixUtils.transformPoint(matrix, scene);
      // Well outside the visible ~3px ring but inside the unchanged 14px target.
      final gesture = await tester.startGesture(point + const Offset(0, 12));
      await gesture.moveBy(const Offset(20, 0));
      await gesture.up();
      await tester.pump();
      final points = (draft!['points'] as List).cast<Map<String, dynamic>>();
      expect(points, hasLength(2));
      expect(points.first['x'], closeTo(.45, 1e-8));
      expect(points.first['y'], closeTo(.5, 1e-8));
      expect(points.last['x'], closeTo(.475 + 20 / scale / size.width, 1e-8));
      expect(points.last['y'], closeTo(.5 + 12 / scale / size.height, 1e-8));
    });
  }

  testWidgets('overlapping touch regions select the nearest thin-trunk point',
      (tester) async {
    Map<String, dynamic>? draft;
    await _open(tester, const [Offset(.48, .5), Offset(.5, .5)],
        onDraftChanged: (value) => draft = value);
    final point = tester.getCenter(_canvas);
    // The preceding point is also within 14px, but is further away.
    final gesture = await tester.startGesture(point + const Offset(-2, 0));
    await gesture.moveBy(const Offset(0, 25));
    await gesture.up();
    await tester.pump();
    final points = (draft!['points'] as List).cast<Map<String, dynamic>>();
    expect(points.first, {'x': .48, 'y': .5});
    expect(points.last['y'], greaterThan(.5));
  });

  testWidgets('tall-photo preview keeps both actions visible at large font',
      (tester) async {
    await tester.binding.setSurfaceSize(const Size(360, 640));
    addTearDown(() => tester.binding.setSurfaceSize(null));
    final photo = img.encodePng(img.Image(width: 100, height: 200));
    bool? result;
    await tester.pumpWidget(MaterialApp(
      builder: (context, child) => MediaQuery(
          data: MediaQuery.of(context)
              .copyWith(textScaler: const TextScaler.linear(2)),
          child: child!),
      home: Builder(
          builder: (context) => Scaffold(
                body: FilledButton(
                  onPressed: () async {
                    result = await showDialog<bool>(
                      context: context,
                      builder: (dialogContext) => MaskPreviewDialog(
                        previewBytes: photo,
                        onEdit: () => Navigator.pop(dialogContext, false),
                        onConfirm: () => Navigator.pop(dialogContext, true),
                      ),
                    );
                  },
                  child: const Text('Открыть'),
                ),
              )),
    ));
    for (final shouldConfirm in [false, true]) {
      await tester.tap(find.text('Открыть'));
      await tester.pump();
      await tester.runAsync(() async {
        await Future<void>.delayed(const Duration(milliseconds: 100));
      });
      await tester.pumpAndSettle();
      final confirm = find.widgetWithText(FilledButton, 'Подтвердить');
      final edit = find.widgetWithText(OutlinedButton, 'Исправить');
      expect(confirm, findsOneWidget);
      expect(edit, findsOneWidget);
      expect(tester.getRect(confirm).bottom, lessThan(640));
      expect(tester.getRect(edit).top, greaterThan(0));
      expect(tester.takeException(), isNull);
      // Neither action requires scrolling the tall photograph.
      await tester.tap(shouldConfirm ? confirm : edit);
      await tester.pumpAndSettle();
      expect(result, shouldConfirm);
      expect(find.byType(MaskPreviewDialog), findsNothing);
    }
  });
}
