import 'dart:convert';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:arborscan_app/binary_mask_overlay.dart';
import 'package:arborscan_app/mask_drawing_page.dart';
import 'package:arborscan_app/corrections_service.dart';
import 'package:arborscan_app/saved_corrections_page.dart';
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:image/image.dart' as im;
import 'package:shared_preferences/shared_preferences.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  Uint8List maskBytes() {
    // An actual PNG, including white RGB under fully transparent alpha.
    final png = im.Image(width: 4, height: 1, numChannels: 4);
    png.setPixelRgba(0, 0, 0, 0, 0, 255);
    png.setPixelRgba(1, 0, 255, 255, 255, 255);
    png.setPixelRgba(2, 0, 255, 255, 255, 0);
    png.setPixelRgba(3, 0, 255, 255, 255, 128);
    return Uint8List.fromList(im.encodePng(png));
  }

  Uint8List rgbMaskBytes() {
    final png = im.Image(width: 4, height: 1, numChannels: 3);
    png.setPixelRgb(0, 0, 0, 0, 0);
    png.setPixelRgb(1, 0, 255, 255, 255);
    png.setPixelRgb(2, 0, 0, 0, 0);
    png.setPixelRgb(3, 0, 128, 128, 128);
    return Uint8List.fromList(im.encodePng(png));
  }

  Future<ui.Image> maskImage(Uint8List? bytes) async {
    final codec = await ui.instantiateImageCodec(bytes ?? maskBytes());
    final frame = await codec.getNextFrame();
    codec.dispose();
    return frame.image;
  }

  Future<List<List<int>>> renderedPixels(ui.Color tint,
      {bool rgb = false}) async {
    final mask = await maskImage(rgb ? rgbMaskBytes() : null);
    final recorder = ui.PictureRecorder();
    final canvas = ui.Canvas(recorder);
    const target = ui.Rect.fromLTWH(0, 0, 128, 32);
    canvas.drawRect(target, ui.Paint()..color = const ui.Color(0xffcc6633));
    drawBinaryMaskOverlay(
        canvas, mask, const ui.Rect.fromLTWH(0, 0, 4, 1), target, tint);
    final picture = recorder.endRecording();
    final image = await picture.toImage(128, 32);
    final bytes = await image.toByteData(format: ui.ImageByteFormat.rawRgba);
    final rgba = bytes!.buffer.asUint8List();
    final pixels = [16, 48, 80, 112]
        .map((x) => rgba.sublist((16 * 128 + x) * 4, (16 * 128 + x) * 4 + 4))
        .toList();
    image.dispose();
    picture.dispose();
    mask.dispose();
    return pixels;
  }

  void expectPixel(List<int> actual, List<int> expected) {
    for (var i = 0; i < 4; i++) {
      expect(actual[i], closeTo(expected[i], 2), reason: 'RGBA channel $i');
    }
  }

  test('opaque black/white PNG reveals original outside and blends blue inside',
      () async {
    final pixels = await renderedPixels(
        const ui.Color(0xff0000ff).withValues(alpha: .35),
        rgb: true);
    expectPixel(pixels[0], [204, 102, 51, 255]);
    expectPixel(pixels[1], [133, 66, 122, 255]);
    expectPixel(pixels[2], [204, 102, 51, 255]);
    expectPixel(pixels[3], [168, 84, 87, 255]);
  });

  test('green overlay respects mask alpha and keeps original photo visible',
      () async {
    final pixels =
        await renderedPixels(const ui.Color(0xff00ff00).withValues(alpha: .25));
    expectPixel(pixels[0], [204, 102, 51, 255]);
    expectPixel(pixels[1], [153, 140, 38, 255]);
    expectPixel(pixels[2], [204, 102, 51, 255]);
    expectPixel(pixels[3], [178, 121, 45, 255]);
  });

  testWidgets('editor and its PNG preview show photo through mask background',
      (tester) async {
    final original = im.Image(width: 128, height: 128);
    im.fill(original, color: im.ColorRgb8(204, 102, 51));
    await tester.pumpWidget(MaterialApp(
        home: MaskDrawingPage(
      originalImageBase64: base64Encode(im.encodePng(original)),
      initialMaskBase64: base64Encode(maskBytes()),
      initialPoints: const [Offset(.05, .8), Offset(.15, .8), Offset(.1, .95)],
    )));
    await tester.runAsync(
        () => Future<void>.delayed(const Duration(milliseconds: 100)));
    await tester.pumpAndSettle();
    final paint = tester.widget<CustomPaint>(find.descendant(
        of: find.byType(InteractiveViewer),
        matching: find.byType(CustomPaint)));
    await tester.runAsync(() async {
      final recorder = ui.PictureRecorder();
      paint.painter!.paint(ui.Canvas(recorder), const Size(128, 128));
      final picture = recorder.endRecording();
      final image = await picture.toImage(128, 128);
      final bytes = await image.toByteData(format: ui.ImageByteFormat.rawRgba);
      final rgba = bytes!.buffer.asUint8List();
      List<int> pixel(int x) =>
          rgba.sublist((10 * 128 + x) * 4, (10 * 128 + x) * 4 + 4);
      expectPixel(pixel(16), [204, 102, 51, 255]);
      expectPixel(pixel(48), [144, 119, 118, 255]);
      expectPixel(pixel(80), [204, 102, 51, 255]);
      image.dispose();
      picture.dispose();
    });

    await tester.tap(find.byTooltip('Быстрый превью'));
    await tester.runAsync(
        () => Future<void>.delayed(const Duration(milliseconds: 100)));
    await tester.pumpAndSettle();
    final preview = tester.widget<Image>(find.descendant(
        of: find.byType(AlertDialog), matching: find.byType(Image)));
    final png = im.decodePng((preview.image as MemoryImage).bytes)!;
    List<int> previewPixel(int x) {
      final pixel = png.getPixel(x, 10);
      return [
        pixel.r.toInt(),
        pixel.g.toInt(),
        pixel.b.toInt(),
        pixel.a.toInt()
      ];
    }

    expectPixel(previewPixel(37), [204, 102, 51, 255]);
    expectPixel(previewPixel(112), [144, 119, 118, 255]);
    expectPixel(previewPixel(187), [204, 102, 51, 255]);
  });

  testWidgets('saved contour overlay blends foreground without dimming photo',
      (tester) async {
    SharedPreferences.setMockInitialValues({'arborscan_auth_token': 'alice'});
    final original = im.Image(width: 128, height: 32);
    im.fill(original, color: im.ColorRgb8(204, 102, 51));
    final service = CorrectionsService(
        clientFactory: () => MockClient((_) async => http.Response(
            jsonEncode({
              'analysis_id': 'demo',
              'original_image_base64': base64Encode(im.encodePng(original)),
              'mask_png_base64': base64Encode(rgbMaskBytes()),
              'review_status': 'pending_review'
            }),
            200)));
    final boundaryKey = GlobalKey();
    await tester.pumpWidget(RepaintBoundary(
        key: boundaryKey,
        child: MaterialApp(
            home: SavedCorrectionsPage(
                service: service,
                correctionId: 'demo',
                sessionToken: 'alice'))));
    await tester.runAsync(
        () => Future<void>.delayed(const Duration(milliseconds: 100)));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Наложение'));
    await tester.pump();
    await tester.runAsync(
        () => Future<void>.delayed(const Duration(milliseconds: 100)));
    await tester.pumpAndSettle();
    final mask = find.byType(BinaryMaskOverlay);
    expect(mask, findsOneWidget);
    expect(find.descendant(of: mask, matching: find.byType(CustomPaint)),
        findsOneWidget);
    final rect = tester.getRect(mask);
    await tester.runAsync(() async {
      final boundary = boundaryKey.currentContext!.findRenderObject()
          as RenderRepaintBoundary;
      final image = await boundary.toImage();
      final bytes = await image.toByteData(format: ui.ImageByteFormat.rawRgba);
      final rgba = bytes!.buffer.asUint8List();
      List<int> pixel(double fraction) {
        final x = (rect.left + rect.width * fraction).round();
        final y = rect.center.dy.round();
        final offset = (y * image.width + x) * 4;
        return rgba.sublist(offset, offset + 4);
      }

      expectPixel(pixel(.125), [204, 102, 51, 255]);
      expectPixel(pixel(.375), [144, 119, 118, 255]);
      expectPixel(pixel(.625), [204, 102, 51, 255]);
      expectPixel(pixel(.875), [174, 110, 85, 255]);
      image.dispose();
    });
  });
}
