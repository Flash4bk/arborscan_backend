import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:crypto/crypto.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:arborscan_app/report_export_loader.dart';
import 'package:arborscan_app/corrections_service.dart';
import 'package:image/image.dart' as im;
import 'package:arborscan_app/report_export_data.dart';
import 'package:arborscan_app/report_pdf.dart';

// Deliberately synthetic. This verifies rendering and coordinate binding,
// not segmentation or measurements of the user's trees.
ReportExportData combinedFixture(
    {bool wrongMask = false, bool correction = false}) {
  final original = im.Image(width: 1920, height: 1080);
  im.fill(original, color: im.ColorRgb8(200, 200, 200));
  final mask =
      im.Image(width: wrongMask ? 1200 : 1600, height: 900, numChannels: 4);
  im.fill(mask, color: im.ColorRgba8(0, 255, 0, 0));
  im.fillRect(mask,
      x1: 790,
      y1: 100,
      x2: 810,
      y2: 800,
      color: im.ColorRgba8(0, 255, 0, 110),
      alphaBlend: false);
  final png = Uint8List.fromList(im.encodePng(mask));
  return ReportExportData(
      local: true,
      photo: Uint8List.fromList(im.encodePng(original)),
      mask: correction ? png : null,
      record: {
        'analysis_id': 'DEMO-combined',
        'version_id': 'DEMO-v1'
      },
      snapshot: {
        'kind': 'reference',
        'report': {
          'height_m': 4.0,
          'crown_width_m': 1.5,
          'images': {
            'width': 1920,
            'height': 1080,
            'mask_image_base64': base64Encode(png)
          }
        },
        'reference': {
          'coordinates': 'normalized_oriented_image',
          'width': 1920,
          'height': 1080,
          'reference': [
            {'x': .2, 'y': .8},
            {'x': .2, 'y': .6}
          ],
          'tree': [
            {'x': .5, 'y': .8},
            {'x': .5, 'y': .1}
          ],
          'crown': [
            {'x': .4, 'y': .2},
            {'x': .6, 'y': .2}
          ]
        }
      });
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  setUp(() => SharedPreferences.setMockInitialValues({
        'arborscan_auth_token': 'test',
        'arborscan_user_id': 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa',
      }));
  test(
      'saved AI alpha mask and reference lines appear together on the original',
      () async {
    final data = combinedFixture();
    expect(data.modelPreviewMask, isTrue);
    final images = prepareReportImages(data);
    final merged = im.decodePng(images.annotated!)!;
    final contour = im.decodePng(images.contour!)!;
    // Transparent green RGB must not turn the whole image into foreground.
    expect(merged.getPixel(300, 400).r, 200);
    expect(merged.getPixel(300, 400).g, 200);
    expect(contour.getPixel(800, 450).r, 110);
    expect(contour.getPixel(800, 450).g, 220);
    // Independent reference line remains blue over the same coordinate frame.
    expect(merged.getPixel(320, 630).b, greaterThan(150));
    expect(merged.getPixel(320, 630).r, lessThan(30));
    expect(images.legend.join(' '), contains('маска модели'));
    expect(images.legend.join(' '), contains('Эталон'));
    expect(data.metrics.first.value, 4);
    final root = Directory('../output/combined-reference');
    await root.create(recursive: true);
    await File('${root.path}/synthetic-combined-overlay.png')
        .writeAsBytes(images.annotated!);
    final pdf = await buildReportPdf(data);
    expect(ascii.decode(pdf.take(4).toList()), '%PDF');
    await File('${root.path}/DEMO-combined.pdf').writeAsBytes(pdf);
  });
  test(
      'an arbitrary resized mask is rejected; a correction must be original-size',
      () {
    expect(() => prepareReportImages(combinedFixture(wrongMask: true)),
        throwsFormatException);
    expect(() => prepareReportImages(combinedFixture(correction: true)),
        throwsFormatException);
  });
  test('valid model preview uses the server half-pixel round-to-even transform',
      () {
    final original = im.Image(width: 2560, height: 1604);
    final mask = im.Image(width: 1600, height: 1002, numChannels: 4);
    final data = ReportExportData(
        local: true,
        record: {},
        photo: Uint8List.fromList(im.encodePng(original)),
        snapshot: {
          'report': {
            'images': {'mask_image_base64': base64Encode(im.encodePng(mask))}
          }
        });
    expect(prepareReportImages(data).contour, isNotNull);
  });
  test('legacy reference without model has lines and no fabricated mask', () {
    final d = combinedFixture();
    final s = d.snapshot;
    s['report']['images'] = {};
    final legacy = ReportExportData(
        snapshot: s, record: d.record, photo: d.photo, local: true);
    final images = prepareReportImages(legacy);
    expect(images.annotated, isNotNull);
    expect(images.contour, isNull);
    expect(legacy.mask, isNull);
  });
  test(
      'explicit saved correction wins over AI and remains bound to original/version',
      () {
    final ai = combinedFixture();
    final binary = im.Image(width: 1920, height: 1080);
    im.fillRect(binary,
        x1: 940, y1: 200, x2: 960, y2: 900, color: im.ColorRgb8(255, 255, 255));
    final snapshot = ai.snapshot;
    snapshot['correction_id'] = 'DEMO-correction-v2';
    snapshot['report']['contour_preview'] = {
      'correction_id': 'DEMO-correction-v2',
      'photo_sha256': sha256.convert(ai.photo!).toString(),
      'width': 1920,
      'height': 1080,
      'mask_png_base64': base64Encode(im.encodePng(binary))
    };
    ReportExportData data() => ReportExportData(
        snapshot: snapshot, record: ai.record, photo: ai.photo, local: true);
    expect(data().modelPreviewMask, isFalse);
    final overlay = im.decodePng(prepareReportImages(data()).annotated!)!;
    expect(overlay.getPixel(790, 450).g, 220);
    expect(overlay.getPixel(300, 450).g, 200);
    snapshot['correction_id'] = 'DEMO-other-revision';
    expect(() => prepareReportImages(data()), throwsFormatException);
    snapshot['correction_id'] = 'DEMO-correction-v2';
    snapshot['report']['contour_preview']['photo_sha256'] = 'wrong-original';
    expect(() => prepareReportImages(data()), throwsFormatException);
  });
  test(
      'complete pinned local correction exports offline without accessing a server',
      () async {
    final fixture = combinedFixture();
    final snapshot = fixture.snapshot;
    snapshot['correction_id'] = 'DEMO-pinned';
    snapshot['report']['contour_preview'] = {
      'correction_id': 'DEMO-pinned',
      'photo_sha256': sha256.convert(fixture.photo!).toString(),
      'mask_png_base64':
          base64Encode(im.encodePng(im.Image(width: 1920, height: 1080))),
      'width': 1920,
      'height': 1080
    };
    var requests = 0;
    final auth = CorrectionsService(
        clientFactory: () => MockClient((_) async {
              requests++;
              throw const SocketException('offline');
            }));
    final exported = await loadReportExport(
        snapshot: snapshot,
        record: fixture.record,
        photo: fixture.photo,
        local: true,
        token: 'test',
        service: auth);
    expect(exported.modelPreviewMask, isFalse);
    expect(exported.mask, isNotNull);
    expect(requests, 0);
  });
  test('cache from another revision of the same photo is rejected', () async {
    final folder = await Directory.systemTemp.createTemp('combined-cache');
    try {
      final fixture = combinedFixture();
      var requests = 0;
      final auth = CorrectionsService(
          clientFactory: () => MockClient((_) async {
                requests++;
                return http.Response(
                    jsonEncode({
                      'correction_id': 'DEMO-pinned',
                      'original_image_base64': base64Encode(fixture.photo!),
                      'mask_png_base64': base64Encode(
                          im.encodePng(im.Image(width: 1920, height: 1080)))
                    }),
                    200);
              }));
      Future<ReportExportData> load() => loadReportExport(
          snapshot: {'correction_id': 'DEMO-pinned', 'report': {}},
          record: {},
          photo: fixture.photo,
          local: false,
          token: 'test',
          service: auth,
          cacheDirectory: () async => folder);
      await load();
      final file = (await Directory('${folder.path}/report-export-assets')
              .list()
              .toList())
          .single as File;
      final asset = jsonDecode(await file.readAsString());
      asset['correction_id'] = 'DEMO-other-revision';
      await file.writeAsString(jsonEncode(asset));
      await expectLater(load(), throwsFormatException);
      expect(requests, 1);
    } finally {
      await folder.delete(recursive: true);
    }
  });
}
