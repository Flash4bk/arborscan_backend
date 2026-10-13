import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import 'package:image/image.dart' as im;
import 'package:arborscan_app/combined_reference.dart';
import 'package:arborscan_app/corrections_service.dart';
import 'package:arborscan_app/photo_analysis_service.dart';
import 'package:arborscan_app/reference_preview.dart';

class InspectClient extends http.BaseClient {
  final Future<http.StreamedResponse> Function(http.BaseRequest) callback;
  InspectClient(this.callback);
  @override
  Future<http.StreamedResponse> send(http.BaseRequest request) =>
      callback(request);
}

http.StreamedResponse response(Map<String, dynamic> data, [int status = 200]) =>
    http.StreamedResponse(Stream.value(utf8.encode(jsonEncode(data))), status);

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  setUp(() => SharedPreferences.setMockInitialValues({
        'arborscan_auth_token': 'test',
        'arborscan_user_id': 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa',
      }));
  test(
      'analysis sends exact original, selected original coordinates and HTTPS auth',
      () async {
    final image = Uint8List.fromList([1, 2, 3, 4]);
    final service = PhotoAnalysisService(
        clientFactory: () => InspectClient((request) async {
              final multipart = request as http.MultipartRequest;
              expect(multipart.url.scheme, 'https');
              expect(multipart.url.path, '/api/v4/v4/analyze-tree');
              expect(multipart.headers['Authorization'], 'Bearer test');
              expect(multipart.fields,
                  {'include_images': 'true', 'tap_x': '0.4', 'tap_y': '0.2'});
              expect(multipart.files.single.field, 'file');
              expect(await multipart.files.single.finalize().toBytes(),
                  [1, 2, 3, 4]);
              return response({
                'tree': {'detected': true}
              });
            }));
    final pending =
        service.analyze(token: 'test', image: image, tap: const Offset(.4, .2));
    image[0] = 9;
    expect((await pending)['tree']['detected'], true);
  });
  test(
      'not found is an honest result; server errors and malformed responses are errors',
      () async {
    final original = Uint8List.fromList([1]);
    final notFound = PhotoAnalysisService(
        clientFactory: () => InspectClient((_) async => response({
              'tree': {'detected': false}
            })));
    expect(
        (await notFound.analyze(token: 'test', image: original))['tree']
            ['detected'],
        false);
    for (final status in [401, 403, 500, 503]) {
      final service = PhotoAnalysisService(
          clientFactory: () =>
              InspectClient((_) async => response({}, status)));
      await expectLater(
          service.analyze(token: 'test', image: original),
          throwsA(isA<CorrectionException>()
              .having((e) => e.statusCode, 'status', status)));
    }
    final malformed = PhotoAnalysisService(
        clientFactory: () => InspectClient((_) async => http.StreamedResponse(
            Stream.value(utf8.encode('Internal Server Error')), 200)));
    await expectLater(
        malformed.analyze(token: 'test', image: original),
        throwsA(isA<CorrectionException>()
            .having((e) => e.message, 'message', contains('некорректный'))));
    final offline = PhotoAnalysisService(
        clientFactory: () =>
            InspectClient((_) async => throw const SocketException('offline')));
    await expectLater(
        offline.analyze(token: 'test', image: original),
        throwsA(isA<CorrectionException>()
            .having((e) => e.message, 'message', contains('Нет связи'))));
  });
  test(
      'account switch rejects in-flight response without leaking previous account result',
      () async {
    final sent = Completer<void>(), result = Completer<http.StreamedResponse>();
    final service = PhotoAnalysisService(
        clientFactory: () => InspectClient((_) {
              sent.complete();
              return result.future;
            }));
    final pending =
        service.analyze(token: 'test', image: Uint8List.fromList([1]));
    final assertion = expectLater(pending, throwsA(isA<CorrectionException>()));
    await sent.future;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('arborscan_auth_token', 'second');
    CorrectionsService.authChanges.value++;
    result.complete(response({
      'tree': {'detected': true}
    }));
    await assertion;
  });
  test('out of original coordinate bounds never reaches model', () async {
    var calls = 0;
    final service = PhotoAnalysisService(
        clientFactory: () => InspectClient((_) async {
              calls++;
              return response({
                'tree': {'detected': true}
              });
            }));
    await expectLater(
        service.analyze(
            token: 'test',
            image: Uint8List.fromList([1]),
            tap: const Offset(1.01, .5)),
        throwsA(isA<CorrectionException>()));
    expect(calls, 0);
  });
  test('nearest neighbour is not silently accepted as selected tree', () async {
    final service = PhotoAnalysisService(
        clientFactory: () => InspectClient((_) async => response({
              'tree': {'detected': true},
              'warnings': ['tap_did_not_hit_mask_nearest_tree_selected']
            })));
    await expectLater(
        service.analyze(
            token: 'test',
            image: Uint8List.fromList([1]),
            tap: const Offset(.4, .2)),
        throwsA(isA<CorrectionException>().having(
            (e) => e.message, 'message', contains('предложила соседнее'))));
  });
  test(
      'combined snapshot preserves model measurements, AR, exact revision and reference source',
      () {
    final base = {
      'report': {
        'measurements': {
          'height': {'value_px': 123, 'value_m': null, 'source': 'vision'}
        },
        'species': {'display_name': 'Pinus'},
        'geometry': {
          'crown_height': {
            'value': null,
            'unit': 'm',
            'source': 'unavailable',
            'reason': 'whole_tree_mask_has_no_live_crown_base'
          }
        },
        'segmentation_model_version': 'sha-exact'
      },
      'ar': {'photo_sha256': 'bound'},
      'correction_id': 'exact-revision'
    };
    final snapshot = combinedReferenceSnapshot(
        base: base,
        reference: {'method': 'known_object_segment_v2'},
        manualReport: {
          'height_m': 4.0,
          'geometry': {
            'tree_height': {'source': 'reference'}
          }
        },
        capturedAt: '2026-10-13T00:00:00Z',
        environment: null);
    expect(snapshot['kind'], 'reference');
    expect(snapshot['report']['measurements'],
        (base['report'] as Map)['measurements']);
    expect(snapshot['report']['species'], (base['report'] as Map)['species']);
    expect(snapshot['report']['model_geometry'],
        (base['report'] as Map)['geometry']);
    expect(
        snapshot['report']['geometry']['tree_height']['source'], 'reference');
    final sources = modelReportForSources(snapshot);
    expect(sources['geometry'], (base['report'] as Map)['geometry']);
    expect(sources['measurements'], (base['report'] as Map)['measurements']);
    expect(sources.containsKey('height_m'), false);
    final revised = combinedReferenceSnapshot(
        base: snapshot,
        reference: snapshot['reference'],
        manualReport: {
          'height_m': 5,
          'geometry': {
            'tree_height': {'source': 'reference', 'value': 5}
          }
        },
        capturedAt: '2026-10-13T00:00:00Z',
        environment: null);
    expect(revised['report']['model_geometry'],
        (base['report'] as Map)['geometry']);
    expect(snapshot['ar'], base['ar']);
    expect(snapshot['correction_id'], 'exact-revision');
    expect(snapshot['report']['height_m'], 4);
    expect((base['report'] as Map).containsKey('height_m'), false);
  });
  test('legacy combined manual geometry cannot become model geometry', () {
    final legacy = {
      'kind': 'reference',
      'reference': {'version': 2},
      'report': {
        'method': 'known_object_segment_v2',
        'height_m': 4,
        'measurements': {
          'height': {'value_px': 123, 'source': 'vision'}
        },
        'geometry': {
          'tree_height': {'value': 4, 'source': 'reference'}
        }
      }
    };
    final sources = modelReportForSources(legacy);
    expect(sources.containsKey('geometry'), false);
    expect(sources.containsKey('method'), false);
    expect(sources.containsKey('height_m'), false);
    expect(sources['measurements'], (legacy['report'] as Map)['measurements']);
    final revised = combinedReferenceSnapshot(
        base: legacy,
        reference: legacy['reference'] as Map<String, dynamic>,
        manualReport: {
          'geometry': {
            'tree_height': {'source': 'reference'}
          }
        },
        capturedAt: '2026-10-13T00:00:00Z',
        environment: null);
    expect(revised['report'].containsKey('model_geometry'), false);
  });
  test(
      '1920 preview binds original history dimensions without rewriting model values',
      () {
    final original = {
      'images': {
        'width': 1600,
        'height': 900,
        'mask_image_base64': 'exactbytes'
      },
      'measurements': {
        'height': {'value_px': 1050, 'source': 'vision'}
      },
      'analysis_id': 'execution'
    };
    final bound = boundAnalysisReport(original, 1920, 1080);
    expect(bound['images']['width'], 1920);
    expect(bound['images']['height'], 1080);
    expect(bound['images']['preview_width'], 1600);
    expect(bound['images']['preview_height'], 900);
    expect(bound['images']['mask_image_base64'], 'exactbytes');
    expect(bound['measurements'], original['measurements']);
    expect((original['images'] as Map)['width'], 1600);
    expect(boundAnalysisReport(bound, 1920, 1080), bound);
    expect(
        () => boundAnalysisReport(original, 1920, 1200), throwsFormatException);
  });
  test(
      'mask preview respects transparent green outside tree, binary legacy and exact frame',
      () {
    final rgba = im.Image(width: 4, height: 2, numChannels: 4);
    for (final p in rgba) {
      p.setRgba(0, 255, 0, 0);
    }
    rgba.setPixelRgba(2, 1, 0, 255, 0, 110);
    final overlay = im.decodePng(
        treeMaskOverlay(Uint8List.fromList(im.encodePng(rgba)), 4, 2)!)!;
    expect(overlay.getPixel(0, 0).a, 0);
    expect(overlay.getPixel(2, 1).a, 110);
    final binary = im.Image(width: 4, height: 2, numChannels: 3)
      ..setPixelRgb(1, 1, 255, 255, 255);
    final old = im.decodePng(
        treeMaskOverlay(Uint8List.fromList(im.encodePng(binary)), 4, 2)!)!;
    expect(old.getPixel(1, 1).a, 110);
    expect(old.getPixel(0, 0).a, 0);
    final opaque = im.Image(width: 4, height: 2, numChannels: 4);
    for (final p in opaque) {
      p.setRgba(0, 0, 0, 255);
    }
    opaque.setPixelRgba(2, 1, 255, 255, 255, 255);
    final corrected = im.decodePng(treeMaskOverlay(
        Uint8List.fromList(im.encodePng(opaque)), 4, 2,
        binary: true)!)!;
    expect(corrected.getPixel(0, 0).a, 0);
    expect(corrected.getPixel(2, 1).a, 110);
    expect(
        treeMaskOverlay(Uint8List.fromList(im.encodePng(rgba)), 4, 3), isNull);
  });
}
