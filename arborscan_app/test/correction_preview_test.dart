import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'package:arborscan_app/corrections_service.dart';
import 'package:arborscan_app/saved_corrections_page.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:image/image.dart' as image;
import 'package:shared_preferences/shared_preferences.dart';

Uint8List _jpeg({int width = 16, int height = 24}) => Uint8List.fromList(
    image.encodeJpg(image.Image(width: width, height: height)));
Map<String, dynamic> _response(String id,
        {Uint8List? bytes, int width = 16, int height = 24}) =>
    {
      'correction_id': id,
      'content_type': 'image/jpeg',
      'width': width,
      'height': height,
      'thumbnail_base64': base64Encode(bytes ?? _jpeg()),
    };

class _PreviewService extends CorrectionsService {
  int active = 0, maxActive = 0, fullDetails = 0;
  final calls = <String>[];
  final pending = <Completer<Uint8List?>>[];
  @override
  Future<Map<String, dynamic>> list(String token, int offset) async => {
        'items': List.generate(40, (i) => {'correction_id': 'demo-$i'}),
        'next_offset': null,
      };
  @override
  Future<Uint8List?> preview(String token, String id, {String? ownerId}) async {
    calls.add(id);
    active++;
    if (active > maxActive) maxActive = active;
    final done = Completer<Uint8List?>();
    pending.add(done);
    try {
      return await done.future;
    } finally {
      active--;
    }
  }

  @override
  Future<Map<String, dynamic>> record(String token, String id,
      {String? ownerId}) async {
    fullDetails++;
    throw StateError('A list must never request full correction records');
  }
}

void main() {
  setUp(() => SharedPreferences.setMockInitialValues({
        'arborscan_auth_token': 'demo-preview-token',
        'arborscan_user_id': 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa',
      }));

  test('owner/admin preview is a bounded separate authenticated request',
      () async {
    final urls = <Uri>[];
    final service = CorrectionsService(
        clientFactory: () => MockClient((r) async {
              urls.add(r.url);
              expect(r.headers['Authorization'], 'Bearer demo-preview-token');
              return http.Response(jsonEncode(_response('demo-id')), 200);
            }));
    expect(await service.preview('demo-preview-token', 'demo-id'), isNotEmpty);
    expect(
        await service.preview('demo-preview-token', 'demo-id',
            ownerId: 'demo-owner'),
        isNotEmpty);
    expect(urls[0].path, endsWith('/v4/corrections/demo-id/preview'));
    expect(
        urls[1].path, endsWith('/workflow/review/demo-owner/demo-id/preview'));
    expect(urls.every((u) => u.queryParameters.isEmpty), isTrue);
  });

  for (final status in [404, 405]) {
    test('old API $status leaves icon, with no full-photo fallback', () async {
      var requests = 0;
      final service = CorrectionsService(
          clientFactory: () => MockClient((r) async {
                requests++;
                expect(r.url.path, endsWith('/preview'));
                return http.Response('', status);
              }));
      expect(await service.preview('demo-preview-token', 'demo-id'), isNull);
      expect(requests, 1);
    });
  }

  test('authorization and network errors never become preview bytes', () async {
    for (final status in [401, 403, 503]) {
      final service = CorrectionsService(clientFactory: () => MockClient(
          (_) async => http.Response('', status)));
      await expectLater(service.preview('demo-preview-token', 'demo-id'),
          throwsA(isA<CorrectionException>()));
    }
    final offline = CorrectionsService(clientFactory: () => MockClient(
        (_) async => throw http.ClientException('offline')));
    await expectLater(offline.preview('demo-preview-token', 'demo-id'),
        throwsA(isA<CorrectionException>()));
    final changed = CorrectionsService(clientFactory: () => MockClient((_) async {
      await (await SharedPreferences.getInstance())
          .setString('arborscan_auth_token', 'demo-other-token');
      return http.Response(jsonEncode(_response('demo-id')), 200);
    }));
    await expectLater(changed.preview('demo-preview-token', 'demo-id'),
        throwsA(isA<CorrectionException>()));
  });

  test(
      'oversized body, wrong revision, and oversized decoded dimensions are rejected',
      () async {
    for (final body in [
      'x' * (100 * 1024),
      jsonEncode(_response('wrong-id')),
      jsonEncode(_response('demo-id', bytes: _jpeg(width: 200), width: 16)),
    ]) {
      final service = CorrectionsService(
          clientFactory: () =>
              MockClient((r) async => http.Response(body, 200)));
      await expectLater(service.preview('demo-preview-token', 'demo-id'),
          throwsA(isA<CorrectionException>()));
    }
  });

  testWidgets('visible previews are lazy, two-at-a-time and cache is bounded',
      (t) async {
    t.view.physicalSize = const Size(400, 600);
    t.view.devicePixelRatio = 1;
    addTearDown(t.view.resetPhysicalSize);
    addTearDown(t.view.resetDevicePixelRatio);
    final service = _PreviewService();
    await t
        .pumpWidget(MaterialApp(home: SavedCorrectionsPage(service: service)));
    await t.pumpAndSettle();
    expect(service.active, 2);
    Future<void> finishVisible() async {
      for (var i = 0; i < 15; i++) {
        for (final job
            in service.pending.where((job) => !job.isCompleted).toList()) {
          job.complete(_jpeg());
        }
        await t.pumpAndSettle();
        if (service.active == 0) return;
      }
      fail('Visible previews did not finish');
    }

    await finishVisible();
    expect(service.calls.length, lessThan(40));
    expect(service.fullDetails, 0);
    expect(service.maxActive, 2);
    final initialFirstCalls =
        service.calls.where((id) => id == 'demo-0').length;
    await t.drag(find.byType(Scrollable).first, const Offset(0, -1800));
    await t.pumpAndSettle();
    await finishVisible();
    await t.drag(find.byType(Scrollable).first, const Offset(0, 2400));
    await t.pumpAndSettle();
    await finishVisible();
    expect(service.calls.where((id) => id == 'demo-0').length,
        greaterThan(initialFirstCalls));
    expect(service.maxActive, 2);
    expect(service.fullDetails, 0);
    expect(t.takeException(), isNull);
  });

  testWidgets('account change drops queued and late previews', (t) async {
    final service = _PreviewService();
    await t
        .pumpWidget(MaterialApp(home: SavedCorrectionsPage(service: service)));
    await t.pumpAndSettle();
    final count = service.calls.length;
    await (await SharedPreferences.getInstance())
        .setString('arborscan_auth_token', 'demo-other-token');
    CorrectionsService.authChanges.value++;
    for (final job in service.pending) {
      if (!job.isCompleted) job.complete(_jpeg());
    }
    await t.pumpAndSettle();
    expect(service.calls.length, count);
    expect(find.byType(Image), findsNothing);
    expect(find.textContaining('Сессия изменилась'), findsOneWidget);
    expect(t.takeException(), isNull);
  });

  testWidgets('legacy feedback proposals remain explicitly unverified in review', (t) async {
    final service = CorrectionsService(clientFactory: () => MockClient((r) async => http.Response(
      jsonEncode({
        'analysis_id': 'demo-analysis', 'review_status': 'submitted',
        'original_image_base64': base64Encode(_jpeg()),
        'mask_png_base64': base64Encode(image.encodePng(image.Image(width: 16, height: 24))),
        'feedback_proposal': {
          'version': 1, 'kind': 'legacy_user_feedback', 'training_consent': true,
          'quality': {'params_ok': false},
          'corrections': {'species': 'DEMO manual', 'height_m': 50.0, 'scale_px_to_m': .01},
        },
      }), 200)));
    await t.pumpWidget(MaterialApp(home: SavedCorrectionsPage(
      service: service, correctionId: 'demo-id', reviewOwner: 'demo-owner',
    )));
    await t.pumpAndSettle();
    await t.ensureVisible(find.text('Предложения автора прежней обратной связи'));
    await t.tap(find.text('Предложения автора прежней обратной связи'));
    await t.pumpAndSettle();
    expect(find.text('Не проверены; измерения отчёта не изменены'), findsOneWidget);
    expect(find.text('Предложенная высота (м): 50.0'), findsOneWidget);
    expect(find.textContaining('система пикселей не подтверждена'), findsOneWidget);
    expect(find.textContaining('Допуск к набору и запуск обучения выполняются отдельно'), findsOneWidget);
    expect(t.takeException(), isNull);
  });
}
