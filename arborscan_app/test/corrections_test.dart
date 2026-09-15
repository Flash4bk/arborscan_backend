import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:arborscan_app/corrections_service.dart';
import 'package:arborscan_app/api_config.dart';
import 'package:arborscan_app/saved_corrections_page.dart';
import 'package:arborscan_app/unified_analysis_report_page.dart';
import 'package:arborscan_app/unified_analysis_models.dart';
import 'package:arborscan_app/mask_drawing_page.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  setUp(() => SharedPreferences.setMockInitialValues({'arborscan_auth_token': 'alice'}));
  final bytes = Uint8List.fromList([1, 2, 3, 255]);
  Future<void> save(CorrectionsService service, {String token = 'alice'}) => service.save(
    token: token, analysisId: 'analysis-id', image: bytes, mask: bytes);

  test('multipart preserves original bytes, fields and bearer header', () async {
    final client = _MultipartClient();
    await save(CorrectionsService(clientFactory: () => client));
    expect(client.request!.url, ApiConfig.v4('/v4/corrections'));
    expect(client.request!.headers['Authorization'], 'Bearer alice');
    expect(client.request!.fields, {'analysis_id': 'analysis-id'});
    expect(client.request!.files.map((f) => f.field), ['image', 'mask']);
    for (final file in client.request!.files) {
      expect(await file.finalize().toBytes(), bytes);
    }
  });

  for (final body in ['{"saved":false}', '{}', '{"saved":"true"}', 'invalid']) {
    test('does not accept unconfirmed response $body', () async {
      final service = CorrectionsService(clientFactory: () => MockClient((_) async => http.Response(body, 200)));
      await expectLater(save(service), throwsA(isA<CorrectionException>()));
    });
  }
  for (final status in [401, 403, 503]) {
    test('HTTP $status allows retry', () async {
      var attempts = 0;
      final service = CorrectionsService(clientFactory: () => MockClient((_) async =>
        ++attempts == 1 ? http.Response('', status) : http.Response('{"saved":true}', 200)));
      await expectLater(save(service), throwsA(isA<CorrectionException>()));
      await save(service);
      expect(attempts, 2);
    });
  }
  test('network loss and timeout are recoverable errors', () async {
    for (final client in [MockClient((_) async => throw http.ClientException('offline')),
      MockClient((_) => Completer<http.Response>().future)]) {
      await expectLater(save(CorrectionsService(clientFactory: () => client,
        timeout: const Duration(milliseconds: 10))), throwsA(isA<CorrectionException>()));
    }
  });
  test('missing token never sends request', () async {
    await expectLater(save(CorrectionsService(clientFactory: () => throw StateError('sent')), token: ''),
      throwsA(isA<CorrectionException>()));
  });
  test('account switch rejects pending response and old token', () async {
    final response = Completer<http.Response>();
    final sent = Completer<void>();
    final service = CorrectionsService(clientFactory: () => MockClient((_) {
      sent.complete(); return response.future;
    }));
    final pending = save(service);
    final assertion = expectLater(pending, throwsA(isA<CorrectionException>()));
    await sent.future;
    await (await SharedPreferences.getInstance()).setString('arborscan_auth_token', 'bob');
    response.complete(http.Response('{"saved":true}', 200));
    await assertion;
    await expectLater(save(service), throwsA(isA<CorrectionException>()));
  });

  testWidgets('pagination, cold reopening from server and account invalidation', (tester) async {
    final offsets = <String?>[];
    var details = 0;
    const png = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=';
    final service = CorrectionsService(clientFactory: () => MockClient((request) async {
      expect(request.headers['Authorization'], 'Bearer alice');
      if (request.url.path.endsWith('/id')) {
        details++;
        return http.Response(jsonEncode({'original_image_base64': png, 'mask_png_base64': png}), 200);
      }
      offsets.add(request.url.queryParameters['offset']);
      return http.Response(jsonEncode({'items': [{'correction_id': 'id'}],
        'next_offset': offsets.length == 1 ? 50 : null}), 200);
    }));
    await tester.pumpWidget(MaterialApp(home: SavedCorrectionsPage(service: service)));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Загрузить ещё'));
    await tester.pumpAndSettle();
    expect(offsets, ['0', '50']);
    await tester.tap(find.text('Сохранённый контур'));
    await tester.pumpAndSettle();
    expect(find.byType(Image), findsNWidgets(2));
    await tester.pumpWidget(const SizedBox());
    await tester.pumpWidget(MaterialApp(home: SavedCorrectionsPage(service: service, correctionId: 'id')));
    await tester.pumpAndSettle();
    expect(details, 2);
    expect(find.byType(Image), findsNWidgets(2));
    CorrectionsService.authChanges.value++;
    await tester.pumpAndSettle();
    expect(find.byType(Image), findsNothing);
    expect(find.textContaining('Сессия изменилась'), findsOneWidget);
  });

  testWidgets('report keeps draft on failure, blocks duplicate send and confirms only saved true', (tester) async {
    final response = Completer<http.Response>();
    var attempts = 0;
    final service = CorrectionsService(clientFactory: () => MockClient((_) async {
      attempts++;
      if (attempts == 1) return response.future;
      return http.Response('{"saved":true}', 200);
    }));
    final original = base64Decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=');
    final result = UnifiedAnalysisResult.fromJson({'analysis_id': 'id',
      'measurements': {'height': {'value_m': 12.5}}});
    await tester.pumpWidget(MaterialApp(home: UnifiedAnalysisReportPage(
      result: result, fallbackImageBytes: original, correctionsService: service)));
    await tester.pumpAndSettle();
    await tester.scrollUntilVisible(find.text('Исправить контур'), 200);
    await tester.tap(find.text('Исправить контур'));
    // Return the editor contract without waiting for the native image codec
    // (the editor itself is outside this save-flow test).
    await tester.pump();
    await tester.pump(const Duration(seconds: 1));
    final editor = tester.element(find.byType(MaskDrawingPage));
    Navigator.of(editor).pop({'mask_png_base64': base64Encode(original),
      'points': [{'x': 0.1, 'y': 0.1}, {'x': 0.8, 'y': 0.1}, {'x': 0.5, 'y': 0.9}], 'closed': true});
    await tester.pumpAndSettle();
    await tester.ensureVisible(find.text('Сохранить контур'));
    await tester.tap(find.text('Сохранить контур'));
    await tester.pump();
    expect(tester.widget<FilledButton>(find.widgetWithText(FilledButton, 'Сохранение…')).onPressed, isNull);
    response.complete(http.Response('{"saved":false}', 200));
    await tester.pumpAndSettle();
    expect(find.text('Сохранено'), findsNothing);
    expect(find.textContaining('Черновик остаётся'), findsOneWidget);
    await tester.tap(find.text('Сохранить контур'));
    await tester.pumpAndSettle();
    expect(find.text('Сохранено'), findsOneWidget);
    expect(attempts, 2);
    expect(result.height.valueM, 12.5);
  });
}

class _MultipartClient extends http.BaseClient {
  http.MultipartRequest? request;
  @override
  Future<http.StreamedResponse> send(http.BaseRequest request) async {
    this.request = request as http.MultipartRequest;
    return http.StreamedResponse(Stream.value(utf8.encode('{"saved":true}')), 200);
  }
}
