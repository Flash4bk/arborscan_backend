import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:arborscan_app/corrections_service.dart';
import 'package:arborscan_app/model_quality_service.dart';
import 'package:arborscan_app/model_quality_page.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  setUp(() => SharedPreferences.setMockInitialValues({
        'arborscan_auth_token': 'first',
        'arborscan_user_id': 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa'
      }));
  test('admin failure is explicit, token sent, no report-saved wording',
      () async {
    final service = ModelQualityService(
        clientFactory: () => MockClient((r) async {
              expect(r.headers['Authorization'], 'Bearer first');
              return http.Response('{}', 403);
            }));
    await expectLater(
        service.request(
            'first', http.Request('GET', Uri.https('example.test', '/status'))),
        throwsA(isA<CorrectionException>().having(
            (e) => e.toString(), 'message', contains('администратора'))));
  });
  test('account switch rejects delayed ML response', () async {
    final sent = Completer<void>(), response = Completer<http.Response>();
    final service = ModelQualityService(
        clientFactory: () => MockClient((r) {
              sent.complete();
              return response.future;
            }));
    final result = service.request(
        'first', http.Request('GET', Uri.https('example.test', '/status')));
    final assertion = expectLater(result, throwsA(isA<CorrectionException>()));
    await sent.future;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('arborscan_auth_token', 'second');
    CorrectionsService.authChanges.value++;
    response.complete(http.Response('{"private":"old account"}', 200));
    await assertion;
  });
  testWidgets(
      'admin screen restores durable jobs and invalidates on account change',
      (tester) async {
    final service = ModelQualityService(
        clientFactory: () => MockClient((r) async => http.Response(
            r.url.path.endsWith('/status')
                ? '{"worker":{"online":true},"jobs":[{"state":"running","progress":{"completed_epochs":1}}],"models":[],"snapshots":[],"active":[]}'
                : '{"eligible":[],"excluded":[]}',
            200)));
    await tester
        .pumpWidget(MaterialApp(home: ModelQualityPage(service: service)));
    await tester.pumpAndSettle();
    expect(find.text('Задача: running'), findsOneWidget);
    await tester.pumpWidget(const SizedBox());
    await tester
        .pumpWidget(MaterialApp(home: ModelQualityPage(service: service)));
    await tester.pumpAndSettle();
    expect(find.text('Задача: running'), findsOneWidget);
    CorrectionsService.authChanges.value++;
    await tester.pump();
    expect(find.text('Задача: running'), findsNothing);
    expect(find.text('Аккаунт изменился.'), findsOneWidget);
    await tester.pumpWidget(const SizedBox());
  });
}
