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
  test('200 without the operation ID does not confirm saving', () async {
    final service = ModelQualityService(
        clientFactory: () => MockClient((r) async => http.Response('{}', 200)));
    final request =
        http.Request('POST', Uri.https('example.test', '/snapshots'))
          ..body = '{"operation_id":"pending-operation"}';
    await expectLater(
        service.request('first', request), throwsA(isA<CorrectionException>()));
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
  testWidgets('open confirmation is hidden when the account changes',
      (tester) async {
    final service = ModelQualityService(
        clientFactory: () => MockClient((r) async => http.Response(
            r.url.path.endsWith('/status')
                ? '{"worker":{"online":true},"jobs":[],"models":[],"active":[],"snapshots":[{"id":"fixture","created_at":"test-date","model_type":"segmentation","manifest":{"items":[],"training_ready":true}}]}'
                : '{"eligible":[],"excluded":[]}',
            200)));
    await tester
        .pumpWidget(MaterialApp(home: ModelQualityPage(service: service)));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Снимок test-date'));
    await tester.pumpAndSettle();
    await tester.ensureVisible(find.text('Запустить пробное обучение'));
    await tester.tap(find.text('Запустить пробное обучение'));
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 500));
    expect(find.text('Пробная задача'), findsOneWidget);
    CorrectionsService.authChanges.value++;
    await tester.pump();
    expect(find.text('Пробная задача'), findsNothing);
    expect(find.text('Данные прежней сессии скрыты. Откройте раздел заново.'),
        findsOneWidget);
    await tester.tap(find.text('Закрыть'));
    await tester.pumpAndSettle();
    await tester.pumpWidget(const SizedBox());
  });
}
