import 'dart:async';
import 'dart:convert';

import 'package:arborscan_app/app_theme.dart';
import 'package:arborscan_app/corrections_service.dart';
import 'package:arborscan_app/data_presentation.dart';
import 'package:arborscan_app/model_quality_page.dart';
import 'package:arborscan_app/model_quality_service.dart';
import 'package:arborscan_app/profile_page.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:shared_preferences/shared_preferences.dart';

const _owner = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa';
const _png =
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+j8ioAAAAASUVORK5CYII=';
final _inventory = {
  'eligible': [
    {
      'owner_id': _owner,
      'correction_id': 'revision-a',
      'analysis_id': 'same-analysis'
    }
  ],
  'excluded': [
    {
      'owner_id': _owner,
      'correction_id': 'revision-b',
      'analysis_id': 'same-analysis',
      'reason': 'accepted_revision_required:submitted'
    },
    {
      'owner_id': _owner,
      'correction_id': 'revision-c',
      'analysis_id': 'other-analysis',
      'reason': 'new_server_reason:details'
    }
  ],
};

Future<void> _reach(WidgetTester tester, Finder finder) async {
  for (var i = 0; i < 40; i++) {
    if (finder.evaluate().isNotEmpty) {
      await tester.ensureVisible(finder.first);
      await tester.pumpAndSettle();
      if (finder.hitTestable().evaluate().isNotEmpty) return;
    }
    await tester.drag(find.byType(Scrollable).first, const Offset(0, -140));
    await tester.pumpAndSettle();
  }
  fail('Control was unreachable: $finder');
}

Future<void> _screen(WidgetTester tester, Widget child,
    {double scale = 1}) async {
  tester.view.physicalSize = const Size(360, 800);
  tester.view.devicePixelRatio = 1;
  addTearDown(tester.view.resetPhysicalSize);
  addTearDown(tester.view.resetDevicePixelRatio);
  await tester.pumpWidget(MaterialApp(
      theme: AppTheme.light(),
      builder: (context, child) => MediaQuery(
          data: MediaQuery.of(context)
              .copyWith(textScaler: TextScaler.linear(scale)),
          child: child!),
      home: child));
  await tester.pumpAndSettle();
}

void main() {
  setUp(() => SharedPreferences.setMockInitialValues(
      {'arborscan_auth_token': 'test', 'arborscan_user_id': _owner}));

  test('reasons keep distinct meanings instead of a generic error', () {
    expect(dataReasonLabel('accepted_revision_required:submitted'),
        'На проверке — нужна принятая ревизия');
    expect(dataReasonLabel('accepted_revision_required:rejected'),
        contains('Отклонено'));
    expect(
        dataReasonLabel('polygon_loss:{"iou":0.7}'), contains('теряет детали'));
    expect(dataReasonLabel('separate_confirmed_species_label_required'),
        contains('подтверждение вида'));
    expect(dataReasonLabel('future_code:x'),
        'Запись исключена — подробности доступны ниже');
    expect(modelJobLabel('running'), 'Выполняется');
  });

  testWidgets(
      'real rows drive counts, filters preserve revisions and raw reasons',
      (tester) async {
    final requests = <http.Request>[];
    final service = ModelQualityService(
        clientFactory: () => MockClient((request) async {
              requests.add(request);
              expect(request.headers['Authorization'], 'Bearer test');
              return http.Response(
                  jsonEncode(request.url.path.endsWith('/data')
                      ? _inventory
                      : request.url.path.contains('/review/')
                          ? {'original_image_base64': _png}
                          : {
                              'worker': {'online': false},
                              'jobs': [],
                              'models': [],
                              'snapshots': [],
                              'active': []
                            }),
                  200);
            }));
    await _screen(tester, ModelQualityPage(service: service));
    final summary =
        tester.widget<DatasetSummaryCard>(find.byType(DatasetSummaryCard));
    expect(summary.included, 1);
    expect(summary.excluded, 2);
    await _reach(tester, find.text('Контур 2'));
    expect(find.byKey(const ValueKey('$_owner/revision-a')), findsOneWidget);
    expect(find.byKey(const ValueKey('$_owner/revision-b')), findsOneWidget);
    await _reach(tester, find.text('Исключены'));
    await tester.tap(find.text('Исключены'));
    await tester.pumpAndSettle();
    expect(find.byKey(const ValueKey('$_owner/revision-a')), findsNothing);
    await _reach(tester, find.text('Контур 2'));
    expect(
        find.textContaining('На проверке — нужна принятая ревизия',
            findRichText: true),
        findsOneWidget);
    await tester.tap(find.text('Контур 2'));
    await tester.pumpAndSettle();
    await _reach(tester, find.text('Подробности ревизии').first);
    await tester.tap(find.text('Подробности ревизии').first);
    await tester.pumpAndSettle();
    await _reach(
        tester, find.text('Код причины: accepted_revision_required:submitted'));
    expect(find.text('Ревизия: revision-b\nАнализ: same-analysis'),
        findsOneWidget);
    expect(requests.every((request) => request.method == 'GET'), isTrue);
    expect(requests.any((request) => request.url.path.endsWith('/revision-b')),
        isTrue);
    await tester.pumpWidget(const SizedBox());
  });

  testWidgets(
      'account switch hides photo response and revision details at 200 percent',
      (tester) async {
    final image = Completer<http.Response>();
    final service = ModelQualityService(
        clientFactory: () => MockClient((request) async {
              if (request.url.path.contains('/review/')) return image.future;
              return http.Response(
                  jsonEncode(request.url.path.endsWith('/data')
                      ? _inventory
                      : {
                          'worker': {'online': false},
                          'jobs': [],
                          'models': [],
                          'snapshots': [],
                          'active': []
                        }),
                  200);
            }));
    await _screen(tester, ModelQualityPage(service: service), scale: 2);
    await _reach(tester, find.text('Контур 1'));
    expect(tester.takeException(), isNull);
    CorrectionsService.authChanges.value++;
    image.complete(
        http.Response(jsonEncode({'original_image_base64': _png}), 200));
    await tester.pumpAndSettle();
    expect(find.byType(DataRevisionTile), findsNothing);
    expect(find.byType(Image), findsNothing);
    expect(find.text('Аккаунт изменился.'), findsOneWidget);
    await tester.pumpWidget(const SizedBox());
  });

  testWidgets(
      'password help is reachable at large text and sends no recovery request',
      (tester) async {
    SharedPreferences.setMockInitialValues({});
    var requests = 0;
    await http.runWithClient(() async {
      await _screen(tester, const ProfilePage(), scale: 2);
      await _reach(tester, find.widgetWithText(ChoiceChip, 'Вход'));
      await tester.tap(find.widgetWithText(ChoiceChip, 'Вход'));
      await tester.pumpAndSettle();
      await _reach(tester, find.text('Не помню пароль'));
      await tester.tap(find.text('Не помню пароль'));
      await tester.pumpAndSettle();
      expect(find.textContaining('автоматический сброс пароля пока недоступен'),
          findsOneWidget);
      expect(find.text('Закрыть'), findsOneWidget);
      expect(tester.takeException(), isNull);
      expect(requests, 0);
    },
        () => MockClient((_) async {
              requests++;
              return http.Response('{}', 500);
            }));
  });
}
