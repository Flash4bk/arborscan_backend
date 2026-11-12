import 'dart:convert';
import 'dart:async';
import 'package:arborscan_app/contour_drafts.dart';
import 'package:arborscan_app/corrections_service.dart';
import 'package:arborscan_app/history_tab_page.dart';
import 'package:arborscan_app/report_history_service.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:shared_preferences/shared_preferences.dart';

const _ownerA = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa';
const _ownerB = 'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb';
const _id = 'cccccccc-cccc-4ccc-8ccc-cccccccccccc';

class _EmptyJournal extends ContourDrafts {
  @override
  Future<List<Map<String, dynamic>>> list(String owner) async => [];
  @override
  Future<Map<String, dynamic>?> load(String owner, String id) async => null;
}

class _EmptyHistory extends ReportHistoryService {
  _EmptyHistory() : super(journal: _EmptyJournal());
  @override
  Future<Map<String, dynamic>> list(String token, int offset,
          {String? analysisId}) async =>
      {'items': [], 'next_offset': null};
}

void main() {
  final requests = <http.Request>[];
  setUp(() {
    requests.clear();
    SharedPreferences.setMockInitialValues({
      'arborscan_auth_token': 'demo-token-a',
      'arborscan_user_id': _ownerA,
    });
  });

  Future<void> open(WidgetTester t, {Future<http.Response>? detail}) async {
    await t.pumpWidget(MaterialApp(
        home: HistoryTabPage(
      reportHistory: _EmptyHistory(),
      clientFactory: () => MockClient((request) async {
        requests.add(request);
        if (!request.url.path.endsWith('/my') && detail != null) return detail;
        return http.Response(
            jsonEncode(request.url.path.endsWith('/my')
                ? {
                    'items': [
                      {
                        'analysis_id': _id,
                        'species': 'DEMO historical',
                        'created_at': '2026-11-12T00:00:00Z',
                        'risk_category': 'низкий',
                      }
                    ],
                  }
                : {
                    'analysis': {
                      'species': 'DEMO historical',
                      'risk': {'available': false}
                    },
                  }),
            200,
            headers: {'content-type': 'application/json'});
      }),
    )));
    await t.pumpAndSettle();
  }

  testWidgets('legacy report request uses bearer header without URL token',
      (t) async {
    await open(t);
    await t.ensureVisible(find.text('DEMO historical'));
    await t.tap(find.text('DEMO historical'));
    await t.pumpAndSettle();
    final detail = requests.singleWhere((r) => r.url.path.endsWith('/$_id'));
    expect(detail.headers['Authorization'], 'Bearer demo-token-a');
    expect(detail.url.queryParameters, isEmpty);
    expect(detail.url.toString(), isNot(contains('demo-token-a')));
    expect(find.text('БЕЗОПАСНО'), findsNothing);
    expect(t.takeException(), isNull);
  });

  testWidgets(
      'hiding server history survives screen recreation and can be undone',
      (t) async {
    await open(t);
    await t.ensureVisible(find.byKey(const ValueKey('hide-history-$_id')));
    await t.tap(find.byKey(const ValueKey('hide-history-$_id')));
    await t.pumpAndSettle();
    expect(find.textContaining('Серверные отчёты и оригиналы останутся'),
        findsOneWidget);
    await t.tap(find.text('Скрыть'));
    await t.pumpAndSettle();
    expect(find.text('DEMO historical'), findsNothing);
    final prefs = await SharedPreferences.getInstance();
    expect(prefs.getStringList('arborscan_history_hidden_v1_$_ownerA'), [_id]);
    expect(requests.every((r) => r.method == 'GET'), isTrue);
    await t.pumpWidget(const SizedBox());
    await open(t);
    expect(find.text('DEMO historical'), findsNothing);
    await t.tap(find.byTooltip('Показать скрытые записи'));
    await t.pumpAndSettle();
    expect(find.text('DEMO historical'), findsOneWidget);
    expect(
        prefs.getStringList('arborscan_history_hidden_v1_$_ownerA'), isEmpty);
  });

  testWidgets('hide decisions belong to the captured account only', (t) async {
    await open(t);
    await t.ensureVisible(find.byKey(const ValueKey('hide-history-$_id')));
    await t.tap(find.byKey(const ValueKey('hide-history-$_id')));
    await t.pumpAndSettle();
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('arborscan_auth_token', 'demo-token-b');
    await prefs.setString('arborscan_user_id', _ownerB);
    CorrectionsService.authChanges.value++;
    await t.pumpAndSettle();
    await t.tap(find.text('Скрыть'));
    await t.pumpAndSettle();
    expect(prefs.getStringList('arborscan_history_hidden_v1_$_ownerB'), isNull);
    expect(prefs.getStringList('arborscan_history_hidden_v1_$_ownerA'), isNull);
    expect(t.takeException(), isNull);
  });
  testWidgets(
      'reused token cannot accept a detail response from an older account generation',
      (t) async {
    final detail = Completer<http.Response>();
    await open(t, detail: detail.future);
    await t.ensureVisible(find.text('DEMO historical'));
    await t.tap(find.text('DEMO historical'));
    await t.pump();
    CorrectionsService.authChanges.value++;
    CorrectionsService.authChanges.value++;
    detail.complete(http.Response(
        '{"analysis":{"species":"DEMO private old response"}}', 200));
    await t.pumpAndSettle();
    expect(find.text('СОХРАНЁННЫЙ ОТЧЁТ'), findsNothing);
    expect(find.textContaining('DEMO private old response'), findsNothing);
    expect(t.takeException(), isNull);
  });
}
