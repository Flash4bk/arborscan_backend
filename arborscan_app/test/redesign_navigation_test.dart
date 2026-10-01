import 'dart:convert';

import 'package:arborscan_app/app_navigation.dart';
import 'package:arborscan_app/app_theme.dart';
import 'package:arborscan_app/history_tab_page.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:shared_preferences/shared_preferences.dart';

void main() {
  testWidgets('Retained history refreshes on activation and preserves filters',
      (tester) async {
    const token = 'synthetic-history-session';
    const owner = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa';
    SharedPreferences.setMockInitialValues({
      'arborscan_auth_token': token,
      'arborscan_user_id': owner,
      'arborscan_history': <String>[],
    });
    const pathChannel = MethodChannel('plugins.flutter.io/path_provider');
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(pathChannel, (_) async {
      // The nested server-history panel's disk journal is unrelated to tab
      // activation. Keep it unavailable rather than reading host/app files.
      throw PlatformException(code: 'synthetic-journal-unavailable');
    });
    addTearDown(() {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(pathChannel, null);
    });
    tester.view.physicalSize = const Size(400, 1400);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);

    Map<String, dynamic> row(String id, String species) => {
          'analysis_id': id,
          'species': species,
          'created_at': '2026-09-30T12:00:00Z',
          'lat': 53.9,
          'lon': 27.6,
        };
    final rows = [
      row('demo-old', 'Сосна DEMO старая'),
      row('demo-other', 'Берёза DEMO'),
    ];
    var historyRequests = 0;
    final requests = <http.Request>[];
    await http.runWithClient(() async {
      await tester.pumpWidget(
          MaterialApp(theme: AppTheme.light(), home: const HistoryTabPage()));
      await tester.pumpAndSettle();
      expect(historyRequests, 1);
      await tester.enterText(find.byType(TextField), 'сосна');
      await tester.ensureVisible(find.widgetWithText(FilterChip, 'Только GPS'));
      await tester.pumpAndSettle();
      await tester.tap(find.widgetWithText(FilterChip, 'Только GPS'));
      await tester.pumpAndSettle();
      expect(find.text('Сосна DEMO старая'), findsOneWidget);
      expect(find.text('Берёза DEMO'), findsNothing);

      // A new analysis arrives while the retained page is not recreated.
      rows.add(row('demo-new', 'Сосна DEMO новая'));
      AppNavigation.activate(1);
      await tester.pumpAndSettle();
      expect(historyRequests, 2);
      expect(find.text('Сосна DEMO новая'), findsOneWidget);
      expect(find.text('Сосна DEMO старая'), findsOneWidget);
      expect(find.text('Берёза DEMO'), findsNothing);
      expect(tester.widget<TextField>(find.byType(TextField)).controller!.text,
          'сосна');
      expect(
          tester
              .widget<FilterChip>(find.widgetWithText(FilterChip, 'Только GPS'))
              .selected,
          isTrue);
      expect(tester.takeException(), isNull);
      expect(
          requests.every((r) => r.headers['Authorization'] == 'Bearer $token'),
          isTrue);
      await tester.pumpWidget(const SizedBox.shrink());
      // A visit after disposal must not call a stale State listener.
      AppNavigation.activate(1);
      await tester.pump();
      expect(historyRequests, 2);
      expect(tester.takeException(), isNull);
    },
        () => MockClient((request) async {
              requests.add(request);
              if (request.url.path.endsWith('/analyses/my')) {
                historyRequests++;
                return http.Response(jsonEncode({'items': rows}), 200,
                    headers: {
                      'content-type': 'application/json; charset=utf-8'
                    });
              }
              return http.Response('{"items":[],"next_offset":null}', 200);
            }));
  });

  testWidgets('Disabled and busy primary actions stay readable and block taps',
      (tester) async {
    var presses = 0;
    await tester.pumpWidget(MaterialApp(
        theme: AppTheme.light(),
        home: Scaffold(
            body: Column(children: [
          AppActionButton(
              primary: true,
              enabled: false,
              label: 'Недоступно',
              onPressed: () => presses++),
          AppActionButton(
              primary: true,
              loading: true,
              label: 'Сохранение',
              onPressed: () => presses++),
        ]))));
    final background = Color.alphaBlend(
        AppTheme.surface2.withValues(alpha: .3), AppTheme.background);
    double contrast(Color foreground) {
      final a = foreground.computeLuminance();
      final b = background.computeLuminance();
      return (a > b ? (a + .05) / (b + .05) : (b + .05) / (a + .05));
    }

    for (final label in ['Недоступно', 'Сохранение']) {
      final text = tester.widget<Text>(find.text(label));
      expect(contrast(text.style!.color!), greaterThanOrEqualTo(4.5));
      await tester.tap(find.text(label));
    }
    final spinner = tester.widget<CircularProgressIndicator>(
        find.byType(CircularProgressIndicator));
    expect(contrast(spinner.valueColor!.value!), greaterThanOrEqualTo(4.5));
    expect(presses, 0);
    expect(tester.takeException(), isNull);
    await tester.pumpWidget(const SizedBox.shrink());
  });
}
