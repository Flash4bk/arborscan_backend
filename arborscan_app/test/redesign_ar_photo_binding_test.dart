import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:arborscan_app/analyze_page.dart';
import 'package:arborscan_app/app_theme.dart';
import 'package:arborscan_app/corrections_service.dart';
import 'package:crypto/crypto.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:shared_preferences/shared_preferences.dart';

const arFixture = <String, dynamic>{
  'height_m': 12.0,
  'trunk_diameter_m': .3,
  'trunk_measurement_height_m': 1.3,
  'distance_m': 4.0,
  'quality': .8,
  'overall_status': 'usable',
};

void main() {
  const picker = MethodChannel('plugins.flutter.io/image_picker');
  const ar = MethodChannel('arborscan/ar_measure');
  final photo = File('test/fixtures/reference_exif6.jpg').absolute;

  Future<void> tapAsync(WidgetTester tester, Finder finder) async {
    await tester.ensureVisible(finder);
    await tester.pump(const Duration(milliseconds: 350));
    await tester.runAsync(() async {
      await tester.tap(finder);
      await Future<void>.delayed(const Duration(milliseconds: 60));
    });
    await tester.pump(const Duration(milliseconds: 350));
    await tester.pump(const Duration(milliseconds: 350));
  }

  Future<void> setup(WidgetTester tester,
      {Future<dynamic> Function(MethodCall)? arHandler,
      double fontScale = 1}) async {
    tester.view.physicalSize = const Size(360, 1200);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    SharedPreferences.setMockInitialValues({
      'arborscan_auth_token': 'synthetic-owner-token',
      'arborscan_user_id': 'synthetic-owner',
    });
    final messenger = tester.binding.defaultBinaryMessenger;
    messenger.setMockMethodCallHandler(picker, (_) async => photo.path);
    messenger.setMockMethodCallHandler(ar, arHandler ?? (_) async => arFixture);
    addTearDown(() {
      messenger.setMockMethodCallHandler(picker, null);
      messenger.setMockMethodCallHandler(ar, null);
    });
    await tester.pumpWidget(MaterialApp(
      theme: AppTheme.light(),
      home: MediaQuery(
        data: MediaQueryData(textScaler: TextScaler.linear(fontScale)),
        child: const ArborScanPage(),
      ),
    ));
    await tester.pumpAndSettle();
  }

  for (final arFirst in [true, false]) {
    testWidgets(
        '${arFirst ? 'AR → photo' : 'Photo → AR'} binds only after explicit confirmation',
        (tester) async {
      final requests = <http.Request>[];
      await http.runWithClient(() async {
        await setup(tester);
        if (!arFirst) await tapAsync(tester, find.text('Галерея'));
        await tapAsync(tester, find.byKey(const ValueKey('open-ar')));
        if (arFirst) {
          expect(find.text('Добавьте фото измеренного дерева'), findsOneWidget);
          expect(
              tester
                  .widget<FilledButton>(
                      find.byKey(const ValueKey('analyze-photo')))
                  .onPressed,
              isNull);
          await tapAsync(tester, find.text('Галерея'));
          expect(find.text('Связать фото с измерением AR'), findsOneWidget);
        } else {
          expect(find.text('Связать AR с выбранным фото'), findsOneWidget);
        }
        await tapAsync(tester, find.text('Это то же дерево'));
        expect(find.text('Связано с этим фото'), findsOneWidget);
        await tapAsync(tester, find.byKey(const ValueKey('analyze-photo')));
        expect(requests, hasLength(1));
        final body =
            utf8.decode(requests.single.bodyBytes, allowMalformed: true);
        expect(requests.single.headers['Authorization'],
            'Bearer synthetic-owner-token');
        expect(body, contains('name="ar_same_tree_confirmed"\r\n\r\ntrue'));
        expect(body, contains('name="ar_height_m"\r\n\r\n12.0000'));
        final hash = await tester.runAsync(
            () async => sha256.convert(await photo.readAsBytes()).toString());
        expect(body, contains('name="ar_photo_sha256"\r\n\r\n$hash'));
        expect(body, isNot(contains('name="ar_crown_width_m"')));
        expect(find.text('Связано с этим фото'), findsOneWidget);
        expect(tester.takeException(), isNull);
      },
          () => MockClient((r) async {
                requests.add(r);
                return http.Response('{"detail":"synthetic_retry"}', 503);
              }));
    });
  }

  testWidgets(
      'A different tree explicitly drops AR; canceled photo binding keeps pending AR',
      (tester) async {
    final bodies = <String>[];
    await http.runWithClient(() async {
      await setup(tester);
      await tapAsync(tester, find.byKey(const ValueKey('open-ar')));
      await tapAsync(tester, find.text('Галерея'));
      await tapAsync(tester, find.text('Отмена'));
      expect(find.text('Добавьте фото измеренного дерева'), findsOneWidget);
      expect(find.text('Фото добавлено'), findsNothing);
      await tapAsync(tester, find.text('Галерея'));
      await tapAsync(tester, find.text('Другое дерево — без AR'));
      expect(find.text('Фото добавлено'), findsOneWidget);
      expect(find.text('Добавьте фото измеренного дерева'), findsNothing);
      await tapAsync(tester, find.byKey(const ValueKey('analyze-photo')));
      expect(bodies, hasLength(1));
      expect(bodies.single, isNot(contains('name="ar_height_m"')));
      expect(bodies.single, isNot(contains('name="ar_photo_sha256"')));
      expect(tester.takeException(), isNull);
    },
        () => MockClient((r) async {
              bodies.add(utf8.decode(r.bodyBytes, allowMalformed: true));
              return http.Response('{"detail":"synthetic_retry"}', 503);
            }));
  });

  testWidgets(
      'Photo-first cancellation keeps old AR and photo; reset removes binding',
      (tester) async {
    await setup(tester, fontScale: 2);
    await tapAsync(tester, find.byKey(const ValueKey('open-ar')));
    await tapAsync(tester, find.text('Галерея'));
    await tapAsync(tester, find.text('Это то же дерево'));
    await tapAsync(tester, find.byKey(const ValueKey('open-ar')));
    await tapAsync(tester, find.text('Отмена'));
    expect(find.text('Связано с этим фото'), findsOneWidget);
    await tester.drag(find.byType(Scrollable).first, const Offset(0, 900));
    await tester.pumpAndSettle();
    expect(find.text('Фото добавлено'), findsOneWidget);
    await tapAsync(tester, find.byTooltip('Начать заново'));
    expect(find.text('Фото добавлено'), findsNothing);
    expect(find.text('Связано с этим фото'), findsNothing);
    expect(tester.takeException(), isNull);
  });

  testWidgets(
      'Pending native call disables photo changes and a second AR entry',
      (tester) async {
    final completer = Completer<dynamic>();
    var calls = 0;
    await setup(tester, arHandler: (_) {
      calls++;
      return completer.future;
    });
    await tester.tap(find.byKey(const ValueKey('open-ar')));
    await tester.pump();
    expect(calls, 1);
    final button = tester.widget<OutlinedButton>(find.descendant(
        of: find.byKey(const ValueKey('open-ar')),
        matching: find.byType(OutlinedButton)));
    expect(button.onPressed, isNull);
    final camera = tester
        .widget<FilledButton>(find.widgetWithText(FilledButton, 'Камера'));
    expect(camera.onPressed, isNull);
    completer.complete(null);
    await tester.pumpAndSettle();
    expect(find.byKey(const ValueKey('open-ar')), findsOneWidget);
    expect(tester.takeException(), isNull);
  });

  testWidgets(
      'Account change while confirming AR photo discards the old selection and binding',
      (tester) async {
    await setup(tester);
    await tapAsync(tester, find.byKey(const ValueKey('open-ar')));
    await tapAsync(tester, find.text('Галерея'));
    expect(find.text('Связать фото с измерением AR'), findsOneWidget);

    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('arborscan_auth_token', 'different-synthetic-token');
    await prefs.setString('arborscan_user_id', 'different-synthetic-owner');
    CorrectionsService.authChanges.value++;
    await tester.pump();
    await tapAsync(tester, find.text('Это то же дерево'));

    expect(find.text('Фото добавлено'), findsNothing);
    expect(find.text('Связано с этим фото'), findsNothing);
    expect(find.text('Добавьте фото измеренного дерева'), findsNothing);
    final analyze = tester.widget<FilledButton>(
        find.byKey(const ValueKey('analyze-photo')));
    expect(analyze.onPressed, isNull);
    expect(tester.takeException(), isNull);
    await tester.pumpWidget(const SizedBox());
  });
}
