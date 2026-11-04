import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:ui' as ui;

import 'package:arborscan_app/admin_panel_page.dart';
import 'package:arborscan_app/app_theme.dart';
import 'package:arborscan_app/corrections_service.dart';
import 'package:arborscan_app/model_quality_page.dart';
import 'package:arborscan_app/model_quality_service.dart';
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:shared_preferences/shared_preferences.dart';

const _sha = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa';

Map<String, dynamic> _status() => {
      'baseline': {
        'version': 4,
        'weights_sha256': _sha,
        'classes': {'0': 'tree'},
      },
      'worker': {'online': true},
      'jobs': [],
      'models': [],
      'snapshots': [],
      'active': [
        {'model_type': 'segmentation', 'model_id': null, 'generation': 0},
      ],
    };

http.Response _json(Object value, [int status = 200]) =>
    http.Response(jsonEncode(value), status,
        headers: {'content-type': 'application/json; charset=utf-8'});

http.Response _data() => _json({'eligible': [], 'excluded': []});

ModelQualityService _service(
        Future<http.Response> Function(http.Request) handler) =>
    ModelQualityService(clientFactory: () => MockClient(handler));

Future<void> _openModels(WidgetTester tester) async {
  await tester.tap(find.text('Модели'));
  await tester.pumpAndSettle();
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  setUp(() => SharedPreferences.setMockInitialValues({
        'arborscan_auth_token': 'first',
        'arborscan_user_id': 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa',
      }));

  testWidgets(
      'baseline weights remain visible with generation zero and no jobs',
      (tester) async {
    await tester.pumpWidget(MaterialApp(
        home: ModelQualityPage(
            service: _service((r) async =>
                r.url.path.endsWith('/status') ? _json(_status()) : _data()))));
    await tester.pumpAndSettle();
    await _openModels(tester);
    expect(find.text('Исходная модель v4'), findsOneWidget);
    expect(find.text('SHA-256 весов: $_sha'), findsOneWidget);
    expect(
        find.text(
            'Активная сегментация: Исходная модель v4. Версия реестра: 0'),
        findsOneWidget);
    expect(find.text('Задач и кандидатов пока нет.'), findsOneWidget);
    expect(find.text('Откатить к исходной модели'), findsNothing);
    expect(tester.takeException(), isNull);
    await tester.pumpWidget(const SizedBox());
  });

  testWidgets(
      'data 503 does not hide loaded model status; data retry is separate',
      (tester) async {
    var failData = true, statusRequests = 0, dataRequests = 0;
    await tester.pumpWidget(
        MaterialApp(home: ModelQualityPage(service: _service((r) async {
      if (r.url.path.endsWith('/status')) {
        statusRequests++;
        return _json(_status());
      }
      dataRequests++;
      return failData ? _json({}, 503) : _data();
    }))));
    await tester.pumpAndSettle();
    expect(find.textContaining('Данные:'), findsOneWidget);
    await _openModels(tester);
    expect(find.text('Исходная модель v4'), findsOneWidget);
    expect(find.text('SHA-256 весов: $_sha'), findsOneWidget);
    expect(find.textContaining('Данные:'), findsNothing);
    await tester.tap(find.text('Данные'));
    await tester.pumpAndSettle();
    failData = false;
    await tester.tap(find.text('Повторить загрузку данных'));
    await tester.pumpAndSettle();
    expect(find.textContaining('Данные:'), findsNothing);
    expect(statusRequests, 1);
    expect(dataRequests, 2);
    await tester.pumpWidget(const SizedBox());
  });

  testWidgets(
      'a pending dataset does not postpone the model status or leak after logout',
      (tester) async {
    final data = Completer<http.Response>();
    await tester.pumpWidget(MaterialApp(
        home: ModelQualityPage(
            service: _service((r) async => r.url.path.endsWith('/status')
                ? _json(_status())
                : await data.future))));
    for (var i = 0; i < 8; i++) {
      await tester.pump(const Duration(milliseconds: 20));
    }
    await tester.tap(find.text('Модели'));
    await tester.pump();
    expect(find.text('Исходная модель v4'), findsOneWidget);
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('arborscan_auth_token', 'second');
    CorrectionsService.authChanges.value++;
    await tester.pump();
    expect(find.text('Исходная модель v4'), findsNothing);
    expect(find.text('SHA-256 весов: $_sha'), findsNothing);
    expect(find.text('Аккаунт изменился.'), findsOneWidget);
    data.complete(_data());
    await tester.pumpAndSettle();
    expect(find.text('Исходная модель v4'), findsNothing);
    await tester.pumpWidget(const SizedBox());
  });

  testWidgets(
      'status failure does not claim an empty registry; model retry is separate',
      (tester) async {
    var failStatus = true, dataRequests = 0;
    await tester.pumpWidget(
        MaterialApp(home: ModelQualityPage(service: _service((r) async {
      if (r.url.path.endsWith('/status')) {
        return failStatus ? _json({}, 503) : _json(_status());
      }
      dataRequests++;
      return _data();
    }))));
    await tester.pumpAndSettle();
    await _openModels(tester);
    expect(find.textContaining('Модели:'), findsOneWidget);
    expect(find.text('Задач и кандидатов пока нет.'), findsNothing);
    failStatus = false;
    await tester.tap(find.text('Повторить загрузку моделей'));
    await tester.pumpAndSettle();
    expect(find.text('Исходная модель v4'), findsOneWidget);
    expect(find.textContaining('Модели:'), findsNothing);
    expect(dataRequests, 1);
    await tester.pumpWidget(const SizedBox());
  });

  for (final deniedCode in [401, 403]) {
    testWidgets('$deniedCode clears the cached model and data on refresh',
        (tester) async {
      var denied = false;
      await tester.pumpWidget(
          MaterialApp(home: ModelQualityPage(service: _service((r) async {
        if (r.url.path.endsWith('/status')) {
          return denied ? _json({}, deniedCode) : _json(_status());
        }
        return _data();
      }))));
      await tester.pumpAndSettle();
      await _openModels(tester);
      expect(find.text('Исходная модель v4'), findsOneWidget);
      denied = true;
      await tester.tap(find.byTooltip('Обновить модели и данные'));
      await tester.pumpAndSettle();
      expect(find.text('Исходная модель v4'), findsNothing);
      expect(find.text('SHA-256 весов: $_sha'), findsNothing);
      expect(find.text('Модели'), findsNothing);
      expect(find.text('Повторить проверку доступа'), findsOneWidget);
      await tester.pumpWidget(const SizedBox());
    });
  }

  testWidgets(
      'v3 archive lists versions without a pretend selector or activation POST',
      (tester) async {
    final requests = <http.Request>[];
    await http.runWithClient(() async {
      await tester.pumpWidget(const MaterialApp(
          home: AdminPanelPage(baseUrl: 'https://example.invalid')));
      await tester.pumpAndSettle();
      await tester.scrollUntilVisible(find.text('Архив v3'), 180);
      await tester.tap(find.text('Архив v3'));
      await tester.pumpAndSettle();
      await tester.scrollUntilVisible(find.text('Прежние версии моделей'), 150);
      await tester.scrollUntilVisible(find.text('Версия 1'), 100);
      expect(find.byType(DropdownButtonFormField<int>), findsNothing);
      expect(find.text('Версия 1'), findsOneWidget);
      expect(find.text('Версия 2'), findsOneWidget);
      expect(find.text('Версия 4'), findsOneWidget);
      expect(find.textContaining('не выбирают модель для текущего анализа v4'),
          findsOneWidget);
      await tester.scrollUntilVisible(find.text('Версия 1').hitTestable(), 120,
          scrollable: find.byType(Scrollable).first);
      expect(find.text('Версия 1').hitTestable(), findsOneWidget);
      await tester.tap(find.text('Версия 1'));
      await tester.pumpAndSettle();
      expect(requests.every((r) => r.method == 'GET'), isTrue);
      expect(requests.any((r) => r.url.path.endsWith('/admin/models')), isTrue);
      await tester.pumpWidget(const SizedBox());
    },
        () => MockClient((r) async {
              requests.add(r);
              if (r.url.path.endsWith('/admin/me')) {
                return _json({
                  'user': {'id': 'test-admin', 'role': 'admin', 'name': 'Test'}
                });
              }
              if (r.url.path.endsWith('/admin/models')) {
                return _json({
                  'models': [1, 2, 4],
                  'active_model_version': 4
                });
              }
              if (r.url.path.endsWith('/admin/training-events')) {
                return _json({'events': []});
              }
              return _json({'active_model_version': 4});
            }));
  });

  testWidgets('baseline details render in the app theme', (tester) async {
    await tester.binding.setSurfaceSize(const Size(411, 891));
    addTearDown(() => tester.binding.setSurfaceSize(null));
    for (final entry in {
      'Atlas': 'assets/fonts/DejaVuSans.ttf',
      'Literata': 'assets/fonts/Literata.ttf',
      'MaterialIcons': 'fonts/MaterialIcons-Regular.otf',
    }.entries) {
      final loader = FontLoader(entry.key)
        ..addFont(rootBundle.load(entry.value));
      await loader.load();
    }
    final capture = GlobalKey();
    await tester.pumpWidget(RepaintBoundary(
        key: capture,
        child: MaterialApp(
            theme: AppTheme.light(),
            debugShowCheckedModeBanner: false,
            home: ModelQualityPage(
                service: _service((r) async => r.url.path.endsWith('/status')
                    ? _json(_status())
                    : _data())))));
    await tester.pumpAndSettle();
    await _openModels(tester);
    final boundary =
        capture.currentContext!.findRenderObject() as RenderRepaintBoundary;
    await tester.runAsync(() async {
      final folder = Directory(
          Platform.environment['ARBORSCAN_MODEL_UI_OUTPUT'] ??
              '../output/model-comparison-20261104/ui')
        ..createSync(recursive: true);
      final picture = await boundary.toImage(pixelRatio: 2);
      final png = await picture.toByteData(format: ui.ImageByteFormat.png);
      await File('${folder.path}/model-registry-baseline.png')
          .writeAsBytes(png!.buffer.asUint8List());
      picture.dispose();
    });
    expect(tester.takeException(), isNull);
    await tester.pumpWidget(const SizedBox());
  });
}
