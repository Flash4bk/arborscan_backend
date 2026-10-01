import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:ui' as ui;

import 'package:arborscan_app/app_theme.dart';
import 'package:arborscan_app/corrections_service.dart';
import 'package:arborscan_app/feedback_page.dart';
import 'package:arborscan_app/model_quality_page.dart';
import 'package:arborscan_app/model_quality_service.dart';
import 'package:arborscan_app/onboarding_page.dart';
import 'package:arborscan_app/unified_analysis_models.dart';
import 'package:arborscan_app/unified_analysis_report_page.dart';
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:shared_preferences/shared_preferences.dart';

const _token = 'synthetic-additional-session';
const _owner = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa';
const _png =
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNImXYCAAMkAcMgVWSjAAAAAElFTkSuQmCC';
final _capture = GlobalKey();

ThemeData _theme() {
  final theme = AppTheme.light();
  ButtonStyle font(ButtonStyle? style) =>
      (style ?? const ButtonStyle()).copyWith(
          textStyle: WidgetStatePropertyAll(
              (style?.textStyle?.resolve({}) ?? const TextStyle())
                  .copyWith(fontFamily: 'AS15ExtraEvidence')));
  return theme.copyWith(
    textTheme: theme.textTheme.apply(fontFamily: 'AS15ExtraEvidence'),
    primaryTextTheme:
        theme.primaryTextTheme.apply(fontFamily: 'AS15ExtraEvidence'),
    appBarTheme: theme.appBarTheme.copyWith(
        titleTextStyle: theme.appBarTheme.titleTextStyle
            ?.copyWith(fontFamily: 'AS15ExtraEvidence')),
    filledButtonTheme:
        FilledButtonThemeData(style: font(theme.filledButtonTheme.style)),
    outlinedButtonTheme:
        OutlinedButtonThemeData(style: font(theme.outlinedButtonTheme.style)),
    textButtonTheme:
        TextButtonThemeData(style: font(theme.textButtonTheme.style)),
  );
}

Future<void> _screen(WidgetTester tester, Widget page,
    {double width = 360, double scale = 2}) async {
  tester.view.physicalSize = Size(width, 800);
  tester.view.devicePixelRatio = 1;
  addTearDown(tester.view.resetPhysicalSize);
  addTearDown(tester.view.resetDevicePixelRatio);
  await tester.pumpWidget(MaterialApp(
    theme: _theme(),
    builder: (context, child) => MediaQuery(
        data: MediaQuery.of(context)
            .copyWith(textScaler: TextScaler.linear(scale)),
        child: RepaintBoundary(key: _capture, child: child!)),
    home: page,
  ));
}

Future<void> _snapshot(WidgetTester tester, String name) async {
  await tester.pump();
  expect(tester.takeException(), isNull,
      reason: 'Screenshots must not conceal layout exceptions.');
  await tester.runAsync(() async {
    final boundary =
        _capture.currentContext!.findRenderObject()! as RenderRepaintBoundary;
    final image = await boundary.toImage(pixelRatio: 2);
    final bytes = await image.toByteData(format: ui.ImageByteFormat.png);
    final directory = Directory('../output/as15/widget-extra');
    await directory.create(recursive: true);
    await File('${directory.path}/$name.png')
        .writeAsBytes(bytes!.buffer.asUint8List());
    image.dispose();
  });
}

Future<void> _reach(WidgetTester tester, Finder target) async {
  final scroll = find
      .descendant(of: find.byType(ListView), matching: find.byType(Scrollable))
      .first;
  for (var i = 0; i < 30; i++) {
    if (target.evaluate().isNotEmpty) {
      await tester.ensureVisible(target);
      await tester.pumpAndSettle();
      return;
    }
    await tester.drag(scroll, const Offset(0, -220));
    await tester.pumpAndSettle();
  }
  expect(target, findsOneWidget);
}

void main() {
  setUpAll(() async {
    final font = FontLoader('AS15ExtraEvidence')
      ..addFont(rootBundle.load('assets/fonts/DejaVuSans.ttf'));
    await font.load();
    final icons = FontLoader('MaterialIcons')
      ..addFont(rootBundle.load('fonts/MaterialIcons-Regular.otf'));
    await icons.load();
  });
  setUp(() => SharedPreferences.setMockInitialValues({
        'arborscan_auth_token': _token,
        'arborscan_user_id': _owner,
        'arborscan_profile_role': 'admin',
      }));

  testWidgets('Models loading, empty data and account guard at 160% font',
      (tester) async {
    final response = Completer<http.Response>();
    final requests = <http.Request>[];
    final service = ModelQualityService(
        clientFactory: () => MockClient((request) async {
              requests.add(request);
              expect(request.headers['Authorization'], 'Bearer $_token');
              if (request.url.path.endsWith('/status')) return response.future;
              return http.Response('{"eligible":[],"excluded":[]}', 200);
            }));
    await _screen(tester, ModelQualityPage(service: service), scale: 1.6);
    await tester.pump();
    await tester.pump();
    expect(find.byType(LinearProgressIndicator), findsOneWidget);
    final refresh = find.widgetWithIcon(IconButton, Icons.refresh);
    expect(tester.widget<IconButton>(refresh).onPressed, isNull);
    await _snapshot(tester, 'models-loading-160');
    response.complete(http.Response(
        jsonEncode({
          'worker': {'online': false},
          'active': [],
          'models': [],
          'snapshots': [],
          'jobs': [],
        }),
        200));
    await tester.pumpAndSettle();
    expect(tester.widget<IconButton>(refresh).onPressed, isNotNull);
    await _reach(tester, find.text('Допущено: 0. Исключено: 0.'));
    expect(find.text('Запустить пробное обучение'), findsNothing);
    expect(find.text('Активировать после просмотра'), findsNothing);
    await _snapshot(tester, 'models-empty-160');
    CorrectionsService.authChanges.value++;
    await tester.pumpAndSettle();
    expect(find.text('Аккаунт изменился.'), findsOneWidget);
    expect(find.byType(DropdownButton<String>), findsNothing);
    expect(requests, hasLength(2));
    expect(requests.every((request) => request.method == 'GET'), isTrue);
    await _snapshot(tester, 'models-account-guard-160');
    await tester.pumpWidget(const SizedBox.shrink());
  });

  testWidgets('Models forbidden role remains explicit and cannot train / 200%',
      (tester) async {
    SharedPreferences.setMockInitialValues({
      'arborscan_auth_token': _token,
      'arborscan_user_id': _owner,
      'arborscan_profile_role': 'user',
    });
    final requests = <http.Request>[];
    final service = ModelQualityService(
        clientFactory: () => MockClient((request) async {
              requests.add(request);
              return http.Response('{}', 403);
            }));
    await _screen(tester, ModelQualityPage(service: service));
    await tester.pumpAndSettle();
    expect(find.textContaining('нужны права администратора'), findsOneWidget);
    expect(find.text('Запустить пробное обучение'), findsNothing);
    expect(find.text('Активировать после просмотра'), findsNothing);
    await _snapshot(tester, 'models-forbidden-200');
    await tester.tap(find.widgetWithIcon(IconButton, Icons.refresh));
    await tester.pumpAndSettle();
    expect(find.textContaining('нужны права администратора'), findsOneWidget);
    expect(requests, hasLength(2));
    expect(requests.every((request) => request.method == 'GET'), isTrue);
    await tester.pumpWidget(const SizedBox.shrink());
  });

  testWidgets(
      'Feedback keeps unavailable dimensions and validates edits / 200%',
      (tester) async {
    var requests = 0;
    await http.runWithClient(() async {
      await _screen(
          tester,
          const FeedbackPage(
              analysisId: 'DEMO-feedback',
              originalImageBase64: _png,
              species: 'DEMO сосна',
              authToken: _token),
          width: 320);
      await tester.pumpAndSettle();
      await _snapshot(tester, 'feedback-top-200');
      final species = find.byWidgetPredicate((widget) =>
          widget is TextField && widget.decoration?.labelText == 'Вид дерева');
      await _reach(tester, species);
      await tester.enterText(species, 'DEMO исправленная сосна');
      FocusManager.instance.primaryFocus?.unfocus();
      await tester.pumpAndSettle();
      final height = find.byWidgetPredicate((widget) =>
          widget is TextField && widget.decoration?.labelText == 'Высота, м');
      await _reach(tester, height);
      expect(tester.widget<TextField>(height).controller!.text, isEmpty);
      await tester.enterText(height, '999');
      FocusManager.instance.primaryFocus?.unfocus();
      await tester.pumpAndSettle();
      await tester.tap(find.byTooltip('Отправить'));
      await tester.pumpAndSettle();
      expect(find.textContaining('допустимо от 0.5 до 100'), findsOneWidget);
      expect(tester.widget<TextField>(height).decoration!.errorMaxLines, 4);
      await _snapshot(tester, 'feedback-validation-200');
      await _reach(tester, find.text('Использовать для обучения'));
      final use = find.byType(SwitchListTile);
      await tester.tap(use);
      await tester.pumpAndSettle();
      expect(tester.widget<SwitchListTile>(use).value, isFalse);
      await _reach(tester, find.text('Отмена'));
      await _snapshot(tester, 'feedback-actions-200');
      expect(requests, 0);
      await tester.pumpWidget(const SizedBox.shrink());
    },
        () => MockClient((_) async {
              requests++;
              return http.Response('{}', 500);
            }));
  });

  testWidgets('Onboarding instructions scroll at 200% and finish returns home',
      (tester) async {
    await _screen(tester, Scaffold(body: Builder(builder: (context) {
      return TextButton(
          onPressed: () => Navigator.push(context,
              MaterialPageRoute(builder: (_) => const OnboardingPage())),
          child: const Text('DEMO открыть помощь'));
    })), width: 320);
    await tester.tap(find.text('DEMO открыть помощь'));
    await tester.pumpAndSettle();
    final titles = [
      'Анализ фотографии',
      'Измерения и эталон',
      'Границы результата',
    ];
    for (var i = 0; i < titles.length; i++) {
      expect(find.text(titles[i]), findsOneWidget);
      expect(find.text('${i + 1} из 3'), findsOneWidget);
      await _snapshot(tester, 'onboarding-${i + 1}-top-200');
      final pageList = find.byType(ListView).hitTestable();
      final position = tester
          .state<ScrollableState>(find
              .descendant(of: pageList, matching: find.byType(Scrollable))
              .first)
          .position;
      for (var drag = 0; drag < 6 && position.extentAfter > 0; drag++) {
        await tester.drag(pageList, const Offset(0, -500));
        await tester.pumpAndSettle();
      }
      expect(position.extentAfter, closeTo(0, .1),
          reason: 'The end of every instruction must remain reachable.');
      await _snapshot(tester, 'onboarding-${i + 1}-scrolled-200');
      await tester.tap(find.text(i == 2 ? 'Готово' : 'Далее'));
      await tester.pumpAndSettle();
    }
    expect(find.text('DEMO открыть помощь'), findsOneWidget);
    expect(find.byType(OnboardingPage), findsNothing);
    expect(tester.takeException(), isNull);
    await tester.pumpWidget(const SizedBox.shrink());
  });

  testWidgets('Actual v4 report keeps missing metric values and original photo',
      (tester) async {
    final result = UnifiedAnalysisResult.fromJson({
      'analysis_id': 'DEMO-unavailable',
      'api_version': 'v4',
      'analysis_status': 'absolute_scale_required',
      'tree': {'detected': true, 'segmentation_confidence': .8},
      'species': {'display_name': 'DEMO сосна', 'confidence': .82},
      'measurements': {
        'height': {'value_m': null, 'value_px': 850, 'source': 'unavailable'},
        'crown_width': {
          'value_m': null,
          'value_px': 450,
          'source': 'unavailable'
        },
        'trunk_diameter': {'value_m': null, 'source': 'unavailable'},
      },
      'quality': {'overall': .7},
      'risk': {'available': false, 'reason': 'DEMO: нет физических данных'},
    });
    final image = (await tester.runAsync(
        () => File('test/fixtures/reference_exif6.jpg').readAsBytes()))!;
    await _screen(
        tester,
        UnifiedAnalysisReportPage(
            result: result, fallbackImageBytes: image, allowServerSave: false),
        scale: 1.6);
    await tester.pumpAndSettle();
    final preview = tester.widget<Image>(find.byType(Image).first);
    expect((preview.image as MemoryImage).bytes, image);
    expect(find.text('Сохранить отчёт в аккаунте'), findsNothing);
    await _snapshot(tester, 'v4-report-photo-160');
    for (final title in ['Высота дерева', 'Ширина кроны', 'Диаметр ствола']) {
      await _reach(tester, find.text(title));
      final card =
          find.ancestor(of: find.text(title), matching: find.byType(Card));
      expect(
          find.descendant(of: card, matching: find.text('—')), findsOneWidget);
      expect(find.descendant(of: card, matching: find.text('Недоступно')),
          findsOneWidget);
      expect(find.text('0.00 м'), findsNothing);
      expect(find.text('0.000 м'), findsNothing);
      await _snapshot(
          tester,
          'v4-report-${title == 'Высота дерева' ? 'height' : title == 'Ширина кроны' ? 'crown' : 'dbh'}-160');
    }
    expect(result.height.valueM, isNull);
    expect(result.crownWidth.valueM, isNull);
    expect(result.trunkDiameter.valueM, isNull);
    expect(tester.takeException(), isNull);
    await tester.pumpWidget(const SizedBox.shrink());
  });
}
