import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:ui' as ui;

import 'package:arborscan_app/app_theme.dart';
import 'package:arborscan_app/corrections_service.dart';
import 'package:arborscan_app/profile_page.dart';
import 'package:arborscan_app/saved_corrections_page.dart';
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:shared_preferences/shared_preferences.dart';

const _owner = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa';
const _png =
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNImXYCAAMkAcMgVWSjAAAAAElFTkSuQmCC';
const _reason = 'DEMO: Граница кроны проходит по соседнему дереву. '
    'Исправьте верхнюю часть контура и исключите фон, сохранив ветви текущего '
    'дерева. После исправления отправьте новую ревизию на проверку. '
    'Этот длинный комментарий должен переноситься и оставаться полностью '
    'доступным на маленьком экране с увеличенным системным шрифтом.';

final _capture = GlobalKey();
final _verticalScroll = find
    .descendant(of: find.byType(ListView), matching: find.byType(Scrollable))
    .first;

Future<void> _screen(WidgetTester tester, Widget page, double scale) async {
  tester.view.physicalSize = const Size(360, 800);
  tester.view.devicePixelRatio = 1;
  addTearDown(tester.view.resetPhysicalSize);
  addTearDown(tester.view.resetDevicePixelRatio);
  await tester.pumpWidget(MaterialApp(
    theme: _evidenceTheme(),
    builder: (context, child) => MediaQuery(
      data:
          MediaQuery.of(context).copyWith(textScaler: TextScaler.linear(scale)),
      child: RepaintBoundary(key: _capture, child: child!),
    ),
    home: page,
  ));
}

ThemeData _evidenceTheme() {
  final theme = AppTheme.light();
  ButtonStyle font(ButtonStyle? style) =>
      (style ?? const ButtonStyle()).copyWith(
          textStyle: WidgetStatePropertyAll(
              (style?.textStyle?.resolve({}) ?? const TextStyle())
                  .copyWith(fontFamily: 'AS15Evidence')));
  return theme.copyWith(
    textTheme: theme.textTheme.apply(fontFamily: 'AS15Evidence'),
    primaryTextTheme: theme.primaryTextTheme.apply(fontFamily: 'AS15Evidence'),
    appBarTheme: theme.appBarTheme.copyWith(
        titleTextStyle: theme.appBarTheme.titleTextStyle
            ?.copyWith(fontFamily: 'AS15Evidence')),
    filledButtonTheme:
        FilledButtonThemeData(style: font(theme.filledButtonTheme.style)),
    outlinedButtonTheme:
        OutlinedButtonThemeData(style: font(theme.outlinedButtonTheme.style)),
    textButtonTheme:
        TextButtonThemeData(style: font(theme.textButtonTheme.style)),
    snackBarTheme: theme.snackBarTheme.copyWith(
        contentTextStyle:
            (theme.snackBarTheme.contentTextStyle ?? const TextStyle())
                .copyWith(fontFamily: 'AS15Evidence')),
  );
}

Future<void> _snapshot(WidgetTester tester, String name) async {
  await tester.pump();
  expect(tester.takeException(), isNull,
      reason: 'The real screen must not overflow before recording evidence.');
  await tester.runAsync(() async {
    final boundary =
        _capture.currentContext!.findRenderObject()! as RenderRepaintBoundary;
    final image = await boundary.toImage(pixelRatio: 2);
    final bytes = await image.toByteData(format: ui.ImageByteFormat.png);
    final directory = Directory('../output/as15/widget-visual');
    await directory.create(recursive: true);
    await File('${directory.path}/$name.png')
        .writeAsBytes(bytes!.buffer.asUint8List());
    image.dispose();
  });
}

Future<void> _reach(WidgetTester tester, Finder finder) async {
  for (var attempt = 0; attempt < 25; attempt++) {
    if (finder.evaluate().isNotEmpty) {
      final point = tester.getCenter(finder);
      if (point.dy >= 80 && point.dy <= 780) return;
      await tester.drag(
          _verticalScroll, Offset(0, point.dy > 780 ? -180 : 180));
    } else {
      await tester.drag(_verticalScroll, const Offset(0, -180));
    }
    await tester.pumpAndSettle();
  }
  expect(finder, findsOneWidget);
  expect(tester.getCenter(finder).dy, inInclusiveRange(80, 780));
}

void main() {
  setUpAll(() async {
    // Widget evidence uses the bundled Cyrillic font instead of test Ahem boxes.
    // Device typography remains a separate visual check.
    final font = FontLoader('AS15Evidence')
      ..addFont(rootBundle.load('assets/fonts/DejaVuSans.ttf'));
    await font.load();
    final icons = FontLoader('MaterialIcons')
      ..addFont(rootBundle.load('fonts/MaterialIcons-Regular.otf'));
    await icons.load();
  });
  setUp(() => SharedPreferences.setMockInitialValues({
        'arborscan_auth_token': 'synthetic-session',
        'arborscan_user_id': _owner,
        'arborscan_profile_logged_in': true,
      }));

  testWidgets(
      'guest login supports 200% font and validates password without a request',
      (tester) async {
    SharedPreferences.setMockInitialValues({});
    var requests = 0;
    await http.runWithClient(() async {
      await _screen(tester, const ProfilePage(), 2);
      await tester.pumpAndSettle();
      final registration = find.descendant(
          of: find.byType(ChoiceChip), matching: find.text('Регистрация'));
      await _reach(tester, registration);
      final label = tester.renderObject<RenderParagraph>(registration);
      expect(
          label
              .getBoxesForSelection(const TextSelection(
                  baseOffset: 0, extentOffset: 'Регистрация'.length))
              .length,
          1,
          reason: 'The registration label must not split off a final letter.');
      await _snapshot(tester, 'registration-large-font');
      await _reach(tester, find.text('Вход'));
      await tester.tap(find.widgetWithText(ChoiceChip, 'Вход'));
      await tester.pumpAndSettle();
      final email = find.byWidgetPredicate((widget) =>
          widget is TextField && widget.decoration?.labelText == 'Почта');
      await tester.ensureVisible(email);
      await tester.enterText(email, 'demo@example.invalid');
      FocusManager.instance.primaryFocus?.unfocus();
      await tester.pumpAndSettle();
      await _reach(tester, find.text('Войти'));
      await tester.tap(find.text('Войти'));
      await tester.pumpAndSettle();
      expect(find.text('Введите пароль.'), findsOneWidget);
      expect(find.text('Админ-панель'), findsNothing);
      expect(requests, 0);
      await _snapshot(tester, 'login-large-validation');
    },
        () => MockClient((_) async {
              requests++;
              return http.Response('{}', 500);
            }));
  });

  testWidgets(
      'offline profile clearly keeps a cached user rather than claiming a server session',
      (tester) async {
    SharedPreferences.setMockInitialValues({
      'arborscan_auth_token': 'synthetic-session',
      'arborscan_profile_logged_in': true,
      'arborscan_profile_name': 'DEMO сохранённый профиль',
      'arborscan_profile_email': 'offline@example.invalid',
      'arborscan_profile_role': 'user',
    });
    await http.runWithClient(() async {
      await _screen(tester, const ProfilePage(), 2);
      await tester.pumpAndSettle();
      await tester.scrollUntilVisible(find.text('Локальная копия'), 200,
          scrollable: _verticalScroll);
      expect(find.text('Локальная копия'), findsOneWidget);
      expect(find.text('Серверная сессия'), findsNothing);
      expect(find.text('Админ-панель'), findsNothing);
      await _snapshot(tester, 'profile-offline-cached');
    },
        () => MockClient((_) async {
              throw const SocketException('synthetic profile offline');
            }));
  });

  for (final scale in [1.6, 2.0]) {
    for (final role in ['user', 'admin']) {
      testWidgets('real $role profile and navigation at 360px / $scale text',
          (tester) async {
        final requests = <Uri>[];
        await http.runWithClient(() async {
          await _screen(tester, const ProfilePage(), scale);
          await tester.pumpAndSettle();
          expect(tester.takeException(), isNull);
          await tester.scrollUntilVisible(find.text('DEMO ArborScan'), 150,
              scrollable: _verticalScroll);
          expect(find.text('DEMO ArborScan'), findsOneWidget);
          await _snapshot(tester, 'profile-$role-$scale');
          await tester.scrollUntilVisible(find.text('Выйти'), 150,
              scrollable: _verticalScroll);
          for (final label in ['Выйти', 'Очистить локальную сессию']) {
            final paragraph =
                tester.renderObject<RenderParagraph>(find.text(label));
            final lines = paragraph.getBoxesForSelection(TextSelection(
                baseOffset: 0, extentOffset: label.split(' ').first.length));
            expect(lines.length, 1,
                reason:
                    '$label must remain readable without splitting the word into letters.');
          }
          if (role == 'admin') {
            await tester.ensureVisible(find.text('Проверка контуров'));
            await tester.pumpAndSettle();
            await _snapshot(tester, 'profile-admin-actions-$scale');
            await tester.tap(find.text('Проверка контуров'));
            await tester.pumpAndSettle();
            expect(find.byType(SavedCorrectionsPage), findsOneWidget);
            expect(find.text('Сохранённых контуров пока нет.'), findsOneWidget);
            expect(requests.any((uri) => uri.path.endsWith('/workflow/queue')),
                isTrue);
            await _snapshot(tester, 'queue-empty-$scale');
          } else {
            expect(find.text('Админ-панель'), findsNothing);
            expect(find.text('Проверка контуров'), findsNothing);
            await tester.scrollUntilVisible(find.text('Мои анализы'), 200,
                scrollable: _verticalScroll);
            await tester.pumpAndSettle();
            expect(find.text('Пока нет серверных анализов.'), findsOneWidget);
            await _snapshot(tester, 'profile-empty-stats-$scale');
          }
        },
            () => MockClient((request) async {
                  requests.add(request.url);
                  expect(request.headers['Authorization'],
                      'Bearer synthetic-session');
                  if (request.url.path.endsWith('/auth/me')) {
                    return http.Response(
                        jsonEncode({
                          'user': {
                            'id': _owner,
                            'name': 'DEMO ArborScan',
                            'email': 'demo@example.invalid',
                            'role': role,
                          }
                        }),
                        200,
                        headers: {
                          'content-type': 'application/json; charset=utf-8'
                        });
                  }
                  if (request.url.path.endsWith('/workflow/queue')) {
                    return http.Response(
                        '{"items":[],"next_offset":null}', 200);
                  }
                  return http.Response('{"stats":{"total_analyses":0}}', 200);
                }));
      });
    }

    testWidgets(
        'real moderation rejects empty reason, preserves long comment and refreshes decision / $scale',
        (tester) async {
      final decisions = <Map<String, dynamic>>[];
      var status = 'submitted';
      final service = CorrectionsService(
          clientFactory: () => MockClient((request) async {
                if (request.method == 'POST') {
                  final body = jsonDecode(request.body) as Map<String, dynamic>;
                  decisions.add(body);
                  status = body['decision'] as String;
                  return http.Response('{"saved":true}', 200);
                }
                return http.Response(
                    jsonEncode({
                      'analysis_id': 'DEMO-analysis',
                      'original_image_base64': _png,
                      'mask_png_base64': _png,
                      'review_status': status,
                      'decisions': decisions
                          .map((d) => {
                                'action': d['decision'],
                                'at': '2026-10-01T09:00:00Z',
                                'reason': d['reason'],
                                'actor_id': 'DEMO-admin',
                              })
                          .toList(),
                    }),
                    200,
                    headers: {
                      'content-type': 'application/json; charset=utf-8'
                    });
              }));
      await _screen(
          tester,
          SavedCorrectionsPage(
              service: service,
              correctionId: 'DEMO-contour',
              reviewOwner: _owner),
          scale);
      await tester.pumpAndSettle();
      await tester.scrollUntilVisible(find.text('Отклонить с причиной'), 250,
          scrollable: _verticalScroll);
      await tester.tap(find.text('Отклонить с причиной'));
      await tester.pumpAndSettle();
      await tester.scrollUntilVisible(
          find.text('Укажите причину отклонения.'), -250,
          scrollable: _verticalScroll);
      expect(find.text('Укажите причину отклонения.'), findsOneWidget);
      expect(decisions, isEmpty);
      await tester.scrollUntilVisible(find.byType(TextField), 250,
          scrollable: _verticalScroll);
      await tester.enterText(find.byType(TextField), _reason);
      await tester.ensureVisible(find.text('Отклонить с причиной'));
      await tester.pumpAndSettle();
      await _snapshot(tester, 'moderation-long-input-$scale');
      FocusManager.instance.primaryFocus?.unfocus();
      await tester.pumpAndSettle();
      await _reach(tester, find.text('Отклонить с причиной'));
      await tester.tap(find.text('Отклонить с причиной'));
      await tester.pumpAndSettle();
      expect(decisions.single, {'decision': 'rejected', 'reason': _reason});
      await tester.scrollUntilVisible(find.text('Статус: отклонён'), -200,
          scrollable: _verticalScroll);
      expect(find.text('Статус: отклонён'), findsOneWidget);
      await tester.scrollUntilVisible(find.text(_reason), -200,
          scrollable: _verticalScroll);
      await tester.pumpAndSettle();
      expect(find.text('Принять маску'), findsNothing);
      expect(find.text('Отклонить с причиной'), findsNothing);
      await _snapshot(tester, 'moderation-long-rejection-$scale');
      final paragraph =
          tester.renderObject<RenderParagraph>(find.text(_reason));
      final tail = paragraph
          .getBoxesForSelection(const TextSelection(
              baseOffset: _reason.length - 10, extentOffset: _reason.length))
          .last;
      // The lazy list estimates its extent until later children are built.
      // Real scroll gestures discover that extent and must reach the comment end.
      for (var attempts = 0;
          paragraph.localToGlobal(Offset(tail.right, tail.bottom)).dy > 744 &&
              attempts < 8;
          attempts++) {
        await tester.drag(_verticalScroll, const Offset(0, -180));
        await tester.pumpAndSettle();
      }
      final finalTail =
          paragraph.localToGlobal(Offset(tail.right, tail.bottom));
      expect(finalTail.dy, inInclusiveRange(56, 744),
          reason:
              'The end of a long rejection reason must be reachable by scrolling.');
      await _snapshot(tester, 'moderation-rejection-tail-$scale');
    });
  }

  for (final status in [403, 500]) {
    testWidgets(
        'real correction $status state can retry to empty list without exposing decision controls',
        (tester) async {
      var requests = 0;
      final service = CorrectionsService(
          clientFactory: () => MockClient((_) async {
                requests++;
                return requests == 1
                    ? http.Response('{}', status)
                    : http.Response('{"items":[],"next_offset":null}', 200);
              }));
      await _screen(
          tester, SavedCorrectionsPage(service: service, adminQueue: true), 2);
      await tester.pumpAndSettle();
      expect(
          find.textContaining(status == 403 ? 'Нет доступа' : 'Ошибка сервера'),
          findsOneWidget);
      expect(find.text('Принять маску'), findsNothing);
      await _snapshot(tester, 'queue-error-$status');
      await tester.tap(find.text('Повторить'));
      await tester.pumpAndSettle();
      expect(requests, 2);
      expect(find.text('Сохранённых контуров пока нет.'), findsOneWidget);
      expect(find.text('Повторить'), findsNothing);
    });
  }

  testWidgets(
      'real offline correction state and session invalidation clear old records',
      (tester) async {
    final service = CorrectionsService(
        clientFactory: () => MockClient((_) async {
              throw const SocketException('synthetic offline');
            }));
    await _screen(tester, SavedCorrectionsPage(service: service), 2);
    await tester.pumpAndSettle();
    expect(find.textContaining('Нет связи с сервером'), findsOneWidget);
    await _snapshot(tester, 'contours-offline');
    CorrectionsService.authChanges.value++;
    await tester.pumpAndSettle();
    expect(find.textContaining('Сессия изменилась'), findsOneWidget);
    expect(find.text('Повторить'), findsNothing);
    expect(find.byType(Image), findsNothing);
    await _snapshot(tester, 'contours-session-changed');
  });

  testWidgets('real loading disables refresh until response finishes',
      (tester) async {
    final response = Completer<http.Response>();
    var requests = 0;
    final service = CorrectionsService(
        clientFactory: () => MockClient((_) async {
              requests++;
              return response.future;
            }));
    await _screen(
        tester, SavedCorrectionsPage(service: service, adminQueue: true), 1.6);
    await tester.pump();
    await tester.pump();
    expect(find.byType(LinearProgressIndicator), findsOneWidget);
    final refresh = find.widgetWithIcon(IconButton, Icons.refresh);
    expect(tester.widget<IconButton>(refresh).onPressed, isNull);
    response.complete(http.Response('{"items":[],"next_offset":null}', 200));
    await tester.pumpAndSettle();
    expect(requests, 1);
    expect(find.byType(LinearProgressIndicator), findsNothing);
    expect(tester.widget<IconButton>(refresh).onPressed, isNotNull);
  });
}
