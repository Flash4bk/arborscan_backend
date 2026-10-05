import 'dart:convert';

import 'package:arborscan_app/corrections_service.dart';
import 'package:arborscan_app/app_theme.dart';
import 'package:arborscan_app/saved_corrections_page.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:shared_preferences/shared_preferences.dart';

const _png =
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=';

void main() {
  setUp(() => SharedPreferences.setMockInitialValues(
      {'arborscan_auth_token': 'synthetic-admin'}));

  http.Response response(Object body) => http.Response(jsonEncode(body), 200,
      headers: {'content-type': 'application/json; charset=utf-8'});

  testWidgets('Actual admin queue status is shown as submitted, never draft',
      (tester) async {
    var queueReads = 0;
    final service = CorrectionsService(
        clientFactory: () => MockClient((r) async {
              expect(r.headers['Authorization'], 'Bearer synthetic-admin');
              if (r.url.path.endsWith('/queue')) {
                queueReads++;
                return response({
                  'items': [
                    {
                      'owner_id': 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa',
                      'correction_id': 'submitted-revision',
                      'status': 'submitted',
                      'created_at': '2026-10-05T12:00:00Z',
                    }
                  ],
                  'next_offset': null,
                });
              }
              return response({
                'original_image_base64': _png,
                'mask_png_base64': _png,
                'review_status': 'submitted',
              });
            }));
    await tester.pumpWidget(MaterialApp(
        home: SavedCorrectionsPage(service: service, adminQueue: true)));
    await tester.pumpAndSettle();
    expect(queueReads, 1);
    expect(find.textContaining('отправлен на проверку'), findsOneWidget);
    final date = DateTime.parse('2026-10-05T12:00:00Z').toLocal();
    final shortDate = '05.10.2026 '
        '${date.hour.toString().padLeft(2, '0')}:00';
    expect(find.textContaining(shortDate), findsOneWidget);
    expect(find.textContaining('2026-10-05T12:00:00Z'), findsNothing);
    expect(find.textContaining('черновик'), findsNothing);
    expect(tester.takeException(), isNull);
  });

  testWidgets(
      'Old personal index does not invent draft; opened decisions remain readable',
      (tester) async {
    var status = 'accepted';
    final service = CorrectionsService(
        clientFactory: () => MockClient((r) async {
              if (r.url.path.endsWith('/v4/corrections')) {
                return response({
                  'items': [
                    {
                      'correction_id': 'old-index-revision',
                      'created_at': '2026-10-05T12:00:00Z',
                    }
                  ],
                  'next_offset': null,
                });
              }
              return response({
                'analysis_id': 'synthetic-analysis',
                'original_image_base64': _png,
                'mask_png_base64': _png,
                'review_status': status,
              });
            }));
    await tester
        .pumpWidget(MaterialApp(home: SavedCorrectionsPage(service: service)));
    await tester.pumpAndSettle();
    expect(
        find.textContaining('Статус доступен после открытия'), findsOneWidget);
    expect(find.textContaining('черновик'), findsNothing);
    await tester.tap(find.text('Сохранённый контур'));
    await tester.pumpAndSettle();
    expect(find.text('Статус: маска принята'), findsOneWidget);
    status = 'rejected';
    await tester.tap(find.byTooltip('Обновить'));
    await tester.pumpAndSettle();
    expect(find.text('Статус: отклонён'), findsOneWidget);
    expect(find.text('Статус: маска принята'), findsNothing);
    expect(tester.takeException(), isNull);
  });

  testWidgets(
      'Decision dates are compact at 200 percent, original timestamps and moderation controls stay accessible',
      (tester) async {
    tester.view.physicalSize = const Size(360, 800);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    const timestamp = '2026-10-05T17:57:26.760725+00:00';
    final service = CorrectionsService(
        clientFactory: () => MockClient((_) async => response({
              'analysis_id': 'synthetic-analysis',
              'created_at': timestamp,
              'original_image_base64': _png,
              'mask_png_base64': _png,
              'review_status': 'submitted',
              'decisions': [
                {
                  'action': 'rejected',
                  'at': timestamp,
                  'actor_id': 'synthetic-moderator',
                  'reason': 'DEMO причина отклонения для повторной правки',
                }
              ],
            })));
    await tester.pumpWidget(MaterialApp(
      theme: AppTheme.light(),
      builder: (context, child) => MediaQuery(
        data: MediaQuery.of(context)
            .copyWith(textScaler: const TextScaler.linear(2)),
        child: child!,
      ),
      home: SavedCorrectionsPage(
          service: service,
          correctionId: 'revision-with-decision',
          reviewOwner: 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa'),
    ));
    await tester.pumpAndSettle();
    final date = DateTime.parse(timestamp).toLocal();
    final shortDate = '05.10.2026 '
        '${date.hour.toString().padLeft(2, '0')}:${date.minute.toString().padLeft(2, '0')}';
    final title = find.text('отклонён · $shortDate');
    final scrollable = find.byType(Scrollable).first;
    await tester.scrollUntilVisible(title, 250, scrollable: scrollable);
    await tester.pumpAndSettle();
    expect(title, findsOneWidget);
    expect(find.textContaining(timestamp), findsNothing);
    await tester.tap(title);
    await tester.pumpAndSettle();
    await tester.scrollUntilVisible(
        find.text('Время решения (исходное): $timestamp'), 200,
        scrollable: scrollable);
    await tester.pumpAndSettle();
    expect(find.text('Время решения (исходное): $timestamp'), findsOneWidget);
    await tester.scrollUntilVisible(find.text('Отклонить с причиной'), 250,
        scrollable: scrollable);
    await tester.pumpAndSettle();
    expect(find.text('Принять маску'), findsOneWidget);
    expect(find.text('Отклонить с причиной'), findsOneWidget);
    expect(tester.takeException(), isNull);
  });

  testWidgets(
      'Explicit review_status takes precedence over the queue status field',
      (tester) async {
    final service = CorrectionsService(
        clientFactory: () => MockClient((r) async {
              if (r.url.path.endsWith('/v4/corrections')) {
                return response({
                  'items': [
                    {
                      'correction_id': 'reviewed-revision',
                      'review_status': 'accepted',
                      'status': 'submitted',
                    }
                  ],
                  'next_offset': null,
                });
              }
              return response({'original_image_base64': _png});
            }));
    await tester
        .pumpWidget(MaterialApp(home: SavedCorrectionsPage(service: service)));
    await tester.pumpAndSettle();
    expect(find.textContaining('принят'), findsOneWidget);
    expect(find.textContaining('отправлен на проверку'), findsNothing);
    expect(tester.takeException(), isNull);
  });
}
