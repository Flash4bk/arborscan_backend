import 'dart:convert';

import 'package:arborscan_app/corrections_service.dart';
import 'package:arborscan_app/saved_corrections_page.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:shared_preferences/shared_preferences.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  setUp(() => SharedPreferences.setMockInitialValues(
      {'arborscan_auth_token': 'listing-owner'}));

  testWidgets('empty filtered page still loads and opens next committed record',
      (tester) async {
    const png =
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=';
    final offsets = <String?>[];
    final openedIds = <String>[];
    final service = CorrectionsService(
        clientFactory: () => MockClient((request) async {
              expect(request.headers['Authorization'], 'Bearer listing-owner');
              if (request.url.path.endsWith('/committed-id')) {
                openedIds.add(request.url.pathSegments.last);
                return http.Response(
                    jsonEncode({
                      'analysis_id': 'demo-analysis',
                      'original_image_base64': png,
                      'mask_png_base64': png,
                      'review_status': 'draft',
                    }),
                    200);
              }
              final offset = request.url.queryParameters['offset'];
              offsets.add(offset);
              return http.Response(
                  jsonEncode(offset == '0'
                      ? {'items': [], 'next_offset': 50}
                      : {
                          'items': [
                            {
                              'correction_id': 'committed-id',
                              'created_at': '2026-10-06T12:00:00Z',
                            }
                          ],
                          'next_offset': null,
                        }),
                  200);
            }));

    await tester
        .pumpWidget(MaterialApp(home: SavedCorrectionsPage(service: service)));
    await tester.pumpAndSettle();

    expect(find.text('Сохранённых контуров пока нет.'), findsNothing);
    expect(find.text('На этой странице нет записей. Загрузите следующую.'),
        findsOneWidget);
    await tester.tap(find.text('Загрузить ещё'));
    await tester.pumpAndSettle();
    expect(offsets, ['0', '50']);
    expect(find.text('Загрузить ещё'), findsNothing);
    expect(find.text('Загружено записей: 1'), findsOneWidget);

    await tester.tap(find.text('Сохранённый контур'));
    await tester.pumpAndSettle();
    expect(find.byType(Image), findsOneWidget);
    expect(find.text('Оригинал'), findsOneWidget);
    expect(find.text('PNG-маска'), findsOneWidget);
    expect(openedIds, isNotEmpty);
    expect(openedIds.every((id) => id == 'committed-id'), isTrue);
    expect(tester.takeException(), isNull);
  });
}
