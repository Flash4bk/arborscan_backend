import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:arborscan_app/profile_page.dart';
import 'package:arborscan_app/saved_corrections_page.dart';

void main() {
  for (final role in ['admin', 'user']) {
    testWidgets('profile $role exposes only appropriate moderation navigation', (tester) async {
      SharedPreferences.setMockInitialValues({
        'arborscan_auth_token': 'test-session',
        'arborscan_profile_logged_in': true,
      });
      final requests = <Uri>[];
      await http.runWithClient(() async {
        await tester.pumpWidget(const MaterialApp(home: ProfilePage()));
        await tester.pumpAndSettle();
        if (role == 'user') {
          expect(find.text('Админ-панель'), findsNothing);
          expect(find.text('Проверка контуров'), findsNothing);
          return;
        }
        expect(find.text('Админ-панель'), findsOneWidget);
        await tester.ensureVisible(find.text('Проверка контуров'));
        await tester.pumpAndSettle();
        await tester.tap(find.text('Проверка контуров'));
        await tester.pumpAndSettle();
        expect(tester.widget<SavedCorrectionsPage>(find.byType(SavedCorrectionsPage)).adminQueue, isTrue);
        expect(requests.any((uri) => uri.path == '/api/v4/v4/corrections/workflow/queue'), isTrue);
        expect(requests.any((uri) => uri.path.contains('/admin/')), isFalse);
      }, () => MockClient((request) async {
        requests.add(request.url);
        expect(request.headers['Authorization'], 'Bearer test-session');
        if (request.url.path.endsWith('/auth/me')) {
          return http.Response(jsonEncode({'user': {'id':'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa',
            'name':'Test', 'email':'test@example.invalid', 'role':role}}), 200);
        }
        if (request.url.path.endsWith('/workflow/queue')) {
          return http.Response('{"items":[],"next_offset":null}', 200);
        }
        return http.Response('{}', 200);
      }));
    });
  }
}
