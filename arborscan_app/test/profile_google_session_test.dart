// Real ProfilePage and google_sign_in 6.x orchestration; native transport and
// HTTP are synthetic. These are not evidence of a real Google OAuth login.
import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:arborscan_app/api_config.dart';
import 'package:arborscan_app/app_theme.dart';
import 'package:arborscan_app/contour_drafts.dart';
import 'package:arborscan_app/profile_page.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:google_sign_in_platform_interface/google_sign_in_platform_interface.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:shared_preferences/shared_preferences.dart';

const _ownerA = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa';
const _ownerB = 'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb';
const _draftKey = 'as16_synthetic_draft_owner_a';
const _expires = '2099-10-07T12:00:00Z';

class _NativeGoogle extends GoogleSignInPlatform {
  SignInInitParameters? initialized;
  int signIns = 0;
  int signOuts = 0;
  int tokenRequests = 0;
  Future<GoogleSignInUserData?> Function()? next;
  String? idToken = 'synthetic-google-id-token';

  GoogleSignInUserData account() => GoogleSignInUserData(
        email: 'google@example.invalid',
        id: 'synthetic-google-subject',
        displayName: 'Native Google',
        idToken: idToken,
      );

  @override
  Future<void> initWithParams(SignInInitParameters params) async {
    initialized = params;
  }

  @override
  Future<void> signOut() async {
    signOuts++;
  }

  @override
  Future<GoogleSignInUserData?> signIn() async {
    signIns++;
    if (next == null) return account();
    return await next!();
  }

  @override
  Future<GoogleSignInTokenData> getTokens({
    required String email,
    bool? shouldRecoverAuth,
  }) async {
    tokenRequests++;
    return GoogleSignInTokenData(idToken: idToken);
  }
}

Map<String, Object> _saved({
  String owner = _ownerA,
  String token = 'opaque-a',
}) =>
    {
      'arborscan_auth_token': token,
      'arborscan_auth_expires_at': _expires,
      'arborscan_user_id': owner,
      'arborscan_profile_logged_in': false,
      'arborscan_profile_name': 'Before',
      'arborscan_profile_email': 'before@example.invalid',
      'arborscan_profile_role': 'user',
      'arborscan_is_admin': false,
      _draftKey: 'synthetic unsaved contour state',
    };

Map<String, Object> _user(String owner, {String name = 'Server owner'}) => {
      'id': owner,
      'name': name,
      'email': 'server@example.invalid',
      'role': 'user',
    };

http.Response _session({String owner = _ownerA, String token = 'opaque-new'}) =>
    http.Response(
      jsonEncode({
        'ok': true,
        'token': token,
        'expires_at': _expires,
        'user': _user(owner),
      }),
      200,
    );

Future<void> _open(WidgetTester tester) async {
  await tester.pumpWidget(
    MaterialApp(theme: AppTheme.light(), home: const ProfilePage()),
  );
  await tester.pumpAndSettle();
}

Future<void> _google(WidgetTester tester) async {
  final button = find.widgetWithText(OutlinedButton, 'Войти через Google');
  await tester.ensureVisible(button);
  await tester.pumpAndSettle();
  await tester.tap(button);
  await tester.pump();
}

Future<void> _email(WidgetTester tester) async {
  final mode = find.widgetWithText(ChoiceChip, 'Вход');
  await tester.ensureVisible(mode);
  await tester.tap(mode);
  await tester.pumpAndSettle();
  final email = find.byWidgetPredicate(
    (w) => w is TextField && w.decoration?.labelText == 'Почта',
  );
  final password = find.byWidgetPredicate(
    (w) => w is TextField && w.decoration?.labelText == 'Пароль',
  );
  await tester.enterText(email, 'server@example.invalid');
  await tester.enterText(password, 'synthetic-password');
  final button = find.widgetWithText(FilledButton, 'Войти');
  await tester.ensureVisible(button);
  await tester.tap(button);
  await tester.pump();
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  late GoogleSignInPlatform previous;
  late _NativeGoogle native;
  setUp(() {
    previous = GoogleSignInPlatform.instance;
    native = _NativeGoogle();
    GoogleSignInPlatform.instance = native;
  });
  tearDown(() => GoogleSignInPlatform.instance = previous);

  testWidgets('Google cancellation preserves real owner-isolated disk drafts', (
    tester,
  ) async {
    SharedPreferences.setMockInitialValues(_saved());
    native.next = () async => null;
    final folder = (await tester.runAsync(
      () async =>
          Directory.systemTemp.createTemp('arborscan-google-draft-test-'),
    ))!;
    final original = Uint8List.fromList([1, 3, 5, 7, 9]);
    final state = {
      'analysis_id': 'synthetic-analysis',
      'editor_state': {
        'version': 1,
        'width': 20,
        'height': 40,
        'closed': false,
        'points': [
          {'x': 0.1, 'y': 0.2},
          {'x': 0.8, 'y': 0.9},
        ],
      },
    };
    try {
      await tester.runAsync(() async {
        final drafts = ContourDrafts(directory: () async => folder);
        await drafts.save(_ownerA, 'draft-a', state, original);
        await drafts.save(_ownerB, 'draft-b', {'owner': 'B'}, original);
      });
      await http.runWithClient(
        () async {
          await _open(tester);
          await _google(tester);
          await tester.pumpAndSettle();
          await tester.pumpWidget(const SizedBox());
          await tester.runAsync(() async {
            final recreated = ContourDrafts(directory: () async => folder);
            final draft = await recreated.load(_ownerA, 'draft-a');
            expect(draft?['editor_state'], state['editor_state']);
            expect(draft?['image'], original);
            expect(await recreated.load(_ownerB, 'draft-a'), isNull);
            expect((await recreated.load(_ownerB, 'draft-b'))?['owner'], 'B');
          });
        },
        () => MockClient(
          (request) async =>
              http.Response('{"detail":"synthetic unavailable"}', 503),
        ),
      );
    } finally {
      await tester.runAsync(() async {
        final absolute = await folder.resolveSymbolicLinks();
        final temp = await Directory.systemTemp.resolveSymbolicLinks();
        expect(absolute.startsWith('$temp${Platform.pathSeparator}'), isTrue);
        await folder.delete(recursive: true);
      });
    }
  });

  for (final status in [200, 401]) {
    testWidgets(
      'late auth/me $status cannot restore or clear another account',
      (tester) async {
        SharedPreferences.setMockInitialValues({
          ..._saved(),
          'arborscan_profile_logged_in': true,
          'arborscan_profile_role': 'admin',
          'arborscan_is_admin': true,
        });
        final reply = Completer<http.Response>();
        await http.runWithClient(
          () async {
            await tester.pumpWidget(
              MaterialApp(theme: AppTheme.light(), home: const ProfilePage()),
            );
            await tester.pump();
            final prefs = await SharedPreferences.getInstance();
            await prefs.setString('arborscan_auth_token', 'opaque-b');
            await prefs.setString('arborscan_user_id', _ownerB);
            await prefs.setString('arborscan_profile_name', 'Account B');
            await prefs.setString('arborscan_profile_role', 'user');
            await prefs.setBool('arborscan_is_admin', false);
            reply.complete(
              http.Response(
                jsonEncode(
                  status == 200
                      ? {'user': _user(_ownerA, name: 'Late account A')}
                      : {'detail': 'expired A'},
                ),
                status,
              ),
            );
            await tester.pumpAndSettle();
            expect(prefs.getString('arborscan_auth_token'), 'opaque-b');
            expect(prefs.getString('arborscan_user_id'), _ownerB);
            expect(prefs.getString('arborscan_profile_name'), 'Account B');
            expect(prefs.getBool('arborscan_profile_logged_in'), isTrue);
            expect(prefs.getBool('arborscan_is_admin'), isFalse);
            expect(find.text('Late account A'), findsNothing);
            expect(find.text('Before'), findsNothing);
            expect(find.text('Администратор'), findsNothing);
            expect(find.text('Админ-панель'), findsNothing);
            expect(find.text('Модели и данные'), findsNothing);
            expect(find.text('Проверка контуров'), findsNothing);
            expect(
              prefs.getString(_draftKey),
              'synthetic unsaved contour state',
            );
          },
          () => MockClient(
            (request) => request.url.path.endsWith('/auth/me')
                ? reply.future
                : Future.value(http.Response('{}', 200)),
          ),
        );
      },
    );
  }

  testWidgets('disposed email login never persists a late session', (
    tester,
  ) async {
    SharedPreferences.setMockInitialValues({_draftKey: 'draft survives'});
    final reply = Completer<http.Response>();
    var loginRequests = 0;
    await http.runWithClient(
      () async {
        await _open(tester);
        await _email(tester);
        await tester.pump();
        expect(loginRequests, 1);
        await tester.pumpWidget(const SizedBox());
        reply.complete(_session());
        await tester.pumpAndSettle();
        final prefs = await SharedPreferences.getInstance();
        expect(prefs.getString('arborscan_auth_token'), isNull);
        expect(prefs.getString('arborscan_user_id'), isNull);
        expect(prefs.getString(_draftKey), 'draft survives');
        expect(tester.takeException(), isNull);
      },
      () => MockClient((request) async {
        if (request.url.path.endsWith('/auth/login')) {
          loginRequests++;
          return reply.future;
        }
        return http.Response('{}', 200);
      }),
    );
  });

  for (final mode in [
    'cancel',
    'native-error',
    'missing-token',
    'offline',
    'bad-200',
  ]) {
    testWidgets('Google $mode preserves saved state and permits retry', (
      tester,
    ) async {
      final before = _saved();
      SharedPreferences.setMockInitialValues(before);
      if (mode == 'cancel') native.next = () async => null;
      if (mode == 'native-error') {
        native.next = () async => throw PlatformException(
              code: 'sign_in_failed',
              message: 'private-native-detail',
            );
      }
      if (mode == 'missing-token') native.idToken = null;
      var exchanges = 0;
      await http.runWithClient(
        () async {
          await _open(tester);
          await _google(tester);
          await tester.pumpAndSettle();
          final prefs = await SharedPreferences.getInstance();
          for (final item in before.entries) {
            expect(prefs.get(item.key), item.value, reason: item.key);
          }
          final button = find.widgetWithText(
            OutlinedButton,
            'Войти через Google',
          );
          expect(tester.widget<OutlinedButton>(button).onPressed, isNotNull);
          expect(find.textContaining('private-native-detail'), findsNothing);
          native.next = null;
          native.idToken = 'synthetic-google-id-token';
          await _google(tester);
          await tester.pumpAndSettle();
          expect(native.signIns, 2);
          expect(prefs.getString('arborscan_user_id'), _ownerA);
          expect(prefs.getString('arborscan_auth_token'), 'opaque-new');
          expect(prefs.getString(_draftKey), before[_draftKey]);
        },
        () => MockClient((request) async {
          if (request.url.path.endsWith('/auth/me')) {
            return http.Response('{"detail":"unavailable"}', 503);
          }
          if (request.url.path.endsWith('/auth/google')) {
            exchanges++;
            if (exchanges == 1 && mode == 'offline') {
              throw http.ClientException('synthetic offline');
            }
            if (exchanges == 1 && mode == 'bad-200') {
              return http.Response('{}', 200);
            }
            return _session();
          }
          return http.Response('{}', 200);
        }),
      );
    });
  }

  testWidgets('repeated Google tap starts only one native flow and exchange', (
    tester,
  ) async {
    SharedPreferences.setMockInitialValues({_draftKey: 'draft survives'});
    final nativeReply = Completer<GoogleSignInUserData?>();
    native.next = () => nativeReply.future;
    var exchanges = 0;
    await http.runWithClient(
      () async {
        await _open(tester);
        await _google(tester);
        await tester.pump();
        await tester.tap(
          find.widgetWithText(OutlinedButton, 'Войти через Google'),
          warnIfMissed: false,
        );
        await tester.pump();
        expect(native.signIns, 1);
        nativeReply.complete(native.account());
        await tester.pumpAndSettle();
        expect(native.signIns, 1);
        expect(exchanges, 1);
        expect(native.initialized?.serverClientId, ApiConfig.googleWebClientId);
        expect(native.initialized?.scopes, ['email', 'profile']);
        expect(native.initialized?.clientId, isNull);
      },
      () => MockClient((request) async {
        if (request.url.path.endsWith('/auth/google')) {
          exchanges++;
          final body = jsonDecode(request.body) as Map;
          expect(body.keys, unorderedEquals(['id_token']));
          expect(body['id_token'], 'synthetic-google-id-token');
          return _session();
        }
        return http.Response('{}', 200);
      }),
    );
  });

  for (final mode in ['google', 'email']) {
    testWidgets('$mode session restores same owner after screen recreation', (
      tester,
    ) async {
      SharedPreferences.setMockInitialValues({_draftKey: 'draft survives'});
      var meRequests = 0;
      await http.runWithClient(
        () async {
          await _open(tester);
          if (mode == 'google') {
            await _google(tester);
          } else {
            await _email(tester);
          }
          await tester.pumpAndSettle();
          final prefs = await SharedPreferences.getInstance();
          expect(prefs.getString('arborscan_user_id'), _ownerA);
          expect(prefs.getString('arborscan_auth_token'), 'opaque-new');
          expect(prefs.getString('arborscan_auth_expires_at'), _expires);
          await tester.pumpWidget(const SizedBox());
          await tester.pumpAndSettle();
          await _open(tester);
          expect(meRequests, 1);
          expect(prefs.getString('arborscan_user_id'), _ownerA);
          expect(prefs.getString('arborscan_auth_token'), 'opaque-new');
          expect(prefs.getString('arborscan_auth_expires_at'), _expires);
          expect(prefs.getString(_draftKey), 'draft survives');
          expect(find.text('Сессия подтверждена сервером.'), findsOneWidget);
        },
        () => MockClient((request) async {
          if (request.url.path.endsWith('/auth/google') ||
              request.url.path.endsWith('/auth/login')) {
            return _session();
          }
          if (request.url.path.endsWith('/auth/me')) {
            meRequests++;
            expect(request.headers['Authorization'], 'Bearer opaque-new');
            return http.Response(
              jsonEncode({'ok': true, 'user': _user(_ownerA)}),
              200,
            );
          }
          return http.Response('{}', 200);
        }),
      );
    });
  }
}
