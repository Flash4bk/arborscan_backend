// Direct session queue tests run in their own isolate. Widget tests use
// separate FakeAsync zones and must not inherit this service's real-zone queue.
import 'dart:async';
import 'package:arborscan_app/profile_session_guard.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

const _ownerA = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa';
const _ownerB = 'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  test(
    'logout follows an already started session commit without losing logout',
    () async {
      SharedPreferences.setMockInitialValues({
        'arborscan_auth_token': 'opaque-a',
        'arborscan_user_id': _ownerA,
      });
      final started = Completer<void>();
      final release = Completer<void>();
      final oldTicket = await ProfileSessionGuard.capture();
      final oldWrite = ProfileSessionGuard.write(oldTicket, (prefs) async {
        started.complete();
        await release.future;
        await prefs.setString('arborscan_auth_token', 'opaque-a-updated');
      }, active: () => true);
      await started.future;
      final logout = () async {
        final ticket = await ProfileSessionGuard.capture(supersede: true);
        return ProfileSessionGuard.write(ticket, (prefs) async {
          await prefs.remove('arborscan_auth_token');
          await prefs.remove('arborscan_user_id');
        }, active: () => true);
      }();
      // Let the snapshot request run while the old commit is deliberately held.
      await Future<void>.delayed(Duration.zero);
      release.complete();
      await oldWrite;
      expect(await logout, isTrue);
      final prefs = await SharedPreferences.getInstance();
      expect(prefs.getString('arborscan_auth_token'), isNull);
      expect(prefs.getString('arborscan_user_id'), isNull);
    },
  );

  test(
    'failed profile commit does not poison later session operations',
    () async {
      SharedPreferences.setMockInitialValues({});
      final failed = await ProfileSessionGuard.capture();
      await expectLater(
        ProfileSessionGuard.write(failed, (_) async {
          throw StateError('synthetic write failure');
        }, active: () => true),
        throwsStateError,
      );
      final next = await ProfileSessionGuard.capture(supersede: true);
      expect(
        await ProfileSessionGuard.write(next, (prefs) async {
          await prefs.setString('arborscan_auth_token', 'opaque-b');
          await prefs.setString('arborscan_user_id', _ownerB);
        }, active: () => true),
        isTrue,
      );
    },
  );

  test('another owner makes a captured session write stale', () async {
    SharedPreferences.setMockInitialValues({
      'arborscan_auth_token': 'opaque-a',
      'arborscan_user_id': _ownerA,
    });
    final ticket = await ProfileSessionGuard.capture();
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('arborscan_auth_token', 'opaque-b');
    await prefs.setString('arborscan_user_id', _ownerB);
    var committed = false;
    expect(
      await ProfileSessionGuard.write(ticket, (_) async {
        committed = true;
      }, active: () => true),
      isFalse,
    );
    expect(committed, isFalse);
    expect(prefs.getString('arborscan_auth_token'), 'opaque-b');
    expect(prefs.getString('arborscan_user_id'), _ownerB);
  });

  test('authenticated legacy owner hydration keeps the same session', () async {
    SharedPreferences.setMockInitialValues({
      'arborscan_auth_token': 'opaque-a',
    });
    final ticket = await ProfileSessionGuard.capture();
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('arborscan_user_id', _ownerA);
    expect(
        await ProfileSessionGuard.current(ticket, active: () => true), isFalse);
    expect(
      await ProfileSessionGuard.write(ticket, (preferences) async {
        await preferences.setString('arborscan_profile_name', 'Authenticated');
      }, active: () => true, authenticatedOwner: _ownerA),
      isTrue,
    );
    expect(prefs.getString('arborscan_auth_token'), 'opaque-a');
    expect(prefs.getString('arborscan_user_id'), _ownerA);
    expect(prefs.getString('arborscan_profile_name'), 'Authenticated');
  });

  for (final change in [
    'wrong-authenticated-owner',
    'different-token',
    'new-revision',
    'missing-token',
    'known-owner-changed',
    'no-authenticated-response',
  ]) {
    test('legacy owner hydration rejects $change', () async {
      SharedPreferences.setMockInitialValues({
        if (change != 'missing-token') 'arborscan_auth_token': 'opaque-a',
        if (change == 'known-owner-changed') 'arborscan_user_id': _ownerA,
      });
      final ticket = await ProfileSessionGuard.capture();
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString('arborscan_user_id',
          change == 'known-owner-changed' ? _ownerB : _ownerA);
      if (change == 'different-token') {
        await prefs.setString('arborscan_auth_token', 'opaque-b');
      } else if (change == 'new-revision') {
        await ProfileSessionGuard.capture(supersede: true);
      }
      final authenticatedOwner = change == 'no-authenticated-response'
          ? null
          : (change == 'wrong-authenticated-owner' ||
                  change == 'known-owner-changed'
              ? _ownerB
              : _ownerA);
      var committed = false;
      expect(
        await ProfileSessionGuard.write(ticket, (_) async {
          committed = true;
        }, active: () => true, authenticatedOwner: authenticatedOwner),
        isFalse,
      );
      expect(committed, isFalse);
      expect(prefs.getString('arborscan_user_id'),
          change == 'known-owner-changed' ? _ownerB : _ownerA);
      expect(
          prefs.getString('arborscan_auth_token'),
          change == 'missing-token'
              ? null
              : (change == 'different-token' ? 'opaque-b' : 'opaque-a'));
    });
  }
}
