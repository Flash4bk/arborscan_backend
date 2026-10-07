import 'package:shared_preferences/shared_preferences.dart';

class ProfileSessionTicket {
  const ProfileSessionTicket(this.revision, this.token, this.owner);

  final int revision;
  final String token;
  final String owner;
}

/// Serializes profile session writes and rejects replies belonging to an old
/// operation, account or disposed screen. It does not replace server auth.
class ProfileSessionGuard {
  static int _revision = 0;
  static Future<void>? _writes;

  static Future<ProfileSessionTicket> capture({bool supersede = false}) async {
    final revision = supersede ? ++_revision : _revision;
    // A write that already started must finish before the next operation takes
    // its snapshot. Invalidate pending replies immediately, then join the queue.
    final pending = _writes;
    if (pending != null) await pending;
    final prefs = await SharedPreferences.getInstance();
    return ProfileSessionTicket(
        revision,
        prefs.getString('arborscan_auth_token') ?? '',
        prefs.getString('arborscan_user_id') ?? '');
  }

  static Future<bool> current(ProfileSessionTicket ticket,
      {required bool Function() active, String? authenticatedOwner}) async {
    if (!active() || ticket.revision != _revision) return false;
    final prefs = await SharedPreferences.getInstance();
    final owner = prefs.getString('arborscan_user_id') ?? '';
    // An authenticated /auth/me reply may agree with a concurrent bootstrap
    // of a legacy empty owner. It never permits a different known owner,
    // missing session, changed token or superseded operation.
    final ownerMatches = owner == ticket.owner ||
        (ticket.owner.isEmpty &&
            ticket.token.isNotEmpty &&
            authenticatedOwner != null &&
            authenticatedOwner.isNotEmpty &&
            owner == authenticatedOwner);
    return active() &&
        ticket.revision == _revision &&
        (prefs.getString('arborscan_auth_token') ?? '') == ticket.token &&
        ownerMatches;
  }

  static Future<bool> write(ProfileSessionTicket ticket,
      Future<void> Function(SharedPreferences) commit,
      {required bool Function() active, String? authenticatedOwner}) {
    final previous = _writes;
    final result = () async {
      if (previous != null) await previous;
      if (!await current(ticket,
          active: active, authenticatedOwner: authenticatedOwner)) return false;
      await commit(await SharedPreferences.getInstance());
      return true;
    }();
    late final Future<void> queued;
    void release() {
      if (identical(_writes, queued)) _writes = null;
    }

    queued = result.then<void>((_) => release(),
        onError: (Object _, StackTrace __) => release());
    _writes = queued;
    return result;
  }
}
