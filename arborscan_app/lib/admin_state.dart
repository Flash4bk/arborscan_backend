import 'package:flutter/foundation.dart';

/// Локальное представление роли пользователя.
///
/// Оно не выдаёт права самостоятельно: роль должна приходить от backend.
class AdminState extends ChangeNotifier {
  String _role = 'user';

  String get role => _role;
  bool get isAdmin => _role == 'admin';

  void applyServerRole(String? role) {
    final normalized = (role ?? 'user').trim().toLowerCase();
    final nextRole = normalized == 'admin' ? 'admin' : 'user';
    if (_role == nextRole) return;
    _role = nextRole;
    notifyListeners();
  }

  void clear() {
    if (_role == 'user') return;
    _role = 'user';
    notifyListeners();
  }
}
