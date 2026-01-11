import 'package:flutter/foundation.dart';

class AdminState extends ChangeNotifier {
  bool _isAdmin = false;

  bool get isAdmin => _isAdmin;

  void enable() {
    _isAdmin = true;
    notifyListeners();
  }

  void disable() {
    _isAdmin = false;
    notifyListeners();
  }

  void toggle() {
    _isAdmin = !_isAdmin;
    notifyListeners();
  }
}
