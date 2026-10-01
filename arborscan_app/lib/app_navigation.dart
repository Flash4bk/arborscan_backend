import 'package:flutter/foundation.dart';

/// Refresh data when a retained tab is revisited without discarding its inputs.
class AppNavigation {
  static final historyVisits = ValueNotifier<int>(0);
  static final mapVisits = ValueNotifier<int>(0);
  static final profileVisits = ValueNotifier<int>(0);
  static void activate(int index) {
    if (index == 1) historyVisits.value++;
    if (index == 2) mapVisits.value++;
    if (index == 3) profileVisits.value++;
  }
}
