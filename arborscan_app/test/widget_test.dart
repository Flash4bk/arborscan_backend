import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:arborscan_app/main.dart';
import 'package:arborscan_app/splash_screen_new.dart';

void main() {
  testWidgets('ArborScan opens its splash screen', (WidgetTester tester) async {
    try {
      await tester.pumpWidget(const ArborScanApp());
      await tester.pump(const Duration(milliseconds: 100));

      expect(find.byType(ArborScanApp), findsOneWidget);
      expect(find.byType(MaterialApp), findsOneWidget);
      expect(find.byType(SplashScreen), findsOneWidget);
      expect(find.byType(Scaffold), findsOneWidget);
      expect(tester.takeException(), isNull);
    } finally {
      // Dispose the splash before its delayed navigation to AppRoot.
      // The existing splash has a 2800 ms timer with a mounted guard.
      // Advance the fake clock after disposal to leave no pending timer.
      await tester.pumpWidget(const SizedBox.shrink());
      await tester.pump(const Duration(seconds: 3));
    }
    expect(tester.takeException(), isNull);
  });
}
