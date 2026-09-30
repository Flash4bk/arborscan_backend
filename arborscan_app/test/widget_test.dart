import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:arborscan_app/main.dart';
import 'package:arborscan_app/app_root.dart';

void main() {
  testWidgets('Launch reaches the home without an artificial delay',
      (tester) async {
    SharedPreferences.setMockInitialValues({});
    await tester.pumpWidget(const ArborScanApp());
    await tester.pump(const Duration(milliseconds: 250));
    await tester.pump();
    expect(find.byType(AppRoot), findsOneWidget);
    expect(find.text('Камера'), findsOneWidget);
    expect(find.text('Галерея'), findsOneWidget);
    expect(tester.takeException(), isNull);
    await tester.pumpWidget(const SizedBox.shrink());
  });
}
