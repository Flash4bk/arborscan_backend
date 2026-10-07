import 'dart:io';

import 'package:arborscan_app/analyze_page.dart';
import 'package:arborscan_app/app_theme.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

void main() {
  for (final unavailable in [false, true]) {
    testWidgets(
        'AR ${unavailable ? 'unavailable' : 'cancel'} preserves selected photo',
        (tester) async {
      SharedPreferences.setMockInitialValues({});
      const picker = MethodChannel('plugins.flutter.io/image_picker');
      const availability = MethodChannel('arborscan/image_selection');
      const ar = MethodChannel('arborscan/ar_measure');
      final photo = File('test/fixtures/reference_exif6.jpg').absolute.path;
      final messenger = tester.binding.defaultBinaryMessenger;
      var arCalls = 0;
      messenger.setMockMethodCallHandler(picker, (_) async => photo);
      messenger.setMockMethodCallHandler(availability, (_) async => true);
      messenger.setMockMethodCallHandler(ar, (_) async {
        arCalls++;
        if (unavailable) {
          throw PlatformException(
              code: 'ar_unavailable',
              message:
                  'AR недоступен на этом устройстве. Можно продолжить анализ фото.');
        }
        return null;
      });
      addTearDown(() {
        messenger.setMockMethodCallHandler(picker, null);
        messenger.setMockMethodCallHandler(availability, null);
        messenger.setMockMethodCallHandler(ar, null);
      });
      await tester.pumpWidget(
          MaterialApp(theme: AppTheme.light(), home: const ArborScanPage()));
      await tester.runAsync(() => tester.tap(find.text('Галерея')));
      await tester.runAsync(
          () async => Future<void>.delayed(const Duration(milliseconds: 30)));
      await tester.pumpAndSettle();
      expect(find.byTooltip('Начать заново'), findsOneWidget);
      final selected = tester.widget<Image>(find.byType(Image).first).image;
      await tester.scrollUntilVisible(
          find.byKey(const ValueKey('open-ar')), 250,
          scrollable: find.byType(Scrollable).first);
      await tester.pumpAndSettle();
      await tester.runAsync(() async {
        await tester.tap(find.byKey(const ValueKey('open-ar')));
        await Future<void>.delayed(const Duration(milliseconds: 200));
      });
      await tester.pump(const Duration(milliseconds: 300));
      await tester.pump(const Duration(milliseconds: 300));
      expect(find.text('Связать AR с выбранным фото'), findsOneWidget);
      await tester.runAsync(() async {
        await tester.tap(find.text('Это то же дерево'));
        await Future<void>.delayed(const Duration(milliseconds: 50));
      });
      await tester.pumpAndSettle();
      expect(arCalls, 1);
      expect(find.textContaining('AR недоступен на этом устройстве'),
          unavailable ? findsOneWidget : findsNothing);
      await tester.drag(find.byType(ListView).first, const Offset(0, 2000));
      await tester.pumpAndSettle();
      expect(tester.widget<Image>(find.byType(Image).first).image, selected);
      expect(find.byTooltip('Начать заново'), findsOneWidget);
      expect(find.textContaining('PlatformException'), findsNothing);
      expect(tester.takeException(), isNull);
    });
  }
}
