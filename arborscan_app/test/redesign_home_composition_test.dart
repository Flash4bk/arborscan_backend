import 'dart:io';
import 'dart:ui' as ui;

import 'package:arborscan_app/analyze_page.dart';
import 'package:arborscan_app/app_theme.dart';
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

void main() {
  setUpAll(() async {
    final font = FontLoader('AS15HomeEvidence')
      ..addFont(rootBundle.load('assets/fonts/DejaVuSans.ttf'));
    await font.load();
    final icons = FontLoader('MaterialIcons')
      ..addFont(rootBundle.load('fonts/MaterialIcons-Regular.otf'));
    await icons.load();
  });
  for (final scale in [1.0, 2.0]) {
    testWidgets(
        'Home composition exposes help and actions at ${scale * 100}% text with reduced motion',
        (tester) async {
      SharedPreferences.setMockInitialValues({});
      tester.view.physicalSize = const Size(360, 800);
      tester.view.devicePixelRatio = 1;
      addTearDown(tester.view.resetPhysicalSize);
      addTearDown(tester.view.resetDevicePixelRatio);
      final capture = GlobalKey();
      final theme = AppTheme.light();
      ButtonStyle withEvidenceFont(ButtonStyle? style) =>
          (style ?? const ButtonStyle()).copyWith(
            textStyle: WidgetStatePropertyAll(
                (style?.textStyle?.resolve({}) ?? const TextStyle())
                    .copyWith(fontFamily: 'AS15HomeEvidence')),
          );
      var profileVisits = 0;
      await tester.pumpWidget(MaterialApp(
        theme: theme.copyWith(
          filledButtonTheme: FilledButtonThemeData(
              style: withEvidenceFont(theme.filledButtonTheme.style)),
          outlinedButtonTheme: OutlinedButtonThemeData(
              style: withEvidenceFont(theme.outlinedButtonTheme.style)),
          textButtonTheme: TextButtonThemeData(
              style: withEvidenceFont(theme.textButtonTheme.style)),
          textTheme: theme.textTheme.apply(fontFamily: 'AS15HomeEvidence'),
          appBarTheme: theme.appBarTheme.copyWith(
              titleTextStyle: theme.appBarTheme.titleTextStyle
                  ?.copyWith(fontFamily: 'AS15HomeEvidence')),
        ),
        builder: (context, child) => MediaQuery(
          data: MediaQuery.of(context).copyWith(
              textScaler: TextScaler.linear(scale), disableAnimations: true),
          child: RepaintBoundary(key: capture, child: child!),
        ),
        home: ArborScanPage(onOpenProfile: () => profileVisits++),
      ));
      await tester.runAsync(
          () async => Future<void>.delayed(const Duration(milliseconds: 120)));
      await tester.pumpAndSettle();
      expect(tester.takeException(), isNull);
      expect(
          tester
              .widget<AnimatedSwitcher>(find.byType(AnimatedSwitcher).first)
              .duration,
          Duration.zero);
      expect(find.textContaining('AR оценивает'), findsNothing);
      await tester.tap(find.byTooltip('Профиль'));
      expect(profileVisits, 1);
      await tester.pumpAndSettle();
      await tester.runAsync(() async {
        final boundary = capture.currentContext!.findRenderObject()!
            as RenderRepaintBoundary;
        final image = await boundary.toImage(pixelRatio: 2);
        final png = await image.toByteData(format: ui.ImageByteFormat.png);
        final output = Directory('../output/as15/concept-widget');
        await output.create(recursive: true);
        await File('${output.path}/home-empty-${(scale * 100).toInt()}.png')
            .writeAsBytes(png!.buffer.asUint8List());
        image.dispose();
      });
      final help = find.byTooltip('Как выбрать способ измерения');
      await tester.scrollUntilVisible(help, 180,
          scrollable: find.byType(Scrollable).first);
      await tester.tap(help);
      await tester.pumpAndSettle();
      expect(find.text('Способы измерения'), findsOneWidget);
      expect(find.textContaining('Можно начать до выбора'), findsOneWidget);
      expect(tester.takeException(), isNull);
      await tester.pumpWidget(const SizedBox.shrink());
    });
  }
}
