import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:arborscan_app/app_theme.dart';
import 'package:arborscan_app/analyze_page.dart';

void main() {
  test('Text, status and action colors meet normal text contrast', () {
    double contrast(Color a, Color b) {
      final x = a.computeLuminance(), y = b.computeLuminance();
      return x > y ? (x + .05) / (y + .05) : (y + .05) / (x + .05);
    }

    for (final fg in [
      AppTheme.text,
      AppTheme.muted,
      AppTheme.primary,
      AppTheme.danger,
      AppTheme.warning,
      AppTheme.success
    ]) {
      expect(contrast(fg, AppTheme.background), greaterThanOrEqualTo(4.5));
    }
    expect(contrast(Colors.white, AppTheme.primary), greaterThanOrEqualTo(4.5));
  });
  testWidgets('Long shared actions and status wrap at 320px and 200% font',
      (t) async {
    t.view.physicalSize = const Size(320, 800);
    t.view.devicePixelRatio = 1;
    addTearDown(t.view.resetPhysicalSize);
    addTearDown(t.view.resetDevicePixelRatio);
    var calls = 0;
    await t.pumpWidget(MaterialApp(
        theme: AppTheme.light(),
        home: MediaQuery(
            data: const MediaQueryData(textScaler: TextScaler.linear(2)),
            child: Scaffold(
                body: SingleChildScrollView(
                    child: Column(children: [
              AppActionButton(
                  expanded: true,
                  primary: true,
                  label: 'Сохранить исправленный контур в аккаунте',
                  onPressed: () => calls++),
              Ui.badge(
                  text: 'Отклонено: необходимо исправить границу дерева',
                  color: AppTheme.danger),
              AppActionButton(
                  expanded: true,
                  loading: true,
                  label: 'Сохранение',
                  onPressed: () => calls++),
            ]))))));
    expect(t.takeException(), isNull);
    await t.tap(find.text('Сохранить исправленный контур в аккаунте'));
    expect(calls, 1);
    await t.ensureVisible(find.text('Сохранение'));
    await t.tap(find.text('Сохранение'));
    expect(calls, 1);
  });
  testWidgets('Home actions fit small display at large system font', (t) async {
    SharedPreferences.setMockInitialValues({});
    t.view.physicalSize = const Size(320, 800);
    t.view.devicePixelRatio = 1;
    addTearDown(t.view.resetPhysicalSize);
    addTearDown(t.view.resetDevicePixelRatio);
    await t.pumpWidget(MaterialApp(
        theme: AppTheme.light(),
        home: MediaQuery(
            data: const MediaQueryData(textScaler: TextScaler.linear(1.8)),
            child: ArborScanPage())));
    await t.pump();
    expect(t.takeException(), isNull);
    await t.scrollUntilVisible(find.text('Галерея'), 200);
    expect(t.takeException(), isNull);
    expect(t.getSize(find.text('Галерея')).width, lessThan(290));
    await t.pumpWidget(const SizedBox.shrink());
  });
}
