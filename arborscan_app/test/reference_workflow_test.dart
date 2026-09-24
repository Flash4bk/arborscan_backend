import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:arborscan_app/reference_measurement_page.dart';
import 'package:arborscan_app/contour_drafts.dart';

void main() {
  testWidgets(
      'EXIF photo through markup, report, save, cold reopen and recalculation',
      (t) async {
    const owner = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa';
    SharedPreferences.setMockInitialValues(
        {'arborscan_auth_token': 'test', 'arborscan_user_id': owner});
    final folder = (await t
        .runAsync(() => Directory.systemTemp.createTemp('reference-flow-')))!;
    final image = (await t.runAsync(
        () => File('test/fixtures/reference_exif6.jpg').readAsBytes()))!;
    final store = ContourDrafts(directory: () async => folder);
    Future<void> settle() async {
      for (var i = 0; i < 8; i++) {
        await t.runAsync(
            () async => Future<void>.delayed(const Duration(milliseconds: 25)));
        await t.pump(const Duration(milliseconds:100));
      }
    }

    Future<void> tapText(String text) async {
      await t.scrollUntilVisible(find.text(text), text.startsWith('Эталон и дерево') ? -150 : 150,
          scrollable: find.byType(Scrollable).first);
      await Scrollable.ensureVisible(t.element(find.text(text)), alignment: .5);
      await t.pumpAndSettle();
      await t.runAsync(() => t.tap(find.text(text)));
      await settle();
    }

    await t.pumpWidget(MaterialApp(
        home: ReferenceMeasurementPage(
            drafts: store, pickPhoto: () async => image)));
    await settle();
    await tapText('Выбрать фото');
    await t.runAsync(() => t.enterText(find.byType(TextField), '1'));
    await settle();
    await tapText('Обвести известный объект');
    final rect = t.getRect(find.byType(InteractiveViewer));
    for (final p in [
      const Offset(.1, .8),
      const Offset(.3, .8),
      const Offset(.3, .5),
      const Offset(.1, .5)
    ]) {
      await t
          .tapAt(rect.topLeft + Offset(rect.width * p.dx, rect.height * p.dy));
      await t.pump();
    }
    await t.tap(find.byTooltip('Замкнуть контур'));
    await t.pump();
    await t.runAsync(() => t.tap(find.byType(FloatingActionButton)));
    await settle();
    await t.ensureVisible(find.text('Подтвердить'));
    await t.pump();
    await t.runAsync(() => t.tap(find.text('Подтвердить')));
    await settle();
    Future<void> line(String label, Offset a, Offset b) async {
      await tapText(label);
      final box = t.getRect(find.byKey(const ValueKey('image-line-canvas')));
      for (final p in [a, b]) {
        await t
            .tapAt(box.topLeft + Offset(box.width * p.dx, box.height * p.dy));
        await t.pump();
      }
      await t.runAsync(() => t.tap(find.text('Применить отрезок')));
      await settle();
    }

    await line(
        'Отметить высоту эталона', const Offset(.2, .8), const Offset(.2, .6));
    await line(
        'Отметить высоту дерева', const Offset(.5, .9), const Offset(.5, .1));
    await line(
        'Отметить ширину кроны', const Offset(.2, .3), const Offset(.8, .3));
    await line('Низ живой кроны и верх', const Offset(.5,.5), const Offset(.5,.1));
    await line('Края ствола в выбранном сечении', const Offset(.45,.7), const Offset(.55,.7));
    await line('Ось участка ствола у сечения', const Offset(.5,.8), const Offset(.5,.6));
    await tapText(
        'Эталон и дерево примерно на одной глубине; перспектива мала');
    await tapText('Сохранить отчёт на устройстве');
    await t.scrollUntilVisible(find.textContaining('Высота:'), -200, scrollable: find.byType(Scrollable).first);
    expect(find.text('Высота: 4.00 м'), findsOneWidget);
    final rows = (await t.runAsync(() => store.list(owner)))!;
    final id = rows.single['draft_id'] as String;
    expect(rows.single['width'], 100);
    expect(rows.single['height'], 200);
    expect(rows.single['report']['height_m'], closeTo(4, 1e-8));
    expect(rows.single['report']['geometry']['crown_height']['value'], closeTo(2,1e-8));
    expect(rows.single['report']['geometry']['trunk_diameter']['value'], closeTo(.25,1e-8));
    await t.pumpWidget(const SizedBox());
    await t.runAsync(() => t.pumpWidget(MaterialApp(
        home: ReferenceMeasurementPage(
            draftId: id,
            drafts: ContourDrafts(directory: () async => folder)))));
    await settle();
    await t.scrollUntilVisible(find.byType(TextField), 150,
        scrollable: find.byType(Scrollable).first);
    await t.runAsync(() => t.enterText(find.byType(TextField), '100'));
    await settle();
    await t.tap(find.byType(DropdownButton<String>));
    await t.pumpAndSettle();
    await t.runAsync(() => t.tap(find.text('Сантиметры').last));
    await settle();
    await tapText('Сохранить отчёт на устройстве');
    await t.scrollUntilVisible(find.textContaining('Высота:'), -200, scrollable: find.byType(Scrollable).first);
    expect(find.text('Высота: 4.00 м'), findsOneWidget);
    await t.scrollUntilVisible(find.byType(TextField), -200,
        scrollable: find.byType(Scrollable).first);
    await t.runAsync(() => t.enterText(find.byType(TextField), '200'));
    await settle();
    await tapText('Сохранить отчёт на устройстве');
    await t.scrollUntilVisible(find.textContaining('Высота:'), -200, scrollable: find.byType(Scrollable).first);
    expect(find.text('Высота: 8.00 м'), findsOneWidget);
    final restored = (await t.runAsync(() => store.load(owner, id)))!;
    expect(restored['report']['height_m'], closeTo(8, 1e-8));
    expect(restored['report']['geometry']['trunk_diameter']['value'], closeTo(.5,1e-8));
    expect(restored['trunk_axis'], rows.single['trunk_axis']);
    await t.pumpWidget(const SizedBox());
    await t.runAsync(() => folder.delete(recursive: true));
  });
}
