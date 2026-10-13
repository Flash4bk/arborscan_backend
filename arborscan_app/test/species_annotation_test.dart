import 'package:arborscan_app/species_annotation.dart';
import 'package:arborscan_app/report_export_data.dart';
import 'package:arborscan_app/geometry_report.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test(
      'manual species keeps prediction and dimensions unchanged and reaches export',
      () {
    final report = <String, dynamic>{
      'species': {'display_name': 'Предсказание модели', 'confidence': .6},
      'height_m': 4.0,
      'geometry': {
        'crown_height': {'value': 2.0, 'unit': 'm'}
      },
    };
    final amended = withSpeciesAnnotation(report, '  Сосна обыкновенная  ');
    expect(speciesDisplayName(amended), 'Сосна обыкновенная');
    expect(amended['species'], report['species']);
    expect(amended['height_m'], report['height_m']);
    expect(amended['geometry'], report['geometry']);
    expect(report.containsKey('species_annotation'), false);
    expect(amended['species_annotation']['confirmed'], false);
    final export = ReportExportData(
        snapshot: {'report': amended}, record: {}, local: true);
    expect(export.speciesLines.join('\n'), contains('Сосна обыкновенная'));
    expect(export.speciesLines.join('\n'), contains('Предсказание модели'));
    expect(export.speciesLines.join('\n'), contains('не подтверждено'));
    expect(withSpeciesAnnotation(amended, ''), report);
    expect(speciesDisplayName({'species': 'Старый вид'}), 'Старый вид');
    expect(speciesDisplayName({}), 'Вид не указан');
  });

  testWidgets(
      'compact geometry lists measured values only; details remain available',
      (t) async {
    await t.pumpWidget(const MaterialApp(
        home: Scaffold(
            body: Column(children: [
      GeometryReport(compact: true, data: {
        'crown_height': {'value': 2, 'unit': 'm'},
        'trunk_lean': {'value': double.nan, 'unit': 'deg'},
      }),
      ExpansionTile(
          title: Text('Расширенный анализ'),
          children: [GeometryReport(data: {})]),
    ]))));
    expect(find.text('Высота отмеченной живой кроны (проекция): 2.00 м'),
        findsOneWidget);
    expect(find.textContaining('нет данных'), findsNothing);
    await t.tap(find.text('Расширенный анализ'));
    await t.pumpAndSettle();
    expect(find.textContaining('DBH: нет данных'), findsOneWidget);
  });
}
