import 'dart:convert';
import 'dart:io';

import 'package:arborscan_app/app_theme.dart';
import 'package:arborscan_app/geometry_report.dart';
import 'package:arborscan_app/reference_measurement.dart';
import 'package:arborscan_app/report_export_data.dart';
import 'package:arborscan_app/report_pdf.dart';
import 'package:arborscan_app/unified_analysis_models.dart';
import 'package:arborscan_app/unified_analysis_report_page.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

ReferenceMeasurement measurement(double lengthM) => ReferenceMeasurement(
      version: 2,
      width: 100,
      height: 200,
      lengthM: lengthM,
      reference: const [Offset(.2, .8), Offset(.2, .6)],
      tree: const [Offset(.5, .9), Offset(.5, .1)],
      crown: const [Offset(.2, .3), Offset(.8, .3)],
      crownHeight: const [Offset(.5, .5), Offset(.5, .1)],
      trunk: const [Offset(.45, .7), Offset(.55, .7)],
      trunkAxis: const [Offset(.5, .8), Offset(.5, .6)],
      outline: const [Offset(.1, .8), Offset(.3, .8), Offset(.3, .5)],
      samePlane: true,
    );

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  setUp(() => SharedPreferences.setMockInitialValues({}));

  test('nonfinite legacy metric values remain unavailable in UI and PDF', () {
    for (final input in ['NaN', 'Infinity', '-Infinity', double.nan]) {
      final metric = UnifiedMetric.fromJson({
        'value_m': input,
        'value_px': input,
        'measurement_height_m': input,
        'confidence': input,
      });
      expect(metric.valueM, isNull);
      expect(metric.valuePx, isNull);
      expect(metric.measurementHeightM, isNull);
      expect(metric.confidence, 0);
    }
    final metric = UnifiedMetric.fromJson({'value_m': '0,25'});
    expect(metric.valueM, .25);
  });

  testWidgets('saved geometry is shown instead of a forced missing snapshot',
      (tester) async {
    final geometry = measurement(1).geometry;
    await tester.pumpWidget(MaterialApp(
        theme: AppTheme.light(),
        home: UnifiedAnalysisReportPage(
            allowServerSave: false,
            result: UnifiedAnalysisResult.fromJson({
              'analysis_id': 'DEMO-AS16-GEOMETRY',
              'measurement_method_version': 2,
              'geometry': geometry,
              'measurements': {
                'trunk_diameter': {
                  'value_m': .25,
                  'source': 'ar',
                  'measurement_height_m': 1.3,
                  'standard': 'dbh_1_3m',
                }
              }
            }))));
    await tester.pumpAndSettle();
    await tester.scrollUntilVisible(find.text('DBH (историческая метка)'), 200,
        scrollable: find.byType(Scrollable).first);
    expect(find.textContaining('Историческая метка DBH из сохранённого отчёта'),
        findsOneWidget);
    expect(find.textContaining('Сохранённый уровень сечения: 1.30 м'),
        findsOneWidget);
    expect(find.textContaining('DBH подтверждён'), findsNothing);
    await tester.scrollUntilVisible(
        find.text('Высота отмеченной живой кроны (проекция): 2.00 m'), 150,
        scrollable: find.byType(Scrollable).first);
    expect(find.text('Ширина сечения ствола (фотооценка диаметра): 0.25 m'),
        findsOneWidget);
    expect(find.text('Угол оси ствола к эталону (2D): 0.00 °'), findsOneWidget);
    expect(geometry['dbh']['value'], isNull);
    expect(geometry['crown_porosity']['value'], isNull);
    expect(tester.takeException(), isNull);
  });

  testWidgets('AR section at 1.3 m never becomes a confirmed DBH',
      (tester) async {
    await tester.pumpWidget(MaterialApp(
        theme: AppTheme.light(),
        home: UnifiedAnalysisReportPage(
            allowServerSave: false,
            result: UnifiedAnalysisResult.fromJson({
              'measurements': {
                'trunk_diameter': {
                  'value_m': .25,
                  'source': 'ar',
                  'measurement_height_m': 1.3,
                }
              }
            }))));
    await tester.pumpAndSettle();
    await tester.scrollUntilVisible(find.text('Диаметр ствола'), 200,
        scrollable: find.byType(Scrollable).first);
    expect(find.textContaining('уровень 1,30 м сам по себе недостаточен'),
        findsOneWidget);
    expect(find.textContaining('DBH подтверждён'), findsNothing);
    expect(tester.takeException(), isNull);
  });

  testWidgets('nonfinite saved geometry does not render a physical value',
      (tester) async {
    await tester.pumpWidget(const MaterialApp(
        home: Scaffold(
            body: GeometryReport(data: {
      'crown_height': {'value': double.nan, 'unit': 'm'},
      'trunk_lean': {'value': 0, 'unit': 'deg'},
    }))));
    expect(find.textContaining('NaN'), findsNothing);
    expect(
        find.textContaining(
            'Высота отмеченной живой кроны (проекция): нет данных'),
        findsOneWidget);
    expect(find.text('Угол оси ствола к эталону (2D): 0.00 °'), findsOneWidget);
  });

  test('two EXIF reference versions preserve all geometry and generate PDFs',
      () async {
    final image = await File('test/fixtures/reference_exif6.jpg').readAsBytes();
    final directory = Directory(
        Platform.environment['ARBORSCAN_GEOMETRY_PDF_TEST_OUTPUT'] ??
            '../output/as16/geometry-pdf')
      ..createSync(recursive: true);
    final sources = <Map<String, dynamic>>[];
    for (final entry in {1: 1.0, 2: 2.0}.entries) {
      final restored = ReferenceMeasurement.fromJson(
          jsonDecode(jsonEncode(measurement(entry.value).toJson())));
      final snapshot = {
        'version': 1,
        'kind': 'reference',
        'reference': restored.toJson(),
        'report': restored.report,
        'captured_at': '2026-10-06T09:00:00Z',
      };
      final data = ReportExportData(
          snapshot: snapshot,
          record: {
            'analysis_id': 'DEMO-AS16-GEOMETRY',
            'version_id': 'geometry-v${entry.key}',
            'created_at': '2026-10-06T09:00:00Z',
          },
          photo: image,
          local: false);
      final metrics = {for (final m in data.metrics) m.label: m};
      expect(metrics['Высота дерева (проекция)']!.value,
          closeTo(4 * entry.value, 1e-8));
      expect(metrics['Ширина кроны (проекция)']!.value,
          closeTo(1.5 * entry.value, 1e-8));
      expect(metrics[GeometryReport.labels['crown_height']]!.value,
          closeTo(2 * entry.value, 1e-8));
      expect(metrics[GeometryReport.labels['trunk_diameter']]!.value,
          closeTo(.25 * entry.value, 1e-8));
      expect(metrics[GeometryReport.labels['trunk_lean']]!.display, '0 °');
      expect(metrics['DBH']!.value, isNull);
      expect(metrics['Пористость кроны']!.value, isNull);
      final images = prepareReportImages(data);
      expect(images.legend, hasLength(7));
      expect(images.legend.join('\n'), contains('Ось участка ствола'));
      expect(data.reference['trunk_axis'], restored.toJson()['trunk_axis']);
      final bytes = await buildReportPdf(data,
          generatedAt: DateTime.utc(2026, 10, 6, 10));
      expect(ascii.decode(bytes.take(5).toList()), '%PDF-');
      await File('${directory.path}/${data.filename}').writeAsBytes(bytes);
      sources.add({
        'record': data.record,
        'snapshot': data.snapshot,
        'pdf': data.filename,
        'expected_metrics': {
          for (final m in data.metrics)
            m.label: {'value': m.value, 'unit': m.unit},
        },
      });
    }
    await File('${directory.path}/geometry-versions.json')
        .writeAsString(const JsonEncoder.withIndent('  ').convert(sources));
  });
}
