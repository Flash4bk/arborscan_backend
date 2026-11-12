import 'package:arborscan_app/analysis_report_page.dart';
import 'package:arborscan_app/geometry_report.dart';
import 'package:arborscan_app/measurement_results_summary.dart';
import 'package:arborscan_app/report_export_data.dart';
import 'package:arborscan_app/species_annotation.dart';
import 'package:arborscan_app/survey_map_repository.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  const owner = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa';

  test('map keeps manual species separate from original inference', () {
    final report = withSpeciesAnnotation({
      'species': {'display_name': 'DEMO original prediction'},
    }, 'DEMO manual name');
    final entry = SurveyMapEntry.version({
      'version_id': 'demo-version',
      'analysis_id': 'demo-analysis',
      'snapshot': {'kind': 'v4', 'report': report},
    }, owner, server: false);
    expect(entry.species, speciesDisplayName(report));
    expect(entry.speciesStatus, contains('пользовател'));
    expect(entry.speciesStatus, contains('не подтвержден'));
    expect(report['species']['display_name'], 'DEMO original prediction');
  });

  test('map and PDF select reference section instead of retained AR source',
      () {
    final snapshot = <String, dynamic>{
      'kind': 'reference',
      'reference': {'version': 2, 'method': 'known_object_segment_v2'},
      'report': {
        'height_m': 12.0,
        'crown_width_m': 4.0,
        'measurements': {
          'trunk_diameter': {'value_m': .4, 'source': 'ar'},
        },
        'geometry': {
          'trunk_diameter': {
            'value': .25,
            'unit': 'm',
            'source': 'known_object_segment_v2',
            'available': true,
          },
        },
      },
    };
    final entry = SurveyMapEntry.version({
      'version_id': 'demo-version',
      'analysis_id': 'demo-analysis',
      'snapshot': snapshot,
    }, owner, server: false);
    final exported =
        ReportExportData(snapshot: snapshot, record: {}, local: true);
    final selected = exported.metrics
        .firstWhere((m) => m.label == GeometryReport.labels['trunk_diameter']);
    expect(selected.value, .25);
    expect(entry.trunkM, selected.value);
    expect(snapshot['report']['measurements']['trunk_diameter']['value_m'], .4);
  });

  for (final metric in [
    {'value_m': 99.0, 'source': 'unavailable', 'available': false},
    {'value_m': 99.0, 'source': 'vision'},
    {'value_m': 99.0, 'source': 'ar', 'unit': 'cm'},
    {'value_m': null, 'source': 'ar'},
  ]) {
    test(
        'unavailable or incompatible metric cannot fall back to old value: $metric',
        () {
      final snapshot = <String, dynamic>{
        'kind': 'v4',
        'report': {
          'height_m': 77.0,
          'measurements': {'height': metric}
        },
      };
      final entry = SurveyMapEntry.version({
        'version_id': 'demo-version',
        'analysis_id': 'demo-analysis',
        'snapshot': snapshot,
      }, owner, server: false);
      final exported =
          ReportExportData(snapshot: snapshot, record: {}, local: true);
      expect(entry.heightM, isNull);
      expect(exported.metrics.first.value, isNull);
    });
  }

  for (final invalid in [
    {'source': 'ar', 'available': false},
    {'source': 'unavailable'},
    {'source': 'vision'},
    {'source': 'ar', 'unit': 'cm'},
  ]) {
    test('unavailable nested trunk cannot reappear as historical DBH: $invalid',
        () {
      final data = ReportExportData.measurementView({
        'kind': 'v4',
        'report': {
          'measurements': {
            'trunk_diameter': {
              'value_m': .99,
              'standard': 'dbh_1_3m',
              ...invalid,
            }
          }
        }
      });
      expect(
          data.metrics
              .firstWhere((m) => m.label == GeometryReport.labels['dbh'])
              .value,
          isNull);
    });
    testWidgets('additional summary cannot display rejected geometry: $invalid',
        (tester) async {
      final snapshot = <String, dynamic>{
        'kind': 'v4',
        'report': {
          'geometry': {
            'crown_height': {'value': 999, 'unit': 'm', ...invalid}
          }
        }
      };
      final metric = ReportExportData.measurementView(snapshot)
          .metrics
          .firstWhere((m) => m.label == GeometryReport.labels['crown_height']);
      expect(metric.value, isNull);
      await tester.pumpWidget(MaterialApp(
          home: Scaffold(
              body: SingleChildScrollView(
                  child: MeasurementResultsSummary(snapshot: snapshot)))));
      expect(find.text('Другие измеренные параметры'), findsNothing);
      expect(find.textContaining('999'), findsNothing);
    });
  }
  test(
      'supported nested DBH remains historical; explicit null geometry is authoritative',
      () {
    final report = <String, dynamic>{
      'measurements': {
        'trunk_diameter': {
          'value_m': .31,
          'source': 'ar',
          'unit': 'm',
          'standard': 'dbh_1_3m'
        }
      }
    };
    ExportMetric selected() =>
        ReportExportData.measurementView({'kind': 'v4', 'report': report})
            .metrics
            .firstWhere((m) => m.label == GeometryReport.labels['dbh']);
    expect(selected().value, .31);
    expect(selected().limitation, contains('Историческая'));
    report['geometry'] = {
      'dbh': {'value': null, 'unit': 'm', 'source': 'unavailable'}
    };
    expect(selected().value, isNull);
  });

  testWidgets('legacy low risk is presented as a historical estimate',
      (tester) async {
    await tester.pumpWidget(MaterialApp(
      home: AnalysisReportPageV2.fromHistory(
        species: 'DEMO tree',
        riskCategory: 'низкий',
        riskIndex: .2,
      ),
    ));
    expect(find.text('БЕЗОПАСНО'), findsNothing);
    expect(find.text('ИСТОРИЧЕСКАЯ ОЦЕНКА'), findsOneWidget);
    expect(find.text('0.20'), findsOneWidget);
  });

  testWidgets('unsupported legacy category does not become critical verdict',
      (tester) async {
    await tester.pumpWidget(MaterialApp(
      home: AnalysisReportPageV2.fromHistory(
        species: 'DEMO tree',
        riskCategory: 'DEMO unsupported category',
      ),
    ));
    expect(find.text('КРИТИЧНО'), findsNothing);
    expect(find.text('ИСТОРИЧЕСКАЯ ОЦЕНКА'), findsOneWidget);
  });
}
