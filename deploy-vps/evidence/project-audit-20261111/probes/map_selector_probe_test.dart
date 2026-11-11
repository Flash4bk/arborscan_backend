// Audit probes: these assert observed divergences, not approved behaviour.
// Synthetic DEMO records only. No server, device, account, file or auth writes.
import 'package:flutter_test/flutter_test.dart';
import 'package:arborscan_app/survey_map_repository.dart';
import 'package:arborscan_app/species_annotation.dart';
import 'package:arborscan_app/report_export_data.dart';
import 'package:arborscan_app/geometry_report.dart';

void main() {
  const owner = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa';
  test('audit reproduces manual species omitted by map', () {
    final report = withSpeciesAnnotation({
      'species': {'display_name': 'DEMO original prediction'},
    }, 'DEMO manual name');
    final entry = SurveyMapEntry.version({
      'version_id': 'demo-version',
      'analysis_id': 'demo-analysis',
      'snapshot': {'kind': 'v4', 'report': report},
    }, owner, server: false);
    expect(speciesDisplayName(report), 'DEMO manual name');
    expect(entry.species, 'DEMO original prediction');
    expect(entry.species, isNot(speciesDisplayName(report)));
  });

  test('audit reproduces map selecting old AR diameter in reference version', () {
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
    final exported = ReportExportData(snapshot: snapshot, record: {}, local: true);
    final selected = exported.metrics.firstWhere(
        (m) => m.label == GeometryReport.labels['trunk_diameter']);
    expect(selected.value, .25);
    expect(entry.trunkM, .4);
    expect(entry.trunkM, isNot(selected.value));
  });

  test('audit reproduces map displaying explicitly unavailable measurement', () {
    final entry = SurveyMapEntry.version({
      'version_id': 'demo-version',
      'analysis_id': 'demo-analysis',
      'snapshot': {
        'kind': 'v4',
        'report': {
          'measurements': {
            'height': {
              'value_m': 99.0,
              'source': 'unavailable',
              'available': false,
            },
          },
        },
      },
    }, owner, server: false);
    expect(entry.heightM, 99.0);
  });
}
