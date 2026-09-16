import 'package:flutter_test/flutter_test.dart';
import 'package:arborscan_app/ar_measure_channel.dart';

void main() {
  final fixture = <String, dynamic>{
    'height_m': 12.0,
    'trunk_diameter_m': .3,
    'trunk_measurement_height_m': 1.4,
    'distance_m': 4.0,
    'quality': .8,
  };
  test('AR exports direct measurements without inventing photo or crown scale',
      () {
    final result = ArMeasureResult.fromJson(fixture);
    expect(result.distanceCm, 400);
    expect(result.crownWidthMeters, isNull);
    final fields = result.toV4FormFields();
    expect(fields['ar_trunk_measurement_height_m'], '1.4000');
    expect(fields.containsKey('manual_scale_m_per_px'), isFalse);
    expect(fields.containsKey('ar_crown_width_m'), isFalse);
  });
  test('invalid AR distance is rejected instead of replaced with tree height',
      () {
    for (final v in [null, 0, -1, double.nan, double.infinity]) {
      expect(() => ArMeasureResult.fromJson({...fixture, 'distance_m': v}),
          throwsFormatException);
    }
    expect(() => ArMeasureResult.fromJson({...fixture, 'quality': double.nan}),
        throwsFormatException);
  });
}
