import 'dart:math' as math;
import 'package:flutter/material.dart';

/// All coordinates refer to the uncropped, EXIF-oriented image, normalized [0,1].
class ReferenceMeasurement {
  final int width, height;
  final int version;
  final List<Offset> crownHeight, trunk, trunkAxis;
  final double lengthM;
  final List<Offset> reference, tree, crown, outline;
  final bool samePlane;
  ReferenceMeasurement(
      {this.version = 1,
      this.crownHeight = const [],
      this.trunk = const [],
      this.trunkAxis = const [],
      required this.width,
      required this.height,
      required this.lengthM,
      required this.reference,
      required this.tree,
      required this.crown,
      required this.outline,
      required this.samePlane}) {
    if (![1, 2].contains(version) ||
        width <= 0 ||
        height <= 0 ||
        width * height > 25000000 ||
        !lengthM.isFinite ||
        lengthM <= 0 ||
        !samePlane ||
        outline.length < 3 ||
        outline.length > 4096 ||
        reference.length != 2 ||
        tree.length != 2 ||
        crown.length != 2) {
      throw const FormatException(
          'Укажите высоту, контур, три отрезка и условия съёмки.');
    }
    for (final line in [crownHeight, trunk, trunkAxis]) {
      if (line.isNotEmpty && (version != 2 || line.length != 2)) {
        throw const FormatException("Неверная дополнительная разметка.");
      }
    }
    for (final p in [
      ...reference,
      ...tree,
      ...crown,
      ...outline,
      ...crownHeight,
      ...trunk,
      ...trunkAxis
    ]) {
      if (!p.dx.isFinite ||
          !p.dy.isFinite ||
          p.dx < 0 ||
          p.dx > 1 ||
          p.dy < 0 ||
          p.dy > 1) {
        throw const FormatException('Точка вне исходного изображения.');
      }
    }
    if (pixels(reference) <= 0 || pixels(tree) <= 0 || pixels(crown) <= 0) {
      throw const FormatException('Концы отрезка должны различаться.');
    }
    for (final line in [crownHeight, trunk, trunkAxis]) {
      if (line.isNotEmpty && pixels(line) <= 0) {
        throw const FormatException('Концы дополнительного отрезка совпадают.');
      }
    }
    if ((crownHeightM != null && crownHeightM! <= 0) ||
        (trunkM != null && trunkM! <= 0)) {
      throw const FormatException(
          'Дополнительный отрезок имеет нулевую проекцию.');
    }
    if (!heightM.isFinite || !crownM.isFinite || heightM <= 0 || crownM <= 0) {
      throw const FormatException(
          'Высота и ширина должны иметь ненулевую проекцию на оси эталона.');
    }
  }
  String get method => 'known_object_segment_v$version';
  double projection(List<Offset> line) {
    final v = vector(line);
    return (v.dx * vertical.dx + v.dy * vertical.dy).abs() * metresPerPixel;
  }

  double? get crownHeightM =>
      crownHeight.isEmpty ? null : projection(crownHeight);
  double? get trunkM => trunk.isEmpty || trunkAxis.isEmpty
      ? null
      : (vector(trunk).dx * vector(trunkAxis).dy -
                  vector(trunk).dy * vector(trunkAxis).dx)
              .abs() /
          pixels(trunkAxis) *
          metresPerPixel;
  double? get trunkLevelM =>
      trunk.isEmpty ? null : projection([tree[0], (trunk[0] + trunk[1]) / 2]);
  double? get leanDeg => trunkAxis.isEmpty
      ? null
      : math.atan2(
              (vector(trunkAxis).dx * vertical.dy -
                      vector(trunkAxis).dy * vertical.dx)
                  .abs(),
              (vector(trunkAxis).dx * vertical.dx +
                      vector(trunkAxis).dy * vertical.dy)
                  .abs()) *
          180 /
          math.pi;
  Map<String, dynamic> get geometry {
    Map<String, dynamic> metric(
            double? value, String unit, String definition, String? reason) =>
        {
          'value': value,
          'unit': unit,
          'method': method,
          'source': 'reference',
          'definition': definition,
          'reason': value == null ? reason : null,
          'limitations': [
            '2D projection; weak perspective; same depth',
            'field accuracy unvalidated'
          ],
        };
    return {
      'tree_height':
          metric(heightM, 'm', 'vertical projection along reference', null),
      'tree_segment_length': metric(pixels(tree) * metresPerPixel, 'm',
          'base-top straight segment in photo plane; not trunk path', null),
      'crown_width': metric(
          crownM, 'm', 'marked crown span perpendicular to reference', null),
      'crown_height': metric(
          crownHeightM,
          'm',
          'marked live crown base to top, along reference',
          'Mark live crown base and top'),
      'trunk_diameter': metric(
          trunkM,
          'm',
          'marked stem width perpendicular to local axis; circular-section assumption',
          'Mark stem edges and local stem axis'),
      'trunk_measurement_height': metric(
          trunkLevelM,
          'm',
          'marked section midpoint above marked tree base in projection; not forestry DBH rule',
          'Mark stem edges'),
      'trunk_lean': metric(
          leanDeg,
          'deg',
          'acute 2D angle of marked local axis to reference; curvature and out-of-plane lean unresolved',
          'Mark local stem axis'),
      'dbh': metric(
          null,
          'm',
          'DBH requires field measurement position protocol',
          '1.3 m alone does not establish DBH: base, slope, forks and lean require field verification'),
      'crown_porosity': metric(null, '1', 'crown gap fraction',
          'No validated crown region and gap-preserving mask; filled polygon is insufficient'),
    };
  }

  Map<String, dynamic> get report => {
        'method': method,
        'height_m': heightM,
        'crown_width_m': crownM,
        'dbh_m': null,
        'beta_kg_s': null,
        if (version == 2) 'geometry': geometry,
      };
  static double parseLength(String text, String unit) {
    final n = double.tryParse(text.trim().replaceAll(',', '.'));
    if (n == null || !n.isFinite || n <= 0 || !['m', 'cm'].contains(unit)) {
      throw const FormatException(
          'Высота должна быть конечным числом больше нуля.');
    }
    return unit == 'cm' ? n / 100 : n;
  }

  double pixels(List<Offset> line) =>
      math.sqrt(math.pow((line[1].dx - line[0].dx) * width, 2) +
          math.pow((line[1].dy - line[0].dy) * height, 2));
  double get metresPerPixel => lengthM / pixels(reference);
  Offset vector(List<Offset> line) => Offset(
      (line[1].dx - line[0].dx) * width, (line[1].dy - line[0].dy) * height);
  Offset get vertical => vector(reference) / pixels(reference);
  double get heightM {
    final v = vector(tree);
    return (v.dx * vertical.dx + v.dy * vertical.dy).abs() * metresPerPixel;
  }

  double get crownM {
    final v = vector(crown);
    return (v.dx * vertical.dy - v.dy * vertical.dx).abs() * metresPerPixel;
  }

  Map<String, dynamic> toJson() => {
        'version': version,
        'method': method,
        if (version == 2) ...{
          'crown_height': encode(crownHeight),
          'trunk': encode(trunk),
          'trunk_axis': encode(trunkAxis),
          'scale_origin': 'vertical_reference_same_depth_user_confirmed',
        },
        'coordinates': 'normalized_oriented_image',
        'width': width,
        'height': height,
        'length_m': lengthM,
        'same_plane': samePlane,
        'reference': encode(reference),
        'tree': encode(tree),
        'crown': encode(crown),
        'outline': encode(outline)
      };
  static List<Map<String, double>> encode(List<Offset> p) =>
      p.map((v) => {'x': v.dx, 'y': v.dy}).toList();
  static List<Offset> decode(dynamic p) => (p as List)
      .map(
          (v) => Offset((v['x'] as num).toDouble(), (v['y'] as num).toDouble()))
      .toList();
  factory ReferenceMeasurement.fromJson(Map<String, dynamic> j) {
    if (![1, 2].contains(j['version']) ||
        j['coordinates'] != 'normalized_oriented_image' ||
        j['method'] != 'known_object_segment_v${j['version']}') {
      throw const FormatException('Неизвестная версия измерения.');
    }
    return ReferenceMeasurement(
        version: j['version'],
        crownHeight: decode(j['crown_height'] ?? []),
        trunk: decode(j['trunk'] ?? []),
        trunkAxis: decode(j['trunk_axis'] ?? []),
        width: j['width'],
        height: j['height'],
        lengthM: (j['length_m'] as num).toDouble(),
        samePlane: j['same_plane'] == true,
        reference: decode(j['reference']),
        tree: decode(j['tree']),
        crown: decode(j['crown']),
        outline: decode(j['outline']));
  }
}
