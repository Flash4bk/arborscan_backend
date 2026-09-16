import 'dart:math' as math;
import 'package:flutter/material.dart';

/// All coordinates refer to the uncropped, EXIF-oriented image, normalized [0,1].
class ReferenceMeasurement {
  final int width, height;
  final double lengthM;
  final List<Offset> reference, tree, crown, outline;
  final bool samePlane;
  ReferenceMeasurement(
      {required this.width,
      required this.height,
      required this.lengthM,
      required this.reference,
      required this.tree,
      required this.crown,
      required this.outline,
      required this.samePlane}) {
    if (width <= 0 ||
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
    for (final p in [...reference, ...tree, ...crown, ...outline]) {
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
    if (!heightM.isFinite || !crownM.isFinite || heightM <= 0 || crownM <= 0) {
      throw const FormatException(
          'Высота и ширина должны иметь ненулевую проекцию на оси эталона.');
    }
  }
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
        'version': 1,
        'method': 'known_object_segment_v1',
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
    if (j['version'] != 1 ||
        j['coordinates'] != 'normalized_oriented_image' ||
        j['method'] != 'known_object_segment_v1') {
      throw const FormatException('Неизвестная версия измерения.');
    }
    return ReferenceMeasurement(
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
