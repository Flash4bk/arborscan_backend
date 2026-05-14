import 'dart:convert';
import 'package:flutter/services.dart';

class ArMeasureResult {
  final double distanceMeters;
  final double distanceCm;
  final int points;

  final double? heightMeters;
  final double? crownWidthMeters;
  final double? trunkDiameterMeters;

  const ArMeasureResult({
    required this.distanceMeters,
    required this.distanceCm,
    required this.points,
    this.heightMeters,
    this.crownWidthMeters,
    this.trunkDiameterMeters,
  });

  static double? _d(dynamic v) {
    if (v == null) return null;
    if (v is num) return v.toDouble();
    return double.tryParse(v.toString());
  }

  static int _i(dynamic v) {
    if (v is int) return v;
    if (v is num) return v.toInt();
    return int.tryParse(v.toString()) ?? 0;
  }

  factory ArMeasureResult.fromJson(Map<String, dynamic> json) {
    final h = _d(json['height_m']);
    final c = _d(json['crown_width_m']);
    final t = _d(json['trunk_diameter_m']);

    final meters = h ?? _d(json['distance_m']) ?? 0;

    return ArMeasureResult(
      distanceMeters: meters,
      distanceCm: meters * 100,
      points: _i(json['points_count']),
      heightMeters: h,
      crownWidthMeters: c,
      trunkDiameterMeters: t,
    );
  }
}

class ArMeasureChannel {
  static const _ch = MethodChannel('arborscan/ar_measure');

  // СТАРОЕ имя метода — возвращаем для совместимости
  static Future<ArMeasureResult?> openArMeasure() async {
    final raw = await _ch.invokeMethod('start');
    if (raw == null) return null;

    final map = raw is String ? json.decode(raw) : Map<String, dynamic>.from(raw);
    return ArMeasureResult.fromJson(map);
  }

  // Новое имя (можешь потом перейти на него)
  static Future<ArMeasureResult?> start() async {
    return openArMeasure();
  }
}
