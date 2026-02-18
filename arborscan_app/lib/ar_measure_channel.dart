import 'dart:convert';
import 'package:flutter/services.dart';

class ArMeasureResult {
  /// Tree height in meters (points 1-2).
  final double heightMeters;

  /// Trunk diameter in meters (points 3-4). Nullable until measured.
  final double? trunkDiameterMeters;

  /// Crown width in meters (points 5-6). Nullable until measured.
  final double? crownWidthMeters;

  /// Number of placed points.
  final int points;

  ArMeasureResult({
    required this.heightMeters,
    required this.trunkDiameterMeters,
    required this.crownWidthMeters,
    required this.points,
  });

  // Backward compatibility with older UI that expected a single distance.
  double get distanceMeters => heightMeters;
  double get distanceCm => heightMeters * 100.0;

  static double? _asDouble(dynamic v) {
    if (v == null) return null;
    if (v is num) return v.toDouble();
    if (v is String) return double.tryParse(v);
    return null;
  }

  static int? _asInt(dynamic v) {
    if (v == null) return null;
    if (v is int) return v;
    if (v is num) return v.toInt();
    if (v is String) return int.tryParse(v);
    return null;
  }

  factory ArMeasureResult.fromJson(Map<String, dynamic> json) {
    // Android keys: height_m, trunk_diameter_m, crown_width_m, points_count
    // Older keys (if any): distanceMeters / distanceCm / points
    final height = _asDouble(json['height_m']) ??
        _asDouble(json['distanceMeters']) ??
        _asDouble(json['distance_m']) ??
        _asDouble(json['meters']);

    final trunk = _asDouble(json['trunk_diameter_m']) ??
        _asDouble(json['trunkDiameter_m']) ??
        _asDouble(json['trunk_diameter']);

    final crown = _asDouble(json['crown_width_m']) ??
        _asDouble(json['crownWidth_m']) ??
        _asDouble(json['crown_width']);

    final pts = _asInt(json['points_count']) ??
        _asInt(json['points']) ??
        _asInt(json['pointCount']) ??
        0;

    if (height == null) {
      throw FormatException('AR result has no height/distance. json=$json');
    }

    return ArMeasureResult(
      heightMeters: height,
      trunkDiameterMeters: trunk,
      crownWidthMeters: crown,
      points: pts,
    );
  }
}

class ArMeasureChannel {
  static const _ch = MethodChannel('arborscan/ar_measure');

  /// Opens native AR measurement screen.
  /// Returns null if user cancelled.
  static Future<ArMeasureResult?> start() async {
    final dynamic raw = await _ch.invokeMethod('start');
    if (raw == null) return null;

    // Android may return String (JSON) or Map.
    Map<String, dynamic> map;
    if (raw is String) {
      if (raw.isEmpty) return null;
      map = json.decode(raw) as Map<String, dynamic>;
    } else if (raw is Map) {
      map = Map<String, dynamic>.from(raw);
    } else {
      throw FormatException(
          'Unexpected AR result type: ${raw.runtimeType}, value=$raw');
    }

    return ArMeasureResult.fromJson(map);
  }

  /// Backward-compatible alias used by older Dart code.
  static Future<ArMeasureResult?> openArMeasure() => start();
}
