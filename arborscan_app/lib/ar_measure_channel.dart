import 'dart:convert';
import 'package:flutter/services.dart';

class ArMeasureResult {
  /// Height in meters (AR).
  final double heightMeters;

  /// Trunk diameter in meters (AR). May be null if user measured only height.
  final double? trunkDiameterMeters;

  /// Crown width in meters (AR). May be null if user measured only height/height+trunk.
  final double? crownWidthMeters;

  /// How many points were actually placed in AR session.
  final int pointsCount;

  /// How many points were required by the chosen mode (2 / 4 / 6). Optional.
  final int? requiredPoints;

  /// Path to AR snapshot JPEG saved in app cache (Android). May be null.
  final String? capturePath;

  ArMeasureResult({
    required this.heightMeters,
    required this.pointsCount,
    this.trunkDiameterMeters,
    this.crownWidthMeters,
    this.requiredPoints,
    this.capturePath,
  });

  /// Backward compatibility for old UI code.
  double get distanceMeters => heightMeters;
  double get distanceCm => heightMeters * 100.0;
  int get points => pointsCount;

  bool get isFull =>
      trunkDiameterMeters != null && crownWidthMeters != null && pointsCount >= 6;

  bool get isHeightOnly =>
      trunkDiameterMeters == null && crownWidthMeters == null && pointsCount >= 2;

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
    // Supported keys:
    // - Android: height_m, trunk_diameter_m, crown_width_m, points_count, required_points
    // - Backward: distanceMeters / distance_m / meters

    final height = _asDouble(json['height_m']) ??
        _asDouble(json['distanceMeters']) ??
        _asDouble(json['distance_m']) ??
        _asDouble(json['meters']);

    final trunk = _asDouble(json['trunk_diameter_m']) ??
        _asDouble(json['trunkDiameter_m']) ??
        _asDouble(json['trunk_m']);

    final crown = _asDouble(json['crown_width_m']) ??
        _asDouble(json['crownWidth_m']) ??
        _asDouble(json['crown_m']);

    final pts = _asInt(json['points_count']) ??
        _asInt(json['points']) ??
        _asInt(json['pointCount']) ??
        0;

    final req = _asInt(json['required_points']) ?? _asInt(json['requiredPoints']);

    final cap = (json['capture_path'] ?? json['capturePath'])?.toString();

    if (height == null) {
      throw FormatException('AR result has no height. json=$json');
    }

    return ArMeasureResult(
      heightMeters: height,
      trunkDiameterMeters: trunk,
      crownWidthMeters: crown,
      pointsCount: pts,
      requiredPoints: req,
      capturePath: cap,
    );
  }
}

class ArMeasureChannel {
  static const _ch = MethodChannel('arborscan/ar_measure');

  /// Opens native AR measurement screen.
  ///
  /// [requiredPoints] can be 2 / 4 / 6 (default: 6).
  /// Returns null if user cancelled.
  static Future<ArMeasureResult?> openArMeasure({int requiredPoints = 6}) async {
    final int rp = (requiredPoints == 2 || requiredPoints == 4 || requiredPoints == 6)
        ? requiredPoints
        : 6;

    final dynamic raw = await _ch.invokeMethod('start', {'required_points': rp});
    if (raw == null) return null;

    // Android can return String(JSON) or Map.
    Map<String, dynamic> map;
    if (raw is String) {
      if (raw.isEmpty) return null;
      map = json.decode(raw) as Map<String, dynamic>;
    } else if (raw is Map) {
      map = Map<String, dynamic>.from(raw);
    } else {
      throw FormatException('Unexpected AR result type: ${raw.runtimeType}, value=$raw');
    }

    return ArMeasureResult.fromJson(map);
  }

  /// Backward compatible alias.
  static Future<ArMeasureResult?> start() => openArMeasure(requiredPoints: 6);
}
