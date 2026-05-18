import 'dart:convert';
import 'package:flutter/services.dart';

class ArMeasureResult {
  /// Generic distance. For legacy Android this is usually height_m.
  final double distanceMeters;
  final double distanceCm;
  final int points;

  /// Full tree measurement values returned by the native AR screen, if available.
  final double? heightMeters;
  final double? crownWidthMeters;
  final double? trunkDiameterMeters;

  ArMeasureResult({
    required this.distanceMeters,
    required this.distanceCm,
    required this.points,
    this.heightMeters,
    this.crownWidthMeters,
    this.trunkDiameterMeters,
  });

  static double? _asDouble(dynamic v) {
    if (v == null) return null;
    if (v is num) return v.toDouble();
    if (v is String) return double.tryParse(v.replaceAll(',', '.'));
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
    // Full 6-point AR workflow:
    // height: base -> top
    // crown: left crown -> right crown
    // trunk: left trunk -> right trunk
    final height = _asDouble(json['height_m']) ??
        _asDouble(json['heightMeters']) ??
        _asDouble(json['tree_height_m']) ??
        _asDouble(json['treeHeightM']);

    final crown = _asDouble(json['crown_width_m']) ??
        _asDouble(json['crownWidthM']) ??
        _asDouble(json['crown_width']) ??
        _asDouble(json['crownMeters']);

    final trunk = _asDouble(json['trunk_diameter_m']) ??
        _asDouble(json['trunkDiameterM']) ??
        _asDouble(json['diameter_m']) ??
        _asDouble(json['trunkMeters']);

    // Backward-compatible generic distance.
    final meters = _asDouble(json['distanceMeters']) ??
        _asDouble(json['distance_m']) ??
        height ??
        _asDouble(json['meters']);

    final cm = _asDouble(json['distanceCm']) ??
        _asDouble(json['distance_cm']) ??
        _asDouble(json['height_cm']) ??
        (meters != null ? meters * 100.0 : null);

    final pts = _asInt(json['points']) ??
        _asInt(json['points_count']) ??
        _asInt(json['pointCount']) ??
        0;

    if (meters == null) {
      throw FormatException('AR result has no distance/height. json=$json');
    }

    return ArMeasureResult(
      distanceMeters: meters,
      distanceCm: cm ?? (meters * 100.0),
      points: pts,
      heightMeters: height ?? meters,
      crownWidthMeters: crown,
      trunkDiameterMeters: trunk,
    );
  }
}

class ArMeasureChannel {
  static const _ch = MethodChannel('arborscan/ar_measure');

  /// Opens native AR measurement screen.
  /// Current Android implementation is a full 6-point tree wizard.
  /// Returns null if user cancelled.
  static Future<ArMeasureResult?> start() async {
    final dynamic raw = await _ch.invokeMethod('start');
    if (raw == null) return null;

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

  static Future<ArMeasureResult?> openArMeasure() => start();
}
