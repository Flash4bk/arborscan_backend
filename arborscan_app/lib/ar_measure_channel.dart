import 'dart:convert';
import 'package:flutter/services.dart';

class ArMeasureResult {
  final double distanceMeters;
  final double distanceCm;
  final int points;
  final double? heightMeters;
  final double? crownWidthMeters;
  final double? trunkDiameterMeters;
  final double? zoomAssist;
  final bool? usedFeaturePoint;
  final bool? centerPlacement;

  ArMeasureResult({
    required this.distanceMeters,
    required this.distanceCm,
    required this.points,
    this.heightMeters,
    this.crownWidthMeters,
    this.trunkDiameterMeters,
    this.zoomAssist,
    this.usedFeaturePoint,
    this.centerPlacement,
  });

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

  static bool? _asBool(dynamic v) {
    if (v == null) return null;
    if (v is bool) return v;
    if (v is String) {
      final s = v.toLowerCase().trim();
      if (s == 'true') return true;
      if (s == 'false') return false;
    }
    return null;
  }

  factory ArMeasureResult.fromJson(Map<String, dynamic> json) {
    final height = _asDouble(json['height_m']) ??
        _asDouble(json['distanceMeters']) ??
        _asDouble(json['distance_m']) ??
        _asDouble(json['meters']);

    final crown = _asDouble(json['crown_width_m']) ??
        _asDouble(json['crownWidthMeters']);

    final trunk = _asDouble(json['trunk_diameter_m']) ??
        _asDouble(json['trunkDiameterMeters']);

    if (height == null) {
      throw FormatException('AR result has no height/distance. json=$json');
    }

    final points = _asInt(json['points_count']) ??
        _asInt(json['points']) ??
        _asInt(json['pointCount']) ??
        0;

    return ArMeasureResult(
      distanceMeters: height,
      distanceCm: height * 100.0,
      points: points,
      heightMeters: height,
      crownWidthMeters: crown,
      trunkDiameterMeters: trunk,
      zoomAssist: _asDouble(json['zoom_assist']),
      usedFeaturePoint: _asBool(json['used_feature_point']),
      centerPlacement: _asBool(json['center_placement']),
    );
  }
}

class ArMeasureChannel {
  static const _ch = MethodChannel('arborscan/ar_measure');

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
