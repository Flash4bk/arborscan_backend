import 'dart:convert';
import 'package:flutter/services.dart';

class ArMeasureResult {
  final double distanceMeters;
  final double distanceCm;
  final int points;

  final double? heightMeters;
  final double? crownWidthMeters;
  final double? trunkDiameterMeters;
  final double? trunkMeasurementHeightMeters;

  /// Internal engineering fusion weight. It is NOT shown to the user as
  /// metrological accuracy.
  final double quality;
  final String overallStatus;
  final String schemaVersion;
  final double? trackingRatio;
  final int? trackingLossEvents;
  final String? baseSurface;
  final double? dbhRepeatSpreadMeters;
  final double? dbhRepeatSpreadRatio;
  final List<String> warnings;
  final Map<String, dynamic> raw;

  const ArMeasureResult({
    required this.distanceMeters,
    required this.distanceCm,
    required this.points,
    required this.heightMeters,
    required this.crownWidthMeters,
    required this.trunkDiameterMeters,
    required this.trunkMeasurementHeightMeters,
    required this.quality,
    required this.overallStatus,
    required this.schemaVersion,
    required this.warnings,
    required this.raw,
    this.trackingRatio,
    this.trackingLossEvents,
    this.baseSurface,
    this.dbhRepeatSpreadMeters,
    this.dbhRepeatSpreadRatio,
  });

  static double? _asDouble(dynamic value) {
    if (value == null) return null;
    if (value is num) return value.toDouble();
    if (value is String) return double.tryParse(value.replaceAll(',', '.'));
    return null;
  }

  static int? _asInt(dynamic value) {
    if (value == null) return null;
    if (value is int) return value;
    if (value is num) return value.toInt();
    if (value is String) return int.tryParse(value);
    return null;
  }

  static double _requiredPositive(Map<String, dynamic> json, String key) {
    final value = _asDouble(json[key]);
    if (value == null || !value.isFinite || value <= 0) {
      throw FormatException('AR result has invalid $key: ${json[key]}');
    }
    return value;
  }

  factory ArMeasureResult.fromJson(Map<String, dynamic> json) {
    final height = _requiredPositive(json, 'height_m');
    final trunk = _requiredPositive(json, 'trunk_diameter_m');
    final trunkHeight = _requiredPositive(json, 'trunk_measurement_height_m');

    final crownRaw = _asDouble(json['crown_width_m']);
    final crown = crownRaw != null && crownRaw.isFinite && crownRaw > 0
        ? crownRaw
        : null;

    final distance = _requiredPositive(json, 'distance_m');

    final rawQuality = _asDouble(json['quality']) ?? 0.0;
    if (!rawQuality.isFinite) throw const FormatException('Неконечная диагностика AR.');
    final quality = rawQuality.clamp(0.0, 1.0).toDouble();
    final warningsRaw = json['warnings'];
    final warnings = warningsRaw is List
        ? warningsRaw.map((e) => e.toString()).toList(growable: false)
        : const <String>[];

    return ArMeasureResult(
      distanceMeters: distance,
      distanceCm: distance * 100.0,
      points: _asInt(json['points_count']) ?? 0,
      heightMeters: height,
      crownWidthMeters: crown,
      trunkDiameterMeters: trunk,
      trunkMeasurementHeightMeters: trunkHeight,
      quality: quality,
      overallStatus: json['overall_status']?.toString() ?? 'unknown',
      schemaVersion: json['schema_version']?.toString() ?? 'unknown',
      trackingRatio: _asDouble(json['tracking_ratio']),
      trackingLossEvents: _asInt(json['tracking_loss_events']),
      baseSurface: json['base_surface']?.toString(),
      dbhRepeatSpreadMeters: _asDouble(json['dbh_repeat_spread_m']),
      dbhRepeatSpreadRatio: _asDouble(json['dbh_repeat_spread_ratio']),
      warnings: warnings,
      raw: Map<String, dynamic>.from(json),
    );
  }

  String get statusLabelRu {
    switch (overallStatus) {
      case 'good':
        return 'условия хорошие';
      case 'usable':
        return 'условия допустимые';
      case 'repeat_recommended':
        return 'желательно повторить';
      default:
        return 'статус не определён';
    }
  }

  /// Direct spatial height and diameter at the recorded measurement height.
  /// Crown is not measured here; AR height does not calibrate photo width.
  Map<String, String> toV4FormFields() {
    final height = heightMeters;
    final trunk = trunkDiameterMeters;
    final trunkHeight = trunkMeasurementHeightMeters;
    if (height == null || trunk == null || trunkHeight == null) {
      throw StateError('Unified AR result is incomplete');
    }
    return {
      'ar_height_m': height.toStringAsFixed(4),
      'ar_trunk_diameter_m': trunk.toStringAsFixed(4),
      'ar_trunk_measurement_height_m': trunkHeight.toStringAsFixed(4),
      'ar_quality': quality.toStringAsFixed(3),
    };
  }
}

class ArMeasureChannel {
  static const MethodChannel _channel = MethodChannel('arborscan/ar_measure');

  static Future<ArMeasureResult?> start() async {
    final dynamic raw = await _channel.invokeMethod('start');
    if (raw == null) return null;

    late final Map<String, dynamic> map;
    if (raw is String) {
      if (raw.trim().isEmpty) return null;
      final decoded = jsonDecode(raw);
      if (decoded is! Map) {
        throw const FormatException('AR result JSON is not an object');
      }
      map = Map<String, dynamic>.from(decoded);
    } else if (raw is Map) {
      map = Map<String, dynamic>.from(raw);
    } else {
      throw FormatException('Unexpected AR result type: ${raw.runtimeType}');
    }

    return ArMeasureResult.fromJson(map);
  }

  static Future<ArMeasureResult?> openArMeasure() => start();
}
