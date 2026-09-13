import 'dart:convert';
import 'dart:typed_data';

class UnifiedMetric {
  final double? valueM;
  final double? valuePx;
  final String source;
  final double confidence;
  final double? measurementHeightM;
  final String? standard;
  final List<String> notes;

  const UnifiedMetric({
    required this.valueM,
    required this.valuePx,
    required this.source,
    required this.confidence,
    required this.measurementHeightM,
    required this.standard,
    required this.notes,
  });

  factory UnifiedMetric.fromJson(Map<String, dynamic>? json) {
    final data = json ?? const <String, dynamic>{};
    return UnifiedMetric(
      valueM: _asDouble(data['value_m']),
      valuePx: _asDouble(data['value_px']),
      source: data['source']?.toString() ?? 'unavailable',
      confidence: (_asDouble(data['confidence']) ?? 0.0).clamp(0.0, 1.0).toDouble(),
      measurementHeightM: _asDouble(data['measurement_height_m']),
      standard: data['standard']?.toString(),
      notes: _stringList(data['notes']),
    );
  }
}

class UnifiedAnalysisResult {
  final String analysisId;
  final String apiVersion;
  final String schemaVersion;
  final String status;

  final bool treeDetected;
  final double segmentationConfidence;
  final bool touchesImageEdge;

  final String speciesName;
  final String? scientificName;
  final double? speciesConfidence;

  final UnifiedMetric height;
  final UnifiedMetric crownWidth;
  final UnifiedMetric trunkDiameter;

  final bool calibrationAvailable;
  final double? pxToM;
  final String? calibrationSource;
  final double calibrationConfidence;
  final bool calibrationConflict;

  final double overallQuality;
  final String qualityStatus;

  final bool mechanicalProfileAvailable;
  final String? mechanicalProfileReason;

  final bool riskAvailable;
  final String? riskReason;

  final Uint8List? annotatedImageBytes;
  final Uint8List? maskImageBytes;

  final List<String> warnings;
  final Map<String, dynamic> raw;

  const UnifiedAnalysisResult({
    required this.analysisId,
    required this.apiVersion,
    required this.schemaVersion,
    required this.status,
    required this.treeDetected,
    required this.segmentationConfidence,
    required this.touchesImageEdge,
    required this.speciesName,
    required this.scientificName,
    required this.speciesConfidence,
    required this.height,
    required this.crownWidth,
    required this.trunkDiameter,
    required this.calibrationAvailable,
    required this.pxToM,
    required this.calibrationSource,
    required this.calibrationConfidence,
    required this.calibrationConflict,
    required this.overallQuality,
    required this.qualityStatus,
    required this.mechanicalProfileAvailable,
    required this.mechanicalProfileReason,
    required this.riskAvailable,
    required this.riskReason,
    required this.annotatedImageBytes,
    required this.maskImageBytes,
    required this.warnings,
    required this.raw,
  });

  factory UnifiedAnalysisResult.fromJson(Map<String, dynamic> json) {
    final tree = _map(json['tree']);
    final species = _map(json['species']);
    final measurements = _map(json['measurements']);
    final calibration = _map(json['calibration']);
    final quality = _map(json['quality']);
    final mechanical = _map(json['mechanical_profile']);
    final risk = _map(json['risk']);
    final images = _map(json['images']);

    final allWarnings = <String>{
      ..._stringList(json['warnings']),
      ..._stringList(quality['warnings']),
    }.toList(growable: false);

    return UnifiedAnalysisResult(
      analysisId: json['analysis_id']?.toString() ?? '',
      apiVersion: json['api_version']?.toString() ?? '',
      schemaVersion: json['schema_version']?.toString() ?? '',
      status: json['analysis_status']?.toString() ?? 'unknown',
      treeDetected: tree['detected'] == true,
      segmentationConfidence:
          (_asDouble(tree['segmentation_confidence']) ?? 0.0).clamp(0.0, 1.0).toDouble(),
      touchesImageEdge: tree['touches_image_edge'] == true,
      speciesName: species['display_name']?.toString() ?? 'Неизвестно',
      scientificName: species['scientific_name']?.toString(),
      speciesConfidence: _asDouble(species['confidence']),
      height: UnifiedMetric.fromJson(_mapOrNull(measurements['height'])),
      crownWidth:
          UnifiedMetric.fromJson(_mapOrNull(measurements['crown_width'])),
      trunkDiameter:
          UnifiedMetric.fromJson(_mapOrNull(measurements['trunk_diameter'])),
      calibrationAvailable: calibration['available'] == true,
      pxToM: _asDouble(calibration['px_to_m']),
      calibrationSource: calibration['source']?.toString(),
      calibrationConfidence:
          (_asDouble(calibration['confidence']) ?? 0.0).clamp(0.0, 1.0).toDouble(),
      calibrationConflict: calibration['conflict'] == true,
      overallQuality:
          (_asDouble(quality['overall']) ?? 0.0).clamp(0.0, 1.0).toDouble(),
      qualityStatus: quality['status']?.toString() ?? 'unknown',
      mechanicalProfileAvailable: mechanical['available'] == true,
      mechanicalProfileReason: mechanical['reason']?.toString(),
      riskAvailable: risk['available'] == true,
      riskReason: risk['reason']?.toString(),
      annotatedImageBytes: _decodeB64(images['annotated_image_base64']),
      maskImageBytes: _decodeB64(images['mask_image_base64']),
      warnings: allWarnings,
      raw: Map<String, dynamic>.from(json),
    );
  }

  bool get hasMetricMeasurements =>
      height.valueM != null ||
      crownWidth.valueM != null ||
      trunkDiameter.valueM != null;

  bool get fullyMeasured =>
      height.valueM != null &&
      crownWidth.valueM != null &&
      trunkDiameter.valueM != null;
}

Map<String, dynamic> _map(dynamic value) {
  if (value is Map<String, dynamic>) return value;
  if (value is Map) return Map<String, dynamic>.from(value);
  return <String, dynamic>{};
}

Map<String, dynamic>? _mapOrNull(dynamic value) {
  if (value is Map<String, dynamic>) return value;
  if (value is Map) return Map<String, dynamic>.from(value);
  return null;
}

double? _asDouble(dynamic value) {
  if (value == null) return null;
  if (value is num) return value.toDouble();
  if (value is String) return double.tryParse(value.replaceAll(',', '.'));
  return null;
}

List<String> _stringList(dynamic value) {
  if (value is! List) return const <String>[];
  return value.map((e) => e.toString()).toList(growable: false);
}

Uint8List? _decodeB64(dynamic value) {
  if (value == null) return null;
  var raw = value.toString().trim();
  if (raw.isEmpty) return null;
  if (raw.startsWith('data:')) {
    final comma = raw.indexOf(',');
    if (comma >= 0) raw = raw.substring(comma + 1);
  }
  try {
    return base64Decode(raw);
  } catch (_) {
    return null;
  }
}
