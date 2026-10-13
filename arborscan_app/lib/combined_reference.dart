import 'dart:convert';
import 'dart:math' as math;

/// Exact visual transform used by v4 encode_visuals, including Python's
/// round-to-even at half pixels. This is not a crop or a coordinate change.
int _roundEven(double value) {
  final lower = value.floor();
  final fraction = value - lower;
  if ((fraction - .5).abs() < 1e-9) {
    return lower.isEven ? lower : lower + 1;
  }
  return value.round();
}

List<int> modelPreviewSize(int width, int height) {
  final scale = math.min(1.0, 1600 / math.max(width, height));
  return [
    math.max(1, _roundEven(width * scale)),
    math.max(1, _roundEven(height * scale))
  ];
}

/// The server history contract binds images.width/height to the ORIGINAL.
/// v4's returned visual bytes may be a bounded preview. Preserve their actual
/// frame separately; never replace the original photo with its model preview.
Map<String, dynamic> boundAnalysisReport(
    Map<String, dynamic> report, int width, int height) {
  if (width <= 0 || height <= 0 || width * height > 25000000) {
    throw const FormatException('Размеры оригинального фото недопустимы.');
  }
  final copy = Map<String, dynamic>.from(jsonDecode(jsonEncode(report)));
  final images = copy['images'];
  if (images is! Map || images.isEmpty) {
    return copy;
  }
  final source = modelPreviewSize(width, height);
  final actualWidth = images['preview_width'] ?? images['width'];
  final actualHeight = images['preview_height'] ?? images['height'];
  if (actualWidth != width || actualHeight != height) {
    if (actualWidth != source[0] || actualHeight != source[1]) {
      throw const FormatException(
          'Изображение модели не соответствует исходному фото.');
    }
  }
  images['preview_width'] = actualWidth;
  images['preview_height'] = actualHeight;
  images['preview_transform'] = 'uniform_fit_max1600';
  images['width'] = width;
  images['height'] = height;
  return copy;
}

Map<String, dynamic> combinedReferenceSnapshot({
  required Map<String, dynamic>? base,
  required Map<String, dynamic> reference,
  required Map<String, dynamic> manualReport,
  required String capturedAt,
  required Map<String, dynamic>? environment,
}) {
  final previous = base?['report'] as Map?;
  final originalGeometry = previous?['model_geometry'] ??
      ((base?['kind']) != 'reference' ? (previous?['geometry']) : null);
  return {
    'version': 1,
    'kind': 'reference',
    'reference': reference,
    // Keep the model's nested measurements, species and model identity exactly.
    // Projection results have separate top-level fields + geometry/source.
    'report': {
      ...?previous,
      ...manualReport,
      if (originalGeometry is Map) 'model_geometry': originalGeometry,
    },
    'ar': base?['ar'],
    'environment': environment,
    'captured_at': capturedAt,
    'change_source': 'reference',
    if (base?['correction_id'] != null) 'correction_id': base!['correction_id'],
  };
}

/// The source view must not present manually marked reference geometry as AI.
/// Legacy combined snapshots lack the original geometry: omit it honestly.
Map<String, dynamic> modelReportForSources(Map<String, dynamic> snapshot) {
  final copy = Map<String, dynamic>.from(
      jsonDecode(jsonEncode(snapshot['report'] ?? {})));
  if (snapshot['reference'] is Map || snapshot['kind'] == 'reference') {
    final originalGeometry = copy['model_geometry'];
    if (originalGeometry is Map) {
      copy['geometry'] = originalGeometry;
    } else {
      copy.remove('geometry');
    }
    for (final key in [
      'method',
      'height_m',
      'crown_width_m',
      'dbh_m',
      'beta_kg_s'
    ]) {
      copy.remove(key);
    }
  }
  copy.remove('model_geometry');
  copy.remove('contour_preview');
  return copy;
}
