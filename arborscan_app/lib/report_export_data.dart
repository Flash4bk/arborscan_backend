import 'dart:convert';
import 'dart:typed_data';
import 'package:crypto/crypto.dart';
import 'geometry_report.dart';

Map<String, dynamic> exportMap(dynamic v) =>
    v is Map ? Map<String, dynamic>.from(v) : <String, dynamic>{};

// Only allow human-readable fields, never dump provider responses into a PDF.
String exportText(dynamic value) {
  if (value == null) return 'не сохранено';
  var s = value.toString();
  s = s.replaceAll(
      RegExp(r'https?://\S+|file://\S+|Bearer\s+\S+', caseSensitive: false),
      '[ссылка скрыта]');
  s = s.replaceAll(
      RegExp(r'[A-Za-z]:[\\/]\S+|/(?:home|opt|data|storage|tmp)/\S+'),
      '[путь скрыт]');
  s = s.replaceAll(
      RegExp(r'\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b'),
      '[скрыто]');
  s = s.replaceAll(RegExp(r'\b(?:NaN|Infinity|-Infinity)\b'), 'нет данных');
  s = s.replaceAll(
      RegExp(r'(?:token|api_key|password|secret)\s*[:=]\s*\S+',
          caseSensitive: false),
      '[секрет скрыт]');
  return s;
}

String exportNumber(dynamic v) {
  if (v is! num || !v.isFinite) return 'нет данных';
  return v.toStringAsFixed(2).replaceFirst(RegExp(r'\.?0+$'), '');
}

Uint8List? exportDecode(dynamic v) {
  if (v is! String || v.isEmpty) return null;
  try {
    return base64Decode(
        v.startsWith('data:') ? v.substring(v.indexOf(',') + 1) : v);
  } catch (_) {
    return null;
  }
}

class ExportMetric {
  final String label, unit, source, limitation;
  final num? value;
  const ExportMetric(
      this.label, this.value, this.unit, this.source, this.limitation);
  String get display =>
      value == null ? 'нет данных' : '${exportNumber(value)} $unit';
}

/// Deep-copied snapshot of one selected version. No inference or metric math.
class ReportExportData {
  final String _snapshotJson, _recordJson;
  final Uint8List? _photo, _annotation, _mask;
  final bool local;
  ReportExportData(
      {required Map<String, dynamic> snapshot,
      required Map<String, dynamic> record,
      Uint8List? photo,
      Uint8List? annotation,
      Uint8List? mask,
      required this.local})
      : _snapshotJson = jsonEncode(_finiteJson(snapshot)),
        _recordJson = jsonEncode(_finiteJson(record)),
        _photo = photo == null ? null : Uint8List.fromList(photo),
        _annotation =
            annotation == null ? null : Uint8List.fromList(annotation),
        _mask = mask == null ? null : Uint8List.fromList(mask);
  Uint8List? get mask => _mask == null ? null : Uint8List.fromList(_mask!);
  Map<String, dynamic> get snapshot => jsonDecode(_snapshotJson);
  Map<String, dynamic> get record => jsonDecode(_recordJson);
  Map<String, dynamic> get report => exportMap(snapshot['report']);
  Map<String, dynamic> get reference => exportMap(snapshot['reference']);
  Uint8List? get photo => _photo == null
      ? exportDecode(exportMap(snapshot['image'])['original_base64'] ??
          exportMap(report['images'])['original_image_base64'])
      : Uint8List.fromList(_photo!);
  Uint8List? get annotation => _annotation == null
      ? exportDecode(exportMap(report['images'])['annotated_image_base64'] ??
          report['annotated_image_base64'])
      : Uint8List.fromList(_annotation!);
  String get id => exportText(
      record['analysis_id'] ?? report['analysis_id'] ?? 'локальная-запись');
  String get version => exportText(record['version_id'] ??
      'снимок-${sha256.convert(utf8.encode(_snapshotJson)).toString().substring(0, 12)}');
  String get filename {
    String safe(String v) => v
        .replaceAll(RegExp('[^a-zA-Z0-9_-]'), '_')
        .substring(0, v.length.clamp(0, 64));
    return 'ArborScan_${safe(id)}_${safe(version)}.pdf';
  }

  String get method => reference.isNotEmpty
      ? 'Эталон: проекционные размеры'
      : snapshot['kind'] == 'v4'
          ? 'Фото / AR; источник указан для каждого размера'
          : 'Сохранённый отчёт прежнего формата';
  List<ExportMetric> get metrics {
    final r = report,
        g = exportMap(r['geometry']),
        m = exportMap(r['measurements']);
    final ref = reference.isNotEmpty;
    ExportMetric metric(String key, String name, String legacy) {
      final a = exportMap(m[key]);
      final value = ref ? r[legacy] : a['value_m'] ?? r[legacy];
      return ExportMetric(
          name,
          value is num && value.isFinite ? value : null,
          'м',
          exportText(ref ? reference['method'] : a['source'] ?? r['method']),
          ref
              ? 'Проекция; общая глубина и малая перспектива. Полевая точность не подтверждена.'
              : exportText((a['notes'] as List?)?.join('; ') ??
                  'Метод и точность ограничены данными выбранной версии.'));
    }

    final result = <ExportMetric>[
      metric(
          'height',
          r['measurement_method_version'] == 2 &&
                  exportMap(m['height'])['source'] != 'ar' &&
                  !ref
              ? 'Вертикальный размер маски на фото'
              : ref
                  ? 'Высота дерева (проекция)'
                  : 'Высота дерева',
          'height_m'),
      metric('crown_width', ref ? 'Ширина кроны (проекция)' : 'Ширина кроны',
          'crown_width_m'),
      if (!ref)
        metric('trunk_diameter', 'Диаметр ствола (не автоматически DBH)',
            'trunk_diameter_m'),
    ];
    for (final e in GeometryReport.labels.entries) {
      final a = exportMap(g[e.key]);
      final legacyDbh = e.key == 'dbh'
          ? (r['dbh_m'] ??
              (exportMap(m['trunk_diameter'])['standard'] == 'dbh_1_3m'
                  ? exportMap(m['trunk_diameter'])['value_m']
                  : null))
          : null;
      final value = a['value'] ?? legacyDbh;
      result.add(ExportMetric(
          e.value,
          value is num && value.isFinite ? value : null,
          a['unit'] == 'deg'
              ? '°'
              : exportText(a['unit'] == 'm'
                  ? 'м'
                  : a['unit'] ?? (e.key == 'crown_porosity' ? '1' : 'м')),
          exportText(a['method'] ?? a['source'] ?? r['method']),
          legacyDbh != null && a['value'] == null
              ? 'Историческая метка DBH из сохранённой версии; соблюдение полевого протокола этим экспортом не подтверждается.'
              : value == null
                  ? GeometryReport.reasons[e.key]!
                  : exportText(a['limitations'] is List
                      ? (a['limitations'] as List).join('; ')
                      : a['definition'] ??
                          'Фотооценка; полевая точность не подтверждена.')));
    }
    return result;
  }

  List<String> get speciesLines {
    final s = report['species'];
    if (s is! Map) {
      return [
        'Название: ${exportText(s)}',
        'Подтверждение и источник: не сохранены'
      ];
    }
    return [
      'Название: ${exportText(s['display_name'])}',
      'Научное название: ${exportText(s['scientific_name'])}',
      'Источник: ${exportText(s['source'])}',
      'Сохранённое предсказание; подтверждённая метка в этом снимке не зафиксирована.'
    ];
  }

  List<String> get provenance => [
        'Метод: $method',
        'Версия измерительного метода: ${exportText(report['measurement_method_version'] ?? reference['method'])}',
        'Версия модели сегментации: ${exportText(report['segmentation_model_version'])}',
        'Схема отчёта: ${exportText(report['schema_version'] ?? snapshot['version'])}',
        if (snapshot['correction_id'] != null)
          'Ревизия контура: ${exportText(snapshot['correction_id'])}',
        if (reference.isNotEmpty) ...[
          'Высота эталона: ${exportNumber(reference['length_m'])} м',
          'Координаты: ${exportText(reference['coordinates'])}; изображение ${exportText(reference['width'])} × ${exportText(reference['height'])} пикселей',
          'Масштаб: ${exportText(reference['scale_origin'] ?? reference['method'])}',
          'Расстояния до камеры должны быть близки. Перспектива автоматически не исправляется.',
        ],
        if (snapshot['ar'] is Map)
          'AR сохранён отдельно; привязка к фото не доказывает совпадение физического объекта. Геометрия и tracking ограничивают результат.',
        ..._arDetails(exportMap(snapshot['ar'])),
      ];
  List<String> _arDetails(Map<String, dynamic> ar) {
    final result = <String>[];
    void walk(Map<String, dynamic> m) {
      for (final e in m.entries) {
        if ([
              'height_method',
              'dbh_method',
              'crown_method',
              'measurement_height_m',
              'height_m',
              'diameter_m',
              'tracking_state',
              'tracking_quality',
              'measured_at',
              'captured_at',
              'diameter_limitations'
            ].contains(e.key) &&
            e.value is! Map &&
            e.value is! List) {
          result.add('AR / ${e.key}: ${exportText(e.value)}');
        } else if (e.value is Map &&
            ['audit', 'provenance', 'geometry', 'measurement']
                .contains(e.key)) {
          walk(exportMap(e.value));
        }
      }
    }

    walk(ar);
    return result;
  }

  List<String> get environment {
    final env =
        exportMap(snapshot['environment'] ?? report['environment_snapshot']);
    final rows = <String>[];
    void add(String label, dynamic data) {
      if (data == null) return;
      if (data is Map) {
        for (final e in data.entries) {
          if ([
            'lat',
            'lon',
            'latitude',
            'longitude',
            'temperature',
            'temperature_c',
            'wind_speed',
            'wind_speed_m_s',
            'wind_m_s',
            'humidity',
            'description',
            'soil_type',
            'ph',
            'source',
            'retrieved_at',
            'observed_at',
            'unit',
            'value'
          ].contains(e.key)) {
            if (e.value is Map) {
              add('$label / ${e.key}', e.value);
            } else if (e.value is! List) {
              rows.add('$label / ${e.key}: ${exportText(e.value)}');
            }
          }
        }
      }
    }

    for (final key in ['gps', 'location', 'weather', 'soil']) {
      add(key, env[key] ?? report[key]);
    }
    return rows.isEmpty
        ? ['Геолокация, погода и почва: доступные сведения не сохранены.']
        : rows;
  }
}

dynamic _finiteJson(dynamic v) {
  if (v is num && !v.isFinite) return null;
  if (v is Map) {
    return {for (final e in v.entries) e.key.toString(): _finiteJson(e.value)};
  }
  if (v is List) return v.map(_finiteJson).toList();
  return v;
}
