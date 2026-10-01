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

num? exportNumeric(dynamic value) {
  final parsed = value is num
      ? value
      : value is String
          ? num.tryParse(value.trim())
          : null;
  return parsed != null && parsed.isFinite ? parsed : null;
}

String exportMethodLabel(dynamic value) {
  final code = value?.toString();
  if (code == null || code.isEmpty) return 'Источник не сохранён';
  if (code.startsWith('known_object_segment_v')) {
    return 'По известному объекту (проекция на фото)';
  }
  return const {
        'ar': 'AR-измерение',
        'ar+vision': 'Фото с масштабом AR',
        'reference+vision': 'Фото с масштабом эталона',
        'manual_scale+vision': 'Фото с сохранённым ручным масштабом',
        'image': 'Фото и модель',
        'photo': 'Фото',
        'vision': 'Изображение без подтверждённого масштаба',
        'reference': 'По известному объекту',
        'reference_object': 'Масштаб по известному объекту',
        'reference_scaled': 'Фото с масштабом эталона',
        'reference_calibrated': 'Фото с масштабом эталона',
        'ar_scaled': 'Фото с масштабом AR',
        'ar_calibrated': 'Фото с масштабом AR',
        'legacy_manual_scale': 'Сохранённый ручной масштаб',
        'manual': 'Введено вручную',
        'unavailable': 'Измерение недоступно',
        'fixed_vertical_tree_plane_ray_intersection_v1':
            'AR: лучи в вертикальной плоскости дерева',
        'cylindrical_tangent_rays_median_v2':
            'AR: касательные лучи к сечению ствола',
        'not_measured_requires_explicit_photo_calibration':
            'Не измерено; нужен масштаб фотографии',
        'vertical_reference_same_depth_user_confirmed':
            'Вертикальный эталон на той же глубине (подтверждено пользователем)',
      }[code] ??
      'Метод указан в служебных сведениях';
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
          exportMap(report['images'])['original_image_base64'] ??
          report['original_image_base64'] ??
          report['image_base64'])
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
      final aliases = {
        'height_m': 'height',
        'crown_width_m': 'crown',
        'trunk_diameter_m': 'trunk'
      };
      final raw =
          ref ? r[legacy] : a['value_m'] ?? r[legacy] ?? r[aliases[legacy]];
      final value = exportNumeric(raw);
      final rawSource = ref
          ? reference['method']
          : a['source'] ??
              exportMap(r['measurement_sources'])[legacy] ??
              r['method'] ??
              r['dimensions_source'];
      return ExportMetric(
          name,
          value,
          'м',
          exportMethodLabel(rawSource),
          raw != null && value == null
              ? 'Поле $legacy присутствует, но его формат не поддерживается экспортом. Требуется проверка исходной версии.'
              : ref
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
      final rawValue = a['value'] ?? legacyDbh;
      final value = exportNumeric(rawValue);
      result.add(ExportMetric(
          e.value,
          value,
          a['unit'] == 'deg'
              ? '°'
              : exportText(a['unit'] == 'm'
                  ? 'м'
                  : a['unit'] ?? (e.key == 'crown_porosity' ? '1' : 'м')),
          exportMethodLabel(a['method'] ?? a['source'] ?? r['method']),
          rawValue != null && value == null
              ? 'Поле geometry.${e.key} присутствует, но его формат не поддерживается экспортом.'
              : legacyDbh != null && a['value'] == null
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
        if (report['species_scientific_name'] != null ||
            exportMap(report['classification'])['scientific_name'] != null)
          'Научное название: ${exportText(report['species_scientific_name'] ?? exportMap(report['classification'])['scientific_name'])}',
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
          'Масштаб: ${exportMethodLabel(reference['scale_origin'] ?? reference['method'])}',
          'Код масштаба: ${exportText(reference['scale_origin'] ?? reference['method'])}',
          'Расстояния до камеры должны быть близки. Перспектива автоматически не исправляется.',
        ],
        if (snapshot['ar'] is Map)
          'AR сохранён отдельно; привязка к фото не доказывает совпадение физического объекта. Геометрия и tracking ограничивают результат.',
        ..._metricCodes,
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

  List<String> get historicalValues {
    final rows = <String>[];
    void saved(String name, dynamic value, String unit) {
      if (value == null) return;
      rows.add(
          '$name: ${exportNumeric(value) == null ? exportText(value) : exportNumber(exportNumeric(value))}${unit.isEmpty ? '' : ' $unit'}');
    }

    saved('Сохранённый масштаб', report['scale_px_to_m'], 'м/пиксель');
    saved('Прежний угол наклона', report['lean_angle_deg'], '°');
    final risk = exportMap(report['risk']);
    saved('Прежний индекс риска', risk['index'] ?? report['risk_index'], '');
    saved('Прежняя категория риска',
        risk['category'] ?? report['risk_category'], '');
    for (final line
        in risk['explanation'] is List ? risk['explanation'] as List : []) {
      rows.add('Прежнее пояснение: ${exportText(line)}');
    }
    final beta = exportMap(report['beta']);
    saved(
        'Прежнее значение β', beta['beta_kg_s'] ?? report['beta_kg_s'], 'кг/с');
    saved('Прежний сценарий β', beta['beta_max_scenario'], 'кг/с');
    saved('Источник прежнего β', beta['source'], '');
    saved('Код прежнего метода β', beta['method'], '');
    saved('Сохранённая сила ветра', beta['wind_force_n'], 'Н');
    final mechanical =
        exportMap(exportMap(report['analytic_wind_model'])['outputs']);
    saved('Прежняя суммарная сила', mechanical['total_force_n'], 'Н');
    saved('Прежний центр нагрузки', mechanical['center_of_load_m'], 'м');
    saved('Прежний момент у основания', mechanical['base_moment_nm'], 'Н·м');
    saved('Прежний механический индекс', mechanical['analytical_score'], '');
    return rows;
  }

  /// Approved public attribution destinations, never a URL from a report.
  Map<String, String> get environmentAttributionLinks {
    final env =
        exportMap(snapshot['environment'] ?? report['environment_snapshot']);
    return {
      if (exportMap(env['weather'] ?? report['weather'])['source'] ==
          'OpenWeather')
        'OpenWeather': 'https://openweathermap.org/',
      if (exportMap(env['soil'] ?? report['soil'])['source'] ==
          'SoilGrids') ...{
        'ISRIC SoilGrids': 'https://soilgrids.org/',
        'Лицензия SoilGrids CC BY 4.0':
            'https://creativecommons.org/licenses/by/4.0/',
      },
    };
  }

  List<String> get _metricCodes {
    final rows = <String>[];
    final measurements = exportMap(report['measurements']);
    final sources = exportMap(report['measurement_sources']);
    for (final entry in measurements.entries) {
      final value = exportMap(entry.value);
      if (value['source'] != null) {
        rows.add('Код источника ${entry.key}: ${exportText(value['source'])}');
      }
    }
    for (final entry in sources.entries) {
      rows.add('Код источника ${entry.key}: ${exportText(entry.value)}');
    }
    for (final entry in exportMap(report['geometry']).entries) {
      final value = exportMap(entry.value);
      if (value['method'] != null) {
        rows.add('Код метода ${entry.key}: ${exportText(value['method'])}');
      }
    }
    if (report['method'] != null) {
      rows.add('Код метода отчёта: ${exportText(report['method'])}');
    }
    return rows;
  }

  List<String> get environment {
    final env =
        exportMap(snapshot['environment'] ?? report['environment_snapshot']);
    final rows = <String>[];
    final gps = exportMap(env['gps'] ?? env['location'] ?? report['gps']);
    final coordinate = gps['value'] is Map ? exportMap(gps['value']) : gps;
    final lat = exportNumeric(coordinate['lat'] ?? coordinate['latitude']);
    final lon = exportNumeric(coordinate['lon'] ?? coordinate['longitude']);
    if (lat != null &&
        lon != null &&
        lat >= -90 &&
        lat <= 90 &&
        lon >= -180 &&
        lon <= 180 &&
        (coordinate['crs'] == null || coordinate['crs'] == 'EPSG:4326')) {
      rows.add('Координаты: $lat, $lon (EPSG:4326)');
      rows.add('Источник положения: ${const {
            'exif': 'EXIF оригинала',
            'device': 'устройство',
            'manual': 'выбор на карте'
          }[gps['source']] ?? 'не сохранён в прежнем формате'}');
      rows.add(gps['position_kind'] == 'tree'
          ? 'Отмечено положение дерева.'
          : 'Положение камеры / устройства; точная точка дерева не подтверждена.');
      if (exportNumeric(gps['accuracy_m']) != null) {
        rows.add(
            'Указанная устройством точность: ±${exportNumber(exportNumeric(gps['accuracy_m']))} м');
      }
      if (gps['is_approximate'] == true) {
        rows.add('Разрешена приблизительная геолокация.');
      }
      if (gps['is_last_known'] == true) {
        rows.add(
            'Последнее известное положение, подтверждено при выборе. Это не новый GPS-замер.');
      }
      if (gps['observed_at'] != null) {
        rows.add('Время положения: ${exportText(gps['observed_at'])}');
      }
      if (gps['captured_local'] != null) {
        rows.add(
            'Время EXIF: ${exportText(gps['captured_local'])}; часовой пояс неизвестен.');
      }
      if (gps['retrieved_at'] != null) {
        rows.add('Координаты получены: ${exportText(gps['retrieved_at'])}');
      }
    } else {
      rows.add(gps.isEmpty
          ? 'Место не сохранено.'
          : 'Координаты присутствуют, но формат или диапазон не поддерживается. Точка на карте не восстановлена.');
    }
    if (report['address'] is String &&
        (report['address'] as String).isNotEmpty) {
      rows.add('Сохранённый адрес: ${exportText(report['address'])}');
    }
    for (final key in ['weather', 'soil']) {
      final item = exportMap(env[key] ?? report[key]);
      final name = key == 'weather' ? 'Погода' : 'Почва';
      if (item.isEmpty) {
        rows.add('$name: сведения не сохранены.');
        continue;
      }
      final value = item['value'] is Map ? exportMap(item['value']) : item;
      rows.add('$name / источник: ${exportText(item['source'])}');
      rows.add(
          '$name / получено: ${exportText(item['retrieved_at'])}${item['cached'] == true ? ' (из кеша)' : ''}');
      rows.add(
          '$name / время данных: ${exportText(item['data_at'] ?? item['observed_at'])}');
      final requested = exportMap(item['request_point']);
      if (requested.isNotEmpty) {
        rows.add(
            '$name / точка запроса: ${exportText(requested['lat'])}, ${exportText(requested['lon'])}');
      }
      if (item['status'] == 'unavailable') {
        rows.add(
            '$name: ${_environmentReason(item['reason'])} (код: ${exportText(item['reason'])}).');
      } else if (key == 'weather') {
        rows.add(item['kind'] == 'current'
            ? 'Текущие условия на момент запроса; не погода на дату старого снимка.'
            : 'Тип погоды (наблюдение, прогноз или архив) в прежнем формате не сохранён.');
        const fields = {
          'temperature_c': ['Температура', '°C'],
          'wind_speed_m_s': ['Ветер', 'м/с'],
          'wind_m_s': ['Ветер', 'м/с'],
          'wind_gust_m_s': ['Порывы', 'м/с'],
          'wind_direction_deg': ['Направление ветра', '°'],
          'pressure_hpa': ['Давление', 'гПа'],
          'relative_humidity_pct': ['Влажность', '%'],
        };
        for (final entry in fields.entries) {
          if (!value.containsKey(entry.key)) continue;
          rows.add(
              '${entry.value[0]}: ${exportNumeric(value[entry.key]) == null ? 'нет данных' : '${exportNumber(exportNumeric(value[entry.key]))} ${entry.value[1]}'}');
        }
        for (final field in const {
          'temperature': 'Температура',
          'wind_speed': 'Ветер',
          'humidity': 'Влажность',
          'description': 'Описание'
        }.entries) {
          if (value[field.key] != null) {
            rows.add(
                '${field.value}: ${exportText(value[field.key])}${field.key == 'description' ? '' : ' (единица: ${exportText(item['unit'] ?? value['unit'])})'}');
          }
        }
      } else {
        rows.add(
            'Модель сетки: не проба почвы у корня дерева. Не определяет состояние корней и устойчивость.');
        if (item['resolution_m'] != null) {
          rows.add(
              'Размер ячейки: ${exportNumber(exportNumeric(item['resolution_m']))} м');
        }
        if (item['dataset_version'] != null) {
          rows.add('Версия набора: ${exportText(item['dataset_version'])}');
        }
        if (item['access_method'] != null) {
          rows.add('Способ получения: ${exportText(item['access_method'])}');
        }
        for (final raw
            in value['properties'] is List ? value['properties'] as List : []) {
          final property = exportMap(raw);
          final label = const {
                'clay': 'Глина',
                'sand': 'Песок',
                'silt': 'Ил',
                'soc': 'Органический углерод',
                'phh2o': 'pH (вода)'
              }[property['name']] ??
              exportText(property['name']);
          final unit = property['unit'] == 'g/kg'
              ? 'г/кг'
              : exportText(property['unit']);
          final depth = property['depth_cm'] is List
              ? (property['depth_cm'] as List).map(exportText).join('-')
              : 'не указан';
          rows.add(
              '$label: ${exportNumber(exportNumeric(property['value']))} $unit; слой $depth см');
          rows.add(
              '$label / квантили 0,05-0,95: ${exportNumber(exportNumeric(property['q05']))} - ${exportNumber(exportNumeric(property['q95']))} $unit');
        }
        for (final field
            in const {'soil_type': 'Тип почвы', 'ph': 'pH'}.entries) {
          if (value[field.key] != null) {
            rows.add('${field.value}: ${exportText(value[field.key])}');
          }
        }
      }
      if (item['status'] == 'partial') {
        rows.add(
            '$name: получена часть параметров, отсутствующие значения не равны нулю.');
      }
      if (item['attribution'] != null) {
        rows.add('Атрибуция: ${exportText(item['attribution'])}');
      }
      for (final limitation
          in item['limitations'] is List ? item['limitations'] as List : []) {
        rows.add(exportText(limitation));
      }
    }
    return rows;
  }
}

String _environmentReason(dynamic reason) =>
    const {
      'provider_not_configured': 'источник не настроен',
      'provider_timeout': 'источник не ответил вовремя',
      'provider_rate_limited': 'лимит запросов источника',
      'provider_busy': 'источник занят',
      'provider_authentication': 'источник не предоставил доступ',
      'no_coverage': 'для точки нет данных',
      'provider_invalid_response': 'неподдерживаемый ответ источника',
    }[reason] ??
    'источник недоступен';

dynamic _finiteJson(dynamic v) {
  if (v is num && !v.isFinite) return null;
  if (v is Map) {
    return {for (final e in v.entries) e.key.toString(): _finiteJson(e.value)};
  }
  if (v is List) return v.map(_finiteJson).toList();
  return v;
}
