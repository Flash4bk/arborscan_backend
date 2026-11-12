/// Contract of one explicit server-calculated section scenario. No mechanical
/// calculation, species lookup or photo-derived strength is performed here.
const sectionMethodId = 'healthy_circular_section_mor_v1';
bool sectionVersionId(dynamic value) =>
    value is String &&
    RegExp(r'^[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}$')
        .hasMatch(value);
const sectionOrigins = {
  'measured': 'Измерение, указанное пользователем',
  'derived': 'Расчёт из других величин',
  'external_prior': 'Справочные данные',
  'scenario_assumption': 'Сценарное допущение',
};

Map<String, dynamic> sectionInputs(Map<String, dynamic> form) {
  double number(String key) {
    final parsed =
        double.tryParse('${form[key] ?? ''}'.trim().replaceAll(',', '.'));
    if (parsed == null || !parsed.isFinite || parsed <= 0) {
      throw const FormatException(
          'Введите конечные положительные диаметр и прочность.');
    }
    return parsed;
  }

  String source(String key, int max) {
    final value = '${form[key] ?? ''}'.trim();
    if (value.isEmpty || value.length > max) {
      throw FormatException(
          'Укажите источник или определение величины (до $max символов).');
    }
    return value;
  }

  final unit = form['diameter_unit'];
  if (unit != 'm' && unit != 'mm') {
    throw const FormatException('Выберите метры или миллиметры.');
  }
  if (form['assumptions_confirmed'] != true) {
    throw const FormatException('Подтвердите допущения сценария сечения.');
  }
  final d = number('diameter_text') / (unit == 'mm' ? 1000 : 1);
  final mor = number('mor_mpa_text') * 1000000;
  if (!d.isFinite || d <= 0 || !mor.isFinite || mor <= 0) {
    throw const FormatException(
        'Значения вне поддержанного численного диапазона.');
  }
  for (final key in ['diameter_origin', 'mor_origin']) {
    if (!sectionOrigins.containsKey(form[key])) {
      throw const FormatException('Укажите происхождение каждой величины.');
    }
  }
  return {
    'schema_version': 1,
    'diameter': {
      'value': d,
      'unit': 'm',
      'origin': form['diameter_origin'],
      'source': source('diameter_source', 512),
      'uncertainty': null
    },
    'effective_mor': {
      'value': mor,
      'unit': 'Pa',
      'origin': form['mor_origin'],
      'source': source('mor_source', 512),
      'uncertainty': null
    },
    'section_model': 'healthy_solid_circular',
    'strength_definition': source('strength_definition', 1024),
  };
}

bool sameScienceInputs(dynamic a, dynamic b) {
  if (a is num && b is num) return a.isFinite && b.isFinite && a == b;
  if (a is Map && b is Map) {
    return a.length == b.length &&
        a.keys.every(
            (key) => b.containsKey(key) && sameScienceInputs(a[key], b[key]));
  }
  if (a is List && b is List) {
    return a.length == b.length &&
        List.generate(a.length, (i) => i)
            .every((i) => sameScienceInputs(a[i], b[i]));
  }
  return a == b;
}

class SectionScienceResult {
  final Map<String, dynamic> raw;
  final double capacityNm;
  const SectionScienceResult._(this.raw, this.capacityNm);

  static SectionScienceResult? parse(dynamic value) {
    if (value is! Map || value.keys.any((key) => key is! String)) return null;
    final map = Map<String, dynamic>.from(value);
    final capacity = map['capacity_nm'];
    final inputs = map['input_snapshot'];
    bool parameter(dynamic input, String unit) =>
        input is Map &&
        input['unit'] == unit &&
        input['value'] is num &&
        (input['value'] as num).isFinite &&
        (input['value'] as num) > 0 &&
        input['source'] is String &&
        (input['source'] as String).trim().isNotEmpty;
    if (map['schema_version'] != 1 ||
        map['method_id'] != sectionMethodId ||
        map['quantity'] != 'single_section_bending_capacity' ||
        map['unit'] != 'N m' ||
        map['validation_status'] != 'scenario_software_only' ||
        map['field_validated'] != false ||
        map['method_version'] != '1.0.0' ||
        map['source_doi'] != '10.1016/j.envsoft.2015.01.016' ||
        capacity is! num ||
        !capacity.isFinite ||
        capacity <= 0 ||
        inputs is! Map ||
        inputs['schema_version'] != 1 ||
        inputs['strength_definition'] is! String ||
        (inputs['strength_definition'] as String).trim().isEmpty ||
        inputs['section_model'] != 'healthy_solid_circular' ||
        !parameter(inputs['diameter'], 'm') ||
        !parameter(inputs['effective_mor'], 'Pa')) {
      return null;
    }
    for (final key in [
      'input_sha256',
      'result_sha256',
      'implementation_sha256',
      'source_payload_sha256',
      'image_sha256'
    ]) {
      if (map[key] is! String ||
          !RegExp(r'^[a-f0-9]{64}$').hasMatch(map[key])) {
        return null;
      }
    }
    for (final key in ['result_id', 'source_version_id']) {
      if (!sectionVersionId(map[key])) return null;
    }
    return SectionScienceResult._(map, capacity.toDouble());
  }

  List<String> get lines {
    final inputs = raw['input_snapshot'] as Map;
    final diameter = inputs['diameter'] as Map;
    final strength = inputs['effective_mor'] as Map;
    return [
      'Расчётный предельный момент заданного сечения: ${capacityNm.toStringAsFixed(2)} Н·м.',
      'Сценарий: здоровое сплошное круглое сечение; M = effective MOR × π × d³ / 32.',
      'Это не устойчивость дерева целиком, не критический ветер, не β и не оценка безопасности.',
      'Диаметр: ${diameter['value']} м. Источник: ${diameter['source']}.',
      'Effective MOR: ${strength['value']} Па. Источник: ${strength['source']}.',
      'Определение прочности и учтённые поправки: ${inputs['strength_definition']}.',
      'Входы указаны пользователем и не проверены полевым экспериментом. Неопределённость не определена.',
      'Метод: ${raw['method_id']}; версия: ${raw['method_version']}.',
      'Исходная серверная версия: ${raw['source_version_id']}.',
      'SHA входов: ${raw['input_sha256']}',
      'SHA результата: ${raw['result_sha256']}',
      'SHA реализации: ${raw['implementation_sha256']}',
      'SHA исходного снимка: ${raw['source_payload_sha256']}',
      'Источник метода: DOI ${raw['source_doi']}.',
    ];
  }
}
