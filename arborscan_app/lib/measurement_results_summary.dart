import 'package:flutter/material.dart';

import 'geometry_report.dart';
import 'report_export_data.dart';

/// Presents the selected frozen version; never runs inference or metric math.
class MeasurementResultsSummary extends StatelessWidget {
  final Map<String, dynamic> snapshot;

  const MeasurementResultsSummary({super.key, required this.snapshot});

  @override
  Widget build(BuildContext context) {
    // Share the PDF's snapshot copying and reference/legacy selection. Explicit
    // nested measurements remain authoritative even when unavailable.
    // Keep full-resolution base64 photos/masks out of repeated UI JSON copies.
    // Selection and units still use the same frozen metric fields as the PDF.
    final inputReport = exportMap(snapshot['report']);
    final inputReference = exportMap(snapshot['reference']);
    final data = ReportExportData(
      snapshot: {
        'kind': snapshot['kind'],
        'report': {
          for (final key in [
            'method',
            'dimensions_source',
            'measurement_method_version',
            'height_m',
            'crown_width_m',
            'trunk_diameter_m',
            'height',
            'crown',
            'trunk',
            'measurement_sources',
            'measurements',
            'geometry',
          ])
            if (inputReport.containsKey(key)) key: inputReport[key],
        },
        if (inputReference.isNotEmpty)
          'reference': {
            'method': inputReference['method'],
            'version': inputReference['version'],
          },
      },
      record: const {},
      local: true,
    );
    final report = data.report;
    final reference = data.reference.isNotEmpty;
    final metrics = data.metrics;
    final geometry = exportMap(report['geometry']);
    final measurements = exportMap(report['measurements']);
    final legacy = data.snapshot['kind'] == 'legacy';

    ExportMetric primary(int index, String key) {
      final metric = metrics[index];
      if (reference) return metric;
      final input = exportMap(measurements[key]);
      if (!measurements.containsKey(key)) {
        return ExportMetric(
          metric.label,
          metric.source == exportMethodLabel('unavailable')
              ? null
              : metric.value,
          metric.unit,
          'Сохранённое значение · ${metric.source}',
          metric.limitation,
        );
      }
      final source = input['source']?.toString();
      // value_m is an SI field. An explicitly conflicting unit is unsupported.
      final available = source != null &&
          source.isNotEmpty &&
          source != 'unavailable' &&
          source != 'vision' &&
          input['available'] != false &&
          (input['unit'] == null || input['unit'] == 'm');
      return ExportMetric(
        metric.label,
        available ? exportNumeric(input['value_m']) : null,
        'м',
        exportMethodLabel(source),
        metric.limitation,
      );
    }

    final height = primary(0, 'height');
    final crown = primary(1, 'crown_width');
    final trunk = reference
        ? metrics.firstWhere(
            (m) => m.label == GeometryReport.labels['trunk_diameter'],
          )
        : primary(2, 'trunk_diameter');
    final trunkUnit =
        reference ? exportMap(geometry['trunk_diameter'])['unit'] : 'm';
    final additional = <Widget>[];
    for (final entry in GeometryReport.labels.entries) {
      if (entry.key == 'trunk_diameter') continue;
      final raw = exportMap(geometry[entry.key]);
      if (raw['source'] == 'unavailable' || raw['available'] == false) continue;
      final value = exportNumeric(raw['value']);
      final expectedUnit = entry.key == 'trunk_lean'
          ? 'deg'
          : entry.key == 'crown_porosity'
              ? '1'
              : 'm';
      if (value == null || raw['unit'] != expectedUnit || value < 0) continue;
      if (entry.key == 'crown_porosity' && value > 1) continue;
      if (expectedUnit == 'm' && value == 0) continue;
      final metric = metrics.firstWhere((m) => m.label == entry.value);
      additional.add(
        _SummaryRow(
          label: entry.key == 'dbh' ? 'DBH (сохранённая метка)' : entry.value,
          value: _number(value, 2, metric.unit),
          source: reference ? null : metric.source,
        ),
      );
    }

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Результат измерения и расчётов',
              style: Theme.of(context).textTheme.titleLarge,
            ),
            const SizedBox(height: 6),
            Text(
              reference
                  ? 'Эталон · фотооценка проекции'
                  : legacy
                      ? 'Сохранённые размеры прежнего формата'
                      : 'Фото / AR · источник указан у каждого размера',
              style: Theme.of(context).textTheme.bodySmall,
            ),
            const SizedBox(height: 10),
            _SummaryRow(
              key: const ValueKey('measurement-summary-height'),
              label: height.label,
              value: _positive(height.value)
                  ? _number(height.value!, 2, 'м')
                  : 'нет данных · нужен масштаб и измерение',
              source:
                  !reference && _positive(height.value) ? height.source : null,
            ),
            _SummaryRow(
              key: const ValueKey('measurement-summary-crown'),
              label: 'Ширина кроны',
              value: _positive(crown.value)
                  ? _number(crown.value!, 2, 'м')
                  : 'нет данных · нужна разметка',
              source:
                  !reference && _positive(crown.value) ? crown.source : null,
            ),
            _SummaryRow(
              key: const ValueKey('measurement-summary-trunk'),
              label: 'Диаметр отмеченного сечения ствола',
              value: _positive(trunk.value) &&
                      trunkUnit == 'm' &&
                      (!reference ||
                          (exportMap(geometry['trunk_diameter'])['source'] !=
                                  'unavailable' &&
                              exportMap(geometry['trunk_diameter'])[
                                      'available'] !=
                                  false))
                  ? _number(trunk.value!, 3, 'м')
                  : 'нет данных · нужна разметка',
              source:
                  !reference && _positive(trunk.value) ? trunk.source : null,
            ),
            const Divider(height: 24),
            const _SummaryRow(
              key: ValueKey('measurement-summary-beta'),
              label: 'Коэффициент β (кг/с)',
              value: 'не рассчитан',
            ),
            const _SummaryRow(
              key: ValueKey('measurement-summary-critical-wind'),
              label: 'Критическая скорость ветра (м/с)',
              value: 'не рассчитана',
            ),
            const _SummaryRow(
              key: ValueKey('measurement-summary-damage-probability'),
              label: 'Вероятность повреждения',
              value: 'не рассчитана',
            ),
            if (additional.isNotEmpty)
              ExpansionTile(
                tilePadding: EdgeInsets.zero,
                title: const Text('Другие измеренные параметры'),
                children: additional,
              ),
            const ExpansionTile(
              tilePadding: EdgeInsets.zero,
              childrenPadding: EdgeInsets.only(bottom: 8),
              title: Text('Что нужно для расчётов'),
              children: [
                Text(
                  'Для β нужны измеренные траектории движения во времени, массы, упругость и начальные условия дерева, а также проверенная динамическая модель.',
                ),
                SizedBox(height: 8),
                Text(
                  'Для критического ветра нужны параметры материала, корней и основания, ветровая нагрузка и проверенная модель разрушения или опрокидывания.',
                ),
                SizedBox(height: 8),
                Text(
                  'Для вероятности повреждения нужны реальные наблюдения, заданные условия и период оценки, экспериментальная валидация вероятностной модели.',
                ),
                SizedBox(height: 8),
                Text(
                  'Фото даёт проекции. Диаметр отмеченного сечения не подтверждает DBH; нужен полевой протокол. Полевая точность не подтверждена. Сохранённые показатели прежних методов не используются как новые расчёты.',
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  static bool _positive(num? value) =>
      value != null && value.isFinite && value > 0;

  static String _number(num value, int digits, String unit) =>
      '${value.toStringAsFixed(digits).replaceAll('.', ',')} $unit';
}

class _SummaryRow extends StatelessWidget {
  final String label, value;
  final String? source;

  const _SummaryRow(
      {super.key, required this.label, required this.value, this.source});

  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 6),
        child: LayoutBuilder(
          builder: (context, constraints) {
            final vertical = constraints.maxWidth < 300 ||
                MediaQuery.textScalerOf(context).scale(14) > 21;
            final description = Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(label),
                if (source != null)
                  Text(source!, style: Theme.of(context).textTheme.bodySmall),
              ],
            );
            final result = Text(
              value,
              style: Theme.of(
                context,
              ).textTheme.bodyMedium?.copyWith(fontWeight: FontWeight.w600),
              textAlign: vertical ? TextAlign.start : TextAlign.end,
            );
            return vertical
                ? Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [description, const SizedBox(height: 3), result],
                  )
                : Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Expanded(flex: 3, child: description),
                      const SizedBox(width: 12),
                      Expanded(flex: 2, child: result),
                    ],
                  );
          },
        ),
      );
}
