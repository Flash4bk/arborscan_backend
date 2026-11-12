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
    final data = ReportExportData.measurementView(snapshot);
    final report = data.report;
    final reference = data.reference.isNotEmpty;
    final metrics = data.metrics;
    final geometry = exportMap(report['geometry']);
    final legacy = data.snapshot['kind'] == 'legacy';
    final height = metrics[0];
    final crown = metrics[1];
    final trunk = reference
        ? metrics.firstWhere(
            (m) => m.label == GeometryReport.labels['trunk_diameter'],
          )
        : metrics[2];
    final trunkUnit =
        reference ? exportMap(geometry['trunk_diameter'])['unit'] : 'm';
    final additional = <Widget>[];
    for (final entry in GeometryReport.labels.entries) {
      if (entry.key == 'trunk_diameter') continue;
      final metric = metrics.firstWhere((m) => m.label == entry.value);
      final value = metric.value;
      if (value == null) continue;
      additional.add(
        _SummaryRow(
          label: entry.key == 'dbh' ? 'DBH (сохранённая метка)' : entry.value,
          value: _number(value, 2, metric.unit),
          source: reference ? null : metric.source,
        ),
      );
    }

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Результат измерений и расчётов',
            style: Theme.of(context).textTheme.titleMedium,
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
            source: !reference && _positive(crown.value) ? crown.source : null,
          ),
          _SummaryRow(
            key: const ValueKey('measurement-summary-trunk'),
            label: 'Диаметр отмеченного сечения ствола',
            value: _positive(trunk.value) &&
                    trunkUnit == 'm' &&
                    (!reference ||
                        (exportMap(geometry['trunk_diameter'])['source'] !=
                                'unavailable' &&
                            exportMap(
                                    geometry['trunk_diameter'])['available'] !=
                                false))
                ? _number(trunk.value!, 3, 'м')
                : 'нет данных · нужна разметка',
            source: !reference && _positive(trunk.value) ? trunk.source : null,
          ),
          const SizedBox(height: 6),
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
        padding: const EdgeInsets.symmetric(vertical: 2),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // One flowing paragraph keeps a value beside its name. It wraps
            // naturally at larger text scales instead of splitting the pair
            // into distant columns or clipping a long source label.
            Text.rich(
              TextSpan(text: '$label: ', children: [
                TextSpan(
                    text: value,
                    style: const TextStyle(fontWeight: FontWeight.w600)),
              ]),
              style: Theme.of(context).textTheme.bodyMedium,
            ),
            if (source != null)
              Text(source!, style: Theme.of(context).textTheme.bodySmall),
          ],
        ),
      );
}
