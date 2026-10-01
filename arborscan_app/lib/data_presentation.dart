import 'dart:typed_data';

import 'package:flutter/material.dart';

import 'app_theme.dart';

/// Presentation only. The original server code stays available in details.
String dataReasonLabel(String? code) {
  if (code == null || code.isEmpty) return 'Допущено';
  final kind = code.split(':').first;
  if (kind == 'accepted_revision_required') {
    return switch (code.split(':').skip(1).join(':')) {
      'submitted' => 'На проверке — нужна принятая ревизия',
      'draft' => 'Черновик — отправьте контур на проверку',
      'rejected' => 'Отклонено — исправьте контур',
      'pending_review' => 'Старая запись — требуется проверка',
      _ => 'Требуется принятая ревизия контура',
    };
  }
  return const {
        'separate_confirmed_species_label_required':
            'Нужно отдельное подтверждение вида',
        'confirmed_taxon_identifier_required':
            'Не указан подтверждённый таксон',
        'polygon_loss': 'Контур теряет детали при экспорте',
        'empty_mask': 'Маска пуста',
        'polygon_complexity_limit_original_retained':
            'Контур слишком сложен для экспорта; оригинал сохранён',
        'polygon_degenerate_or_too_complex': 'Непригодная геометрия контура',
        'oriented_dimensions_mismatch': 'Размер маски не совпадает с фото',
        'revision_original_mismatch': 'Ревизия не соответствует оригиналу',
        'original_checksum_mismatch': 'Не совпала контрольная сумма фото',
        'mask_checksum_mismatch': 'Не совпала контрольная сумма маски',
        'image_too_large': 'Фото превышает допустимый размер',
        'synthetic_export_smoke_not_evaluation_data':
            'Тестовый пример не допускается к обучению',
        'exact_duplicate_or_other_revision': 'Дубликат или другая ревизия фото',
        'snapshot_memory_budget_deferred_use_smaller_selection':
            'Превышен объём набора — уменьшите выборку',
      }[kind] ??
      'Запись исключена — подробности доступны ниже';
}

String modelJobLabel(String? state) =>
    const {
      'queued': 'В очереди',
      'running': 'Выполняется',
      'cancel_requested': 'Отмена запрошена',
      'cancelled': 'Отменено',
      'succeeded': 'Завершено',
      'completed': 'Завершено',
      'failed': 'Ошибка выполнения',
      'interrupted': 'Прервано',
    }[state] ??
    'Статус задачи неизвестен';

class DatasetSummaryCard extends StatelessWidget {
  final int included, excluded;
  const DatasetSummaryCard(
      {super.key, required this.included, required this.excluded});

  @override
  Widget build(BuildContext context) => Semantics(
        label: 'Допущено: $included. Исключено: $excluded.',
        child: Container(
          padding: const EdgeInsets.all(22),
          decoration: BoxDecoration(
            color: AppTheme.primary,
            borderRadius: BorderRadius.circular(22),
          ),
          child: LayoutBuilder(builder: (context, constraints) {
            final compact = constraints.maxWidth < 280 ||
                MediaQuery.textScalerOf(context).scale(1) > 1.5;
            Widget count(int value, String label, IconData icon) => Row(
                  children: [
                    Icon(icon, color: Colors.white, size: 23),
                    const SizedBox(width: 12),
                    Expanded(
                        child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text('$value',
                            style: Theme.of(context)
                                .textTheme
                                .headlineLarge
                                ?.copyWith(
                                    color: Colors.white,
                                    fontWeight: FontWeight.w700)),
                        Text(label,
                            style: const TextStyle(color: Colors.white)),
                      ],
                    )),
                  ],
                );
            final a = count(included, 'Допущено', Icons.check_circle_outline);
            final b = count(excluded, 'Исключено', Icons.block);
            return compact
                ? Column(children: [
                    a,
                    const Divider(color: Colors.white38, height: 24),
                    b
                  ])
                : Row(children: [
                    Expanded(child: a),
                    const SizedBox(width: 20),
                    Expanded(child: b)
                  ]);
          }),
        ),
      );
}

/// Loads only visible private photos and never reuses them across row identities.
class DataRevisionTile extends StatefulWidget {
  final String title;
  final bool eligible;
  final String? reason;
  final Future<Uint8List?> Function() loadImage;
  final List<Widget> details;
  const DataRevisionTile(
      {super.key,
      required this.title,
      required this.eligible,
      required this.reason,
      required this.loadImage,
      required this.details});

  @override
  State<DataRevisionTile> createState() => _DataRevisionTileState();
}

class _DataRevisionTileState extends State<DataRevisionTile> {
  late Future<Uint8List?> _image = widget.loadImage();

  @override
  Widget build(BuildContext context) {
    final pending = widget.reason?.contains(':submitted') == true ||
        widget.reason?.contains(':pending_review') == true;
    final color = widget.eligible
        ? AppTheme.success
        : pending
            ? AppTheme.warning
            : AppTheme.danger;
    return ExpansionTile(
      tilePadding: const EdgeInsets.symmetric(vertical: 6),
      childrenPadding: const EdgeInsets.only(bottom: 12),
      leading: ClipRRect(
        borderRadius: BorderRadius.circular(12),
        child: SizedBox(
          width: 72,
          height: 72,
          child: FutureBuilder<Uint8List?>(
              future: _image,
              builder: (context, snapshot) {
                if (snapshot.hasData) {
                  return Image.memory(snapshot.data!,
                      fit: BoxFit.cover,
                      cacheWidth: 180,
                      errorBuilder: (_, __, ___) =>
                          const Icon(Icons.broken_image_outlined));
                }
                return ColoredBox(
                    color: AppTheme.surface2,
                    child: IconButton(
                      tooltip: snapshot.hasError
                          ? 'Повторить загрузку фото'
                          : 'Фото записи',
                      icon: Icon(snapshot.hasError
                          ? Icons.refresh
                          : Icons.image_outlined),
                      onPressed: snapshot.hasError
                          ? () => setState(() => _image = widget.loadImage())
                          : null,
                    ));
              }),
        ),
      ),
      title: Text(widget.title, style: Theme.of(context).textTheme.titleMedium),
      subtitle: Padding(
          padding: const EdgeInsets.only(top: 6),
          child: Text.rich(
              TextSpan(children: [
                WidgetSpan(
                    alignment: PlaceholderAlignment.middle,
                    child: Padding(
                        padding: const EdgeInsets.only(right: 5),
                        child: Icon(
                            widget.eligible
                                ? Icons.check_circle
                                : pending
                                    ? Icons.schedule
                                    : Icons.block,
                            size: 18,
                            color: color))),
                TextSpan(text: dataReasonLabel(widget.reason)),
              ]),
              style: TextStyle(color: color))),
      children: widget.details,
    );
  }
}
