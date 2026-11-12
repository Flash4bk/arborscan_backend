import 'package:flutter/material.dart';
import 'science_section.dart';

/// A separate server scenario, never a replacement for the photo measurements.
class ScienceResultCard extends StatelessWidget {
  final dynamic result;
  const ScienceResultCard({super.key, required this.result});
  @override
  Widget build(BuildContext context) {
    final parsed = SectionScienceResult.parse(result);
    return Card(
        child: Padding(
            padding: const EdgeInsets.all(16),
            child: parsed == null
                ? const Text(
                    'Формат сохранённого научного сценария не поддерживается этой версией приложения. Значения не интерпретированы.')
                : Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                        Text('Сценарий одного сечения',
                            style: Theme.of(context).textTheme.titleLarge),
                        const SizedBox(height: 8),
                        Text('${parsed.capacityNm.toStringAsFixed(2)} Н·м',
                            style: Theme.of(context).textTheme.headlineSmall),
                        const Text(
                            'Расчётный предельный момент заданного здорового круглого сечения.'),
                        const SizedBox(height: 8),
                        const Text(
                            'Программный сценарий. Полевая точность не подтверждена; устойчивость дерева целиком, β и критический ветер не определены.'),
                        ExpansionTile(
                            tilePadding: EdgeInsets.zero,
                            title: const Text('Входы, источники и метод'),
                            children: [
                              for (final line in parsed.lines.skip(1))
                                Padding(
                                    padding: const EdgeInsets.only(bottom: 8),
                                    child: SelectableText(line))
                            ]),
                      ])));
  }
}
