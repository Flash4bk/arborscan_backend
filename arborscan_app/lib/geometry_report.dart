import 'package:flutter/material.dart';

/// Displays frozen values; never recalculates a historical report.
class GeometryReport extends StatelessWidget {
  final Map? data;
  const GeometryReport({super.key, required this.data});
  static const labels = {
    'tree_segment_length': 'Прямой отрезок основание–верх (2D)',
    'crown_height': 'Высота отмеченной живой кроны (проекция)',
    'trunk_diameter': 'Ширина сечения ствола (фотооценка диаметра)',
    'trunk_measurement_height':
        'Уровень сечения над отмеченным основанием (проекция)',
    'trunk_lean': 'Угол оси ствола к эталону (2D)',
    'dbh': 'DBH',
    'crown_porosity': 'Пористость кроны',
  };
  static const reasons = {
    'tree_segment_length': 'нет данных в этой версии отчёта',
    'crown_height': 'нужны нижняя граница живой кроны и её верх',
    'trunk_diameter': 'нужны края коры и ось участка ствола',
    'trunk_measurement_height': 'нужна разметка сечения',
    'trunk_lean':
        'нужны ось участка ствола и вертикальный эталон; 3D-наклон не определяется',
    'dbh':
        'не подтверждено место измерения по методике: 1,3 м само по себе недостаточно; важны основание, уклон, наклон и развилки',
    'crown_porosity':
        'нет отдельной области кроны и проверенной маски просветов; заполненный контур недостаточен',
  };
  @override
  Widget build(BuildContext context) => Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          for (final e in labels.entries)
            Padding(
                padding: const EdgeInsets.symmetric(vertical: 4),
                child: Text(
                    '${e.value}: ${data?[e.key]?['value'] is num ? '${(data![e.key]['value'] as num).toStringAsFixed(2)} ${data![e.key]['unit'] == 'deg' ? '°' : data![e.key]['unit']}' : 'нет данных — ${reasons[e.key]}'}')),
          const Text(
              'Фотооценки требуют общей глубины и малой перспективы. Наклон относится только к выбранному прямому участку; кривизна и глубина не восстановлены. Полевая точность не подтверждена.'),
        ],
      );
}
