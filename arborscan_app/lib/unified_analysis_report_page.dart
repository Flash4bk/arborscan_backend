import 'dart:typed_data';

import 'package:flutter/material.dart';

import 'app_theme.dart';
import 'corrections_service.dart';
import 'contour_workspace_page.dart';
import 'unified_analysis_models.dart';

class UnifiedAnalysisReportPage extends StatefulWidget {
  final UnifiedAnalysisResult result;
  final Uint8List? fallbackImageBytes;
  final CorrectionsService? correctionsService;

  const UnifiedAnalysisReportPage({
    super.key,
    required this.result,
    this.fallbackImageBytes,
    this.correctionsService,
  });

  @override
  State<UnifiedAnalysisReportPage> createState() => _UnifiedAnalysisReportPageState();
}

class _UnifiedAnalysisReportPageState extends State<UnifiedAnalysisReportPage> {
  UnifiedAnalysisResult get result => widget.result;
  Uint8List? get fallbackImageBytes => widget.fallbackImageBytes;
  bool _editing = false;
  Future<void> _editContour() async {
    if (fallbackImageBytes == null || fallbackImageBytes!.isEmpty || _editing) return;
    setState(() => _editing = true);
    try {
      await Navigator.of(context).push(MaterialPageRoute(builder: (_) =>
        ContourWorkspacePage(analysisId:result.analysisId, image:fallbackImageBytes,
          aiMask:result.maskImageBytes, service:widget.correctionsService)));
    } finally { if (mounted) setState(() => _editing = false); }
  }

  @override
  Widget build(BuildContext context) {
    final displayImage = result.annotatedImageBytes ?? fallbackImageBytes;
    final status = _statusPresentation(result.status);

    return WillPopScope(
      onWillPop: () async => true,
      child: Scaffold(
      appBar: AppBar(title: const Text('Результат сканирования')),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.fromLTRB(16, 12, 16, 28),
          children: [
            if (displayImage != null) ...[
              ClipRRect(
                borderRadius: BorderRadius.circular(18),
                child: AspectRatio(
                  aspectRatio: 4 / 3,
                  child: Image.memory(
                    displayImage,
                    fit: BoxFit.contain,
                    errorBuilder: (_, __, ___) => const _ImageErrorBox(),
                  ),
                ),
              ),
              const SizedBox(height: 14),
            ],
            OutlinedButton.icon(
              onPressed: fallbackImageBytes == null || fallbackImageBytes!.isEmpty
                  ? null : _editContour,
              icon: const Icon(Icons.edit_outlined),
              label: const Text('Исправить контур'),
            ),
            const SizedBox(height: 8),
            Text(
              fallbackImageBytes == null || fallbackImageBytes!.isEmpty
                  ? 'Для редактирования нужно исходное фото. Откройте результат сразу после анализа.'
                  : 'Сохраните исправленный контур отдельной кнопкой. Исходные измерения не пересчитываются.',
              style: const TextStyle(color: AppTheme.muted, fontSize: 13),
            ),
            const SizedBox(height: 14),
            _StatusCard(
              title: status.$1,
              description: status.$2,
              icon: status.$3,
              color: status.$4,
            ),
            const SizedBox(height: 14),
            _SpeciesCard(result: result),
            const SizedBox(height: 18),
            const _SectionTitle(
              title: 'Основные измерения',
              subtitle:
                  'Высота, ширина кроны и DBH объединены в один результат.',
            ),
            const SizedBox(height: 10),
            _MetricCard(
              title: 'Высота дерева',
              icon: Icons.height,
              metric: result.height,
              valueDigits: 2,
            ),
            const SizedBox(height: 10),
            _MetricCard(
              title: 'Ширина кроны',
              icon: Icons.nature,
              metric: result.crownWidth,
              valueDigits: 2,
            ),
            const SizedBox(height: 10),
            _MetricCard(
              title: result.trunkDiameter.standard == 'dbh_1_3m' ? 'DBH ствола' : 'Диаметр ствола',
              icon: Icons.circle_outlined,
              metric: result.trunkDiameter,
              valueDigits: 3,
              dbh: true,
            ),
            const SizedBox(height: 18),
            const _SectionTitle(
              title: 'Диагностика исходных данных',
              subtitle:
                  'Баллы ниже — инженерные эвристики, не точность и не доверительный интервал измерения.',
            ),
            const SizedBox(height: 10),
            _QualityCard(result: result),
            const Card(child:Padding(padding:EdgeInsets.all(16),child:Column(
              crossAxisAlignment:CrossAxisAlignment.start,children:[
                Text('β — коэффициент сопротивления, кг/с',style:TextStyle(fontWeight:FontWeight.bold)),
                Text('Пока не определён. Фото, маска и AR-размеры не задают β однозначно. Нужны положения элементов дерева во времени, массы, упругость и проверенная динамическая модель.'),
              ]))),
            const SizedBox(height: 18),
            const _SectionTitle(
              title: 'Механическая оценка и риск',
              subtitle:
                  'Расчёт выполняется только при наличии валидированных физических данных.',
            ),
            const SizedBox(height: 10),
            _RiskCard(result: result),
            if (result.warnings.isNotEmpty) ...[
              const SizedBox(height: 18),
              const _SectionTitle(
                title: 'Замечания',
                subtitle: 'Причины снижения качества или ограничения анализа.',
              ),
              const SizedBox(height: 10),
              _WarningsCard(warnings: result.warnings),
            ],
            const SizedBox(height: 18),
            _TechnicalCard(result: result),
          ],
        ),
      ),
      ),
    );
  }
}

(String, String, IconData, Color) _statusPresentation(String status) {
  switch (status) {
    case 'measured':
      return (
        'Измерение завершено',
        'Показаны отдельные AR-размеры и/или оценки по эталону. Полевая точность ещё не подтверждена.',
        Icons.check_circle_outline,
        AppTheme.success,
      );
    case 'partial_measurement':
      return (
        'Частичный результат',
        'Часть параметров измерена, но для полного результата данных недостаточно.',
        Icons.info_outline,
        AppTheme.warning,
      );
    case 'absolute_scale_required':
      return (
        'Нужен физический масштаб',
        'Дерево найдено, но без AR или эталона метры не рассчитываются.',
        Icons.straighten,
        AppTheme.warning,
      );
    case 'tree_not_detected':
      return (
        'Дерево не найдено',
        'Система не создаёт искусственную маску. Сделайте другой кадр.',
        Icons.search_off,
        AppTheme.danger,
      );
    default:
      return (
        'Результат анализа',
        'Статус: $status',
        Icons.analytics_outlined,
        AppTheme.primary,
      );
  }
}

class _StatusCard extends StatelessWidget {
  final String title;
  final String description;
  final IconData icon;
  final Color color;

  const _StatusCard({
    required this.title,
    required this.description,
    required this.icon,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              width: 44,
              height: 44,
              decoration: BoxDecoration(
                color: color.withOpacity(0.10),
                borderRadius: BorderRadius.circular(14),
              ),
              child: Icon(icon, color: color),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: Theme.of(context).textTheme.titleMedium?.copyWith(
                          fontWeight: FontWeight.w800,
                        ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    description,
                    style: Theme.of(context).textTheme.bodySmall?.copyWith(
                          color: AppTheme.muted,
                          height: 1.35,
                        ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _SpeciesCard extends StatelessWidget {
  final UnifiedAnalysisResult result;

  const _SpeciesCard({required this.result});

  @override
  Widget build(BuildContext context) {
    final conf = result.speciesConfidence;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Icon(Icons.park_outlined, color: AppTheme.primary),
                const SizedBox(width: 8),
                Text(
                  'Порода',
                  style: Theme.of(context).textTheme.titleMedium?.copyWith(
                        fontWeight: FontWeight.w800,
                      ),
                ),
              ],
            ),
            const SizedBox(height: 10),
            Text(
              result.speciesName,
              style: Theme.of(context).textTheme.titleLarge?.copyWith(
                    fontWeight: FontWeight.w800,
                  ),
            ),
            if (result.scientificName?.isNotEmpty == true) ...[
              const SizedBox(height: 3),
              Text(
                result.scientificName!,
                style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                      fontStyle: FontStyle.italic,
                      color: AppTheme.muted,
                    ),
              ),
            ],
            if (conf != null) ...[
              const SizedBox(height: 10),
              Ui.badge(
                text: 'PlantNet ${(conf * 100).toStringAsFixed(1)}%',
                color: conf >= 0.80 ? AppTheme.success : AppTheme.warning,
                icon: Icons.eco_outlined,
              ),
            ],
          ],
        ),
      ),
    );
  }
}

class _MetricCard extends StatelessWidget {
  final String title;
  final IconData icon;
  final UnifiedMetric metric;
  final int valueDigits;
  final bool dbh;

  const _MetricCard({
    required this.title,
    required this.icon,
    required this.metric,
    required this.valueDigits,
    this.dbh = false,
  });

  @override
  Widget build(BuildContext context) {
    final value = metric.valueM;
    final source = _sourceLabel(metric.source);
    final sourceColor = metric.source == 'ar' || metric.source == 'ar+vision'
        ? AppTheme.success
        : value != null
            ? AppTheme.primary
            : AppTheme.muted;

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              width: 44,
              height: 44,
              decoration: BoxDecoration(
                color: AppTheme.primary.withOpacity(0.08),
                borderRadius: BorderRadius.circular(14),
              ),
              child: Icon(icon, color: AppTheme.primary),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: Theme.of(context).textTheme.titleSmall?.copyWith(
                          fontWeight: FontWeight.w800,
                        ),
                  ),
                  const SizedBox(height: 5),
                  Text(
                    value != null ? '${value.toStringAsFixed(valueDigits)} м' : '—',
                    style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                          fontWeight: FontWeight.w900,
                        ),
                  ),
                  const SizedBox(height: 6),
                  Wrap(
                    spacing: 8,
                    runSpacing: 8,
                    children: [
                      Ui.badge(
                        text: source,
                        color: sourceColor,
                        icon: _sourceIcon(metric.source),
                      ),
                      Ui.badge(
                        text:
                            'Эвристический балл ${metric.confidence.toStringAsFixed(2)}',
                        color: _qualityColor(metric.confidence),
                        icon: Icons.verified_outlined,
                      ),
                    ],
                  ),
                  if (value == null && metric.valuePx != null) ...[
                    const SizedBox(height: 8),
                    Text(
                      'Диагностическая геометрия маски: ${metric.valuePx!.toStringAsFixed(1)} px. Нужны явные границы измеряемой части и физический эталон. AR не задаёт масштаб фото.',
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                            color: AppTheme.muted,
                            height: 1.35,
                          ),
                    ),
                  ],
                  if (dbh) ...[
                    const SizedBox(height: 8),
                    Text(
                      metric.standard == 'dbh_1_3m'
                          ? 'DBH подтверждён на высоте ${metric.measurementHeightM?.toStringAsFixed(2) ?? '1.30'} м.'
                          : 'DBH считается валидным только при измерении диаметра на высоте 1,30 м.',
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                            color: AppTheme.muted,
                            height: 1.35,
                          ),
                    ),
                  ],
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _QualityCard extends StatelessWidget {
  final UnifiedAnalysisResult result;

  const _QualityCard({required this.result});

  @override
  Widget build(BuildContext context) {
    final value = result.overallQuality;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Text(
                  value.toStringAsFixed(2),
                  style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                        fontWeight: FontWeight.w900,
                        color: _qualityColor(value),
                      ),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: Text(
                    _qualityStatusLabel(result.qualityStatus),
                    style: Theme.of(context).textTheme.titleSmall?.copyWith(
                          fontWeight: FontWeight.w800,
                        ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 10),
            ClipRRect(
              borderRadius: BorderRadius.circular(999),
              child: LinearProgressIndicator(
                value: value,
                minHeight: 8,
                backgroundColor: AppTheme.border,
              ),
            ),
            const SizedBox(height: 12),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                Ui.badge(
                  text:
                      'Оценка модели ${result.segmentationConfidence.toStringAsFixed(2)}',
                  color: _qualityColor(result.segmentationConfidence),
                  icon: Icons.auto_awesome_motion_outlined,
                ),
                Ui.badge(
                  text: result.calibrationAvailable
                      ? 'Эвристика масштаба ${result.calibrationConfidence.toStringAsFixed(2)}'
                      : 'Масштаб отсутствует',
                  color: result.calibrationAvailable
                      ? _qualityColor(result.calibrationConfidence)
                      : AppTheme.muted,
                  icon: Icons.straighten,
                ),
              ],
            ),
            if (result.calibrationConflict) ...[
              const SizedBox(height: 10),
              const Text(
                'Обнаружено противоречие между источниками масштаба. Производные метрические размеры ограничены.',
                style: TextStyle(color: AppTheme.warning, height: 1.35),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

class _RiskCard extends StatelessWidget {
  final UnifiedAnalysisResult result;

  const _RiskCard({required this.result});

  @override
  Widget build(BuildContext context) {
    if (result.riskAvailable) {
      return Card(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Row(
            children: [
              const Icon(Icons.shield_outlined, color: AppTheme.primary),
              const SizedBox(width: 10),
              Expanded(
                child: Text(
                  'Механическая оценка доступна.',
                  style: Theme.of(context).textTheme.bodyMedium,
                ),
              ),
            ],
          ),
        ),
      );
    }

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Icon(Icons.science_outlined, color: AppTheme.warning),
            const SizedBox(width: 10),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Риск пока не рассчитывается',
                    style: Theme.of(context).textTheme.titleSmall?.copyWith(
                          fontWeight: FontWeight.w800,
                        ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    result.mechanicalProfileAvailable
                        ? 'Механический профиль найден, но Risk Engine отключён до общей валидации.'
                        : 'Для ${result.scientificName ?? result.speciesName} нет валидированного механического профиля. Параметры другой породы не подставляются.',
                    style: Theme.of(context).textTheme.bodySmall?.copyWith(
                          color: AppTheme.muted,
                          height: 1.35,
                        ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _WarningsCard extends StatelessWidget {
  final List<String> warnings;

  const _WarningsCard({required this.warnings});

  @override
  Widget build(BuildContext context) {
    final labels = warnings.map(_warningLabel).toSet().toList();
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            for (var i = 0; i < labels.length; i++) ...[
              Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Padding(
                    padding: EdgeInsets.only(top: 2),
                    child: Icon(
                      Icons.info_outline,
                      size: 18,
                      color: AppTheme.warning,
                    ),
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      labels[i],
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                            height: 1.35,
                          ),
                    ),
                  ),
                ],
              ),
              if (i != labels.length - 1) const Divider(height: 18),
            ],
          ],
        ),
      ),
    );
  }
}

class _TechnicalCard extends StatelessWidget {
  final UnifiedAnalysisResult result;

  const _TechnicalCard({required this.result});

  @override
  Widget build(BuildContext context) {
    return Card(
      child: ExpansionTile(
        title: const Text(
          'Технические данные',
          style: TextStyle(fontWeight: FontWeight.w800),
        ),
        childrenPadding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
        children: [
          _TechnicalRow(label: 'Analysis ID', value: result.analysisId),
          _TechnicalRow(label: 'API', value: result.apiVersion),
          _TechnicalRow(label: 'Schema', value: result.schemaVersion),
          _TechnicalRow(
            label: 'px → m',
            value: result.pxToM?.toStringAsFixed(9) ?? 'нет',
          ),
          _TechnicalRow(
            label: 'Источник масштаба',
            value: result.calibrationSource ?? 'нет',
          ),
        ],
      ),
    );
  }
}

class _TechnicalRow extends StatelessWidget {
  final String label;
  final String value;

  const _TechnicalRow({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 5),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 130,
            child: Text(
              label,
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                    color: AppTheme.muted,
                  ),
            ),
          ),
          Expanded(
            child: SelectableText(
              value,
              style: Theme.of(context).textTheme.bodySmall,
            ),
          ),
        ],
      ),
    );
  }
}

class _SectionTitle extends StatelessWidget {
  final String title;
  final String subtitle;

  const _SectionTitle({required this.title, required this.subtitle});

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          title,
          style: Theme.of(context).textTheme.titleMedium?.copyWith(
                fontWeight: FontWeight.w800,
              ),
        ),
        const SizedBox(height: 3),
        Text(
          subtitle,
          style: Theme.of(context).textTheme.bodySmall?.copyWith(
                color: AppTheme.muted,
                height: 1.35,
              ),
        ),
      ],
    );
  }
}

class _ImageErrorBox extends StatelessWidget {
  const _ImageErrorBox();

  @override
  Widget build(BuildContext context) {
    return Container(
      color: AppTheme.surface,
      alignment: Alignment.center,
      child: const Icon(Icons.broken_image_outlined, color: AppTheme.muted),
    );
  }
}

String _sourceLabel(String source) {
  switch (source) {
    case 'ar':
      return 'AR';
    case 'ar+vision':
      return 'AR + CV';
    case 'reference+vision':
      return 'Эталон + CV';
    case 'manual_scale+vision':
      return 'Масштаб + CV';
    case 'vision':
      return 'CV, без метров';
    default:
      return 'Недоступно';
  }
}

IconData _sourceIcon(String source) {
  switch (source) {
    case 'ar':
    case 'ar+vision':
      return Icons.view_in_ar_outlined;
    case 'reference+vision':
    case 'manual_scale+vision':
      return Icons.straighten;
    default:
      return Icons.visibility_outlined;
  }
}

Color _qualityColor(double value) {
  if (value >= 0.80) return AppTheme.success;
  if (value >= 0.60) return AppTheme.primary;
  if (value >= 0.40) return AppTheme.warning;
  return AppTheme.danger;
}

String _qualityStatusLabel(String value) {
  switch (value) {
    case 'good':
      return 'Хорошее качество';
    case 'usable':
      return 'Допустимое качество';
    case 'low_confidence':
      return 'Низкое качество';
    case 'invalid_for_measurement':
      return 'Недостаточно данных';
    default:
      return value;
  }
}

String _warningLabel(String code) {
  if (code.startsWith('ar_scale_conflict:')) {
    return 'AR-измерения дают противоречивые масштабы. Проверьте повторяемость измерений.';
  }
  if (code.startsWith('calibration_crosscheck_conflict:')) {
    return 'Физический эталон и AR-калибровка расходятся между собой.';
  }
  switch (code) {
    case 'mechanical_profile_not_available_risk_not_calculated':
      return 'Механический профиль породы не валидирован, поэтому риск не рассчитан.';
    case 'tree_mask_touches_image_edge':
      return 'Дерево касается края кадра. Часть кроны или ствола может быть обрезана.';
    case 'reference_depth_relative_to_tree_not_confirmed':
      return 'Эталон может находиться в другой плоскости относительно дерева.';
    case 'tree_not_detected':
      return 'Нейросеть не обнаружила дерево на изображении.';
    case 'camera_distance_m_not_used_without_calibrated_intrinsics_and_pose':
      return 'Одна дистанция до дерева не используется без калиброванной геометрии камеры.';
    default:
      return code.replaceAll('_', ' ');
  }
}
