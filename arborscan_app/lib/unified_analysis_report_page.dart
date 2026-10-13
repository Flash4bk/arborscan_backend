import 'dart:typed_data';

import 'package:flutter/material.dart';

import 'app_theme.dart';
import 'corrections_service.dart';
import 'contour_workspace_page.dart';
import 'unified_analysis_models.dart';
import 'geometry_report.dart';
import 'report_history_service.dart';
import 'report_export_button.dart';
import 'report_export_data.dart';
import 'survey_environment.dart';
import 'survey_environment_ui.dart';
import 'reference_measurement_page.dart';
import 'combined_reference.dart';
import 'species_annotation.dart';

class UnifiedAnalysisReportPage extends StatefulWidget {
  final UnifiedAnalysisResult result;
  final Uint8List? fallbackImageBytes;
  final CorrectionsService? correctionsService;
  final bool allowServerSave;
  final ExportLoader? exportLoader;
  // The immutable report lineage differs from a later model execution ID.
  final String? historyAnalysisId;
  final String? referenceParentId;
  final Map<String, dynamic>? referenceSnapshot;
  final ReportHistoryService? history;

  const UnifiedAnalysisReportPage({
    super.key,
    required this.result,
    this.fallbackImageBytes,
    this.correctionsService,
    this.allowServerSave = true,
    this.exportLoader,
    this.historyAnalysisId,
    this.referenceParentId,
    this.referenceSnapshot,
    this.history,
  });

  @override
  State<UnifiedAnalysisReportPage> createState() =>
      _UnifiedAnalysisReportPageState();
}

class _UnifiedAnalysisReportPageState extends State<UnifiedAnalysisReportPage> {
  UnifiedAnalysisResult get result => widget.result;
  String? _manualSpecies;
  bool _speciesEdited = false, _reportStaged = false;
  String? _reportParentId;
  late final String _capturedAt =
      result.raw['captured_at'] ?? DateTime.now().toUtc().toIso8601String();
  Map<String, dynamic> get _report => _speciesEdited
      ? withSpeciesAnnotation(result.raw, _manualSpecies ?? '')
      : result.raw;
  String get _analysisId => widget.historyAnalysisId ?? result.analysisId;
  Uint8List? get fallbackImageBytes => widget.fallbackImageBytes;
  bool _editing = false;
  late final _history = widget.history ?? ReportHistoryService();
  String? _reportToken, _saveMessage;
  bool _savingReport = false, _invalidSession = false;
  @override
  void initState() {
    super.initState();
    _reportParentId = widget.referenceParentId;
    CorrectionsService.authChanges.addListener(_invalidateSession);
    if (widget.allowServerSave && fallbackImageBytes != null) _stageReport();
  }

  void _invalidateSession() {
    if (mounted) setState(() => _invalidSession = true);
  }

  @override
  void dispose() {
    CorrectionsService.authChanges.removeListener(_invalidateSession);
    super.dispose();
  }

  Future<void> _stageReport() async {
    _savingReport = true;
    _reportStaged = false;
    try {
      _reportToken = await CorrectionsService.currentToken();
      if (_invalidSession) return;
      final raw = _report;
      final size = await referenceImageSize(fallbackImageBytes!);
      final report =
          boundAnalysisReport(raw, size.width.toInt(), size.height.toInt());
      await _history.stage(
          token: _reportToken!,
          localId: _analysisId,
          analysisId: _analysisId,
          parentId: _reportParentId,
          image: fallbackImageBytes!,
          snapshot: {
            'version': 1,
            'kind': 'v4',
            'report': report,
            'reference': null,
            'ar': raw['ar_provenance'],
            'environment': raw['environment_snapshot'],
            'captured_at': _capturedAt,
            'change_source': _speciesEdited ? 'manual' : 'analysis'
          });
      _reportStaged = true;
      if (mounted && !_invalidSession) {
        setState(() => _saveMessage = 'Полный отчёт сохранён на устройстве.');
      }
    } catch (e) {
      if (mounted) setState(() => _saveMessage = '$e');
    } finally {
      if (mounted) setState(() => _savingReport = false);
    }
  }

  Future<void> _saveReport() async {
    if (_savingReport || _editing || _invalidSession) return;
    if (!_reportStaged) await _stageReport();
    if (!_reportStaged || _invalidSession) return;
    setState(() => _savingReport = true);
    try {
      final response = await _history.upload(_reportToken ?? '', _analysisId);
      _reportParentId = response['record']['version_id'];
      if (mounted && !_invalidSession) {
        setState(() => _saveMessage = 'Сохранено в аккаунте');
      }
    } catch (e) {
      if (mounted) setState(() => _saveMessage = '$e');
    } finally {
      if (mounted) setState(() => _savingReport = false);
    }
  }

  Future<void> _editSpecies() async {
    if (_editing || _savingReport || _invalidSession) return;
    setState(() => _editing = true);
    try {
      final token = await CorrectionsService.currentToken();
      await _history.auth.checkSession(token);
      if (!mounted || _invalidSession) return;
      final name = await editSpeciesName(context, _report);
      await _history.auth.checkSession(token);
      if (name == null || !mounted || _invalidSession) return;
      setState(() {
        _speciesEdited = true;
        _manualSpecies = name;
      });
      await _stageReport();
    } catch (e) {
      if (mounted && !_invalidSession) setState(() => _saveMessage = '$e');
    } finally {
      if (mounted) setState(() => _editing = false);
    }
  }

  Future<void> _editContour() async {
    if (fallbackImageBytes == null || fallbackImageBytes!.isEmpty || _editing) {
      return;
    }
    setState(() => _editing = true);
    try {
      final service = widget.correctionsService ?? _history.auth;
      final token = await CorrectionsService.currentToken();
      await service.checkSession(token);
      final pinnedId = widget.referenceSnapshot?['correction_id'] as String?;
      final record =
          pinnedId == null ? null : await service.record(token, pinnedId);
      await service.checkSession(token);
      if (!mounted || _invalidSession) return;
      if (record != null) {
        validateBoundCorrection(
            record: record,
            correctionId: pinnedId!,
            analysisId: _analysisId,
            photo: fallbackImageBytes!);
      }
      await Navigator.of(context).push(MaterialPageRoute(
          builder: (_) => ContourWorkspacePage(
              analysisId: _analysisId,
              image: fallbackImageBytes,
              aiMask: result.maskImageBytes,
              record: record,
              service: service)));
    } catch (e) {
      if (mounted && !_invalidSession) setState(() => _saveMessage = '$e');
    } finally {
      if (mounted) setState(() => _editing = false);
    }
  }

  Future<void> _addReference() async {
    if (_editing || _invalidSession || fallbackImageBytes == null) return;
    setState(() => _editing = true);
    try {
      final token = await CorrectionsService.currentToken();
      await _history.auth.checkSession(token);
      final owner = await _history.auth.owner(token);
      final existing = await _history.journal.load(owner, _analysisId);
      await _history.auth.checkSession(token);
      if (!mounted || _invalidSession) return;
      final raw = _report;
      await Navigator.push(
          context,
          MaterialPageRoute(
              builder: (_) => ReferenceMeasurementPage(
                  initialImage: fallbackImageBytes,
                  analysisId: _analysisId,
                  parentId: widget.referenceParentId ??
                      (existing?['saved'] == true
                          ? existing!['version_id']
                          : null),
                  initialSnapshot: widget.referenceSnapshot != null
                      ? {
                          ...widget.referenceSnapshot!,
                          'report': _speciesEdited
                              ? withSpeciesAnnotation(
                                  widget.referenceSnapshot!['report'] as Map,
                                  _manualSpecies ?? '')
                              : widget.referenceSnapshot!['report'],
                        }
                      : {
                          'report': raw,
                          'ar': raw['ar_provenance'],
                          'environment': raw['environment_snapshot'],
                          'captured_at': raw['captured_at'] ??
                              DateTime.now().toUtc().toIso8601String(),
                        })));
    } catch (e) {
      if (mounted && !_invalidSession) setState(() => _saveMessage = '$e');
    } finally {
      if (mounted) setState(() => _editing = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_invalidSession) {
      return const Scaffold(
          body: Center(
              child: Text('Аккаунт изменился. Откройте свой отчёт заново.')));
    }
    final displayImage = result.annotatedImageBytes ?? fallbackImageBytes;
    final status = _statusPresentation(context, result.status);

    return WillPopScope(
      onWillPop: () async => true,
      child: Scaffold(
        appBar: AppBar(title: const Text('Результат сканирования')),
        body: SafeArea(
          child: ListView(
            padding: const EdgeInsets.fromLTRB(16, 12, 16, 28),
            children: [
              ReportExportButton(
                  load: widget.exportLoader ??
                      () async => ReportExportData(snapshot: {
                            'kind': 'v4',
                            'report': _report,
                            'captured_at': _capturedAt,
                            'ar': result.raw['ar_provenance'],
                            'environment': result.raw['environment_snapshot']
                          }, record: {
                            'analysis_id': _analysisId
                          }, photo: fallbackImageBytes, local: true)),
              if (widget.allowServerSave && fallbackImageBytes != null) ...[
                FilledButton(
                    onPressed: _savingReport || _editing ? null : _saveReport,
                    child: Text(_savingReport
                        ? 'Сохранение…'
                        : 'Сохранить отчёт в аккаунте')),
              ],
              if (_saveMessage != null) Text(_saveMessage!),
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
                onPressed: _editing ||
                        fallbackImageBytes == null ||
                        fallbackImageBytes!.isEmpty
                    ? null
                    : _addReference,
                icon: const Icon(Icons.straighten),
                label: const Text('Добавить эталон к этому фото'),
              ),
              const SizedBox(height: 8),
              OutlinedButton.icon(
                onPressed: _editing ||
                        fallbackImageBytes == null ||
                        fallbackImageBytes!.isEmpty
                    ? null
                    : _editContour,
                icon: const Icon(Icons.edit_outlined),
                label: const Text('Исправить контур'),
              ),
              const SizedBox(height: 8),
              Text(
                fallbackImageBytes == null || fallbackImageBytes!.isEmpty
                    ? 'Для редактирования нужно исходное фото. Откройте результат сразу после анализа.'
                    : 'Сохраните исправленный контур отдельной кнопкой. Исходные измерения не пересчитываются.',
                style: TextStyle(
                    color: AppTheme.color(context, AppTheme.muted),
                    fontSize: 13),
              ),
              const SizedBox(height: 14),
              _StatusCard(
                title: status.$1,
                description: status.$2,
                icon: status.$3,
                color: status.$4,
              ),
              const SizedBox(height: 14),
              SpeciesAnnotationCard(
                  report: _report,
                  onEdit: widget.allowServerSave &&
                          !_editing &&
                          !_savingReport &&
                          fallbackImageBytes != null
                      ? _editSpecies
                      : null),
              if (result.raw['environment_snapshot'] is Map)
                Card(
                    child: ExpansionTile(
                        title: const Text('Место и условия'),
                        childrenPadding: const EdgeInsets.all(16),
                        children: [
                      EnvironmentSummary(
                          snapshot:
                              surveyMap(result.raw['environment_snapshot']))
                    ])),
              const SizedBox(height: 18),
              const _SectionTitle(
                title: 'Основные измерения',
                subtitle:
                    'Размеры сохраняют свой источник. DBH требует отдельного подтверждения места измерения.',
              ),
              const SizedBox(height: 10),
              _MetricCard(
                title: result.raw['measurement_method_version'] == 2 &&
                        result.height.source != 'ar'
                    ? 'Вертикальный размер маски на фото'
                    : 'Высота дерева',
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
              if (result.trunkDiameter.valueM != null) ...[
                const SizedBox(height: 10),
                _MetricCard(
                  title: result.trunkDiameter.standard == 'dbh_1_3m'
                      ? 'DBH (историческая метка)'
                      : 'Диаметр ствола',
                  icon: Icons.circle_outlined,
                  metric: result.trunkDiameter,
                  valueDigits: 3,
                  dbh: true,
                ),
              ],
              if (result.raw['measurement_method_version'] == 2)
                GeometryReport(
                    compact: true,
                    data: result.raw['geometry'] is Map
                        ? result.raw['geometry'] as Map
                        : null),
              const SizedBox(height: 18),
              ExpansionTile(
                key: const PageStorageKey('analysis-advanced'),
                title: const Text('Расширенный анализ'),
                childrenPadding: const EdgeInsets.all(12),
                children: [
                  Text('Предсказание вида: ${result.speciesName}'),
                  if (result.scientificName != null)
                    Text(result.scientificName!),
                  if (result.speciesConfidence != null)
                    Text(
                        'Оценка Pl@ntNet ${result.speciesConfidence!.toStringAsFixed(3)} - не точность'),
                  if (userSpeciesName(_report) != null)
                    const Text(
                        'Название указано пользователем. Предсказание и механические параметры модели не изменены; подтверждённая метка для обучения не назначена.'),
                  GeometryReport(
                      data: result.raw['geometry'] is Map
                          ? result.raw['geometry'] as Map
                          : null),
                  if (result.trunkDiameter.valueM == null)
                    _MetricCard(
                        title: 'Диаметр ствола',
                        icon: Icons.circle_outlined,
                        metric: result.trunkDiameter,
                        valueDigits: 3,
                        dbh: true),
                  const _SectionTitle(
                    title: 'Диагностика исходных данных',
                    subtitle:
                        'Баллы ниже — инженерные эвристики, не точность и не доверительный интервал измерения.',
                  ),
                  const SizedBox(height: 10),
                  _QualityCard(result: result),
                  const Card(
                      child: Padding(
                          padding: EdgeInsets.all(16),
                          child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text('β — коэффициент сопротивления, кг/с',
                                    style:
                                        TextStyle(fontWeight: FontWeight.bold)),
                                Text(
                                    'Пока не определён. Фото, маска и AR-размеры не задают β однозначно. Нужны положения элементов дерева во времени, массы, упругость и проверенная динамическая модель.'),
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
                      subtitle:
                          'Причины снижения качества или ограничения анализа.',
                    ),
                    const SizedBox(height: 10),
                    _WarningsCard(warnings: result.warnings),
                  ],
                  const SizedBox(height: 18),
                  _TechnicalCard(result: result),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}

(String, String, IconData, Color) _statusPresentation(
    BuildContext context, String status) {
  switch (status) {
    case 'measured':
      return (
        'Измерение завершено',
        'Показаны отдельные AR-размеры и/или оценки по эталону. Полевая точность ещё не подтверждена.',
        Icons.check_circle_outline,
        AppTheme.color(context, AppTheme.success),
      );
    case 'partial_measurement':
      return (
        'Частичный результат',
        'Часть параметров измерена, но для полного результата данных недостаточно.',
        Icons.info_outline,
        AppTheme.color(context, AppTheme.warning),
      );
    case 'absolute_scale_required':
      return (
        'Нужен физический масштаб',
        'Дерево найдено, но без AR или эталона метры не рассчитываются.',
        Icons.straighten,
        AppTheme.color(context, AppTheme.warning),
      );
    case 'tree_not_detected':
      return (
        'Дерево не найдено',
        'Система не создаёт искусственную маску. Сделайте другой кадр.',
        Icons.search_off,
        AppTheme.color(context, AppTheme.danger),
      );
    default:
      return (
        'Результат анализа',
        'Статус: $status',
        Icons.analytics_outlined,
        AppTheme.color(context, AppTheme.primary),
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
                          color: AppTheme.color(context, AppTheme.muted),
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
        ? AppTheme.color(context, AppTheme.success)
        : value != null
            ? AppTheme.color(context, AppTheme.primary)
            : AppTheme.color(context, AppTheme.muted);

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
                color:
                    AppTheme.color(context, AppTheme.primary).withOpacity(0.08),
                borderRadius: BorderRadius.circular(14),
              ),
              child:
                  Icon(icon, color: AppTheme.color(context, AppTheme.primary)),
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
                    value != null
                        ? '${value.toStringAsFixed(valueDigits)} м'
                        : '—',
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
                        color: _qualityColor(context, metric.confidence),
                        icon: Icons.verified_outlined,
                      ),
                    ],
                  ),
                  if (value == null && metric.valuePx != null) ...[
                    const SizedBox(height: 8),
                    Text(
                      'Диагностическая геометрия маски: ${metric.valuePx!.toStringAsFixed(1)} px. Нужны явные границы измеряемой части и физический эталон. AR не задаёт масштаб фото.',
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                            color: AppTheme.color(context, AppTheme.muted),
                            height: 1.35,
                          ),
                    ),
                  ],
                  if (dbh) ...[
                    const SizedBox(height: 8),
                    if (metric.measurementHeightM != null)
                      Text(
                        'Сохранённый уровень сечения: ${metric.measurementHeightM!.toStringAsFixed(2)} м.',
                        style: Theme.of(context).textTheme.bodySmall?.copyWith(
                              color: AppTheme.color(context, AppTheme.muted),
                              height: 1.35,
                            ),
                      ),
                    Text(
                      metric.standard == 'dbh_1_3m'
                          ? 'Историческая метка DBH из сохранённого отчёта; соблюдение полевого протокола не подтверждено.'
                          : 'DBH не подтверждён: уровень 1,30 м сам по себе недостаточен; нужно проверить основание, уклон, наклон и развилки по полевой методике.',
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                            color: AppTheme.color(context, AppTheme.muted),
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
                        color: _qualityColor(context, value),
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
                backgroundColor: AppTheme.color(context, AppTheme.border),
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
                  color: _qualityColor(context, result.segmentationConfidence),
                  icon: Icons.auto_awesome_motion_outlined,
                ),
                Ui.badge(
                  text: result.calibrationAvailable
                      ? 'Эвристика масштаба ${result.calibrationConfidence.toStringAsFixed(2)}'
                      : 'Масштаб отсутствует',
                  color: result.calibrationAvailable
                      ? _qualityColor(context, result.calibrationConfidence)
                      : AppTheme.color(context, AppTheme.muted),
                  icon: Icons.straighten,
                ),
              ],
            ),
            if (result.calibrationConflict) ...[
              const SizedBox(height: 10),
              Text(
                'Обнаружено противоречие между источниками масштаба. Производные метрические размеры ограничены.',
                style: TextStyle(
                    color: AppTheme.color(context, AppTheme.warning),
                    height: 1.35),
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
              Icon(Icons.shield_outlined,
                  color: AppTheme.color(context, AppTheme.primary)),
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
            Icon(Icons.science_outlined,
                color: AppTheme.color(context, AppTheme.warning)),
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
                          color: AppTheme.color(context, AppTheme.muted),
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
                  Padding(
                    padding: const EdgeInsets.only(top: 2),
                    child: Icon(
                      Icons.info_outline,
                      size: 18,
                      color: AppTheme.color(context, AppTheme.warning),
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
        key: const PageStorageKey('analysis-technical'),
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
                    color: AppTheme.color(context, AppTheme.muted),
                  ),
            ),
          ),
          Expanded(
            child: SelectableText(
              value,
              key: PageStorageKey('technical-field-$label'),
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
                color: AppTheme.color(context, AppTheme.muted),
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
      color: AppTheme.color(context, AppTheme.surface),
      alignment: Alignment.center,
      child: Icon(Icons.broken_image_outlined,
          color: AppTheme.color(context, AppTheme.muted)),
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

Color _qualityColor(BuildContext context, double value) {
  if (value >= 0.80) return AppTheme.color(context, AppTheme.success);
  if (value >= 0.60) return AppTheme.color(context, AppTheme.primary);
  if (value >= 0.40) return AppTheme.color(context, AppTheme.warning);
  return AppTheme.color(context, AppTheme.danger);
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
