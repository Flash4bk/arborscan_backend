import 'app_theme.dart';
import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'corrections_service.dart';
import 'report_history_service.dart';
import 'reference_measurement_page.dart';
import 'reference_measurement.dart';
import 'geometry_report.dart';
import 'report_export_button.dart';
import 'report_export_loader.dart';
import 'contour_workspace_page.dart';
import 'unified_analysis_models.dart';
import 'unified_analysis_report_page.dart';
import 'survey_environment.dart';
import 'survey_environment_ui.dart';
import 'environment_revision_page.dart';
import 'map_page.dart';
import 'reference_preview.dart';
import 'combined_reference.dart';
import 'species_annotation.dart';
import 'measurement_results_summary.dart';
import 'science_section_page.dart';
import 'science_section_service.dart';
import 'science_result_card.dart';

class ServerReportPage extends StatefulWidget {
  final String? versionId, localId;
  final String? expectedLocalVersionId;
  final ReportHistoryService? service;
  const ServerReportPage(
      {super.key,
      this.versionId,
      this.localId,
      this.expectedLocalVersionId,
      this.service});
  @override
  State<ServerReportPage> createState() => _ServerReportPageState();
}

class _ServerReportPageState extends State<ServerReportPage> {
  late final _service = widget.service ?? ReportHistoryService();
  Map<String, dynamic>? _data;
  String? _token, _error;
  bool _busy = true, _invalid = false;
  Set<String> _hiddenPhotoLayers = {};
  @override
  void initState() {
    super.initState();
    CorrectionsService.authChanges.addListener(_invalidate);
    _load();
  }

  void _invalidate() {
    if (mounted) {
      setState(() {
        _invalid = true;
        _data = null;
        _error = 'Аккаунт изменился.';
      });
    }
  }

  @override
  void dispose() {
    CorrectionsService.authChanges.removeListener(_invalidate);
    super.dispose();
  }

  Future<void> _load() async {
    try {
      _token = await CorrectionsService.currentToken();
      Map<String, dynamic> data;
      if (widget.versionId != null) {
        data = await _service.record(_token!, widget.versionId!);
      } else {
        final owner = await _service.auth.owner(_token!);
        final d = await _service.journal.load(owner, widget.localId!);
        if (d == null) throw const FormatException('Отчёт не найден.');
        if (widget.expectedLocalVersionId != null &&
            d['version_id'] != widget.expectedLocalVersionId) {
          throw const FormatException(
              'Локальная запись уже изменена. Откройте нужную версию из истории аккаунта.');
        }
        data = {
          'record': d,
          'snapshot': {
            ...d['snapshot'],
            'image': {'original_base64': base64Encode(d['image'])}
          }
        };
      }
      await _service.auth.checkSession(_token!);
      if (!_invalid) _data = data;
    } catch (e) {
      _error = '$e';
    }
    if (mounted) setState(() => _busy = false);
  }

  Future<void> _act(Future<void> Function() action) async {
    if (_busy || _invalid) return;
    setState(() {
      _busy = true;
      _error = null;
    });
    try {
      await _service.auth.checkSession(_token!);
      await action();
    } catch (e) {
      _error = '$e';
    }
    if (mounted) setState(() => _busy = false);
  }

  Future<void> _editSpecies(Map snapshot, Map row, Uint8List photo) async {
    final name = await editSpeciesName(context, snapshot['report'] as Map?);
    await _service.auth.checkSession(_token!);
    if (name == null || !mounted || _invalid) return;
    final localId = 'species-${reportUuid()}';
    await _service.stage(
      token: _token!,
      localId: localId,
      analysisId: row['analysis_id'],
      parentId: widget.versionId != null || row['saved'] == true
          ? row['version_id']
          : row['parent_id'],
      image: photo,
      snapshot: {
        for (final key in [
          'version',
          'kind',
          'reference',
          'ar',
          'environment',
          'captured_at',
          'correction_id'
        ])
          if (snapshot.containsKey(key)) key: snapshot[key],
        'report': withSpeciesAnnotation(snapshot['report'] as Map, name),
        'change_source': 'manual',
      },
    );
    await _service.auth.checkSession(_token!);
    if (!mounted || _invalid) return;
    await Navigator.push(
        context,
        MaterialPageRoute(
            builder: (_) =>
                ServerReportPage(localId: localId, service: _service)));
  }

  @override
  Widget build(BuildContext context) {
    final s = _data?['snapshot'] as Map?;
    final row = _data?['record'] as Map?;
    final ref = s?['reference'];
    final photo =
        s == null ? null : base64Decode(s['image']['original_base64']);
    return Scaffold(
        appBar: AppBar(title: const Text('Сохранённый отчёт')),
        body: AppContentList(padding: const EdgeInsets.all(16), children: [
          if (_busy) const LinearProgressIndicator(),
          if (_error != null) Text(_error!),
          if (s != null && !_invalid) ...[
            Text(widget.versionId != null
                ? (_data?['cached'] == true
                    ? 'Локальная копия серверной версии · без сети'
                    : 'Сохранено в аккаунте')
                : 'Локальная запись'),
            ReportExportButton(
                enabled: !_busy,
                load: () => loadReportExport(
                    snapshot: Map<String, dynamic>.from(s),
                    record: Map<String, dynamic>.from(row!),
                    photo: photo,
                    local: widget.versionId == null && row['saved'] != true,
                    hiddenLayers: _hiddenPhotoLayers,
                    token: _token!)),
            if (photo != null && ref != null)
              SizedBox(
                  height: 280,
                  child: Center(
                      child: AspectRatio(
                          aspectRatio: ref['width'] / ref['height'],
                          child: ReferencePreview(
                              image: photo,
                              width: ref['width'],
                              height: ref['height'],
                              report: Map<String, dynamic>.from(s['report']),
                              reference: Map<String, dynamic>.from(ref),
                              correctionId: s['correction_id'],
                              hiddenLayers: _hiddenPhotoLayers,
                              onHiddenLayersChanged: (hidden) => setState(
                                  () => _hiddenPhotoLayers = hidden)))))
            else if (photo != null)
              Image.memory(photo, height: 240, fit: BoxFit.contain),
            SpeciesAnnotationCard(
                report: s['report'] as Map?,
                onEdit: photo == null || _busy
                    ? null
                    : () => _act(() => _editSpecies(s, row!, photo))),
            MeasurementResultsSummary(snapshot: Map<String, dynamic>.from(s)),
            if (s.containsKey('science_result'))
              ScienceResultCard(result: s['science_result']),
            if (widget.versionId != null || row?['saved'] == true)
              OutlinedButton.icon(
                  icon: const Icon(Icons.calculate_outlined),
                  label: const Text('Расчёты: сценарий сечения'),
                  onPressed: _busy
                      ? null
                      : () async {
                          final version = await Navigator.push<String>(
                              context,
                              MaterialPageRoute(
                                  builder: (_) => ScienceSectionPage(
                                      sourceVersionId: row!['version_id'],
                                      service: ScienceSectionService(
                                          history: _service))));
                          if (version != null && context.mounted && !_invalid) {
                            Navigator.push(
                                context,
                                MaterialPageRoute(
                                    builder: (_) => ServerReportPage(
                                        versionId: version,
                                        service: _service)));
                          }
                        })
            else
              const Text(
                  'Для серверного сценария сечения сначала сохраните отчёт в аккаунте.'),
            if (ref != null) ...[
              ExpansionTile(
                  title: const Text('Расширенный анализ'),
                  childrenPadding: const EdgeInsets.all(12),
                  children: [
                    GeometryReport(data: s['report']['geometry']),
                    const Text(
                        'β не рассчитан. Вид, указанный пользователем, не является подтверждённой меткой для обучения.'),
                  ]),
            ],
            Text('Снимок отчёта от ${s['captured_at'] ?? 'дата неизвестна'}'),
            const Text('Повторный анализ и обновление погоды не выполнялись.'),
            if (SurveyPoint.fromJson(surveyMap(s['environment'])['gps']) !=
                null)
              OutlinedButton.icon(
                  icon: const Icon(Icons.map_outlined),
                  label: const Text('Показать эту версию на карте'),
                  onPressed: _busy
                      ? null
                      : () => Navigator.push(
                          context,
                          MaterialPageRoute(
                              builder: (_) =>
                                  MapPage(focusVersionId: row?['version_id']))))
            else
              const Text(
                  'Нет GPS данных, карта недоступна. Добавьте точку через «Уточнить место / условия».'),
            Card(
                child: ExpansionTile(
                    title: const Text('Место и условия'),
                    childrenPadding: const EdgeInsets.all(16),
                    children: [
                  EnvironmentSummary(snapshot: surveyMap(s['environment']))
                ])),
            if (photo != null)
              OutlinedButton.icon(
                  icon: const Icon(Icons.edit_location_alt_outlined),
                  label: const Text('Уточнить место / условия'),
                  onPressed: _busy
                      ? null
                      : () async {
                          final version = await Navigator.push<String>(
                              context,
                              MaterialPageRoute(
                                  builder: (_) => EnvironmentRevisionPage(
                                      localId: widget.localId,
                                      snapshot: Map<String, dynamic>.from(s),
                                      record: Map<String, dynamic>.from(row!),
                                      photo: photo)));
                          if (version != null && context.mounted && !_invalid) {
                            Navigator.push(
                                context,
                                MaterialPageRoute(
                                    builder: (_) =>
                                        ServerReportPage(versionId: version)));
                          }
                        }),
            if (s['correction_id'] == null)
              OutlinedButton(
                  onPressed: _busy
                      ? null
                      : () => Navigator.push(
                          context,
                          MaterialPageRoute(
                              builder: (_) => ContourWorkspacePage(
                                  analysisId: row!['analysis_id'],
                                  image: photo))),
                  child: const Text('Создать / исправить контур дерева')),
            if (s['kind'] == 'v4')
              OutlinedButton(
                  onPressed: _busy
                      ? null
                      : () => Navigator.push(
                          context,
                          MaterialPageRoute(
                              builder: (_) => UnifiedAnalysisReportPage(
                                  result: UnifiedAnalysisResult.fromJson({
                                    ...Map<String, dynamic>.from(s['report']),
                                    'environment_snapshot': s['environment']
                                  }),
                                  fallbackImageBytes: photo,
                                  historyAnalysisId: row!['analysis_id'],
                                  referenceParentId: widget.versionId != null ||
                                          row['saved'] == true
                                      ? row['version_id']
                                      : null,
                                  referenceSnapshot:
                                      Map<String, dynamic>.from(s),
                                  exportLoader: () => loadReportExport(
                                      snapshot: Map<String, dynamic>.from(s),
                                      record: Map<String, dynamic>.from(row),
                                      photo: photo,
                                      local: widget.versionId == null &&
                                          row['saved'] != true,
                                      token: _token!),
                                  allowServerSave: false))),
                  child: const Text('Открыть отчёт анализа')),
            if (photo != null && ref == null && s['kind'] == 'v4')
              OutlinedButton(
                  onPressed: _busy
                      ? null
                      : () => _act(() async {
                            if (!context.mounted || _invalid) return;
                            await Navigator.push(
                                context,
                                MaterialPageRoute(
                                    builder: (_) => ReferenceMeasurementPage(
                                        initialImage: photo,
                                        analysisId: row!['analysis_id'],
                                        parentId: widget.versionId != null ||
                                                row['saved'] == true
                                            ? row['version_id']
                                            : null,
                                        initialSnapshot:
                                            Map<String, dynamic>.from(s))));
                          }),
                  child: const Text('Добавить эталон к этому фото')),
            if (ref != null && s['report']['tree'] is Map) ...[
              Text(s['correction_id'] != null &&
                      s['report']['contour_preview'] is Map
                  ? 'Зелёная маска - закреплённая исправленная ревизия контура. Размеры - отдельно по эталону.'
                  : s['report']['tree']['detected'] == true
                      ? 'Зелёная маска - сохранённое выделение дерева моделью. Размеры - отдельно по эталону.'
                      : 'В этой версии модель не нашла дерево. Размеры получены ручной разметкой по эталону.'),
              if (s['correction_id'] != null &&
                  s['report']['contour_preview'] is Map &&
                  s['report']['tree']['detected'] != true)
                const Text(
                    'Модель не нашла дерево; исправленный контур создан вручную.'),
              OutlinedButton(
                  onPressed: _busy
                      ? null
                      : () => Navigator.push(
                          context,
                          MaterialPageRoute(
                              builder: (_) => UnifiedAnalysisReportPage(
                                  result: UnifiedAnalysisResult.fromJson({
                                    ...modelReportForSources(
                                        Map<String, dynamic>.from(s)),
                                    'environment_snapshot': s['environment'],
                                  }),
                                  fallbackImageBytes: photo,
                                  historyAnalysisId: row!['analysis_id'],
                                  referenceParentId: widget.versionId != null ||
                                          row['saved'] == true
                                      ? row['version_id']
                                      : null,
                                  referenceSnapshot:
                                      Map<String, dynamic>.from(s),
                                  allowServerSave: false))),
                  child: const Text('Исходный результат модели / источники')),
            ],
            if (ref != null)
              OutlinedButton(
                  onPressed: _busy
                      ? null
                      : () => _act(() async {
                            final owner = await _service.auth.owner(_token!);
                            final r = ReferenceMeasurement.fromJson(
                                Map<String, dynamic>.from(ref));
                            final id = 'server-${row!['version_id']}';
                            if (await referenceStore().load(owner, id) ==
                                null) {
                              await referenceStore().save(
                                  owner,
                                  id,
                                  {
                                    'version': 1,
                                    'width': r.width,
                                    'height': r.height,
                                    'length_text': '${r.lengthM}',
                                    'unit': 'm',
                                    'same_plane': r.samePlane,
                                    'outline': {
                                      'version': 1,
                                      'coordinates':
                                          'normalized_oriented_image',
                                      'width': r.width,
                                      'height': r.height,
                                      'closed': true,
                                      'points': ref['outline']
                                    },
                                    'reference': ref['reference'],
                                    'tree': ref['tree'],
                                    'crown': ref['crown'],
                                    'measurement_version': r.version,
                                    'crown_height': ref['crown_height'] ?? [],
                                    'trunk': ref['trunk'] ?? [],
                                    'trunk_axis': ref['trunk_axis'] ?? [],
                                    'server_analysis_id': row['analysis_id'],
                                    'server_parent_id':
                                        widget.versionId != null ||
                                                row['saved'] == true
                                            ? row['version_id']
                                            : null,
                                    'server_snapshot': s
                                  },
                                  photo!);
                            }
                            await _service.auth.checkSession(_token!);
                            if (!context.mounted || _invalid) return;
                            await Navigator.push(
                                context,
                                MaterialPageRoute(
                                    builder: (_) =>
                                        ReferenceMeasurementPage(draftId: id)));
                          }),
                  child: const Text('Изменить эталон / новая версия')),
            if (s['correction_id'] != null)
              OutlinedButton(
                  onPressed: _busy
                      ? null
                      : () => _act(() async {
                            final c = await _service.auth
                                .record(_token!, s['correction_id']);
                            await _service.auth.checkSession(_token!);
                            if (!context.mounted || _invalid) return;
                            if (photo == null) {
                              throw const FormatException(
                                  'Оригинальное фото недоступно.');
                            }
                            validateBoundCorrection(
                                record: c,
                                correctionId: s['correction_id'],
                                analysisId: row!['analysis_id'],
                                photo: photo);
                            await Navigator.push(
                                context,
                                MaterialPageRoute(
                                    builder: (_) => ContourWorkspacePage(
                                        analysisId: row['analysis_id'],
                                        image: photo,
                                        service: _service.auth,
                                        record: c)));
                          }),
                  child: const Text('Связанный контур и модерация')),
            if (widget.versionId != null)
              OutlinedButton(
                  onPressed: _busy
                      ? null
                      : () => _act(() async {
                            final choices = <Map<String, dynamic>>[];
                            int? offset = 0;
                            while (offset != null) {
                              final page =
                                  await _service.auth.list(_token!, offset);
                              choices.addAll((page['items'] as List)
                                  .where((c) =>
                                      c['analysis_id'] == row!['analysis_id'])
                                  .map((c) => Map<String, dynamic>.from(c)));
                              offset = page['next_offset'];
                            }
                            if (!context.mounted || _invalid) return;
                            if (choices.isEmpty) {
                              throw const FormatException(
                                  'Для этого анализа пока нет сохранённого контура. Откройте отчёт и сохраните контур.');
                            }
                            final id = await showDialog<String>(
                                context: context,
                                builder: (c) => SimpleDialog(
                                    title: const Text(
                                        'Связать конкретную ревизию'),
                                    children: choices
                                        .map((r) => SimpleDialogOption(
                                            onPressed: () => Navigator.pop(
                                                c, r['correction_id']),
                                            child: Text(
                                                '${r['created_at']} · ${r['review_status']}')))
                                        .toList()));
                            if (id == null) return;
                            final payload = <String, dynamic>{
                              for (final key in [
                                'version',
                                'kind',
                                'report',
                                'reference',
                                'ar',
                                'environment',
                                'captured_at'
                              ])
                                if (s.containsKey(key)) key: s[key],
                              'change_source': 'contour_link',
                              'correction_id': id
                            };
                            final local = 'link-${row!['version_id']}-$id';
                            await _service.stage(
                                token: _token!,
                                localId: local,
                                analysisId: row['analysis_id'],
                                parentId: row['version_id'],
                                snapshot: payload,
                                image: photo!);
                            await _service.upload(_token!, local);
                            if (mounted && !_invalid) {
                              setState(() => _error =
                                  'Связь сохранена новой версией отчёта. Решение модерации не изменено.');
                            }
                          }),
                  child:
                      const Text('Связать сохранённый контур новой версией')),
            if (widget.versionId != null)
              OutlinedButton(
                  onPressed: _busy
                      ? null
                      : () => _act(() async {
                            if (!context.mounted) return;
                            await Navigator.push(
                                context,
                                MaterialPageRoute(
                                    builder: (_) => Scaffold(
                                        appBar: AppBar(
                                            title: const Text('Версии отчёта')),
                                        body: SingleChildScrollView(
                                            child: ServerHistoryPanel(
                                                analysisId:
                                                    row!['analysis_id'])))));
                          }),
                  child: const Text('Все версии')),
            if (widget.localId != null)
              FilledButton(
                  onPressed: _busy
                      ? null
                      : () => _act(() async {
                            await _service.upload(_token!, widget.localId!);
                            if (mounted) {
                              setState(() => _error = 'Сохранено в аккаунте');
                            }
                          }),
                  child: const Text('Сохранить на сервере / повторить')),
            ExpansionTile(
                title: const Text('Исходные данные и происхождение'),
                children: [
                  SelectableText(const JsonEncoder.withIndent('  ').convert({
                    for (final e in s.entries)
                      if (e.key != 'image' && e.key != 'report') e.key: e.value,
                    'report': {
                      for (final e in (s['report'] as Map).entries)
                        if (e.key != 'images') e.key: e.value
                    }
                  })),
                ]),
          ],
        ]));
  }
}

class ServerHistoryPanel extends StatefulWidget {
  final String? analysisId;
  final ReportHistoryService? service;
  const ServerHistoryPanel({super.key, this.analysisId, this.service});
  @override
  State<ServerHistoryPanel> createState() => _ServerHistoryPanelState();
}

class _ServerHistoryPanelState extends State<ServerHistoryPanel> {
  late final _service = widget.service ?? ReportHistoryService();
  final List<Map<String, dynamic>> _rows = [];
  List<Map<String, dynamic>> _local = [];
  String? _error;
  bool _busy = false;
  int? _next = 0;
  int _epoch = 0;
  @override
  void initState() {
    super.initState();
    CorrectionsService.authChanges.addListener(_changed);
    _load(true);
  }

  @override
  void dispose() {
    CorrectionsService.authChanges.removeListener(_changed);
    super.dispose();
  }

  void _changed() {
    _epoch++;
    setState(() {
      _rows.clear();
      _local = [];
      _error = null;
      _busy = false;
    });
    _load(true);
  }

  Future<void> _load(bool reset) async {
    if (_busy) return;
    final epoch = _epoch;
    setState(() {
      _busy = true;
      _error = null;
      if (reset) {
        _rows.clear();
        _next = 0;
      }
    });
    try {
      final token = await CorrectionsService.currentToken();
      final owner = await _service.auth.owner(token);
      final local = await _service.journal.list(owner);
      await _service.auth.checkSession(token);
      if (epoch != _epoch || !mounted) return;
      setState(() => _local = local);
      final page =
          await _service.list(token, _next ?? 0, analysisId: widget.analysisId);
      if (epoch != _epoch || !mounted) return;
      final next = page['next_offset'];
      if (next != null && (next is! int || next <= (_next ?? 0))) {
        throw const FormatException('Неверная страница истории.');
      }
      setState(() {
        for (final r in page['items'] as List) {
          if (!_rows.any((old) => old['version_id'] == r['version_id'])) {
            _rows.add(Map<String, dynamic>.from(r));
          }
        }
        _local.removeWhere((r) =>
            r['saved'] == true &&
            _rows.any((s) => s['version_id'] == r['version_id']));
        _next = next;
      });
    } catch (e) {
      if (epoch == _epoch && mounted) setState(() => _error = '$e');
    } finally {
      if (epoch == _epoch && mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) => Column(children: [
        ListTile(
            title: const Text('Отчёты аккаунта'),
            trailing: IconButton(
                tooltip: 'Обновить отчёты аккаунта',
                onPressed: _busy ? null : () => _load(true),
                icon: const Icon(Icons.refresh))),
        if (_busy) const LinearProgressIndicator(),
        if (_error != null) Text(_error!),
        if (!_busy && _error == null && _rows.isEmpty && _local.isEmpty)
          const Text('Сохранённых отчётов пока нет.'),
        if (widget.analysisId == null)
          for (final r in _local)
            ListTile(
                leading: const Icon(Icons.phone_android),
                title: Text(r['saved'] == true
                    ? 'Локальная копия серверного отчёта'
                    : 'Локально — отправка не подтверждена'),
                subtitle: Text('${r['created_at']}'),
                onTap: () => Navigator.push(
                    context,
                    MaterialPageRoute(
                        builder: (_) =>
                            ServerReportPage(localId: r['draft_id'])))),
        for (final r in _rows)
          ListTile(
              leading: const Icon(Icons.cloud_done),
              title: Text(summarySpeciesName(r['summary'] as Map?)),
              subtitle: Text(
                  '${r['created_at']} · ${r['parent_id'] == null ? 'Исходная версия' : 'Новая версия'}'),
              onTap: () => Navigator.push(
                  context,
                  MaterialPageRoute(
                      builder: (_) =>
                          ServerReportPage(versionId: r['version_id'])))),
        if (!_busy && (_next != null || _error != null))
          TextButton(
              onPressed: () => _load(false),
              child: Text(_error == null ? 'Загрузить ещё' : 'Повторить')),
      ]);
}
