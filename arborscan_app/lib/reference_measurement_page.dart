import 'app_theme.dart';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';
import 'package:crypto/crypto.dart';
import 'contour_drafts.dart';
import 'contour_editor_state.dart';
import 'corrections_service.dart';
import 'image_line_page.dart';
import 'mask_drawing_page.dart';
import 'reference_measurement.dart';
import 'report_history_service.dart';
import 'geometry_report.dart';
import 'report_export_button.dart';
import 'report_export_data.dart';
import 'survey_environment.dart';
import 'survey_environment_ui.dart';
import 'image_selection_service.dart';
import 'photo_analysis_service.dart';
import 'reference_preview.dart';
import 'combined_reference.dart';
import 'contour_workspace_page.dart';
import 'species_annotation.dart';
import 'measurement_results_summary.dart';

ContourDrafts referenceStore() => ContourDrafts(
    directory: () async => Directory(
        '${(await getApplicationSupportDirectory()).path}/reference-measurements-v1'));

Future<Size> referenceImageSize(Uint8List bytes) async {
  if (bytes.length > 15 * 1024 * 1024) {
    throw const FormatException('Фото больше 15 МБ.');
  }
  final buffer = await ui.ImmutableBuffer.fromUint8List(bytes);
  try {
    final descriptor = await ui.ImageDescriptor.encoded(buffer);
    try {
      if (descriptor.width * descriptor.height > 25000000) {
        throw const FormatException('Фото больше 25 МП.');
      }
      final codec = await descriptor.instantiateCodec();
      try {
        final frame = await codec.getNextFrame();
        final size =
            Size(frame.image.width.toDouble(), frame.image.height.toDouble());
        frame.image.dispose();
        return size;
      } finally {
        codec.dispose();
      }
    } finally {
      descriptor.dispose();
    }
  } finally {
    buffer.dispose();
  }
}

class ReferenceMeasurementPage extends StatefulWidget {
  final String? draftId;
  final CorrectionsService? service;
  final ContourDrafts? drafts;
  final Future<Uint8List?> Function()? pickPhoto;
  final Uint8List? initialImage;
  final Map<String, dynamic>? initialSnapshot;
  final String? analysisId, parentId;
  final ReportHistoryService? history;
  final PhotoAnalysisService? analysisService;
  final ContourDrafts? contourDrafts;
  const ReferenceMeasurementPage(
      {super.key,
      this.draftId,
      this.service,
      this.drafts,
      this.pickPhoto,
      this.initialImage,
      this.initialSnapshot,
      this.analysisId,
      this.parentId,
      this.history,
      this.analysisService,
      this.contourDrafts});
  @override
  State<ReferenceMeasurementPage> createState() =>
      _ReferenceMeasurementPageState();
}

class _ReferenceMeasurementPageState extends State<ReferenceMeasurementPage> {
  late final _service = widget.service ?? CorrectionsService(),
      _store = widget.drafts ?? referenceStore();
  late final _history = widget.history ?? ReportHistoryService(auth: _service);
  late final _analysis =
      widget.analysisService ?? PhotoAnalysisService(auth: _service);
  final _length = TextEditingController();
  final _environment = SurveyEnvironmentController();
  void _environmentChanged() {
    if (_busy || _invalid || _image == null || _owner == null) return;
    _save().catchError((Object e) {
      if (mounted) setState(() => _error = '$e');
    });
  }

  String? _token, _owner, _id, _error, _calculationError;
  String _unit = 'm';
  bool _busy = true, _invalid = false, _plane = false;
  bool _saved = false;
  String? _serverAnalysisId, _serverParentId, _serverMessage;
  Map<String, dynamic>? _serverSnapshot;
  Uint8List? _image;
  int _width = 0, _height = 0;
  Map<String, dynamic>? _outline;
  List<Offset> _reference = [], _tree = [], _crown = [];
  List<Offset> _crownHeight = [], _trunk = [], _trunkAxis = [];
  int _measurementVersion = 2;
  Offset? _treeTap;
  Set<String> _hiddenPhotoLayers = {};
  @override
  void initState() {
    super.initState();
    CorrectionsService.authChanges.addListener(_invalidate);
    _environment.addListener(_environmentChanged);
    _load();
  }

  void _invalidate() {
    if (mounted) {
      setState(() {
        _invalid = true;
        _image = null;
        _error = 'Аккаунт изменился. Откройте экран заново.';
      });
    }
  }

  @override
  void dispose() {
    CorrectionsService.authChanges.removeListener(_invalidate);
    _length.dispose();
    _environment.dispose();
    super.dispose();
  }

  Future<void> _guard() async {
    await _service.checkSession(_token ?? '');
    if (_invalid) throw const CorrectionException('Аккаунт изменился.');
  }

  Future<void> _load() async {
    try {
      _token = await CorrectionsService.currentToken();
      _owner = await _service.owner(_token!);
      await _guard();
      _id = widget.draftId ??
          'reference-${DateTime.now().microsecondsSinceEpoch}';
      if (widget.draftId != null) {
        final d = await _store.load(_owner!, _id!);
        await _guard();
        if (d == null || d['version'] != 1) {
          throw const FormatException('Запись недоступна.');
        }
        if (d['photo_sha256'] != null &&
            d['photo_sha256'] != sha256.convert(d['image']).toString()) {
          throw const FormatException(
              'Фото не соответствует сохранённой разметке.');
        }
        final size = await referenceImageSize(d['image']);
        await _guard();
        if (size.width != d['width'] || size.height != d['height']) {
          throw const FormatException(
              'Размеры фото не соответствуют разметке.');
        }
        _image = d['image'];
        _width = d['width'];
        _height = d['height'];
        _length.text = d['length_text'];
        _unit = d['unit'];
        _plane = d['same_plane'] == true;
        _outline = d['outline'];
        _reference = ReferenceMeasurement.decode(d['reference']);
        _tree = ReferenceMeasurement.decode(d['tree']);
        _crown = ReferenceMeasurement.decode(d['crown']);
        _measurementVersion = d['measurement_version'] ?? 1;
        _crownHeight = ReferenceMeasurement.decode(d['crown_height'] ?? []);
        _trunk = ReferenceMeasurement.decode(d['trunk'] ?? []);
        _trunkAxis = ReferenceMeasurement.decode(d['trunk_axis'] ?? []);
        _saved = true;
        _serverAnalysisId = d['server_analysis_id'];
        _serverParentId = d['server_parent_id'];
        _serverSnapshot = d['server_snapshot'] == null
            ? null
            : Map<String, dynamic>.from(d['server_snapshot']);
        final tap = d['tree_tap'];
        if (tap is Map) {
          final points = ReferenceMeasurement.decode([tap]);
          _treeTap = points.single;
        }
        if (_serverSnapshot?['report'] is Map) {
          _serverSnapshot!['report'] = boundAnalysisReport(
              Map<String, dynamic>.from(_serverSnapshot!['report']),
              _width,
              _height);
        }
        _environment.replace(
            surveyMap(d['environment'] ?? _serverSnapshot?['environment']));
      } else if (widget.initialImage != null) {
        // Snapshot a selected original immediately; never use the API's resized
        // JPEG as the original to which reference coordinates belong.
        final original = Uint8List.fromList(widget.initialImage!);
        final size = await referenceImageSize(original);
        await _guard();
        _image = original;
        _width = size.width.toInt();
        _height = size.height.toInt();
        _serverAnalysisId = widget.analysisId;
        _serverParentId = widget.parentId;
        if (widget.initialSnapshot != null) {
          _serverSnapshot = Map<String, dynamic>.from(
              jsonDecode(jsonEncode(widget.initialSnapshot)));
          final binding = _serverSnapshot!['image'];
          if (binding is Map &&
              binding['sha256'] != null &&
              binding['sha256'] != sha256.convert(original).toString()) {
            throw const FormatException('Отчёт относится к другому фото.');
          }
          if (_serverSnapshot!['report'] is Map) {
            _serverSnapshot!['report'] = boundAnalysisReport(
                Map<String, dynamic>.from(_serverSnapshot!['report']),
                _width,
                _height);
          }
          _environment.replace(surveyMap(_serverSnapshot!['environment']));
          final initialReference = _serverSnapshot!['reference'];
          if (initialReference is Map) {
            final measurement = ReferenceMeasurement.fromJson(
                Map<String, dynamic>.from(initialReference));
            if (measurement.width != _width || measurement.height != _height) {
              throw const FormatException('Эталон относится к другому фото.');
            }
            _length.text = '${measurement.lengthM}';
            _unit = 'm';
            _plane = measurement.samePlane;
            _measurementVersion = measurement.version;
            _outline = ContourEditorState(
                    width: _width,
                    height: _height,
                    closed: true,
                    points: measurement.outline)
                .toJson();
            _reference = measurement.reference;
            _tree = measurement.tree;
            _crown = measurement.crown;
            _crownHeight = measurement.crownHeight;
            _trunk = measurement.trunk;
            _trunkAxis = measurement.trunkAxis;
          }
        } else {
          _environment.setPoint(surveyPointFromExif(original));
        }
        await _save();
      }
    } catch (e) {
      _image = null;
      _error = '$e';
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _save() async {
    await _guard();
    if (_image == null) return;
    await _store.save(
        _owner!,
        _id!,
        {
          'version': 1,
          'server_analysis_id': _serverAnalysisId,
          'server_parent_id': _serverParentId,
          'server_snapshot': _serverSnapshot,
          'tree_tap': _treeTap == null
              ? null
              : ReferenceMeasurement.encode([_treeTap!]).single,
          'environment':
              _environment.snapshot.isEmpty ? null : _environment.snapshot,
          'method': 'known_object_segment_v$_measurementVersion',
          'coordinates': 'normalized_oriented_image',
          'width': _width,
          'height': _height,
          'photo_sha256': sha256.convert(_image!).toString(),
          'length_text': _length.text,
          'unit': _unit,
          'same_plane': _plane,
          'outline': _outline,
          'measurement_version': _measurementVersion,
          'crown_height': ReferenceMeasurement.encode(_crownHeight),
          'trunk': ReferenceMeasurement.encode(_trunk),
          'trunk_axis': ReferenceMeasurement.encode(_trunkAxis),
          'reference': ReferenceMeasurement.encode(_reference),
          'tree': ReferenceMeasurement.encode(_tree),
          'crown': ReferenceMeasurement.encode(_crown),
          'report': _result?.report,
          'saved_at': DateTime.now().toUtc().toIso8601String()
        },
        _image!);
    await _guard();
    if (mounted) setState(() => _saved = true);
  }

  Future<void> _run(Future<void> Function() action) async {
    if (_busy || _invalid) return;
    setState(() {
      _busy = true;
      _saved = false;
      _serverMessage = null;
      _error = null;
    });
    try {
      await _guard();
      await action();
      await _guard();
      await _save();
    } catch (e) {
      if (mounted && !_invalid) _error = '$e';
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _pick() async {
    final bytes = widget.pickPhoto != null
        ? await widget.pickPhoto!()
        : await (await ImageSelectionService().pick(ImageSource.gallery))
            ?.readAsBytes();
    if (bytes == null) return;
    final size = await referenceImageSize(bytes);
    await _guard();
    _width = size.width.toInt();
    _height = size.height.toInt();
    _image = bytes;
    _id = 'reference-${DateTime.now().microsecondsSinceEpoch}';
    _outline = null;
    _reference = [];
    _tree = [];
    _crown = [];
    _crownHeight = [];
    _trunk = [];
    _trunkAxis = [];
    _measurementVersion = 2;
    _plane = false;
    _serverAnalysisId = null;
    _serverParentId = null;
    _serverSnapshot = null;
    _serverMessage = null;
    _treeTap = null;
    _hiddenPhotoLayers = {};
    _environment.setPoint(surveyPointFromExif(bytes));
  }

  Future<void> _uploadReport() async {
    final result = _result;
    if (result == null) return;
    _serverAnalysisId ??= reportUuid();
    await _save();
    // Capture time belongs to the measurement, not to each network retry.
    _serverSnapshot ??= {
      'captured_at': DateTime.now().toUtc().toIso8601String()
    };
    _serverSnapshot!['captured_at'] ??=
        DateTime.now().toUtc().toIso8601String();
    final snapshot = _snapshot(result);
    await _save();
    await _history.stage(
        token: _token!,
        localId: _id!,
        analysisId: _serverAnalysisId!,
        image: _image!,
        snapshot: snapshot,
        parentId: _serverParentId);
    await _guard();
    final response = await _history.upload(_token!, _id!);
    await _guard();
    _serverParentId = response['record']['version_id'];
    _serverSnapshot = snapshot;
    _serverMessage = 'Сохранено в аккаунте';
  }

  Map<String, dynamic> _snapshot(ReferenceMeasurement result) =>
      combinedReferenceSnapshot(
          base: _serverSnapshot,
          reference: result.toJson(),
          manualReport: result.report,
          capturedAt: _serverSnapshot?['captured_at'] ??
              DateTime.now().toUtc().toIso8601String(),
          environment:
              _environment.snapshot.isEmpty ? null : _environment.snapshot);

  Future<void> _editSpecies() async {
    final report = _serverSnapshot?['report'] as Map?;
    final name = await editSpeciesName(context, report);
    await _guard();
    if (name == null) return;
    _serverSnapshot = {
      ...?_serverSnapshot,
      'report': withSpeciesAnnotation(report ?? {}, name),
    };
  }

  Future<void> _selectTree() async {
    final point = await Navigator.push<Offset>(
        context,
        MaterialPageRoute(
            builder: (_) => SelectTreePointPage(
                image: _image!,
                width: _width,
                height: _height,
                initial: _treeTap,
                sessionToken: _token!)));
    await _guard();
    if (point != null) _treeTap = point;
  }

  Future<void> _analyzeTree() async {
    // Durable reference draft before any network request. No model failure can
    // erase the dimensions, manual lines or known original photo.
    await _save();
    final original = Uint8List.fromList(_image!);
    final digest = sha256.convert(original).toString();
    final raw =
        await _analysis.analyze(token: _token!, image: original, tap: _treeTap);
    await _guard();
    if (_image == null || sha256.convert(_image!).toString() != digest) {
      throw const FormatException(
          'Фото изменилось. Повторите выделение дерева.');
    }
    final report = boundAnalysisReport(raw, _width, _height);
    if (report['geometry'] is Map) {
      // A fresh inference has fresh model geometry even when the outer record
      // is already a reference version. Keep it before the next manual merge.
      report['model_geometry'] = report['geometry'];
    }
    final pinned = (_serverSnapshot?['report'] as Map?)?['contour_preview'];
    if (pinned != null) report['contour_preview'] = pinned;
    final annotation =
        (_serverSnapshot?['report'] as Map?)?['species_annotation'];
    if (annotation != null) report['species_annotation'] = annotation;
    // Keep a saved analysis's identity/parent. Inference does not silently
    // replace an immutable server version; the next save creates its child.
    _serverAnalysisId ??= raw['analysis_id'] as String? ?? reportUuid();
    _serverSnapshot = {
      ...?_serverSnapshot,
      'report': report,
      'captured_at': _serverSnapshot?['captured_at'] ??
          DateTime.now().toUtc().toIso8601String(),
    };
    _serverMessage = raw['tree']['detected'] == true
        ? 'Выделение дерева добавлено. Размеры ниже определяются отмеченными отрезками и эталоном.'
        : 'Модель не нашла дерево. Разметка и измерения по эталону сохранены; выберите дерево или исправьте контур вручную.';
  }

  Future<void> _editTreeContour() async {
    _serverAnalysisId ??= reportUuid();
    await _save();
    final id = _serverSnapshot?['correction_id'];
    final record = id == null ? null : await _service.record(_token!, id);
    await _guard();
    if (!mounted || _invalid) return;
    if (record != null) {
      validateBoundCorrection(
          record: record,
          correctionId: id,
          analysisId: _serverAnalysisId!,
          photo: _image!);
    }
    final selected = await Navigator.push<Map<String, dynamic>>(
        context,
        MaterialPageRoute(
            builder: (_) => ContourWorkspacePage(
                analysisId: _serverAnalysisId!,
                image: _image,
                record: record,
                aiMask: savedTreeMask(_serverSnapshot?['report'] is Map
                    ? Map<String, dynamic>.from(_serverSnapshot!['report'])
                    : null),
                service: _service,
                drafts: widget.contourDrafts,
                returnSelection: true)));
    await _guard();
    if (selected == null) return;
    if (selected['analysis_id'] != _serverAnalysisId ||
        selected['photo_sha256'] != sha256.convert(_image!).toString() ||
        selected['mask'] is! Uint8List ||
        selected['correction_id'] is! String) {
      throw const FormatException(
          'Контур не соответствует этому фото и отчёту.');
    }
    _serverSnapshot = {
      ...?_serverSnapshot,
      'captured_at': _serverSnapshot?['captured_at'] ??
          DateTime.now().toUtc().toIso8601String(),
      'correction_id': selected['correction_id'],
      'report': {
        ...?(_serverSnapshot?['report'] as Map?),
        'contour_preview': {
          'correction_id': selected['correction_id'],
          'photo_sha256': selected['photo_sha256'],
          'mask_png_base64': base64Encode(selected['mask']),
          'width': _width,
          'height': _height,
        }
      }
    };
    _serverMessage =
        'В отчёте закреплена эта сохранённая ревизия контура. Размеры по эталону и решение модерации не изменились.';
  }

  Future<void> _line(String kind, String title) async {
    final points = kind == 'reference'
        ? _reference
        : kind == 'tree'
            ? _tree
            : kind == 'crown'
                ? _crown
                : kind == 'crown_height'
                    ? _crownHeight
                    : kind == 'trunk'
                        ? _trunk
                        : _trunkAxis;
    final result = await Navigator.of(context).push<List<Offset>>(
        MaterialPageRoute(
            builder: (_) => ImageLinePage(
                image: _image!,
                width: _width,
                height: _height,
                title: kind == 'reference'
                    ? '$title: ${_length.text} ${_unit == 'm' ? 'м' : 'см'}'
                    : title,
                initial: points,
                sessionToken: _token)));
    await _guard();
    if (result == null) return;
    if (kind == 'reference') {
      _reference = result;
    } else if (kind == 'tree') {
      _tree = result;
    } else if (kind == 'crown_height') {
      _measurementVersion = 2;
      _crownHeight = result;
    } else if (kind == 'trunk') {
      _measurementVersion = 2;
      _trunk = result;
    } else if (kind == 'trunk_axis') {
      _measurementVersion = 2;
      _trunkAxis = result;
    } else {
      _crown = result;
    }
  }

  ReferenceMeasurement? get _result {
    _calculationError = null;
    if (_outline?['closed'] != true) return null;
    if (_outline?['width'] != _width || _outline?['height'] != _height) {
      return null;
    }
    try {
      return ReferenceMeasurement(
          version: _measurementVersion,
          crownHeight: _crownHeight,
          trunk: _trunk,
          trunkAxis: _trunkAxis,
          width: _width,
          height: _height,
          lengthM: ReferenceMeasurement.parseLength(_length.text, _unit),
          samePlane: _plane,
          reference: _reference,
          tree: _tree,
          crown: _crown,
          outline: _outline == null
              ? []
              : ContourEditorState.fromJson(_outline!).points);
    } catch (e) {
      _calculationError =
          e is FormatException ? e.message : 'Неверная разметка.';
      return null;
    }
  }

  Future<void> _export() async {
    final result = _result;
    if (result == null) return;
    final json = const JsonEncoder.withIndent('  ').convert({
      'measurement': result.toJson(),
      'photo_sha256': sha256.convert(_image!).toString(),
      ...(_snapshot(result)['report'] as Map),
      'crown_width_m': result.crownM,
      'dbh_m': null,
      'beta_kg_s': null,
      'metrological_accuracy': null,
      'conditions': 'same_depth_weak_perspective_user_confirmed'
    });
    if (!mounted) return;
    await showDialog<void>(
        context: context,
        builder: (c) => AlertDialog(
                title: const Text('Экспорт измерения'),
                content: SingleChildScrollView(child: SelectableText(json)),
                actions: [
                  TextButton(
                      onPressed: () => Navigator.pop(c),
                      child: const Text('Закрыть'))
                ]));
  }

  @override
  Widget build(BuildContext context) {
    final result = _result;
    final choosePhoto = OutlinedButton(
        onPressed: _busy ? null : () => _run(_pick),
        child: const Text('Выбрать фото'));
    const measurementHelp = ExpansionTile(
        title: Text('Как измерять по эталону'),
        leading: Icon(Icons.info_outline),
        children: [
          Padding(
              padding: EdgeInsets.all(16),
              child: Text(
                  'Вертикальный эталон должен стоять рядом с деревом примерно на той же глубине. Снимайте целиком, без сильного наклона камеры. Его высота задаёт вертикальную ось; ширина кроны считается поперёк неё. Перспектива ограничивает метод: один отрезок её не исправляет. Результат — оценка проекции, не подтверждённая точность.'))
        ]);
    return Scaffold(
        appBar: AppBar(title: const Text('По известному объекту')),
        body: AppContentList(padding: const EdgeInsets.all(16), children: [
          if (_busy) const LinearProgressIndicator(),
          if (_error != null) Text(_error!),
          if (!_invalid) ...[
            // Retain list slots when a size becomes incomplete so the lazy
            // list does not dispose the focused length field.
            if (result == null) measurementHelp else const SizedBox.shrink(),
            if (result == null) choosePhoto else const SizedBox.shrink(),
            OutlinedButton(
                onPressed: _busy
                    ? null
                    : () => Navigator.push(
                        context,
                        MaterialPageRoute(
                            builder: (_) => const ReferenceHistoryPage())),
                child: const Text('Сохранённые измерения по эталону')),
            if (_image != null) ...[
              SizedBox(
                  height: 280,
                  child: Center(
                      child: AspectRatio(
                          aspectRatio: _width / _height,
                          child: ReferencePreview(
                              image: _image!,
                              width: _width,
                              height: _height,
                              report: _serverSnapshot?['report'] is Map
                                  ? Map<String, dynamic>.from(
                                      _serverSnapshot!['report'])
                                  : null,
                              reference: result?.toJson(),
                              correctionId: _serverSnapshot?['correction_id'],
                              hiddenLayers: _hiddenPhotoLayers,
                              onHiddenLayersChanged: (hidden) =>
                                  setState(() => _hiddenPhotoLayers = hidden),
                              tap: _treeTap)))),
              SpeciesAnnotationCard(
                  report: _serverSnapshot?['report'] as Map?,
                  onEdit: _busy ? null : () => _run(_editSpecies)),
              // Keep one list slot so an invalid/partial size does not dispose
              // the focused height field when the summary disappears.
              if (result == null)
                const SizedBox.shrink()
              else
                MeasurementResultsSummary(snapshot: _snapshot(result)),
              if (result == null)
                const SizedBox.shrink()
              else
                ExpansionTile(
                  key: const PageStorageKey('reference-photo-actions'),
                  title: const Text('Фото и экспорт'),
                  childrenPadding: const EdgeInsets.all(12),
                  children: [
                    choosePhoto,
                    ReportExportButton(
                        enabled: !_busy,
                        load: () async => ReportExportData(
                            hiddenLayers: _hiddenPhotoLayers,
                            snapshot: _snapshot(result),
                            record: {'analysis_id': _serverAnalysisId ?? _id},
                            photo: _image,
                            local: true)),
                    measurementHelp,
                  ],
                ),
              Ui.sectionTitle(context, 'Выделение дерева на этом фото'),
              Text(_serverSnapshot?['correction_id'] != null
                  ? 'Сохранённый контур · зелёная маска'
                  : 'Проверьте выделение нужного дерева.'),
              if ((_serverSnapshot?['report'] as Map?)?['tree'] is Map &&
                  _serverSnapshot!['report']['tree']['detected'] == false)
                const Text(
                    'В сохранённом результате модель не нашла дерево. Ручная разметка остаётся доступна.'),
              OutlinedButton(
                  onPressed: _busy ? null : () => _run(_editTreeContour),
                  child: const Text('Создать / исправить контур дерева')),
              OutlinedButton(
                  onPressed: _busy ? null : () => _run(_selectTree),
                  child: const Text('Выбрать дерево для модели')),
              OutlinedButton(
                  onPressed: _busy ? null : () => _run(_analyzeTree),
                  child: const Text('Выделить дерево на этом фото')),
              if (_serverMessage != null) Text(_serverMessage!),
              SurveyEnvironmentEditor(
                  controller: _environment, original: _image, enabled: !_busy),
              Ui.sectionTitle(context, '1. Размер эталона'),
              TextField(
                  key: const ValueKey('reference-length-input'),
                  enabled: !_busy,
                  controller: _length,
                  keyboardType:
                      const TextInputType.numberWithOptions(decimal: true),
                  decoration: const InputDecoration(
                      labelText: 'Реальная высота эталона'),
                  onChanged: (_) {
                    setState(() {
                      _saved = false;
                      _serverMessage = null;
                    });
                    _save().catchError((Object e) {
                      if (mounted) setState(() => _error = '$e');
                    });
                  }),
              DropdownButton<String>(
                  value: _unit,
                  items: const [
                    DropdownMenuItem(value: 'm', child: Text('Метры')),
                    DropdownMenuItem(value: 'cm', child: Text('Сантиметры'))
                  ],
                  onChanged: _busy
                      ? null
                      : (v) => _run(() async {
                            _unit = v!;
                          })),
              CheckboxListTile(
                  value: _plane,
                  title: const Text(
                      'Эталон и дерево примерно на одной глубине; перспектива мала'),
                  onChanged: _busy
                      ? null
                      : (v) => _run(() async {
                            _plane = v == true;
                          })),
              OutlinedButton(
                  onPressed: _busy
                      ? null
                      : () => _run(() async {
                            final r =
                                await Navigator.push<Map<String, dynamic>>(
                                    context,
                                    MaterialPageRoute(
                                        builder: (_) => MaskDrawingPage(
                                            originalImageBase64:
                                                base64Encode(_image!),
                                            editorState: _outline == null
                                                ? null
                                                : ContourEditorState.fromJson(
                                                    _outline!),
                                            sessionToken: _token)));
                            await _guard();
                            if (r != null) {
                              _outline =
                                  Map<String, dynamic>.from(r['editor_state']);
                            }
                          }),
                  child: Text(_outline == null
                      ? 'Обвести известный объект'
                      : 'Изменить контур эталона')),
              OutlinedButton(
                  onPressed: _busy
                      ? null
                      : () => _run(
                          () => _line('reference', 'Основание и верх эталона')),
                  child: const Text('Отметить высоту эталона')),
              OutlinedButton(
                  onPressed: _busy
                      ? null
                      : () => _run(
                          () => _line('tree', 'Основание и вершина дерева')),
                  child: const Text('Отметить высоту дерева')),
              OutlinedButton(
                  onPressed: _busy
                      ? null
                      : () => _run(() =>
                          _line('crown', 'Левый и правый край именно кроны')),
                  child: const Text('Отметить ширину кроны')),
              ExpansionTile(
                key: const PageStorageKey('reference-extra-measurements'),
                title: const Text('Дополнительные измерения'),
                childrenPadding: const EdgeInsets.all(12),
                children: [
                  const Text(
                      'Дополнительно: отметьте нижнюю границу живой кроны и её верх; для сечения ствола — края коры и локальную ось рядом с сечением. Не включайте листву. Разметка необязательна.'),
                  for (final entry in {
                    'crown_height': 'Низ живой кроны и верх',
                    'trunk': 'Края ствола в выбранном сечении',
                    'trunk_axis': 'Ось участка ствола у сечения'
                  }.entries)
                    OutlinedButton(
                        onPressed: _busy
                            ? null
                            : () => _run(() => _line(entry.key, entry.value)),
                        child: Text(entry.value)),
                  if (_crownHeight.isNotEmpty ||
                      _trunk.isNotEmpty ||
                      _trunkAxis.isNotEmpty)
                    TextButton(
                        onPressed: _busy
                            ? null
                            : () => _run(() async {
                                  _crownHeight = [];
                                  _trunk = [];
                                  _trunkAxis = [];
                                }),
                        child: const Text('Очистить дополнительную разметку')),
                ],
              ),
              if (result == null)
                Text(_calculationError ??
                    'Для расчёта введите положительную высоту, обведите эталон, отметьте три отрезка и подтвердите условия съёмки.'),
              if (result != null) ...[
                ExpansionTile(
                  title: const Text('Расширенный анализ'),
                  childrenPadding: const EdgeInsets.all(12),
                  children: [
                    if (_measurementVersion == 2)
                      GeometryReport(data: result.geometry),
                    const Text(
                        'Высота и ширина - проекции по вертикальному эталону. DBH не измерен; β по одному фото не определяется. Нужны отдельные измерения и динамический эксперимент.'),
                    const Text(
                        'Ручной вид не меняет предсказание модели и не подтверждает метку для обучения.'),
                  ],
                ),
                FilledButton(
                    onPressed: _busy ? null : () => _run(() async {}),
                    child: const Text('Сохранить отчёт на устройстве')),
                FilledButton(
                    onPressed: _busy ? null : () => _run(_uploadReport),
                    child: const Text('Сохранить в аккаунте / новую версию')),
                OutlinedButton(
                    onPressed: () => _run(_export),
                    child: const Text('Экспорт результата и разметки')),
              ],
              const ExpansionTile(
                title: Text('Хранение и ограничения'),
                childrenPadding: EdgeInsets.all(12),
                children: [
                  Text(
                      'Локальная копия остаётся на этом устройстве. После «Сохранить в аккаунте» отчёт доступен в истории на других устройствах. Неотправленные данные не переживут удаление приложения. AR и исходный отчёт не перезаписываются.'),
                  Text(
                      'Эталон и дерево должны быть на одной глубине. Перспектива автоматически не исправляется; полевая точность не подтверждена.'),
                  Text(
                      'Маска показывает дерево целиком. Принятие контура не подтверждает точность размеров.'),
                ],
              ),
              if (_saved) const Text('Сохранено на этом устройстве'),
            ],
          ],
        ]));
  }
}

class ReferenceHistoryPage extends StatefulWidget {
  const ReferenceHistoryPage({super.key});
  @override
  State<ReferenceHistoryPage> createState() => _ReferenceHistoryPageState();
}

class _ReferenceHistoryPageState extends State<ReferenceHistoryPage> {
  List<Map<String, dynamic>> rows = [];
  String? error;
  bool invalid = false;
  bool loading = true;
  @override
  void initState() {
    super.initState();
    CorrectionsService.authChanges.addListener(_invalidate);
    _load();
  }

  void _invalidate() {
    if (mounted) {
      setState(() {
        invalid = true;
        rows = [];
        error = 'Аккаунт изменился.';
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
      final s = CorrectionsService(),
          t = await CorrectionsService.currentToken();
      final result = await referenceStore().list(await s.owner(t));
      await s.checkSession(t);
      if (mounted && !invalid) {
        setState(() {
          rows = result;
          loading = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          error = '$e';
          loading = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) => Scaffold(
      appBar: AppBar(title: const Text('Измерения по эталону')),
      body: AppContentList(children: [
        if (loading) const LinearProgressIndicator(),
        if (error != null) Text(error!),
        if (!loading && rows.isEmpty && error == null)
          const Text('На этом устройстве пока нет измерений по эталону.'),
        for (final r in rows)
          ListTile(
              title: const Text('По известному объекту'),
              subtitle: Text(r['saved_at'] ?? ''),
              onTap: () async {
                await Navigator.push(
                    context,
                    MaterialPageRoute(
                        builder: (_) =>
                            ReferenceMeasurementPage(draftId: r['draft_id'])));
                if (!invalid) await _load();
              })
      ]));
}
