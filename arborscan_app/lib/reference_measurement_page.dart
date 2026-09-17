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
  const ReferenceMeasurementPage(
      {super.key, this.draftId, this.service, this.drafts, this.pickPhoto});
  @override
  State<ReferenceMeasurementPage> createState() =>
      _ReferenceMeasurementPageState();
}

class _ReferenceMeasurementPageState extends State<ReferenceMeasurementPage> {
  late final _service = widget.service ?? CorrectionsService(),
      _store = widget.drafts ?? referenceStore();
  final _length = TextEditingController();
  String? _token, _owner, _id, _error;
  String _unit = 'm';
  bool _busy = true, _invalid = false, _plane = false;
  bool _saved = false;
  String? _serverAnalysisId, _serverParentId, _serverMessage;
  Map<String, dynamic>? _serverSnapshot;
  Uint8List? _image;
  int _width = 0, _height = 0;
  Map<String, dynamic>? _outline;
  List<Offset> _reference = [], _tree = [], _crown = [];
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
        _image = null;
        _error = 'Аккаунт изменился. Откройте экран заново.';
      });
    }
  }

  @override
  void dispose() {
    CorrectionsService.authChanges.removeListener(_invalidate);
    _length.dispose();
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
        _saved = true;
        _serverAnalysisId = d['server_analysis_id'];
        _serverParentId = d['server_parent_id'];
        _serverSnapshot = d['server_snapshot'] == null
            ? null
            : Map<String, dynamic>.from(d['server_snapshot']);
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
          'method': 'known_object_segment_v1',
          'coordinates': 'normalized_oriented_image',
          'width': _width,
          'height': _height,
          'photo_sha256': sha256.convert(_image!).toString(),
          'length_text': _length.text,
          'unit': _unit,
          'same_plane': _plane,
          'outline': _outline,
          'reference': ReferenceMeasurement.encode(_reference),
          'tree': ReferenceMeasurement.encode(_tree),
          'crown': ReferenceMeasurement.encode(_crown),
          'report': _result == null
              ? null
              : {
                  'method': 'known_object_segment_v1',
                  'height_m': _result!.heightM,
                  'crown_width_m': _result!.crownM,
                  'dbh_m': null,
                  'beta_kg_s': null,
                },
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
      if (mounted) _error = '$e';
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _pick() async {
    final bytes = widget.pickPhoto != null
        ? await widget.pickPhoto!()
        : await (await ImagePicker().pickImage(source: ImageSource.gallery))
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
    _plane = false;
    _serverAnalysisId = null;
    _serverParentId = null;
    _serverSnapshot = null;
    _serverMessage = null;
  }

  Future<void> _uploadReport() async {
    final result = _result;
    if (result == null) return;
    _serverAnalysisId ??= reportUuid();
    await _save();
    final snapshot = <String, dynamic>{
      'version': 1,
      'kind': 'reference',
      'reference': result.toJson(),
      'report': {
        ...?_serverSnapshot?['report'],
        'method': 'known_object_segment_v1',
        'height_m': result.heightM,
        'crown_width_m': result.crownM,
        'dbh_m': null,
        'beta_kg_s': null
      },
      'ar': _serverSnapshot?['ar'],
      'environment': _serverSnapshot?['environment'],
      'captured_at': _serverSnapshot?['captured_at'],
      'change_source': 'reference',
      if (_serverSnapshot?['correction_id'] != null)
        'correction_id': _serverSnapshot!['correction_id'],
    };
    // Capture time belongs to the measurement, not to each network retry.
    _serverSnapshot ??= {
      'captured_at': DateTime.now().toUtc().toIso8601String()
    };
    snapshot['captured_at'] = _serverSnapshot!['captured_at'];
    await _save();
    final history = ReportHistoryService(auth: _service);
    await history.stage(
        token: _token!,
        localId: _id!,
        analysisId: _serverAnalysisId!,
        image: _image!,
        snapshot: snapshot,
        parentId: _serverParentId);
    await _guard();
    final response = await history.upload(_token!, _id!);
    await _guard();
    _serverParentId = response['record']['version_id'];
    _serverMessage = 'Сохранено в аккаунте';
  }

  Future<void> _line(String kind, String title) async {
    final points = kind == 'reference'
        ? _reference
        : kind == 'tree'
            ? _tree
            : _crown;
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
    } else {
      _crown = result;
    }
  }

  ReferenceMeasurement? get _result {
    if (_outline?['closed'] != true) return null;
    if (_outline?['width'] != _width || _outline?['height'] != _height) {
      return null;
    }
    try {
      return ReferenceMeasurement(
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
    } catch (_) {
      return null;
    }
  }

  Future<void> _export() async {
    final result = _result;
    if (result == null) return;
    final json = const JsonEncoder.withIndent('  ').convert({
      'measurement': result.toJson(),
      'photo_sha256': sha256.convert(_image!).toString(),
      'height_m': result.heightM,
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
    return Scaffold(
        appBar: AppBar(title: const Text('По известному объекту')),
        body: ListView(padding: const EdgeInsets.all(16), children: [
          if (_busy) const LinearProgressIndicator(),
          if (_error != null) Text(_error!),
          if (!_invalid) ...[
            const Text(
                'Вертикальный эталон должен стоять рядом с деревом примерно на той же глубине. Снимайте целиком, без сильного наклона камеры. Его высота задаёт вертикальную ось; ширина кроны считается поперёк неё. Перспектива ограничивает метод: один отрезок её не исправляет. Результат — оценка проекции, не подтверждённая точность.'),
            OutlinedButton(
                onPressed: _busy ? null : () => _run(_pick),
                child: const Text('Выбрать фото')),
            OutlinedButton(
                onPressed: _busy
                    ? null
                    : () => Navigator.push(
                        context,
                        MaterialPageRoute(
                            builder: (_) => const ReferenceHistoryPage())),
                child: const Text('Сохранённые измерения по эталону')),
            if (_image != null) ...[
              Image.memory(_image!, height: 200, fit: BoxFit.contain),
              TextField(
                  enabled: !_busy,
                  controller: _length,
                  keyboardType:
                      const TextInputType.numberWithOptions(decimal: true),
                  decoration: const InputDecoration(
                      labelText: 'Реальная высота эталона'),
                  onChanged: (_) {
                    setState(() { _saved = false; _serverMessage = null; });
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
              if (result == null)
                const Text(
                    'Для расчёта введите положительную высоту, обведите эталон, отметьте три отрезка и подтвердите условия съёмки.'),
              if (result != null) ...[
                const Text('Источник: По известному объекту'),
                Text('Высота: ${result.heightM.toStringAsFixed(2)} м'),
                Text('Ширина кроны: ${result.crownM.toStringAsFixed(2)} м'),
                const Text(
                    'DBH не измерен. β (кг/с) не определяется по одному фото: нужны динамический эксперимент и модель.'),
                FilledButton(
                    onPressed: _busy ? null : () => _run(() async {}),
                    child: const Text('Сохранить отчёт на устройстве')),
                FilledButton(
                    onPressed: _busy ? null : () => _run(_uploadReport),
                    child: const Text('Сохранить в аккаунте / новую версию')),
                if (_serverMessage != null) Text(_serverMessage!),
                OutlinedButton(
                    onPressed: () => _run(_export),
                    child: const Text('Экспорт результата и разметки')),
              ],
              const Text(
                  'Локальная копия остаётся на этом устройстве. После «Сохранить в аккаунте» отчёт доступен в истории на других устройствах. Неотправленные данные не переживут удаление приложения. AR и исходный отчёт не перезаписываются.'),
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
      if (mounted && !invalid) setState(() => rows = result);
    } catch (e) {
      if (mounted) setState(() => error = '$e');
    }
  }

  @override
  Widget build(BuildContext context) => Scaffold(
      appBar: AppBar(title: const Text('Измерения по эталону')),
      body: ListView(children: [
        if (error != null) Text(error!),
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
