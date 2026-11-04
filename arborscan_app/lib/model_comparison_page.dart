import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import 'app_theme.dart';
import 'corrections_service.dart';
import 'image_selection_service.dart';
import 'model_comparison_service.dart';

class ModelComparisonPage extends StatefulWidget {
  final ModelComparisonService? service;
  final Future<Uint8List?> Function()? pickImage;
  const ModelComparisonPage({super.key, this.service, this.pickImage});

  @override
  State<ModelComparisonPage> createState() => _ModelComparisonPageState();
}

class _ModelComparisonPageState extends State<ModelComparisonPage> {
  late final _service = widget.service ?? ModelComparisonService();
  ComparisonModels? _catalog;
  ComparisonPhoto? _photo;
  ModelComparisonResult? _result;
  ComparisonModel? _baseline, _candidate;
  String? _token, _error, _operation;
  bool _invalid = false, _unsupported = false;
  int _epoch = 0;
  bool get _busy => _operation != null;

  @override
  void initState() {
    super.initState();
    CorrectionsService.authChanges.addListener(_invalidate);
    _loadModels();
  }

  void _invalidate() {
    _epoch++;
    _service.cancel();
    if (!mounted) return;
    setState(() {
      _invalid = true;
      _catalog = null;
      _baseline = null;
      _candidate = null;
      _photo = null;
      _result = null;
      _token = null;
      _operation = null;
      _error = 'Аккаунт изменился. Откройте сравнение заново.';
    });
  }

  @override
  void dispose() {
    _epoch++;
    _service.cancel();
    CorrectionsService.authChanges.removeListener(_invalidate);
    super.dispose();
  }

  bool _current(int epoch) => mounted && !_invalid && epoch == _epoch;

  Future<bool> _identity(int epoch) async {
    if (!_current(epoch)) return false;
    if (await CorrectionsService.currentToken() != _token) {
      _invalidate();
      return false;
    }
    return _current(epoch);
  }

  Future<void> _failure(Object e, int epoch) async {
    if (!await _identity(epoch)) return;
    setState(() {
      _error = '$e';
      if (e is CorrectionException) {
        _unsupported = [404, 405, 501].contains(e.statusCode);
        if (e.statusCode == 401 || e.statusCode == 403) {
          _catalog = null;
          _baseline = null;
          _candidate = null;
          _photo = null;
          _result = null;
        }
      }
    });
  }

  Future<void> _loadModels() async {
    if (_busy || _invalid) return;
    final epoch = ++_epoch;
    setState(() {
      _operation = 'models';
      _error = null;
    });
    try {
      _token ??= await CorrectionsService.currentToken();
      if (!_current(epoch)) return;
      final catalog = await _service.models(_token!);
      if (!await _identity(epoch)) return;
      final available = catalog.available;
      ComparisonModel? candidate;
      for (final model in available) {
        if (model.version == catalog.defaultVersion) candidate = model;
      }
      candidate ??= available.isEmpty ? null : available.last;
      ComparisonModel? baseline;
      for (final model in available) {
        if (model.version == 2 && model.version != candidate?.version) {
          baseline = model;
        }
      }
      for (final model in available) {
        if (model.version != candidate?.version) {
          baseline ??= model;
        }
      }
      setState(() {
        _catalog = catalog;
        _baseline = baseline;
        _candidate = candidate;
        _unsupported = false;
        _result = null;
        if (available.length < 2) {
          _error =
              'На сервере нужны две доступные версии с сохранёнными весами.';
        }
      });
    } catch (e) {
      await _failure(e, epoch);
    } finally {
      if (_current(epoch)) setState(() => _operation = null);
    }
  }

  Future<void> _pick() async {
    if (_busy || _invalid) return;
    final epoch = ++_epoch;
    setState(() {
      _operation = 'picker';
      _error = null;
    });
    try {
      final raw = widget.pickImage != null
          ? await widget.pickImage!()
          : await (await ImageSelectionService().pick(ImageSource.gallery))
              ?.readAsBytes();
      if (!await _identity(epoch) || raw == null) return;
      final photo = ComparisonPhoto.decode(raw);
      if (!await _identity(epoch)) return;
      setState(() {
        _photo = photo;
        _result = null;
      });
    } catch (e) {
      await _failure(e, epoch);
    } finally {
      if (_current(epoch)) setState(() => _operation = null);
    }
  }

  Future<void> _compare() async {
    if (_busy ||
        _invalid ||
        _photo == null ||
        _baseline == null ||
        _candidate == null ||
        _baseline!.version == _candidate!.version) {
      return;
    }
    final epoch = ++_epoch;
    setState(() {
      _operation = 'comparison';
      _error = null;
      _result = null;
    });
    try {
      final result =
          await _service.compare(_token!, _photo!, _baseline!, _candidate!);
      if (!await _identity(epoch)) return;
      setState(() => _result = result);
    } catch (e) {
      await _failure(e, epoch);
    } finally {
      if (_current(epoch)) setState(() => _operation = null);
    }
  }

  void _cancel() {
    _epoch++;
    _service.cancel();
    setState(() {
      _operation = null;
      _error = 'Запрос отменён. Выбранное фото остаётся на экране.';
    });
  }

  Widget _controls() =>
      Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
        if (_catalog != null && _error != null)
          Text(_error!, key: const ValueKey('comparison-error')),
        if (_catalog != null)
          FilledButton.icon(
              onPressed: _busy ||
                      _unsupported ||
                      _photo == null ||
                      _baseline == null ||
                      _candidate == null ||
                      _baseline!.version == _candidate!.version
                  ? null
                  : _compare,
              icon: const Icon(Icons.compare),
              label: const Text('Сравнить на этом фото')),
        if (_busy && _operation != 'picker')
          TextButton(onPressed: _cancel, child: const Text('Отменить запрос')),
      ]);

  Widget _modelChoice(String label, ComparisonModel? value, bool first) =>
      DropdownButtonFormField<int>(
        key: ValueKey(
            'comparison-${first ? 'baseline' : 'candidate'}-${value?.version}'),
        initialValue: value?.version,
        isExpanded: true,
        decoration: InputDecoration(labelText: label),
        items: [
          for (final model in _catalog?.available ?? <ComparisonModel>[])
            DropdownMenuItem(
                value: model.version,
                child: Text(
                    'v${model.version} · ${model.weightsSha256.substring(0, 8)}')),
        ],
        onChanged: _busy
            ? null
            : (version) => setState(() {
                  final selected = _catalog!.available
                      .firstWhere((model) => model.version == version);
                  if (first) {
                    _baseline = selected;
                  } else {
                    _candidate = selected;
                  }
                  _result = null;
                  _error = null;
                }),
      );

  Widget _preview(ComparedModel model) => Card(
      child: Padding(
          padding: const EdgeInsets.all(12),
          child:
              Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Text('Модель v${model.model.version}',
                style: Theme.of(context).textTheme.titleMedium),
            Text(model.detected ? 'Маска получена' : 'Маска не получена'),
            const SizedBox(height: 8),
            LayoutBuilder(builder: (context, constraints) {
              final aspect = _photo!.width / _photo!.height;
              final height = (constraints.maxWidth / aspect).clamp(0.0, 300.0);
              return Center(
                  child: SizedBox(
                      width: height * aspect,
                      height: height,
                      child: Stack(fit: StackFit.expand, children: [
                        Image.memory(_photo!.previewPng,
                            fit: BoxFit.fill, gaplessPlayback: true),
                        Image.memory(model.overlayPng,
                            key: ValueKey(
                                'comparison-overlay-${model.model.version}'),
                            fit: BoxFit.fill,
                            gaplessPlayback: true),
                      ])));
            }),
            if (model.warnings.isNotEmpty)
              ExpansionTile(
                  title: const Text('Сведения о выделении'),
                  children: [
                    for (final warning in model.warnings) Text(warning)
                  ]),
          ])));

  Widget _table(ModelComparisonResult result) {
    Widget cell(String text, {bool heading = false}) => Padding(
        padding: const EdgeInsets.all(8),
        child: Text(text,
            style:
                heading ? const TextStyle(fontWeight: FontWeight.w600) : null));
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Text('Сравнение результатов',
          style: Theme.of(context).textTheme.titleMedium),
      const SizedBox(height: 8),
      Table(
          key: const ValueKey('model-comparison-table'),
          columnWidths: const {
            0: FlexColumnWidth(1.15),
            1: FlexColumnWidth(1),
            2: FlexColumnWidth(1)
          },
          border: TableBorder.all(color: Theme.of(context).dividerColor),
          children: [
            TableRow(children: [
              cell('Показатель', heading: true),
              for (final m in result.models)
                cell('Модель v${m.model.version}', heading: true),
            ]),
            for (final metric in [
              'Precision',
              'Recall',
              'mAP50',
              'mAP50-95',
              'Средний IoU'
            ])
              TableRow(children: [
                cell(metric),
                cell('Не оценено'),
                cell('Не оценено')
              ]),
            TableRow(children: [
              cell('Время модели, с/фото'),
              for (final m in result.models)
                cell(m.meanSeconds.toStringAsFixed(3).replaceAll('.', ',')),
            ]),
          ]),
      const SizedBox(height: 8),
      const Text('Для Precision, Recall, mAP и IoU нужна проверенная разметка. '
          'Сравнение двух предсказанных масок её не заменяет.'),
      Text(
          'Время: среднее трёх вычислений после одного прогрева, без загрузки фото. '
          'CPU, 1 поток; imgsz=${result.imageSize}, порог=${result.confidenceThreshold}.'),
      Text('Фото ${_photo!.originalWidth}×${_photo!.originalHeight}; '
          'обработка обеих версий ${_photo!.width}×${_photo!.height}, максимум $comparisonProcessingMaxSide по стороне. '
          'Размеры дерева здесь не рассчитываются.'),
    ]);
  }

  @override
  Widget build(BuildContext context) => Scaffold(
      appBar: AppBar(title: const Text('Сравнение моделей'), actions: [
        IconButton(
            tooltip: 'Обновить список моделей',
            onPressed: _busy || _invalid ? null : _loadModels,
            icon: const Icon(Icons.refresh)),
      ]),
      body: AppContentList(children: [
        if (_busy) const LinearProgressIndicator(),
        if (_error != null && (_invalid || _catalog == null))
          Text(_error!, key: const ValueKey('comparison-error')),
        if (!_invalid) ...[
          const Text('Одна фотография и одинаковые настройки для обеих версий. '
              'Запрос не переключает действующую модель и не сохраняет новый отчёт.'),
          if (_catalog != null) ...[
            _modelChoice('Первая модель', _baseline, true),
            _modelChoice('Вторая модель', _candidate, false),
            if (_baseline?.version == _candidate?.version && _baseline != null)
              const Text('Выберите две разные версии.'),
            OutlinedButton.icon(
                onPressed: _busy ? null : _pick,
                icon: const Icon(Icons.photo_library_outlined),
                label: const Text('Выбрать фото для сравнения')),
          ],
          if (_photo != null && _result == null)
            Image.memory(_photo!.previewPng,
                key: const ValueKey('comparison-selected-photo'),
                fit: BoxFit.contain,
                height: 300),
          _controls(),
          if (_result != null) ...[
            for (final model in _result!.models) _preview(model),
            _table(_result!),
            const Text('Результат доступен только на этом экране. '
                'Сохранённые обследования и решения по контурам не меняются.'),
          ],
        ],
      ]));
}
