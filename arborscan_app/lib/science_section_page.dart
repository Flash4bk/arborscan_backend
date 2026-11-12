import 'package:flutter/material.dart';
import 'app_theme.dart';
import 'corrections_service.dart';
import 'science_section.dart';
import 'science_section_service.dart';
import 'science_result_card.dart';

class ScienceSectionPage extends StatefulWidget {
  final String sourceVersionId;
  final ScienceSectionService? service;
  const ScienceSectionPage(
      {super.key, required this.sourceVersionId, this.service});
  @override
  State<ScienceSectionPage> createState() => _ScienceSectionPageState();
}

class _ScienceSectionPageState extends State<ScienceSectionPage> {
  late final _service = widget.service ?? ScienceSectionService();
  final _controllers = {
    for (final key in [
      'diameter_text',
      'mor_mpa_text',
      'diameter_source',
      'mor_source',
      'strength_definition'
    ])
      key: TextEditingController()
  };
  String _unit = 'mm';
  String? _diameterOrigin, _morOrigin, _token, _error;
  bool _assumptions = false, _loading = true, _busy = false, _invalid = false;
  bool? _available;
  Map<String, dynamic>? _operation;
  Future<bool> _lastWrite = Future.value(true);
  bool get _locked =>
      _operation?['attempted'] == true || _operation?['saved'] == true;
  bool get _editable => !_loading && !_busy && !_invalid && !_locked;

  Map<String, dynamic> get _form => {
        for (final entry in _controllers.entries) entry.key: entry.value.text,
        'diameter_unit': _unit,
        'diameter_origin': _diameterOrigin,
        'mor_origin': _morOrigin,
        'assumptions_confirmed': _assumptions,
      };
  @override
  void initState() {
    super.initState();
    CorrectionsService.authChanges.addListener(_invalidate);
    for (final c in _controllers.values) {
      c.addListener(_changed);
    }
    _load();
  }

  void _invalidate() {
    if (!mounted) return;
    setState(() {
      _invalid = true;
      _operation = null;
      _error = 'Аккаунт изменился. Откройте сценарий из своего отчёта заново.';
      for (final c in _controllers.values) {
        c.clear();
      }
      _diameterOrigin = null;
      _morOrigin = null;
      _assumptions = false;
    });
  }

  @override
  void dispose() {
    CorrectionsService.authChanges.removeListener(_invalidate);
    for (final c in _controllers.values) {
      c.dispose();
    }
    super.dispose();
  }

  Future<void> _load() async {
    try {
      _token = await CorrectionsService.currentToken();
      final draft = await _service.draft(_token!, widget.sourceVersionId);
      if (!mounted || _invalid) return;
      _operation = draft;
      final form = draft?['form'];
      if (form is Map) {
        for (final entry in _controllers.entries) {
          entry.value.text = form[entry.key] is String ? form[entry.key] : '';
        }
        if (['m', 'mm'].contains(form['diameter_unit'])) {
          _unit = form['diameter_unit'];
        }
        _diameterOrigin = sectionOrigins.containsKey(form['diameter_origin'])
            ? form['diameter_origin']
            : null;
        _morOrigin = sectionOrigins.containsKey(form['mor_origin'])
            ? form['mor_origin']
            : null;
        _assumptions = form['assumptions_confirmed'] == true;
      }
      try {
        _available = await _service.available(_token!);
      } catch (e) {
        _error = '$e';
      }
    } catch (e) {
      _error = '$e';
    }
    if (mounted && !_invalid) setState(() => _loading = false);
  }

  void _changed() {
    if (!_editable || _token == null) return;
    _lastWrite = _persist();
  }

  Future<bool> _persist() async {
    try {
      final draft =
          await _service.saveForm(_token!, widget.sourceVersionId, _form);
      if (mounted && !_invalid) setState(() => _operation = draft);
      return true;
    } catch (e) {
      if (mounted && !_invalid) setState(() => _error = '$e');
      return false;
    }
  }

  Future<void> _calculate() async {
    if (_busy ||
        _invalid ||
        _available != true ||
        _operation?['saved'] == true) {
      return;
    }
    setState(() {
      _busy = true;
      _error = null;
    });
    try {
      if (!await _lastWrite) {
        throw const CorrectionException(
            'Черновик не записан. Повторите сохранение входов.');
      }
      final inputs = _locked
          ? Map<String, dynamic>.from(_operation!['inputs'])
          : sectionInputs(_form);
      if (!_locked) {
        final saved =
            await _service.saveForm(_token!, widget.sourceVersionId, _form);
        if (!mounted || _invalid) return;
        _operation = saved;
      }
      await _service.calculate(_token!, widget.sourceVersionId, inputs);
    } catch (e) {
      if (mounted && !_invalid) {
        _error = e is FormatException ? e.message : '$e';
      }
    } finally {
      if (mounted && !_invalid) {
        try {
          _operation = await _service.draft(_token!, widget.sourceVersionId);
        } catch (e) {
          _error = '$e';
        }
        if (mounted && !_invalid) setState(() => _busy = false);
      }
    }
  }

  Widget _field(String key, String label,
          {int lines = 1, bool number = false}) =>
      Padding(
          padding: const EdgeInsets.only(bottom: 12),
          child: TextField(
              key: ValueKey(key),
              controller: _controllers[key],
              enabled: _editable,
              maxLines: lines,
              keyboardType: number
                  ? const TextInputType.numberWithOptions(decimal: true)
                  : TextInputType.multiline,
              decoration: InputDecoration(labelText: label)));
  Widget _origin(String key, String? value, ValueChanged<String?> change) =>
      Padding(
          padding: const EdgeInsets.only(bottom: 12),
          child: DropdownButtonFormField<String>(
              key: ValueKey(key),
              initialValue: value,
              isExpanded: true,
              decoration:
                  const InputDecoration(labelText: 'Происхождение величины'),
              items: [
                for (final e in sectionOrigins.entries)
                  DropdownMenuItem(value: e.key, child: Text(e.value))
              ],
              onChanged: _editable
                  ? (value) {
                      setState(() => change(value));
                      _changed();
                    }
                  : null));
  @override
  Widget build(BuildContext context) => Scaffold(
      appBar: AppBar(title: const Text('Сценарий сечения')),
      body: AppContentList(padding: const EdgeInsets.all(16), children: [
        if (_loading || _busy) const LinearProgressIndicator(),
        const Text(
            'Введите диаметр одного сечения и эффективную прочность при изгибе (MOR). Приложение не получает прочность из фотографии или вида дерева.'),
        const SizedBox(height: 12),
        if (_invalid)
          Text(_error!)
        else ...[
          if (_error != null)
            Padding(
                padding: const EdgeInsets.only(bottom: 12),
                child: Text(_error!, key: const ValueKey('science-error'))),
          if (_available == false)
            const Text(
                'Этот сервер пока не поддерживает сценарии сечения. Черновик можно заполнить на устройстве; анализ и прежние отчёты доступны.'),
          if (!_loading && _available == null)
            OutlinedButton(
                onPressed: _busy
                    ? null
                    : () {
                        setState(() => _loading = true);
                        _load();
                      },
                child: const Text('Проверить доступность сервера')),
          const SizedBox(height: 12),
          _field('diameter_text', 'Диаметр сечения', number: true),
          DropdownButtonFormField<String>(
              key: const ValueKey('diameter_unit'),
              initialValue: _unit,
              decoration: const InputDecoration(labelText: 'Единицы диаметра'),
              items: const [
                DropdownMenuItem(value: 'mm', child: Text('Миллиметры')),
                DropdownMenuItem(value: 'm', child: Text('Метры'))
              ],
              onChanged: _editable
                  ? (v) {
                      setState(() => _unit = v!);
                      _changed();
                    }
                  : null),
          const SizedBox(height: 12),
          _origin(
              'diameter_origin', _diameterOrigin, (v) => _diameterOrigin = v),
          _field('diameter_source', 'Как и где измерен диаметр / источник',
              lines: 2),
          _field('mor_mpa_text', 'Effective MOR, МПа', number: true),
          _origin('mor_origin', _morOrigin, (v) => _morOrigin = v),
          _field('mor_source', 'Источник effective MOR', lines: 2),
          _field('strength_definition',
              'Что означает прочность и какие поправки учтены',
              lines: 3),
          CheckboxListTile(
              contentPadding: EdgeInsets.zero,
              value: _assumptions,
              title: const Text(
                  'Сценарий здорового сплошного круглого сечения; дефекты и корни не учтены'),
              onChanged: _editable
                  ? (v) {
                      setState(() => _assumptions = v == true);
                      _changed();
                    }
                  : null),
          const Text(
              'Результат - предельный момент только этого сечения. Он не определяет устойчивость дерева целиком, критический ветер, β или вероятность повреждения. Входы не проверены полевым экспериментом.'),
          const SizedBox(height: 12),
          if (_locked &&
              _operation?['saved'] != true &&
              _operation?['validation_rejected'] != true)
            const Text(
                'Результат отправки пока не подтверждён. Входы закреплены; повторяется тот же запрос без создания второй версии.'),
          if (_operation?['validation_rejected'] == true)
            const Text(
                'Сервер отклонил входные данные (422). Версия не сохранена; исправьте входы.'),
          if (_operation?['validation_rejected'] == true)
            OutlinedButton(
                key: const ValueKey('edit-rejected-section'),
                onPressed: _busy
                    ? null
                    : () async {
                        try {
                          final draft = await _service.editRejected(
                              _token!, widget.sourceVersionId);
                          if (mounted && !_invalid) {
                            setState(() {
                              _operation = draft;
                              _error = null;
                            });
                          }
                        } catch (e) {
                          if (mounted && !_invalid) {
                            setState(() => _error = '$e');
                          }
                        }
                      },
                child: const Text('Исправить входы')),
          if (_operation?['saved'] == true) ...[
            const Text('Расчёт и новая версия подтверждены сервером.'),
            ScienceResultCard(result: _operation!['science_result']),
            FilledButton(
                onPressed: _busy
                    ? null
                    : () => Navigator.pop(
                        context, _operation!['version_id'] as String),
                child: const Text('Открыть сохранённую версию')),
          ] else
            FilledButton(
                key: const ValueKey('calculate-section'),
                onPressed: _loading ||
                        _busy ||
                        _available != true ||
                        _operation?['validation_rejected'] == true
                    ? null
                    : _calculate,
                child: Text(_locked
                    ? 'Повторить тот же запрос'
                    : 'Рассчитать и сохранить новую версию')),
          const SizedBox(height: 12),
          Text(_operation == null
              ? 'Черновик ещё не записан.'
              : 'Черновик входов хранится на этом устройстве отдельно для аккаунта. После подтверждения результат доступен в серверной истории. Удаление приложения удаляет неотправленный черновик.'),
          ExpansionTile(
              title: const Text('Исходная версия'),
              children: [SelectableText(widget.sourceVersionId)]),
        ],
      ]));
}
