import 'dart:convert';
import 'dart:async';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import 'api_config.dart';
import 'corrections_service.dart';
import 'report_history_service.dart';
import 'model_quality_service.dart';
import 'saved_corrections_page.dart';

class ModelQualityPage extends StatefulWidget {
  final ModelQualityService? service;
  const ModelQualityPage({super.key, this.service});
  @override
  State<ModelQualityPage> createState() => _ModelQualityPageState();
}

class _ModelQualityPageState extends State<ModelQualityPage> {
  late final _service = widget.service ?? ModelQualityService();
  String _kind = 'segmentation';
  String? _token, _owner, _error;
  bool _busy = true, _invalid = false;
  Map<String, dynamic>? _status, _data;
  Timer? _poll;
  bool _polling = false;
  @override
  void initState() {
    super.initState();
    CorrectionsService.authChanges.addListener(_invalidate);
    _load();
    _poll = Timer.periodic(const Duration(seconds: 5), (_) => _refreshStatus());
  }

  Future<void> _refreshStatus() async {
    if (_busy || _invalid || _polling || _token == null) return;
    _polling = true;
    try {
      final status = await _request('/status');
      if (mounted && !_invalid) setState(() => _status = status);
    } catch (e) {
      if (mounted && !_invalid) {
        setState(() => _error = 'Обновление статуса: $e');
      }
    } finally {
      _polling = false;
    }
  }

  void _invalidate() {
    if (mounted) {
      setState(() {
        _invalid = true;
        _status = null;
        _data = null;
        _error = 'Аккаунт изменился.';
      });
    }
  }

  @override
  void dispose() {
    _poll?.cancel();
    CorrectionsService.authChanges.removeListener(_invalidate);
    super.dispose();
  }

  Future<Map<String, dynamic>> _request(String path,
      {Map<String, dynamic>? body}) {
    final r = http.Request(
        body == null ? 'GET' : 'POST', ApiConfig.v4('/v4/model-quality$path'));
    if (body != null) {
      r.headers['Content-Type'] = 'application/json';
      r.body = jsonEncode(body);
    }
    return _service.request(_token!, r);
  }

  Future<void> _load() async {
    try {
      _token ??= await CorrectionsService.currentToken();
      _owner ??= await _service.auth.owner(_token!);
      final s = await _request('/status');
      final d = await _request('/data?model_type=$_kind');
      if (mounted && !_invalid) {
        setState(() {
          _status = s;
          _data = d;
        });
      }
    } catch (e) {
      if (mounted) setState(() => _error = 'ML-сервис: $e');
    }
    if (mounted) setState(() => _busy = false);
  }

  Future<void> _run(Future<void> Function() action) async {
    if (_busy || _invalid) return;
    setState(() {
      _busy = true;
      _error = null;
    });
    try {
      await _service.auth.checkSession(_token!);
      await action();
      if (!_invalid) await _load();
    } catch (e) {
      if (mounted) setState(() => _error = '$e');
    }
    if (mounted) setState(() => _busy = false);
  }

  Future<String> _operation(String name) async {
    final prefs = await SharedPreferences.getInstance();
    final key = 'ml_pending_$_owner$name';
    final id = prefs.getString(key) ?? reportUuid();
    await prefs.setString(key, id);
    return id;
  }

  Future<void> _clearOperation(String name) async {
    final p = await SharedPreferences.getInstance();
    await p.remove('ml_pending_$_owner$name');
  }

  Future<void> _diagnostic(String id, String key, int index) async {
    final parts = key.split('_');
    final data = await _request(
        '/models/$id/diagnostics/${parts[0]}/${parts[1]}/$index');
    if (!mounted || _invalid) return;
    await showDialog<void>(
        context: context,
        builder: (c) => AlertDialog(
                title: Text('$key · ${data['case']['correction_id']}'),
                content: SingleChildScrollView(
                    child: Column(mainAxisSize: MainAxisSize.min, children: [
                  Image.memory(base64Decode(data['image_base64'])),
                  const Text(
                      'Зелёный: совпадение; красный: пропущено моделью; синий: лишнее. Изображение уменьшено только для просмотра.'),
                  Text('${data['case']}'),
                ])),
                actions: [
                  TextButton(
                      onPressed: () => Navigator.pop(c),
                      child: const Text('Закрыть'))
                ]));
  }

  Future<void> _confirmLabel(Map r) async {
    final history = await _request(
        '/labels/${r['owner_id']}/${Uri.encodeComponent(r['correction_id'])}');
    final items = history['items'] as List;
    final previous = items.isEmpty ? null : items.first;
    final operationName = 'label-${r['owner_id']}-${r['correction_id']}';
    final prefs = await SharedPreferences.getInstance();
    if (previous != null &&
        prefs.getString('ml_pending_$_owner$operationName') == previous['id']) {
      await _clearOperation(operationName);
      return; // A lost response was already committed.
    }
    final scientific =
        TextEditingController(text: previous?['label']?['scientific_name']);
    final taxon = TextEditingController(text: previous?['label']?['taxon_id']);
    final evidence =
        TextEditingController(text: previous?['label']?['evidence']);
    final russian =
        TextEditingController(text: previous?['label']?['russian_name']);
    final group = TextEditingController(text: previous?['label']?['group_id']);
    var authority = previous?['label']?['authority'] ?? 'GBIF';
    var rank = previous?['label']?['rank'] ?? 'species';
    bool confirmed = false;
    if (!mounted || _invalid) return;
    final payload = await showDialog<Map<String, dynamic>>(
        context: context,
        builder: (c) => StatefulBuilder(
            builder: (c, set) => AlertDialog(
                    title: const Text('Отдельная метка породы'),
                    content: SingleChildScrollView(
                        child:
                            Column(mainAxisSize: MainAxisSize.min, children: [
                      const Text(
                          'Принятие контура не подтверждает вид. Укажите источник определения. Род не является видом.'),
                      TextField(
                          controller: scientific,
                          decoration: const InputDecoration(
                              labelText: 'Научное название')),
                      DropdownButton<String>(
                          value: rank,
                          items: const [
                            DropdownMenuItem(
                                value: 'species', child: Text('Вид')),
                            DropdownMenuItem(
                                value: 'genus', child: Text('Только род'))
                          ],
                          onChanged: (v) => set(() => rank = v!)),
                      DropdownButton<String>(
                          value: authority,
                          items: ['GBIF', 'POWO', 'manual_reference']
                              .map((s) =>
                                  DropdownMenuItem(value: s, child: Text(s)))
                              .toList(),
                          onChanged: (v) => set(() => authority = v!)),
                      TextField(
                          controller: taxon,
                          decoration: const InputDecoration(
                              labelText: 'Идентификатор таксона в источнике')),
                      TextField(
                          controller: russian,
                          decoration: const InputDecoration(
                              labelText:
                                  'Достоверное русское название (необязательно)')),
                      TextField(
                          controller: evidence,
                          decoration: const InputDecoration(
                              labelText: 'Основание определения / источник')),
                      TextField(
                          controller: group,
                          decoration: const InputDecoration(
                              labelText:
                                  'ID одного дерева/серии, если известен')),
                      CheckboxListTile(
                          value: confirmed,
                          onChanged: (v) => set(() => confirmed = v == true),
                          title: const Text(
                              'Я подтверждаю эту метку для классификации')),
                      if (previous != null)
                        ExpansionTile(
                            title: const Text('История меток'),
                            children: [
                              SelectableText(const JsonEncoder.withIndent('  ')
                                  .convert(items))
                            ]),
                    ])),
                    actions: [
                      TextButton(
                          onPressed: () => Navigator.pop(c),
                          child: const Text('Отмена')),
                      FilledButton(
                          onPressed: () => Navigator.pop(c, {
                                'scientific_name': scientific.text.trim(),
                                'taxon_id': taxon.text.trim(),
                                'authority': authority,
                                'rank': rank,
                                'russian_name': russian.text.trim().isEmpty
                                    ? null
                                    : russian.text.trim(),
                                'evidence': evidence.text.trim(),
                                'confirmed': confirmed,
                                'group_id': group.text.trim().isEmpty
                                    ? null
                                    : group.text.trim()
                              }),
                          child: const Text('Сохранить метку'))
                    ])));
    for (final controller in [scientific, taxon, evidence, russian, group]) {
      controller.dispose();
    }
    if (payload == null || _invalid) return;
    final name = operationName;
    await _request('/labels', body: {
      ...payload,
      'operation_id': await _operation(name),
      'owner_id': r['owner_id'],
      'correction_id': r['correction_id'],
      'parent_id': previous?['id']
    });
    await _clearOperation(name);
  }

  @override
  Widget build(BuildContext context) => Scaffold(
      appBar: AppBar(title: const Text('Модели и качество'), actions: [
        IconButton(
            onPressed: _busy ? null : () => _run(() async {}),
            icon: const Icon(Icons.refresh))
      ]),
      body: ListView(padding: const EdgeInsets.all(16), children: [
        if (_busy) const LinearProgressIndicator(),
        if (_error != null) Text(_error!),
        if (!_invalid) ...[
          const Text(
              'Сегментация и классификация обучаются раздельно. Завершение задачи не включает модель в production.'),
          if (_status != null)
            Text(_status!['worker']?['online'] == true
                ? 'Worker доступен (CPU).'
                : 'Worker недоступен; прогресс сейчас не подтверждён.'),
          for (final a in (_status?['active'] as List? ?? []))
            if (a['model_type'] == 'segmentation') ...[
              Text(
                  'Активная сегментация: ${a['model_id'] ?? 'исходная модель'}. Версия реестра: ${a['generation']}'),
              if (a['model_id'] != null)
                OutlinedButton(
                    onPressed: _busy
                        ? null
                        : () => _run(() async {
                              final ok = await showDialog<bool>(
                                  context: context,
                                  builder: (c) => AlertDialog(
                                          title: const Text(
                                              'Откат к исходной модели?'),
                                          content: const Text(
                                              'Новые запросы будут использовать исходные веса. Сохранённые отчёты останутся прежними.'),
                                          actions: [
                                            TextButton(
                                                onPressed: () =>
                                                    Navigator.pop(c, false),
                                                child: const Text('Отмена')),
                                            TextButton(
                                                onPressed: () =>
                                                    Navigator.pop(c, true),
                                                child: const Text('Откатить'))
                                          ]));
                              if (ok == true) {
                                await _request('/activate', body: {
                                  'model_id': null,
                                  'expected_generation': a['generation']
                                });
                              }
                            }),
                    child: const Text('Откатить к исходной модели')),
            ],
          DropdownButton<String>(
              value: _kind,
              items: const [
                DropdownMenuItem(
                    value: 'segmentation', child: Text('Сегментация: дерево')),
                DropdownMenuItem(
                    value: 'classification',
                    child: Text('Локальная классификация: эксперимент'))
              ],
              onChanged: _busy
                  ? null
                  : (v) => _run(() async {
                        _kind = v!;
                      })),
          if (_kind == 'classification')
            const Text(
                'Pl@ntNet остаётся внешним распознавателем. Эта задача обучает отдельного локального кандидата, а не Pl@ntNet.'),
          Text(
              'Допущено: ${(_data?['eligible'] as List?)?.length ?? 0}. Исключено: ${(_data?['excluded'] as List?)?.length ?? 0}.'),
          const Text(
              'Старый каталог verified сам по себе не доказывает подтверждение. Такие записи вне выборки до явной модерации.'),
          for (final r in [...?_data?['eligible'], ...?_data?['excluded']])
            ExpansionTile(
                title: Text(r['reason'] ?? 'Допущено к $_kind'),
                subtitle: Text('${r['analysis_id']}'),
                children: [
                  OutlinedButton(
                      onPressed: _busy
                          ? null
                          : () => Navigator.push(
                              context,
                              MaterialPageRoute(
                                  builder: (_) => SavedCorrectionsPage(
                                      correctionId: r['correction_id'],
                                      reviewOwner: r['owner_id']))),
                      child: const Text('Оригинал, маска и модерация')),
                  OutlinedButton(
                      onPressed:
                          _busy ? null : () => _run(() => _confirmLabel(r)),
                      child: const Text('Исправить / подтвердить породу')),
                  if (r['fidelity'] != null)
                    Text('Потери преобразования маски: ${r['fidelity']}'),
                ]),
          FilledButton(
              onPressed: _busy
                  ? null
                  : () => _run(() async {
                        final name = 'snapshot-$_kind';
                        await _request('/snapshots', body: {
                          'operation_id': await _operation(name),
                          'model_type': _kind
                        });
                        await _clearOperation(name);
                      }),
              child: const Text('Создать зафиксированный снимок')),
          for (final s in (_status?['snapshots'] as List? ?? []))
            if (s['model_type'] == _kind)
              ExpansionTile(
                  title: Text('Снимок ${s['created_at']}'),
                  subtitle: Text(
                      'Примеров: ${(s['manifest']['items'] as List).length}'),
                  children: [
                    SelectableText(const JsonEncoder.withIndent('  ')
                        .convert(s['manifest'])),
                    FilledButton(
                        onPressed: _busy ||
                                s['manifest']['training_ready'] != true
                            ? null
                            : () => _run(() async {
                                  final approved = await showDialog<bool>(
                                      context: context,
                                      builder: (c) => AlertDialog(
                                              title:
                                                  const Text('Пробная задача'),
                                              content: const Text(
                                                  '1 эпоха, CPU, 320 px, batch 1, лимит 30 минут. Это не доказательство улучшения. Текущая модель останется прежней.'),
                                              actions: [
                                                TextButton(
                                                    onPressed: () =>
                                                        Navigator.pop(c, false),
                                                    child:
                                                        const Text('Отмена')),
                                                FilledButton(
                                                    onPressed: () =>
                                                        Navigator.pop(c, true),
                                                    child:
                                                        const Text('Запустить'))
                                              ]));
                                  if (approved != true) return;
                                  final name = 'job-${s['id']}';
                                  await _request('/jobs', body: {
                                    'operation_id': await _operation(name),
                                    'snapshot_id': s['id'],
                                    'epochs': 1,
                                    'imgsz': 320
                                  });
                                  await _clearOperation(name);
                                }),
                        child: const Text('Запустить пробное обучение')),
                    if (s['manifest']['training_ready'] != true)
                      const Text(
                          'Недостаточно независимых групп для train / validation / test.'),
                  ]),
          for (final j in (_status?['jobs'] as List? ?? []))
            Card(
                child: ListTile(
                    title: Text('Задача: ${j['state']}'),
                    subtitle: Text('${j['progress']}'),
                    trailing: ['queued', 'running', 'cancel_requested']
                            .contains(j['state'])
                        ? TextButton(
                            onPressed: _busy
                                ? null
                                : () => _run(() async {
                                      await _request('/jobs/${j['id']}/cancel',
                                          body: {});
                                    }),
                            child: const Text('Отменить'))
                        : null)),
          for (final m in (_status?['models'] as List? ?? []))
            ExpansionTile(
                title: Text('Кандидат ${m['model_type']}'),
                children: [
                  SelectableText(const JsonEncoder.withIndent('  ')
                      .convert(m['metadata'])),
                  for (final key in [
                    'baseline_val',
                    'candidate_val',
                    'baseline_test',
                    'candidate_test'
                  ])
                    for (var i = 0;
                        i <
                            (m['metadata']['evaluation']?[key]?['cases']
                                        as List? ??
                                    [])
                                .length;
                        i++)
                      TextButton(
                          onPressed: _busy
                              ? null
                              : () => _run(() => _diagnostic(m['id'], key, i)),
                          child: Text('Наложение $key · пример ${i + 1}')),
                  if (m['metadata']['eligible_for_activation'] == true)
                    FilledButton(
                        onPressed: _busy
                            ? null
                            : () => _run(() async {
                                  final ok = await showDialog<bool>(
                                      context: context,
                                      builder: (c) => AlertDialog(
                                              title: const Text(
                                                  'Изменить production-модель?'),
                                              content: const Text(
                                                  'Это отдельное переключение после просмотра метрик. Старые отчёты не пересчитываются.'),
                                              actions: [
                                                TextButton(
                                                    onPressed: () =>
                                                        Navigator.pop(c, false),
                                                    child:
                                                        const Text('Отмена')),
                                                TextButton(
                                                    onPressed: () =>
                                                        Navigator.pop(c, true),
                                                    child: const Text(
                                                        'Активировать'))
                                              ]));
                                  if (ok != true) return;
                                  final active = (_status!['active'] as List)
                                      .firstWhere((a) =>
                                          a['model_type'] == 'segmentation');
                                  await _request('/activate', body: {
                                    'model_id': m['id'],
                                    'expected_generation': active['generation']
                                  });
                                }),
                        child: const Text('Активировать после просмотра')),
                ]),
        ],
      ]));
}
