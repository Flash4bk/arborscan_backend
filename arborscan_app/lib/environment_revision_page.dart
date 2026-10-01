import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'corrections_service.dart';
import 'report_history_service.dart';
import 'survey_environment.dart';
import 'survey_environment_ui.dart';

/// Location updates deliberately copy the chosen immutable report version.
class EnvironmentRevisionPage extends StatefulWidget {
  final Map<String, dynamic> snapshot, record;
  final Uint8List photo;
  final String? localId;
  const EnvironmentRevisionPage(
      {super.key,
      required this.snapshot,
      required this.record,
      required this.photo,
      this.localId});
  @override
  State<EnvironmentRevisionPage> createState() =>
      _EnvironmentRevisionPageState();
}

class _EnvironmentRevisionPageState extends State<EnvironmentRevisionPage> {
  final _service = ReportHistoryService();
  late final _environment =
      SurveyEnvironmentController(surveyMap(widget.snapshot['environment']));
  String? _token, _localId, _message;
  bool _busy = true, _invalid = false, _changed = false;
  Future<void> _pending = Future.value();
  String? get _parentId =>
      widget.localId != null && widget.record['saved'] != true
          ? widget.record['parent_id']
          : widget.record['version_id'];
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
        _message = 'Аккаунт изменился.';
      });
    }
  }

  @override
  void dispose() {
    CorrectionsService.authChanges.removeListener(_invalidate);
    _environment.dispose();
    super.dispose();
  }

  Future<void> _load() async {
    try {
      _token = await CorrectionsService.currentToken();
      final owner = await _service.auth.owner(_token!);
      _localId = widget.localId ?? 'environment-${widget.record['version_id']}';
      final saved = await _service.journal.load(owner, _localId!);
      await _service.auth.checkSession(_token!);
      if (_invalid || !mounted) return;
      if (saved != null && saved['saved'] != true) {
        _environment.replace(surveyMap(saved['snapshot']['environment']));
        _changed = true;
        _message = 'Восстановлен локальный черновик новой версии.';
      }
      // A new edit of an old version must retain its original parent and conflict,
      // not silently build on the last locally saved child of that old version.
      if (saved?['saved'] == true) {
        _localId = 'environment-${widget.record['version_id']}-${reportUuid()}';
      }
      _environment.addListener(_onChange);
    } catch (e) {
      _message = '$e';
    }
    if (mounted) setState(() => _busy = false);
  }

  Map<String, dynamic> _payload(Map<String, dynamic> environment) => {
        for (final key in [
          'version',
          'kind',
          'report',
          'reference',
          'ar',
          'captured_at',
          'correction_id'
        ])
          if (widget.snapshot.containsKey(key)) key: widget.snapshot[key],
        'environment': environment.isEmpty ? null : environment,
        'change_source': 'environment_edit',
      };
  void _onChange() {
    if (_invalid || _token == null || _localId == null || _environment.busy) {
      return;
    }
    final value = _environment.snapshot;
    _changed = true;
    _pending = _pending.then((_) async {
      if (_invalid) return;
      try {
        await _service.stage(
            token: _token!,
            localId: _localId!,
            analysisId: widget.record['analysis_id'],
            parentId: _parentId,
            snapshot: _payload(value),
            image: widget.photo);
        if (mounted && !_invalid) {
          setState(
              () => _message = 'Черновик новой версии сохранён на устройстве.');
        }
      } catch (e) {
        if (mounted && !_invalid) setState(() => _message = '$e');
      }
    });
  }

  Future<void> _upload() async {
    if (_busy || _invalid || !_changed) return;
    setState(() => _busy = true);
    try {
      await _pending;
      await _service.stage(
          token: _token!,
          localId: _localId!,
          analysisId: widget.record['analysis_id'],
          parentId: _parentId,
          snapshot: _payload(_environment.snapshot),
          image: widget.photo);
      final saved = await _service.upload(_token!, _localId!);
      if (mounted && !_invalid) {
        Navigator.pop(context, saved['record']['version_id']);
      }
    } catch (e) {
      if (mounted) setState(() => _message = '$e');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) => Scaffold(
      appBar: AppBar(title: const Text('Место и условия · новая версия')),
      body: ListView(padding: const EdgeInsets.all(16), children: [
        if (_busy) const LinearProgressIndicator(),
        if (_message != null) Text(_message!),
        if (!_invalid) ...[
          const Text(
              'Изменение места или обновление условий создаёт новую версию. Размеры, фото и прежний отчёт сохраняются.'),
          const SizedBox(height: 12),
          ClipRRect(
              borderRadius: BorderRadius.circular(20),
              child:
                  Image.memory(widget.photo, height: 180, fit: BoxFit.contain)),
          SurveyEnvironmentEditor(
              controller: _environment,
              original: widget.photo,
              enabled: !_busy),
          ListenableBuilder(
              listenable: _environment,
              builder: (context, _) => FilledButton(
                  onPressed:
                      _busy || _environment.busy || !_changed ? null : _upload,
                  child: const Text('Сохранить новую версию в аккаунте'))),
          const Text(
              'При отсутствии сети черновик доступен в истории на этом устройстве. После сохранения в аккаунте версия доступна на других устройствах.'),
        ],
      ]));
}
