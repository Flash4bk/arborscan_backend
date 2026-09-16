import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'contour_drafts.dart';
import 'contour_editor_state.dart';
import 'corrections_service.dart';
import 'mask_drawing_page.dart';

class ContourWorkspacePage extends StatefulWidget {
  final String analysisId;
  final Uint8List? image;
  final Uint8List? aiMask;
  final Map<String, dynamic>? record;
  final String? draftId;
  final CorrectionsService? service;
  final ContourDrafts? drafts;
  const ContourWorkspacePage({super.key, required this.analysisId, this.image,
    this.aiMask, this.record, this.draftId, this.service, this.drafts});
  @override
  State<ContourWorkspacePage> createState() => _ContourWorkspacePageState();
}

class _ContourWorkspacePageState extends State<ContourWorkspacePage> with WidgetsBindingObserver {
  late final _service = widget.service ?? CorrectionsService();
  late final _drafts = widget.drafts ?? ContourDrafts();
  String? _token, _owner, _draftId, _parent, _savedId;
  Uint8List? _image, _mask;
  Map<String, dynamic>? _state;
  String _status = 'draft';
  String? _error;
  bool _busy = true, _invalid = false;
  bool? _supported;
  late final int _generation;

  @override
  void initState() {
    super.initState();
    _generation = CorrectionsService.authChanges.value;
    CorrectionsService.authChanges.addListener(_invalidate);
    WidgetsBinding.instance.addObserver(this);
    _load();
  }

  void _invalidate() {
    if (!mounted) return;
    setState(() { _invalid = true; _image = null; _mask = null; _state = null;
      _error = 'Сессия изменилась. Откройте экран из своего аккаунта.'; });
  }

  Future<void> _guard() async {
    await _service.checkSession(_token ?? '');
    if (_invalid || _generation != CorrectionsService.authChanges.value) {
      throw const CorrectionException('Сессия изменилась.');
    }
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) async {
    if (state == AppLifecycleState.resumed && _token != null) {
      try { await _guard(); } catch (_) { _invalidate(); }
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    CorrectionsService.authChanges.removeListener(_invalidate);
    super.dispose();
  }

  Future<void> _load() async {
    try {
      _token = await CorrectionsService.currentToken();
      _owner = await _service.owner(_token!);
      await _guard();
      final record = widget.record;
      _parent = record?['correction_id'] as String?;
      _savedId = _parent;
      _status = record?['review_status'] as String? ?? 'draft';
      _draftId = widget.draftId ?? '${widget.analysisId}_${_parent ?? 'root'}';
      final draft = await _drafts.load(_owner!, _draftId!);
      await _guard();
      _image = draft?['image'] as Uint8List? ?? widget.image;
      _state = (draft?['editor_state'] ?? record?['editor_state']) as Map<String, dynamic>?;
      if (_state != null) _state = ContourEditorState.fromJson(_state!).toJson();
      final encoded = draft?['mask_png_base64'] ?? record?['mask_png_base64'];
      _mask = encoded is String ? base64Decode(encoded) : null;
      if (draft != null) {
        _parent = draft['parent_id'] as String?;
        _savedId = draft['saved_id'] as String?;
        _status = draft['status'] as String? ?? 'draft';
        if (_savedId == record?['correction_id'] && record != null) {
          _status = record['review_status'] as String? ?? _status;
        }
      }
      if (_image == null) throw const CorrectionException('Оригинальное фото недоступно.');
      if (mounted) setState(() => _busy = false);
      _supported = await _service.workflowAvailable(_token!);
      await _guard();
    } catch (e) {
      if (!_invalid) _error = e.toString();
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _persist() async {
    await _guard();
    await _drafts.save(_owner!, _draftId!, {
      'analysis_id':widget.analysisId, 'parent_id':_parent, 'saved_id':_savedId,
      'status':_status, 'editor_state':_state,
      'mask_png_base64':_mask == null ? null : base64Encode(_mask!),
    }, _image!);
  }

  void _changed(Map<String, dynamic> state) {
    if (_invalid || jsonEncode(_state) == jsonEncode(state)) return;
    if (_savedId != null) { _parent = _savedId; _savedId = null; }
    _state = state; _mask = null; _status = 'draft';
    _persist().catchError((Object e) {
      if (mounted && !_invalid) setState(() => _error = 'Не удалось записать черновик: $e');
    });
  }

  Future<void> _edit() async {
    if (_busy || _invalid || _image == null) return;
    setState(() { _busy = true; _error = null; });
    try {
      await _persist();
      if (!mounted || _invalid) return;
      final result = await Navigator.of(context).push<Map<String, dynamic>>(MaterialPageRoute(
        builder: (_) => MaskDrawingPage(originalImageBase64:base64Encode(_image!),
          initialMaskBase64:_mask == null ? null : base64Encode(_mask!),
          aiMaskBase64:widget.aiMask == null ? null : base64Encode(widget.aiMask!),
          editorState:_state == null ? null : ContourEditorState.fromJson(_state!),
          sessionToken:_token, onDraftChanged:_changed)));
      await _guard();
      if (result != null) {
        final state = Map<String, dynamic>.from(result['editor_state'] as Map);
        _changed(state);
        _mask = base64Decode(result['mask_png_base64'] as String);
        await _persist();
      }
    } catch (e) {
      if (!_invalid) _error = e.toString();
    } finally { if (mounted) setState(() => _busy = false); }
  }

  Future<void> _save({bool legacy = false}) async {
    if (_busy || _invalid || _mask == null || _state == null) return;
    setState(() { _busy = true; _error = null; });
    try {
      await _persist();
      if (legacy) {
        await _service.save(token:_token!, analysisId:widget.analysisId, image:_image!, mask:_mask!);
        await _guard();
        _error = 'PNG сохранён старым API. Точки остались в локальном черновике; ревизии и модерация недоступны.';
      } else {
        if (!await _service.workflowAvailable(_token!)) {
          _supported = false;
          throw const CorrectionException('Сервер пока не поддерживает ревизии. Черновик сохранён.');
        }
        final saved = await _service.saveRevision(token:_token!, analysisId:widget.analysisId,
          image:_image!, mask:_mask!, editorState:_state!, parentId:_parent);
        await _guard();
        _savedId = saved['correction_id'] as String;
        _status = saved['review_status'] as String? ?? 'draft';
        await _persist();
      }
    } catch (e) { if (!_invalid) _error = '$e Черновик остаётся на экране.'; }
    finally { if (mounted) setState(() => _busy = false); }
  }

  Future<void> _submit() async {
    if (_busy || _savedId == null) return;
    setState(() { _busy = true; _error = null; });
    try {
      final record = await _service.submit(_token!, _savedId!);
      await _guard();
      _status = record['review_status'] as String;
      await _persist();
    } catch (e) { if (!_invalid) _error = e.toString(); }
    finally { if (mounted) setState(() => _busy = false); }
  }

  @override
  Widget build(BuildContext context) => Scaffold(
    appBar: AppBar(title:const Text('Работа с контуром')),
    body: ListView(padding:const EdgeInsets.all(16), children:[
      if (_busy) const LinearProgressIndicator(),
      if (_error != null) Text(_error!, style:TextStyle(color:Theme.of(context).colorScheme.error)),
      if (!_invalid && _image != null) ...[
        Image.memory(_image!, height:240, fit:BoxFit.contain),
        if (_mask != null) Image.memory(_mask!, height:160, fit:BoxFit.contain),
        Text('Статус: ${contourStatus(_status)}'),
        const Text('Черновик хранится на устройстве в вашем аккаунте. Измерения отчёта не изменяются.'),
        if (widget.record != null && _state == null)
          const Text('Старая запись содержит только PNG. Исходных точек нет. Новый контур рисуется явно по фото; PNG служит подсказкой.'),
        OutlinedButton(onPressed:_busy ? null : _edit,
          child:Text(_state == null ? 'Создать новый контур по фото' : 'Продолжить редактирование')),
        FilledButton(onPressed:_busy || _mask == null || _savedId != null || _state?.containsKey('points') != true
            ? null : () => _save(), child:Text(_savedId == null ? 'Сохранить контур' : 'Сохранено')),
        if (_supported == false) ...[
          const Text('Этот сервер не поддерживает состояние редактора и модерацию. Доступно сохранение только PNG.'),
          OutlinedButton(onPressed:_busy || _mask == null ? null : () => _save(legacy:true),
            child:const Text('Сохранить только PNG (старый API)')),
        ],
        if (_savedId != null && _supported == true && (_status == 'draft' || _status == 'pending_review'))
          FilledButton(onPressed:_busy ? null : _submit, child:const Text('Отправить на проверку')),
      ],
    ]),
  );
}

String contourStatus(String status) => const {'draft':'черновик', 'submitted':'отправлен на проверку',
  'pending_review':'ожидает проверки (старая запись)', 'accepted':'маска принята', 'rejected':'отклонён'}[status] ?? status;
