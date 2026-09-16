import 'dart:typed_data';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'app_theme.dart';
import 'corrections_service.dart';
import 'contour_drafts.dart';
import 'contour_workspace_page.dart';

/// Fetches private server records anew on every opening, including cold starts.
/// Old PNG-only records remain viewable alongside versioned editor records.
class SavedCorrectionsPage extends StatefulWidget {
  final CorrectionsService? service;
  final String? correctionId;
  final String? sessionToken;
  final bool localDrafts;
  final bool adminQueue;
  final String? reviewOwner;
  const SavedCorrectionsPage({super.key, this.service, this.correctionId, this.sessionToken,
    this.localDrafts = false, this.adminQueue = false, this.reviewOwner});
  @override
  State<SavedCorrectionsPage> createState() => _SavedCorrectionsPageState();
}

class _SavedCorrectionsPageState extends State<SavedCorrectionsPage>
    with WidgetsBindingObserver {
  late final _service = widget.service ?? CorrectionsService();
  late final Future<String> _session;
  final List<Map<String, dynamic>> _items = [];
  int? _next = 0;
  bool _busy = false;
  bool _invalidSession = false;
  String? _error;
  Uint8List? _image;
  Uint8List? _mask;
  Map<String, dynamic>? _record;
  bool _overlay = false;
  final _reason = TextEditingController();

  @override
  void initState() {
    super.initState();
    _session = widget.sessionToken == null
        ? CorrectionsService.currentToken() : Future.value(widget.sessionToken!);
    CorrectionsService.authChanges.addListener(_invalidate);
    WidgetsBinding.instance.addObserver(this);
    _load();
  }

  void _invalidate() {
    if (!mounted) return;
    setState(() {
      _invalidSession = true;
      _items.clear(); _image = null; _mask = null;
      _record = null; _reason.clear();
      _error = 'Сессия изменилась. Откройте список заново из Истории.';
    });
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) async {
    if (state == AppLifecycleState.resumed &&
        await _session != await CorrectionsService.currentToken()) {
      _invalidate();
    }
  }

  @override
  void dispose() {
    _reason.dispose();
    WidgetsBinding.instance.removeObserver(this);
    CorrectionsService.authChanges.removeListener(_invalidate);
    super.dispose();
  }

  Future<void> _load({bool refresh = false}) async {
    if (_busy || _invalidSession) return;
    setState(() { _busy = true; _error = null; });
    try {
      final token = await _session;
      if (widget.correctionId != null) {
        final record = await _service.record(token, widget.correctionId!, ownerId:widget.reviewOwner);
        final image = base64Decode(record['original_image_base64'] as String);
        final mask = base64Decode(record['mask_png_base64'] as String);
        if (!mounted || _invalidSession) return;
        setState(() { _image = image; _mask = mask; _record = record; });
      } else if (widget.localDrafts) {
        final rows = await ContourDrafts().list(await _service.owner(token));
        await _service.checkSession(token);
        if (!mounted || _invalidSession) return;
        setState(() { _items.clear(); _items.addAll(rows); _next = null; });
      } else {
        final page = widget.adminQueue ? await _service.queue(token, refresh ? 0 : _next ?? 0)
          : await _service.list(token, refresh ? 0 : _next ?? 0);
        if (!mounted || _invalidSession) return;
        setState(() {
          if (refresh) _items.clear();
          for (final item in page['items'] as List) {
            final row = Map<String, dynamic>.from(item as Map);
            if (!_items.any((e) => e['correction_id'] == row['correction_id'] && e['owner_id'] == row['owner_id'])) _items.add(row);
          }
          _next = page['next_offset'] as int?;
        });
      }
    } catch (e) {
      if (mounted && !_invalidSession) {
        if (await _session != await CorrectionsService.currentToken()) {
          _invalidate();
        } else if (mounted) {
          setState(() { _error = e.toString(); _image = null; _mask = null; });
        }
      }
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _decide(String decision) async {
    if (_busy || _invalidSession) return;
    if (decision == 'rejected' && _reason.text.trim().isEmpty) {
      setState(() => _error = 'Укажите причину отклонения.'); return;
    }
    setState(() { _busy = true; _error = null; });
    try {
      await _service.decide(await _session, widget.reviewOwner!, widget.correctionId!, decision, _reason.text.trim());
      if (!mounted || _invalidSession) return;
      setState(() => _busy = false);
      await _load(refresh:true);
    } catch (e) { if (mounted && !_invalidSession) setState(() => _error = '$e'); }
    finally { if (mounted) setState(() => _busy = false); }
  }

  @override
  Widget build(BuildContext context) => Scaffold(
    appBar: AppBar(title: Text(widget.correctionId == null
        ? widget.localDrafts ? 'Черновики контуров' : widget.adminQueue ? 'Проверка контуров' : 'Сохранённые контуры'
        : 'Сохранённый контур'), actions: [
      IconButton(tooltip: 'Обновить', icon: const Icon(Icons.refresh),
        onPressed: _busy || _invalidSession ? null : () => _load(refresh: true)),
    ]),
    body: ListView(padding: const EdgeInsets.all(16), children: [
      if (_busy) const LinearProgressIndicator(),
      if (_error != null) ...[
        Text(_error!, style: const TextStyle(color: AppTheme.danger)),
        if (!_invalidSession) OutlinedButton(onPressed: _busy ? null : _load,
          child: const Text('Повторить')),
      ],
      if (widget.correctionId == null && !_invalidSession) ...[
        if (!widget.localDrafts && !widget.adminQueue)
          OutlinedButton(onPressed:() => Navigator.of(context).push(MaterialPageRoute(
            builder:(_) => SavedCorrectionsPage(service:_service, localDrafts:true))), child:const Text('Черновики на устройстве')),
        if (!_busy && _error == null && _items.isEmpty) const Text('Сохранённых контуров пока нет.'),
        for (final item in _items) Card(child: ListTile(
          leading: const Icon(Icons.layers_outlined, color: AppTheme.primary),
          title: Text(widget.localDrafts ? 'Черновик контура' : 'Сохранённый контур'),
          subtitle: Text('${item['created_at'] ?? item['analysis_id'] ?? ''}\n${item['correction_id'] ?? item['draft_id']}'),
          onTap: () async {
            final token = await _session;
            if (!context.mounted || _invalidSession) return;
            await Navigator.of(context).push(MaterialPageRoute(builder: (_) => widget.localDrafts
              ? ContourWorkspacePage(analysisId:item['analysis_id'] as String, draftId:item['draft_id'] as String, service:_service)
              : SavedCorrectionsPage(service: _service, reviewOwner:widget.adminQueue ? item['owner_id'] as String : null,
                correctionId: item['correction_id'] as String, sessionToken: token)));
            if (mounted && !_invalidSession) await _load(refresh:true);
          },
        )),
        if (_next != null && _items.isNotEmpty && _error == null)
          OutlinedButton(onPressed: _busy ? null : _load, child: const Text('Загрузить ещё')),
      ],
      if (_image != null && _mask != null) ...[
        const Text('Оригинальное фото'),
        Image.memory(_image!, fit: BoxFit.contain, errorBuilder: _imageError),
        const SizedBox(height: 16),
        const Text('Сохранённая PNG-маска'),
        Image.memory(_mask!, fit: BoxFit.contain, errorBuilder: _imageError),
        const SizedBox(height: 12),
        Text('Статус: ${contourStatus(_record?['review_status'] as String? ?? 'pending_review')}'),
        if (widget.reviewOwner == null && _record?['next_revision_id'] is String)
          OutlinedButton(onPressed:() => Navigator.of(context).push(MaterialPageRoute(builder:(_) =>
            SavedCorrectionsPage(service:_service, correctionId:_record!['next_revision_id'] as String))),
            child:const Text('Открыть следующую ревизию')),
        for (final event in (_record?['decisions'] as List? ?? []))
          Text('${event['action']} · ${event['at']} · ${event['actor_id']}\n${event['reason'] ?? ''}'),
        const Text('Принятие касается только маски: высота, DBH и механическая оценка не подтверждаются. Обучение не запускается.'),
        SwitchListTile(title:const Text('Наложение маски на оригинал'), value:_overlay,
          onChanged:(v) => setState(() => _overlay = v)),
        if (_overlay) Stack(children:[
          Image.memory(_image!, fit:BoxFit.contain),
          Positioned.fill(child:Opacity(opacity:0.4, child:Image.memory(_mask!, fit:BoxFit.fill))),
        ]),
        if (widget.reviewOwner == null)
          OutlinedButton(onPressed:_busy ? null : () async {
            await Navigator.of(context).push(MaterialPageRoute(builder:(_) => ContourWorkspacePage(
              analysisId:_record!['analysis_id'] as String, image:_image, record:{..._record!, 'correction_id':widget.correctionId}, service:_service)));
            if (mounted && !_invalidSession) await _load(refresh:true);
          }, child:Text(_record?['editor_state'] == null ? 'Создать новый контур по фото' : 'Редактировать ревизию')),
        if (widget.reviewOwner == null && _record?['editor_state'] == null)
          const Text('Старая PNG-запись: исходные точки не сохранены. Можно явно нарисовать новый контур по фото.'),
        if (widget.reviewOwner != null && _record?['review_status'] == 'submitted') ...[
          TextField(controller:_reason, maxLength:2000, decoration:const InputDecoration(labelText:'Причина / комментарий')),
          FilledButton(onPressed:_busy ? null : () => _decide('accepted'), child:const Text('Принять маску')),
          OutlinedButton(onPressed:_busy ? null : () => _decide('rejected'), child:const Text('Отклонить с причиной')),
        ],
      ],
    ]),
  );

  Widget _imageError(BuildContext context, Object error, StackTrace? stack) =>
      const Text('Изображение повреждено. Попробуйте обновить.');
}
