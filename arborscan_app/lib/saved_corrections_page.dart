import 'dart:typed_data';
import 'dart:convert';
import 'dart:async';
import 'dart:collection';
import 'package:flutter/material.dart';
import 'app_theme.dart';
import 'binary_mask_overlay.dart';
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
  const SavedCorrectionsPage(
      {super.key,
      this.service,
      this.correctionId,
      this.sessionToken,
      this.localDrafts = false,
      this.adminQueue = false,
      this.reviewOwner});
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
  String _view = 'photo';
  final _previews = <String, Uint8List>{};
  final _previewWaiters = Queue<Completer<void>>();
  int _activePreviews = 0, _previewGeneration = 0;
  final _reason = TextEditingController();

  @override
  void initState() {
    super.initState();
    _session = widget.sessionToken == null
        ? CorrectionsService.currentToken()
        : Future.value(widget.sessionToken!);
    CorrectionsService.authChanges.addListener(_invalidate);
    WidgetsBinding.instance.addObserver(this);
    _load();
  }

  void _invalidate() {
    if (!mounted) return;
    setState(() {
      _invalidSession = true;
      _items.clear();
      _previews.clear();
      _image = null;
      _mask = null;
      _record = null;
      _reason.clear();
      _error = 'Сессия изменилась. Откройте список заново из Истории.';
    });
    _wakePreviewWaiters();
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
    _invalidSession = true;
    _previews.clear();
    _wakePreviewWaiters();
    _reason.dispose();
    WidgetsBinding.instance.removeObserver(this);
    CorrectionsService.authChanges.removeListener(_invalidate);
    super.dispose();
  }

  Future<void> _load({bool refresh = false}) async {
    if (_busy || _invalidSession) return;
    setState(() {
      _busy = true;
      _error = null;
      if (refresh) {
        _previewGeneration++;
        _previews.clear();
      }
    });
    try {
      final token = await _session;
      if (widget.correctionId != null) {
        final record = await _service.record(token, widget.correctionId!,
            ownerId: widget.reviewOwner);
        final image = base64Decode(record['original_image_base64'] as String);
        final mask = base64Decode(record['mask_png_base64'] as String);
        if (!mounted || _invalidSession) return;
        setState(() {
          _image = image;
          _mask = mask;
          _record = record;
        });
      } else if (widget.localDrafts) {
        final rows = await ContourDrafts().list(await _service.owner(token));
        await _service.checkSession(token);
        if (!mounted || _invalidSession) return;
        setState(() {
          _items.clear();
          _items.addAll(rows);
          _next = null;
        });
      } else {
        final page = widget.adminQueue
            ? await _service.queue(token, refresh ? 0 : _next ?? 0)
            : await _service.list(token, refresh ? 0 : _next ?? 0);
        if (!mounted || _invalidSession) return;
        setState(() {
          if (refresh) _items.clear();
          for (final item in page['items'] as List) {
            final row = Map<String, dynamic>.from(item as Map);
            if (!_items.any((e) =>
                e['correction_id'] == row['correction_id'] &&
                e['owner_id'] == row['owner_id'])) {
              _items.add(row);
            }
          }
          _next = page['next_offset'] as int?;
        });
      }
    } catch (e) {
      if (mounted && !_invalidSession) {
        if (await _session != await CorrectionsService.currentToken()) {
          _invalidate();
        } else if (mounted) {
          setState(() {
            _error = e.toString();
            _image = null;
            _mask = null;
          });
        }
      }
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _decide(String decision) async {
    if (_busy || _invalidSession) return;
    if (decision == 'rejected' && _reason.text.trim().isEmpty) {
      setState(() => _error = 'Укажите причину отклонения.');
      return;
    }
    setState(() {
      _busy = true;
      _error = null;
    });
    try {
      await _service.decide(await _session, widget.reviewOwner!,
          widget.correctionId!, decision, _reason.text.trim());
      if (!mounted || _invalidSession) return;
      setState(() => _busy = false);
      await _load(refresh: true);
    } catch (e) {
      if (mounted && !_invalidSession) setState(() => _error = '$e');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  void _wakePreviewWaiters() {
    while (_previewWaiters.isNotEmpty) {
      _previewWaiters.removeFirst().complete();
    }
  }

  Future<bool> _previewSlot(bool Function() visible) async {
    while (_activePreviews >= 2) {
      final waiter = Completer<void>();
      _previewWaiters.add(waiter);
      await waiter.future;
      if (!mounted || _invalidSession || !visible()) return false;
    }
    if (!mounted || _invalidSession || !visible()) return false;
    _activePreviews++;
    return true;
  }

  Future<Uint8List?> _preview(
      Map<String, dynamic> item, bool Function() visible) async {
    final key = '${item['owner_id']}/${item['correction_id']}';
    if (_invalidSession || widget.localDrafts) return null;
    final cached = _previews.remove(key);
    if (cached != null) {
      _previews[key] = cached;
      return cached;
    }
    if (!await _previewSlot(visible)) return null;
    try {
      final bytes = await _service.preview(
          await _session, item['correction_id'] as String,
          ownerId: widget.adminQueue ? item['owner_id'] as String : null);
      if (_invalidSession || !mounted) return null;
      if (bytes != null) {
        _previews[key] = bytes;
        while (_previews.length > 8) {
          _previews.remove(_previews.keys.first);
        }
      }
      return bytes;
    } catch (_) {
      // A preview is optional. The original still opens through its normal
      // guarded detail action; do not fetch that full resource for a list icon.
      return null;
    } finally {
      _activePreviews--;
      _wakePreviewWaiters();
    }
  }

  String _listStatus(Map<String, dynamic> item) {
    final status = item['review_status'] ?? item['status'];
    if (status is String && status.isNotEmpty) return contourStatus(status);
    return widget.localDrafts
        ? contourStatus('draft')
        : 'Статус доступен после открытия';
  }

  String _displayDate(dynamic value) {
    final raw = value?.toString();
    final date = raw == null ? null : DateTime.tryParse(raw)?.toLocal();
    if (date == null) return raw ?? 'Дата не указана';
    String two(int value) => value.toString().padLeft(2, '0');
    return '${two(date.day)}.${two(date.month)}.${date.year} '
        '${two(date.hour)}:${two(date.minute)}';
  }

  @override
  Widget build(BuildContext context) => Scaffold(
        appBar: AppBar(
            title: Text(widget.correctionId == null
                ? widget.localDrafts
                    ? 'Черновики контуров'
                    : widget.adminQueue
                        ? 'Проверка контуров'
                        : 'Сохранённые контуры'
                : 'Сохранённый контур'),
            actions: [
              IconButton(
                  tooltip: 'Обновить',
                  icon: const Icon(Icons.refresh),
                  onPressed: _busy || _invalidSession
                      ? null
                      : () => _load(refresh: true)),
            ]),
        body: AppContentList(padding: const EdgeInsets.all(16), children: [
          if (_busy) const LinearProgressIndicator(),
          if (_error != null) ...[
            Text(_error!,
                style:
                    TextStyle(color: AppTheme.color(context, AppTheme.danger))),
            if (!_invalidSession)
              OutlinedButton(
                  onPressed: _busy ? null : _load,
                  child: const Text('Повторить')),
          ],
          if (widget.correctionId == null && !_invalidSession) ...[
            if (!widget.localDrafts && !widget.adminQueue)
              OutlinedButton(
                  onPressed: () => Navigator.of(context).push(MaterialPageRoute(
                      builder: (_) => SavedCorrectionsPage(
                          service: _service, localDrafts: true))),
                  child: const Text('Черновики на устройстве')),
            if (!_busy && _error == null && _items.isEmpty)
              Text(_next == null
                  ? 'Сохранённых контуров пока нет.'
                  : 'На этой странице нет записей. Загрузите следующую.'),
            if (_items.isNotEmpty)
              Text('Загружено записей: ${_items.length}',
                  style: Theme.of(context).textTheme.labelLarge),
            for (final item in _items)
              Card(
                  child: ListTile(
                leading: widget.localDrafts
                    ? const Icon(Icons.edit_note)
                    : ClipRRect(
                        borderRadius: BorderRadius.circular(10),
                        child: SizedBox(
                            width: 56,
                            height: 56,
                            child: _CorrectionPreview(
                                key: ValueKey(
                                    '${widget.adminQueue}/${item['owner_id']}/${item['correction_id']}/$_previewGeneration'),
                                load: (visible) => _preview(item, visible)))),
                trailing: const Icon(Icons.chevron_right),
                title: Text(widget.localDrafts
                    ? 'Черновик контура'
                    : 'Сохранённый контур'),
                subtitle: Text(
                    '${_displayDate(item['created_at'] ?? (widget.localDrafts ? 'На этом устройстве' : null))}\n${_listStatus(item)}'),
                onTap: () async {
                  final token = await _session;
                  if (!context.mounted || _invalidSession) return;
                  await Navigator.of(context).push(MaterialPageRoute(
                      builder: (_) => widget.localDrafts
                          ? ContourWorkspacePage(
                              analysisId: item['analysis_id'] as String,
                              draftId: item['draft_id'] as String,
                              service: _service)
                          : SavedCorrectionsPage(
                              service: _service,
                              reviewOwner: widget.adminQueue
                                  ? item['owner_id'] as String
                                  : null,
                              correctionId: item['correction_id'] as String,
                              sessionToken: token)));
                  if (mounted && !_invalidSession) await _load(refresh: true);
                },
              )),
            if (_next != null && _error == null)
              OutlinedButton(
                  onPressed: _busy ? null : _load,
                  child: const Text('Загрузить ещё')),
          ],
          if (_image != null && _mask != null) ...[
            ClipRRect(
                borderRadius: BorderRadius.circular(20),
                child: Container(
                  color: AppTheme.color(context, AppTheme.surface2),
                  height: 320,
                  child: _view == 'mask'
                      ? Image.memory(_mask!,
                          fit: BoxFit.contain, errorBuilder: _imageError)
                      : Stack(
                          fit: StackFit.expand,
                          alignment: Alignment.center,
                          children: [
                              Image.memory(_image!,
                                  fit: BoxFit.contain,
                                  errorBuilder: _imageError),
                              if (_view == 'overlay')
                                Positioned.fill(
                                    child: BinaryMaskOverlay(
                                        bytes: _mask!,
                                        color: Colors.blue
                                            .withValues(alpha: .35))),
                            ]),
                )),
            Wrap(spacing: 8, runSpacing: 4, children: [
              for (final view in const {
                'photo': 'Оригинал',
                'mask': 'PNG-маска',
                'overlay': 'Наложение'
              }.entries)
                ChoiceChip(
                    label: Text(view.value),
                    selected: _view == view.key,
                    onSelected: (_) => setState(() => _view = view.key)),
            ]),
            Text(
                'Статус: ${contourStatus(_record?['review_status'] as String? ?? 'pending_review')}',
                style: Theme.of(context).textTheme.titleMedium),
            ExpansionTile(title: const Text('Сведения о ревизии'), children: [
              SelectableText(
                  'Ревизия: ${widget.correctionId}\nАнализ: ${_record?['analysis_id']}'),
              if (_record?['parent_id'] != null)
                SelectableText(
                    'Родительская ревизия: ${_record!['parent_id']}'),
              SelectableText(
                  'Код статуса: ${_record?['review_status'] ?? 'pending_review'}'),
              if (_record?['created_at'] != null)
                SelectableText(
                    'Создано (исходное время): ${_record!['created_at']}'),
            ]),
            if (widget.reviewOwner == null &&
                _record?['next_revision_id'] is String)
              OutlinedButton(
                  onPressed: () => Navigator.of(context).push(MaterialPageRoute(
                      builder: (_) => SavedCorrectionsPage(
                          service: _service,
                          correctionId:
                              _record!['next_revision_id'] as String))),
                  child: const Text('Открыть следующую ревизию')),
            for (final event in (_record?['decisions'] as List? ?? []))
              ExpansionTile(
                  title: Text(
                      '${contourStatus(event['action'] as String? ?? '')} · ${_displayDate(event['at'])}'),
                  subtitle: Text(event['reason']?.toString() ?? ''),
                  children: [
                    if (event['at'] != null)
                      SelectableText(
                          'Время решения (исходное): ${event['at']}'),
                    SelectableText('Автор решения: ${event['actor_id']}')
                  ]),
            if (_record?['feedback_proposal'] is Map)
              _FeedbackProposal(
                  proposal:
                      Map<String, dynamic>.from(_record!['feedback_proposal'])),
            const Text(
                'Принятие касается только маски: высота, DBH и механическая оценка не подтверждаются. Обучение не запускается.'),
            if (widget.reviewOwner == null)
              OutlinedButton(
                  onPressed: _busy
                      ? null
                      : () async {
                          await Navigator.of(context).push(MaterialPageRoute(
                              builder: (_) => ContourWorkspacePage(
                                  analysisId: _record!['analysis_id'] as String,
                                  image: _image,
                                  record: {
                                    ..._record!,
                                    'correction_id': widget.correctionId
                                  },
                                  service: _service)));
                          if (mounted && !_invalidSession) {
                            await _load(refresh: true);
                          }
                        },
                  child: Text(_record?['editor_state'] == null
                      ? 'Создать новый контур по фото'
                      : 'Редактировать ревизию')),
            if (widget.reviewOwner == null && _record?['editor_state'] == null)
              const Text(
                  'Старая PNG-запись: исходные точки не сохранены. Можно явно нарисовать новый контур по фото.'),
            if (widget.reviewOwner != null &&
                _record?['review_status'] == 'submitted') ...[
              TextField(
                  controller: _reason,
                  minLines: 2,
                  maxLines: 5,
                  maxLength: 2000,
                  decoration: const InputDecoration(
                      labelText: 'Причина / комментарий')),
              FilledButton(
                  onPressed: _busy ? null : () => _decide('accepted'),
                  child: const Text('Принять маску')),
              OutlinedButton(
                  onPressed: _busy ? null : () => _decide('rejected'),
                  child: const Text('Отклонить с причиной')),
            ],
          ],
        ]),
      );

  Widget _imageError(BuildContext context, Object error, StackTrace? stack) =>
      const Text('Изображение повреждено. Попробуйте обновить.');
}

/// Loading starts when this element enters the lazy list's viewport/cache area,
/// rather than while the parent constructs every loaded Card widget.
class _CorrectionPreview extends StatefulWidget {
  final Future<Uint8List?> Function(bool Function() visible) load;
  const _CorrectionPreview({super.key, required this.load});
  @override
  State<_CorrectionPreview> createState() => _CorrectionPreviewState();
}

class _CorrectionPreviewState extends State<_CorrectionPreview> {
  late final Future<Uint8List?> _image;
  @override
  void initState() {
    super.initState();
    _image = widget.load(() => mounted);
  }

  @override
  Widget build(BuildContext context) => FutureBuilder<Uint8List?>(
      future: _image,
      builder: (_, snapshot) => snapshot.hasData
          ? Image.memory(snapshot.data!,
              cacheWidth: 140,
              fit: BoxFit.cover,
              errorBuilder: (_, __, ___) =>
                  const Icon(Icons.broken_image_outlined))
          : ColoredBox(
              color: AppTheme.color(context, AppTheme.surface2),
              child: Icon(Icons.layers_outlined,
                  color: AppTheme.color(context, AppTheme.primary))));
}

class _FeedbackProposal extends StatelessWidget {
  final Map<String, dynamic> proposal;
  const _FeedbackProposal({required this.proposal});
  @override
  Widget build(BuildContext context) {
    final corrections = proposal['corrections'] is Map
        ? proposal['corrections'] as Map
        : const {};
    final quality =
        proposal['quality'] is Map ? proposal['quality'] as Map : const {};
    return ExpansionTile(
      title: const Text('Предложения автора прежней обратной связи'),
      subtitle: const Text('Не проверены; измерения отчёта не изменены'),
      childrenPadding: const EdgeInsets.all(12),
      children: [
        const Text(
            'Это предложения пользователя, а не результаты проверки или новый расчёт. Принятие PNG подтверждает только маску.'),
        if (corrections['species'] is String)
          Text('Предложенный вид: ${corrections['species']}'),
        for (final field in const {
          'height_m': 'Предложенная высота (м)',
          'crown_width_m': 'Предложенная ширина кроны (м)',
          'trunk_diameter_m': 'Предложенный диаметр сечения (м)',
          'scale_px_to_m':
              'Предложенный масштаб (м/пиксель; система пикселей не подтверждена)',
        }.entries)
          if (corrections[field.key] is num &&
              (corrections[field.key] as num).isFinite)
            Text('${field.value}: ${corrections[field.key]}'),
        for (final field in const {
          'tree_ok': 'Выделение дерева',
          'stick_ok': 'Выделение эталона',
          'params_ok': 'Размеры',
          'species_ok': 'Вид',
        }.entries)
          if (quality[field.key] is bool)
            Text(
                'По мнению автора: ${field.value} ${quality[field.key] == true ? 'верны' : 'требуют исправления'}'),
        Text(proposal['training_consent'] == true
            ? 'Автор разрешил использовать пример для обучения. Допуск к набору и запуск обучения выполняются отдельно.'
            : 'Разрешение автора на использование в обучении не дано.'),
      ],
    );
  }
}
