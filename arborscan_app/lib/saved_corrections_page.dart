import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'app_theme.dart';
import 'corrections_service.dart';

/// Fetches private server records anew on every opening, including cold starts.
/// PNGs are displayed as images; editor points are not stored by the backend.
class SavedCorrectionsPage extends StatefulWidget {
  final CorrectionsService? service;
  final String? correctionId;
  final String? sessionToken;
  const SavedCorrectionsPage({super.key, this.service, this.correctionId, this.sessionToken});
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
        final detail = await _service.detail(token, widget.correctionId!);
        if (!mounted || _invalidSession) return;
        setState(() { _image = detail.image; _mask = detail.mask; });
      } else {
        final page = await _service.list(token, refresh ? 0 : _next ?? 0);
        if (!mounted || _invalidSession) return;
        setState(() {
          if (refresh) _items.clear();
          for (final item in page['items'] as List) {
            final row = Map<String, dynamic>.from(item as Map);
            if (!_items.any((e) => e['correction_id'] == row['correction_id'])) _items.add(row);
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

  @override
  Widget build(BuildContext context) => Scaffold(
    appBar: AppBar(title: Text(widget.correctionId == null
        ? 'Сохранённые контуры' : 'Сохранённый контур'), actions: [
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
        if (!_busy && _error == null && _items.isEmpty) const Text('Сохранённых контуров пока нет.'),
        for (final item in _items) Card(child: ListTile(
          leading: const Icon(Icons.layers_outlined, color: AppTheme.primary),
          title: const Text('Сохранённый контур'),
          subtitle: Text('${item['created_at'] ?? ''}\n${item['correction_id']}'),
          onTap: () async {
            final token = await _session;
            if (!context.mounted || _invalidSession) return;
            await Navigator.of(context).push(MaterialPageRoute(builder: (_) =>
              SavedCorrectionsPage(service: _service,
                correctionId: item['correction_id'] as String, sessionToken: token)));
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
        const Text('Доступен просмотр PNG. Точки редактора не сохраняются и не восстанавливаются. Исходные измерения не изменены. Сохранение не подтверждает маску и не запускает обучение.'),
      ],
    ]),
  );

  Widget _imageError(BuildContext context, Object error, StackTrace? stack) =>
      const Text('Изображение повреждено. Попробуйте обновить.');
}
