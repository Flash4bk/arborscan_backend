import 'dart:typed_data';

import 'package:flutter/material.dart';

import 'admin_service.dart';

class TrainingDatasetPage extends StatefulWidget {
  const TrainingDatasetPage({super.key});

  @override
  State<TrainingDatasetPage> createState() => _TrainingDatasetPageState();
}

class _TrainingDatasetPageState extends State<TrainingDatasetPage> {
  final _service = AdminService();
  late Future<List<VerifiedItem>> _future;
  String? _error;

  @override
  void initState() {
    super.initState();
    _future = _service.getVerifiedList();
  }

  void _reload() {
    setState(() {
      _error = null;
      _future = _service.getVerifiedList();
    });
  }

  Future<void> _toggleTraining(VerifiedItem item, bool include) async {
    try {
      await _service.setTrainingFlag(item.id, includeInTraining: include);
      setState(() {
        item.includeInTraining = include;
      });
    } catch (e) {
      setState(() => _error = e.toString());
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Датасет для обучения'),
        actions: [
          IconButton(
            tooltip: 'Обновить',
            onPressed: _reload,
            icon: const Icon(Icons.refresh_rounded),
          )
        ],
      ),
      body: FutureBuilder<List<VerifiedItem>>(
        future: _future,
        builder: (context, snap) {
          if (snap.connectionState != ConnectionState.done) {
            return const Center(child: CircularProgressIndicator());
          }
          if (snap.hasError) {
            return _ErrorView(
              title: 'Не удалось загрузить датасет',
              message: snap.error.toString(),
              onRetry: _reload,
            );
          }

          final items = snap.data ?? const <VerifiedItem>[];
          if (items.isEmpty) {
            return _ErrorView(
              title: 'Пока пусто',
              message: 'Нет подтверждённых изображений для обучения.',
              onRetry: _reload,
            );
          }

          return Column(
            children: [
              if (_error != null)
                _BannerError(
                  message: _error!,
                  onClose: () => setState(() => _error = null),
                ),
              Expanded(
                child: ListView.separated(
                  padding: const EdgeInsets.all(12),
                  itemCount: items.length,
                  separatorBuilder: (_, __) => const SizedBox(height: 10),
                  itemBuilder: (context, i) {
                    final it = items[i];
                    return _VerifiedCard(
                      item: it,
                      onToggle: (v) => _toggleTraining(it, v),
                      onOpen: () async {
                        await Navigator.of(context).push(
                          MaterialPageRoute(
                            builder: (_) => _VerifiedDetailsPage(item: it),
                          ),
                        );
                      },
                    );
                  },
                ),
              ),
            ],
          );
        },
      ),
    );
  }
}

class _VerifiedCard extends StatelessWidget {
  final VerifiedItem item;
  final VoidCallback onOpen;
  final ValueChanged<bool> onToggle;

  const _VerifiedCard({
    required this.item,
    required this.onOpen,
    required this.onToggle,
  });

  String _fmtDate(String? iso) {
    if (iso == null || iso.isEmpty) return '—';
    try {
      final dt = DateTime.parse(iso).toLocal();
      String two(int n) => n.toString().padLeft(2, '0');
      return '${two(dt.day)}.${two(dt.month)}.${dt.year}  ${two(dt.hour)}:${two(dt.minute)}';
    } catch (_) {
      return iso;
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final idShort = item.id.length > 8 ? item.id.substring(0, 8) : item.id;

    return Material(
      color: theme.colorScheme.surface,
      borderRadius: BorderRadius.circular(16),
      child: InkWell(
        onTap: onOpen,
        borderRadius: BorderRadius.circular(16),
        child: Padding(
          padding: const EdgeInsets.all(14),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Expanded(
                    child: Text(
                      'ID: $idShort…',
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: theme.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700),
                    ),
                  ),
                  Switch.adaptive(
                    value: item.includeInTraining,
                    onChanged: onToggle,
                  ),
                ],
              ),
              const SizedBox(height: 6),
              Text(
                'Добавлено: ${_fmtDate(item.verifiedAt)}',
                style: theme.textTheme.bodySmall?.copyWith(
                  color: theme.colorScheme.onSurfaceVariant,
                ),
              ),
              const SizedBox(height: 4),
              Text(
                item.includeInTraining ? 'Будет использовано в обучении' : 'Исключено из обучения',
                style: theme.textTheme.bodyMedium?.copyWith(
                  color: item.includeInTraining
                      ? theme.colorScheme.primary
                      : theme.colorScheme.onSurfaceVariant,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _VerifiedDetailsPage extends StatefulWidget {
  final VerifiedItem item;
  const _VerifiedDetailsPage({required this.item});

  @override
  State<_VerifiedDetailsPage> createState() => _VerifiedDetailsPageState();
}

class _VerifiedDetailsPageState extends State<_VerifiedDetailsPage> {
  final _service = AdminService();
  VerifiedAnalysis? _details;
  String? _error;
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final d = await _service.getVerifiedAnalysis(widget.item.id);
      setState(() => _details = d);
    } catch (e) {
      setState(() => _error = e.toString());
    } finally {
      setState(() => _loading = false);
    }
  }

  String _fmtDate(String? iso) {
    if (iso == null || iso.isEmpty) return '—';
    try {
      final dt = DateTime.parse(iso).toLocal();
      String two(int n) => n.toString().padLeft(2, '0');
      return '${two(dt.day)}.${two(dt.month)}.${dt.year}  ${two(dt.hour)}:${two(dt.minute)}';
    } catch (_) {
      return iso;
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Детали'),
        actions: [
          IconButton(
            tooltip: 'Обновить',
            onPressed: _load,
            icon: const Icon(Icons.refresh_rounded),
          ),
        ],
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : (_error != null)
              ? _ErrorView(
                  title: 'Не удалось загрузить детали',
                  message: _error!,
                  onRetry: _load,
                )
              : (_details == null)
                  ? _ErrorView(
                      title: 'Нет данных',
                      message: 'Сервер не вернул детали.',
                      onRetry: _load,
                    )
                  : ListView(
                      padding: const EdgeInsets.all(16),
                      children: [
                        _InfoRow(label: 'ID', value: widget.item.id),
                        _InfoRow(label: 'Добавлено', value: _fmtDate(widget.item.verifiedAt)),
                        const SizedBox(height: 14),
                        Text(
                          'Просмотр разметки',
                          style: theme.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w800),
                        ),
                        const SizedBox(height: 10),
                        _MaskPreviewCard(
                          inputImage: _details!.inputImage,
                          userMaskPng: _details!.userMaskPng,
                          fallbackAnnotated: _details!.annotatedImage,
                        ),
                        const SizedBox(height: 14),
                        Text(
                          'Подсказка',
                          style: theme.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w800),
                        ),
                        const SizedBox(height: 6),
                        Text(
                          _details!.userMaskPng != null
                              ? 'Показывается ваша итоговая маска (как вы обвели). Нажмите на картинку, чтобы открыть на весь экран и приблизить.'
                              : 'Для этого образца нет вашей сохранённой маски, поэтому показывается разметка от ИИ. Нажмите на картинку, чтобы открыть на весь экран и приблизить.',
                          style: theme.textTheme.bodyMedium?.copyWith(
                            color: theme.colorScheme.onSurfaceVariant,
                          ),
                        ),
                      ],
                    ),
    );
  }
}

class _MaskPreviewCard extends StatelessWidget {
  final Uint8List inputImage;
  final Uint8List? userMaskPng;
  final Uint8List fallbackAnnotated;

  const _MaskPreviewCard({
    required this.inputImage,
    required this.userMaskPng,
    required this.fallbackAnnotated,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    final overlay = userMaskPng != null
        ? Opacity(
            opacity: 0.65,
            child: Image.memory(
              userMaskPng!,
              fit: BoxFit.contain,
            ),
          )
        : Opacity(
            opacity: 0.75,
            child: Image.memory(
              fallbackAnnotated,
              fit: BoxFit.contain,
            ),
          );

    return Material(
      color: theme.colorScheme.surface,
      borderRadius: BorderRadius.circular(18),
      clipBehavior: Clip.antiAlias,
      child: InkWell(
        onTap: () {
          Navigator.of(context).push(
            MaterialPageRoute(
              builder: (_) => _FullImageViewer(
                inputImage: inputImage,
                overlayPng: userMaskPng ?? fallbackAnnotated,
                overlayIsMask: userMaskPng != null,
              ),
            ),
          );
        },
        child: AspectRatio(
          aspectRatio: 4 / 3,
          child: Stack(
            fit: StackFit.expand,
            children: [
              Image.memory(inputImage, fit: BoxFit.cover),
              Align(
                alignment: Alignment.center,
                child: overlay,
              ),
              Positioned(
                left: 12,
                bottom: 12,
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                  decoration: BoxDecoration(
                    color: Colors.black.withOpacity(0.55),
                    borderRadius: BorderRadius.circular(999),
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      const Icon(Icons.open_in_full_rounded, color: Colors.white, size: 16),
                      const SizedBox(width: 6),
                      Text(
                        'Открыть',
                        style: theme.textTheme.labelMedium?.copyWith(
                          color: Colors.white,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _FullImageViewer extends StatelessWidget {
  final Uint8List inputImage;
  final Uint8List overlayPng;
  final bool overlayIsMask;

  const _FullImageViewer({
    required this.inputImage,
    required this.overlayPng,
    required this.overlayIsMask,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(
        title: Text(overlayIsMask ? 'Ваша маска' : 'Разметка ИИ'),
      ),
      body: InteractiveViewer(
        minScale: 1,
        maxScale: 8,
        child: Center(
          child: Stack(
            children: [
              Image.memory(inputImage),
              Opacity(
                opacity: overlayIsMask ? 0.65 : 0.75,
                child: Image.memory(overlayPng),
              ),
            ],
          ),
        ),
      ),
      bottomNavigationBar: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(12),
          child: Text(
            'Жестами: два пальца для увеличения/уменьшения, перетаскивание для перемещения.',
            textAlign: TextAlign.center,
            style: theme.textTheme.bodySmall?.copyWith(
              color: theme.colorScheme.onSurfaceVariant,
            ),
          ),
        ),
      ),
    );
  }
}

class _InfoRow extends StatelessWidget {
  final String label;
  final String value;
  const _InfoRow({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          SizedBox(
            width: 110,
            child: Text(
              label,
              style: theme.textTheme.bodyMedium?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              value,
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
              style: theme.textTheme.bodyMedium?.copyWith(fontWeight: FontWeight.w700),
            ),
          ),
        ],
      ),
    );
  }
}

class _BannerError extends StatelessWidget {
  final String message;
  final VoidCallback onClose;
  const _BannerError({required this.message, required this.onClose});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Container(
      margin: const EdgeInsets.fromLTRB(12, 12, 12, 0),
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: theme.colorScheme.errorContainer,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        children: [
          Expanded(
            child: Text(
              message,
              maxLines: 3,
              overflow: TextOverflow.ellipsis,
              style: theme.textTheme.bodyMedium?.copyWith(
                color: theme.colorScheme.onErrorContainer,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
          IconButton(
            onPressed: onClose,
            icon: Icon(Icons.close_rounded, color: theme.colorScheme.onErrorContainer),
          ),
        ],
      ),
    );
  }
}

class _ErrorView extends StatelessWidget {
  final String title;
  final String message;
  final VoidCallback onRetry;
  const _ErrorView({required this.title, required this.message, required this.onRetry});

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(title, style: Theme.of(context).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w800)),
            const SizedBox(height: 8),
            Text(message, textAlign: TextAlign.center),
            const SizedBox(height: 12),
            ElevatedButton.icon(
              onPressed: onRetry,
              icon: const Icon(Icons.refresh_rounded),
              label: const Text('Повторить'),
            ),
          ],
        ),
      ),
    );
  }
}
