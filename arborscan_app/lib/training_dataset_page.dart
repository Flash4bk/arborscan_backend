import 'dart:typed_data';

import 'package:flutter/material.dart';

import 'admin_service.dart';
import 'app_theme.dart';
import 'corrections_service.dart';

/// Экран «Датасет для обучения» — показывает подтверждённые примеры
/// и даёт возможность исключать/включать их в дообучение.
class TrainingDatasetPage extends StatefulWidget {
  final AdminService service;

  const TrainingDatasetPage({super.key, required this.service});

  @override
  State<TrainingDatasetPage> createState() => _TrainingDatasetPageState();
}

class _TrainingDatasetPageState extends State<TrainingDatasetPage> {
  bool _loading = true;
  String? _error;
  String _filter = 'all';
  bool _invalid = false;
  final Set<String> _saving = {};

  List<VerifiedItem> _items = const [];

  // кеш деталей, чтобы не грузить одно и то же много раз
  final Map<String, VerifiedAnalysis> _detailsCache = {};

  @override
  void initState() {
    super.initState();
    CorrectionsService.authChanges.addListener(_invalidate);
    _load();
  }

  void _invalidate() {
    if (!mounted) return;
    setState(() {
      _invalid = true;
      _loading = false;
      _items = [];
      _detailsCache.clear();
      _error = 'Аккаунт изменился. Откройте архив заново.';
    });
  }

  @override
  void dispose() {
    CorrectionsService.authChanges.removeListener(_invalidate);
    super.dispose();
  }

  Future<void> _load() async {
    if (_invalid) return;
    final generation = CorrectionsService.authChanges.value;
    setState(() {
      _loading = true;
      _error = null;
    });

    try {
      final list = await widget.service.getVerifiedList();
      if (!mounted ||
          _invalid ||
          generation != CorrectionsService.authChanges.value) {
        return;
      }
      setState(() {
        _items = list;
        _loading = false;
      });
    } catch (e) {
      if (!mounted ||
          _invalid ||
          generation != CorrectionsService.authChanges.value) {
        return;
      }
      setState(() {
        _error = e.toString();
        _loading = false;
      });
    }
  }

  int get _includedCount => _items.where((e) => !e.excludeFromTraining).length;
  int get _excludedCount => _items.where((e) => e.excludeFromTraining).length;

  Future<VerifiedAnalysis> _getDetails(String analysisId) async {
    final cached = _detailsCache[analysisId];
    if (cached != null) return cached;
    final generation = CorrectionsService.authChanges.value;
    final d = await widget.service.getVerifiedAnalysis(analysisId);
    if (_invalid || generation != CorrectionsService.authChanges.value) {
      throw const AdminApiException('Аккаунт изменился.');
    }
    _detailsCache[analysisId] = d;
    return d;
  }

  Future<void> _toggleInclude(VerifiedItem it) async {
    if (_invalid || _saving.contains(it.analysisId)) return;
    final generation = CorrectionsService.authChanges.value;
    _saving.add(it.analysisId);
    final newInclude = it.excludeFromTraining; // если был excluded -> включаем

    // optimistic
    setState(() {
      _items = _items
          .map((x) => x.analysisId == it.analysisId
              ? VerifiedItem(
                  analysisId: x.analysisId,
                  verified: x.verified,
                  excludeFromTraining: !newInclude,
                  species: x.species,
                  riskCategory: x.riskCategory,
                  trustScore: x.trustScore,
                  verifiedAt: x.verifiedAt,
                )
              : x)
          .toList();
    });

    try {
      await widget.service
          .setTrainingInclude(it.analysisId, include: newInclude);
    } catch (e) {
      // Roll back only within the initiating account.
      if (!mounted ||
          _invalid ||
          generation != CorrectionsService.authChanges.value) {
        return;
      }
      setState(() {
        _items = _items
            .map((x) => x.analysisId == it.analysisId
                ? VerifiedItem(
                    analysisId: x.analysisId,
                    verified: x.verified,
                    excludeFromTraining: it.excludeFromTraining,
                    species: x.species,
                    riskCategory: x.riskCategory,
                    trustScore: x.trustScore,
                    verifiedAt: x.verifiedAt,
                  )
                : x)
            .toList();
        _error = e.toString();
      });
    } finally {
      if (mounted && !_invalid) setState(() => _saving.remove(it.analysisId));
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Архив примеров'),
        actions: [
          IconButton(
            tooltip: 'Обновить датасет',
            onPressed: _loading || _invalid ? null : _load,
            icon: const Icon(Icons.refresh),
          ),
        ],
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : RefreshIndicator(
              onRefresh: _load,
              child: ListView(
                padding: const EdgeInsets.all(16),
                children: [
                  if (_error != null) ...[
                    _ErrorBanner(message: _error!),
                    const SizedBox(height: 12),
                  ],
                  if (!_invalid)
                    _SummaryCard(
                      total: _items.length,
                      included: _includedCount,
                      excluded: _excludedCount,
                    ),
                  const SizedBox(height: 12),
                  if (!_invalid)
                    Wrap(spacing: 8, runSpacing: 4, children: [
                      for (final filter in const {
                        'all': 'Все',
                        'included': 'Включены',
                        'excluded': 'Исключены'
                      }.entries)
                        ChoiceChip(
                            label: Text(filter.value),
                            selected: _filter == filter.key,
                            onSelected: (_) =>
                                setState(() => _filter = filter.key)),
                    ]),
                  if (_items.isEmpty && !_invalid)
                    const Padding(
                      padding: EdgeInsets.only(top: 40),
                      child: Center(
                        child: Text(
                          'В архиве пока нет примеров.',
                          textAlign: TextAlign.center,
                          style: TextStyle(color: AppTheme.muted),
                        ),
                      ),
                    )
                  else
                    ..._items
                        .where((it) =>
                            _filter == 'all' ||
                            (_filter == 'excluded'
                                ? it.excludeFromTraining
                                : !it.excludeFromTraining))
                        .map((it) => _DatasetItemCard(
                              key: ValueKey(it.analysisId),
                              item: it,
                              loadDetails: _getDetails,
                              onToggleInclude: _saving.contains(it.analysisId)
                                  ? null
                                  : () => _toggleInclude(it),
                            )),
                ],
              ),
            ),
    );
  }
}

class _SummaryCard extends StatelessWidget {
  final int total;
  final int included;
  final int excluded;

  const _SummaryCard(
      {required this.total, required this.included, required this.excluded});

  @override
  Widget build(BuildContext context) {
    final tt = Theme.of(context).textTheme;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Записей в архиве: $total',
                style: tt.titleMedium?.copyWith(fontWeight: FontWeight.w700)),
            const SizedBox(height: 8),
            Wrap(
              spacing: 10,
              runSpacing: 8,
              children: [
                _Chip(
                    label: 'Включены: $included',
                    icon: Icons.check_circle_outline),
                _Chip(label: 'Исключено: $excluded', icon: Icons.block),
              ],
            ),
            const SizedBox(height: 10),
            const Text(
              'Это прежний каталог. Его флаг не допускает запись к новому обучению: нужна принятая ревизия в разделе «Модели и данные».',
              style: TextStyle(color: AppTheme.muted),
            ),
          ],
        ),
      ),
    );
  }
}

class _DatasetItemCard extends StatefulWidget {
  final VerifiedItem item;
  final Future<VerifiedAnalysis> Function(String analysisId) loadDetails;
  final VoidCallback? onToggleInclude;

  const _DatasetItemCard({
    super.key,
    required this.item,
    required this.loadDetails,
    required this.onToggleInclude,
  });

  @override
  State<_DatasetItemCard> createState() => _DatasetItemCardState();
}

class _DatasetItemCardState extends State<_DatasetItemCard> {
  bool _expanded = false;
  bool _loading = false;
  VerifiedAnalysis? _details;
  String? _err;

  Future<void> _toggleExpand() async {
    setState(() {
      _expanded = !_expanded;
      _err = null;
    });
    if (!_expanded) return;
    if (_details != null) return;

    setState(() => _loading = true);
    try {
      final d = await widget.loadDetails(widget.item.analysisId);
      if (!mounted) return;
      setState(() {
        _details = d;
        _loading = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _err = e.toString();
        _loading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final it = widget.item;
    final excluded = it.excludeFromTraining;

    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        it.species ?? 'Без вида',
                        style: const TextStyle(
                            fontSize: 16, fontWeight: FontWeight.w700),
                      ),
                      const SizedBox(height: 2),
                      Text(
                        'ID: ${it.analysisId}',
                        style: const TextStyle(
                            fontSize: 12, color: AppTheme.muted),
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                    ],
                  ),
                ),
                const SizedBox(width: 10),
                _Badge(
                  text: excluded ? 'Исключён' : 'Включён',
                  icon: excluded ? Icons.block : Icons.check_circle_outline,
                ),
              ],
            ),
            const SizedBox(height: 10),
            Wrap(
              spacing: 10,
              runSpacing: 8,
              children: [
                if (it.riskCategory != null)
                  _Chip(
                      label: 'Прежняя оценка: ${it.riskCategory}',
                      icon: Icons.warning_amber_rounded),
                if (it.trustScore != null)
                  _Chip(
                      label: 'Прежний индекс: ${it.trustScore}',
                      icon: Icons.verified_user_outlined),
              ],
            ),
            const SizedBox(height: 12),
            Wrap(spacing: 8, runSpacing: 8, children: [
              OutlinedButton.icon(
                  onPressed: widget.onToggleInclude,
                  icon: Icon(excluded
                      ? Icons.add_circle_outline
                      : Icons.remove_circle_outline),
                  label: Text(excluded ? 'Включить в архиве' : 'Исключить')),
              TextButton.icon(
                  onPressed: _toggleExpand,
                  icon: Icon(_expanded ? Icons.expand_less : Icons.expand_more),
                  label: Text(_expanded ? 'Свернуть' : 'Просмотр')),
            ]),
            if (_expanded) ...[
              const SizedBox(height: 12),
              if (_loading)
                const Center(
                    child: Padding(
                        padding: EdgeInsets.all(12),
                        child: CircularProgressIndicator())),
              if (_err != null)
                Padding(
                    padding: const EdgeInsets.only(bottom: 8),
                    child: _ErrorBanner(message: _err!)),
              if (_details != null) _DetailsBlock(details: _details!),
            ],
          ],
        ),
      ),
    );
  }
}

class _DetailsBlock extends StatelessWidget {
  final VerifiedAnalysis details;

  const _DetailsBlock({required this.details});

  @override
  Widget build(BuildContext context) {
    final meta = details.meta;
    final risk = (meta['risk'] is Map) ? (meta['risk'] as Map) : const {};
    final trust = meta['trust_score'];
    final height = meta['height_m'] ?? meta['height'] ?? meta['heightMeters'];

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _TwoImages(
          left: details.inputImage,
          right: details.annotatedImage,
          userMask: details.userMaskImage,
        ),
        const SizedBox(height: 10),
        Wrap(
          spacing: 10,
          runSpacing: 8,
          children: [
            if (height != null)
              _Chip(label: 'Высота: $height', icon: Icons.height),
            if (risk['category'] != null)
              _Chip(
                  label: 'Категория: ${risk['category']}',
                  icon: Icons.shield_outlined),
            if (trust != null)
              _Chip(
                  label: 'Прежний индекс: $trust',
                  icon: Icons.verified_outlined),
          ],
        ),
      ],
    );
  }
}

class _TwoImages extends StatelessWidget {
  final Uint8List left;
  final Uint8List right;
  final Uint8List? userMask;

  const _TwoImages({
    required this.left,
    required this.right,
    this.userMask,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Row(
          children: [
            Expanded(child: _ImageTile(bytes: left, label: 'Оригинал')),
            const SizedBox(width: 10),
            Expanded(child: _ImageTile(bytes: right, label: 'Аннотация (ИИ)')),
          ],
        ),
        if (userMask != null) ...[
          const SizedBox(height: 10),
          _ImageTile(
            bytes: left,
            overlay: userMask,
            label: 'Маска автора',
            height: 180,
          ),
        ],
      ],
    );
  }
}

class _ImageTile extends StatelessWidget {
  final Uint8List bytes;
  final Uint8List? overlay;
  final String label;
  final double height;

  const _ImageTile({
    required this.bytes,
    required this.label,
    this.overlay,
    this.height = 140,
  });

  void _open(BuildContext context) {
    showDialog<void>(
      context: context,
      builder: (ctx) => Dialog(
        insetPadding: const EdgeInsets.all(16),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
              child: Row(
                children: [
                  Expanded(
                    child: Text(
                      label,
                      style: const TextStyle(
                          fontSize: 16, fontWeight: FontWeight.w600),
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                  IconButton(
                    tooltip: 'Закрыть изображение',
                    onPressed: () => Navigator.of(ctx).pop(),
                    icon: const Icon(Icons.close),
                  ),
                ],
              ),
            ),
            Flexible(
              child: InteractiveViewer(
                minScale: 0.7,
                maxScale: 6,
                child: Stack(
                  children: [
                    Image.memory(bytes, fit: BoxFit.contain),
                    if (overlay != null)
                      Positioned.fill(
                        child: Opacity(
                          opacity: 0.7,
                          child: Image.memory(overlay!, fit: BoxFit.contain),
                        ),
                      ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 12),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return InkWell(
      borderRadius: BorderRadius.circular(12),
      onTap: () => _open(context),
      child: Stack(
        children: [
          Container(
            height: height,
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: AppTheme.border),
            ),
            clipBehavior: Clip.antiAlias,
            child: Stack(
              fit: StackFit.expand,
              children: [
                Image.memory(bytes, fit: BoxFit.cover),
                if (overlay != null)
                  Opacity(
                    opacity: 0.7,
                    child: Image.memory(overlay!, fit: BoxFit.cover),
                  ),
              ],
            ),
          ),
          Positioned(
            left: 8,
            top: 8,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
              decoration: BoxDecoration(
                color: AppTheme.muted,
                borderRadius: BorderRadius.circular(999),
              ),
              child: Text(
                label,
                style: const TextStyle(color: Colors.white, fontSize: 12),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _Badge extends StatelessWidget {
  final String text;
  final IconData icon;

  const _Badge({required this.text, required this.icon});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.secondaryContainer,
        borderRadius: BorderRadius.circular(999),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 16),
          const SizedBox(width: 6),
          Text(text, style: const TextStyle(fontWeight: FontWeight.w700)),
        ],
      ),
    );
  }
}

class _Chip extends StatelessWidget {
  final String label;
  final IconData icon;

  const _Chip({required this.label, required this.icon});

  @override
  Widget build(BuildContext context) {
    return Chip(
      avatar: Icon(icon, size: 18),
      label: Text(label),
      visualDensity: VisualDensity.compact,
    );
  }
}

class _ErrorBanner extends StatelessWidget {
  final String message;

  const _ErrorBanner({required this.message});

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.errorContainer,
        borderRadius: BorderRadius.circular(14),
      ),
      padding: const EdgeInsets.all(12),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(Icons.error_outline),
          const SizedBox(width: 10),
          Expanded(child: Text(message)),
        ],
      ),
    );
  }
}
