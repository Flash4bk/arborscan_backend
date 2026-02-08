import 'dart:typed_data';

import 'package:flutter/material.dart';

import 'admin_service.dart';

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

  List<VerifiedItem> _items = const [];

  // кеш деталей, чтобы не грузить одно и то же много раз
  final Map<String, VerifiedAnalysis> _detailsCache = {};

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
      final list = await widget.service.getVerifiedList();
      setState(() {
        _items = list;
        _loading = false;
      });
    } catch (e) {
      setState(() {
        _error = e.toString();
        _loading = false;
      });
    }
  }

  int get _includedCount => _items.where((e) => !e.excludeFromTraining).length;
  int get _excludedCount => _items.where((e) => e.excludeFromTraining).length;

  /// Formats timestamp shown in the training dataset list.
  ///
  /// Backend returns `verified_at` as ISO8601 string (or null). Some older code
  /// paths might still pass a DateTime, so we accept both.
  String _fmtDateTime(dynamic value) {
    if (value == null) return '—';

    DateTime? dt;
    if (value is DateTime) {
      dt = value;
    } else if (value is String && value.trim().isNotEmpty) {
      try {
        dt = DateTime.parse(value).toLocal();
      } catch (_) {
        dt = null;
      }
    }

    if (dt == null) return '—';
    String two(int v) => v.toString().padLeft(2, '0');
    return '${two(dt.day)}.${two(dt.month)}.${dt.year} ${two(dt.hour)}:${two(dt.minute)}';
  }

  Future<VerifiedAnalysis> _getDetails(String analysisId) async {
    final cached = _detailsCache[analysisId];
    if (cached != null) return cached;
    final d = await widget.service.getVerifiedAnalysis(analysisId);
    _detailsCache[analysisId] = d;
    return d;
  }

  Future<void> _toggleInclude(VerifiedItem it) async {
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
      await widget.service.setTrainingInclude(it.analysisId, include: newInclude);
    } catch (e) {
      // rollback
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
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Датасет для обучения'),
        actions: [
          IconButton(
            onPressed: _loading ? null : _load,
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

                  _SummaryCard(
                    total: _items.length,
                    included: _includedCount,
                    excluded: _excludedCount,
                  ),
                  const SizedBox(height: 12),

                  if (_items.isEmpty)
                    const Padding(
                      padding: EdgeInsets.only(top: 40),
                      child: Center(
                        child: Text(
                          'Пока нет подтверждённых примеров.\nСначала сделай анализ → нарисуй маску → отправь фидбек → подтверди.',
                          textAlign: TextAlign.center,
                          style: TextStyle(color: Colors.black54),
                        ),
                      ),
                    )
                  else
                    ..._items.map((it) => _DatasetItemCard(
                          item: it,
                          addedAtText: _fmtDateTime(it.verifiedAt),
                          loadDetails: _getDetails,
                          onToggleInclude: () => _toggleInclude(it),
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

  const _SummaryCard({required this.total, required this.included, required this.excluded});

  @override
  Widget build(BuildContext context) {
    final tt = Theme.of(context).textTheme;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Итого в Supabase (verified): $total', style: tt.titleMedium?.copyWith(fontWeight: FontWeight.w700)),
            const SizedBox(height: 8),
            Wrap(
              spacing: 10,
              runSpacing: 8,
              children: [
                _Chip(label: 'В дообучение: $included', icon: Icons.check_circle_outline),
                _Chip(label: 'Исключено: $excluded', icon: Icons.block),
              ],
            ),
            const SizedBox(height: 10),
            const Text(
              'Если пример «Исключён», он останется в verified, но воркер retrain_worker.py пропустит его при сборке датасета.',
              style: TextStyle(color: Colors.black54),
            ),
          ],
        ),
      ),
    );
  }
}

class _DatasetItemCard extends StatefulWidget {
  final VerifiedItem item;
  final String addedAtText;
  final Future<VerifiedAnalysis> Function(String analysisId) loadDetails;
  final VoidCallback onToggleInclude;

  const _DatasetItemCard({
    required this.item,
    required this.addedAtText,
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
                        style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w700),
                      ),
                      const SizedBox(height: 2),
                      Text(
                        'ID: ${it.analysisId}',
                        style: const TextStyle(fontSize: 12, color: Colors.black54),
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                      const SizedBox(height: 2),
                      Text(
                        'Добавлено: ${widget.addedAtText}',
                        style: const TextStyle(fontSize: 12, color: Colors.black54),
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                    ],
                  ),
                ),
                const SizedBox(width: 10),
                _Badge(
                  text: excluded ? 'Исключён' : 'В дообучение',
                  icon: excluded ? Icons.block : Icons.check_circle_outline,
                ),
              ],
            ),
            const SizedBox(height: 10),
            Wrap(
              spacing: 10,
              runSpacing: 8,
              children: [
                if (it.riskCategory != null) _Chip(label: 'Риск: ${it.riskCategory}', icon: Icons.warning_amber_rounded),
                if (it.trustScore != null) _Chip(label: 'Trust: ${it.trustScore}', icon: Icons.verified_user_outlined),
              ],
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: widget.onToggleInclude,
                    icon: Icon(excluded ? Icons.add_circle_outline : Icons.remove_circle_outline),
                    label: Text(excluded ? 'Вернуть в обучение' : 'Исключить'),
                  ),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _toggleExpand,
                    icon: Icon(_expanded ? Icons.expand_less : Icons.expand_more),
                    label: Text(_expanded ? 'Свернуть' : 'Просмотр'),
                  ),
                ),
              ],
            ),

            if (_expanded) ...[
              const SizedBox(height: 12),
              if (_loading) const Center(child: Padding(padding: EdgeInsets.all(12), child: CircularProgressIndicator())),
              if (_err != null) Padding(padding: const EdgeInsets.only(bottom: 8), child: _ErrorBanner(message: _err!)),
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
        ),
        const SizedBox(height: 10),
        Wrap(
          spacing: 10,
          runSpacing: 8,
          children: [
            if (height != null) _Chip(label: 'Высота: $height', icon: Icons.height),
            if (risk['category'] != null) _Chip(label: 'Категория: ${risk['category']}', icon: Icons.shield_outlined),
            if (trust != null) _Chip(label: 'Trust: $trust', icon: Icons.verified_outlined),
          ],
        ),
      ],
    );
  }
}

class _TwoImages extends StatelessWidget {
  final Uint8List left;
  final Uint8List right;

  const _TwoImages({required this.left, required this.right});

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Expanded(child: _ImageTile(bytes: left, label: 'Оригинал')),
        const SizedBox(width: 10),
        Expanded(child: _ImageTile(bytes: right, label: 'Аннотация')),
      ],
    );
  }
}

class _ImageTile extends StatelessWidget {
  final Uint8List bytes;
  final String label;

  const _ImageTile({required this.bytes, required this.label});

  @override
  Widget build(BuildContext context) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(16),
      child: Stack(
        children: [
          AspectRatio(
            aspectRatio: 1,
            child: Image.memory(bytes, fit: BoxFit.cover),
          ),
          Positioned(
            left: 8,
            top: 8,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
              decoration: BoxDecoration(
                color: Colors.black.withOpacity(0.55),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Text(label, style: const TextStyle(color: Colors.white, fontSize: 12, fontWeight: FontWeight.w600)),
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
