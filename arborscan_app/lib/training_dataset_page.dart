import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/material.dart';

import 'admin_service.dart';
import 'app_theme.dart';

class TrainingDatasetPage extends StatefulWidget {
  final AdminService service;

  const TrainingDatasetPage({super.key, required this.service});

  @override
  State<TrainingDatasetPage> createState() => _TrainingDatasetPageState();
}

class _TrainingDatasetPageState extends State<TrainingDatasetPage> {
  bool _loading = true;
  String? _error;

  List<VerifiedItem> _all = const [];
  List<VerifiedItem> _filtered = const [];

  final TextEditingController _searchCtrl = TextEditingController();
  bool _showExcluded = true;
  bool _onlyIncluded = false;

  @override
  void initState() {
    super.initState();
    _load();
    _searchCtrl.addListener(_applyFilters);
  }

  @override
  void dispose() {
    _searchCtrl.dispose();
    super.dispose();
  }

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _error = null;
    });

    try {
      final items = await widget.service.getVerifiedList();
      // newest first if backend includes verifiedAt; otherwise keep as-is
      items.sort((a, b) => (b.verifiedAt ?? '').compareTo(a.verifiedAt ?? ''));

      if (!mounted) return;
      setState(() {
        _all = items;
        _loading = false;
      });
      _applyFilters();
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = e.toString();
        _loading = false;
      });
    }
  }

  void _applyFilters() {
    final q = _searchCtrl.text.trim().toLowerCase();
    var out = List<VerifiedItem>.of(_all);

    if (!_showExcluded) {
      out = out.where((e) => !e.excludeFromTraining).toList();
    }
    if (_onlyIncluded) {
      out = out.where((e) => !e.excludeFromTraining).toList();
    }

    if (q.isNotEmpty) {
      out = out.where((e) {
        final hay = [
          e.analysisId,
          e.species ?? '',
          e.riskCategory ?? '',
        ].join(' ').toLowerCase();
        return hay.contains(q);
      }).toList();
    }

    setState(() => _filtered = out);
  }

  Future<void> _toggleInclude(VerifiedItem item, bool include) async {
    try {
      await widget.service.setTrainingInclude(item.analysisId, include: include);
      // Optimistic update
      final idx = _all.indexWhere((e) => e.analysisId == item.analysisId);
      if (idx != -1) {
        final updated = VerifiedItem(
          analysisId: item.analysisId,
          verified: item.verified,
          excludeFromTraining: !include,
          species: item.species,
          riskCategory: item.riskCategory,
          trustScore: item.trustScore,
          verifiedAt: item.verifiedAt,
        );
        final newAll = List<VerifiedItem>.of(_all);
        newAll[idx] = updated;
        setState(() => _all = newAll);
        _applyFilters();
      }

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text(include ? 'Добавлено в обучение' : 'Исключено из обучения')),
        );
      }
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Ошибка: $e')),
      );
    }
  }

  Future<void> _openPreview(String analysisId) async {
    showModalBottomSheet(
      context: context,
      showDragHandle: true,
      isScrollControlled: true,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(18)),
      ),
      builder: (_) {
        return _PreviewSheet(service: widget.service, analysisId: analysisId);
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Датасет для обучения'),
        actions: [
          IconButton(
            tooltip: 'Обновить',
            icon: const Icon(Icons.refresh),
            onPressed: _loading ? null : _load,
          ),
        ],
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : _error != null
              ? _ErrorState(message: _error!, onRetry: _load)
              : Column(
                  children: [
                    Padding(
                      padding: const EdgeInsets.fromLTRB(16, 12, 16, 10),
                      child: Column(
                        children: [
                          TextField(
                            controller: _searchCtrl,
                            decoration: const InputDecoration(
                              prefixIcon: Icon(Icons.search),
                              labelText: 'Поиск (вид / риск / id)',
                            ),
                          ),
                          const SizedBox(height: 10),
                          Row(
                            children: [
                              FilterChip(
                                label: const Text('Показывать исключённые'),
                                selected: _showExcluded,
                                onSelected: (v) {
                                  setState(() => _showExcluded = v);
                                  _applyFilters();
                                },
                              ),
                              const SizedBox(width: 10),
                              FilterChip(
                                label: const Text('Только в обучении'),
                                selected: _onlyIncluded,
                                onSelected: (v) {
                                  setState(() => _onlyIncluded = v);
                                  _applyFilters();
                                },
                              ),
                            ],
                          ),
                          const SizedBox(height: 8),
                          Row(
                            children: [
                              Expanded(
                                child: Text(
                                  'Всего: ${_all.length} • Показано: ${_filtered.length}',
                                  style: Theme.of(context)
                                      .textTheme
                                      .bodySmall
                                      ?.copyWith(color: AppTheme.muted),
                                ),
                              ),
                            ],
                          ),
                        ],
                      ),
                    ),
                    Expanded(
                      child: _filtered.isEmpty
                          ? ListView(
                              padding: const EdgeInsets.fromLTRB(16, 6, 16, 16),
                              children: [
                                Ui.paddedCard(
                                  context,
                                  child: Row(
                                    crossAxisAlignment: CrossAxisAlignment.start,
                                    children: [
                                      const Icon(Icons.inbox_outlined),
                                      const SizedBox(width: 12),
                                      Expanded(
                                        child: Text(
                                          'Нет примеров под текущие фильтры.\n\n'
                                          'Подтверди примеры (trusted/examples) — и они появятся здесь.',
                                          style: Theme.of(context)
                                              .textTheme
                                              .bodyMedium
                                              ?.copyWith(color: AppTheme.muted),
                                        ),
                                      ),
                                    ],
                                  ),
                                ),
                              ],
                            )
                          : ListView.separated(
                              padding: const EdgeInsets.fromLTRB(16, 6, 16, 16),
                              itemCount: _filtered.length,
                              separatorBuilder: (_, __) => const SizedBox(height: 12),
                              itemBuilder: (context, i) {
                                final item = _filtered[i];
                                final include = !item.excludeFromTraining;

                                return Card(
                                  child: InkWell(
                                    onTap: () => _openPreview(item.analysisId),
                                    borderRadius: BorderRadius.circular(16),
                                    child: Padding(
                                      padding: const EdgeInsets.all(12),
                                      child: Column(
                                        crossAxisAlignment: CrossAxisAlignment.start,
                                        children: [
                                          Row(
                                            children: [
                                              Expanded(
                                                child: Text(
                                                  item.species?.isNotEmpty == true
                                                      ? item.species!
                                                      : 'Неизвестно',
                                                  maxLines: 1,
                                                  overflow: TextOverflow.ellipsis,
                                                  style: Theme.of(context)
                                                      .textTheme
                                                      .titleMedium
                                                      ?.copyWith(fontWeight: FontWeight.w800),
                                                ),
                                              ),
                                              const SizedBox(width: 10),
                                              Ui.badge(
                                                text: include ? 'В обучении' : 'Исключено',
                                                color: include ? AppTheme.success : AppTheme.muted,
                                                icon: include ? Icons.check_circle : Icons.block,
                                              ),
                                            ],
                                          ),
                                          const SizedBox(height: 8),
                                          Wrap(
                                            spacing: 8,
                                            runSpacing: 8,
                                            children: [
                                              if (item.riskCategory?.isNotEmpty == true)
                                                Ui.badge(
                                                  text: item.riskCategory!,
                                                  color: AppTheme.warning,
                                                  icon: Icons.shield,
                                                ),
                                              Ui.badge(
                                                text: 'ID: ${item.analysisId.substring(0, item.analysisId.length > 10 ? 10 : item.analysisId.length)}',
                                                color: AppTheme.primary,
                                                icon: Icons.tag,
                                              ),
                                              if (item.trustScore != null)
                                                Ui.badge(
                                                  text: 'Trust: ${item.trustScore!.toStringAsFixed(2)}',
                                                  color: AppTheme.primary,
                                                  icon: Icons.verified,
                                                ),
                                            ],
                                          ),
                                          const SizedBox(height: 12),
                                          Row(
                                            children: [
                                              Expanded(
                                                child: Text(
                                                  'Тап — предпросмотр примера',
                                                  style: Theme.of(context)
                                                      .textTheme
                                                      .bodySmall
                                                      ?.copyWith(color: AppTheme.muted),
                                                ),
                                              ),
                                              Switch(
                                                value: include,
                                                onChanged: (v) => _toggleInclude(item, v),
                                              ),
                                            ],
                                          ),
                                        ],
                                      ),
                                    ),
                                  ),
                                );
                              },
                            ),
                    ),
                  ],
                ),
    );
  }
}

class _ErrorState extends StatelessWidget {
  final String message;
  final VoidCallback onRetry;

  const _ErrorState({required this.message, required this.onRetry});

  @override
  Widget build(BuildContext context) {
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Ui.paddedCard(
          context,
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Icon(Icons.error_outline, color: AppTheme.danger),
              const SizedBox(width: 12),
              Expanded(
                child: Text(
                  message,
                  style: Theme.of(context)
                      .textTheme
                      .bodyMedium
                      ?.copyWith(color: AppTheme.danger),
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 12),
        ElevatedButton.icon(
          onPressed: onRetry,
          icon: const Icon(Icons.refresh),
          label: const Text('Повторить'),
        ),
      ],
    );
  }
}

class _PreviewSheet extends StatefulWidget {
  final AdminService service;
  final String analysisId;

  const _PreviewSheet({required this.service, required this.analysisId});

  @override
  State<_PreviewSheet> createState() => _PreviewSheetState();
}

class _PreviewSheetState extends State<_PreviewSheet> {
  bool _loading = true;
  String? _error;
  VerifiedAnalysis? _data;

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
      final d = await widget.service.getVerifiedAnalysis(widget.analysisId);
      if (!mounted) return;
      setState(() {
        _data = d;
        _loading = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = e.toString();
        _loading = false;
      });
    }
  }

  Uint8List? _b64(String s) {
    try {
      return base64Decode(s);
    } catch (_) {
      return null;
    }
  }

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
        child: _loading
            ? const SizedBox(height: 240, child: Center(child: CircularProgressIndicator()))
            : _error != null
                ? Column(
                    mainAxisSize: MainAxisSize.min,
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Ошибка: $_error',
                        style: Theme.of(context)
                            .textTheme
                            .bodyMedium
                            ?.copyWith(color: AppTheme.danger),
                      ),
                      const SizedBox(height: 12),
                      ElevatedButton.icon(
                        onPressed: _load,
                        icon: const Icon(Icons.refresh),
                        label: const Text('Повторить'),
                      ),
                    ],
                  )
                : _buildContent(context),
      ),
    );
  }

  Widget _buildContent(BuildContext context) {
    final d = _data!;
    final inBytes = _b64(d.inputBase64);
    final annBytes = _b64(d.annotatedBase64);

    String? species;
    String? risk;
    try {
      species = d.meta['species']?.toString();
      final r = d.meta['risk'];
      if (r is Map) risk = r['category']?.toString();
    } catch (_) {}

    return SingleChildScrollView(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Ui.badge(text: 'ID', color: AppTheme.primary, icon: Icons.tag),
              const SizedBox(width: 10),
              Expanded(
                child: Text(
                  d.analysisId,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: Theme.of(context).textTheme.bodyMedium,
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          Wrap(
            spacing: 10,
            runSpacing: 10,
            children: [
              if (species?.isNotEmpty == true)
                Ui.badge(text: species!, color: AppTheme.success, icon: Icons.park),
              if (risk?.isNotEmpty == true)
                Ui.badge(text: risk!, color: AppTheme.warning, icon: Icons.shield),
            ],
          ),
          const SizedBox(height: 14),
          Text('Входное фото', style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w800)),
          const SizedBox(height: 8),
          _imageBlock(inBytes),
          const SizedBox(height: 14),
          Text('Разметка / Annotated', style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w800)),
          const SizedBox(height: 8),
          _imageBlock(annBytes),
          const SizedBox(height: 14),
          OutlinedButton.icon(
            onPressed: () => Navigator.of(context).pop(),
            icon: const Icon(Icons.close),
            label: const Text('Закрыть'),
          ),
        ],
      ),
    );
  }

  Widget _imageBlock(Uint8List? bytes) {
    if (bytes == null) {
      return Container(
        height: 160,
        alignment: Alignment.center,
        decoration: BoxDecoration(
          color: Colors.black.withOpacity(0.04),
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: AppTheme.border),
        ),
        child: const Icon(Icons.image_not_supported),
      );
    }
    return ClipRRect(
      borderRadius: BorderRadius.circular(14),
      child: Image.memory(
        bytes,
        height: 180,
        width: double.infinity,
        fit: BoxFit.cover,
      ),
    );
  }
}
