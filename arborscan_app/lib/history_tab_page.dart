import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'app_theme.dart';
import 'map_page.dart';

class HistoryTabPage extends StatefulWidget {
  const HistoryTabPage({super.key});

  @override
  State<HistoryTabPage> createState() => _HistoryTabPageState();
}

class _HistoryTabPageState extends State<HistoryTabPage> {
  static const String _historyKey = 'arborscan_history';

  bool _loading = true;
  String? _error;

  List<_HistoryItem> _all = const [];
  List<_HistoryItem> _filtered = const [];

  final TextEditingController _searchCtrl = TextEditingController();

  _FilterMode _filterMode = _FilterMode.all;
  bool _onlyWithGeo = false;

  @override
  void initState() {
    super.initState();
    _load();

    _searchCtrl.addListener(() {
      _applyFilters();
    });
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
      final prefs = await SharedPreferences.getInstance();
      final list = prefs.getStringList(_historyKey) ?? const [];

      final items = <_HistoryItem>[];
      for (final s in list) {
        try {
          final m = jsonDecode(s) as Map<String, dynamic>;
          items.add(_HistoryItem.fromJson(m));
        } catch (_) {}
      }

      // newest first
      items.sort((a, b) => b.timestamp.compareTo(a.timestamp));

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

    List<_HistoryItem> out = List.of(_all);

    if (q.isNotEmpty) {
      out = out.where((e) {
        final hay = [
          e.species,
          e.address ?? '',
          e.riskCategory ?? '',
        ].join(' ').toLowerCase();
        return hay.contains(q);
      }).toList();
    }

    if (_filterMode == _FilterMode.withRisk) {
      out = out.where((e) => e.riskIndex != null || (e.riskCategory?.isNotEmpty ?? false)).toList();
    } else if (_filterMode == _FilterMode.noRisk) {
      out = out.where((e) => e.riskIndex == null && (e.riskCategory?.isEmpty ?? true)).toList();
    }

    if (_onlyWithGeo) {
      out = out.where((e) => e.lat != null && e.lon != null).toList();
    }

    setState(() => _filtered = out);
  }

  Future<void> _clearAll() async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Очистить историю?'),
        content: const Text('Все записи будут удалены.'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Отмена')),
          FilledButton(onPressed: () => Navigator.pop(ctx, true), child: const Text('Очистить')),
        ],
      ),
    );

    if (ok != true) return;

    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_historyKey);
    await _load();
  }

  Future<void> _deleteOne(_HistoryItem item) async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Удалить запись?'),
        content: Text('“${item.species}” будет удалено из истории.'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Отмена')),
          FilledButton(onPressed: () => Navigator.pop(ctx, true), child: const Text('Удалить')),
        ],
      ),
    );

    if (ok != true) return;

    final prefs = await SharedPreferences.getInstance();
    final list = prefs.getStringList(_historyKey) ?? const [];

    // Remove by best-effort stable id
    final newList = <String>[];
    for (final s in list) {
      try {
        final m = jsonDecode(s) as Map<String, dynamic>;
        final other = _HistoryItem.fromJson(m);
        if (other.uniqueKey != item.uniqueKey) {
          newList.add(s);
        }
      } catch (_) {
        // keep broken as-is
        newList.add(s);
      }
    }

    await prefs.setStringList(_historyKey, newList);
    await _load();

    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Запись удалена')));
  }

  void _openDetails(_HistoryItem item) {
    showModalBottomSheet(
      context: context,
      showDragHandle: true,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(18)),
      ),
      builder: (_) {
        final bytes = item.imageBase64.isNotEmpty ? _safeB64(item.imageBase64) : null;

        return SafeArea(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Ui.badge(
                      text: item.riskCategory?.isNotEmpty == true ? item.riskCategory! : 'Анализ',
                      color: AppTheme.primary,
                      icon: Icons.history,
                    ),
                    const Spacer(),
                    Text(
                      item.formattedTs,
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(color: AppTheme.muted),
                    ),
                  ],
                ),
                const SizedBox(height: 12),
                Text(
                  item.species.isNotEmpty ? item.species : 'Неизвестно',
                  style: Theme.of(context).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w800),
                ),
                const SizedBox(height: 10),

                if (bytes != null) ...[
                  ClipRRect(
                    borderRadius: BorderRadius.circular(14),
                    child: Image.memory(
                      bytes,
                      height: 150,
                      width: double.infinity,
                      fit: BoxFit.cover,
                      errorBuilder: (_, __, ___) => Container(
                        height: 150,
                        alignment: Alignment.center,
                        decoration: BoxDecoration(
                          color: Colors.black.withOpacity(0.04),
                          borderRadius: BorderRadius.circular(14),
                          border: Border.all(color: AppTheme.border),
                        ),
                        child: const Icon(Icons.image_not_supported),
                      ),
                    ),
                  ),
                  const SizedBox(height: 12),
                ],

                Wrap(
                  spacing: 10,
                  runSpacing: 10,
                  children: [
                    if (item.height != null)
                      Ui.badge(text: 'H: ${item.height!.toStringAsFixed(2)} м', color: AppTheme.success, icon: Icons.height),
                    if (item.crown != null)
                      Ui.badge(text: 'Крона: ${item.crown!.toStringAsFixed(2)} м', color: AppTheme.success, icon: Icons.nature),
                    if (item.trunk != null)
                      Ui.badge(text: 'Ствол: ${item.trunk!.toStringAsFixed(2)} м', color: AppTheme.success, icon: Icons.circle),
                    if (item.riskIndex != null)
                      Ui.badge(text: 'Риск: ${item.riskIndex!.toStringAsFixed(2)}', color: AppTheme.warning, icon: Icons.shield),
                    Ui.badge(
                      text: (item.lat != null && item.lon != null) ? 'Есть GPS' : 'Нет GPS',
                      color: (item.lat != null && item.lon != null) ? AppTheme.success : AppTheme.muted,
                      icon: Icons.location_on,
                    ),
                  ],
                ),

                if (item.address != null && item.address!.isNotEmpty) ...[
                  const SizedBox(height: 12),
                  Text(item.address!, style: Theme.of(context).textTheme.bodyMedium),
                ],

                const SizedBox(height: 14),

                Row(
                  children: [
                    Expanded(
                      child: OutlinedButton.icon(
                        onPressed: () {
                          Navigator.of(context).pop();
                          _deleteOne(item);
                        },
                        icon: const Icon(Icons.delete_outline),
                        label: const Text('Удалить'),
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: ElevatedButton.icon(
                        onPressed: (item.lat == null || item.lon == null)
                            ? null
                            : () {
                                Navigator.of(context).pop();
                                Navigator.of(context).push(
                                  MaterialPageRoute(
                                    builder: (_) => MapPage(
                                      initialFocus: LatLngFocus(item.lat!, item.lon!, zoom: 17),
                                    ),
                                  ),
                                );
                              },
                        icon: const Icon(Icons.map),
                        label: const Text('На карте'),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  Uint8List? _safeB64(String b64) {
    try {
      return base64Decode(b64);
    } catch (_) {
      return null;
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_loading) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text('История'),
        actions: [
          IconButton(tooltip: 'Обновить', icon: const Icon(Icons.refresh), onPressed: _load),
          IconButton(
            tooltip: 'Очистить всё',
            icon: const Icon(Icons.delete_outline),
            onPressed: _all.isEmpty ? null : _clearAll,
          ),
        ],
      ),
      body: _error != null
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
                          labelText: 'Поиск (вид / адрес / риск)',
                        ),
                      ),
                      const SizedBox(height: 10),
                      Row(
                        children: [
                          Expanded(
                            child: _FilterChipGroup(
                              value: _filterMode,
                              onChanged: (v) {
                                setState(() => _filterMode = v);
                                _applyFilters();
                              },
                            ),
                          ),
                          const SizedBox(width: 10),
                          FilterChip(
                            label: const Text('Только GPS'),
                            selected: _onlyWithGeo,
                            onSelected: (v) {
                              setState(() => _onlyWithGeo = v);
                              _applyFilters();
                            },
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
                                      'Ничего не найдено по текущим фильтрам.\nПопробуй изменить поиск или режим.',
                                      style: Theme.of(context).textTheme.bodyMedium?.copyWith(color: AppTheme.muted),
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
                            final bytes = item.imageBase64.isNotEmpty ? _safeB64(item.imageBase64) : null;

                            return Card(
                              child: InkWell(
                                onTap: () => _openDetails(item),
                                borderRadius: BorderRadius.circular(16),
                                child: Padding(
                                  padding: const EdgeInsets.all(12),
                                  child: Row(
                                    children: [
                                      ClipRRect(
                                        borderRadius: BorderRadius.circular(12),
                                        child: Container(
                                          width: 64,
                                          height: 64,
                                          color: Colors.black.withOpacity(0.04),
                                          child: bytes == null
                                              ? const Icon(Icons.park_outlined)
                                              : Image.memory(bytes, fit: BoxFit.cover),
                                        ),
                                      ),
                                      const SizedBox(width: 12),
                                      Expanded(
                                        child: Column(
                                          crossAxisAlignment: CrossAxisAlignment.start,
                                          children: [
                                            Text(
                                              item.species.isNotEmpty ? item.species : 'Неизвестно',
                                              maxLines: 1,
                                              overflow: TextOverflow.ellipsis,
                                              style: Theme.of(context).textTheme.titleMedium?.copyWith(
                                                    fontWeight: FontWeight.w800,
                                                  ),
                                            ),
                                            const SizedBox(height: 4),
                                            Text(
                                              item.formattedTs,
                                              style: Theme.of(context).textTheme.bodySmall?.copyWith(color: AppTheme.muted),
                                            ),
                                            const SizedBox(height: 8),
                                            Wrap(
                                              spacing: 8,
                                              runSpacing: 8,
                                              children: [
                                                if (item.riskCategory?.isNotEmpty == true)
                                                  Ui.badge(text: item.riskCategory!, color: AppTheme.warning, icon: Icons.shield),
                                                Ui.badge(
                                                  text: (item.lat != null && item.lon != null) ? 'GPS' : 'без GPS',
                                                  color: (item.lat != null && item.lon != null) ? AppTheme.success : AppTheme.muted,
                                                  icon: Icons.location_on,
                                                ),
                                              ],
                                            ),
                                          ],
                                        ),
                                      ),
                                      const SizedBox(width: 10),
                                      const Icon(Icons.chevron_right),
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
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(color: AppTheme.danger),
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

enum _FilterMode { all, withRisk, noRisk }

class _FilterChipGroup extends StatelessWidget {
  final _FilterMode value;
  final ValueChanged<_FilterMode> onChanged;

  const _FilterChipGroup({required this.value, required this.onChanged});

  @override
  Widget build(BuildContext context) {
    return Wrap(
      spacing: 8,
      children: [
        ChoiceChip(
          label: const Text('Все'),
          selected: value == _FilterMode.all,
          onSelected: (_) => onChanged(_FilterMode.all),
        ),
        ChoiceChip(
          label: const Text('С риском'),
          selected: value == _FilterMode.withRisk,
          onSelected: (_) => onChanged(_FilterMode.withRisk),
        ),
        ChoiceChip(
          label: const Text('Без риска'),
          selected: value == _FilterMode.noRisk,
          onSelected: (_) => onChanged(_FilterMode.noRisk),
        ),
      ],
    );
  }
}

class _HistoryItem {
  final String analysisId;
  final String species;
  final double? height;
  final double? crown;
  final double? trunk;
  final double? scale;
  final double? riskIndex;
  final String? riskCategory;

  final double? lat;
  final double? lon;
  final String? address;

  final String imageBase64;
  final DateTime timestamp;

  _HistoryItem({
    required this.analysisId,
    required this.species,
    required this.imageBase64,
    required this.timestamp,
    this.height,
    this.crown,
    this.trunk,
    this.scale,
    this.riskIndex,
    this.riskCategory,
    this.lat,
    this.lon,
    this.address,
  });

  String get uniqueKey {
    // Prefer analysisId if present, otherwise timestamp+species
    if (analysisId.isNotEmpty) return analysisId;
    return '${timestamp.toIso8601String()}|$species|${lat ?? ''}|${lon ?? ''}';
  }

  String get formattedTs {
    String two(int v) => v.toString().padLeft(2, '0');
    return '${two(timestamp.day)}.${two(timestamp.month)}.${timestamp.year} ${two(timestamp.hour)}:${two(timestamp.minute)}';
  }

  factory _HistoryItem.fromJson(Map<String, dynamic> json) {
    return _HistoryItem(
      analysisId: (json['analysisId'] ?? '') as String,
      species: (json['species'] ?? 'Неизвестно') as String,
      height: (json['height'] as num?)?.toDouble(),
      crown: (json['crown'] as num?)?.toDouble(),
      trunk: (json['trunk'] as num?)?.toDouble(),
      scale: (json['scale'] as num?)?.toDouble(),
      riskIndex: (json['riskIndex'] as num?)?.toDouble(),
      riskCategory: json['riskCategory'] as String?,
      lat: (json['lat'] as num?)?.toDouble(),
      lon: (json['lon'] as num?)?.toDouble(),
      address: json['address'] as String?,
      imageBase64: (json['imageBase64'] ?? '') as String,
      timestamp: DateTime.tryParse((json['timestamp'] ?? '') as String) ?? DateTime.now(),
    );
  }
}
