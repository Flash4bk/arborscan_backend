import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

import 'app_theme.dart';
import 'map_page.dart';
import 'api_config.dart';
import 'analysis_report_page.dart'; // Нужно для перехода в отчет

class HistoryTabPage extends StatefulWidget {
  const HistoryTabPage({super.key});

  @override
  State<HistoryTabPage> createState() => _HistoryTabPageState();
}

class _HistoryTabPageState extends State<HistoryTabPage> {
  static const String _historyKey = 'arborscan_history';
  static const String _tokenKey = 'arborscan_auth_token';

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
      final prefs = await SharedPreferences.getInstance();
      final list = prefs.getStringList(_historyKey) ?? const [];

      final items = <_HistoryItem>[];
      for (final s in list) {
        try {
          final m = jsonDecode(s) as Map<String, dynamic>;
          items.add(_HistoryItem.fromJson(m));
        } catch (_) {}
      }

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
    var out = List<_HistoryItem>.of(_all);

    if (q.isNotEmpty) {
      out = out.where((e) {
        final hay = [e.species, e.address ?? '', e.riskCategory ?? ''].join(' ').toLowerCase();
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
    final newList = <String>[];
    for (final s in list) {
      try {
        final m = jsonDecode(s) as Map<String, dynamic>;
        final other = _HistoryItem.fromJson(m);
        if (other.uniqueKey != item.uniqueKey) {
          newList.add(s);
        }
      } catch (_) {
        newList.add(s);
      }
    }
    await prefs.setStringList(_historyKey, newList);
    await _load();

    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Запись удалена')));
  }

  // --- ИСПРАВЛЕНИЕ: СКАЧИВАНИЕ ПОЛНОГО ОТЧЕТА С СЕРВЕРА ---
  Future<void> _openFullReport(_HistoryItem item) async {
    if (item.analysisId.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Этот анализ не сохранен на сервере.')));
      return;
    }

    // Показываем загрузку
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (ctx) => const Center(child: CircularProgressIndicator()),
    );

    try {
      final prefs = await SharedPreferences.getInstance();
      final token = prefs.getString(_tokenKey) ?? '';
      
      final uri = Uri.parse('${ApiConfig.baseUrl}/analyses/${item.analysisId}').replace(queryParameters: {'token': token});
      final res = await http.get(uri).timeout(const Duration(seconds: 15));
      
      if (!mounted) return;
      Navigator.pop(context); // Закрываем крутилку

      if (res.statusCode != 200) {
        throw Exception('Не удалось загрузить отчет: ${res.statusCode}');
      }

      final data = jsonDecode(utf8.decode(res.bodyBytes)) as Map<String, dynamic>;
      final analysisRaw = data['analysis'] as Map<String, dynamic>?;

      if (analysisRaw != null) {
        Navigator.of(context).push(
          MaterialPageRoute(
            builder: (_) => AnalysisReportPageV2.fromRawResult(
              raw: analysisRaw,
              annotatedImageBytes: _tryDecodeImageB64(analysisRaw['annotated_image_base64']),
            ),
          ),
        );
      }
    } catch (e) {
      if (mounted) {
        Navigator.pop(context); // Закрываем крутилку
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Ошибка загрузки: $e')));
      }
    }
  }

  Uint8List? _tryDecodeImageB64(String? b64) {
    if (b64 == null) return null;
    var s = b64.trim();
    if (s.isEmpty) return null;
    final comma = s.indexOf(',');
    if (s.startsWith('data:') && comma != -1) {
      s = s.substring(comma + 1);
    }
    try {
      return base64.decode(s);
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
        title: const Text('ИСТОРИЯ', style: TextStyle(letterSpacing: 1.5)),
        actions: [
          IconButton(tooltip: 'Обновить', icon: const Icon(Icons.refresh, color: AppTheme.primary), onPressed: _load),
          IconButton(
            tooltip: 'Очистить всё',
            icon: const Icon(Icons.delete_outline, color: AppTheme.danger),
            onPressed: _all.isEmpty ? null : _clearAll,
          ),
        ],
      ),
      body: _error != null
          ? _ErrorState(message: _error!, onRetry: _load)
          : Column(
              children: [
                Padding(
                  padding: const EdgeInsets.fromLTRB(16, 10, 16, 8),
                  child: Column(
                    children: [
                      GlassPanel(
                        padding: const EdgeInsets.all(18),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text('ЖУРНАЛ АНАЛИЗОВ', style: Theme.of(context).textTheme.titleLarge),
                            const SizedBox(height: 6),
                            Text(
                              'Поиск, фильтрация и генерация PDF-отчетов.',
                              style: Theme.of(context).textTheme.bodySmall,
                            ),
                            const SizedBox(height: 14),
                            Row(
                              children: [
                                Expanded(
                                  child: AppStatCard(
                                    value: '${_all.length}',
                                    label: 'ВСЕГО',
                                    icon: Icons.history_rounded,
                                    color: AppTheme.primary,
                                  ),
                                ),
                                const SizedBox(width: 12),
                                Expanded(
                                  child: AppStatCard(
                                    value: '${_all.where((e) => e.lat != null && e.lon != null).length}',
                                    label: 'С GPS',
                                    icon: Icons.location_on_rounded,
                                    color: AppTheme.warning,
                                  ),
                                ),
                              ],
                            ),
                          ],
                        ),
                      ),
                      const SizedBox(height: 12),
                      TextField(
                        controller: _searchCtrl,
                        decoration: const InputDecoration(
                          prefixIcon: Icon(Icons.search_rounded),
                          labelText: 'Поиск по виду, адресу или риску',
                        ),
                      ),
                      const SizedBox(height: 10),
                      SingleChildScrollView(
                        scrollDirection: Axis.horizontal,
                        child: Row(
                          children: [
                            _FilterChipGroup(
                              value: _filterMode,
                              onChanged: (v) {
                                setState(() => _filterMode = v);
                                _applyFilters();
                              },
                            ),
                            const SizedBox(width: 8),
                            FilterChip(
                              label: const Text('Только GPS'),
                              selectedColor: AppTheme.primary.withOpacity(0.3),
                              selected: _onlyWithGeo,
                              onSelected: (v) {
                                setState(() => _onlyWithGeo = v);
                                _applyFilters();
                              },
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
                Expanded(
                  child: _filtered.isEmpty
                      ? ListView(
                          padding: const EdgeInsets.fromLTRB(16, 6, 16, 16),
                          children: [
                            GlassPanel(
                              child: Row(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  const Icon(Icons.inbox_outlined, color: AppTheme.muted),
                                  const SizedBox(width: 12),
                                  Expanded(
                                    child: Text(
                                      'Ничего не найдено по текущим фильтрам.',
                                      style: Theme.of(context).textTheme.bodySmall,
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          ],
                        )
                      : ListView.separated(
                          padding: const EdgeInsets.fromLTRB(16, 6, 16, 120),
                          itemCount: _filtered.length,
                          separatorBuilder: (_, __) => const SizedBox(height: 12),
                          itemBuilder: (context, i) {
                            final item = _filtered[i];
                            // По клику теперь загружаем полный отчет с сервера
                            return _HistoryPreviewCard(
                              item: item,
                              onTap: () => _openFullReport(item),
                              onDelete: () => _deleteOne(item),
                            );
                          },
                        ),
                ),
              ],
            ),
    );
  }
}

class _HistoryPreviewCard extends StatelessWidget {
  final _HistoryItem item;
  final VoidCallback onTap;
  final VoidCallback onDelete;

  const _HistoryPreviewCard({
    required this.item,
    required this.onTap,
    required this.onDelete,
  });

  @override
  Widget build(BuildContext context) {
    final isHighRisk = item.riskCategory == 'высокий';
    final riskColor = isHighRisk ? AppTheme.danger : AppTheme.primary;

    return GlassPanel(
      padding: EdgeInsets.zero,
      onTap: onTap,
      border: Border.all(color: isHighRisk ? AppTheme.danger.withOpacity(0.5) : AppTheme.border),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Row(
          children: [
            Container(
              width: 50,
              height: 50,
              decoration: BoxDecoration(
                color: riskColor.withOpacity(0.15),
                shape: BoxShape.circle,
              ),
              child: Icon(Icons.park, color: riskColor),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    item.species.isNotEmpty ? item.species : 'Неизвестно',
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: Theme.of(context).textTheme.titleMedium,
                  ),
                  const SizedBox(height: 4),
                  Text(item.formattedTs, style: Theme.of(context).textTheme.bodySmall),
                  const SizedBox(height: 8),
                  Wrap(
                    spacing: 8,
                    runSpacing: 8,
                    children: [
                      Ui.badge(
                        text: item.riskCategory?.toUpperCase() ?? 'АНАЛИЗ',
                        color: riskColor,
                      ),
                      if (item.lat != null && item.lon != null)
                        Ui.badge(text: 'GPS', color: AppTheme.success, icon: Icons.location_on),
                    ],
                  ),
                ],
              ),
            ),
            IconButton(
              icon: const Icon(Icons.delete_outline, color: AppTheme.danger),
              onPressed: onDelete,
            )
          ],
        ),
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
        GlassPanel(
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
        AppActionButton(
          onTap: onRetry,
          icon: Icons.refresh_rounded,
          title: 'Повторить',
          subtitle: 'Перезагрузить историю',
          primary: true,
          compact: true,
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
          selectedColor: AppTheme.primary.withOpacity(0.3),
          onSelected: (_) => onChanged(_FilterMode.all)
        ),
        ChoiceChip(
          label: const Text('С риском'), 
          selected: value == _FilterMode.withRisk, 
          selectedColor: AppTheme.danger.withOpacity(0.3),
          onSelected: (_) => onChanged(_FilterMode.withRisk)
        ),
        ChoiceChip(
          label: const Text('Без риска'), 
          selected: value == _FilterMode.noRisk, 
          selectedColor: AppTheme.success.withOpacity(0.3),
          onSelected: (_) => onChanged(_FilterMode.noRisk)
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