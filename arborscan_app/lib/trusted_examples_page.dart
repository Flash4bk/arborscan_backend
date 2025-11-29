import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

/// Адрес твоего бэкенда
const String kBaseUrl =
    'https://arborscanbackend-production.up.railway.app';

class TrustedExamplesPage extends StatefulWidget {
  const TrustedExamplesPage({super.key});

  @override
  State<TrustedExamplesPage> createState() => _TrustedExamplesPageState();
}

class _TrustedExamplesPageState extends State<TrustedExamplesPage> {
  late Future<List<TrustedExample>> _future;
  bool _onlyNeedsReview = false;

  @override
  void initState() {
    super.initState();
    _future = _loadExamples();
  }

  Future<List<TrustedExample>> _loadExamples() async {
    final uri = Uri.parse(
      '$kBaseUrl/trusted-examples?limit=100'
      '${_onlyNeedsReview ? '&require_review=true' : ''}',
    );

    final resp = await http.get(uri);
    if (resp.statusCode != 200) {
      throw Exception(
          'Ошибка загрузки доверенных примеров: ${resp.statusCode}');
    }

    final data = jsonDecode(resp.body) as Map<String, dynamic>;
    final list = (data['items'] as List<dynamic>? ?? [])
        .cast<Map<String, dynamic>>();

    return list.map(TrustedExample.fromJson).toList();
  }

  void _reload() {
    setState(() {
      _future = _loadExamples();
    });
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Доверенные примеры'),
        actions: [
          IconButton(
            onPressed: _reload,
            icon: const Icon(Icons.refresh),
          ),
        ],
      ),
      body: Column(
        children: [
          Padding(
            padding:
                const EdgeInsets.symmetric(horizontal: 16.0, vertical: 8),
            child: Row(
              children: [
                Expanded(
                  child: Text(
                    'Показывать только требующие\nручной проверки',
                    style: theme.textTheme.bodyMedium,
                  ),
                ),
                Switch(
                  value: _onlyNeedsReview,
                  onChanged: (v) {
                    setState(() {
                      _onlyNeedsReview = v;
                      _future = _loadExamples();
                    });
                  },
                ),
              ],
            ),
          ),
          const Divider(height: 1),
          Expanded(
            child: FutureBuilder<List<TrustedExample>>(
              future: _future,
              builder: (context, snapshot) {
                if (snapshot.connectionState == ConnectionState.waiting) {
                  return const Center(child: CircularProgressIndicator());
                }
                if (snapshot.hasError) {
                  return Center(
                    child: Padding(
                      padding: const EdgeInsets.all(16.0),
                      child: Text(
                        snapshot.error.toString(),
                        textAlign: TextAlign.center,
                      ),
                    ),
                  );
                }

                final items = snapshot.data ?? [];
                if (items.isEmpty) {
                  return const Center(
                    child: Text(
                      'Пока нет сохранённых доверенных примеров.',
                      textAlign: TextAlign.center,
                    ),
                  );
                }

                return ListView.separated(
                  padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
                  itemCount: items.length,
                  separatorBuilder: (_, __) => const SizedBox(height: 8),
                  itemBuilder: (context, index) {
                    final item = items[index];
                    return _buildItemCard(context, item);
                  },
                );
              },
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildItemCard(BuildContext context, TrustedExample item) {
    final theme = Theme.of(context);

    Color statusColor;
    String statusText;

    if (item.needsManualReview == true) {
      statusColor = const Color(0xFFFFE1E1);
      statusText = 'Требует ручной проверки';
    } else {
      statusColor = const Color(0xFFD9F5DC);
      statusText = 'Подтверждённый пример';
    }

    final chips = <Widget>[];

    if (item.hasUserMask == true) {
      chips.add(_buildChip('Маска', Icons.brush));
    }
    if (item.stickOk == false || item.stickOk == true) {
      chips.add(_buildChip('Палка', Icons.straighten));
    }
    if (item.paramsOk == false || item.paramsOk == true) {
      chips.add(_buildChip('Параметры', Icons.analytics_outlined));
    }

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Icon(Icons.park, size: 20),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    item.species ?? 'Неизвестный вид',
                    style: theme.textTheme.titleMedium
                        ?.copyWith(fontWeight: FontWeight.w600),
                  ),
                ),
                if (item.trustScore != null)
                  Text(
                    'trust ${item.trustScore!.toStringAsFixed(2)}',
                    style: theme.textTheme.labelSmall,
                  ),
              ],
            ),
            const SizedBox(height: 8),
            Container(
              padding:
                  const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
              decoration: BoxDecoration(
                color: statusColor,
                borderRadius: BorderRadius.circular(999),
              ),
              child: Text(
                statusText,
                style: theme.textTheme.labelMedium,
              ),
            ),
            const SizedBox(height: 8),
            Wrap(
              spacing: 6,
              runSpacing: -4,
              children: [
                if (chips.isNotEmpty) ...chips,
                if (item.useForTraining == true)
                  _buildChip('В обучении', Icons.school_outlined),
              ],
            ),
            const SizedBox(height: 8),
            Text(
              'ID: ${item.analysisId}',
              style: theme.textTheme.labelSmall
                  ?.copyWith(color: Colors.black54),
            ),
            if (item.createdAt != null)
              Text(
                'Создано: ${item.createdAt}',
                style: theme.textTheme.labelSmall
                    ?.copyWith(color: Colors.black54),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildChip(String text, IconData icon) {
    return Chip(
      labelPadding: const EdgeInsets.symmetric(horizontal: 4),
      avatar: Icon(icon, size: 16),
      label: Text(text),
      materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
    );
  }
}

class TrustedExample {
  final String analysisId;
  final String? species;
  final double? trustScore;
  final String? createdAt;

  final bool? treeOk;
  final bool? stickOk;
  final bool? paramsOk;
  final bool? speciesOk;

  final bool? hasUserMask;
  final bool? useForTraining;
  final bool? needsManualReview;

  TrustedExample({
    required this.analysisId,
    this.species,
    this.trustScore,
    this.createdAt,
    this.treeOk,
    this.stickOk,
    this.paramsOk,
    this.speciesOk,
    this.hasUserMask,
    this.useForTraining,
    this.needsManualReview,
  });

  factory TrustedExample.fromJson(Map<String, dynamic> json) {
    double? _toDouble(dynamic v) =>
        v == null ? null : (v as num).toDouble();

    bool? _toBool(dynamic v) =>
        v == null ? null : (v as bool);

    return TrustedExample(
      analysisId: json['analysis_id'] as String? ?? '',
      species: json['species'] as String?,
      trustScore: _toDouble(json['trust_score']),
      createdAt: json['created_at'] as String?,
      treeOk: _toBool(json['tree_ok']),
      stickOk: _toBool(json['stick_ok']),
      paramsOk: _toBool(json['params_ok']),
      speciesOk: _toBool(json['species_ok']),
      hasUserMask: _toBool(json['has_user_mask']),
      useForTraining: _toBool(json['use_for_training']),
      needsManualReview: _toBool(json['needs_manual_review']),
    );
  }
}
