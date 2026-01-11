import 'package:flutter/material.dart';

import 'admin_service.dart';

class AdminPanelPage extends StatefulWidget {
  const AdminPanelPage({super.key});

  @override
  State<AdminPanelPage> createState() => _AdminPanelPageState();
}

class _AdminPanelPageState extends State<AdminPanelPage> {
  bool _loading = true;

  // values from backend
  int? _activeModelVersion;
  int? _lastModelVersion;
  bool _trainingInProgress = false;

  // local UI
  int? _selectedVersion;
  String? _error;

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
      final status = await AdminService.getTrainingStatus();
      final active = (status['active_model_version'] as num?)?.toInt();
      final last = (status['last_model_version'] as num?)?.toInt();
      final inProgress = status['training_in_progress'] == true;

      setState(() {
        _activeModelVersion = active;
        _lastModelVersion = last;
        _trainingInProgress = inProgress;
        _selectedVersion = active ?? last;
      });
    } catch (e) {
      setState(() {
        _error = 'Не удалось загрузить статус: $e';
      });
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  Future<void> _setActiveModel(int version) async {
    try {
      await AdminService.setActiveModel(version);
      if (!mounted) return;
      setState(() => _activeModelVersion = version);
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Активная модель: v$version')),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Не удалось переключить модель')),
      );
    }
  }

  Future<void> _requestRetrain() async {
    try {
      await AdminService.requestRetrain();
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Запрос на обучение отправлен')),
      );
      await _load();
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Ошибка запроса обучения: $e')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Admin Panel'),
        actions: [
          IconButton(
            tooltip: 'Обновить',
            icon: const Icon(Icons.refresh),
            onPressed: _load,
          ),
        ],
      ),
      body: SafeArea(
        child: _loading
            ? const Center(child: CircularProgressIndicator())
            : ListView(
                padding: const EdgeInsets.fromLTRB(16, 16, 16, 24),
                children: [
                  if (_error != null) ...[
                    Container(
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: const Color(0xFFFFE1E1),
                        borderRadius: BorderRadius.circular(16),
                      ),
                      child: Text(
                        _error!,
                        style: theme.textTheme.bodySmall?.copyWith(
                          color: const Color(0xFFB71C1C),
                        ),
                      ),
                    ),
                    const SizedBox(height: 12),
                  ],

                  // Status card
                  Card(
                    margin: EdgeInsets.zero,
                    child: Padding(
                      padding: const EdgeInsets.all(16),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'Статус обучения',
                            style: theme.textTheme.titleMedium?.copyWith(
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                          const SizedBox(height: 8),
                          _StatusRow(
                            label: 'Обучение сейчас',
                            value: _trainingInProgress ? 'Да' : 'Нет',
                            valueColor: _trainingInProgress
                                ? cs.tertiary
                                : cs.onSurfaceVariant,
                          ),
                          _StatusRow(
                            label: 'Активная модель',
                            value: _activeModelVersion != null
                                ? 'v${_activeModelVersion}'
                                : '—',
                          ),
                          _StatusRow(
                            label: 'Последняя обученная',
                            value: _lastModelVersion != null
                                ? 'v${_lastModelVersion}'
                                : '—',
                          ),
                        ],
                      ),
                    ),
                  ),

                  const SizedBox(height: 16),

                  // Model switch card
                  Card(
                    margin: EdgeInsets.zero,
                    child: Padding(
                      padding: const EdgeInsets.all(16),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'Переключение модели',
                            style: theme.textTheme.titleMedium?.copyWith(
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                          const SizedBox(height: 8),
                          DropdownButtonFormField<int>(
                            value: _selectedVersion,
                            decoration: const InputDecoration(
                              labelText: 'Версия модели',
                              border: OutlineInputBorder(),
                            ),
                            items: [
                              if (_lastModelVersion != null)
                                DropdownMenuItem(
                                  value: _lastModelVersion,
                                  child: Text('v${_lastModelVersion} (последняя)'),
                                ),
                              if (_activeModelVersion != null &&
                                  _activeModelVersion != _lastModelVersion)
                                DropdownMenuItem(
                                  value: _activeModelVersion,
                                  child: Text('v${_activeModelVersion} (активная)'),
                                ),
                              // fallback for manual selection by common versions
                              for (final v in <int>[1, 2, 3, 4, 5])
                                if (v != _activeModelVersion &&
                                    v != _lastModelVersion)
                                  DropdownMenuItem(
                                    value: v,
                                    child: Text('v$v'),
                                  ),
                            ],
                            onChanged: (v) => setState(() => _selectedVersion = v),
                          ),
                          const SizedBox(height: 12),
                          SizedBox(
                            width: double.infinity,
                            child: FilledButton.icon(
                              onPressed: _selectedVersion == null
                                  ? null
                                  : () => _setActiveModel(_selectedVersion!),
                              icon: const Icon(Icons.swap_horiz),
                              label: const Text('Сделать активной'),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),

                  const SizedBox(height: 16),

                  // Retrain card
                  Card(
                    margin: EdgeInsets.zero,
                    child: Padding(
                      padding: const EdgeInsets.all(16),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'Обучение',
                            style: theme.textTheme.titleMedium?.copyWith(
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                          const SizedBox(height: 8),
                          Text(
                            'Запуск обучения берёт подтверждённые примеры из Supabase и формирует новую версию модели.',
                            style: theme.textTheme.bodySmall?.copyWith(
                              color: cs.onSurfaceVariant,
                            ),
                          ),
                          const SizedBox(height: 12),
                          SizedBox(
                            width: double.infinity,
                            child: FilledButton.icon(
                              onPressed: _trainingInProgress ? null : _requestRetrain,
                              icon: const Icon(Icons.play_arrow_rounded),
                              label: const Text('Запросить обучение'),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ],
              ),
      ),
    );
  }
}

class _StatusRow extends StatelessWidget {
  final String label;
  final String value;
  final Color? valueColor;

  const _StatusRow({
    required this.label,
    required this.value,
    this.valueColor,
  });

  @override
  
  Widget _buildEventsCard() {
    if (_events.isEmpty) {
      return const Card(
        child: Padding(
          padding: EdgeInsets.all(16),
          child: Text("События обучения пока отсутствуют."),
        ),
      );
    }

    String fmtTs(String? iso) {
      if (iso == null) return "";
      try {
        final dt = DateTime.parse(iso).toLocal();
        return "${dt.day.toString().padLeft(2, '0')}.${dt.month.toString().padLeft(2, '0')}.${dt.year} "
            "${dt.hour.toString().padLeft(2, '0')}:${dt.minute.toString().padLeft(2, '0')}:${dt.second.toString().padLeft(2, '0')}";
      } catch (_) {
        return iso;
      }
    }

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              "Последние события",
              style: TextStyle(fontWeight: FontWeight.w700),
            ),
            const SizedBox(height: 8),
            ..._events.take(15).map((e) {
              final ts = fmtTs(e["ts"] as String?);
              final type = (e["type"] ?? "").toString();
              final msg = (e["message"] ?? "").toString();
              return Padding(
                padding: const EdgeInsets.symmetric(vertical: 6),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Expanded(
                      child: Text(
                        "[$type] $msg",
                        style: const TextStyle(fontSize: 13),
                      ),
                    ),
                    const SizedBox(width: 8),
                    Text(
                      ts,
                      style: const TextStyle(fontSize: 11, color: Colors.black54),
                    ),
                  ],
                ),
              );
            }),
          ],
        ),
      ),
    );
  }

Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          Expanded(
            child: Text(
              label,
              style: theme.textTheme.bodySmall?.copyWith(
                color: cs.onSurfaceVariant,
              ),
            ),
          ),
          Text(
            value,
            style: theme.textTheme.bodySmall?.copyWith(
              fontWeight: FontWeight.w700,
              color: valueColor ?? cs.onSurface,
            ),
          ),
        ],
      ),
    );
  }
}
