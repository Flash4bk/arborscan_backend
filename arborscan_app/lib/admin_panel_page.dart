
import 'package:flutter/material.dart';
import 'admin_service.dart';

class AdminPanelPage extends StatefulWidget {
  final String baseUrl;

  const AdminPanelPage({super.key, required this.baseUrl});

  @override
  State<AdminPanelPage> createState() => _AdminPanelPageState();
}

class _AdminPanelPageState extends State<AdminPanelPage> {
  late final AdminService _service = AdminService(baseUrl: widget.baseUrl);

  bool _loading = true;
  String? _error;

  TrainingStatus? _status;
  List<ModelInfo> _models = const [];
  ModelInfo? _selected;

  List<TrainingEvent> _events = const [];
  String? _eventsError;

  @override
  void initState() {
    super.initState();
    _loadAll();
  }

  Future<void> _loadAll() async {
    setState(() {
      _loading = true;
      _error = null;
      _eventsError = null;
    });

    try {
      final results = await Future.wait([
        _service.fetchTrainingStatus(),
        _service.fetchModels(),
        _service.fetchTrainingEvents(limit: 15),
      ]);

      final status = results[0] as TrainingStatus;
      final models = results[1] as List<ModelInfo>;
      final events = results[2] as List<TrainingEvent>;

      ModelInfo? selected = _selected;
      if (selected == null || !models.any((m) => m.version == selected!.version)) {
        selected = models.firstWhere((m) => m.isActive, orElse: () => models.isNotEmpty ? models.first : const ModelInfo(version: 'v1', isActive: true));
      }

      setState(() {
        _status = status;
        _models = models;
        _selected = selected;
        _events = events;
        _loading = false;
      });
    } catch (e) {
      setState(() {
        _error = e.toString();
        _loading = false;
      });
    }
  }

  Future<void> _makeActive() async {
    final sel = _selected;
    if (sel == null) return;

    setState(() {
      _loading = true;
      _error = null;
    });

    try {
      await _service.setActiveModel(sel.version);
      await _loadAll();
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Активная модель: ${sel.version}')),
      );
    } catch (e) {
      setState(() {
        _loading = false;
        _error = e.toString();
      });
    }
  }

  Future<void> _requestTraining() async {
    setState(() {
      _loading = true;
      _error = null;
    });

    try {
      await _service.requestRetrain();
      await _loadAll();
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Запрос обучения отправлен.')),
      );
    } catch (e) {
      setState(() {
        _loading = false;
        _error = e.toString();
      });
    }
  }

  String _fmtDt(DateTime dt) {
    String two(int n) => n.toString().padLeft(2, '0');
    return '${two(dt.day)}.${two(dt.month)}.${dt.year} ${two(dt.hour)}:${two(dt.minute)}:${two(dt.second)}';
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
            icon: const Icon(Icons.refresh),
            tooltip: 'Обновить',
            onPressed: _loading ? null : _loadAll,
          ),
        ],
      ),
      body: Stack(
        children: [
          ListView(
            padding: const EdgeInsets.fromLTRB(16, 16, 16, 24),
            children: [
              if (_error != null)
                _ErrorBanner(text: _error!)
              else
                const SizedBox.shrink(),

              _SectionCard(
                title: 'Статус обучения',
                child: _status == null
                    ? Text('Нет данных', style: theme.textTheme.bodyMedium)
                    : Column(
                        children: [
                          _StatusRow(
                            label: 'Обучение сейчас',
                            value: (_status!.isTraining ? 'Да' : 'Нет'),
                          ),
                          const SizedBox(height: 8),
                          _StatusRow(
                            label: 'Активная модель',
                            value: _status!.activeModel ?? '—',
                          ),
                          const SizedBox(height: 8),
                          _StatusRow(
                            label: 'Последняя обученная',
                            value: _status!.lastTrainedModel ?? '—',
                          ),
                          if ((_status!.message ?? '').isNotEmpty) ...[
                            const SizedBox(height: 12),
                            Align(
                              alignment: Alignment.centerLeft,
                              child: Text(
                                _status!.message!,
                                style: theme.textTheme.bodySmall?.copyWith(
                                  color: cs.onSurfaceVariant,
                                ),
                              ),
                            ),
                          ],
                        ],
                      ),
              ),

              const SizedBox(height: 12),

              _SectionCard(
                title: 'Переключение модели',
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    DropdownButtonFormField<ModelInfo>(
                      value: _selected,
                      decoration: const InputDecoration(
                        labelText: 'Версия модели',
                        border: OutlineInputBorder(),
                      ),
                      items: _models
                          .map(
                            (m) => DropdownMenuItem<ModelInfo>(
                              value: m,
                              child: Text(
                                'Версия ${m.version}${m.isActive ? ' (активна)' : ''}',
                              ),
                            ),
                          )
                          .toList(),
                      onChanged: _loading
                          ? null
                          : (v) {
                              setState(() => _selected = v);
                            },
                    ),
                    const SizedBox(height: 12),
                    SizedBox(
                      width: double.infinity,
                      child: FilledButton.icon(
                        onPressed: _loading ? null : _makeActive,
                        icon: const Icon(Icons.swap_horiz),
                        label: const Text('Сделать активной'),
                      ),
                    ),
                  ],
                ),
              ),

              const SizedBox(height: 12),

              _SectionCard(
                title: 'Обучение',
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
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
                        onPressed: _loading ? null : _requestTraining,
                        icon: const Icon(Icons.play_arrow),
                        label: const Text('Запросить обучение'),
                      ),
                    ),
                  ],
                ),
              ),

              const SizedBox(height: 12),

              _SectionCard(
                title: 'Лог обучения',
                subtitle: 'Показывает последние события: запрос обучения, переключение модели, статусы (если сервер поддерживает).',
                child: _buildEventsCard(context),
              ),
            ],
          ),

          if (_loading)
            Container(
              color: cs.onSurface.withOpacity(0.08),
              child: const Center(child: CircularProgressIndicator()),
            ),
        ],
      ),
    );
  }

  Widget _buildEventsCard(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;

    if (_eventsError != null) {
      return Text(_eventsError!, style: theme.textTheme.bodySmall?.copyWith(color: cs.error));
    }

    if (_events.isEmpty) {
      return Text(
        'События пока отсутствуют (или endpoint /admin/training-events ещё не развёрнут).',
        style: theme.textTheme.bodySmall?.copyWith(color: cs.onSurfaceVariant),
      );
    }

    return Column(
      children: _events.map((e) {
        return Padding(
          padding: const EdgeInsets.only(bottom: 10),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Container(
                width: 10,
                height: 10,
                margin: const EdgeInsets.only(top: 4),
                decoration: BoxDecoration(
                  color: _levelColor(e.level),
                  shape: BoxShape.circle,
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      e.message,
                      style: theme.textTheme.bodyMedium?.copyWith(fontWeight: FontWeight.w600),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      _fmtDt(e.ts),
                      style: theme.textTheme.bodySmall?.copyWith(color: cs.onSurfaceVariant),
                    ),
                  ],
                ),
              ),
            ],
          ),
        );
      }).toList(),
    );
  }

  Color _levelColor(String level) {
    switch (level.toLowerCase()) {
      case 'error':
        return const Color(0xFFB71C1C);
      case 'warning':
        return const Color(0xFF8D6E00);
      case 'success':
        return const Color(0xFF1B5E20);
      default:
        return const Color(0xFF1565C0);
    }
  }
}

class _SectionCard extends StatelessWidget {
  final String title;
  final String? subtitle;
  final Widget child;

  const _SectionCard({
    required this.title,
    required this.child,
    this.subtitle,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;

    return Card(
      margin: EdgeInsets.zero,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              title,
              style: theme.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700),
            ),
            if (subtitle != null) ...[
              const SizedBox(height: 6),
              Text(
                subtitle!,
                style: theme.textTheme.bodySmall?.copyWith(color: cs.onSurfaceVariant),
              ),
            ],
            const SizedBox(height: 12),
            child,
          ],
        ),
      ),
    );
  }
}

class _StatusRow extends StatelessWidget {
  final String label;
  final String value;

  const _StatusRow({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;

    return Row(
      children: [
        Expanded(
          child: Text(
            label,
            style: theme.textTheme.bodyMedium?.copyWith(color: cs.onSurfaceVariant),
          ),
        ),
        const SizedBox(width: 12),
        Text(value, style: theme.textTheme.bodyMedium?.copyWith(fontWeight: FontWeight.w700)),
      ],
    );
  }
}

class _ErrorBanner extends StatelessWidget {
  final String text;
  const _ErrorBanner({required this.text});

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0xFFFFE1E1),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Text(
        text,
        style: const TextStyle(color: Color(0xFFB71C1C)),
      ),
    );
  }
}
