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
  List<TrainingEvent> _events = const [];
  List<int> _models = const [];
  int? _selectedVersion;

  @override
  void initState() {
    super.initState();
    _refresh();
  }

  Future<void> _refresh() async {
    setState(() {
      _loading = true;
      _error = null;
    });

    try {
      final results = await Future.wait([
        _service.getTrainingStatus(),
        _service.getTrainingEvents(limit: 15),
        _service.getModels(),
      ]);

      final status = results[0] as TrainingStatus;
      final events = results[1] as List<TrainingEvent>;
      final modelsResp = results[2] as ModelsResponse;

      // Pick a stable selection: active -> first -> null
      final models = modelsResp.models;
      int? selection = _selectedVersion;
      if (selection == null || !models.contains(selection)) {
        selection = modelsResp.activeModelVersion;
      }
      if (selection == null || !models.contains(selection)) {
        selection = models.isNotEmpty ? models.first : null;
      }

      setState(() {
        _status = status;
        _events = events;
        _models = models;
        _selectedVersion = selection;
        _loading = false;
      });
    } catch (e) {
      setState(() {
        _error = e.toString();
        _loading = false;
      });
    }
  }

  Future<void> _setActive() async {
    final v = _selectedVersion;
    if (v == null) return;

    setState(() => _error = null);
    try {
      await _service.setActiveModel(v);
      await _refresh();
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Активная модель переключена на v$v')),
        );
      }
    } catch (e) {
      setState(() => _error = e.toString());
    }
  }

  Future<void> _requestTraining() async {
    setState(() => _error = null);
    try {
      await _service.requestTraining();
      await _refresh();
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Запрос на обучение отправлен')),
        );
      }
    } catch (e) {
      setState(() => _error = e.toString());
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Admin Panel'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: _loading ? null : _refresh,
          ),
        ],
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : RefreshIndicator(
              onRefresh: _refresh,
              child: ListView(
                padding: const EdgeInsets.all(16),
                children: [
                  if (_error != null) ...[
                    _ErrorBanner(message: _error!),
                    const SizedBox(height: 12),
                  ],

                  _Card(
                    title: 'Статус обучения',
                    child: _StatusBlock(status: _status),
                  ),
                  const SizedBox(height: 16),

                  _Card(
                    title: 'Переключение модели',
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        DropdownButtonFormField<int>(
                          value: (_selectedVersion != null && _models.contains(_selectedVersion))
                              ? _selectedVersion
                              : null,
                          items: _models
                              .map(
                                (v) => DropdownMenuItem<int>(
                                  value: v,
                                  child: Text('Версия $v'),
                                ),
                              )
                              .toList(),
                          decoration: const InputDecoration(
                            labelText: 'Версия модели',
                            border: OutlineInputBorder(),
                          ),
                          onChanged: (v) => setState(() => _selectedVersion = v),
                        ),
                        const SizedBox(height: 12),
                        ElevatedButton.icon(
                          onPressed: _models.isEmpty || _selectedVersion == null ? null : _setActive,
                          icon: const Icon(Icons.swap_horiz),
                          label: const Text('Сделать активной'),
                          style: ElevatedButton.styleFrom(
                            minimumSize: const Size.fromHeight(48),
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 16),

                  _Card(
                    title: 'Обучение',
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        const Text(
                          'Запуск обучения берёт подтверждённые примеры из Supabase и формирует новую версию модели.',
                        ),
                        const SizedBox(height: 12),
                        ElevatedButton.icon(
                          onPressed: _requestTraining,
                          icon: const Icon(Icons.play_arrow),
                          label: const Text('Запросить обучение'),
                          style: ElevatedButton.styleFrom(
                            minimumSize: const Size.fromHeight(48),
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 16),

                  _Card(
                    title: 'Лог обучения',
                    child: _TrainingLog(events: _events),
                  ),
                ],
              ),
            ),
    );
  }
}

class _Card extends StatelessWidget {
  final String title;
  final Widget child;

  const _Card({required this.title, required this.child});

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 0,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              title,
              style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700),
            ),
            const SizedBox(height: 12),
            child,
          ],
        ),
      ),
    );
  }
}

class _ErrorBanner extends StatelessWidget {
  final String message;

  const _ErrorBanner({required this.message});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.red.withOpacity(0.10),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.red.withOpacity(0.25)),
      ),
      child: Text(
        message,
        style: const TextStyle(color: Colors.red),
      ),
    );
  }
}

class _StatusBlock extends StatelessWidget {
  final TrainingStatus? status;

  const _StatusBlock({required this.status});

  @override
  Widget build(BuildContext context) {
    if (status == null) {
      return const Text('Нет данных');
    }

    final s = status!;
    String dash(int? v) => v == null ? '—' : 'v$v';

    return Column(
      children: [
        _StatusRow(label: 'Обучение сейчас', value: s.isTraining ? 'Да' : 'Нет'),
        const SizedBox(height: 8),
        _StatusRow(label: 'Активная модель', value: dash(s.activeModelVersion)),
        const SizedBox(height: 8),
        _StatusRow(label: 'Последняя обученная', value: dash(s.lastTrainedVersion ?? s.lastTrained)),
      ],
    );
  }
}

class _StatusRow extends StatelessWidget {
  final String label;
  final String value;

  const _StatusRow({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Expanded(child: Text(label)),
        Text(value, style: const TextStyle(fontWeight: FontWeight.w600)),
      ],
    );
  }
}

class _TrainingLog extends StatelessWidget {
  final List<TrainingEvent> events;

  const _TrainingLog({required this.events});

  @override
  Widget build(BuildContext context) {
    if (events.isEmpty) {
      return const Text(
        'События пока отсутствуют (или endpoint /admin/training-events ещё не развёрнут).',
      );
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: events.map((e) {
        final metaMap = e.meta;
        final meta = metaMap.isEmpty ? '' : '  $metaMap';
        return Padding(
          padding: const EdgeInsets.only(bottom: 8),
          child: Text('${e.ts}  ${e.level}: ${e.message}$meta'),
        );
      }).toList(),
    );
  }
}
