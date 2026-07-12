import 'package:flutter/material.dart';

import 'admin_service.dart';
import 'training_dataset_page.dart';

class AdminPanelPage extends StatefulWidget {
  final String baseUrl;

  const AdminPanelPage({
    super.key,
    required this.baseUrl,
  });

  @override
  State<AdminPanelPage> createState() => _AdminPanelPageState();
}

class _AdminPanelPageState extends State<AdminPanelPage> {
  late final AdminService _service = AdminService(baseUrl: widget.baseUrl);

  bool _loading = true;
  bool _changingModel = false;
  bool _requestingTraining = false;
  String? _error;
  int? _errorStatusCode;

  AdminIdentity? _identity;
  TrainingStatus? _status;
  List<TrainingEvent> _events = const [];
  List<int> _models = const [];
  int? _selectedVersion;

  bool get _accessDenied =>
      _errorStatusCode == 401 || _errorStatusCode == 403;

  @override
  void initState() {
    super.initState();
    _refresh();
  }

  Future<void> _refresh() async {
    if (mounted) {
      setState(() {
        _loading = true;
        _error = null;
        _errorStatusCode = null;
      });
    }

    try {
      final identity = await _service.verifyAdminAccess();

      final results = await Future.wait<dynamic>([
        _service.getTrainingStatus(),
        _service.getTrainingEvents(limit: 15),
        _service.getModels(),
      ]);

      final status = results[0] as TrainingStatus;
      final events = results[1] as List<TrainingEvent>;
      final modelsResponse = results[2] as ModelsResponse;

      final models = modelsResponse.models;
      int? selection = _selectedVersion;
      if (selection == null || !models.contains(selection)) {
        selection = modelsResponse.activeModelVersion;
      }
      if (selection == null || !models.contains(selection)) {
        selection = models.isNotEmpty ? models.first : null;
      }

      if (!mounted) return;
      setState(() {
        _identity = identity;
        _status = status;
        _events = events;
        _models = models;
        _selectedVersion = selection;
        _loading = false;
      });
    } on AdminApiException catch (error) {
      if (!mounted) return;
      setState(() {
        _error = error.message;
        _errorStatusCode = error.statusCode;
        _loading = false;
      });
    } catch (error) {
      if (!mounted) return;
      setState(() {
        _error = error.toString();
        _errorStatusCode = null;
        _loading = false;
      });
    }
  }

  Future<void> _setActive() async {
    final version = _selectedVersion;
    if (version == null || _changingModel) return;

    setState(() {
      _changingModel = true;
      _error = null;
      _errorStatusCode = null;
    });

    try {
      await _service.setActiveModel(version);
      await _refresh();
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Активная модель переключена на v$version.')),
      );
    } on AdminApiException catch (error) {
      if (!mounted) return;
      setState(() {
        _error = error.message;
        _errorStatusCode = error.statusCode;
      });
    } catch (error) {
      if (!mounted) return;
      setState(() => _error = error.toString());
    } finally {
      if (mounted) setState(() => _changingModel = false);
    }
  }

  Future<void> _requestTraining() async {
    if (_requestingTraining) return;

    final confirmed = await showDialog<bool>(
      context: context,
      builder: (dialogContext) => AlertDialog(
        title: const Text('Запросить переобучение?'),
        content: const Text(
          'Сервер установит флаг переобучения. Worker начнёт работу, '
          'когда увидит запрос и доступные подтверждённые примеры.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(dialogContext, false),
            child: const Text('Отмена'),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(dialogContext, true),
            child: const Text('Запросить'),
          ),
        ],
      ),
    );
    if (confirmed != true) return;

    setState(() {
      _requestingTraining = true;
      _error = null;
      _errorStatusCode = null;
    });

    try {
      await _service.requestTraining();
      await _refresh();
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Запрос обучения отправлен.')),
      );
    } on AdminApiException catch (error) {
      if (!mounted) return;
      setState(() {
        _error = error.message;
        _errorStatusCode = error.statusCode;
      });
    } catch (error) {
      if (!mounted) return;
      setState(() => _error = error.toString());
    } finally {
      if (mounted) setState(() => _requestingTraining = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Панель администратора'),
        actions: [
          IconButton(
            tooltip: 'Обновить',
            icon: const Icon(Icons.refresh),
            onPressed: _loading ? null : _refresh,
          ),
        ],
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : _accessDenied
              ? _AccessDeniedView(
                  message: _error ?? 'Нет доступа.',
                  statusCode: _errorStatusCode,
                  onRetry: _refresh,
                )
              : RefreshIndicator(
                  onRefresh: _refresh,
                  child: ListView(
                    physics: const AlwaysScrollableScrollPhysics(),
                    padding: const EdgeInsets.all(16),
                    children: [
                      if (_error != null) ...[
                        _ErrorBanner(message: _error!),
                        const SizedBox(height: 12),
                      ],
                      if (_identity != null) ...[
                        _AdminIdentityCard(identity: _identity!),
                        const SizedBox(height: 16),
                      ],
                      _SectionCard(
                        title: 'Статус обучения',
                        child: _StatusBlock(status: _status),
                      ),
                      const SizedBox(height: 16),
                      _SectionCard(
                        title: 'Переключение модели',
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.stretch,
                          children: [
                            DropdownButtonFormField<int>(
                              value: _selectedVersion != null &&
                                      _models.contains(_selectedVersion)
                                  ? _selectedVersion
                                  : null,
                              items: _models
                                  .map(
                                    (version) => DropdownMenuItem<int>(
                                      value: version,
                                      child: Text('Версия $version'),
                                    ),
                                  )
                                  .toList(),
                              decoration: const InputDecoration(
                                labelText: 'Версия модели',
                                border: OutlineInputBorder(),
                              ),
                              onChanged: _changingModel
                                  ? null
                                  : (value) {
                                      setState(() => _selectedVersion = value);
                                    },
                            ),
                            const SizedBox(height: 12),
                            FilledButton.icon(
                              onPressed: _models.isEmpty ||
                                      _selectedVersion == null ||
                                      _changingModel
                                  ? null
                                  : _setActive,
                              icon: _changingModel
                                  ? const SizedBox(
                                      width: 18,
                                      height: 18,
                                      child: CircularProgressIndicator(
                                        strokeWidth: 2,
                                      ),
                                    )
                                  : const Icon(Icons.swap_horiz),
                              label: Text(
                                _changingModel
                                    ? 'Переключение...'
                                    : 'Сделать активной',
                              ),
                            ),
                          ],
                        ),
                      ),
                      const SizedBox(height: 16),
                      _SectionCard(
                        title: 'Обучение',
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.stretch,
                          children: [
                            const Text(
                              'Запрос использует подтверждённые примеры из '
                              'Supabase и передаётся отдельному worker.',
                            ),
                            const SizedBox(height: 12),
                            FilledButton.icon(
                              onPressed: _requestingTraining
                                  ? null
                                  : _requestTraining,
                              icon: _requestingTraining
                                  ? const SizedBox(
                                      width: 18,
                                      height: 18,
                                      child: CircularProgressIndicator(
                                        strokeWidth: 2,
                                      ),
                                    )
                                  : const Icon(Icons.play_arrow),
                              label: Text(
                                _requestingTraining
                                    ? 'Отправка...'
                                    : 'Запросить обучение',
                              ),
                            ),
                            const SizedBox(height: 10),
                            OutlinedButton.icon(
                              onPressed: () {
                                Navigator.of(context).push(
                                  MaterialPageRoute<void>(
                                    builder: (_) => TrainingDatasetPage(
                                      service: _service,
                                    ),
                                  ),
                                );
                              },
                              icon: const Icon(Icons.dataset_outlined),
                              label: const Text(
                                'Датасет для последующего обучения',
                              ),
                            ),
                          ],
                        ),
                      ),
                      const SizedBox(height: 16),
                      _SectionCard(
                        title: 'Журнал административных действий',
                        child: _TrainingLog(events: _events),
                      ),
                    ],
                  ),
                ),
    );
  }
}

class _AdminIdentityCard extends StatelessWidget {
  final AdminIdentity identity;

  const _AdminIdentityCard({required this.identity});

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Row(
          children: [
            const CircleAvatar(
              child: Icon(Icons.admin_panel_settings),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    identity.name.isEmpty ? 'Администратор' : identity.name,
                    style: Theme.of(context)
                        .textTheme
                        .titleMedium
                        ?.copyWith(fontWeight: FontWeight.w800),
                  ),
                  const SizedBox(height: 3),
                  Text(identity.email),
                  const SizedBox(height: 3),
                  const Text(
                    'Права подтверждены сервером',
                    style: TextStyle(color: Colors.green),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _AccessDeniedView extends StatelessWidget {
  final String message;
  final int? statusCode;
  final Future<void> Function() onRetry;

  const _AccessDeniedView({
    required this.message,
    required this.statusCode,
    required this.onRetry,
  });

  @override
  Widget build(BuildContext context) {
    final title = statusCode == 401
        ? 'Требуется вход'
        : 'Недостаточно прав';
    final icon = statusCode == 401
        ? Icons.login
        : Icons.admin_panel_settings_outlined;

    return Center(
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(24),
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 440),
          child: Card(
            child: Padding(
              padding: const EdgeInsets.all(24),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(icon, size: 54),
                  const SizedBox(height: 16),
                  Text(
                    title,
                    textAlign: TextAlign.center,
                    style: Theme.of(context)
                        .textTheme
                        .headlineSmall
                        ?.copyWith(fontWeight: FontWeight.w800),
                  ),
                  const SizedBox(height: 10),
                  Text(
                    message,
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 18),
                  FilledButton.icon(
                    onPressed: onRetry,
                    icon: const Icon(Icons.refresh),
                    label: const Text('Проверить снова'),
                  ),
                  const SizedBox(height: 8),
                  TextButton(
                    onPressed: () => Navigator.of(context).maybePop(),
                    child: const Text('Вернуться'),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class _SectionCard extends StatelessWidget {
  final String title;
  final Widget child;

  const _SectionCard({
    required this.title,
    required this.child,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              title,
              style: Theme.of(context)
                  .textTheme
                  .titleMedium
                  ?.copyWith(fontWeight: FontWeight.w800),
            ),
            const SizedBox(height: 10),
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
    final color = Theme.of(context).colorScheme.error;
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: color.withOpacity(0.10),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color.withOpacity(0.25)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(Icons.error_outline, color: color),
          const SizedBox(width: 10),
          Expanded(
            child: Text(message, style: TextStyle(color: color)),
          ),
        ],
      ),
    );
  }
}

class _StatusBlock extends StatelessWidget {
  final TrainingStatus? status;

  const _StatusBlock({required this.status});

  @override
  Widget build(BuildContext context) {
    if (status == null) return const Text('Нет данных.');

    final value = status!;
    String versionText(int? version) => version == null ? '—' : 'v$version';

    return Column(
      children: [
        _StatusRow(
          label: 'Обучение сейчас',
          value: value.isTraining ? 'Да' : 'Нет',
        ),
        const SizedBox(height: 8),
        _StatusRow(
          label: 'Запрос ожидает worker',
          value: value.retrainRequested ? 'Да' : 'Нет',
        ),
        const SizedBox(height: 8),
        _StatusRow(
          label: 'Активная модель',
          value: versionText(value.activeModelVersion),
        ),
        const SizedBox(height: 8),
        _StatusRow(
          label: 'Последняя обученная',
          value: versionText(value.lastTrainedVersion),
        ),
        if (value.lastError != null && value.lastError!.isNotEmpty) ...[
          const SizedBox(height: 10),
          Align(
            alignment: Alignment.centerLeft,
            child: Text(
              'Последняя ошибка: ${value.lastError}',
              style: TextStyle(color: Theme.of(context).colorScheme.error),
            ),
          ),
        ],
      ],
    );
  }
}

class _StatusRow extends StatelessWidget {
  final String label;
  final String value;

  const _StatusRow({
    required this.label,
    required this.value,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Expanded(child: Text(label)),
        Text(
          value,
          style: const TextStyle(fontWeight: FontWeight.w700),
        ),
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
      return const Text('События в текущем процессе пока отсутствуют.');
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: events.map((event) {
        final details = event.meta.isEmpty ? '' : '\n${event.meta}';
        return Padding(
          padding: const EdgeInsets.only(bottom: 10),
          child: Text(
            '${event.ts}  ${event.level}: ${event.message}$details',
          ),
        );
      }).toList(),
    );
  }
}
