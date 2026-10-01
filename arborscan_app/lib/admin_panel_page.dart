import 'package:flutter/material.dart';
import 'app_theme.dart';
import 'corrections_service.dart';

import 'admin_service.dart';
import 'training_dataset_page.dart';
import 'saved_corrections_page.dart';
import 'model_quality_page.dart';

class AdminPanelPage extends StatefulWidget {
  final String baseUrl;

  const AdminPanelPage({super.key, required this.baseUrl});

  @override
  State<AdminPanelPage> createState() => _AdminPanelPageState();
}

class _AdminPanelPageState extends State<AdminPanelPage> {
  late final AdminService _service = AdminService(baseUrl: widget.baseUrl);

  bool _loading = true;
  bool _invalid = false;
  String? _error;
  int? _errorStatusCode;

  AdminIdentity? _identity;
  TrainingStatus? _status;
  List<TrainingEvent> _events = const [];
  List<int> _models = const [];
  int? _selectedVersion;

  bool get _accessDenied => _errorStatusCode == 401 || _errorStatusCode == 403;

  @override
  void initState() {
    super.initState();
    CorrectionsService.authChanges.addListener(_invalidate);
    _refresh();
  }

  void _invalidate() {
    if (!mounted) return;
    setState(() {
      _invalid = true;
      _loading = false;
      _identity = null;
      _status = null;
      _events = [];
      _models = [];
      _error = 'Аккаунт изменился. Откройте раздел заново.';
    });
  }

  @override
  void dispose() {
    CorrectionsService.authChanges.removeListener(_invalidate);
    super.dispose();
  }

  Future<void> _refresh() async {
    if (_invalid) return;
    final generation = CorrectionsService.authChanges.value;
    if (mounted) {
      setState(() {
        _loading = true;
        _error = null;
        _errorStatusCode = null;
      });
    }

    try {
      final identity = await _service.verifyAdminAccess();
      if (!mounted ||
          _invalid ||
          generation != CorrectionsService.authChanges.value) {
        return;
      }
      setState(() => _identity = identity);
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

      if (!mounted ||
          _invalid ||
          generation != CorrectionsService.authChanges.value) {
        return;
      }
      setState(() {
        _identity = identity;
        _status = status;
        _events = events;
        _models = models;
        _selectedVersion = selection;
        _loading = false;
      });
    } on AdminApiException catch (error) {
      if (!mounted ||
          _invalid ||
          generation != CorrectionsService.authChanges.value) {
        return;
      }
      setState(() {
        _error = error.message;
        _errorStatusCode = error.statusCode;
        _loading = false;
      });
    } catch (error) {
      if (!mounted ||
          _invalid ||
          generation != CorrectionsService.authChanges.value) {
        return;
      }
      setState(() {
        _error = error.toString();
        _errorStatusCode = null;
        _loading = false;
      });
    }
  }

  Future<void> _setActive() async {
    await Navigator.push(
        context, MaterialPageRoute(builder: (_) => const ModelQualityPage()));
  }

  @override
  Widget build(BuildContext context) {
    final canOpen = !_invalid && _identity != null && !_accessDenied;
    return Scaffold(
      appBar: AppBar(title: const Text('Администрирование'), actions: [
        IconButton(
            tooltip: 'Обновить данные',
            icon: const Icon(Icons.refresh),
            onPressed: _loading || _invalid ? null : _refresh),
      ]),
      body: _accessDenied
          ? _AccessDeniedView(
              message: _error ?? 'Нет доступа.',
              statusCode: _errorStatusCode,
              onRetry: _refresh)
          : AppContentList(children: [
              if (_loading) const LinearProgressIndicator(),
              if (_error != null) _ErrorBanner(message: _error!),
              if (_identity != null) _AdminIdentityBlock(identity: _identity!),
              if (!_invalid) ...[
                Card(
                    child: ListTile(
                  leading: const Icon(Icons.fact_check_outlined),
                  title: const Text('Проверка контуров'),
                  subtitle: const Text('Фото, ревизии и решения'),
                  trailing: const Icon(Icons.chevron_right),
                  onTap: canOpen
                      ? () => Navigator.push(
                          context,
                          MaterialPageRoute(
                              builder: (_) =>
                                  const SavedCorrectionsPage(adminQueue: true)))
                      : null,
                )),
                Card(
                    child: ListTile(
                  leading: const Icon(Icons.layers_outlined),
                  title: const Text('Модели и данные'),
                  subtitle: const Text('Контуры, породы и снимки наборов'),
                  trailing: const Icon(Icons.chevron_right),
                  onTap: canOpen ? _setActive : null,
                )),
                ExpansionTile(
                    title: const Text('Архив v3'),
                    subtitle: const Text('Прежние данные и события'),
                    children: [
                      _Card(
                          title: 'Архивный статус обучения',
                          child: _StatusBlock(status: _status)),
                      if (_models.isNotEmpty)
                        _Card(
                            title: 'Прежние версии моделей',
                            child: DropdownButtonFormField<int>(
                              initialValue: _selectedVersion,
                              isExpanded: true,
                              items: _models
                                  .map((v) => DropdownMenuItem(
                                      value: v, child: Text('Версия $v')))
                                  .toList(),
                              onChanged: (v) =>
                                  setState(() => _selectedVersion = v),
                            )),
                      ListTile(
                          leading: const Icon(Icons.inventory_2_outlined),
                          title: const Text('Архив примеров'),
                          subtitle: const Text(
                              'Архивный флаг не допускает запись к новому обучению'),
                          trailing: const Icon(Icons.chevron_right),
                          onTap: canOpen
                              ? () => Navigator.push(
                                  context,
                                  MaterialPageRoute(
                                      builder: (_) => TrainingDatasetPage(
                                          service: _service)))
                              : null),
                      _Card(
                          title: 'События архива',
                          child: _TrainingLog(events: _events)),
                    ]),
              ],
            ]),
    );
  }
}

class _AdminIdentityBlock extends StatelessWidget {
  final AdminIdentity identity;

  const _AdminIdentityBlock({required this.identity});

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        const CircleAvatar(child: Icon(Icons.admin_panel_settings)),
        const SizedBox(width: 12),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                identity.name.isEmpty ? 'Администратор' : identity.name,
                style: Theme.of(context)
                    .textTheme
                    .titleSmall
                    ?.copyWith(fontWeight: FontWeight.w800),
              ),
              if (identity.email.isNotEmpty) ...[
                const SizedBox(height: 3),
                Text(identity.email),
              ],
              const SizedBox(height: 3),
              const Text(
                'Права подтверждены сервером',
                style: TextStyle(color: AppTheme.success),
              ),
            ],
          ),
        ),
      ],
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
    final title = statusCode == 401 ? 'Требуется вход' : 'Недостаточно прав';
    final icon =
        statusCode == 401 ? Icons.login : Icons.admin_panel_settings_outlined;

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
                  Text(message, textAlign: TextAlign.center),
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

class _Card extends StatelessWidget {
  final String title;
  final Widget child;

  const _Card({required this.title, required this.child});

  @override
  Widget build(BuildContext context) {
    final tt = Theme.of(context).textTheme;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(title,
                style: tt.titleMedium?.copyWith(fontWeight: FontWeight.w800)),
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
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        // ignore: deprecated_member_use
        color: Theme.of(context).colorScheme.error.withOpacity(0.10),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
            color: Theme.of(context).colorScheme.error.withOpacity(0.25)),
      ),
      child: Text(
        message,
        style: TextStyle(color: Theme.of(context).colorScheme.error),
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
        _StatusRow(
            label: 'Обучение сейчас', value: s.isTraining ? 'Да' : 'Нет'),
        const SizedBox(height: 8),
        _StatusRow(
          label: 'Запрос ожидает worker',
          value: s.retrainRequested ? 'Да' : 'Нет',
        ),
        const SizedBox(height: 8),
        _StatusRow(label: 'Активная модель', value: dash(s.activeModelVersion)),
        const SizedBox(height: 8),
        _StatusRow(
            label: 'Последняя обученная', value: dash(s.lastTrainedVersion)),
        if (s.lastError != null && s.lastError!.isNotEmpty) ...[
          const SizedBox(height: 10),
          Align(
            alignment: Alignment.centerLeft,
            child: Text(
              'Последняя ошибка: ${s.lastError}',
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
        'Архивные события пока недоступны.',
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
