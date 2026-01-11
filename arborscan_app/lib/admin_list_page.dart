import 'package:flutter/material.dart';
import 'admin_service.dart';

class AdminListPage extends StatefulWidget {
  const AdminListPage({Key? key}) : super(key: key);

  @override
  State<AdminListPage> createState() => _AdminListPageState();
}

class _AdminListPageState extends State<AdminListPage> {
  final AdminService _service = AdminService();

  bool _loading = true;
  bool _training = false;
  String? _error;

  List<ModelInfo> _models = [];
  ModelInfo? _selectedModel;

  @override
  void initState() {
    super.initState();
    _loadModels();
  }

  Future<void> _loadModels() async {
    setState(() {
      _loading = true;
      _error = null;
    });

    try {
      final models = await _service.fetchModels();
      setState(() {
        _models = models;
        _selectedModel =
            models.firstWhere((m) => m.isActive, orElse: () => models.first);
      });
    } catch (e) {
      setState(() {
        _error = e.toString();
      });
    } finally {
      setState(() {
        _loading = false;
      });
    }
  }

  Future<void> _activateSelected() async {
    if (_selectedModel == null) return;

    setState(() {
      _loading = true;
      _error = null;
    });

    try {
      await _service.setActiveModel(_selectedModel!.version);
      await _loadModels();

      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Активная модель обновлена')),
      );
    } catch (e) {
      setState(() {
        _error = e.toString();
      });
    } finally {
      setState(() {
        _loading = false;
      });
    }
  }

  Future<void> _requestTraining() async {
    setState(() {
      _training = true;
      _error = null;
    });

    try {
      await _service.requestTraining();

      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Обучение поставлено в очередь')),
      );
    } catch (e) {
      setState(() {
        _error = e.toString();
      });
    } finally {
      setState(() {
        _training = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Admin Panel'),
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : Padding(
              padding: const EdgeInsets.all(16),
              child: ListView(
                children: [
                  // =======================
                  // MODEL CONTROL
                  // =======================
                  Card(
                    child: Padding(
                      padding: const EdgeInsets.all(16),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'Управление моделью',
                            style: theme.textTheme.titleMedium
                                ?.copyWith(fontWeight: FontWeight.w700),
                          ),
                          const SizedBox(height: 12),

                          DropdownButtonFormField<ModelInfo>(
                            value: _selectedModel,
                            items: _models
                                .map(
                                  (m) => DropdownMenuItem<ModelInfo>(
                                    value: m,
                                    child: Text(
                                      'Версия ${m.version}'
                                      '${m.isActive ? ' (активна)' : ''}',
                                    ),
                                  ),
                                )
                                .toList(),
                            onChanged: (v) {
                              setState(() {
                                _selectedModel = v;
                              });
                            },
                            decoration: const InputDecoration(
                              labelText: 'Активная модель',
                              border: OutlineInputBorder(),
                            ),
                          ),
                          const SizedBox(height: 12),

                          SizedBox(
                            width: double.infinity,
                            child: FilledButton.icon(
                              onPressed: (_selectedModel == null ||
                                      _selectedModel!.isActive)
                                  ? null
                                  : _activateSelected,
                              icon: const Icon(Icons.swap_horiz),
                              label: const Text('Сделать активной'),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),

                  const SizedBox(height: 16),

                  // =======================
                  // TRAINING CONTROL
                  // =======================
                  Card(
                    child: Padding(
                      padding: const EdgeInsets.all(16),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'Обучение модели',
                            style: theme.textTheme.titleMedium
                                ?.copyWith(fontWeight: FontWeight.w700),
                          ),
                          const SizedBox(height: 12),
                          Text(
                            'Использует накопленные данные и feedback из приложения.',
                            style: theme.textTheme.bodySmall,
                          ),
                          const SizedBox(height: 12),
                          SizedBox(
                            width: double.infinity,
                            child: FilledButton.icon(
                              onPressed: _training ? null : _requestTraining,
                              icon: const Icon(Icons.play_arrow),
                              label: Text(
                                _training
                                    ? 'Обучение запускается...'
                                    : 'Запустить обучение',
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),

                  const SizedBox(height: 16),

                  // =======================
                  // ERROR
                  // =======================
                  if (_error != null)
                    Container(
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: cs.errorContainer,
                        borderRadius: BorderRadius.circular(16),
                      ),
                      child: Text(
                        _error!,
                        style: theme.textTheme.bodySmall
                            ?.copyWith(color: cs.onErrorContainer),
                      ),
                    ),
                ],
              ),
            ),
    );
  }
}
