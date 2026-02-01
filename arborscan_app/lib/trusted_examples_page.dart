import 'dart:convert';
import 'package:flutter/material.dart';

import 'admin_service.dart';
import 'app_theme.dart';
import 'mask_drawing_page.dart';

class TrustedExamplesPage extends StatefulWidget {
  final String baseUrl;
  const TrustedExamplesPage({super.key, required this.baseUrl});

  @override
  State<TrustedExamplesPage> createState() => _TrustedExamplesPageState();
}

class _TrustedExamplesPageState extends State<TrustedExamplesPage> {
  late final AdminService _service = AdminService(baseUrl: widget.baseUrl);
  List<dynamic> _examples = [];
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _loadData();
  }

  Future<void> _loadData() async {
    setState(() => _isLoading = true);
    try {
      // Имитация загрузки. В реальности: await http.get(...)
      await Future.delayed(const Duration(seconds: 1));
      setState(() {
        _examples = []; // Сюда придут данные с бэкенда
        _isLoading = false;
      });
    } catch (_) {
      setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Верификация данных')),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : _examples.isEmpty
              ? ListView(
                  padding: const EdgeInsets.all(16),
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
                              'Пока нет примеров для верификации. Как только на бэкенде появятся данные — они отобразятся здесь.',
                              style: Theme.of(context).textTheme.bodyMedium?.copyWith(color: AppTheme.muted),
                            ),
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(height: 12),
                    OutlinedButton.icon(
                      onPressed: _loadData,
                      icon: const Icon(Icons.refresh),
                      label: const Text('Обновить'),
                    ),
                  ],
                )
              : GridView.builder(
                  padding: const EdgeInsets.all(12),
                  gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                    crossAxisCount: 2,
                    childAspectRatio: 0.78,
                    mainAxisSpacing: 12,
                    crossAxisSpacing: 12,
                  ),
                  itemCount: _examples.length,
                  itemBuilder: (context, index) {
                    final item = _examples[index];
                    return Card(
                      clipBehavior: Clip.antiAlias,
                      child: InkWell(
                        onTap: () => _openEditor(item),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.stretch,
                          children: [
                            Expanded(
                              child: Container(
                                color: Colors.black.withOpacity(0.04),
                                child: Image.memory(
                                  base64Decode(item['thumbnail'] ?? ''),
                                  fit: BoxFit.cover,
                                  errorBuilder: (_, __, ___) => const Center(
                                    child: Icon(Icons.image_not_supported),
                                  ),
                                ),
                              ),
                            ),
                            Padding(
                              padding: const EdgeInsets.all(10),
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text(
                                    item['species'] ?? 'Пример',
                                    maxLines: 1,
                                    overflow: TextOverflow.ellipsis,
                                    style: Theme.of(context).textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w800),
                                  ),
                                  const SizedBox(height: 6),
                                  Row(
                                    children: [
                                      Ui.badge(
                                        text: 'Открыть',
                                        color: AppTheme.primary,
                                        icon: Icons.edit,
                                      ),
                                      const Spacer(),
                                      Icon(Icons.chevron_right, color: Colors.black.withOpacity(0.35)),
                                    ],
                                  ),
                                ],
                              ),
                            ),
                          ],
                        ),
                      ),
                    );
                  },
                ),
    );
  }

  Future<void> _openEditor(dynamic item) async {
    final result = await Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => MaskDrawingPage(
          originalImageBase64: item['original_image'],
          aiMaskBase64: item['ai_mask'],
        ),
      ),
    );

    if (result != null && result['points'] != null) {
      await _service.verifyExample(item['analysis_id'], result['points'], result['closed']);
      _loadData();
    }
  }
}
