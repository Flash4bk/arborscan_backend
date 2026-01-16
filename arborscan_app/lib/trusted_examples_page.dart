import 'dart:convert';
import 'package:flutter/material.dart';
import 'admin_service.dart';
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
    } catch (e) {
      setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Верификация данных')),
      body: _isLoading 
        ? const Center(child: CircularProgressIndicator())
        : GridView.builder(
            padding: const EdgeInsets.all(10),
            gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
              crossAxisCount: 2, 
              childAspectRatio: 0.8,
              mainAxisSpacing: 10,
              crossAxisSpacing: 10
            ),
            itemCount: _examples.length,
            itemBuilder: (context, index) {
              final item = _examples[index];
              return Card(
                clipBehavior: Clip.antiAlias,
                child: InkWell(
                  onTap: () => _openEditor(item),
                  child: Column(
                    children: [
                      Expanded(
                        child: Image.memory(
                          base64Decode(item['thumbnail'] ?? ''), // Исправлено: теперь доступен base64Decode
                          fit: BoxFit.cover,
                          errorBuilder: (_, __, ___) => const Icon(Icons.image_not_supported),
                        ),
                      ),
                      Padding(
                        padding: const EdgeInsets.all(8.0),
                        child: Text("ID: ${item['analysis_id'].toString().substring(0, 8)}"),
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