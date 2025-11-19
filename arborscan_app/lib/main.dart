import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';
import 'package:shared_preferences/shared_preferences.dart';

void main() {
  runApp(const ArborScanApp());
}

/// ============================
///  Модель результата анализа
/// ============================
class AnalysisResult {
  final String species;
  final double? height;
  final double? crown;
  final double? trunk;
  final double? scale;
  final String imageBase64;
  final DateTime timestamp;

  AnalysisResult({
    required this.species,
    required this.imageBase64,
    required this.timestamp,
    this.height,
    this.crown,
    this.trunk,
    this.scale,
  });

  Map<String, dynamic> toJson() => {
        'species': species,
        'height': height,
        'crown': crown,
        'trunk': trunk,
        'scale': scale,
        'imageBase64': imageBase64,
        'timestamp': timestamp.toIso8601String(),
      };

  factory AnalysisResult.fromJson(Map<String, dynamic> json) => AnalysisResult(
        species: json['species'],
        height: (json['height'] as num?)?.toDouble(),
        crown: (json['crown'] as num?)?.toDouble(),
        trunk: (json['trunk'] as num?)?.toDouble(),
        scale: (json['scale'] as num?)?.toDouble(),
        imageBase64: json['imageBase64'],
        timestamp: DateTime.parse(json['timestamp']),
      );
}

/// ============================
///   Приложение + темы
/// ============================
class ArborScanApp extends StatelessWidget {
  const ArborScanApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'ArborScan',
      themeMode: ThemeMode.system,
      theme: ThemeData(
        useMaterial3: true,
        colorSchemeSeed: Colors.green,
        brightness: Brightness.light,
      ),
      darkTheme: ThemeData(
        useMaterial3: true,
        colorSchemeSeed: Colors.green,
        brightness: Brightness.dark,
      ),
      home: const ArborScanPage(),
    );
  }
}

/// ============================
///      Главный экран
/// ============================
class ArborScanPage extends StatefulWidget {
  const ArborScanPage({super.key});

  @override
  State<ArborScanPage> createState() => _ArborScanPageState();
}

class _ArborScanPageState extends State<ArborScanPage> {
  final ImagePicker _picker = ImagePicker();
  File? _imageFile;

  bool _isLoading = false;
  String? _error;
  Uint8List? _annotatedImageBytes;
  Map<String, dynamic>? _result;

  final String _apiUrl =
      'https://arborscanbackend-production.up.railway.app/analyze-tree';

  final List<AnalysisResult> _history = [];
  static const _historyKey = 'arborscan_history';

  @override
  void initState() {
    super.initState();
    _loadHistory();
  }

  Future<void> _loadHistory() async {
    final prefs = await SharedPreferences.getInstance();
    final list = prefs.getStringList(_historyKey);
    if (list == null) return;

    setState(() {
      _history.clear();
      _history.addAll(
        list.map((e) => AnalysisResult.fromJson(jsonDecode(e))),
      );
    });
  }

  Future<void> _saveHistory() async {
    final prefs = await SharedPreferences.getInstance();
    final list =
        _history.map((e) => jsonEncode(e.toJson())).toList(growable: false);
    await prefs.setStringList(_historyKey, list);
  }

  Future<void> _pickImage(ImageSource source) async {
    setState(() {
      _error = null;
      _result = null;
      _annotatedImageBytes = null;
    });

    final XFile? picked = await _picker.pickImage(
      source: source,
      imageQuality: 92,
    );

    if (picked != null) {
      setState(() => _imageFile = File(picked.path));
    }
  }

  Future<void> _analyze() async {
    if (_imageFile == null) {
      setState(() => _error = 'Сначала выберите изображение.');
      return;
    }

    setState(() {
      _isLoading = true;
      _error = null;
    });

    try {
      final req = http.MultipartRequest('POST', Uri.parse(_apiUrl));
      req.files.add(await http.MultipartFile.fromPath('file', _imageFile!.path));

      final streamed = await req.send();
      final response = await http.Response.fromStream(streamed);

      if (response.statusCode != 200) {
        setState(() => _error = 'Ошибка сервера: ${response.statusCode}');
        return;
      }

      final data = jsonDecode(response.body);

      if (data['error'] != null) {
        setState(() => _error = data['error']);
        return;
      }

      Uint8List? imgBytes;
      if (data['annotated_image_base64'] != null) {
        imgBytes = base64Decode(data['annotated_image_base64']);
      }

      final result = AnalysisResult(
        species: data['species'],
        height: (data['height_m'] as num?)?.toDouble(),
        crown: (data['crown_width_m'] as num?)?.toDouble(),
        trunk: (data['trunk_diameter_m'] as num?)?.toDouble(),
        scale: (data['scale_m_per_px'] as num?)?.toDouble(),
        imageBase64: imgBytes != null ? base64Encode(imgBytes) : '',
        timestamp: DateTime.now(),
      );

      _history.insert(0, result);
      await _saveHistory();

      setState(() {
        _result = data;
        _annotatedImageBytes = imgBytes;
      });
    } catch (e) {
      setState(() => _error = 'Ошибка запроса: $e');
    } finally {
      setState(() => _isLoading = false);
    }
  }

  Widget _buildImageCard() {
    Widget child;

    if (_annotatedImageBytes != null) {
      child = Image.memory(_annotatedImageBytes!, fit: BoxFit.cover);
    } else if (_imageFile != null) {
      child = Image.file(_imageFile!, fit: BoxFit.cover);
    } else {
      child = const Center(
        child: Text(
          'Выберите изображение дерева\nс эталонной палкой',
          textAlign: TextAlign.center,
          style: TextStyle(fontSize: 16),
        ),
      );
    }

    return Card(
      elevation: 3,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
      clipBehavior: Clip.antiAlias,
      child: AspectRatio(aspectRatio: 3 / 4, child: child),
    );
  }

  Widget _buildResultCard() {
    if (_error != null) {
      return Card(
        color: Colors.red.withOpacity(0.1),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Text(_error!, style: const TextStyle(color: Colors.red)),
        ),
      );
    }

    if (_result == null) return const SizedBox.shrink();

    return Card(
      elevation: 2,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
      child: Padding(
        padding: const EdgeInsets.all(18.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Вид: ${_result!['species']}',
              style: const TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 8),
            Text('Высота: ${_result!['height_m'] ?? '-'} м'),
            Text('Ширина кроны: ${_result!['crown_width_m'] ?? '-'} м'),
            Text('Диаметр ствола: ${_result!['trunk_diameter_m'] ?? '-'} м'),
            Text('Масштаб: 1 px = ${_result!['scale_m_per_px'] ?? '-'} м'),
          ],
        ),
      ),
    );
  }

  void _openHistory() {
    Navigator.push(
      context,
      MaterialPageRoute(builder: (_) => HistoryPage(history: _history)),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('ArborScan'),
        actions: [
          IconButton(
            icon: const Icon(Icons.history),
            tooltip: 'История измерений',
            onPressed: _history.isEmpty ? null : _openHistory,
          )
        ],
      ),
      body: Stack(
        children: [
          SafeArea(
            child: SingleChildScrollView(
              padding: const EdgeInsets.all(12),
              child: Column(
                children: [
                  _buildImageCard(),
                  const SizedBox(height: 12),
                  _buildResultCard(),
                  const SizedBox(height: 12),
                  Row(
                    children: [
                      Expanded(
                        child: FilledButton.icon(
                          onPressed: _isLoading
                              ? null
                              : () => _pickImage(ImageSource.camera),
                          icon: const Icon(Icons.photo_camera),
                          label: const Text('Камера'),
                        ),
                      ),
                      const SizedBox(width: 10),
                      Expanded(
                        child: OutlinedButton.icon(
                          onPressed: _isLoading
                              ? null
                              : () => _pickImage(ImageSource.gallery),
                          icon: const Icon(Icons.photo_library),
                          label: const Text('Галерея'),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 14),
                  SizedBox(
                    width: double.infinity,
                    child: FilledButton(
                      onPressed: _isLoading ? null : _analyze,
                      child: const Padding(
                        padding: EdgeInsets.symmetric(vertical: 14),
                        child: Text('Анализировать'),
                      ),
                    ),
                  ),
                  const SizedBox(height: 30),
                ],
              ),
            ),
          ),
          if (_isLoading)
            Container(
              color: Colors.black54,
              child: const Center(
                child: CircularProgressIndicator(strokeWidth: 6),
              ),
            ),
        ],
      ),
    );
  }
}

/// ============================
///      Экран истории
/// ============================
class HistoryPage extends StatelessWidget {
  final List<AnalysisResult> history;

  const HistoryPage({super.key, required this.history});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('История измерений')),
      body: ListView.separated(
        padding: const EdgeInsets.all(12),
        itemCount: history.length,
        separatorBuilder: (_, __) => const SizedBox(height: 10),
        itemBuilder: (context, index) {
          final item = history[index];
          final img = item.imageBase64.isNotEmpty
              ? Image.memory(base64Decode(item.imageBase64), fit: BoxFit.cover)
              : null;

          return Card(
            shape:
                RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
            child: ListTile(
              leading: img != null
                  ? ClipRRect(
                      borderRadius: BorderRadius.circular(12),
                      child: SizedBox(width: 56, height: 56, child: img),
                    )
                  : const Icon(Icons.park),
              title: Text(item.species),
              subtitle: Text(
                  'Высота: ${item.height ?? '-'} м\nКрона: ${item.crown ?? '-'} м'),
            ),
          );
        },
      ),
    );
  }
}
