import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';

void main() {
  runApp(const ArborScanApp());
}

class ArborScanApp extends StatelessWidget {
  const ArborScanApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'ArborScan',
      theme: ThemeData(
        useMaterial3: true,
        colorSchemeSeed: Colors.green,
      ),
      home: const ArborScanPage(),
    );
  }
}

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
  Map<String, dynamic>? _result;
  Uint8List? _annotatedImageBytes;

  // TODO: поменяй на свой реальный URL Railway
  final String _apiUrl = 'https://your-railway-app-url.up.railway.app/analyze-tree';

  Future<void> _pickImage(ImageSource source) async {
    setState(() {
      _error = null;
      _result = null;
      _annotatedImageBytes = null;
    });
    final XFile? picked = await _picker.pickImage(source: source, imageQuality: 95);
    if (picked != null) {
      setState(() {
        _imageFile = File(picked.path);
      });
    }
  }

  Future<void> _analyze() async {
    if (_imageFile == null) {
      setState(() {
        _error = 'Сначала выберите изображение';
      });
      return;
    }

    setState(() {
      _isLoading = true;
      _error = null;
      _result = null;
      _annotatedImageBytes = null;
    });

    try {
      final request = http.MultipartRequest('POST', Uri.parse(_apiUrl));
      request.files.add(
        await http.MultipartFile.fromPath('file', _imageFile!.path),
      );
      final streamed = await request.send();
      final response = await http.Response.fromStream(streamed);

      if (response.statusCode != 200) {
        setState(() {
          _error = 'Ошибка сервера: ${response.statusCode}';
        });
        return;
      }

      final data = json.decode(response.body) as Map<String, dynamic>;

      if (data.containsKey('error')) {
        setState(() {
          _error = data['error'] as String?;
        });
      } else {
        Uint8List? annotatedBytes;
        if (data['annotated_image_base64'] != null) {
          annotatedBytes = base64Decode(data['annotated_image_base64'] as String);
        }
        setState(() {
          _result = data;
          _annotatedImageBytes = annotatedBytes;
        });
      }
    } catch (e) {
      setState(() {
        _error = 'Ошибка запроса: $e';
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  Widget _buildImagePreview() {
    if (_annotatedImageBytes != null) {
      return Image.memory(_annotatedImageBytes!, fit: BoxFit.contain);
    }
    if (_imageFile != null) {
      return Image.file(_imageFile!, fit: BoxFit.contain);
    }
    return const Text(
      'Изображение не выбрано',
      textAlign: TextAlign.center,
    );
  }

  Widget _buildResult() {
    if (_isLoading) {
      return const Center(child: CircularProgressIndicator());
    }

    if (_error != null) {
      return Text(
        _error!,
        style: const TextStyle(color: Colors.red),
      );
    }

    if (_result == null) {
      return const SizedBox.shrink();
    }

    final species = _result!['species'];
    final height = _result!['height_m'];
    final crown = _result!['crown_width_m'];
    final trunk = _result!['trunk_diameter_m'];
    final scale = _result!['scale_m_per_px'];

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('Вид: $species', style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
        const SizedBox(height: 8),
        Text('Высота: ${height ?? '-'} м'),
        Text('Ширина кроны: ${crown ?? '-'} м'),
        Text('Диаметр ствола: ${trunk ?? '-'} м'),
        if (scale != null) Text('Масштаб: 1 px = ${scale.toStringAsFixed(4)} м'),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('ArborScan'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(12.0),
        child: Column(
          children: [
            Expanded(
              child: Center(child: _buildImagePreview()),
            ),
            const SizedBox(height: 12),
            _buildResult(),
            const SizedBox(height: 12),
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: () => _pickImage(ImageSource.camera),
                    icon: const Icon(Icons.photo_camera),
                    label: const Text('Камера'),
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: () => _pickImage(ImageSource.gallery),
                    icon: const Icon(Icons.photo_library),
                    label: const Text('Галерея'),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            SizedBox(
              width: double.infinity,
              child: FilledButton(
                onPressed: _isLoading ? null : _analyze,
                child: const Text('Анализировать'),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
