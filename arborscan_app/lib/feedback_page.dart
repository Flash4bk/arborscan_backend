import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

import 'mask_drawing_page.dart';

class FeedbackPage extends StatefulWidget {
  final String analysisId;
  final Uint8List imageBytes;

  const FeedbackPage({
    super.key,
    required this.analysisId,
    required this.imageBytes,
  });

  @override
  State<FeedbackPage> createState() => _FeedbackPageState();
}

class _FeedbackPageState extends State<FeedbackPage> {
  Uint8List? _userMaskBytes;
  bool _sending = false;
  String? _status;

  // ⚠️ ОБЯЗАТЕЛЬНО ПРОВЕРЬ URL
  static const String _baseUrl =
      'https://arborscanbackend-production.up.railway.app';

  Future<void> _openMaskPage() async {
    final result = await Navigator.of(context).push<Uint8List>(
      MaterialPageRoute(
        builder: (_) => MaskDrawingPage(
          imageBytes: widget.imageBytes,
        ),
      ),
    );

    if (result != null) {
      setState(() {
        _userMaskBytes = result;
      });
    }
  }

  Future<void> _sendToBackend() async {
    if (_userMaskBytes == null) return;

    setState(() {
      _sending = true;
      _status = null;
    });

    try {
      final body = {
        'analysis_id': widget.analysisId,
        'image_base64': base64Encode(widget.imageBytes),
        'mask_base64': base64Encode(_userMaskBytes!),
        'meta': {
          'source': 'user_mask',
          'platform': 'flutter',
        },
      };

      final response = await http.post(
        Uri.parse('$_baseUrl/dataset/user-mask'),
        headers: {
          'Content-Type': 'application/json',
        },
        body: jsonEncode(body),
      );

      if (response.statusCode == 200) {
        setState(() {
          _status = '✅ Успешно отправлено';
        });
      } else {
        setState(() {
          _status =
              '❌ Ошибка ${response.statusCode}: ${response.body.toString()}';
        });
      }
    } catch (e) {
      setState(() {
        _status = '❌ Exception: $e';
      });
    } finally {
      setState(() {
        _sending = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('User feedback'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Image.memory(widget.imageBytes),

            const SizedBox(height: 16),

            OutlinedButton.icon(
              icon: const Icon(Icons.brush),
              label: const Text('Нарисовать маску дерева'),
              onPressed: _openMaskPage,
            ),

            if (_userMaskBytes != null) ...[
              const SizedBox(height: 12),
              const Text(
                'Маска:',
                style: TextStyle(fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 8),
              Image.memory(
                _userMaskBytes!,
                height: 200,
                fit: BoxFit.contain,
              ),
            ],

            const Spacer(),

            ElevatedButton.icon(
              icon: _sending
                  ? const SizedBox(
                      width: 18,
                      height: 18,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Icon(Icons.cloud_upload),
              label: const Text('Отправить в обучающий датасет'),
              onPressed:
                  (_userMaskBytes == null || _sending) ? null : _sendToBackend,
            ),

            if (_status != null) ...[
              const SizedBox(height: 12),
              Text(
                _status!,
                textAlign: TextAlign.center,
              ),
            ],
          ],
        ),
      ),
    );
  }
}
