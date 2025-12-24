import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'admin_service.dart';

class AdminAnalysisDetailPage extends StatefulWidget {
  final String analysisId;

  const AdminAnalysisDetailPage({
    super.key,
    required this.analysisId,
  });

  @override
  State<AdminAnalysisDetailPage> createState() =>
      _AdminAnalysisDetailPageState();
}

class _AdminAnalysisDetailPageState extends State<AdminAnalysisDetailPage> {
  late Future<Map<String, dynamic>> _future;
  bool _showRawJson = false;

  @override
  void initState() {
    super.initState();
    _future = AdminService.fetchVerifiedAnalysis(widget.analysisId);
  }

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<Map<String, dynamic>>(
      future: _future,
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.waiting) {
          return const Scaffold(
            body: Center(child: CircularProgressIndicator()),
          );
        }

        if (snapshot.hasError) {
          return Scaffold(
            body: Center(child: Text('Ошибка: ${snapshot.error}')),
          );
        }

        final data = snapshot.data!;
        final meta = data['meta'] ?? {};
        final images = data['images'] ?? {};

        Uint8List? preview;
        if (images['input_base64'] != null) {
          preview = base64Decode(images['input_base64']);
        }

        return Scaffold(
          appBar: AppBar(title: const Text('Analysis details')),
          body: SingleChildScrollView(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _row('Analysis ID', meta['analysis_id']),
                _row('Species', meta['species']),
                _row('Risk category', meta['risk']?['category']),
                _row('Trust score', meta['trust_score']?.toString()),
                _row('Verified at', meta['verified_at']),
                const SizedBox(height: 16),

                if (preview != null)
                  ClipRRect(
                    borderRadius: BorderRadius.circular(12),
                    child: Image.memory(preview),
                  ),

                const SizedBox(height: 12),

                OutlinedButton(
                  onPressed: () {
                    setState(() {
                      _showRawJson = !_showRawJson;
                    });
                  },
                  child: Text(
                    _showRawJson ? 'Скрыть raw JSON' : 'Показать raw JSON',
                  ),
                ),

                if (_showRawJson)
                  Container(
                    margin: const EdgeInsets.only(top: 12),
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: Colors.grey.shade100,
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: SelectableText(
                      const JsonEncoder.withIndent('  ').convert(data),
                      style: const TextStyle(fontSize: 12),
                    ),
                  ),
              ],
            ),
          ),
        );
      },
    );
  }

  Widget _row(String label, String? value) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 6),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(width: 120, child: Text(label)),
          Expanded(child: Text(value ?? '—')),
        ],
      ),
    );
  }
}
