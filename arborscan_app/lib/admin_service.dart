import 'dart:convert';
import 'package:http/http.dart' as http;
import 'admin_analysis_item.dart';

class AdminService {
  static const String baseUrl =
      'https://arborscanbackend-production.up.railway.app';

  /// Список verified анализов (короткие данные)
  static Future<List<AdminAnalysisItem>> fetchVerifiedAnalyses() async {
    final uri = Uri.parse('$baseUrl/admin/verified-list');

    final res = await http.get(uri);
    if (res.statusCode != 200) {
      throw Exception('Failed to load verified list');
    }

    final data = jsonDecode(res.body) as Map<String, dynamic>;
    final items = data['items'] as List<dynamic>;

    return items
        .map((e) => AdminAnalysisItem.fromJson(e))
        .toList();
  }

  /// Полные детали одного verified анализа
  static Future<Map<String, dynamic>> fetchVerifiedAnalysis(
      String analysisId) async {
    final uri = Uri.parse('$baseUrl/admin/analysis/$analysisId');

    final res = await http.get(uri);
    if (res.statusCode != 200) {
      throw Exception('Failed to load analysis details');
    }

    return jsonDecode(res.body) as Map<String, dynamic>;
  }
}
