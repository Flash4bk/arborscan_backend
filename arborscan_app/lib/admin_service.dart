import 'dart:convert';
import 'package:http/http.dart' as http;

/// Minimal admin API client for ArborScan.
///
/// Note: We intentionally removed "verified analyses" logic.
/// Admin panel is now focused on operations tools: model versions & retraining.
class AdminService {
  static const String _baseUrl = 'https://arborscanbackend-production.up.railway.app';

  static Uri _u(String path) => Uri.parse('$_baseUrl$path');

  /// GET /admin/training-status
  /// Expected keys (as in backend): training_in_progress, last_model_version, active_model_version
  static Future<Map<String, dynamic>> getTrainingStatus() async {
    final resp = await http.get(_u('/admin/training-status'));
    if (resp.statusCode != 200) {
      throw Exception('HTTP ${resp.statusCode}: ${resp.body}');
    }
    final data = jsonDecode(resp.body);
    if (data is Map<String, dynamic>) return data;
    throw Exception('Unexpected response type');
  }

  /// GET /admin/models
  /// Optional endpoint: returns list of available model versions.
  static Future<List<int>> fetchModelVersions() async {
    final resp = await http.get(_u('/admin/models'));
    if (resp.statusCode != 200) {
      throw Exception('HTTP ${resp.statusCode}: ${resp.body}');
    }
    final data = jsonDecode(resp.body);
    // Accept either: {"models":[{"version":1}, ...]} or {"versions":[1,2]} or [1,2]
    if (data is List) {
      return data.map((e) => (e as num).toInt()).toList();
    }
    if (data is Map) {
      final models = data['models'];
      if (models is List) {
        final versions = <int>[];
        for (final m in models) {
          if (m is Map && m['version'] != null) versions.add((m['version'] as num).toInt());
        }
        return versions;
      }
      final versions = data['versions'];
      if (versions is List) {
        return versions.map((e) => (e as num).toInt()).toList();
      }
    }
    return const <int>[];
  }

  /// POST /admin/set-active-model  Body: {"version": int}
  static Future<void> setActiveModel(int version) async {
    final resp = await http.post(
      _u('/admin/set-active-model'),
      headers: const {'Content-Type': 'application/json'},
      body: jsonEncode({'version': version}),
    );
    if (resp.statusCode != 200) {
      throw Exception('HTTP ${resp.statusCode}: ${resp.body}');
    }
  }

  /// POST /admin/request-retrain
  static Future<void> requestRetrain() async {
    final resp = await http.post(_u('/admin/request-retrain'));
    if (resp.statusCode != 200) {
      throw Exception('HTTP ${resp.statusCode}: ${resp.body}');
    }
  }
}
