import 'dart:convert';
import 'package:http/http.dart' as http;

/// Админ API клиент.
/// ВАЖНО: здесь НЕТ verified-analyses (мы их убрали из UI).
class AdminService {
  static const String baseUrl = 'https://arborscanbackend-production.up.railway.app';

  static Uri _u(String path) => Uri.parse('$baseUrl$path');

  /// Возвращает статус обучения и активную/последнюю версию модели.
  /// Ожидаемый ответ сервера:
  /// {
  ///   "training_in_progress": bool,
  ///   "active_model_version": int,
  ///   "last_model_version": int
  /// }
  static Future<Map<String, dynamic>> getTrainingStatus() async {
    final resp = await http.get(_u('/admin/training-status'));
    if (resp.statusCode != 200) {
      throw Exception('HTTP ${resp.statusCode}: ${resp.body}');
    }
    final data = jsonDecode(resp.body);
    if (data is Map<String, dynamic>) return data;
    throw Exception('Unexpected response: ${resp.body}');
  }

  /// Делает указанную версию модели активной.
  static Future<void> setActiveModel(int version) async {
    final resp = await http.post(
      _u('/admin/set-active-model'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'version': version}),
    );
    if (resp.statusCode != 200) {
      throw Exception('HTTP ${resp.statusCode}: ${resp.body}');
    }
  }

  /// Запросить обучение (retrain worker).
  static Future<void> requestRetrain() async {
    final resp = await http.post(_u('/admin/request-retrain'));
    if (resp.statusCode != 200) {
      throw Exception('HTTP ${resp.statusCode}: ${resp.body}');
    }
  }
}
