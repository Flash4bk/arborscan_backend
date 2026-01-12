import 'dart:convert';
import 'package:http/http.dart' as http;

/// Единая точка backend
const String _baseUrl = 'https://arborscanbackend-production.up.railway.app';

class TrainingEvent {
  final String type;
  final String message;
  final DateTime timestamp;

  TrainingEvent({
    required this.type,
    required this.message,
    required this.timestamp,
  });

  factory TrainingEvent.fromJson(Map<String, dynamic> json) {
    return TrainingEvent(
      type: (json['type'] ?? 'info').toString(),
      message: (json['message'] ?? '').toString(),
      timestamp: DateTime.tryParse((json['timestamp'] ?? '').toString()) ??
          DateTime.fromMillisecondsSinceEpoch(0),
    );
  }
}

class TrainingStatus {
  final bool training;
  final String? activeModel;
  final String? lastTrained;

  TrainingStatus({
    required this.training,
    this.activeModel,
    this.lastTrained,
  });

  factory TrainingStatus.fromJson(Map<String, dynamic> json) {
    return TrainingStatus(
      training: json['training'] == true,
      activeModel: json['active_model']?.toString(),
      lastTrained: json['last_trained']?.toString(),
    );
  }
}

class AdminService {
  static Future<TrainingStatus> getTrainingStatus() async {
    final uri = Uri.parse('$_baseUrl/admin/training-status');
    final resp = await http.get(uri);

    if (resp.statusCode != 200) {
      throw Exception('HTTP ${resp.statusCode}: training-status');
    }

    final data = jsonDecode(resp.body) as Map<String, dynamic>;
    return TrainingStatus.fromJson(data);
  }

  static Future<List<TrainingEvent>> getTrainingEvents({int limit = 20}) async {
    final uri = Uri.parse('$_baseUrl/admin/training-events?limit=$limit');
    final resp = await http.get(uri);

    if (resp.statusCode != 200) {
      throw Exception('HTTP ${resp.statusCode}: training-events');
    }

    final list = jsonDecode(resp.body);
    if (list is! List) return [];

    return list
        .whereType<Map<String, dynamic>>()
        .map(TrainingEvent.fromJson)
        .toList();
  }

  static Future<void> requestTraining() async {
    final uri = Uri.parse('$_baseUrl/admin/request-training');
    final resp = await http.post(uri);

    if (resp.statusCode != 200) {
      throw Exception('HTTP ${resp.statusCode}: request-training');
    }
  }
}
