
import 'dart:convert';
import 'package:http/http.dart' as http;

/// Simple DTOs used by Admin Panel UI
class ModelInfo {
  final String version;
  final bool isActive;

  const ModelInfo({required this.version, required this.isActive});

  factory ModelInfo.fromJson(Map<String, dynamic> json) {
    return ModelInfo(
      version: (json['version'] ?? '').toString(),
      isActive: (json['is_active'] ?? json['isActive'] ?? false) == true,
    );
  }
}

class TrainingStatus {
  final bool isTraining;
  final String? activeModel;
  final String? lastTrainedModel;
  final String? message;

  const TrainingStatus({
    required this.isTraining,
    this.activeModel,
    this.lastTrainedModel,
    this.message,
  });

  factory TrainingStatus.fromJson(Map<String, dynamic> json) {
    return TrainingStatus(
      isTraining: (json['is_training'] ?? json['isTraining'] ?? false) == true,
      activeModel: json['active_model']?.toString(),
      lastTrainedModel: json['last_trained_model']?.toString(),
      message: json['message']?.toString(),
    );
  }
}

class TrainingEvent {
  final DateTime ts;
  final String level;
  final String message;

  const TrainingEvent({
    required this.ts,
    required this.level,
    required this.message,
  });

  factory TrainingEvent.fromJson(Map<String, dynamic> json) {
    final rawTs = (json['ts'] ?? '').toString();
    DateTime parsed;
    try {
      parsed = DateTime.parse(rawTs).toLocal();
    } catch (_) {
      parsed = DateTime.now();
    }
    return TrainingEvent(
      ts: parsed,
      level: (json['level'] ?? 'info').toString(),
      message: (json['message'] ?? '').toString(),
    );
  }
}

class AdminService {
  // Keep in sync with main.dart baseUrl
  final String baseUrl;

  const AdminService({required this.baseUrl});

  Uri _u(String path, [Map<String, String>? q]) {
    return Uri.parse('$baseUrl$path').replace(queryParameters: q);
  }

  Future<List<ModelInfo>> fetchModels()  async {
    final resp = await http.get(_u('/admin/models'));
    if (resp.statusCode != 200) {
      throw Exception('HTTP ${resp.statusCode}: models');
    }
    final data = jsonDecode(resp.body) as Map<String, dynamic>;
    final list = (data['models'] as List?) ?? [];
    return list.map((e) => ModelInfo.fromJson(Map<String, dynamic>.from(e as Map))).toList();
  }

  Future<TrainingStatus> fetchTrainingStatus() async {
    final resp = await http.get(_u('/admin/training-status'));
    if (resp.statusCode != 200) {
      throw Exception('HTTP ${resp.statusCode}: training-status');
    }
    return TrainingStatus.fromJson(jsonDecode(resp.body) as Map<String, dynamic>);
  }

  Future<void> setActiveModel(String version) async {
    final resp = await http.post(
      _u('/admin/set-active-model'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'model_version': version}),
    );
    if (resp.statusCode != 200) {
      throw Exception('HTTP ${resp.statusCode}: set-active-model');
    }
  }

  Future<void> requestRetrain() async {
    final resp = await http.post(_u('/admin/request-retrain'));
    if (resp.statusCode != 200) {
      throw Exception('HTTP ${resp.statusCode}: request-retrain');
    }
  }

  Future<List<TrainingEvent>> fetchTrainingEvents({int limit = 15}) async {
    final resp = await http.get(_u('/admin/training-events', {'limit': '$limit'}));
    if (resp.statusCode == 404) {
      // Endpoint not deployed on backend (yet)
      return const <TrainingEvent>[];
    }
    if (resp.statusCode != 200) {
      throw Exception('HTTP ${resp.statusCode}: training-events');
    }
    final data = jsonDecode(resp.body) as Map<String, dynamic>;
    final list = (data['events'] as List?) ?? [];
    return list.map((e) => TrainingEvent.fromJson(Map<String, dynamic>.from(e as Map))).toList();
  }
}
