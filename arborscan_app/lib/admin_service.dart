import 'dart:convert';

import 'package:http/http.dart' as http;

class TrainingStatus {
  final bool isTraining;
  final int? activeModelVersion;
  final int? lastTrainedVersion;

  // Compatibility aliases for older/newer UI code.
  bool get training => isTraining;
  int? get activeModel => activeModelVersion;
  int? get lastTrained => lastTrainedVersion;
  int? get lastTrainedModelVersion => lastTrainedVersion;

  const TrainingStatus({
    required this.isTraining,
    required this.activeModelVersion,
    required this.lastTrainedVersion,
  });

  factory TrainingStatus.fromJson(Map<String, dynamic> json) {
    int? asInt(dynamic v) {
      if (v == null) return null;
      if (v is int) return v;
      if (v is num) return v.toInt();
      return int.tryParse(v.toString());
    }

    return TrainingStatus(
      isTraining: (json['is_training'] ?? json['training'] ?? false) == true,
      activeModelVersion: asInt(json['active_model_version'] ?? json['active_model']),
      lastTrainedVersion: asInt(json['last_trained_version'] ?? json['last_trained']),
    );
  }
}

class TrainingEvent {
  final String ts;
  final String level;
  final String message;
  final Map<String, dynamic> meta;

  const TrainingEvent({
    required this.ts,
    required this.level,
    required this.message,
    required this.meta,
  });

  factory TrainingEvent.fromJson(Map<String, dynamic> json) {
    return TrainingEvent(
      ts: (json['ts'] ?? json['time'] ?? '').toString(),
      level: (json['level'] ?? 'INFO').toString(),
      message: (json['message'] ?? '').toString(),
      meta: (json['meta'] is Map<String, dynamic>)
          ? (json['meta'] as Map<String, dynamic>)
          : <String, dynamic>{},
    );
  }
}

class ModelsResponse {
  final List<int> models;
  final int? activeModelVersion;

  const ModelsResponse({required this.models, required this.activeModelVersion});

  factory ModelsResponse.fromJson(Map<String, dynamic> json) {
    final raw = json['models'];
    final List<int> versions = [];
    int? activeFromItems;
    if (raw is List) {
      for (final v in raw) {
        // Backend may return either:
        // - [1,2,3]
        // - [{"version":1,"is_active":true}, ...]
        if (v is num) {
          versions.add(v.toInt());
          continue;
        }
        if (v is Map) {
          final verVal = v['version'] ?? v['model_version'] ?? v['id'];
          final ver = verVal == null ? null : int.tryParse(verVal.toString());
          if (ver != null) {
            versions.add(ver);
            final isActive = v['is_active'] ?? v['isActive'] ?? v['active'];
            if (isActive == true) activeFromItems = ver;
          }
          continue;
        }
        final parsed = int.tryParse(v.toString());
        if (parsed != null) versions.add(parsed);
      }
    }
    versions.sort();
    final unique = <int>{};
    final deduped = <int>[];
    for (final v in versions) {
      if (unique.add(v)) deduped.add(v);
    }

    int? active;
    final av = json['active_model_version'] ?? json['active_model'];
    if (av != null) {
      active = int.tryParse(av.toString());
    } else {
      active = activeFromItems;
    }

    return ModelsResponse(models: deduped, activeModelVersion: active);
  }
}

class AdminService {
  final String baseUrl;

  const AdminService({required this.baseUrl});

  Uri _u(String path, [Map<String, dynamic>? q]) {
    final p = path.startsWith('/') ? path : '/$path';
    return Uri.parse('$baseUrl$p').replace(
      queryParameters: q?.map((k, v) => MapEntry(k, v.toString())),
    );
  }

  Future<TrainingStatus> getTrainingStatus() async {
    final r = await http
        .get(_u('/admin/training-status'))
        .timeout(const Duration(seconds: 30));
    if (r.statusCode != 200) {
      throw Exception('HTTP ${r.statusCode}: training-status');
    }
    return TrainingStatus.fromJson(jsonDecode(r.body) as Map<String, dynamic>);
  }

  Future<List<TrainingEvent>> getTrainingEvents({int limit = 15}) async {
    final r = await http
        .get(_u('/admin/training-events', {'limit': limit}))
        .timeout(const Duration(seconds: 30));
    if (r.statusCode != 200) {
      throw Exception('HTTP ${r.statusCode}: training-events');
    }
    final decoded = jsonDecode(r.body);
    final eventsRaw = (decoded is Map<String, dynamic>) ? decoded['events'] : null;
    final List<TrainingEvent> events = [];
    if (eventsRaw is List) {
      for (final e in eventsRaw) {
        if (e is Map<String, dynamic>) events.add(TrainingEvent.fromJson(e));
      }
    }
    return events;
  }

  Future<ModelsResponse> getModels() async {
    final r = await http.get(_u('/admin/models')).timeout(const Duration(seconds: 30));
    if (r.statusCode != 200) {
      throw Exception('HTTP ${r.statusCode}: models');
    }
    return ModelsResponse.fromJson(jsonDecode(r.body) as Map<String, dynamic>);
  }

  Future<void> setActiveModel(int modelVersion) async {
    final r = await http
        .post(
          _u('/admin/set-active-model'),
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({'model_version': modelVersion}),
        )
        .timeout(const Duration(seconds: 30));
    if (r.statusCode != 200) {
      throw Exception('HTTP ${r.statusCode}: set-active-model');
    }
  }

  Future<void> requestTraining() async {
    final r = await http.post(_u('/admin/request-retrain')).timeout(const Duration(seconds: 30));
    if (r.statusCode != 200) {
      throw Exception('HTTP ${r.statusCode}: request-retrain');
    }
  }
}
