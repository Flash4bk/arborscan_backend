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

  // ============================
  // Verified dataset management
  // ============================

  Future<List<VerifiedItem>> getVerifiedList() async {
    final r = await http.get(_u('/admin/verified-list')).timeout(const Duration(seconds: 60));
    if (r.statusCode != 200) {
      throw Exception('HTTP ${r.statusCode}: verified-list');
    }
    final decoded = jsonDecode(r.body);
    final raw = (decoded is Map<String, dynamic>) ? decoded['items'] : null;
    final out = <VerifiedItem>[];
    if (raw is List) {
      for (final e in raw) {
        if (e is Map<String, dynamic>) out.add(VerifiedItem.fromJson(e));
      }
    }
    return out;
  }

  Future<VerifiedAnalysis> getVerifiedAnalysis(String analysisId) async {
    final r = await http.get(_u('/admin/analysis/$analysisId')).timeout(const Duration(seconds: 60));
    if (r.statusCode != 200) {
      throw Exception('HTTP ${r.statusCode}: analysis/$analysisId');
    }
    return VerifiedAnalysis.fromJson(jsonDecode(r.body) as Map<String, dynamic>);
  }

  /// include=true  -> будет участвовать в обучении
  /// include=false -> будет исключён (exclude_from_training=true)
  Future<void> setTrainingInclude(String analysisId, {required bool include}) async {
    final r = await http
        .post(
          _u('/admin/verified/$analysisId/set-training'),
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({'include': include}),
        )
        .timeout(const Duration(seconds: 60));
    if (r.statusCode != 200) {
      throw Exception('HTTP ${r.statusCode}: set-training');
    }
  }
}

class VerifiedItem {
  final String analysisId;
  final String? species;
  final String? riskCategory;
  final double? trustScore;
  final bool verified;
  final String? verifiedAt;
  final bool excludeFromTraining;

  const VerifiedItem({
    required this.analysisId,
    required this.verified,
    required this.excludeFromTraining,
    this.species,
    this.riskCategory,
    this.trustScore,
    this.verifiedAt,
  });

  factory VerifiedItem.fromJson(Map<String, dynamic> json) {
    double? asDouble(dynamic v) {
      if (v == null) return null;
      if (v is num) return v.toDouble();
      return double.tryParse(v.toString());
    }

    return VerifiedItem(
      analysisId: (json['analysis_id'] ?? json['analysisId'] ?? '').toString(),
      species: json['species']?.toString(),
      riskCategory: (json['risk_category'] ?? json['riskCategory'])?.toString(),
      trustScore: asDouble(json['trust_score'] ?? json['trustScore']),
      verified: (json['verified'] ?? true) == true,
      verifiedAt: json['verified_at']?.toString(),
      excludeFromTraining: (json['exclude_from_training'] ?? false) == true,
    );
  }
}

class VerifiedAnalysis {
  final String analysisId;
  final String inputBase64;
  final String annotatedBase64;
  final Map<String, dynamic> treePred;
  final Map<String, dynamic> stickPred;
  final Map<String, dynamic> meta;

  const VerifiedAnalysis({
    required this.analysisId,
    required this.inputBase64,
    required this.annotatedBase64,
    required this.treePred,
    required this.stickPred,
    required this.meta,
  });

  factory VerifiedAnalysis.fromJson(Map<String, dynamic> json) {
    final images = (json['images'] is Map<String, dynamic>) ? (json['images'] as Map<String, dynamic>) : <String, dynamic>{};
    return VerifiedAnalysis(
      analysisId: (json['analysis_id'] ?? '').toString(),
      inputBase64: (images['input_base64'] ?? '').toString(),
      annotatedBase64: (images['annotated_base64'] ?? '').toString(),
      treePred: (json['tree_pred'] is Map<String, dynamic>) ? (json['tree_pred'] as Map<String, dynamic>) : <String, dynamic>{},
      stickPred: (json['stick_pred'] is Map<String, dynamic>) ? (json['stick_pred'] as Map<String, dynamic>) : <String, dynamic>{},
      meta: (json['meta'] is Map<String, dynamic>) ? (json['meta'] as Map<String, dynamic>) : <String, dynamic>{},
    );
  }
}
