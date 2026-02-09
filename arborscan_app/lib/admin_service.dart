import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;

/// Simple service wrapper around the backend admin API.
///
/// `baseUrl` must include scheme + host (and optional port), e.g.
///   http://192.168.1.10:8000
class AdminService {
  final String baseUrl;

  const AdminService({required this.baseUrl});

  String get _base => baseUrl.endsWith('/') ? baseUrl.substring(0, baseUrl.length - 1) : baseUrl;

  Uri _uri(String path, [Map<String, String>? query]) {
    final p = path.startsWith('/') ? path : '/$path';
    return Uri.parse('$_base$p').replace(queryParameters: query);
  }

  Future<Map<String, dynamic>> _getJson(String path, {Map<String, String>? query}) async {
    final res = await http.get(_uri(path, query));
    if (res.statusCode < 200 || res.statusCode >= 300) {
      throw Exception('HTTP ${res.statusCode}: ${res.body}');
    }
    return jsonDecode(res.body) as Map<String, dynamic>;
  }

  Future<Map<String, dynamic>> _postJson(String path, Map<String, dynamic> body) async {
    final res = await http.post(
      _uri(path),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode(body),
    );
    if (res.statusCode < 200 || res.statusCode >= 300) {
      throw Exception('HTTP ${res.statusCode}: ${res.body}');
    }
    return res.body.isEmpty ? <String, dynamic>{} : (jsonDecode(res.body) as Map<String, dynamic>);
  }

  Future<List<dynamic>> _getJsonList(String path, {Map<String, String>? query}) async {
    final res = await http.get(_uri(path, query));
    if (res.statusCode < 200 || res.statusCode >= 300) {
      throw Exception('HTTP ${res.statusCode}: ${res.body}');
    }
    final decoded = jsonDecode(res.body);
    if (decoded is List) return decoded;
    throw Exception('Expected JSON list, got: ${res.body}');
  }

  // --- Admin API ---

  Future<TrainingStatus> getTrainingStatus() async {
    final j = await _getJson('/admin/training-status');
    return TrainingStatus.fromJson(j);
  }

  Future<List<TrainingEvent>> getTrainingEvents({int limit = 15}) async {
    final l = await _getJsonList('/admin/training-events', query: {'limit': '$limit'});
    return l.map((e) => TrainingEvent.fromJson(e as Map<String, dynamic>)).toList();
  }

  Future<List<ModelInfo>> listModels() async {
    final l = await _getJsonList('/admin/models');
    return l.map((e) => ModelInfo.fromJson(e as Map<String, dynamic>)).toList();
  }

  Future<void> setActiveModel(int version) async {
    await _postJson('/admin/set-active-model', {'version': version});
  }

  Future<void> requestRetrain({bool force = false}) async {
    await _postJson('/admin/request-retrain', {'force': force});
  }

  Future<List<VerifiedItem>> getVerifiedList() async {
    final l = await _getJsonList('/admin/verified-list');
    return l.map((e) => VerifiedItem.fromJson(e as Map<String, dynamic>)).toList();
  }

  Future<VerifiedAnalysis> getVerifiedAnalysis(String analysisId) async {
    final j = await _getJson('/admin/analysis/$analysisId');
    return VerifiedAnalysis.fromJson(j);
  }

  Future<void> setTrainingFlag(String analysisId, {required bool includeInTraining}) async {
    // Backend expects exclude_from_training (true means excluded)
    await _postJson('/admin/verified/$analysisId/set-training', {
      'exclude_from_training': !includeInTraining,
    });
  }
}

// --- DTOs ---

class TrainingStatus {
  final bool isTraining;
  final int? activeModelVersion;
  final int? lastTrainedVersion;

  const TrainingStatus({
    required this.isTraining,
    this.activeModelVersion,
    this.lastTrainedVersion,
  });

  factory TrainingStatus.fromJson(Map<String, dynamic> j) {
    return TrainingStatus(
      isTraining: (j['is_training'] ?? j['isTraining'] ?? false) as bool,
      activeModelVersion: _asIntOrNull(j['active_model_version'] ?? j['activeModelVersion']),
      lastTrainedVersion: _asIntOrNull(j['last_model_version'] ?? j['lastModelVersion'] ?? j['last_trained_version'] ?? j['lastTrainedVersion']),
    );
  }
}

class TrainingEvent {
  final String type;
  final String message;
  final DateTime? ts;

  const TrainingEvent({required this.type, required this.message, this.ts});

  factory TrainingEvent.fromJson(Map<String, dynamic> j) {
    return TrainingEvent(
      type: (j['type'] ?? 'event').toString(),
      message: (j['message'] ?? '').toString(),
      ts: _asDateTimeOrNull(j['ts'] ?? j['timestamp'] ?? j['time']),
    );
  }
}

class ModelInfo {
  final int version;
  final String? storageKey;

  const ModelInfo({required this.version, this.storageKey});

  factory ModelInfo.fromJson(Map<String, dynamic> j) {
    return ModelInfo(
      version: _asInt(j['version'] ?? j['model_version'] ?? 0),
      storageKey: (j['key'] ?? j['storage_key'] ?? j['path'])?.toString(),
    );
  }
}

class VerifiedItem {
  final String analysisId;
  final bool verified;
  /// true => excluded from training
  final bool excludeFromTraining;
  /// true => already consumed in a completed training run
  final bool usedForTraining;
  final DateTime? createdAt;

  VerifiedItem({
    required this.analysisId,
    required this.verified,
    required this.excludeFromTraining,
    required this.usedForTraining,
    this.createdAt,
  });

  bool get includeInTraining => !excludeFromTraining;

  factory VerifiedItem.fromJson(Map<String, dynamic> j) {
    return VerifiedItem(
      analysisId: (j['analysis_id'] ?? j['id'] ?? j['analysisId']).toString(),
      verified: (j['verified'] ?? true) as bool,
      excludeFromTraining: (j['exclude_from_training'] ?? j['excludeFromTraining'] ?? false) as bool,
      usedForTraining: (j['used_for_training'] ?? j['usedForTraining'] ?? false) as bool,
      createdAt: _asDateTimeOrNull(j['created_at'] ?? j['createdAt'] ?? j['added_at'] ?? j['addedAt']),
    );
  }

  VerifiedItem copyWith({
    bool? excludeFromTraining,
    bool? usedForTraining,
    DateTime? createdAt,
  }) {
    return VerifiedItem(
      analysisId: analysisId,
      verified: verified,
      excludeFromTraining: excludeFromTraining ?? this.excludeFromTraining,
      usedForTraining: usedForTraining ?? this.usedForTraining,
      createdAt: createdAt ?? this.createdAt,
    );
  }
}

class VerifiedAnalysis {
  final String analysisId;
  final String? originalImageUrl;
  final String? annotatedImageUrl;
  // user corrected overlay preview if backend provides it
  final Uint8List? userMaskImage;
  final Map<String, dynamic> meta;

  VerifiedAnalysis({
    required this.analysisId,
    this.originalImageUrl,
    this.annotatedImageUrl,
    this.userMaskImage,
    required this.meta,
  });

  factory VerifiedAnalysis.fromJson(Map<String, dynamic> j) {
    final images = (j['images'] is Map<String, dynamic>) ? (j['images'] as Map<String, dynamic>) : <String, dynamic>{};
    final String? userMaskB64 = (images['user_mask_base64'] ?? images['userMaskBase64'] ?? images['user_mask'])?.toString();
    Uint8List? userMask;
    if (userMaskB64 != null && userMaskB64.isNotEmpty) {
      try {
        userMask = base64Decode(userMaskB64);
      } catch (_) {
        userMask = null;
      }
    }
    return VerifiedAnalysis(
      analysisId: (j['analysis_id'] ?? j['analysisId'] ?? j['id']).toString(),
      originalImageUrl: (images['original_url'] ?? images['originalUrl'] ?? images['original'])?.toString(),
      annotatedImageUrl: (images['annotated_url'] ?? images['annotatedUrl'] ?? images['annotated'])?.toString(),
      userMaskImage: userMask,
      meta: (j['meta'] is Map<String, dynamic>) ? (j['meta'] as Map<String, dynamic>) : <String, dynamic>{},
    );
  }
}

int _asInt(dynamic v) {
  if (v is int) return v;
  if (v is num) return v.toInt();
  return int.tryParse(v?.toString() ?? '') ?? 0;
}

int? _asIntOrNull(dynamic v) {
  if (v == null) return null;
  final s = v.toString();
  final n = int.tryParse(s);
  return n;
}

DateTime? _asDateTimeOrNull(dynamic v) {
  if (v == null) return null;
  if (v is DateTime) return v;
  final s = v.toString();
  if (s.isEmpty) return null;
  try {
    return DateTime.parse(s);
  } catch (_) {
    return null;
  }
}
