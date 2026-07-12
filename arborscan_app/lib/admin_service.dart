import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

class AdminApiException implements Exception {
  final int? statusCode;
  final String message;

  const AdminApiException(this.message, {this.statusCode});

  bool get isUnauthorized => statusCode == 401;
  bool get isForbidden => statusCode == 403;

  @override
  String toString() => message;
}

class AdminIdentity {
  final String id;
  final String name;
  final String email;
  final String role;

  const AdminIdentity({
    required this.id,
    required this.name,
    required this.email,
    required this.role,
  });

  factory AdminIdentity.fromJson(Map<String, dynamic> json) {
    return AdminIdentity(
      id: (json['id'] ?? '').toString(),
      name: (json['name'] ?? '').toString(),
      email: (json['email'] ?? '').toString(),
      role: (json['role'] ?? 'user').toString(),
    );
  }
}

class TrainingStatus {
  final bool isTraining;
  final bool retrainRequested;
  final int? activeModelVersion;
  final int? lastTrainedVersion;
  final String? lastError;
  final String? trainingStartedAt;
  final String? trainingCompletedAt;

  bool get training => isTraining;
  int? get activeModel => activeModelVersion;
  int? get lastTrained => lastTrainedVersion;
  int? get lastTrainedModelVersion => lastTrainedVersion;

  const TrainingStatus({
    required this.isTraining,
    required this.retrainRequested,
    required this.activeModelVersion,
    required this.lastTrainedVersion,
    required this.lastError,
    required this.trainingStartedAt,
    required this.trainingCompletedAt,
  });

  factory TrainingStatus.fromJson(Map<String, dynamic> json) {
    int? asInt(dynamic value) {
      if (value == null) return null;
      if (value is int) return value;
      if (value is num) return value.toInt();
      return int.tryParse(value.toString());
    }

    bool asBool(dynamic value) {
      if (value is bool) return value;
      final normalized = value?.toString().trim().toLowerCase();
      return normalized == 'true' || normalized == '1';
    }

    return TrainingStatus(
      isTraining: asBool(
        json['training_in_progress'] ??
            json['is_training'] ??
            json['training'],
      ),
      retrainRequested: asBool(json['retrain_requested']),
      activeModelVersion: asInt(
        json['active_model_version'] ?? json['active_model'],
      ),
      lastTrainedVersion: asInt(
        json['last_model_version'] ??
            json['last_trained_version'] ??
            json['last_trained'],
      ),
      lastError: json['last_error']?.toString(),
      trainingStartedAt: json['training_started_at']?.toString(),
      trainingCompletedAt: json['training_completed_at']?.toString(),
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
    Map<String, dynamic> asMap(dynamic value) {
      if (value is Map<String, dynamic>) return value;
      if (value is Map) return value.cast<String, dynamic>();
      return <String, dynamic>{};
    }

    return TrainingEvent(
      ts: (json['ts'] ?? json['time'] ?? '').toString(),
      level: (json['level'] ?? 'INFO').toString(),
      message: (json['message'] ?? '').toString(),
      meta: asMap(json['data'] ?? json['meta']),
    );
  }
}

class ModelsResponse {
  final List<int> models;
  final int? activeModelVersion;

  const ModelsResponse({
    required this.models,
    required this.activeModelVersion,
  });

  factory ModelsResponse.fromJson(Map<String, dynamic> json) {
    final raw = json['models'];
    final versions = <int>[];
    int? activeFromItems;

    if (raw is List) {
      for (final item in raw) {
        if (item is num) {
          versions.add(item.toInt());
          continue;
        }
        if (item is Map) {
          final value = item['version'] ?? item['model_version'] ?? item['id'];
          final version = value == null ? null : int.tryParse(value.toString());
          if (version != null) {
            versions.add(version);
            final isActive =
                item['is_active'] ?? item['isActive'] ?? item['active'];
            if (isActive == true) activeFromItems = version;
          }
          continue;
        }
        final version = int.tryParse(item.toString());
        if (version != null) versions.add(version);
      }
    }

    final deduplicated = versions.toSet().toList()..sort();
    final activeRaw = json['active_model_version'] ?? json['active_model'];
    final active = activeRaw == null
        ? activeFromItems
        : int.tryParse(activeRaw.toString());

    return ModelsResponse(
      models: deduplicated,
      activeModelVersion: active,
    );
  }
}

class AdminService {
  static const String _tokenKey = 'arborscan_auth_token';

  final String baseUrl;
  final Future<String?> Function()? tokenProvider;

  const AdminService({
    required this.baseUrl,
    this.tokenProvider,
  });

  Uri _uri(String path, [Map<String, dynamic>? query]) {
    final normalizedPath = path.startsWith('/') ? path : '/$path';
    return Uri.parse('$baseUrl$normalizedPath').replace(
      queryParameters: query?.map(
        (key, value) => MapEntry(key, value.toString()),
      ),
    );
  }

  Future<String> _loadToken() async {
    final supplied = await tokenProvider?.call();
    if (supplied != null && supplied.trim().isNotEmpty) {
      return supplied.trim();
    }

    final preferences = await SharedPreferences.getInstance();
    final token = preferences.getString(_tokenKey)?.trim() ?? '';
    if (token.isEmpty) {
      throw const AdminApiException(
        'Сначала войдите в профиль администратора.',
        statusCode: 401,
      );
    }
    return token;
  }

  Future<Map<String, String>> _headers({bool jsonBody = false}) async {
    final token = await _loadToken();
    return <String, String>{
      'Authorization': 'Bearer $token',
      'Accept': 'application/json',
      if (jsonBody) 'Content-Type': 'application/json',
    };
  }

  Map<String, dynamic> _decodeObject(http.Response response) {
    if (response.bodyBytes.isEmpty) return <String, dynamic>{};
    final decoded = jsonDecode(utf8.decode(response.bodyBytes));
    if (decoded is Map<String, dynamic>) return decoded;
    if (decoded is Map) return decoded.cast<String, dynamic>();
    return <String, dynamic>{};
  }

  Never _throwResponse(http.Response response, String operation) {
    String detail = '';
    try {
      final decoded = _decodeObject(response);
      detail = (decoded['detail'] ?? decoded['message'] ?? '').toString();
    } catch (_) {
      detail = utf8.decode(response.bodyBytes, allowMalformed: true).trim();
    }

    if (response.statusCode == 401) {
      throw const AdminApiException(
        'Сессия отсутствует или истекла. Войдите снова.',
        statusCode: 401,
      );
    }
    if (response.statusCode == 403) {
      throw const AdminApiException(
        'У текущего профиля нет прав администратора.',
        statusCode: 403,
      );
    }

    throw AdminApiException(
      detail.isEmpty
          ? 'Ошибка $operation: HTTP ${response.statusCode}.'
          : 'Ошибка $operation: $detail',
      statusCode: response.statusCode,
    );
  }

  Future<http.Response> _get(
    String path, {
    Map<String, dynamic>? query,
    Duration timeout = const Duration(seconds: 30),
  }) async {
    final response = await http
        .get(
          _uri(path, query),
          headers: await _headers(),
        )
        .timeout(timeout);
    if (response.statusCode < 200 || response.statusCode >= 300) {
      _throwResponse(response, path);
    }
    return response;
  }

  Future<http.Response> _post(
    String path, {
    Map<String, dynamic>? body,
    Duration timeout = const Duration(seconds: 30),
  }) async {
    final response = await http
        .post(
          _uri(path),
          headers: await _headers(jsonBody: body != null),
          body: body == null ? null : jsonEncode(body),
        )
        .timeout(timeout);
    if (response.statusCode < 200 || response.statusCode >= 300) {
      _throwResponse(response, path);
    }
    return response;
  }

  Future<AdminIdentity> verifyAdminAccess() async {
    final response = await _get('/admin/me');
    final data = _decodeObject(response);
    final rawUser = data['user'];
    if (rawUser is! Map) {
      throw const AdminApiException(
        'Сервер не вернул данные администратора.',
      );
    }
    return AdminIdentity.fromJson(rawUser.cast<String, dynamic>());
  }

  Future<TrainingStatus> getTrainingStatus() async {
    final response = await _get('/admin/training-status');
    return TrainingStatus.fromJson(_decodeObject(response));
  }

  Future<List<TrainingEvent>> getTrainingEvents({int limit = 15}) async {
    final response = await _get(
      '/admin/training-events',
      query: {'limit': limit},
    );
    final data = _decodeObject(response);
    final rawEvents = data['events'];
    final events = <TrainingEvent>[];
    if (rawEvents is List) {
      for (final raw in rawEvents) {
        if (raw is Map<String, dynamic>) {
          events.add(TrainingEvent.fromJson(raw));
        } else if (raw is Map) {
          events.add(TrainingEvent.fromJson(raw.cast<String, dynamic>()));
        }
      }
    }
    return events;
  }

  Future<ModelsResponse> getModels() async {
    final response = await _get('/admin/models');
    return ModelsResponse.fromJson(_decodeObject(response));
  }

  Future<void> setActiveModel(int modelVersion) async {
    await _post(
      '/admin/set-active-model',
      body: {'model_version': modelVersion},
      timeout: const Duration(seconds: 120),
    );
  }

  Future<void> requestTraining() async {
    await _post('/admin/request-retrain');
  }

  Future<List<VerifiedItem>> getVerifiedList({bool includeUsed = false}) async {
    final response = await _get(
      '/admin/verified-list',
      query: {'include_used': includeUsed},
      timeout: const Duration(seconds: 45),
    );
    final data = _decodeObject(response);
    final rawItems = data['items'];
    final items = <VerifiedItem>[];
    if (rawItems is List) {
      for (final raw in rawItems) {
        if (raw is Map<String, dynamic>) {
          items.add(VerifiedItem.fromJson(raw));
        } else if (raw is Map) {
          items.add(VerifiedItem.fromJson(raw.cast<String, dynamic>()));
        }
      }
    }
    return items;
  }

  Future<VerifiedAnalysis> getVerifiedAnalysis(String analysisId) async {
    final response = await _get(
      '/admin/analysis/$analysisId',
      timeout: const Duration(seconds: 90),
    );
    return VerifiedAnalysis.fromJson(_decodeObject(response));
  }

  Future<void> setTrainingInclude(
    String analysisId, {
    required bool include,
  }) async {
    await _post(
      '/admin/verified/$analysisId/set-training',
      body: {'include': include},
    );
  }

  Future<void> verifyExample(
    String analysisId,
    dynamic points,
    dynamic closed,
  ) async {
    throw const AdminApiException(
      'Старая операция verifyExample не поддерживается. '
      'Используйте страницу датасета и подтверждённые маски.',
    );
  }
}

class VerifiedItem {
  final String analysisId;
  final bool verified;
  final bool excludeFromTraining;
  final bool hasUserMask;
  final bool usedForTraining;
  final String? species;
  final String? riskCategory;
  final num? trustScore;
  final String? verifiedAt;

  const VerifiedItem({
    required this.analysisId,
    required this.verified,
    required this.excludeFromTraining,
    this.hasUserMask = false,
    this.usedForTraining = false,
    required this.species,
    required this.riskCategory,
    required this.trustScore,
    required this.verifiedAt,
  });

  factory VerifiedItem.fromJson(Map<String, dynamic> json) {
    return VerifiedItem(
      analysisId: (json['analysis_id'] ?? json['analysisId'] ?? '').toString(),
      verified: (json['verified'] ?? true) == true,
      excludeFromTraining:
          (json['exclude_from_training'] ??
                  json['excludeFromTraining'] ??
                  false) ==
              true,
      hasUserMask:
          (json['has_user_mask'] ?? json['hasUserMask'] ?? false) == true,
      usedForTraining:
          (json['used_for_training'] ?? json['usedForTraining'] ?? false) ==
              true,
      species: json['species']?.toString(),
      riskCategory:
          json['risk_category']?.toString() ?? json['riskCategory']?.toString(),
      trustScore: json['trust_score'] as num?,
      verifiedAt: json['verified_at']?.toString(),
    );
  }
}

class VerifiedAnalysis {
  final String analysisId;
  final Uint8List inputImage;
  final Uint8List annotatedImage;
  final Uint8List? userMaskImage;
  final Map<String, dynamic> meta;
  final Map<String, dynamic> treePred;
  final Map<String, dynamic> stickPred;

  const VerifiedAnalysis({
    required this.analysisId,
    required this.inputImage,
    required this.annotatedImage,
    this.userMaskImage,
    required this.meta,
    required this.treePred,
    required this.stickPred,
  });

  factory VerifiedAnalysis.fromJson(Map<String, dynamic> json) {
    final rawImages = json['images'];
    final images = rawImages is Map
        ? rawImages.cast<String, dynamic>()
        : <String, dynamic>{};

    Uint8List decodeRequired(String value, String fieldName) {
      if (value.isEmpty) {
        throw AdminApiException('В ответе отсутствует изображение $fieldName.');
      }
      return base64Decode(value);
    }

    final inputBase64 = (images['input_base64'] ?? '').toString();
    final annotatedBase64 = (images['annotated_base64'] ?? '').toString();
    final userMaskBase64 =
        (images['user_mask_base64'] ??
                images['user_mask_png_base64'] ??
                '')
            .toString();

    Map<String, dynamic> asMap(dynamic value) {
      if (value is Map<String, dynamic>) return value;
      if (value is Map) return value.cast<String, dynamic>();
      return <String, dynamic>{};
    }

    return VerifiedAnalysis(
      analysisId: (json['analysis_id'] ?? json['analysisId'] ?? '').toString(),
      inputImage: decodeRequired(inputBase64, 'input_base64'),
      annotatedImage: decodeRequired(annotatedBase64, 'annotated_base64'),
      userMaskImage:
          userMaskBase64.isEmpty ? null : base64Decode(userMaskBase64),
      meta: asMap(json['meta']),
      treePred: asMap(json['tree_pred']),
      stickPred: asMap(json['stick_pred']),
    );
  }
}
