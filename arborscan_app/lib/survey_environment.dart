import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:image/image.dart' as im;
import 'api_config.dart';
import 'corrections_service.dart';

Map<String, dynamic> surveyMap(dynamic v) =>
    v is Map ? Map<String, dynamic>.from(v) : <String, dynamic>{};
Map<String, dynamic> copySurveyMap(Map<String, dynamic> value) =>
    Map<String, dynamic>.from(jsonDecode(jsonEncode(value)));

/// A saved observation's position, never an implicit map center or phone fallback.
class SurveyPoint {
  final double lat, lon;
  final String source, retrievedAt, positionKind;
  final String? observedAt, capturedLocal;
  final double? accuracyM;
  final bool isLastKnown;
  final bool? isApproximate;
  const SurveyPoint(
      {required this.lat,
      required this.lon,
      required this.source,
      required this.retrievedAt,
      this.positionKind = 'camera_or_device',
      this.observedAt,
      this.capturedLocal,
      this.accuracyM,
      this.isLastKnown = false,
      this.isApproximate});

  static bool valid(double lat, double lon) =>
      lat.isFinite &&
      lon.isFinite &&
      lat >= -90 &&
      lat <= 90 &&
      lon >= -180 &&
      lon <= 180;
  static SurveyPoint? fromJson(dynamic raw) {
    final m = surveyMap(raw), value = surveyMap(surveyMap(raw)['value']);
    final v = value.isEmpty ? m : value;
    final lat = v['lat'] ?? v['latitude'], lon = v['lon'] ?? v['longitude'];
    if (lat is! num || lon is! num || !valid(lat.toDouble(), lon.toDouble())) {
      return null;
    }
    if (v['crs'] != null && v['crs'] != 'EPSG:4326') return null;
    final accuracy = m['accuracy_m'];
    return SurveyPoint(
        lat: lat.toDouble(),
        lon: lon.toDouble(),
        source: m['source']?.toString() ?? 'legacy',
        retrievedAt: m['retrieved_at']?.toString() ?? '',
        observedAt: m['observed_at']?.toString(),
        capturedLocal: m['captured_local']?.toString(),
        positionKind: m['position_kind']?.toString() ?? 'camera_or_device',
        isLastKnown: m['is_last_known'] == true,
        isApproximate:
            m['is_approximate'] is bool ? m['is_approximate'] as bool : null,
        accuracyM: accuracy is num && accuracy.isFinite && accuracy >= 0
            ? accuracy.toDouble()
            : null);
  }

  Map<String, dynamic> toJson() => {
        'value': {'lat': lat, 'lon': lon, 'crs': 'EPSG:4326'},
        'source': source,
        'retrieved_at': retrievedAt,
        'observed_at': observedAt,
        'accuracy_m': accuracyM,
        'is_last_known': isLastKnown,
        'position_kind': positionKind,
        if (capturedLocal != null) 'captured_local': capturedLocal,
        if (isApproximate != null) 'is_approximate': isApproximate,
      };
  String get coordinates =>
      '${lat.toStringAsFixed(6)}, ${lon.toStringAsFixed(6)}';
  String get label => switch (source) {
        'exif' => 'Место съёмки из EXIF',
        'device' => 'Положение устройства',
        'manual' => 'Точка дерева выбрана вручную',
        _ => 'Сохранённые координаты',
      };
  String get identity => '$lat,$lon';
}

/// Reads JPEG metadata without decoding/resizing/orienting pixels.
/// No EXIF accuracy is invented, and an unknown timezone stays unknown.
SurveyPoint? surveyPointFromExif(Uint8List bytes, {DateTime? retrievedAt}) {
  try {
    final exif = im.decodeJpgExif(bytes);
    if (exif == null) return null;
    final gps = exif.gpsIfd;
    double? angle(int tag, int refTag, String positive, String negative) {
      final value = gps[tag];
      final ref = gps[refTag]?.toString().replaceAll('\u0000', '').trim();
      if (value == null ||
          value.type != im.IfdValueType.rational ||
          value.length != 3 ||
          (ref != positive && ref != negative)) {
        return null;
      }
      final d = value.toDouble(0), m = value.toDouble(1), s = value.toDouble(2);
      if (![d, m, s].every((v) => v.isFinite && v >= 0) || m >= 60 || s >= 60) {
        return null;
      }
      return (d + m / 60 + s / 3600) * (ref == negative ? -1 : 1);
    }

    final lat = angle(2, 1, 'N', 'S'), lon = angle(4, 3, 'E', 'W');
    if (lat == null || lon == null || !SurveyPoint.valid(lat, lon)) return null;
    final local =
        exif.exifIfd[0x9003]?.toString().replaceAll('\u0000', '').trim();
    final offset =
        exif.exifIfd[0x9011]?.toString().replaceAll('\u0000', '').trim();
    String? observed;
    if (local != null &&
        RegExp(r'^\d{4}:\d{2}:\d{2} \d{2}:\d{2}:\d{2}$').hasMatch(local) &&
        offset != null &&
        RegExp(r'^[+-]\d{2}:\d{2}$').hasMatch(offset)) {
      final date =
          '${local.substring(0, 10).replaceAll(':', '-')}T${local.substring(11)}$offset';
      observed = DateTime.tryParse(date)?.toUtc().toIso8601String();
    }
    final localIso = local != null &&
            RegExp(r'^\d{4}:\d{2}:\d{2} \d{2}:\d{2}:\d{2}$').hasMatch(local)
        ? '${local.substring(0, 10).replaceAll(':', '-')}T${local.substring(11)}'
        : null;
    return SurveyPoint(
        lat: lat,
        lon: lon,
        source: 'exif',
        retrievedAt: (retrievedAt ?? DateTime.now()).toUtc().toIso8601String(),
        observedAt: observed,
        capturedLocal: observed == null ? localIso : null);
  } catch (_) {
    return null;
  }
}

String environmentReason(dynamic code) => switch (code) {
      'provider_not_configured' => 'Источник пока не настроен на сервере',
      'provider_timeout' => 'Источник не ответил вовремя',
      'provider_rate_limited' => 'Лимит источника. Повторите позже',
      'provider_busy' => 'Источник занят. Повторите позже',
      'provider_authentication' => 'Сервер не получил доступ к источнику',
      'no_coverage' => 'Для этой точки данных нет',
      'partial_data' => 'Получена только часть параметров',
      'provider_invalid_response' => 'Источник вернул неподдерживаемые данные',
      _ => 'Источник временно недоступен',
    };

class EnvironmentService {
  final http.Client Function() clientFactory;
  final Future<void> Function(String) checkSession;
  EnvironmentService(
      {http.Client Function()? clientFactory,
      Future<void> Function(String)? checkSession})
      : clientFactory = clientFactory ?? http.Client.new,
        checkSession = checkSession ?? CorrectionsService().checkSession;
  Future<Map<String, dynamic>> current(String token, SurveyPoint point) async {
    if (!SurveyPoint.valid(point.lat, point.lon)) {
      throw const FormatException('Некорректные координаты.');
    }
    await checkSession(token);
    final generation = CorrectionsService.authChanges.value;
    final client = clientFactory();
    try {
      final response = await client.get(
          ApiConfig.v4('/v4/environment').replace(
              queryParameters: {'lat': '${point.lat}', 'lon': '${point.lon}'}),
          headers: {
            'Authorization': 'Bearer $token'
          }).timeout(const Duration(seconds: 18));
      await checkSession(token);
      if (generation != CorrectionsService.authChanges.value) {
        throw const CorrectionException('Аккаунт изменился.');
      }
      if ([404, 405].contains(response.statusCode)) {
        throw const CorrectionException(
            'Этот сервер пока не поддерживает условия среды. Координаты и отчёт можно сохранить.');
      }
      if ([401, 403].contains(response.statusCode)) {
        throw const CorrectionException(
            'Для получения условий войдите в свой аккаунт.');
      }
      if (response.statusCode == 429) {
        throw const CorrectionException(
            'Слишком много запросов. Повторите позже.');
      }
      if (response.statusCode != 200) {
        throw const CorrectionException(
            'Условия сейчас недоступны. Координаты и отчёт можно сохранить.');
      }
      final value = surveyMap(jsonDecode(utf8.decode(response.bodyBytes)));
      if (value['environment_version'] != 1) {
        throw const FormatException(
            'Сервер вернул неизвестный формат условий.');
      }
      for (final key in ['weather', 'soil']) {
        final requested = surveyMap(surveyMap(value[key])['request_point']);
        if (requested.isNotEmpty &&
            (requested['lat'] != point.lat || requested['lon'] != point.lon)) {
          throw const FormatException('Условия относятся к другой точке.');
        }
      }
      return {
        'version': 1,
        'gps': point.toJson(),
        'weather': value['weather'],
        'soil': value['soil']
      };
    } on CorrectionException {
      rethrow;
    } on FormatException {
      rethrow;
    } catch (_) {
      throw const CorrectionException(
          'Нет связи с источником условий. Сохранённые данные доступны.');
    } finally {
      client.close();
    }
  }
}

/// Owns one editor's snapshot. Changing position/account invalidates late results.
/// Saving reads a deep copy; no later request can mutate the saved version.
class SurveyEnvironmentController extends ChangeNotifier {
  Map<String, dynamic> _value;
  int _generation = 0;
  bool _disposed = false, busy = false;
  String? error;
  SurveyEnvironmentController([Map<String, dynamic>? initial])
      : _value = copySurveyMap(initial ?? {}) {
    CorrectionsService.authChanges.addListener(_accountChanged);
  }
  Map<String, dynamic> get snapshot => copySurveyMap(_value);
  SurveyPoint? get point => SurveyPoint.fromJson(_value['gps']);
  void _accountChanged() {
    _generation++;
    _value = {};
    busy = false;
    error = null;
    notifyListeners();
  }

  void replace(Map<String, dynamic>? value) {
    _generation++;
    _value = copySurveyMap(value ?? {});
    busy = false;
    error = null;
    notifyListeners();
  }

  void setPoint(SurveyPoint? point) {
    _generation++;
    _value = point == null ? {} : {'version': 1, 'gps': point.toJson()};
    busy = false;
    error = null;
    notifyListeners();
  }

  Future<void> refresh(String token, {EnvironmentService? service}) async {
    final selected = point;
    if (selected == null || busy) return;
    final operation = ++_generation;
    busy = true;
    error = null;
    notifyListeners();
    try {
      final response =
          await (service ?? EnvironmentService()).current(token, selected);
      if (!_disposed && operation == _generation) {
        _value = copySurveyMap(response);
      }
    } catch (e) {
      if (!_disposed && operation == _generation) error = '$e';
    } finally {
      if (!_disposed && operation == _generation) {
        busy = false;
        notifyListeners();
      }
    }
  }

  @override
  void dispose() {
    _disposed = true;
    _generation++;
    CorrectionsService.authChanges.removeListener(_accountChanged);
    super.dispose();
  }
}
