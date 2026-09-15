import 'dart:async';
import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

import 'api_config.dart';

class CorrectionException implements Exception {
  final String message;
  const CorrectionException(this.message);
  @override
  String toString() => message;
}

class CorrectionsService {
  // AppRoot signals login/logout; requests also check preferences before and
  // after I/O so an old account's response is never accepted by a new session.
  static final authChanges = ValueNotifier<int>(0);
  final http.Client Function() clientFactory;
  final Duration timeout;
  CorrectionsService({http.Client Function()? clientFactory,
    this.timeout = const Duration(seconds: 120)})
      : clientFactory = clientFactory ?? http.Client.new;

  static Future<String> currentToken() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString('arborscan_auth_token')?.trim() ?? '';
  }

  Future<void> checkSession(String token) async {
    if (token.isEmpty) {
      throw const CorrectionException('Войдите в профиль для доступа к контурам.');
    }
    if (await currentToken() != token) {
      throw const CorrectionException('Аккаунт изменился. Откройте экран заново.');
    }
  }

  Future<Map<String, dynamic>> _send(http.BaseRequest request, String token) async {
    await checkSession(token);
    final generation = authChanges.value;
    final client = clientFactory();
    request.headers['Authorization'] = 'Bearer $token';
    try {
      final response = await (() async => http.Response.fromStream(
        await client.send(request)))().timeout(timeout);
      await checkSession(token);
      if (generation != authChanges.value) {
        throw const CorrectionException('Сессия изменилась. Откройте экран заново.');
      }
      if (response.statusCode == 401 || response.statusCode == 403) {
        throw const CorrectionException('Нет доступа. Войдите в профиль снова.');
      }
      if (response.statusCode < 200 || response.statusCode >= 300) {
        throw CorrectionException('Ошибка сервера (${response.statusCode}). Повторите попытку.');
      }
      final data = jsonDecode(utf8.decode(response.bodyBytes));
      if (data is! Map<String, dynamic>) throw const FormatException();
      return data;
    } on CorrectionException {
      rethrow;
    } on FormatException {
      throw const CorrectionException('Некорректный ответ сервера. Повторите попытку.');
    } catch (_) {
      throw const CorrectionException('Нет связи с сервером или истекло время ожидания. Повторите попытку.');
    } finally {
      client.close();
    }
  }

  Uri _uri([String suffix = '']) => ApiConfig.v4('/v4/corrections$suffix');

  Future<void> save({required String token, required String analysisId,
    required Uint8List image, required Uint8List mask}) async {
    final request = http.MultipartRequest('POST', _uri())
      ..fields['analysis_id'] = analysisId
      ..files.add(http.MultipartFile.fromBytes('image', image, filename: 'original.jpg'))
      ..files.add(http.MultipartFile.fromBytes('mask', mask, filename: 'mask.png'));
    final data = await _send(request, token);
    if (data['saved'] != true) {
      throw const CorrectionException('Сервер не подтвердил сохранение. Повторите попытку.');
    }
  }

  Future<Map<String, dynamic>> list(String token, int offset) async {
    final data = await _send(http.Request('GET', _uri().replace(
      queryParameters: {'offset': '$offset'})), token);
    final next = data['next_offset'];
    if (data['items'] is! List ||
        (data['items'] as List).any((item) => item is! Map || item['correction_id'] is! String) ||
        (next != null && (next is! int || next <= offset))) {
      throw const CorrectionException('Некорректный список контуров.');
    }
    return data;
  }

  Future<({Uint8List image, Uint8List mask})> detail(String token, String id) async {
    final data = await _send(http.Request('GET', _uri('/${Uri.encodeComponent(id)}')), token);
    try {
      final image = base64Decode(data['original_image_base64'] as String);
      final mask = base64Decode(data['mask_png_base64'] as String);
      if (image.isEmpty || mask.isEmpty) throw const FormatException();
      return (image: image, mask: mask);
    } catch (_) {
      throw const CorrectionException('Не удалось прочитать фото или PNG-маску.');
    }
  }
}
