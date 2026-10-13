import 'dart:convert';
import 'dart:typed_data';
import 'dart:ui';

import 'package:http/http.dart' as http;

import 'api_config.dart';
import 'corrections_service.dart';

/// Runs the existing model on the exact original. It never invents a scale,
/// changes measurements from a previous response, or saves a server report.
class PhotoAnalysisService {
  final CorrectionsService auth;
  final http.Client Function() clientFactory;
  final Duration timeout;
  PhotoAnalysisService(
      {CorrectionsService? auth,
      http.Client Function()? clientFactory,
      this.timeout = const Duration(seconds: 180)})
      : auth = auth ?? CorrectionsService(),
        clientFactory = clientFactory ?? http.Client.new;

  Future<Map<String, dynamic>> analyze(
      {required String token, required Uint8List image, Offset? tap}) async {
    // Copy synchronously: callers may change a selected image while I/O awaits.
    final original = Uint8List.fromList(image);
    if (original.isEmpty || original.length > 15 * 1024 * 1024) {
      throw const CorrectionException('Фото пустое или больше 15 МБ.');
    }
    if (tap != null &&
        (!tap.dx.isFinite ||
            !tap.dy.isFinite ||
            tap.dx < 0 ||
            tap.dx > 1 ||
            tap.dy < 0 ||
            tap.dy > 1)) {
      throw const CorrectionException('Точка выбора дерева вне фото.');
    }
    final generation = CorrectionsService.authChanges.value;
    await auth.checkSession(token);
    final request =
        http.MultipartRequest('POST', ApiConfig.v4('/v4/analyze-tree'))
          ..headers['Authorization'] = 'Bearer $token'
          ..fields['include_images'] = 'true'
          ..files.add(http.MultipartFile.fromBytes('file', original,
              filename: 'original.jpg'));
    if (tap != null) {
      request.fields['tap_x'] = '${tap.dx}';
      request.fields['tap_y'] = '${tap.dy}';
    }
    final client = clientFactory();
    try {
      final response = await (() async =>
              http.Response.fromStream(await client.send(request)))()
          .timeout(timeout);
      await auth.checkSession(token);
      if (generation != CorrectionsService.authChanges.value) {
        throw const CorrectionException(
            'Аккаунт изменился. Откройте экран заново.');
      }
      if (response.statusCode < 200 || response.statusCode >= 300) {
        throw CorrectionException(
            response.statusCode == 401 || response.statusCode == 403
                ? 'Войдите в свой аккаунт заново. Разметка сохранена на устройстве.'
                : 'Модель недоступна (${response.statusCode}). Разметка сохранена; повторите позже.',
            statusCode: response.statusCode);
      }
      final data = jsonDecode(utf8.decode(response.bodyBytes));
      if (data is! Map<String, dynamic> ||
          data['tree'] is! Map ||
          data['tree']['detected'] is! bool) {
        throw const FormatException();
      }
      if ((data['warnings'] as List? ?? [])
          .contains('tap_did_not_hit_mask_nearest_tree_selected')) {
        throw const CorrectionException(
            'Модель не выделила выбранное дерево и предложила соседнее. Результат не применён. Выберите крону нужного дерева или создайте контур вручную.');
      }
      return data;
    } on CorrectionException {
      rethrow;
    } on FormatException {
      throw const CorrectionException(
          'Сервер вернул некорректный результат модели. Разметка сохранена.');
    } catch (_) {
      throw const CorrectionException(
          'Нет связи с моделью. Разметка сохранена; повторите позже.');
    } finally {
      client.close();
    }
  }
}
