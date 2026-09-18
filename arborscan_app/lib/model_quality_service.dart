import 'dart:convert';
import 'package:http/http.dart' as http;
import 'corrections_service.dart';
import 'report_history_service.dart';

/// Reuses account guards while keeping ML failures distinct from report saving.
class ModelQualityService extends ReportHistoryService {
  ModelQualityService({super.auth, super.clientFactory});

  @override
  Future<Map<String, dynamic>> request(
      String token, http.BaseRequest request) async {
    await auth.checkSession(token);
    final generation = CorrectionsService.authChanges.value;
    final client = clientFactory();
    try {
      request.headers['Authorization'] = 'Bearer $token';
      final response = await (() async =>
              http.Response.fromStream(await client.send(request)))()
          .timeout(const Duration(seconds: 120));
      await auth.checkSession(token);
      if (generation != CorrectionsService.authChanges.value) {
        throw const CorrectionException(
            'Аккаунт изменился. Откройте раздел заново.');
      }
      if (response.statusCode < 200 || response.statusCode >= 300) {
        final message = switch (response.statusCode) {
          401 => 'Сессия истекла. Войдите снова.',
          403 => 'Для этой операции нужны права администратора.',
          404 || 405 => 'Новая функция или запись недоступна на этом сервере.',
          409 =>
            'Конфликт: обновите список. Возможно, задача уже запущена или версия изменилась.',
          422 =>
            'Проверьте поля и данные: для обучения нужны независимые train, validation и test; для классификации — по каждому виду.',
          503 => 'ML-сервис или миграция недоступны. Повторите позже.',
          _ => 'Операция не подтверждена сервером (${response.statusCode}).',
        };
        throw CorrectionException(message, statusCode: response.statusCode);
      }
      return Map<String, dynamic>.from(
          jsonDecode(utf8.decode(response.bodyBytes)));
    } on CorrectionException {
      rethrow;
    } catch (_) {
      throw const CorrectionException(
          'Нет ответа ML-сервиса. Повторите действие: идентификатор запроса сохранён.');
    } finally {
      client.close();
    }
  }
}
