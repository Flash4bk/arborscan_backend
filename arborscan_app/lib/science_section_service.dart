import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import 'api_config.dart';
import 'corrections_service.dart';
import 'report_history_service.dart';
import 'science_section.dart';

/// Small account-owned input journal. The API calculates and saves the result;
/// this journal only preserves form fields and the exact pending operation.
class ScienceSectionService {
  final ReportHistoryService history;
  CorrectionsService get auth => history.auth;
  ScienceSectionService({ReportHistoryService? history})
      : history = history ?? ReportHistoryService();
  static Future<void>? _writes;
  Future<T> _serial<T>(Future<T> Function() action) {
    final next = (_writes ?? Future<void>.value()).then((_) => action());
    late final Future<void> settled;
    void release() {
      if (identical(_writes, settled)) _writes = null;
    }

    settled = next.then<void>((_) => release(),
        onError: (Object _, StackTrace __) => release());
    _writes = settled;
    return next;
  }

  String _key(String owner, String source) =>
      'arborscan_section_draft_v1_$owner/$source';
  Future<Map<String, dynamic>?> _read(String owner, String source) async {
    if (!sectionVersionId(source)) {
      throw const CorrectionException('Неверная исходная версия отчёта.');
    }
    final encoded =
        (await SharedPreferences.getInstance()).getString(_key(owner, source));
    if (encoded == null) return null;
    final data = jsonDecode(encoded);
    if (data is! Map<String, dynamic> ||
        data['schema_version'] != 1 ||
        data['source_version_id'] != source) {
      throw const CorrectionException(
          'Черновик сценария имеет неподдержанный формат.');
    }
    return data;
  }

  Future<void> _write(String token, String owner, String source,
      Map<String, dynamic> data) async {
    await auth.checkSession(token);
    final saved = await (await SharedPreferences.getInstance())
        .setString(_key(owner, source), jsonEncode(data));
    if (!saved) {
      throw const CorrectionException(
          'Не удалось сохранить черновик на устройстве. Запрос не отправлен.');
    }
    await auth.checkSession(token);
  }

  Future<Map<String, dynamic>?> draft(String token, String source) async {
    final owner = await auth.owner(token);
    await _writes;
    final data = await _read(owner, source);
    await auth.checkSession(token);
    return data;
  }

  Future<Map<String, dynamic>> saveForm(
      String token, String source, Map<String, dynamic> form) {
    final frozen = Map<String, dynamic>.from(jsonDecode(jsonEncode(form)));
    return _serial(() async {
      final owner = await auth.owner(token);
      final old = await _read(owner, source);
      if (old?['attempted'] == true || old?['saved'] == true) {
        if (!sameScienceInputs(old!['form'], frozen)) {
          throw const CorrectionException(
              'Сначала повторите прежнее сохранение или откройте его сохранённую версию.');
        }
        return old;
      }
      final data = <String, dynamic>{
        ...?old,
        'schema_version': 1,
        'source_version_id': source,
        'form': frozen,
        'attempted': false,
        'saved': false,
      };
      await _write(token, owner, source, data);
      return data;
    });
  }

  Future<bool> available(String token) async {
    try {
      final result = await history.request(
          token, http.Request('GET', ApiConfig.v4('/v4/science/capabilities')));
      return result['science_version'] == 1 &&
          result['methods'] is List &&
          (result['methods'] as List).contains(sectionMethodId);
    } on CorrectionException catch (error) {
      if ([404, 405].contains(error.statusCode)) return false;
      rethrow;
    }
  }

  /// Only a definite server validation rejection permits a different intent.
  /// A timeout, 5xx, conflict or missing reply retains the exact pending ID.
  Future<Map<String, dynamic>> editRejected(String token, String source) =>
      _serial(() async {
        final owner = await auth.owner(token);
        final old = await _read(owner, source);
        if (old == null ||
            old['validation_rejected'] != true ||
            old['saved'] == true) {
          throw const CorrectionException(
              'Исход прежнего запроса не подтверждён. Повторите тот же запрос; входы не заменены.');
        }
        final data = <String, dynamic>{
          ...old,
          'attempted': false,
          'saved': false,
          'validation_rejected': false,
          'previous_rejected_version_id': old['version_id']
        }
          ..remove('version_id')
          ..remove('inputs')
          ..remove('science_result');
        await _write(token, owner, source, data);
        return data;
      });

  Future<Map<String, dynamic>> calculate(
      String token, String source, Map<String, dynamic> inputs) async {
    final frozen = Map<String, dynamic>.from(jsonDecode(jsonEncode(inputs)));
    final operation = await _serial(() async {
      final owner = await auth.owner(token);
      final old = await _read(owner, source);
      if (old == null) {
        throw const CorrectionException('Сначала сохраните черновик входов.');
      }
      if (old['inputs'] != null &&
          (old['attempted'] == true || old['saved'] == true) &&
          !sameScienceInputs(old['inputs'], frozen)) {
        throw const CorrectionException(
            'Нельзя менять входы запроса с неизвестным результатом. Повторите прежний запрос.');
      }
      final data = <String, dynamic>{
        ...old,
        'version_id': sectionVersionId(old['version_id']) &&
                old['inputs'] != null &&
                sameScienceInputs(old['inputs'], frozen)
            ? old['version_id']
            : reportUuid(),
        'inputs': frozen,
        'attempted': true,
        'validation_rejected': false,
      };
      // Durable before the POST; service/screen recreation reuses this ID/body.
      await _write(token, owner, source, data);
      return data;
    });
    late Map<String, dynamic> response;
    try {
      final request =
          http.Request('POST', ApiConfig.v4('/v4/science/section-capacity'))
            ..headers['Content-Type'] = 'application/json'
            ..body = jsonEncode({
              'source_version_id': source,
              'version_id': operation['version_id'],
              'inputs': operation['inputs']
            });
      response = await history.request(token, request);
    } on CorrectionException catch (error) {
      if (error.statusCode == 422) {
        await _serial(() async {
          final owner = await auth.owner(token);
          final latest = await _read(owner, source);
          if (latest?['version_id'] == operation['version_id'] &&
              latest?['saved'] != true) {
            await _write(token, owner, source,
                {...latest!, 'validation_rejected': true});
          }
        });
        throw const CorrectionException(
            'Сервер отклонил входные данные (422); версия не сохранена. Можно исправить входы и создать новый запрос.',
            statusCode: 422);
      }
      if ([404, 405].contains(error.statusCode)) {
        throw const CorrectionException(
            'Этот сервер пока не поддерживает сценарии сечения. Черновик сохранён; анализ и прежние отчёты доступны.');
      }
      rethrow;
    }
    final parsed = SectionScienceResult.parse(response['science_result']);
    if (response['saved'] != true ||
        response['persisted'] != true ||
        response['science_version'] != 1 ||
        response['record']?['version_id'] != operation['version_id'] ||
        parsed == null ||
        parsed.raw['result_id'] != operation['version_id'] ||
        parsed.raw['source_version_id'] != source ||
        !sameScienceInputs(parsed.raw['input_snapshot'], operation['inputs'])) {
      throw const CorrectionException(
          'Сервер не подтвердил расчёт и сохранение этой версии. Черновик сохранён; повторите тот же запрос.');
    }
    await _serial(() async {
      final owner = await auth.owner(token);
      final latest = await _read(owner, source);
      if (latest?['version_id'] != operation['version_id']) {
        throw const CorrectionException(
            'Черновик уже изменился. Откройте серверную историю.');
      }
      await _write(token, owner, source,
          {...latest!, 'saved': true, 'science_result': parsed.raw});
    });
    return response;
  }
}
