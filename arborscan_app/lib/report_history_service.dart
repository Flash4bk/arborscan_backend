import 'dart:convert';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:crypto/crypto.dart';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';
import 'api_config.dart';
import 'contour_drafts.dart';
import 'corrections_service.dart';

String reportUuid() {
  final r = Random.secure();
  final b = List<int>.generate(16, (_) => r.nextInt(256));
  b[6] = (b[6] & 15) | 64;
  b[8] = (b[8] & 63) | 128;
  final h = b.map((v) => v.toRadixString(16).padLeft(2, '0')).join();
  return '${h.substring(0, 8)}-${h.substring(8, 12)}-${h.substring(12, 16)}-${h.substring(16, 20)}-${h.substring(20)}';
}

ContourDrafts reportStore() => ContourDrafts(
    directory: () async => Directory(
        '${(await getApplicationSupportDirectory()).path}/report-history-v1'));

class ReportHistoryService {
  final CorrectionsService auth;
  final ContourDrafts journal;
  final http.Client Function() clientFactory;
  ReportHistoryService(
      {CorrectionsService? auth,
      ContourDrafts? journal,
      http.Client Function()? clientFactory})
      : auth = auth ?? CorrectionsService(),
        journal = journal ?? reportStore(),
        clientFactory = clientFactory ?? http.Client.new;

  Future<Map<String, dynamic>> request(
      String token, http.BaseRequest request) async {
    await auth.checkSession(token);
    final generation = CorrectionsService.authChanges.value;
    final client = clientFactory();
    try {
      request.headers['Authorization'] = 'Bearer $token';
      final r = await (() async =>
              http.Response.fromStream(await client.send(request)))()
          .timeout(const Duration(seconds: 120));
      await auth.checkSession(token);
      if (generation != CorrectionsService.authChanges.value) {
        throw const CorrectionException(
            'Аккаунт изменился. Откройте историю заново.');
      }
      if (r.statusCode == 409) {
        throw const CorrectionException(
            'Версия уже изменилась. Черновик сохранён: откройте последнюю версию из истории.',
            statusCode: 409);
      }
      if ([401, 403].contains(r.statusCode)) {
        throw const CorrectionException('Войдите в свой аккаунт заново.',
            statusCode: 401);
      }
      if ([404, 405, 503].contains(r.statusCode)) {
        throw CorrectionException(
            'Серверная история или запись недоступна. Локальные данные сохранены; повторите позже.',
            statusCode: r.statusCode);
      }
      if (r.statusCode < 200 || r.statusCode >= 300) {
        throw CorrectionException(
            'Отчёт не сохранён (${r.statusCode}). Проверьте данные и повторите.');
      }
      return Map<String, dynamic>.from(jsonDecode(utf8.decode(r.bodyBytes)));
    } on CorrectionException {
      rethrow;
    } catch (_) {
      throw const CorrectionException(
          'Нет подтверждения сервера. Данные остаются на устройстве; повторите сохранение.');
    } finally {
      client.close();
    }
  }

  Future<Map<String, dynamic>> list(String token, int offset,
          {String? analysisId}) =>
      request(
          token,
          http.Request(
              'GET',
              ApiConfig.v4('/v4/reports').replace(queryParameters: {
                'offset': '$offset',
                if (analysisId != null) 'analysis_id': analysisId
              })));
  Future<Map<String, dynamic>> record(String token, String id) => request(
      token,
      http.Request(
          'GET', ApiConfig.v4('/v4/reports/${Uri.encodeComponent(id)}')));

  Future<Map<String, dynamic>> stage(
      {required String token,
      required String localId,
      required String analysisId,
      required Map<String, dynamic> snapshot,
      required Uint8List image,
      String? parentId}) async {
    final owner = await auth.owner(token);
    final old = await journal.load(owner, localId);
    await auth.checkSession(token);
    final hash = sha256.convert(utf8.encode(jsonEncode(snapshot))).toString();
    if (old != null && old['snapshot_hash'] == hash) return old;
    final data = <String, dynamic>{
      'analysis_id': analysisId,
      'version_id': reportUuid(),
      'parent_id': old?['saved'] == true
          ? old!['version_id']
          : (old?['parent_id'] ?? parentId),
      'snapshot': snapshot,
      'snapshot_hash': hash,
      'saved': false,
      'created_at': DateTime.now().toUtc().toIso8601String()
    };
    await journal.save(owner, localId, data, image);
    await auth.checkSession(token);
    return {...data, 'image': image, 'draft_id': localId};
  }

  Future<Map<String, dynamic>> upload(String token, String localId) async {
    final owner = await auth.owner(token);
    final d = await journal.load(owner, localId);
    await auth.checkSession(token);
    if (d == null) {
      throw const CorrectionException('Локальный отчёт недоступен.');
    }
    final request = http.MultipartRequest('POST', ApiConfig.v4('/v4/reports'))
      ..fields['analysis_id'] = d['analysis_id']
      ..fields['version_id'] = d['version_id']
      ..fields['snapshot'] = jsonEncode(d['snapshot'])
      ..files.add(http.MultipartFile.fromBytes('image', d['image'],
          filename: 'original.jpg'));
    if (d['parent_id'] != null) request.fields['parent_id'] = d['parent_id'];
    final response = await this.request(token, request);
    if (response['saved'] != true ||
        response['persisted'] != true ||
        response['record']?['version_id'] != d['version_id']) {
      throw const CorrectionException(
          'Сервер не подтвердил сохранение. Повторите запрос.');
    }
    await auth.checkSession(token);
    final latest = await journal.load(owner, localId);
    if (latest?['version_id'] != d['version_id']) {
      throw const CorrectionException(
          'Предыдущая версия отправлена, но локальный черновик уже изменился. Откройте серверную историю.');
    }
    final photo = d.remove('image');
    d.remove('draft_id');
    await journal.save(owner, localId, {...d, 'saved': true}, photo);
    await auth.checkSession(token);
    return response;
  }
}
