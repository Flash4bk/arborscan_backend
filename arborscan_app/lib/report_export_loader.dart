import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:crypto/crypto.dart';
import 'package:path_provider/path_provider.dart';
import 'corrections_service.dart';
import 'report_export_data.dart';

/// Fetches only the pinned immutable correction, never the latest revision or decision.
Future<ReportExportData> loadReportExport(
    {required Map<String, dynamic> snapshot,
    required Map<String, dynamic> record,
    required Uint8List? photo,
    required bool local,
    required String token,
    CorrectionsService? service,
    Future<Directory> Function()? cacheDirectory}) async {
  final frozen = ReportExportData(
      snapshot: snapshot, record: record, photo: photo, local: local);
  final id = frozen.snapshot['correction_id'];
  if (id == null) return frozen;
  final auth = service ?? CorrectionsService();
  final epoch = CorrectionsService.authChanges.value;
  await auth.checkSession(token);
  final owner = await auth.owner(token);
  final root = await (cacheDirectory ?? getApplicationSupportDirectory)();
  final key = sha256.convert(utf8.encode('$owner/$id')).toString();
  final file = File('${root.path}/report-export-assets/$key.json');
  Map<String, dynamic> asset;
  if (await file.exists()) {
    asset = exportMap(jsonDecode(await file.readAsString()));
  } else {
    final response = await auth.record(token, id.toString());
    if (response['correction_id'] != id) {
      throw const FormatException('Сервер вернул другую ревизию контура.');
    }
    asset = {
      'correction_id': id,
      'original_image_base64': response['original_image_base64'],
      'mask_png_base64': response['mask_png_base64']
    };
  }
  await auth.checkSession(token);
  if (epoch != CorrectionsService.authChanges.value) {
    throw const CorrectionException('Аккаунт изменился.');
  }
  final original = exportDecode(asset['original_image_base64']);
  final mask = exportDecode(asset['mask_png_base64']);
  if (frozen.photo == null ||
      original == null ||
      mask == null ||
      sha256.convert(original) != sha256.convert(frozen.photo!)) {
    throw const FormatException(
        'Контур не соответствует оригиналу выбранной версии. Повторите загрузку.');
  }
  if (!await file.exists()) {
    await file.parent.create(recursive: true);
    final temp = File('${file.path}.tmp');
    await temp.writeAsString(jsonEncode(asset), flush: true);
    await temp.rename(file.path);
  }
  return ReportExportData(
      snapshot: frozen.snapshot,
      record: frozen.record,
      photo: frozen.photo,
      local: local,
      mask: mask);
}
