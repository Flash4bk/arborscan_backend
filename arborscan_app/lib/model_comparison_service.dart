import 'dart:convert';
import 'dart:typed_data';

import 'package:crypto/crypto.dart';
import 'package:http/http.dart' as http;
import 'package:image/image.dart' as im;

import 'api_config.dart';
import 'corrections_service.dart';

const _comparisonSchema = 'arborscan.model-comparison.v1';
const comparisonProcessingMaxSide = 2048;
final _weightHash = RegExp(r'^[a-f0-9]{64}$');

/// The original upload is preserved; this separate PNG is only for display.
class ComparisonPhoto {
  final Uint8List original, previewPng;
  final String uploadSha256;
  final int originalWidth, originalHeight, width, height;

  ComparisonPhoto._(this.original, this.previewPng, this.uploadSha256,
      this.originalWidth, this.originalHeight, this.width, this.height);

  factory ComparisonPhoto.decode(Uint8List bytes) {
    if (bytes.isEmpty || bytes.length > 15 * 1024 * 1024) {
      throw const CorrectionException(
          'Выберите фото размером не больше 15 МБ.');
    }
    final original = Uint8List.fromList(bytes);
    final decoder = im.findDecoderForData(original);
    final info = decoder?.startDecode(original);
    if (info == null || info.width * info.height > 25000000) {
      throw const CorrectionException('Фото повреждено или больше 25 МП.');
    }
    final frame = decoder!.decodeFrame(0);
    if (frame == null) {
      throw const CorrectionException('Не удалось прочитать фотографию.');
    }
    final oriented = im.bakeOrientation(frame);
    final scale = comparisonProcessingMaxSide /
        (oriented.width > oriented.height ? oriented.width : oriented.height);
    final width = scale < 1
        ? _roundEven(oriented.width * scale)
            .clamp(1, comparisonProcessingMaxSide)
        : oriented.width;
    final height = scale < 1
        ? _roundEven(oriented.height * scale)
            .clamp(1, comparisonProcessingMaxSide)
        : oriented.height;
    final preview = width == oriented.width && height == oriented.height
        ? oriented
        : im.copyResize(oriented, width: width, height: height);
    return ComparisonPhoto._(
        original,
        Uint8List.fromList(im.encodePng(preview)),
        sha256.convert(original).toString(),
        oriented.width,
        oriented.height,
        width,
        height);
  }
}

int _roundEven(double value) {
  final lower = value.floor();
  if (value - lower == .5) return lower.isEven ? lower : lower + 1;
  return value.round();
}

class ComparisonModel {
  final int version;
  final String weightsSha256;
  const ComparisonModel(this.version, this.weightsSha256);
}

class ComparisonModels {
  final List<ComparisonModel> available;
  final int? defaultVersion;
  const ComparisonModels(this.available, this.defaultVersion);
}

class ComparedModel {
  final ComparisonModel model;
  final bool detected;
  final double? confidence;
  final double meanSeconds, medianSeconds;
  final Uint8List overlayPng;
  final List<String> warnings;
  const ComparedModel(this.model, this.detected, this.confidence,
      this.meanSeconds, this.medianSeconds, this.overlayPng, this.warnings);
}

class ModelComparisonResult {
  final List<ComparedModel> models;
  final int imageSize;
  final double confidenceThreshold;
  const ModelComparisonResult(
      this.models, this.imageSize, this.confidenceThreshold);
}

/// A page-scoped, cancellable diagnostic client. It never activates or saves.
class ModelComparisonService {
  final CorrectionsService auth;
  final http.Client Function() clientFactory;
  final Duration timeout;
  final Set<http.Client> _clients = {};
  int _generation = 0;
  ModelComparisonService(
      {CorrectionsService? auth,
      http.Client Function()? clientFactory,
      this.timeout = const Duration(seconds: 150)})
      : auth = auth ?? CorrectionsService(),
        clientFactory = clientFactory ?? http.Client.new;

  void cancel() {
    _generation++;
    for (final client in _clients.toList()) {
      client.close();
    }
    _clients.clear();
  }

  Future<Map<String, dynamic>> _request(
      String token, http.BaseRequest request) async {
    final session = CorrectionsService.authChanges.value;
    final generation = _generation;
    await auth.checkSession(token);
    if (session != CorrectionsService.authChanges.value ||
        generation != _generation) {
      throw const CorrectionException(
          'Сравнение отменено или аккаунт изменился.');
    }
    final client = clientFactory();
    _clients.add(client);
    request.headers['Authorization'] = 'Bearer $token';
    try {
      final body = await (() async {
        final response = await client.send(request);
        if (response.statusCode < 200 || response.statusCode >= 300) {
          final message = switch (response.statusCode) {
            401 => 'Сессия истекла. Войдите снова.',
            403 => 'Для сравнения моделей нужны права администратора.',
            404 ||
            405 ||
            501 =>
              'Сравнение моделей пока недоступно на сервере. Обычный анализ и история остаются доступны.',
            409 =>
              'Веса моделей изменились. Обновите список и повторите сравнение.',
            413 => 'Фото слишком велико. Выберите файл до 15 МБ и 25 МП.',
            422 => 'Не удалось обработать выбранные фото или версии моделей.',
            429 => 'Сравнение уже выполняется на сервере. Повторите позже.',
            503 ||
            504 =>
              'Сервис сравнения сейчас недоступен. Фото остаётся на экране.',
            _ => 'Сервер не подтвердил сравнение (${response.statusCode}).',
          };
          throw CorrectionException(message, statusCode: response.statusCode);
        }
        final bytes = BytesBuilder(copy: false);
        var size = 0;
        await for (final chunk in response.stream) {
          size += chunk.length;
          if (size > 20 * 1024 * 1024) {
            throw const FormatException('comparison response too large');
          }
          bytes.add(chunk);
        }
        return bytes.takeBytes();
      })()
          .timeout(timeout);
      await auth.checkSession(token);
      if (session != CorrectionsService.authChanges.value ||
          generation != _generation) {
        throw const CorrectionException(
            'Сравнение отменено или аккаунт изменился.');
      }
      final decoded = jsonDecode(utf8.decode(body));
      if (decoded is! Map<String, dynamic> ||
          decoded['schema'] != _comparisonSchema) {
        throw const FormatException('unsupported comparison schema');
      }
      return decoded;
    } on CorrectionException {
      rethrow;
    } on FormatException {
      throw const CorrectionException(
          'Сервер вернул некорректный результат сравнения. Повторите запрос.');
    } catch (_) {
      throw const CorrectionException(
          'Нет ответа сервиса сравнения. Фото остаётся на экране; повторите запрос.');
    } finally {
      _clients.remove(client);
      client.close();
    }
  }

  Future<ComparisonModels> models(String token) async {
    final data = await _request(token,
        http.Request('GET', ApiConfig.v4('/v4/model-comparison/models')));
    try {
      if (data['supported'] != true || data['available'] is! List) {
        throw const FormatException();
      }
      final models = <ComparisonModel>[];
      for (final row in data['available']) {
        if (row is! Map ||
            row['version'] is! int ||
            row['version'] < 0 ||
            row['source'] != 'local_saved_weights' ||
            row['weights_sha256'] is! String ||
            !_weightHash.hasMatch(row['weights_sha256']) ||
            models.any((m) => m.version == row['version'])) {
          throw const FormatException();
        }
        models.add(ComparisonModel(row['version'], row['weights_sha256']));
      }
      models.sort((a, b) => a.version.compareTo(b.version));
      final defaultVersion = data['default_version'];
      if (defaultVersion != null && defaultVersion is! int) {
        throw const FormatException();
      }
      return ComparisonModels(List.unmodifiable(models), defaultVersion);
    } catch (_) {
      throw const CorrectionException(
          'Сервер не вернул проверяемый список версий и хешей весов.');
    }
  }

  Future<ModelComparisonResult> compare(String token, ComparisonPhoto photo,
      ComparisonModel baseline, ComparisonModel candidate) async {
    if (baseline.version == candidate.version) {
      throw const CorrectionException('Выберите две разные версии моделей.');
    }
    final original = Uint8List.fromList(photo.original);
    if (sha256.convert(original).toString() != photo.uploadSha256) {
      throw const CorrectionException(
          'Исходное фото изменилось. Выберите его снова.');
    }
    final request =
        http.MultipartRequest('POST', ApiConfig.v4('/v4/model-comparison'))
          ..fields.addAll({
            'baseline_version': '${baseline.version}',
            'candidate_version': '${candidate.version}',
            'baseline_sha256': baseline.weightsSha256,
            'candidate_sha256': candidate.weightsSha256,
          })
          ..files.add(http.MultipartFile.fromBytes('file', original,
              filename: 'original-photo.jpg'));
    final data = await _request(token, request);
    try {
      final image = data['image'];
      final settings = data['settings'];
      final quality = data['quality_metrics'];
      if (data['persisted'] != false ||
          image is! Map ||
          image['upload_sha256'] != photo.uploadSha256 ||
          image['original_width'] != photo.originalWidth ||
          image['original_height'] != photo.originalHeight ||
          image['width'] != photo.width ||
          image['height'] != photo.height ||
          image['preview_transform'] != 'resize_only' ||
          !_near(
              image['processing_scale_x'], photo.width / photo.originalWidth) ||
          !_near(image['processing_scale_y'],
              photo.height / photo.originalHeight) ||
          settings is! Map ||
          settings['device'] != 'cpu' ||
          settings['threads'] != 1 ||
          settings['warmup_runs'] != 1 ||
          settings['timing_repeats'] != 3 ||
          settings['processing_max_side'] != comparisonProcessingMaxSide ||
          settings['max_detections'] != 16 ||
          settings['imgsz'] is! int ||
          settings['imgsz'] < 320 ||
          settings['imgsz'] > 1536 ||
          !_finite(settings['confidence']) ||
          settings['confidence'] < 0 ||
          settings['confidence'] > 1 ||
          quality is! Map ||
          quality['reason'] != 'verified_reference_masks_required' ||
          [
            'precision',
            'recall',
            'map50',
            'map50_95',
            'mean_iou'
          ].any((key) => !quality.containsKey(key) || quality[key] != null) ||
          data['models'] is! List ||
          data['models'].length != 2) {
        throw const FormatException();
      }
      final results = <ComparedModel>[];
      final selected = [baseline, candidate];
      for (var i = 0; i < 2; i++) {
        final row = data['models'][i];
        final timing = row is Map ? row['inference_timing'] : null;
        if (row is! Map ||
            row['version'] != selected[i].version ||
            row['weights_sha256'] != selected[i].weightsSha256 ||
            row['detected'] is! bool ||
            (row['detected'] == true && row['tree_class_id'] is! int) ||
            (row['confidence'] != null &&
                (!_finite(row['confidence']) ||
                    row['confidence'] < 0 ||
                    row['confidence'] > 1)) ||
            timing is! Map ||
            !_nonnegative(timing['mean_seconds']) ||
            !_nonnegative(timing['median_seconds']) ||
            timing['samples_seconds'] is! List ||
            timing['samples_seconds'].length != 3 ||
            (timing['samples_seconds'] as List).any((v) => !_nonnegative(v)) ||
            row['warnings'] is! List ||
            (row['warnings'] as List).any((v) => v is! String)) {
          throw const FormatException();
        }
        final maskBytes = base64Decode(row['mask_png_base64'] as String);
        if (maskBytes.length < 33 || maskBytes[24] != 8 || maskBytes[25] != 0) {
          throw const FormatException();
        }
        final decoder = im.PngDecoder();
        final info = decoder.startDecode(maskBytes);
        if (info?.width != photo.width || info?.height != photo.height) {
          throw const FormatException();
        }
        final mask = decoder.decodeFrame(0);
        if (mask == null || mask.numChannels != 1) {
          throw const FormatException();
        }
        final overlay =
            im.Image(width: photo.width, height: photo.height, numChannels: 4);
        var pixels = 0;
        for (final p in mask) {
          if (p.r != 0 && p.r != 255) throw const FormatException();
          if (p.r == 255) {
            pixels++;
            overlay.setPixelRgba(p.x, p.y, 36, 150, 80, 112);
          }
        }
        if ((row['detected'] == true) != (pixels > 0)) {
          throw const FormatException();
        }
        results.add(ComparedModel(
            selected[i],
            row['detected'],
            (row['confidence'] as num?)?.toDouble(),
            (timing['mean_seconds'] as num).toDouble(),
            (timing['median_seconds'] as num).toDouble(),
            Uint8List.fromList(im.encodePng(overlay)),
            List<String>.unmodifiable(row['warnings'])));
      }
      return ModelComparisonResult(List.unmodifiable(results),
          settings['imgsz'], (settings['confidence'] as num).toDouble());
    } catch (_) {
      throw const CorrectionException(
          'Результат не соответствует выбранным фото и версиям моделей. Повторите сравнение.');
    }
  }
}

bool _finite(dynamic v) => v is num && v.isFinite;
bool _nonnegative(dynamic v) => _finite(v) && v >= 0;
bool _near(dynamic v, double expected) =>
    _finite(v) && (v - expected).abs() < 1e-8;
