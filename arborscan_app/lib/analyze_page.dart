import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'api_config.dart';
import 'app_theme.dart';
import 'ar_measure_channel.dart';
import 'unified_analysis_models.dart';
import 'unified_analysis_report_page.dart';
import 'reference_measurement_page.dart';
import 'survey_environment.dart';
import 'survey_environment_ui.dart';
import 'corrections_service.dart';
import 'image_selection_service.dart';
import 'package:crypto/crypto.dart';

class ArborScanPage extends StatefulWidget {
  const ArborScanPage({super.key, this.onOpenProfile});

  final VoidCallback? onOpenProfile;

  @override
  State<ArborScanPage> createState() => _ArborScanPageState();
}

class _ArborScanPageState extends State<ArborScanPage> {
  static const String _historyKey = 'arborscan_history';

  final _imageSelection = ImageSelectionService();
  final _environment = SurveyEnvironmentController();
  Uint8List? _original;
  @override
  void initState() {
    super.initState();
    CorrectionsService.authChanges.addListener(_accountChanged);
  }

  void _accountChanged() {
    if (mounted) {
      setState(() {
        _imageFile = null;
        _original = null;
        _arResult = null;
        _arPhotoHash = null;
        _lastResult = null;
      });
    }
  }

  @override
  void dispose() {
    CorrectionsService.authChanges.removeListener(_accountChanged);
    _environment.dispose();
    super.dispose();
  }

  File? _imageFile;
  ImageSource? _imageSource;
  ArMeasureResult? _arResult;
  String? _arPhotoHash;
  UnifiedAnalysisResult? _lastResult;

  bool _loading = false;
  bool _openingAr = false;
  bool _pickingImage = false;
  String? _error;

  String get _apiUrl => ApiConfig.v4('/v4/analyze-tree').toString();

  Future<void> _pickImage(ImageSource source) async {
    if (_loading || _openingAr || _pickingImage) return;
    final authGeneration = CorrectionsService.authChanges.value;
    setState(() => _pickingImage = true);
    try {
      final picked = await _imageSelection.pick(source);
      if (picked == null ||
          !mounted ||
          authGeneration != CorrectionsService.authChanges.value) {
        return;
      }

      // The original file remains intact for EXIF and photo-hash provenance.
      final selectedImage = File(picked.path);
      final original = await selectedImage.readAsBytes();
      if (!mounted || authGeneration != CorrectionsService.authChanges.value) {
        return;
      }
      String? boundHash;
      var retainAr = false;
      if (_arResult != null) {
        final sameTree = await _confirmSameTree(photoSelectedAfterAr: true);
        if (!mounted ||
            sameTree == null ||
            authGeneration != CorrectionsService.authChanges.value) {
          return;
        }
        retainAr = sameTree;
        if (retainAr) {
          // Bind the bytes already read for this photo. Keep this synchronous:
          // an account change must not restore a canceled selection afterward.
          boundHash = sha256.convert(original).toString();
        }
      }

      setState(() {
        _imageFile = selectedImage;
        _original = original;
        _imageSource = source;
        if (!retainAr) _arResult = null;
        _arPhotoHash = boundHash;
        _lastResult = null;
        _error = null;
      });
      _environment.setPoint(surveyPointFromExif(original));
    } catch (e) {
      if (!mounted || authGeneration != CorrectionsService.authChanges.value) {
        return;
      }
      setState(() => _error = 'Не удалось выбрать изображение: $e');
    } finally {
      if (mounted) setState(() => _pickingImage = false);
    }
  }

  Future<bool?> _confirmSameTree({bool photoSelectedAfterAr = false}) =>
      showDialog<bool>(
        context: context,
        builder: (c) => AlertDialog(
          title: Text(photoSelectedAfterAr
              ? 'Связать фото с измерением AR'
              : 'Связать AR с выбранным фото'),
          content: Text(photoSelectedAfterAr
              ? 'На выбранном фото то же дерево, которое вы измерили? '
                  'Приложение не проверяет это автоматически. Размеры AR '
                  'сохраняются отдельно и не задают масштаб фотографии.'
              : 'Измеряйте то же дерево, которое выбрано на фото. '
                  'Приложение не проверяет это автоматически. Размеры AR '
                  'сохраняются отдельно и не задают масштаб фотографии.'),
          actions: [
            TextButton(
                onPressed: () => Navigator.pop(c), child: const Text('Отмена')),
            if (photoSelectedAfterAr)
              TextButton(
                  onPressed: () => Navigator.pop(c, false),
                  child: const Text('Другое дерево — без AR')),
            FilledButton(
                onPressed: () => Navigator.pop(c, true),
                child: const Text('Это то же дерево')),
          ],
        ),
      );

  Future<void> _openAr() async {
    if (_openingAr || _loading || _pickingImage) return;
    final authGeneration = CorrectionsService.authChanges.value;
    setState(() {
      _openingAr = true;
      _error = null;
    });
    try {
      final selectedImage = _imageFile;
      String? boundHash;
      if (selectedImage != null) {
        boundHash =
            sha256.convert(await selectedImage.readAsBytes()).toString();
        if (!mounted) return;
        final sameTree = await _confirmSameTree();
        if (sameTree != true || !mounted) return;
      }

      final result = await ArMeasureChannel.openArMeasure();
      if (!mounted ||
          result == null ||
          authGeneration != CorrectionsService.authChanges.value) {
        return;
      }
      if (_imageFile != selectedImage) {
        throw const FormatException(
            'Фото изменилось. Повторите AR для выбранного дерева.');
      }

      setState(() {
        _arResult = result;
        _arPhotoHash = boundHash;
        _lastResult = null;
      });

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            'AR готов: H ${result.heightMeters!.toStringAsFixed(2)} м · '
            'Диаметр ствола ${result.trunkDiameterMeters!.toStringAsFixed(3)} м · '
            '${selectedImage == null ? 'Теперь добавьте фото этого дерева.' : result.statusLabelRu}',
          ),
        ),
      );
    } on PlatformException catch (e) {
      if (!mounted) return;
      setState(
          () => _error = e.message ?? 'AR сейчас недоступен на устройстве.');
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = 'AR-измерение не завершено: $e');
    } finally {
      if (mounted) setState(() => _openingAr = false);
    }
  }

  Future<void> _analyze() async {
    final imageFile = _imageFile;
    if (imageFile == null || _loading) return;
    final environment = _environment.snapshot;
    final authGeneration = CorrectionsService.authChanges.value;

    setState(() {
      _loading = true;
      _error = null;
      _lastResult = null;
    });

    final client = http.Client();
    try {
      final request = http.MultipartRequest('POST', Uri.parse(_apiUrl));
      final prefs = await SharedPreferences.getInstance();
      final sessionToken = prefs.getString('arborscan_auth_token');
      if (sessionToken != null && sessionToken.isNotEmpty) {
        request.headers['Authorization'] = 'Bearer $sessionToken';
      }
      request.fields['include_images'] = 'true';

      final ar = _arResult;
      if (ar != null) {
        if (_arPhotoHash !=
            sha256.convert(await imageFile.readAsBytes()).toString()) {
          throw const FormatException(
              'AR относится к другому фото. Повторите измерение.');
        }
        request.fields.addAll(ar.toV4FormFields());
        request.fields['ar_photo_sha256'] = _arPhotoHash!;
        request.fields['ar_same_tree_confirmed'] = 'true';
      }

      request.files.add(
        await http.MultipartFile.fromPath('file', imageFile.path),
      );

      final response = await (() async =>
              http.Response.fromStream(await client.send(request)))()
          .timeout(const Duration(seconds: 180));

      if (response.statusCode < 200 || response.statusCode >= 300) {
        throw Exception(_extractServerMessage(response));
      }

      final decoded = jsonDecode(utf8.decode(response.bodyBytes));
      if (decoded is! Map) {
        throw const FormatException('Сервер вернул некорректный JSON');
      }
      if (ar != null && decoded['measurement_method_version'] != 1) {
        final measures = decoded['measurements'];
        if (measures is Map) {
          measures['crown_width'] = {
            'value_m': null,
            'value_px': null,
            'source': 'unavailable',
            'confidence': 0.0,
            'notes': ['old_api_ar_photo_scale_not_validated']
          };
        }
      }

      decoded['captured_at'] ??= DateTime.now().toUtc().toIso8601String();
      decoded['environment_snapshot'] =
          environment.isEmpty ? null : environment;
      decoded['ar_provenance'] = ar == null
          ? null
          : {
              'photo_sha256': _arPhotoHash,
              'association': 'user_confirmed_same_tree',
              'measurement': ar.raw
            };
      if (authGeneration != CorrectionsService.authChanges.value ||
          prefs.getString('arborscan_auth_token') != sessionToken) {
        throw const FormatException('Аккаунт изменился. Повторите анализ.');
      }
      final result = UnifiedAnalysisResult.fromJson(
        Map<String, dynamic>.from(decoded),
      );

      if (!mounted) return;
      setState(() => _lastResult = result);
      await _saveHistory(result, sessionToken);

      Uint8List? fallbackBytes;
      try {
        fallbackBytes = await imageFile.readAsBytes();
      } catch (_) {}

      if (!mounted) return;
      if (prefs.getString('arborscan_auth_token') != sessionToken) {
        return;
      }
      await Navigator.of(context).push(
        MaterialPageRoute(
          builder: (_) => UnifiedAnalysisReportPage(
            result: result,
            fallbackImageBytes: fallbackBytes,
          ),
        ),
      );
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = _humanizeNetworkError(e);
      });
    } finally {
      client.close();
      if (mounted) setState(() => _loading = false);
    }
  }

  Future<void> _saveHistory(
      UnifiedAnalysisResult result, String? sessionToken) async {
    try {
      final prefs = await SharedPreferences.getInstance();
      if (sessionToken == null ||
          prefs.getString('arborscan_auth_token') != sessionToken) {
        return;
      }
      final existing = prefs.getStringList(_historyKey) ?? <String>[];

      final historyRow = <String, dynamic>{
        'owner_id': prefs.getString('arborscan_user_id'),
        'species': result.speciesName,
        'height': result.height.valueM,
        'crown': result.crownWidth.valueM,
        'trunk': result.trunkDiameter.valueM,
        'scale': result.pxToM,
        'riskIndex': null,
        'riskCategory': null,
        'lat': SurveyPoint.fromJson(
                surveyMap(result.raw['environment_snapshot'])['gps'])
            ?.lat,
        'lon': SurveyPoint.fromJson(
                surveyMap(result.raw['environment_snapshot'])['gps'])
            ?.lon,
        'environment_snapshot': result.raw['environment_snapshot'],
        'address': null,
        'imageBase64': '',
        'timestamp': DateTime.now().toIso8601String(),
        'analysisId': result.analysisId,
        'ar_provenance': _arResult == null
            ? null
            : {
                'photo_sha256': _arPhotoHash,
                'association': 'user_confirmed_same_tree',
                'measurement': _arResult!.raw
              },
      };

      final newRow = jsonEncode(historyRow);
      final output = <String>[newRow];

      for (final row in existing) {
        if (output.length >= 30) break;
        try {
          final raw = jsonDecode(row);
          if (raw is Map && raw['analysisId'] == result.analysisId) continue;
        } catch (_) {}
        output.add(row);
      }

      await prefs.setStringList(_historyKey, output);
    } catch (_) {
      // History is auxiliary. A valid analysis should not fail because local
      // persistence is unavailable.
    }
  }

  String _extractServerMessage(http.Response response) {
    var message = 'Ошибка сервера ${response.statusCode}';
    try {
      final raw = jsonDecode(utf8.decode(response.bodyBytes));
      if (raw is Map) {
        final detail = raw['detail'] ?? raw['message'] ?? raw['error'];
        if (detail != null) message = detail.toString();
      }
    } catch (_) {}
    return message;
  }

  String _humanizeNetworkError(Object error) {
    final text = error.toString();
    if (text.contains('Connection refused') ||
        text.contains('Failed host lookup') ||
        text.contains('SocketException')) {
      return 'Не удалось подключиться к Unified Analysis v4. '
          '${ApiConfig.connectionHint(ApiConfig.v4BaseUrl)}';
    }
    if (text.contains('TimeoutException')) {
      return 'Анализ превысил допустимое время ожидания.';
    }
    return text.replaceFirst('Exception: ', '');
  }

  void _reset() {
    _environment.setPoint(null);
    setState(() {
      _imageFile = null;
      _original = null;
      _imageSource = null;
      _arResult = null;
      _arPhotoHash = null;
      _lastResult = null;
      _error = null;
    });
  }

  void _showMeasurementHelp() {
    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      builder: (context) => SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.fromLTRB(24, 8, 24, 24),
          child:
              Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Text('Способы измерения',
                style: Theme.of(context).textTheme.titleLarge),
            const SizedBox(height: 16),
            const Text('AR', style: TextStyle(fontWeight: FontWeight.w700)),
            const Text('Оценивает высоту в вертикальной плоскости и диаметр '
                'приблизительно цилиндрического ствола. Можно начать до выбора '
                'фото. Затем подтвердите, что на фото то же дерево. '
                'AR не задаёт масштаб фотографии и отдельно не измеряет крону. '
                'Наклон дерева ограничивает метод; статус описывает условия, '
                'а не подтверждённую точность.'),
            const SizedBox(height: 16),
            const Text('По эталону',
                style: TextStyle(fontWeight: FontWeight.w700)),
            const Text('Независимое измерение по отрезку известной длины на '
                'фотографии. Эталон и дерево должны быть примерно на одной '
                'глубине. Результаты этого режима сохраняются отдельно.'),
            const SizedBox(height: 16),
            const Text('Анализ фото',
                style: TextStyle(fontWeight: FontWeight.w700)),
            const Text('Выделяет дерево и определяет породу. Без масштаба '
                'физические размеры не подставляются. Источники каждого '
                'размера остаются в отчёте.'),
            const SizedBox(height: 20),
            SizedBox(
                width: double.infinity,
                child: FilledButton(
                  onPressed: () => Navigator.pop(context),
                  child: const Text('Понятно'),
                )),
          ]),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final busy = _loading || _openingAr || _pickingImage;
    return Scaffold(
      appBar: AppBar(
        title: const Text('ArborScan'),
        actions: [
          if (_imageFile != null || _arResult != null)
            IconButton(
              tooltip: 'Начать заново',
              onPressed: busy ? null : _reset,
              icon: const Icon(Icons.refresh),
            ),
          if (widget.onOpenProfile != null)
            Padding(
              padding: const EdgeInsets.only(right: 12),
              child: IconButton.filledTonal(
                tooltip: 'Профиль',
                style: IconButton.styleFrom(backgroundColor: AppTheme.surface3),
                onPressed: widget.onOpenProfile,
                icon: const Icon(Icons.person_outline),
              ),
            ),
        ],
      ),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.fromLTRB(16, 8, 16, 24),
          children: [
            Text('Новое исследование',
                style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                    fontWeight: FontWeight.w800,
                    height: 1.1,
                    fontSize: MediaQuery.textScalerOf(context).scale(1) > 1.3
                        ? 22
                        : null)),
            const SizedBox(height: 18),
            _PhotoCard(
              file: _imageFile,
              source: _imageSource,
              loading: busy,
              onCamera: () => _pickImage(ImageSource.camera),
              onGallery: () => _pickImage(ImageSource.gallery),
            ),
            const SizedBox(height: 12),
            Row(children: [
              Expanded(
                  child: Text('Измерения',
                      style: Theme.of(context).textTheme.titleLarge)),
              IconButton(
                tooltip: 'Как выбрать способ измерения',
                onPressed: _showMeasurementHelp,
                icon: const Icon(Icons.info_outline),
              ),
            ]),
            LayoutBuilder(builder: (context, constraints) {
              final actions = [
                _MeasurementAction(
                    key: const ValueKey('open-ar'),
                    title: _openingAr ? 'Открываем AR…' : 'AR',
                    icon: Icons.view_in_ar_outlined,
                    busy: _openingAr,
                    onPressed: busy ? null : _openAr),
                _MeasurementAction(
                    title: 'По эталону',
                    icon: Icons.straighten,
                    onPressed: busy
                        ? null
                        : () => Navigator.of(context).push(MaterialPageRoute(
                            builder: (_) => const ReferenceMeasurementPage()))),
              ];
              if (MediaQuery.textScalerOf(context).scale(1) > 1.3 ||
                  constraints.maxWidth < 300) {
                return Column(children: [
                  actions[0],
                  const SizedBox(height: 8),
                  actions[1]
                ]);
              }
              return Row(children: [
                Expanded(child: actions[0]),
                const SizedBox(width: 10),
                Expanded(child: actions[1])
              ]);
            }),
            if (_arResult != null) ...[
              const SizedBox(height: 12),
              _ArSummary(result: _arResult!, bound: _arPhotoHash != null),
            ],
            const SizedBox(height: 16),
            SizedBox(
              width: double.infinity,
              child: FilledButton.icon(
                key: const ValueKey('analyze-photo'),
                onPressed: _imageFile == null || busy ? null : _analyze,
                icon: _loading
                    ? const SizedBox(
                        width: 20,
                        height: 20,
                        child: CircularProgressIndicator(
                            strokeWidth: 2, color: Colors.white))
                    : const Icon(Icons.arrow_forward),
                iconAlignment: IconAlignment.end,
                label: Text(_loading ? 'Анализируем…' : 'Анализировать'),
              ),
            ),
            // AS-09: the location/conditions panel belongs after the main action.
            if (_imageFile != null)
              SurveyEnvironmentEditor(
                  controller: _environment,
                  original: _original,
                  enabled: !busy),
            if (_imageFile == null)
              const Padding(
                padding: EdgeInsets.fromLTRB(4, 8, 4, 0),
                child: Text('Добавьте фото для анализа.',
                    style: TextStyle(color: AppTheme.muted)),
              ),
            if (_error != null) ...[
              const SizedBox(height: 14),
              _ErrorCard(message: _error!),
            ],
            if (_lastResult != null) ...[
              const SizedBox(height: 14),
              _LastResultCard(
                result: _lastResult!,
                onOpen: () async {
                  Uint8List? fallbackBytes;
                  try {
                    fallbackBytes = await _imageFile?.readAsBytes();
                  } catch (_) {}
                  if (!context.mounted) return;
                  await Navigator.of(context).push(MaterialPageRoute(
                    builder: (_) => UnifiedAnalysisReportPage(
                        result: _lastResult!,
                        fallbackImageBytes: fallbackBytes),
                  ));
                },
              ),
            ],
            Align(
              alignment: Alignment.centerRight,
              child: TextButton.icon(
                onPressed: () => showModalBottomSheet<void>(
                    context: context,
                    isScrollControlled: true,
                    builder: (_) => SafeArea(
                        child: SingleChildScrollView(
                            padding: const EdgeInsets.all(16),
                            child: _DebugEndpointCard(url: _apiUrl)))),
                icon: const Icon(Icons.settings_outlined, size: 18),
                label: const Text('Диагностика'),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _PhotoCard extends StatelessWidget {
  final File? file;
  final ImageSource? source;
  final bool loading;
  final VoidCallback onCamera;
  final VoidCallback onGallery;
  const _PhotoCard(
      {required this.file,
      required this.source,
      required this.loading,
      required this.onCamera,
      required this.onGallery});

  @override
  Widget build(BuildContext context) => LayoutBuilder(builder: (context, size) {
        final largeText = MediaQuery.textScalerOf(context).scale(1) > 1.3;
        final height = largeText ? 480.0 : size.maxWidth.clamp(280.0, 440.0);
        final controls = [
          _PhotoButton(
              icon: Icons.camera_alt_outlined,
              label: 'Камера',
              onPressed: loading ? null : onCamera),
          _PhotoButton(
              icon: Icons.photo_library_outlined,
              label: 'Галерея',
              onPressed: loading ? null : onGallery),
        ];
        return ClipRRect(
          borderRadius: BorderRadius.circular(22),
          child: SizedBox(
              height: height,
              width: double.infinity,
              child: Stack(
                fit: StackFit.expand,
                children: [
                  AnimatedSwitcher(
                    duration: MediaQuery.disableAnimationsOf(context)
                        ? Duration.zero
                        : const Duration(milliseconds: 180),
                    child: file == null
                        ? Container(
                            key: const ValueKey('empty-photo'),
                            color: AppTheme.surface2,
                            padding: EdgeInsets.fromLTRB(
                                28, 28, 28, largeText ? 160 : 80),
                            alignment: Alignment.center,
                            child: Column(
                                mainAxisSize: MainAxisSize.min,
                                children: [
                                  Image.asset('assets/icons/arborscan_icon.png',
                                      width: 64,
                                      height: 64,
                                      semanticLabel: 'ArborScan'),
                                  const SizedBox(height: 16),
                                  Text('Фото дерева',
                                      textAlign: TextAlign.center,
                                      style: Theme.of(context)
                                          .textTheme
                                          .titleLarge),
                                ]))
                        : Image.file(file!,
                            key: ValueKey(file!.path),
                            width: double.infinity,
                            height: double.infinity,
                            fit: BoxFit.cover,
                            semanticLabel: 'Выбранная фотография дерева',
                            errorBuilder: (_, __, ___) => const ColoredBox(
                                color: AppTheme.surface2,
                                child: Center(
                                    child: Icon(Icons.broken_image_outlined,
                                        size: 48)))),
                  ),
                  if (file != null)
                    const DecoratedBox(
                        decoration: BoxDecoration(
                            gradient: LinearGradient(
                      begin: Alignment.topCenter,
                      end: Alignment.bottomCenter,
                      colors: [
                        Color(0x35000000),
                        Colors.transparent,
                        Color(0x55000000)
                      ],
                      stops: [0, .5, 1],
                    ))),
                  if (file != null)
                    Positioned(
                        top: 14,
                        left: 14,
                        right: 14,
                        child: Align(
                          alignment: Alignment.centerLeft,
                          child: Semantics(
                            label: source == ImageSource.camera
                                ? 'Фото добавлено с камеры'
                                : 'Фото добавлено из галереи',
                            child: Container(
                              padding: const EdgeInsets.symmetric(
                                  horizontal: 12, vertical: 9),
                              decoration: BoxDecoration(
                                  color: AppTheme.primary.withValues(alpha: .9),
                                  borderRadius: BorderRadius.circular(24)),
                              child: const Row(
                                  mainAxisSize: MainAxisSize.min,
                                  children: [
                                    Icon(Icons.check_circle,
                                        color: AppTheme.surface3, size: 20),
                                    SizedBox(width: 8),
                                    Flexible(
                                        child: Text('Фото добавлено',
                                            style: TextStyle(
                                                color: Colors.white,
                                                fontWeight: FontWeight.w600))),
                                  ]),
                            ),
                          ),
                        )),
                  Positioned(
                      bottom: 14,
                      left: 14,
                      right: 14,
                      child: largeText
                          ? Column(children: [
                              controls[0],
                              const SizedBox(height: 8),
                              controls[1]
                            ])
                          : Row(children: [
                              Expanded(child: controls[0]),
                              const SizedBox(width: 10),
                              Expanded(child: controls[1])
                            ])),
                ],
              )),
        );
      });
}

class _PhotoButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback? onPressed;
  const _PhotoButton({required this.icon, required this.label, this.onPressed});
  @override
  Widget build(BuildContext context) => SizedBox(
        width: double.infinity,
        child: FilledButton.icon(
          onPressed: onPressed,
          style: FilledButton.styleFrom(
              backgroundColor: AppTheme.primary.withValues(alpha: .92),
              foregroundColor: Colors.white,
              side: const BorderSide(color: Color(0xFFCED5BA)),
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 13),
              shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(14))),
          icon: Icon(icon, size: 24),
          label: Text(label),
        ),
      );
}

class _MeasurementAction extends StatelessWidget {
  final String title;
  final IconData icon;
  final bool busy;
  final VoidCallback? onPressed;
  const _MeasurementAction(
      {super.key,
      required this.title,
      required this.icon,
      this.busy = false,
      this.onPressed});
  @override
  Widget build(BuildContext context) => OutlinedButton(
        onPressed: onPressed,
        style: OutlinedButton.styleFrom(
            minimumSize: const Size(double.infinity, 64),
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 16)),
        child: Row(children: [
          if (busy)
            const SizedBox(
                width: 25,
                height: 25,
                child: CircularProgressIndicator(strokeWidth: 2))
          else
            Icon(icon, size: 28),
          const SizedBox(width: 12),
          Expanded(
              child: Text(title,
                  style: const TextStyle(fontWeight: FontWeight.w700))),
          const SizedBox(width: 4),
          const Icon(Icons.chevron_right, size: 20),
        ]),
      );
}

class _ArSummary extends StatelessWidget {
  final ArMeasureResult result;
  final bool bound;
  const _ArSummary({required this.result, required this.bound});
  @override
  Widget build(BuildContext context) => Card(
        child: ExpansionTile(
          leading: Icon(bound ? Icons.check_circle_outline : Icons.link,
              color: AppTheme.primary2),
          title: Text('AR · ${result.heightMeters!.toStringAsFixed(2)} м'),
          subtitle: Text(bound
              ? 'Связано с этим фото'
              : 'Добавьте фото измеренного дерева'),
          childrenPadding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
          expandedCrossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
                'Диаметр: ${(result.trunkDiameterMeters! * 100).toStringAsFixed(1)} см '
                'на высоте ${result.trunkMeasurementHeightMeters!.toStringAsFixed(2)} м. '
                '${result.statusLabelRu}. Крона отдельно не измерена.'),
            if (result.dbhRepeatSpreadMeters != null)
              Text('Разброс повторов диаметра: '
                  '${(result.dbhRepeatSpreadMeters! * 1000).toStringAsFixed(0)} мм.'),
            const SizedBox(height: 8),
            const Text('Масштаб фото по AR не переносится. '
                'Статус условий не подтверждает точность измерения.'),
            if (result.warnings.isNotEmpty)
              ExpansionTile(
                  title: const Text('Диагностика AR'),
                  children: [SelectableText(result.warnings.join('\n'))]),
          ],
        ),
      );
}

class _ErrorCard extends StatelessWidget {
  final String message;

  const _ErrorCard({required this.message});

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Icon(Icons.error_outline, color: AppTheme.danger),
            const SizedBox(width: 10),
            Expanded(
              child: Text(
                message,
                style: Theme.of(context).textTheme.bodySmall?.copyWith(
                      color: AppTheme.danger,
                      height: 1.4,
                    ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _LastResultCard extends StatelessWidget {
  final UnifiedAnalysisResult result;
  final VoidCallback onOpen;

  const _LastResultCard({required this.result, required this.onOpen});

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Последний результат',
              style: Theme.of(context).textTheme.titleSmall?.copyWith(
                    fontWeight: FontWeight.w800,
                  ),
            ),
            const SizedBox(height: 6),
            Text(result.speciesName),
            const SizedBox(height: 10),
            SizedBox(
              width: double.infinity,
              child: OutlinedButton.icon(
                onPressed: onOpen,
                icon: const Icon(Icons.description_outlined),
                label: const Text('Открыть отчёт'),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _DebugEndpointCard extends StatelessWidget {
  final String url;

  const _DebugEndpointCard({required this.url});

  @override
  Widget build(BuildContext context) {
    assert(() {
      return true;
    }());

    return Card(
      child: ExpansionTile(
        title: const Text(
          'Диагностика подключения',
          style: TextStyle(fontWeight: FontWeight.w800),
        ),
        subtitle: Text(
          Uri.parse(url).scheme == 'https'
              ? 'Подключение по HTTPS'
              : 'Подключение к настроенному серверу',
        ),
        childrenPadding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
        children: [
          SelectableText(
            url,
            style: Theme.of(context).textTheme.bodySmall,
          ),
          const SizedBox(height: 8),
          Text(
            ApiConfig.connectionHint(url),
            style: Theme.of(context).textTheme.bodySmall?.copyWith(
                  color: AppTheme.muted,
                ),
          ),
        ],
      ),
    );
  }
}
