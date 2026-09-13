import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'api_config.dart';
import 'app_theme.dart';
import 'ar_measure_channel.dart';
import 'unified_analysis_models.dart';
import 'unified_analysis_report_page.dart';

class ArborScanPage extends StatefulWidget {
  const ArborScanPage({super.key});

  @override
  State<ArborScanPage> createState() => _ArborScanPageState();
}

class _ArborScanPageState extends State<ArborScanPage> {
  static const String _historyKey = 'arborscan_history';

  final ImagePicker _picker = ImagePicker();

  File? _imageFile;
  ImageSource? _imageSource;
  ArMeasureResult? _arResult;
  UnifiedAnalysisResult? _lastResult;

  bool _loading = false;
  bool _openingAr = false;
  String? _error;

  String get _apiUrl => '${ApiConfig.v4BaseUrl}/v4/analyze-tree';

  Future<void> _pickImage(ImageSource source) async {
    try {
      final picked = await _picker.pickImage(
        source: source,
      );
      if (picked == null || !mounted) return;

      setState(() {
        _imageFile = File(picked.path);
        _imageSource = source;
        _arResult = null;
        _lastResult = null;
        _error = null;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = 'Не удалось выбрать изображение: $e');
    }
  }

  Future<void> _openAr() async {
    if (_openingAr) return;

    setState(() {
      _openingAr = true;
      _error = null;
    });

    try {
      final result = await ArMeasureChannel.openArMeasure();
      if (!mounted || result == null) return;

      setState(() {
        _arResult = result;
        _lastResult = null;
      });

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            'AR готов: H ${result.heightMeters!.toStringAsFixed(2)} м · '
            'DBH ${result.trunkDiameterMeters!.toStringAsFixed(3)} м · '
            '${result.statusLabelRu}. Крона будет рассчитана по фото.',
          ),
        ),
      );
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

    setState(() {
      _loading = true;
      _error = null;
      _lastResult = null;
    });

    try {
      final request = http.MultipartRequest('POST', Uri.parse(_apiUrl));
      request.fields['include_images'] = 'true';

      final ar = _arResult;
      if (ar != null) {
        request.fields.addAll(ar.toV4FormFields());
      }

      request.files.add(
        await http.MultipartFile.fromPath('file', imageFile.path),
      );

      final streamed = await request.send().timeout(
            const Duration(seconds: 180),
          );
      final response = await http.Response.fromStream(streamed);

      if (response.statusCode < 200 || response.statusCode >= 300) {
        throw Exception(_extractServerMessage(response));
      }

      final decoded = jsonDecode(utf8.decode(response.bodyBytes));
      if (decoded is! Map) {
        throw const FormatException('Сервер вернул некорректный JSON');
      }

      final result = UnifiedAnalysisResult.fromJson(
        Map<String, dynamic>.from(decoded),
      );

      if (!mounted) return;
      setState(() => _lastResult = result);
      await _saveHistory(result);

      Uint8List? fallbackBytes;
      try {
        fallbackBytes = await imageFile.readAsBytes();
      } catch (_) {}

      if (!mounted) return;
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
      if (mounted) setState(() => _loading = false);
    }
  }

  Future<void> _saveHistory(UnifiedAnalysisResult result) async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final existing = prefs.getStringList(_historyKey) ?? <String>[];

      final historyRow = <String, dynamic>{
        'species': result.speciesName,
        'height': result.height.valueM,
        'crown': result.crownWidth.valueM,
        'trunk': result.trunkDiameter.valueM,
        'scale': result.pxToM,
        'riskIndex': null,
        'riskCategory': null,
        'lat': null,
        'lon': null,
        'address': null,
        'imageBase64': '',
        'timestamp': DateTime.now().toIso8601String(),
        'analysisId': result.analysisId,
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
          'Для локального debug-подключения проверьте SSH tunnel и adb reverse.';
    }
    if (text.contains('TimeoutException')) {
      return 'Анализ превысил допустимое время ожидания.';
    }
    return text.replaceFirst('Exception: ', '');
  }

  void _reset() {
    setState(() {
      _imageFile = null;
      _imageSource = null;
      _arResult = null;
      _lastResult = null;
      _error = null;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('ArborScan'),
            Text(
              'Unified Analysis v4',
              style: TextStyle(
                fontSize: 11,
                fontWeight: FontWeight.w600,
                color: AppTheme.muted,
              ),
            ),
          ],
        ),
        actions: [
          if (_imageFile != null)
            IconButton(
              tooltip: 'Начать заново',
              onPressed: _loading ? null : _reset,
              icon: const Icon(Icons.refresh),
            ),
        ],
      ),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.fromLTRB(16, 12, 16, 28),
          children: [
            const _IntroCard(),
            const SizedBox(height: 16),
            _StepHeader(
              number: '1',
              title: 'Фотография дерева',
              subtitle:
                  'Фото используется для сегментации, геометрии и определения породы. Фото и AR можно выполнять в любом порядке.',
              done: _imageFile != null,
            ),
            const SizedBox(height: 8),
            _PhotoCard(
              file: _imageFile,
              source: _imageSource,
              loading: _loading,
              onCamera: () => _pickImage(ImageSource.camera),
              onGallery: () => _pickImage(ImageSource.gallery),
            ),
            const SizedBox(height: 18),
            _StepHeader(
              number: '2',
              title: 'AR-измерение',
              subtitle:
                  'Можно запускать независимо от фото. Помощник подбирает позицию; AR измеряет высоту и DBH, а крона затем рассчитывается по фото через CV + AR-масштаб.',
              done: _arResult != null,
              optional: true,
            ),
            const SizedBox(height: 8),
            _ArCard(
              enabled: !_loading,
              opening: _openingAr,
              result: _arResult,
              onOpen: _openAr,
            ),
            const SizedBox(height: 18),
            _StepHeader(
              number: '3',
              title: 'Единый анализ',
              subtitle: _arResult != null
                  ? 'AR + CV + PlantNet будут объединены в один результат.'
                  : 'Без AR система не будет придумывать физические размеры: метры останутся пустыми.',
              done: _lastResult != null,
            ),
            const SizedBox(height: 8),
            _AnalyzeCard(
              enabled: _imageFile != null && !_openingAr,
              loading: _loading,
              hasAr: _arResult != null,
              onAnalyze: _analyze,
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
                  if (!mounted) return;
                  await Navigator.of(context).push(
                    MaterialPageRoute(
                      builder: (_) => UnifiedAnalysisReportPage(
                        result: _lastResult!,
                        fallbackImageBytes: fallbackBytes,
                      ),
                    ),
                  );
                },
              ),
            ],
            const SizedBox(height: 18),
            _DebugEndpointCard(url: _apiUrl),
          ],
        ),
      ),
    );
  }
}

class _IntroCard extends StatelessWidget {
  const _IntroCard();

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              width: 48,
              height: 48,
              decoration: BoxDecoration(
                color: AppTheme.primary.withOpacity(0.10),
                borderRadius: BorderRadius.circular(16),
              ),
              child: const Icon(Icons.park, color: AppTheme.primary),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Одно дерево — один анализ',
                    style: Theme.of(context).textTheme.titleMedium?.copyWith(
                          fontWeight: FontWeight.w800,
                        ),
                  ),
                  const SizedBox(height: 5),
                  Text(
                    'Высота, крона и DBH больше не считаются отдельными режимами. '
                    'AR даёт физическую геометрию, CV анализирует изображение, PlantNet определяет породу.',
                    style: Theme.of(context).textTheme.bodySmall?.copyWith(
                          color: AppTheme.muted,
                          height: 1.4,
                        ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _StepHeader extends StatelessWidget {
  final String number;
  final String title;
  final String subtitle;
  final bool done;
  final bool optional;

  const _StepHeader({
    required this.number,
    required this.title,
    required this.subtitle,
    required this.done,
    this.optional = false,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Container(
          width: 30,
          height: 30,
          alignment: Alignment.center,
          decoration: BoxDecoration(
            color: done ? AppTheme.success : AppTheme.primary,
            shape: BoxShape.circle,
          ),
          child: done
              ? const Icon(Icons.check, color: Colors.white, size: 18)
              : Text(
                  number,
                  style: const TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.w900,
                  ),
                ),
        ),
        const SizedBox(width: 10),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Flexible(
                    child: Text(
                      title,
                      style: Theme.of(context).textTheme.titleMedium?.copyWith(
                            fontWeight: FontWeight.w800,
                          ),
                    ),
                  ),
                  if (optional) ...[
                    const SizedBox(width: 8),
                    Ui.badge(
                      text: 'рекомендуется',
                      color: AppTheme.primary,
                    ),
                  ],
                ],
              ),
              const SizedBox(height: 3),
              Text(
                subtitle,
                style: Theme.of(context).textTheme.bodySmall?.copyWith(
                      color: AppTheme.muted,
                      height: 1.35,
                    ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}

class _PhotoCard extends StatelessWidget {
  final File? file;
  final ImageSource? source;
  final bool loading;
  final VoidCallback onCamera;
  final VoidCallback onGallery;

  const _PhotoCard({
    required this.file,
    required this.source,
    required this.loading,
    required this.onCamera,
    required this.onGallery,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          children: [
            if (file != null) ...[
              ClipRRect(
                borderRadius: BorderRadius.circular(14),
                child: AspectRatio(
                  aspectRatio: 4 / 3,
                  child: Image.file(
                    file!,
                    fit: BoxFit.cover,
                    errorBuilder: (_, __, ___) => Container(
                      alignment: Alignment.center,
                      color: AppTheme.bg,
                      child: const Icon(Icons.broken_image_outlined),
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 10),
              Row(
                children: [
                  Ui.badge(
                    text: source == ImageSource.camera ? 'Камера' : 'Галерея',
                    color: AppTheme.success,
                    icon: source == ImageSource.camera
                        ? Icons.camera_alt_outlined
                        : Icons.photo_library_outlined,
                  ),
                  const Spacer(),
                  TextButton.icon(
                    onPressed: loading ? null : onGallery,
                    icon: const Icon(Icons.swap_horiz),
                    label: const Text('Заменить'),
                  ),
                ],
              ),
            ] else ...[
              Container(
                width: double.infinity,
                padding: const EdgeInsets.symmetric(vertical: 34),
                decoration: BoxDecoration(
                  color: AppTheme.bg,
                  borderRadius: BorderRadius.circular(14),
                  border: Border.all(color: AppTheme.border),
                ),
                child: const Column(
                  children: [
                    Icon(
                      Icons.add_a_photo_outlined,
                      size: 38,
                      color: AppTheme.muted,
                    ),
                    SizedBox(height: 8),
                    Text(
                      'Добавьте одно хорошо видимое дерево',
                      style: TextStyle(fontWeight: FontWeight.w700),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 12),
              Row(
                children: [
                  Expanded(
                    child: ElevatedButton.icon(
                      onPressed: loading ? null : onCamera,
                      icon: const Icon(Icons.camera_alt_outlined),
                      label: const Text('Камера'),
                    ),
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    child: OutlinedButton.icon(
                      onPressed: loading ? null : onGallery,
                      icon: const Icon(Icons.photo_library_outlined),
                      label: const Text('Галерея'),
                    ),
                  ),
                ],
              ),
            ],
          ],
        ),
      ),
    );
  }
}

class _ArCard extends StatelessWidget {
  final bool enabled;
  final bool opening;
  final ArMeasureResult? result;
  final VoidCallback onOpen;

  const _ArCard({
    required this.enabled,
    required this.opening,
    required this.result,
    required this.onOpen,
  });

  @override
  Widget build(BuildContext context) {
    final ar = result;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (ar == null) ...[
              Text(
                'Measurement Coach оценивает позицию и стабильность. AR даёт физическую высоту и DBH, а крона рассчитывается по фотографии.',
                style: Theme.of(context).textTheme.bodySmall?.copyWith(
                      color: AppTheme.muted,
                      height: 1.4,
                    ),
              ),
              const SizedBox(height: 12),
              SizedBox(
                width: double.infinity,
                child: ElevatedButton.icon(
                  onPressed: enabled && !opening ? onOpen : null,
                  icon: opening
                      ? const SizedBox(
                          width: 18,
                          height: 18,
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                            color: Colors.white,
                          ),
                        )
                      : const Icon(Icons.view_in_ar_outlined),
                  label: Text(opening ? 'Открываем AR…' : 'Измерить дерево в AR'),
                ),
              ),
            ] else ...[
              Wrap(
                spacing: 8,
                runSpacing: 8,
                children: [
                  Ui.badge(
                    text: 'H ${ar.heightMeters!.toStringAsFixed(2)} м',
                    color: AppTheme.success,
                    icon: Icons.height,
                  ),
                  Ui.badge(
                    text: 'DBH ${ar.trunkDiameterMeters!.toStringAsFixed(3)} м',
                    color: AppTheme.success,
                    icon: Icons.circle_outlined,
                  ),
                  Ui.badge(
                    text: ar.statusLabelRu,
                    color: ar.overallStatus == 'good'
                        ? AppTheme.success
                        : AppTheme.warning,
                    icon: Icons.assistant_outlined,
                  ),
                ],
              ),
              const SizedBox(height: 10),
              Text(
                'DBH измерен на высоте ${ar.trunkMeasurementHeightMeters!.toStringAsFixed(2)} м. '
                'Крона будет вычислена как CV + AR-масштаб по измеренной высоте.${ar.dbhRepeatSpreadMeters != null ? ' Разброс DBH: ${(ar.dbhRepeatSpreadMeters! * 1000).toStringAsFixed(0)} мм.' : ''}',
                style: Theme.of(context).textTheme.bodySmall?.copyWith(
                      color: AppTheme.muted,
                    ),
              ),
              if (ar.warnings.isNotEmpty) ...[
                const SizedBox(height: 6),
                Text(
                  'AR предупреждения: ${ar.warnings.join(', ')}',
                  style: Theme.of(context).textTheme.bodySmall?.copyWith(
                        color: AppTheme.warning,
                      ),
                ),
              ],
              const SizedBox(height: 10),
              SizedBox(
                width: double.infinity,
                child: OutlinedButton.icon(
                  onPressed: enabled && !opening ? onOpen : null,
                  icon: const Icon(Icons.replay),
                  label: const Text('Повторить AR'),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

class _AnalyzeCard extends StatelessWidget {
  final bool enabled;
  final bool loading;
  final bool hasAr;
  final VoidCallback onAnalyze;

  const _AnalyzeCard({
    required this.enabled,
    required this.loading,
    required this.hasAr,
    required this.onAnalyze,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Icon(
                  hasAr ? Icons.hub_outlined : Icons.visibility_outlined,
                  color: hasAr ? AppTheme.success : AppTheme.warning,
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: Text(
                    hasAr
                        ? 'Готово к объединению AR + CV + PlantNet.'
                        : 'Будет выполнен анализ изображения без метрических догадок.',
                    style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                          fontWeight: FontWeight.w700,
                        ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            SizedBox(
              width: double.infinity,
              child: ElevatedButton.icon(
                onPressed: enabled && !loading ? onAnalyze : null,
                icon: loading
                    ? const SizedBox(
                        width: 18,
                        height: 18,
                        child: CircularProgressIndicator(
                          strokeWidth: 2,
                          color: Colors.white,
                        ),
                      )
                    : const Icon(Icons.analytics_outlined),
                label: Text(
                  loading
                      ? 'Анализируем…'
                      : hasAr
                          ? 'Создать единый анализ'
                          : 'Проанализировать фото',
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
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
          'Alpha / подключение',
          style: TextStyle(fontWeight: FontWeight.w800),
        ),
        subtitle: const Text(
          'В production этот блок можно убрать после перехода на HTTPS.',
        ),
        childrenPadding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
        children: [
          SelectableText(
            url,
            style: Theme.of(context).textTheme.bodySmall,
          ),
          const SizedBox(height: 8),
          Text(
            'Для debug APK через USB: SSH tunnel на ПК + adb reverse tcp:8001 tcp:8001.',
            style: Theme.of(context).textTheme.bodySmall?.copyWith(
                  color: AppTheme.muted,
                ),
          ),
        ],
      ),
    );
  }
}
