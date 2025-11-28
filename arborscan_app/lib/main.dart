import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:lottie/lottie.dart';

import 'feedback_page.dart';

void main() {
  runApp(const ArborScanApp());
}

/// ============================
///  Модель результата анализа
/// ============================
class AnalysisResult {
  final String species;
  final double? height;
  final double? crown;
  final double? trunk;
  final double? scale;
  final double? riskIndex;
  final String? riskCategory;
  final double? lat;
  final double? lon;
  final String? address;
  final String imageBase64;
  final DateTime timestamp;

  // analysis ID
  final String analysisId;

  AnalysisResult({
    required this.species,
    required this.imageBase64,
    required this.timestamp,
    required this.analysisId,
    this.height,
    this.crown,
    this.trunk,
    this.scale,
    this.riskIndex,
    this.riskCategory,
    this.lat,
    this.lon,
    this.address,
  });

  Map<String, dynamic> toJson() => {
        'species': species,
        'height': height,
        'crown': crown,
        'trunk': trunk,
        'scale': scale,
        'riskIndex': riskIndex,
        'riskCategory': riskCategory,
        'lat': lat,
        'lon': lon,
        'address': address,
        'imageBase64': imageBase64,
        'timestamp': timestamp.toIso8601String(),
        'analysisId': analysisId,
      };

  factory AnalysisResult.fromJson(Map<String, dynamic> json) => AnalysisResult(
        species: json['species'] ?? 'Неизвестно',
        height: (json['height'] as num?)?.toDouble(),
        crown: (json['crown'] as num?)?.toDouble(),
        trunk: (json['trunk'] as num?)?.toDouble(),
        scale: (json['scale'] as num?)?.toDouble(),
        riskIndex: (json['riskIndex'] as num?)?.toDouble(),
        riskCategory: json['riskCategory'] as String?,
        lat: (json['lat'] as num?)?.toDouble(),
        lon: (json['lon'] as num?)?.toDouble(),
        address: json['address'] as String?,
        imageBase64: json['imageBase64'] ?? '',
        timestamp: DateTime.parse(json['timestamp']),
        analysisId: json['analysisId'] ?? '',
      );
}

/// ============================
///   Приложение + темы
/// ============================
class ArborScanApp extends StatelessWidget {
  const ArborScanApp({super.key});

  @override
  Widget build(BuildContext context) {
    final baseTheme = ThemeData(
      useMaterial3: true,
      colorScheme: ColorScheme.fromSeed(
        seedColor: const Color(0xFF2F6B3A),
        background: const Color(0xFFF5F6EC),
      ),
      scaffoldBackgroundColor: const Color(0xFFF5F6EC),
    );

    return MaterialApp(
      title: 'ArborScan',
      theme: baseTheme.copyWith(
        appBarTheme: baseTheme.appBarTheme.copyWith(
          backgroundColor: baseTheme.scaffoldBackgroundColor,
          elevation: 0,
          centerTitle: false,
          titleTextStyle: const TextStyle(
            fontSize: 22,
            fontWeight: FontWeight.w700,
            color: Colors.black,
          ),
        ),
        cardTheme: baseTheme.cardTheme.copyWith(
          elevation: 0,
          color: Colors.white,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(24),
          ),
        ),
        filledButtonTheme: FilledButtonThemeData(
          style: FilledButton.styleFrom(
            padding: const EdgeInsets.symmetric(vertical: 16),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(999),
            ),
          ),
        ),
      ),
      home: const ArborScanPage(),
    );
  }
}

/// ============================
///      Главный экран
/// ============================
class ArborScanPage extends StatefulWidget {
  const ArborScanPage({super.key});

  @override
  State<ArborScanPage> createState() => _ArborScanPageState();
}

class _ArborScanPageState extends State<ArborScanPage> {
  final ImagePicker _picker = ImagePicker();

  File? _imageFile;
  Uint8List? _annotatedImageBytes;
  Map<String, dynamic>? _result;

  bool _isLoading = false;
  String? _error;

  static const String _apiUrl =
      'https://arborscanbackend-production.up.railway.app/analyze-tree';

  static const String _feedbackUrl =
      'https://arborscanbackend-production.up.railway.app/feedback';

  static const String _historyKey = 'arborscan_history';
  final List<AnalysisResult> _history = [];

  @override
  void initState() {
    super.initState();
    _loadHistory();
  }

  Future<void> _loadHistory() async {
    final prefs = await SharedPreferences.getInstance();
    final list = prefs.getStringList(_historyKey);
    if (list == null) return;

    setState(() {
      _history
        ..clear()
        ..addAll(list.map((e) {
          final json = jsonDecode(e) as Map<String, dynamic>;
          return AnalysisResult.fromJson(json);
        }));
    });
  }

  Future<void> _saveHistory() async {
    final prefs = await SharedPreferences.getInstance();
    final encoded = _history.map((e) => jsonEncode(e.toJson())).toList();
    await prefs.setStringList(_historyKey, encoded);
  }

  Future<void> _clearHistory() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_historyKey);
    setState(() {
      _history.clear();
    });
  }

  Future<void> _pickImage(ImageSource source) async {
    try {
      final picked = await _picker.pickImage(source: source, imageQuality: 90);
      if (picked == null) return;

      setState(() {
        _imageFile = File(picked.path);
        _annotatedImageBytes = null;
        _result = null;
        _error = null;
      });
    } catch (e) {
      setState(() {
        _error = 'Ошибка при выборе изображения: $e';
      });
    }
  }

  Future<void> _analyze() async {
    if (_imageFile == null) return;

    setState(() {
      _isLoading = true;
      _error = null;
    });

    try {
      final uri = Uri.parse(_apiUrl);
      final request = http.MultipartRequest('POST', uri);
      request.files.add(
        await http.MultipartFile.fromPath('file', _imageFile!.path),
      );

      final streamed = await request.send();
      final response = await http.Response.fromStream(streamed);

      if (response.statusCode != 200) {
        dynamic body;
        try {
          body = jsonDecode(response.body);
        } catch (_) {}
        final msg = body is Map && body['error'] != null
            ? body['error'].toString()
            : 'Ошибка сервера (${response.statusCode})';
        throw Exception(msg);
      }

      final data = jsonDecode(response.body) as Map<String, dynamic>;

      final annotatedB64 = data['annotated_image_base64'] as String?;
      Uint8List? annotatedBytes;
      if (annotatedB64 != null && annotatedB64.isNotEmpty) {
        annotatedBytes = base64Decode(annotatedB64);
      }

      final risk = (data['risk'] ?? {}) as Map<String, dynamic>;
      final gps = data['gps'] as Map<String, dynamic>?;
      final String? address = data['address'] as String?;

      final double? height = (data['height_m'] as num?)?.toDouble();
      final double? crown = (data['crown_width_m'] as num?)?.toDouble();
      final double? trunk = (data['trunk_diameter_m'] as num?)?.toDouble();
      final double? scale = (data['scale_px_to_m'] as num?)?.toDouble();

      final double? riskIndex = (risk['index'] as num?)?.toDouble();
      final String? riskCategory = risk['category'] as String?;

      final analysisId = data['analysis_id'] as String? ?? '';

      final historyItem = AnalysisResult(
        species: data['species'] as String? ?? 'Неизвестно',
        height: height,
        crown: crown,
        trunk: trunk,
        scale: scale,
        riskIndex: riskIndex,
        riskCategory: riskCategory,
        lat: (gps?['lat'] as num?)?.toDouble(),
        lon: (gps?['lon'] as num?)?.toDouble(),
        address: address,
        imageBase64: annotatedB64 ?? '',
        timestamp: DateTime.now(),
        analysisId: analysisId,
      );

      setState(() {
        _annotatedImageBytes = annotatedBytes;
        _result = data;
        _history.insert(0, historyItem);
      });

      await _saveHistory();
    } catch (e) {
      setState(() {
        _error = e.toString();
      });
    } finally {
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
      }
    }
  }
  String _capitalise(String s) {
    if (s.isEmpty) return s;
    return s[0].toUpperCase() + s.substring(1);
  }

  void _showRiskDetails() {
    final risk = _result?['risk'] as Map<String, dynamic>?;

    final explanation = (risk?['explanation'] as List?)?.cast<String>() ?? [];

    showModalBottomSheet(
      context: context,
      showDragHandle: true,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
      ),
      builder: (ctx) {
        final index = (risk?['index'] as num?)?.toDouble();
        final cat = risk?['category'] as String?;

        return Padding(
          padding: const EdgeInsets.fromLTRB(16, 8, 16, 24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  const Icon(Icons.warning_amber_rounded, color: Colors.orange),
                  const SizedBox(width: 12),
                  Text(
                    'Детальный разбор риска',
                    style: Theme.of(context).textTheme.titleMedium?.copyWith(
                          fontWeight: FontWeight.w700,
                        ),
                  ),
                ],
              ),
              const SizedBox(height: 8),
              if (index != null && cat != null)
                Text(
                  'Индекс: ${index.toStringAsFixed(2)} (${_capitalise(cat)})',
                  style: Theme.of(context).textTheme.bodyMedium,
                ),
              const SizedBox(height: 12),
              if (explanation.isNotEmpty)
                ...explanation.map(
                  (line) => Padding(
                    padding: const EdgeInsets.symmetric(vertical: 4),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text('• '),
                        Expanded(child: Text(line)),
                      ],
                    ),
                  ),
                ),
            ],
          ),
        );
      },
    );
  }

  Widget _buildRiskChip() {
    final risk = _result?['risk'] as Map<String, dynamic>?;
    if (risk == null) return const SizedBox.shrink();

    final double? index = (risk['index'] as num?)?.toDouble();
    final String? category = risk['category'] as String?;

    if (index == null || category == null) return const SizedBox.shrink();

    Color bg;
    Color fg;

    switch (category) {
      case 'низкий':
        bg = const Color(0xFFD9F5DC);
        fg = const Color(0xFF1B5E20);
        break;
      case 'средний':
        bg = const Color(0xFFFFF4D1);
        fg = const Color(0xFF8D6E00);
        break;
      default:
        bg = const Color(0xFFFFE1E1);
        fg = const Color(0xFFB71C1C);
        break;
    }

    return GestureDetector(
      onTap: _showRiskDetails,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
        decoration: BoxDecoration(
          color: bg,
          borderRadius: BorderRadius.circular(999),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.warning_amber_rounded, size: 18, color: fg),
            const SizedBox(width: 6),
            Text(
              'Риск: ${_capitalise(category)} (${index.toStringAsFixed(2)})',
              style: TextStyle(
                color: fg,
                fontWeight: FontWeight.w600,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildImageCard() {
    final theme = Theme.of(context);

    Widget content;
    if (_imageFile == null && _annotatedImageBytes == null) {
      content = Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          SizedBox(
            height: 140,
            child: Lottie.asset(
              'assets/lottie/tree.json',
              repeat: true,
            ),
          ),
          const SizedBox(height: 12),
          Text(
            'Добавьте фото дерева\nиз камеры или галереи',
            textAlign: TextAlign.center,
            style: theme.textTheme.bodyMedium?.copyWith(
              color: Colors.black54,
            ),
          ),
        ],
      );
    } else {
      final imageWidget = _annotatedImageBytes != null
          ? Image.memory(
              _annotatedImageBytes!,
              fit: BoxFit.cover,
            )
          : (_imageFile != null
              ? Image.file(
                  _imageFile!,
                  fit: BoxFit.cover,
                )
              : const SizedBox());

      content = ClipRRect(
        borderRadius: BorderRadius.circular(20),
        child: AspectRatio(
          aspectRatio: 3 / 4,
          child: imageWidget,
        ),
      );
    }

    return Card(
      margin: EdgeInsets.zero,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: AnimatedSize(
          duration: const Duration(milliseconds: 200),
          curve: Curves.easeInOut,
          child: content,
        ),
      ),
    );
  }

  Widget _buildResultCard() {
    final theme = Theme.of(context);

    if (_result == null) {
      return Card(
        margin: EdgeInsets.zero,
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Icon(Icons.info_outline, color: Colors.black45),
              const SizedBox(width: 12),
              Expanded(
                child: Text(
                  'Результаты появятся после анализа.\n'
                  'Загрузите фото дерева и нажмите «Анализировать».',
                  style: theme.textTheme.bodyMedium?.copyWith(
                    color: Colors.black54,
                  ),
                ),
              ),
            ],
          ),
        ),
      );
    }

    final species = _result!['species'] as String? ?? '—';
    final height = (_result!['height_m'] as num?)?.toDouble();
    final crown = (_result!['crown_width_m'] as num?)?.toDouble();
    final trunk = (_result!['trunk_diameter_m'] as num?)?.toDouble();
    final scale = (_result!['scale_px_to_m'] as num?)?.toDouble();

    final gps = _result!['gps'] as Map<String, dynamic>?;
    final String? address = _result!['address'] as String?;

    String formatValue(double? v, {String suffix = 'м'}) {
      if (v == null) return '—';
      return '${v.toStringAsFixed(2)} $suffix';
    }

    String scaleText;
    if (scale == null) {
      scaleText = 'Масштаб не найден (нет палки 1 м).';
    } else {
      scaleText = '1 px ≈ ${scale.toStringAsFixed(4)} м';
    }

    return Card(
      margin: EdgeInsets.zero,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Результаты анализа',
                        style: theme.textTheme.titleMedium?.copyWith(
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        'Вид дерева: $species',
                        style: theme.textTheme.bodyMedium,
                      ),
                    ],
                  ),
                ),
                const SizedBox(width: 8),
                _buildRiskChip(),
              ],
            ),
            const SizedBox(height: 12),

            Row(
              children: [
                Expanded(
                  child: _MetricTile(
                    label: 'Высота',
                    value: formatValue(height),
                    icon: Icons.height,
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: _MetricTile(
                    label: 'Крона',
                    value: formatValue(crown),
                    icon: Icons.filter_hdr,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Row(
              children: [
                Expanded(
                  child: _MetricTile(
                    label: 'Диаметр ствола',
                    value: formatValue(trunk),
                    icon: Icons.circle_outlined,
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: _MetricTile(
                    label: 'Масштаб',
                    value: scaleText,
                    icon: Icons.straighten,
                    isSecondary: true,
                  ),
                ),
              ],
            ),

            const SizedBox(height: 12),

            if (address != null && address.isNotEmpty)
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
                decoration: BoxDecoration(
                  color: const Color(0xFFE8F3FF),
                  borderRadius: BorderRadius.circular(16),
                ),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Icon(Icons.location_on_outlined,
                        size: 20, color: Color(0xFF1565C0)),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        address,
                        style: theme.textTheme.bodySmall?.copyWith(
                          color: const Color(0xFF0D47A1),
                        ),
                      ),
                    ),
                  ],
                ),
              )
            else if (gps != null)
              Text(
                'Координаты: ${gps['lat']}, ${gps['lon']}',
                style: theme.textTheme.bodySmall?.copyWith(
                  color: Colors.black54,
                ),
              )
            else
              Text(
                'GPS-данные в фото не найдены.',
                style: theme.textTheme.bodySmall?.copyWith(
                  color: Colors.black54,
                ),
              ),
          ],
        ),
      ),
    );
  }
  Future<void> _sendFeedbackToServer(
    Map<String, dynamic> feedback,
    String analysisId,
  ) async {
    final body = {
      "analysis_id": analysisId,
      "use_for_training": true,
      "tree_ok": feedback["tree_ok"],
      "stick_ok": feedback["stick_ok"],
      "params_ok": feedback["params_ok"],
      "species_ok": feedback["species_ok"],
      "correct_species": feedback["correct_species"],
      "user_mask_base64": null, // будет добавлено позже на шаге 3.2
    };

    try {
      final uri = Uri.parse(_feedbackUrl);
      final resp = await http.post(
        uri,
        headers: {"Content-Type": "application/json"},
        body: jsonEncode(body),
      );

      if (!mounted) return;

      if (resp.statusCode == 200) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text("Спасибо! Анализ подтверждён.")),
        );
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
                "Ошибка отправки фидбека: ${resp.statusCode.toString()}"),
          ),
        );
      }
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Ошибка отправки: $e")),
      );
    }
  }

  Future<void> _openFeedback() async {
    if (_result == null) return;

    final data = _result!;
    final analysisId = data['analysis_id']?.toString();
    final annotatedB64 = data['annotated_image_base64'] as String?;

    if (analysisId == null || analysisId.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Сервер не прислал analysis_id")),
      );
      return;
    }

    if (_annotatedImageBytes == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Нет аннотированного изображения.")),
      );
      return;
    }

    // Загружаем оригинальное фото
final Uint8List originalBytes = await _imageFile!.readAsBytes();

// Раскодируем аннотированное фото
final Uint8List annotatedBytes = base64Decode(
  annotatedB64 ?? "",
);

// Открываем FeedbackPage
final feedback = await Navigator.push<Map<String, dynamic>?>(
  context,
  MaterialPageRoute(
    builder: (_) => FeedbackPage(
      analysisId: analysisId,
      originalBytes: originalBytes,
      annotatedBytes: annotatedBytes,
      species: data['species'] ?? 'Неизвестно',
      heightM: (data['height_m'] as num?)?.toDouble(),
      crownWidthM: (data['crown_width_m'] as num?)?.toDouble(),
      trunkDiameterM: (data['trunk_diameter_m'] as num?)?.toDouble(),
    ),
  ),
);



    if (feedback != null) {
      await _sendFeedbackToServer(feedback, analysisId);
    }
  }

  Future<void> _openHistory() async {
    final cleared = await Navigator.of(context).push<bool>(
      MaterialPageRoute(
        builder: (_) => HistoryPage(
          items: _history,
        ),
      ),
    );

    if (cleared == true) {
      await _clearHistory();
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(
        title: const Text('ArborScan'),
        actions: [
          IconButton(
            icon: const Icon(Icons.history),
            tooltip: 'История',
            onPressed: _history.isEmpty ? null : _openHistory,
          ),
        ],
      ),
      body: Stack(
        children: [
          SafeArea(
            child: SingleChildScrollView(
              padding: const EdgeInsets.fromLTRB(16, 16, 16, 24),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Анализ деревьев\nс помощью ИИ',
                    style: theme.textTheme.headlineSmall?.copyWith(
                      fontWeight: FontWeight.w800,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    'Определение породы, размеров и оценки риска падения.',
                    style: theme.textTheme.bodyMedium?.copyWith(
                      color: Colors.black54,
                    ),
                  ),
                  const SizedBox(height: 16),

                  // Фото
                  _buildImageCard(),
                  const SizedBox(height: 16),

                  // Результат
                  _buildResultCard(),
                  const SizedBox(height: 12),

                  // >>> КНОПКА ПОДТВЕРЖДЕНИЯ АНАЛИЗА <<<
                  if (_annotatedImageBytes != null &&
                      _result != null &&
                      _result?['analysis_id'] != null)
                    SizedBox(
                      width: double.infinity,
                      child: FilledButton.icon(
                        onPressed: _openFeedback,
                        icon: const Icon(Icons.check_circle_outline),
                        label: const Text('Подтвердить анализ'),
                        style: FilledButton.styleFrom(
                          padding: const EdgeInsets.symmetric(vertical: 16),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(999),
                          ),
                        ),
                      ),
                    ),

                  if (_annotatedImageBytes != null &&
                      _result != null &&
                      _result?['analysis_id'] != null)
                    const SizedBox(height: 16),

                  // Кнопки выбора изображения
                  Row(
                    children: [
                      Expanded(
                        child: OutlinedButton.icon(
                          onPressed: () => _pickImage(ImageSource.camera),
                          icon: const Icon(Icons.photo_camera_outlined),
                          label: const Text('Камера'),
                          style: OutlinedButton.styleFrom(
                            padding: const EdgeInsets.symmetric(vertical: 14),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(999),
                            ),
                          ),
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: OutlinedButton.icon(
                          onPressed: () => _pickImage(ImageSource.gallery),
                          icon: const Icon(Icons.photo_library_outlined),
                          label: const Text('Галерея'),
                          style: OutlinedButton.styleFrom(
                            padding: const EdgeInsets.symmetric(vertical: 14),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(999),
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  Align(
                    alignment: Alignment.centerRight,
                    child: TextButton.icon(
                      onPressed:
                          _imageFile == null && _annotatedImageBytes == null
                              ? null
                              : () {
                                  setState(() {
                                    _imageFile = null;
                                    _annotatedImageBytes = null;
                                    _result = null;
                                    _error = null;
                                  });
                                },
                      icon: const Icon(Icons.clear),
                      label: const Text('Очистить'),
                    ),
                  ),
                  const SizedBox(height: 8),

                  // Кнопка анализа
                  SizedBox(
                    width: double.infinity,
                    child: FilledButton.icon(
                      onPressed: _imageFile == null || _isLoading
                          ? null
                          : _analyze,
                      icon: const Icon(Icons.play_arrow_rounded),
                      label: const Text(
                        'Анализировать',
                        style: TextStyle(fontWeight: FontWeight.w600),
                      ),
                    ),
                  ),

                  if (_error != null) ...[
                    const SizedBox(height: 12),
                    Container(
                      width: double.infinity,
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: const Color(0xFFFFE1E1),
                        borderRadius: BorderRadius.circular(16),
                      ),
                      child: Text(
                        _error!,
                        style: theme.textTheme.bodySmall?.copyWith(
                          color: const Color(0xFFB71C1C),
                        ),
                      ),
                    ),
                  ],
                ],
              ),
            ),
          ),

          // Лоадер поверх
          if (_isLoading)
            Container(
              color: Colors.black.withOpacity(0.2),
              child: Center(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    SizedBox(
                      height: 120,
                      child: Lottie.asset(
                        'assets/lottie/analysis.json',
                      ),
                    ),
                    const SizedBox(height: 8),
                    const Text(
                      'Анализ изображения...',
                      style: TextStyle(
                        color: Colors.white,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ],
                ),
              ),
            ),
        ],
      ),
    );
  }
}

/// Карточка маленькой метрики (высота, крона и т.п.)
class _MetricTile extends StatelessWidget {
  final String label;
  final String value;
  final IconData icon;
  final bool isSecondary;

  const _MetricTile({
    required this.label,
    required this.value,
    required this.icon,
    this.isSecondary = false,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: isSecondary ? const Color(0xFFF3F3F3) : const Color(0xFFF0F8F2),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Row(
        children: [
          Icon(icon, size: 20, color: Colors.black54),
          const SizedBox(width: 8),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  label,
                  style: theme.textTheme.labelSmall?.copyWith(
                    color: Colors.black54,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  value,
                  style: theme.textTheme.bodyMedium?.copyWith(
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

/// ============================
///     История анализов
/// ============================
class HistoryPage extends StatelessWidget {
  final List<AnalysisResult> items;

  const HistoryPage({super.key, required this.items});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(
        title: const Text('История анализов'),
        actions: [
          if (items.isNotEmpty)
            IconButton(
              icon: const Icon(Icons.delete_outline),
              tooltip: 'Очистить историю',
              onPressed: () async {
                final confirm = await showDialog<bool>(
                  context: context,
                  builder: (ctx) => AlertDialog(
                    title: const Text('Очистить историю?'),
                    content: const Text(
                        'Все сохранённые результаты анализов будут удалены.'),
                    actions: [
                      TextButton(
                        onPressed: () => Navigator.of(ctx).pop(false),
                        child: const Text('Отмена'),
                      ),
                      FilledButton(
                        onPressed: () => Navigator.of(ctx).pop(true),
                        child: const Text('Очистить'),
                      ),
                    ],
                  ),
                );
                if (confirm == true && context.mounted) {
                  Navigator.of(context).pop(true);
                }
              },
            ),
        ],
      ),
      body: items.isEmpty
          ? Center(
              child: Text(
                'История пуста.\nПроведите анализ, чтобы он здесь появился.',
                textAlign: TextAlign.center,
                style: theme.textTheme.bodyMedium?.copyWith(
                  color: Colors.black54,
                ),
              ),
            )
          : ListView.separated(
              padding: const EdgeInsets.fromLTRB(16, 16, 16, 24),
              itemCount: items.length,
              separatorBuilder: (_, __) => const SizedBox(height: 12),
              itemBuilder: (context, index) {
                final item = items[index];

                Widget? thumb;
                if (item.imageBase64.isNotEmpty) {
                  try {
                    final bytes = base64Decode(item.imageBase64);
                    thumb = ClipRRect(
                      borderRadius: BorderRadius.circular(12),
                      child: Image.memory(
                        bytes,
                        width: 64,
                        height: 64,
                        fit: BoxFit.cover,
                      ),
                    );
                  } catch (_) {}
                }

                String subtitle = [
                  'Дата: '
                      '${item.timestamp.day.toString().padLeft(2, '0')}.'
                      '${item.timestamp.month.toString().padLeft(2, '0')}.'
                      '${item.timestamp.year}  '
                      '${item.timestamp.hour.toString().padLeft(2, '0')}:'
                      '${item.timestamp.minute.toString().padLeft(2, '0')}',
                  if (item.height != null)
                    'Высота: ${item.height!.toStringAsFixed(2)} м',
                  if (item.crown != null)
                    'Крона: ${item.crown!.toStringAsFixed(2)} м',
                  if (item.trunk != null)
                    'Ствол: ${item.trunk!.toStringAsFixed(2)} м',
                  if (item.address != null && item.address!.isNotEmpty)
                    'Место: ${item.address}',
                ].join('\n');

                Color chipBg = const Color(0xFFEEEEEE);
                Color chipFg = Colors.black87;
                final cat = item.riskCategory;
                if (cat != null) {
                  switch (cat) {
                    case 'низкий':
                      chipBg = const Color(0xFFD9F5DC);
                      chipFg = const Color(0xFF1B5E20);
                      break;
                    case 'средний':
                      chipBg = const Color(0xFFFFF4D1);
                      chipFg = const Color(0xFF8D6E00);
                      break;
                    default:
                      chipBg = const Color(0xFFFFE1E1);
                      chipFg = const Color(0xFFB71C1C);
                      break;
                  }
                }

                return Card(
                  child: ListTile(
                    leading: thumb ??
                        Container(
                          width: 64,
                          height: 64,
                          decoration: BoxDecoration(
                            color: const Color(0xFFE0E0E0),
                            borderRadius: BorderRadius.circular(12),
                          ),
                          child: const Icon(Icons.park, color: Colors.green),
                        ),
                    title: Text(
                      'Вид: ${item.species}',
                      style: const TextStyle(fontWeight: FontWeight.w600),
                    ),
                    subtitle: Text(subtitle),
                    trailing: cat == null
                        ? null
                        : Container(
                            padding: const EdgeInsets.symmetric(
                                horizontal: 10, vertical: 6),
                            decoration: BoxDecoration(
                              color: chipBg,
                              borderRadius: BorderRadius.circular(999),
                            ),
                            child: Text(
                              'Риск: ${cat[0].toUpperCase()}${cat.substring(1)}',
                              style: TextStyle(
                                color: chipFg,
                                fontSize: 12,
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                          ),
                  ),
                );
              },
            ),
    );
  }
}
