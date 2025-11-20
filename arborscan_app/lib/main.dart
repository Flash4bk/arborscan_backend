import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:url_launcher/url_launcher.dart';
import 'package:lottie/lottie.dart';
import 'splash_screen.dart';

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
  final String imageBase64;
  final DateTime timestamp;
  final String? riskCategory;
  final double? riskIndex;

  AnalysisResult({
    required this.species,
    required this.imageBase64,
    required this.timestamp,
    this.height,
    this.crown,
    this.trunk,
    this.scale,
    this.riskCategory,
    this.riskIndex,
  });

  Map<String, dynamic> toJson() => {
        'species': species,
        'height': height,
        'crown': crown,
        'trunk': trunk,
        'scale': scale,
        'imageBase64': imageBase64,
        'timestamp': timestamp.toIso8601String(),
        'riskCategory': riskCategory,
        'riskIndex': riskIndex,
      };

  factory AnalysisResult.fromJson(Map<String, dynamic> json) => AnalysisResult(
        species: json['species'] as String,
        height: (json['height'] as num?)?.toDouble(),
        crown: (json['crown'] as num?)?.toDouble(),
        trunk: (json['trunk'] as num?)?.toDouble(),
        scale: (json['scale'] as num?)?.toDouble(),
        imageBase64: json['imageBase64'] as String,
        timestamp: DateTime.parse(json['timestamp'] as String),
        riskCategory: json['riskCategory'] as String?,
        riskIndex: (json['riskIndex'] as num?)?.toDouble(),
      );
}


/// ============================
///   Приложение + темы
/// ============================
class ArborScanApp extends StatelessWidget {
  const ArborScanApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'ArborScan',
      themeMode: ThemeMode.system,
      theme: ThemeData(
        useMaterial3: true,
        colorSchemeSeed: Colors.green,
        brightness: Brightness.light,
      ),
      darkTheme: ThemeData(
        useMaterial3: true,
        colorSchemeSeed: Colors.green,
        brightness: Brightness.dark,
      ),
      home: const SplashScreen(),
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

  bool _isLoading = false;
  String? _error;
  Uint8List? _annotatedImageBytes;
  Map<String, dynamic>? _result;

  // URL backend на Railway
  final String _apiUrl =
      'https://arborscanbackend-production.up.railway.app/analyze-tree';

  final List<AnalysisResult> _history = [];
  static const _historyKey = 'arborscan_history';

  static const _animDuration = Duration(milliseconds: 350);

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
      _history.clear();
      _history.addAll(
        list.map((e) => AnalysisResult.fromJson(jsonDecode(e))),
      );
    });
  }

  Future<void> _saveHistory() async {
    final prefs = await SharedPreferences.getInstance();
    final list =
        _history.map((e) => jsonEncode(e.toJson())).toList(growable: false);
    await prefs.setStringList(_historyKey, list);
  }

  Future<void> _pickImage(ImageSource source) async {
    setState(() {
      _error = null;
      _result = null;
      _annotatedImageBytes = null;
    });

    final XFile? picked = await _picker.pickImage(
      source: source,
      imageQuality: 92,
    );

    if (picked != null) {
      setState(() => _imageFile = File(picked.path));
    }
  }

  Future<void> _analyze() async {
    if (_imageFile == null) {
      setState(() => _error = 'Сначала выберите изображение.');
      return;
    }

    setState(() {
      _isLoading = true;
      _error = null;
    });

    try {
      final req = http.MultipartRequest('POST', Uri.parse(_apiUrl));
      req.files.add(
        await http.MultipartFile.fromPath('file', _imageFile!.path),
      );

      final streamed = await req.send();
      final response = await http.Response.fromStream(streamed);

      if (response.statusCode != 200) {
        setState(() => _error = 'Ошибка сервера: ${response.statusCode}');
        return;
      }

      final data = jsonDecode(response.body) as Map<String, dynamic>;

      if (data['error'] != null) {
        setState(() => _error = data['error'] as String);
        return;
      }

      Uint8List? imgBytes;
      if (data['annotated_image_base64'] != null) {
        imgBytes =
            base64Decode(data['annotated_image_base64'] as String);
      }

      // масштаб: старая (scale_m_per_px) или новая (scale_px_to_m)
      final scaleNum =
          data['scale_m_per_px'] ?? data['scale_px_to_m']; // оба варианта
      final scaleVal = (scaleNum is num) ? scaleNum.toDouble() : null;
      final riskMap = data['risk'] as Map<String, dynamic>?;

      final result = AnalysisResult(
        species: data['species'] as String,
        height: (data['height_m'] as num?)?.toDouble(),
        crown: (data['crown_width_m'] as num?)?.toDouble(),
        trunk: (data['trunk_diameter_m'] as num?)?.toDouble(),
        scale: scaleVal,
        imageBase64: imgBytes != null ? base64Encode(imgBytes) : '',
        timestamp: DateTime.now(),
        riskCategory: riskMap?['category'] as String?,
        riskIndex: (riskMap?['index'] as num?)?.toDouble(),
      );

      _history.insert(0, result);
      await _saveHistory();

      setState(() {
        _result = data;
        _annotatedImageBytes = imgBytes;
      });
    } catch (e) {
      setState(() => _error = 'Ошибка запроса: $e');
    } finally {
      setState(() => _isLoading = false);
    }
  }

  Widget _buildImageCard() {
    Widget child;

    if (_annotatedImageBytes != null) {
      child = Image.memory(_annotatedImageBytes!, fit: BoxFit.cover);
    } else if (_imageFile != null) {
      child = Image.file(_imageFile!, fit: BoxFit.cover);
    } else {
      child = Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Lottie.asset(
            'assets/lottie/tree.json',
            width: 160,
            repeat: true,
          ),
          const SizedBox(height: 12),
          const Text(
            'Выберите изображение дерева\nс эталонной палкой',
            textAlign: TextAlign.center,
            style: TextStyle(fontSize: 16),
          ),
        ],
      );
    }

    return Card(
      elevation: 3,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
      clipBehavior: Clip.antiAlias,
      child: AspectRatio(aspectRatio: 3 / 4, child: child),
    );
  }

  Widget _buildResultCard() {
    if (_error != null) {
      return Card(
        color: Colors.red.withOpacity(0.08),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Text(
            _error!,
            style: const TextStyle(color: Colors.red),
          ),
        ),
      );
    }

    if (_result == null) return const SizedBox.shrink();

    // поддерживаем оба варианта имени поля масштаба
    final scaleNum =
        _result!['scale_m_per_px'] ?? _result!['scale_px_to_m'];
    final scaleStr =
        (scaleNum is num) ? scaleNum.toStringAsFixed(4) : '-';

    return Card(
      elevation: 2,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
      child: Padding(
        padding: const EdgeInsets.all(18.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Вид: ${_result!['species']}',
              style: const TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.w600,
              ),
            ),
            const SizedBox(height: 8),
            Text('Высота: ${_result!['height_m'] ?? '-'} м'),
            Text('Ширина кроны: ${_result!['crown_width_m'] ?? '-'} м'),
            Text('Диаметр ствола: ${_result!['trunk_diameter_m'] ?? '-'} м'),
            Text('Масштаб: 1 px = $scaleStr м'),
          ],
        ),
      ),
    );
  }

  Future<void> _openInMaps(double lat, double lon) async {
    final uri = Uri.parse('https://www.google.com/maps?q=$lat,$lon');
    if (!await launchUrl(uri, mode: LaunchMode.externalApplication)) {
      setState(() {
        _error = 'Не удалось открыть карту';
      });
    }
  }

  Widget _buildLocationCard() {
    if (_result == null) return const SizedBox.shrink();

    final gps = _result!['gps'];
    final address = _result!['address'];

    if (gps == null) return const SizedBox.shrink();

    final lat = (gps['lat'] as num).toDouble();
    final lon = (gps['lon'] as num).toDouble();

    return Card(
      elevation: 2,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Lottie.asset(
              'assets/lottie/Location Icon Animation.json',
              width: 60,
              repeat: true,
            ),
            const SizedBox(width: 10),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'Местоположение',
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  const SizedBox(height: 6),
                  Text(
                    'Координаты: ${lat.toStringAsFixed(5)}°, ${lon.toStringAsFixed(5)}°',
                  ),
                  if (address != null && (address as String).isNotEmpty) ...[
                    const SizedBox(height: 4),
                    Text(
                      'Адрес: $address',
                      style: const TextStyle(fontSize: 14),
                    ),
                  ],
                  const SizedBox(height: 12),
                  Align(
                    alignment: Alignment.centerRight,
                    child: FilledButton.icon(
                      onPressed: () => _openInMaps(lat, lon),
                      icon: const Icon(Icons.map),
                      label: const Text('Открыть в Google Maps'),
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

  Widget _buildWeatherCard() {
    if (_result == null) return const SizedBox.shrink();

    final weather = _result!['weather'];
    if (weather == null) return const SizedBox.shrink();

    final temp = weather['temperature'] ?? weather['temperature_c'];
    final windSpeed = weather['wind_speed'] ?? weather['wind_speed_ms'];
    final windGust = weather['wind_gust'] ?? weather['wind_gust_ms'];
    final humidity = weather['humidity'] ?? weather['humidity_pct'];
    final pressure = weather['pressure'] ?? weather['pressure_hpa'];

    return Card(
      elevation: 2,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Lottie.asset(
              'assets/lottie/Weather-partly shower.json',
              width: 70,
              repeat: true,
            ),
            const SizedBox(width: 10),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'Погодные условия',
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  const SizedBox(height: 6),
                  if (temp != null)
                    Text('Температура: ${temp.toStringAsFixed(1)} °C'),
                  if (windSpeed != null)
                    Text('Скорость ветра: ${windSpeed.toStringAsFixed(1)} м/с'),
                  if (windGust != null)
                    Text('Порывы ветра: ${windGust.toStringAsFixed(1)} м/с'),
                  if (humidity != null)
                    Text('Влажность: ${humidity.toString()} %'),
                  if (pressure != null)
                    Text('Давление: ${pressure.toString()} гПа'),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSoilCard() {
    if (_result == null) return const SizedBox.shrink();

    final soil = _result!['soil'];
    if (soil == null) return const SizedBox.shrink();

    num? clay = soil['clay'] ?? soil['clay_pct'];
    num? sand = soil['sand'] ?? soil['sand_pct'];
    num? silt = soil['silt'] ?? soil['silt_pct'];
    num? org = soil['soc'] ?? soil['organic_carbon'];
    num? ph = soil['phh2o'] ?? soil['ph'];

    String soilType = 'Не определён';
    if (clay != null && sand != null) {
      final c = clay.toDouble();
      final s = sand.toDouble();
      if (c > 40) {
        soilType = 'Тяжёлая глинистая почва';
      } else if (s > 60) {
        soilType = 'Лёгкая песчаная почва';
      } else if (c > 25 && s > 25) {
        soilType = 'Суглинок';
      } else {
        soilType = 'Супесь / смешанный тип';
      }
    }

    String phText;
    if (ph == null) {
      phText = '-';
    } else {
      final p = ph.toDouble();
      if (p < 5.5) {
        phText = '$p (кислая)';
      } else if (p <= 7.5) {
        phText = '$p (нейтральная)';
      } else {
        phText = '$p (щелочная)';
      }
    }

    return Card(
      elevation: 2,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Почва',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.w600,
              ),
            ),
            const SizedBox(height: 6),
            Text('Тип: $soilType'),
            if (clay != null)
              Text('Глина: ${clay.toStringAsFixed(0)} %'),
            if (sand != null)
              Text('Песок: ${sand.toStringAsFixed(0)} %'),
            if (silt != null)
              Text('Ил: ${silt.toStringAsFixed(0)} %'),
            if (org != null)
              Text('Органическое вещество (SOC): '
                  '${org.toStringAsFixed(0)}'),
            Text('pH: $phText'),
          ],
        ),
      ),
    );
  }

  Widget _buildRiskCard() {
    if (_result == null) return const SizedBox.shrink();

    final risk = _result!['risk'];
    if (risk == null) return const SizedBox.shrink();

    final index = (risk['index'] as num?)?.toDouble() ?? 0.0;
    final category = (risk['category'] ?? 'неизвестно') as String;
    final explanation = (risk['explanation'] as List<dynamic>? ?? [])
        .map((e) => e.toString())
        .toList();

    Color color;
    if (category == 'низкий') {
      color = Colors.green;
    } else if (category == 'средний') {
      color = Colors.orange;
    } else {
      color = Colors.red;
    }

    return Card(
      elevation: 2,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(20),
      ),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Lottie.asset(
              'assets/lottie/Alert.json',
              width: 60,
              repeat: true,
            ),
            const SizedBox(width: 10),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Text(
                        'Риск падения',
                        style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.w600,
                          color: color,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 10),
                  Text(
                    'Индекс: ${index.toStringAsFixed(2)} ($category)',
                    style: TextStyle(fontSize: 16, color: color),
                  ),
                  const SizedBox(height: 12),
                  const Text(
                    'Факторы риска:',
                    style: TextStyle(fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 6),
                  ...explanation.map(
                    (e) => Padding(
                      padding: const EdgeInsets.symmetric(vertical: 2.0),
                      child: Text('• $e'),
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

  Widget _buildLoadingOverlay() {
    if (_isLoading == false) return const SizedBox.shrink();

    final steps = [
      'Детекция дерева',
      'Измерение размеров',
      'Определение породы',
      'Анализ местоположения',
      'Погодные условия',
      'Анализ почвы и расчёт риска',
    ];

    return AnimatedOpacity(
      opacity: _isLoading ? 1.0 : 0.0,
      duration: const Duration(milliseconds: 200),
      child: Container(
        color: Colors.black54,
        child: Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 340),
            child: Card(
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(20),
              ),
              child: Padding(
                padding: const EdgeInsets.all(18.0),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Lottie.asset(
                      'assets/lottie/Leaf scanning.json',
                      width: 120,
                      repeat: true,
                    ),
                    const SizedBox(height: 8),
                    const Text(
                      'Идёт анализ дерева',
                      style:
                          TextStyle(fontSize: 18, fontWeight: FontWeight.w600),
                    ),
                    const SizedBox(height: 12),
                    const LinearProgressIndicator(minHeight: 4),
                    const SizedBox(height: 12),
                    Align(
                      alignment: Alignment.centerLeft,
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          for (final s in steps)
                            Padding(
                              padding:
                                  const EdgeInsets.symmetric(vertical: 2.0),
                              child: Row(
                                children: [
                                  const Icon(Icons.circle,
                                      size: 8, color: Colors.grey),
                                  const SizedBox(width: 6),
                                  Expanded(child: Text(s)),
                                ],
                              ),
                            )
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }

  void _openHistory() {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => HistoryPage(history: _history),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final hasResult = _result != null || _error != null;

    return Scaffold(
      appBar: AppBar(
        title: const Text('ArborScan'),
        actions: [
          IconButton(
            icon: const Icon(Icons.history),
            tooltip: 'История измерений',
            onPressed: _history.isEmpty ? null : _openHistory,
          )
        ],
      ),
      body: Stack(
        children: [
          SafeArea(
            child: SingleChildScrollView(
              padding: const EdgeInsets.all(12),
              child: Column(
                children: [
                  _buildImageCard(),
                  const SizedBox(height: 12),
                  AnimatedSwitcher(
                    duration: _animDuration,
                    child: hasResult ? _buildResultCard() : const SizedBox(),
                  ),
                  const SizedBox(height: 12),
                  AnimatedSwitcher(
                    duration: _animDuration,
                    child: hasResult ? _buildLocationCard() : const SizedBox(),
                  ),
                  const SizedBox(height: 12),
                  AnimatedSwitcher(
                    duration: _animDuration,
                    child: hasResult ? _buildWeatherCard() : const SizedBox(),
                  ),
                  const SizedBox(height: 12),
                  AnimatedSwitcher(
                    duration: _animDuration,
                    child: hasResult ? _buildSoilCard() : const SizedBox(),
                  ),
                  const SizedBox(height: 12),
                  AnimatedSwitcher(
                    duration: _animDuration,
                    child: hasResult ? _buildRiskCard() : const SizedBox(),
                  ),
                  const SizedBox(height: 12),
                  Row(
                    children: [
                      Expanded(
                        child: FilledButton.icon(
                          onPressed: _isLoading
                              ? null
                              : () => _pickImage(ImageSource.camera),
                          icon: const Icon(Icons.photo_camera),
                          label: const Text('Камера'),
                        ),
                      ),
                      const SizedBox(width: 10),
                      Expanded(
                        child: OutlinedButton.icon(
                          onPressed: _isLoading
                              ? null
                              : () => _pickImage(ImageSource.gallery),
                          icon: const Icon(Icons.photo_library),
                          label: const Text('Галерея'),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 14),
                  SizedBox(
                    width: double.infinity,
                    child: FilledButton(
                      onPressed: _isLoading ? null : _analyze,
                      child: const Padding(
                        padding: EdgeInsets.symmetric(vertical: 14),
                        child: Text('Анализировать'),
                      ),
                    ),
                  ),
                  const SizedBox(height: 30),
                ],
              ),
            ),
          ),
          _buildLoadingOverlay(),
        ],
      ),
    );
  }
}

/// ============================
///      Экран истории
/// ============================
class HistoryPage extends StatelessWidget {
  final List<AnalysisResult> history;

  const HistoryPage({super.key, required this.history});

  Widget _buildEmptyHistory() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Lottie.asset(
            'assets/lottie/Tree in the wind.json',
            width: 220,
            repeat: true,
          ),
          const SizedBox(height: 12),
          const Text(
            'История пока пуста',
            style: TextStyle(fontSize: 18, fontWeight: FontWeight.w600),
          ),
          const SizedBox(height: 4),
          const Text(
            'Сделайте первое измерение,\nчтобы сохранить его здесь.',
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }

  String _formatDate(DateTime dt) {
    final d = dt.day.toString().padLeft(2, '0');
    final m = dt.month.toString().padLeft(2, '0');
    final y = dt.year.toString();
    final hh = dt.hour.toString().padLeft(2, '0');
    final mm = dt.minute.toString().padLeft(2, '0');
    return '$d.$m.$y $hh:$mm';
  }

  Color _riskColor(String? category) {
    switch (category) {
      case 'низкий':
        return Colors.green;
      case 'средний':
        return Colors.orange;
      case 'высокий':
        return Colors.red;
      default:
        return Colors.grey;
    }
  }

  @override
  Widget build(BuildContext context) {
    if (history.isEmpty) {
      return Scaffold(
        appBar: AppBar(title: const Text('История измерений')),
        body: _buildEmptyHistory(),
      );
    }

    return Scaffold(
      appBar: AppBar(title: const Text('История измерений')),
      body: ListView.separated(
        padding: const EdgeInsets.all(12),
        itemCount: history.length,
        separatorBuilder: (_, __) => const SizedBox(height: 10),
        itemBuilder: (context, index) {
          final item = history[index];
          final imgWidget = item.imageBase64.isNotEmpty
              ? Image.memory(
                  base64Decode(item.imageBase64),
                  fit: BoxFit.cover,
                )
              : null;
          final dateStr = _formatDate(item.timestamp);
          final riskCategory = item.riskCategory;

          return Card(
            shape:
                RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
            child: ListTile(
              leading: imgWidget != null
                  ? ClipRRect(
                      borderRadius: BorderRadius.circular(12),
                      child: SizedBox(width: 56, height: 56, child: imgWidget),
                    )
                  : const Icon(Icons.park),
              title: Text(item.species),
              subtitle: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    dateStr,
                    style: const TextStyle(fontSize: 12, color: Colors.grey),
                  ),
                  const SizedBox(height: 4),
                  Text('Высота: ${item.height ?? '-'} м'),
                  Text('Крона: ${item.crown ?? '-'} м'),
                ],
              ),
              trailing: riskCategory == null
                  ? null
                  : Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        CircleAvatar(
                          radius: 7,
                          backgroundColor: _riskColor(riskCategory),
                        ),
                        const SizedBox(height: 4),
                        Text(
                          riskCategory,
                          style: const TextStyle(fontSize: 10),
                        ),
                      ],
                    ),
            ),
          );
        },
      ),
    );
  }
}

