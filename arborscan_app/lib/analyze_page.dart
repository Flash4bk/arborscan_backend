import 'dart:convert';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'dart:ui'; 

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:lottie/lottie.dart';

import 'api_config.dart';
import 'admin_gate.dart';
import 'admin_panel_page.dart';
import 'feedback_page.dart';
import 'ar_measure_channel.dart';
import 'app_theme.dart';
import 'analysis_report_page.dart';
import 'location_service.dart';
import 'stick_page.dart'; 

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

class ArborScanPage extends StatefulWidget {
  const ArborScanPage({super.key});

  @override
  State<ArborScanPage> createState() => _ArborScanPageState();
}

class _ArborScanPageState extends State<ArborScanPage> with SingleTickerProviderStateMixin {
  final ImagePicker _picker = ImagePicker();

  File? _imageFile;
  ImageSource? _imageSource; // ТУТ СОХРАНЯЕМ ИСТОЧНИК ФОТО

  Uint8List? _annotatedImageBytes;
  Map<String, dynamic>? _result;

  bool _isLoading = false;
  String? _error;
  String? _gpsStatusText;
  bool _lastGpsOk = false;
  
  double? _lastArMeters;
  double? _arHeightM;
  double? _arCrownWidthM;
  double? _arTrunkDiameterM;
  
  double? _manualScale; 
  double _manualWindSpeedMS = 25.0; 

  double _aiConf = 0.15; 
  double _aiSmoothness = 5.0; 
  bool _aiUseRembg = false; 

  bool _isAdmin = false;
  String? _authToken;
  static const String _adminFlagKey = 'arborscan_is_admin';

  static String get _apiUrl => '${ApiConfig.baseUrl}/analyze-tree';
  static const String _historyKey = 'arborscan_history';
  static const String _authTokenKey = 'arborscan_auth_token';
  final List<AnalysisResult> _history = [];

  late AnimationController _pulseController;

  @override
  void initState() {
    super.initState();
    _loadHistory().then((_) => _syncServerHistory());
    _loadSessionState();

    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 2),
    )..repeat(reverse: true);
  }

  @override
  void dispose() {
    _pulseController.dispose();
    super.dispose();
  }

  Future<void> _loadHistory() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final list = prefs.getStringList(_historyKey);
      if (list == null) return;
      final loaded = <AnalysisResult>[];
      for (final e in list.take(30)) {
        try {
          final jsonMap = jsonDecode(e) as Map<String, dynamic>;
          jsonMap['imageBase64'] = '';
          loaded.add(AnalysisResult.fromJson(jsonMap));
        } catch (_) {}
      }
      if (mounted) setState(() => _history..clear()..addAll(loaded));
    } catch (_) {}
  }

  Future<void> _saveHistory() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final encoded = _history.take(30).map((e) => jsonEncode(e.toJson()..['imageBase64']='')).toList();
      await prefs.setStringList(_historyKey, encoded);
    } catch (_) {}
  }

  Future<void> _clearHistory() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_historyKey);
    setState(() => _history.clear());
  }

  Future<void> _syncServerHistory() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final token = prefs.getString(_authTokenKey) ?? '';
      if (token.isEmpty) return;
      final uri = Uri.parse('${ApiConfig.baseUrl}/analyses/my')
          .replace(queryParameters: {'limit': '100'});
      final res = await http.get(
        uri,
        headers: {'Authorization': 'Bearer $token'},
      ).timeout(const Duration(seconds: 12));
      if (res.statusCode != 200) return;
      final data = jsonDecode(utf8.decode(res.bodyBytes)) as Map<String, dynamic>;
      final items = (data['items'] as List? ?? const []);
      final serverHistory = <AnalysisResult>[];
      for (final raw in items) {
        if (raw is! Map) continue;
        final m = raw.cast<String, dynamic>();
        serverHistory.add(AnalysisResult(
          species: m['species']?.toString() ?? 'Неизвестно',
          height: (m['height_m'] as num?)?.toDouble(),
          crown: (m['crown_width_m'] as num?)?.toDouble(),
          trunk: (m['trunk_diameter_m'] as num?)?.toDouble(),
          scale: null,
          riskIndex: (m['risk_index'] as num?)?.toDouble(),
          riskCategory: m['risk_category']?.toString(),
          lat: (m['lat'] as num?)?.toDouble(),
          lon: (m['lon'] as num?)?.toDouble(),
          address: m['address']?.toString(),
          imageBase64: '',
          timestamp: m['created_at'] != null ? DateTime.parse(m['created_at'].replaceFirst('Z', '')) : DateTime.now(),
          analysisId: m['analysis_id']?.toString() ?? '',
        ));
      }
      if (!mounted || serverHistory.isEmpty) return;
      setState(() {
        final byId = <String, AnalysisResult>{for (final h in _history) if (h.analysisId.isNotEmpty) h.analysisId: h};
        for (final s in serverHistory) {
          if (s.analysisId.isNotEmpty) byId[s.analysisId] = s;
        }
        _history..clear()..addAll(byId.values.toList()..sort((a, b) => b.timestamp.compareTo(a.timestamp)));
      });
      await _saveHistory();
    } catch (_) {}
  }

  Future<void> _loadSessionState() async {
    final prefs = await SharedPreferences.getInstance();
    final isAdmin = prefs.getBool(_adminFlagKey) ?? false;
    final token = prefs.getString(_authTokenKey)?.trim();
    if (!mounted) return;
    setState(() {
      _isAdmin = isAdmin;
      _authToken = token == null || token.isEmpty ? null : token;
    });
  }

  void _openAiSettings() {
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      isScrollControlled: true,
      builder: (ctx) => StatefulBuilder(
        builder: (ctx, setModalState) => GlassPanel(
          padding: const EdgeInsets.all(24),
          radius: 30,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Center(child: Text("НАСТРОЙКИ ИИ", style: TextStyle(fontSize: 16, fontWeight: FontWeight.w900, color: AppTheme.primary, letterSpacing: 2))),
              const SizedBox(height: 24),

              Text("ЧУВСТВИТЕЛЬНОСТЬ: ${(_aiConf * 100).toInt()}%", style: const TextStyle(fontWeight: FontWeight.w900, color: AppTheme.primary2, fontSize: 12, letterSpacing: 1.0)),
              const SizedBox(height: 4),
              const Text("Меньше = цепляется за голые ветки. Больше = игнорирует фон.", style: TextStyle(fontSize: 11, color: AppTheme.muted, height: 1.2)),
              SliderTheme(
                data: SliderThemeData(activeTrackColor: AppTheme.primary2, thumbColor: AppTheme.primary2, inactiveTrackColor: AppTheme.surface3),
                child: Slider(
                  value: _aiConf, min: 0.05, max: 0.95,
                  onChanged: (v) { setModalState(() => _aiConf = v); setState(() => _aiConf = v); }
                ),
              ),
              const SizedBox(height: 16),

              Text("СГЛАЖИВАНИЕ МАСКИ: ${_aiSmoothness.toInt()}", style: const TextStyle(fontWeight: FontWeight.w900, color: AppTheme.warning, fontSize: 12, letterSpacing: 1.0)),
              const SizedBox(height: 4),
              const Text("Заливка 'дырок' в кроне. 1 = пиксельно точно, 15 = монолитная шапка.", style: TextStyle(fontSize: 11, color: AppTheme.muted, height: 1.2)),
              SliderTheme(
                data: SliderThemeData(activeTrackColor: AppTheme.warning, thumbColor: AppTheme.warning, inactiveTrackColor: AppTheme.surface3),
                child: Slider(
                  value: _aiSmoothness, min: 1, max: 15, divisions: 14,
                  onChanged: (v) { setModalState(() => _aiSmoothness = v); setState(() => _aiSmoothness = v); }
                ),
              ),
              const SizedBox(height: 16),

              SwitchListTile(
                contentPadding: EdgeInsets.zero,
                activeColor: AppTheme.primary,
                title: const Text("ГЛУБОКАЯ ОЧИСТКА ФОНА", style: TextStyle(fontWeight: FontWeight.w900, color: AppTheme.primary, fontSize: 12, letterSpacing: 1.0)),
                subtitle: const Text("Использовать U-2-Net для вырезания зданий и неба.", style: TextStyle(fontSize: 11, color: AppTheme.muted, height: 1.2)),
                value: _aiUseRembg,
                onChanged: (v) { setModalState(() => _aiUseRembg = v); setState(() => _aiUseRembg = v); }
              ),
              
              const SizedBox(height: 24),
              SizedBox(
                width: double.infinity,
                child: FilledButton(
                  onPressed: () => Navigator.pop(ctx),
                  style: FilledButton.styleFrom(backgroundColor: AppTheme.primary, padding: const EdgeInsets.symmetric(vertical: 16)),
                  child: const Text("СОХРАНИТЬ", style: TextStyle(color: Colors.black, fontWeight: FontWeight.w900, letterSpacing: 1.5)),
                )
              )
            ],
          )
        )
      )
    );
  }

  void _pickImageSource() {
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      builder: (ctx) => GlassPanel(
        padding: const EdgeInsets.all(24),
        radius: 30,
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Text("ИСТОЧНИК ФОТО", style: TextStyle(fontSize: 16, fontWeight: FontWeight.w900, color: AppTheme.primary, letterSpacing: 2)),
            const SizedBox(height: 24),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                _ModalBtn(icon: Icons.camera_alt_outlined, label: "КАМЕРА", onTap: () { Navigator.pop(ctx); _pickImage(ImageSource.camera); }),
                _ModalBtn(icon: Icons.photo_library_outlined, label: "ГАЛЕРЕЯ", onTap: () { Navigator.pop(ctx); _pickImage(ImageSource.gallery); }),
              ],
            ),
            const SizedBox(height: 24),
          ],
        ),
      ),
    );
  }

  Future<void> _pickImage(ImageSource source) async {
    try {
      final picked = await _picker.pickImage(source: source, imageQuality: 72, maxWidth: 1600, maxHeight: 1600);
      if (picked == null) return;
      setState(() {
        _imageFile = File(picked.path);
        _imageSource = source; // Сохраняем источник фото
        _annotatedImageBytes = null;
        _result = null;
        _error = null;
        _arHeightM = null;
        _arCrownWidthM = null;
        _arTrunkDiameterM = null;
        _manualScale = null;
      });
    } catch (e) {
      setState(() => _error = 'Ошибка при выборе изображения: $e');
    }
  }

  Future<void> _openArMeasure() async {
    try {
      final result = await ArMeasureChannel.openArMeasure();
      if (!mounted || result == null) return;
      setState(() {
        _arHeightM = result.heightMeters ?? result.distanceMeters;
        _arCrownWidthM = result.crownWidthMeters;
        _arTrunkDiameterM = result.trunkDiameterMeters;
        _manualScale = null;
      });
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Row(
            children: const [Icon(Icons.check_circle, color: AppTheme.primary), SizedBox(width: 10), Expanded(child: Text('AR-измерения сохранены'))],
          ),
          backgroundColor: AppTheme.surface2,
        ),
      );
    } catch (e) {
      if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('AR ошибка: $e')));
    }
  }

  void _onAnalyzeTap() {
    if (_imageFile == null) return;
    if (_arHeightM != null || _arCrownWidthM != null || _arTrunkDiameterM != null) {
      _analyze();
    } else {
      _showScaleOptionsModal();
    }
  }

  void _showScaleOptionsModal() {
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      isScrollControlled: true,
      builder: (ctx) => GlassPanel(
        padding: const EdgeInsets.all(24),
        radius: 30,
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Text("МАСШТАБ ФОТО", style: TextStyle(fontSize: 16, fontWeight: FontWeight.w900, color: AppTheme.primary, letterSpacing: 2)),
            const SizedBox(height: 12),
            const Text(
              "На фото нет AR-замеров. Как вычислим реальные размеры дерева?",
              textAlign: TextAlign.center,
              style: TextStyle(color: AppTheme.muted, fontSize: 13, height: 1.3),
            ),
            const SizedBox(height: 24),
            _ScaleOptionBtn(
              icon: Icons.auto_awesome,
              title: "Автоматически (ИИ)",
              subtitle: "Нейросеть подберет средние размеры по породе",
              onTap: () {
                Navigator.pop(ctx);
                _analyze();
              }
            ),
            const SizedBox(height: 12),
            _ScaleOptionBtn(
              icon: Icons.straighten,
              title: "Знаю один размер",
              subtitle: "Например, толщину ствола или высоту",
              onTap: () {
                Navigator.pop(ctx);
                _showSingleSizeInputDialog();
              }
            ),
            const SizedBox(height: 12),
            _ScaleOptionBtn(
              icon: Icons.design_services,
              title: "По объекту в кадре",
              subtitle: "Провести 1-метровую линию (человек, лопата и т.д.)",
              onTap: () async {
                Navigator.pop(ctx);
                final bytes = await _imageFile!.readAsBytes();
                final b64 = base64Encode(bytes);
                if (!mounted) return;
                
                final scale = await Navigator.push<double>(
                  context,
                  MaterialPageRoute(builder: (_) => StickPage(originalImageBase64: b64, currentScalePxToM: 0.0)),
                );
                
                if (scale != null) {
                  setState(() => _manualScale = scale);
                  _analyze();
                }
              }
            ),
            const SizedBox(height: 24),
          ],
        ),
      ),
    );
  }

  void _showSingleSizeInputDialog() {
    String selectedType = 'trunk'; 
    final ctrl = TextEditingController();

    showDialog(
      context: context,
      builder: (ctx) => StatefulBuilder(
        builder: (ctx, setDialogState) {
          return Dialog(
            backgroundColor: Colors.transparent,
            child: GlassPanel(
              radius: 24,
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Text("ИЗВЕСТНЫЙ РАЗМЕР", style: TextStyle(fontSize: 16, fontWeight: FontWeight.w900, color: AppTheme.primary, letterSpacing: 2)),
                  const SizedBox(height: 20),
                  DropdownButtonFormField<String>(
                    value: selectedType,
                    dropdownColor: AppTheme.surface2,
                    decoration: const InputDecoration(labelText: 'Что вы можете оценить?'),
                    items: const [
                      DropdownMenuItem(value: 'trunk', child: Text('Толщина ствола (м)')),
                      DropdownMenuItem(value: 'height', child: Text('Высота дерева (м)')),
                      DropdownMenuItem(value: 'crown', child: Text('Ширина кроны (м)')),
                    ],
                    onChanged: (v) => setDialogState(() => selectedType = v!),
                  ),
                  const SizedBox(height: 16),
                  TextField(
                    controller: ctrl,
                    keyboardType: const TextInputType.numberWithOptions(decimal: true),
                    decoration: const InputDecoration(
                      labelText: 'Значение в метрах',
                      hintText: 'Например, 0.4',
                    ),
                  ),
                  const SizedBox(height: 24),
                  SizedBox(
                    width: double.infinity,
                    child: FilledButton(
                      style: FilledButton.styleFrom(
                        backgroundColor: AppTheme.primary,
                        padding: const EdgeInsets.symmetric(vertical: 16),
                      ),
                      onPressed: () {
                        final val = double.tryParse(ctrl.text.replaceAll(',', '.'));
                        if (val != null && val > 0) {
                          setState(() {
                            if (selectedType == 'trunk') _arTrunkDiameterM = val;
                            if (selectedType == 'height') _arHeightM = val;
                            if (selectedType == 'crown') _arCrownWidthM = val;
                          });
                          Navigator.pop(ctx);
                          _analyze();
                        }
                      },
                      child: const Text('ПРОДОЛЖИТЬ', style: TextStyle(color: Colors.black, fontWeight: FontWeight.w900, letterSpacing: 1.0)),
                    )
                  )
                ]
              )
            )
          );
        }
      )
    );
  }

  Future<void> _analyze() async {
    if (_imageFile == null) return;
    await _loadSessionState();
    if (!mounted) return;
    setState(() { _isLoading = true; _error = null; });

    try {
      final uri = Uri.parse(_apiUrl);
      final request = http.MultipartRequest('POST', uri);
      
      // ИСПРАВЛЕНИЕ: Берем GPS телефона ТОЛЬКО если фото сделано на камеру
      if (_imageSource == ImageSource.camera) {
        final locationResult = await LocationService.getCurrentPositionDetailed();
        final pos = locationResult.position;
        if (mounted) setState(() { _lastGpsOk = pos != null; _gpsStatusText = locationResult.message; });
        if (pos != null) {
          request.fields['lat'] = pos.latitude.toString();
          request.fields['lon'] = pos.longitude.toString();
        }
      } else {
        if (mounted) setState(() { _lastGpsOk = false; _gpsStatusText = "Фото из галереи (используются EXIF-данные)"; });
      }

      final token = _authToken;
      if (token != null && token.isNotEmpty) {
        request.fields['auth_token'] = token;
      }

      if (_arHeightM != null) request.fields['ar_height_m'] = _arHeightM!.toStringAsFixed(3);
      if (_arCrownWidthM != null) request.fields['ar_crown_width_m'] = _arCrownWidthM!.toStringAsFixed(3);
      if (_arTrunkDiameterM != null) request.fields['ar_trunk_diameter_m'] = _arTrunkDiameterM!.toStringAsFixed(3);
      if (_manualScale != null) request.fields['manual_scale'] = _manualScale!.toStringAsFixed(6);

      request.fields['manual_wind_speed_m_s'] = _manualWindSpeedMS.toStringAsFixed(3);
      request.fields['ai_conf'] = _aiConf.toStringAsFixed(2);
      request.fields['ai_smoothness'] = _aiSmoothness.toInt().toString();
      request.fields['ai_use_rembg'] = _aiUseRembg.toString();

      request.files.add(await http.MultipartFile.fromPath('file', _imageFile!.path));

      final streamed = await request.send();
      final response = await http.Response.fromStream(streamed);

      if (response.statusCode < 200 || response.statusCode >= 300) {
        String message = 'Ошибка сервера: ${response.statusCode}';
        try {
          final decoded = jsonDecode(utf8.decode(response.bodyBytes));
          if (decoded is Map) {
            message = (decoded['detail'] ??
                    decoded['error'] ??
                    decoded['message'] ??
                    message)
                .toString();
          }
        } catch (_) {}
        throw Exception(message);
      }

      final data = jsonDecode(utf8.decode(response.bodyBytes))
          as Map<String, dynamic>;
      final annotatedB64 = data['annotated_image_base64'] as String?;
      
      setState(() {
        if (annotatedB64 != null) _annotatedImageBytes = base64Decode(annotatedB64);
        _result = data;
        _history.insert(0, AnalysisResult(
          species: data['species'] ?? 'Неизвестно',
          height: (data['height_m'] as num?)?.toDouble(),
          crown: (data['crown_width_m'] as num?)?.toDouble(),
          trunk: (data['trunk_diameter_m'] as num?)?.toDouble(),
          scale: (data['scale_px_to_m'] as num?)?.toDouble(),
          riskIndex: ((data['risk'] ?? {})['index'] as num?)?.toDouble(),
          riskCategory: (data['risk'] ?? {})['category'],
          lat: data['gps']?['lat'], // Берем GPS с бэкенда (так как он может вытянуть EXIF)
          lon: data['gps']?['lon'],
          address: data['address']?.toString(),
          imageBase64: '',
          timestamp: DateTime.now(),
          analysisId: data['analysis_id'] ?? '',
        ));
      });
      await _saveHistory();
    } catch (e) {
      setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  Future<void> _applyFeedbackResult(Map<String, dynamic> feedback) async {
    final corrected = feedback['corrected'];
    if (corrected is! Map || _result == null) return;

    final updated = Map<String, dynamic>.from(_result!);
    updated['species'] = corrected['species'] ?? updated['species'];
    updated['height_m'] = corrected['height_m'] ?? updated['height_m'];
    updated['crown_width_m'] =
        corrected['crown_width_m'] ?? updated['crown_width_m'];
    updated['trunk_diameter_m'] =
        corrected['trunk_diameter_m'] ?? updated['trunk_diameter_m'];
    updated['scale_px_to_m'] =
        corrected['scale_px_to_m'] ?? updated['scale_px_to_m'];
    updated['feedback_status'] = feedback['status'];
    updated['feedback_submitted'] = true;
    updated['feedback_verified'] = feedback['verified'] == true;

    final analysisId = updated['analysis_id']?.toString() ?? '';
    final historyIndex = _history.indexWhere(
      (item) => item.analysisId == analysisId,
    );

    if (!mounted) return;
    setState(() {
      _result = updated;
      if (historyIndex >= 0) {
        final old = _history[historyIndex];
        _history[historyIndex] = AnalysisResult(
          species: updated['species']?.toString() ?? old.species,
          height: (updated['height_m'] as num?)?.toDouble(),
          crown: (updated['crown_width_m'] as num?)?.toDouble(),
          trunk: (updated['trunk_diameter_m'] as num?)?.toDouble(),
          scale: (updated['scale_px_to_m'] as num?)?.toDouble(),
          riskIndex: old.riskIndex,
          riskCategory: old.riskCategory,
          lat: old.lat,
          lon: old.lon,
          address: old.address,
          imageBase64: old.imageBase64,
          timestamp: old.timestamp,
          analysisId: old.analysisId,
        );
      }
    });
    await _saveHistory();
  }

  Future<void> _openFeedback() async {
    final data = _result;
    if (data == null) return;

    final analysisId = data['analysis_id']?.toString() ?? '';
    if (analysisId.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('analysis_id отсутствует.')),
      );
      return;
    }

    String originalB64 = data['original_image_base64']?.toString() ?? '';
    if (originalB64.isEmpty && _imageFile != null) {
      originalB64 = base64Encode(await _imageFile!.readAsBytes());
    }
    if (originalB64.isEmpty) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Изображение для проверки недоступно.')),
      );
      return;
    }

    await _loadSessionState();
    if (!mounted) return;

    final feedback = await Navigator.of(context).push<Map<String, dynamic>>(
      MaterialPageRoute(
        builder: (_) => FeedbackPage(
          baseUrl: ApiConfig.baseUrl,
          authToken: _authToken,
          analysisId: analysisId,
          originalImageBase64: originalB64,
          annotatedImageBase64:
              data['annotated_image_base64']?.toString(),
          maskImageBase64: data['mask_image_base64']?.toString(),
          species: data['species']?.toString() ?? 'Неизвестно',
          heightM: (data['height_m'] as num?)?.toDouble(),
          crownWidthM: (data['crown_width_m'] as num?)?.toDouble(),
          trunkDiameterM:
              (data['trunk_diameter_m'] as num?)?.toDouble(),
          scalePxToM: (data['scale_px_to_m'] as num?)?.toDouble(),
        ),
      ),
    );

    if (feedback != null && feedback['submitted'] == true) {
      await _applyFeedbackResult(feedback);
    }
  }

  Future<void> _openReport() async {
    final data = _result;
    if (data == null) return;
    await Navigator.of(context).push<void>(
      MaterialPageRoute(
        builder: (_) => AnalysisReportPageV2.fromRawResult(
          raw: data,
          annotatedImageBytes: _annotatedImageBytes,
          onOpenFeedback: _openFeedback,
        ),
      ),
    );
  }

  Future<void> _openAdminPanel() async {
    await Navigator.of(context).push<void>(
      MaterialPageRoute(
        builder: (_) => const AdminPanelPage(baseUrl: ApiConfig.baseUrl),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.background,
      appBar: AppBar(
        title: const Text('ARBORSCAN'),
        actions: [
          IconButton(
            icon: const Icon(Icons.tune_rounded, color: AppTheme.primary2),
            tooltip: 'Настройки ИИ',
            onPressed: _openAiSettings,
          ),
        ],
      ),
      body: Stack(
        children: [
          Positioned(
            top: 200,
            right: -100,
            child: Container(
              width: 250, height: 250,
              decoration: BoxDecoration(shape: BoxShape.circle, color: AppTheme.primary.withOpacity(0.08)),
              child: BackdropFilter(filter: ImageFilter.blur(sigmaX: 100, sigmaY: 100), child: const SizedBox()),
            ),
          ),
          
          SafeArea(
            child: SingleChildScrollView(
              padding: const EdgeInsets.fromLTRB(16, 16, 16, 120), 
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _buildMainGlassButton(),
                  const SizedBox(height: 24),
                  
                  if (_isAdmin) ...[
                    AdminGate(
                      isAdmin: true,
                      onOpenFeedback: () { _openFeedback(); },
                      onOpenAdminPanel: _openAdminPanel,
                    ),
                    const SizedBox(height: 24),
                  ],

                  _buildWindSlidersCard(),
                  const SizedBox(height: 24),

                  _buildArMeasurementsCard(),
                  const SizedBox(height: 24),

                  if (_result != null) _buildResultCard(),

                  if (_error != null) ...[
                    const SizedBox(height: 16),
                    GlassPanel(
                      color: AppTheme.danger.withOpacity(0.1),
                      border: Border.all(color: AppTheme.danger),
                      child: Text(_error!, style: const TextStyle(color: AppTheme.danger)),
                    ),
                  ],

                  if (_imageFile != null && !_isLoading) ...[
                    const SizedBox(height: 24),
                    SizedBox(
                      width: double.infinity,
                      child: FilledButton.icon(
                        onPressed: _onAnalyzeTap,
                        icon: const Icon(Icons.analytics_outlined, color: Colors.black),
                        label: const Text(
                          "АНАЛИЗИРОВАТЬ", 
                          style: TextStyle(color: Colors.black, fontWeight: FontWeight.w900, letterSpacing: 1.5, fontSize: 16)
                        ),
                        style: FilledButton.styleFrom(
                          backgroundColor: AppTheme.primary,
                          padding: const EdgeInsets.symmetric(vertical: 20),
                          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
                          elevation: 10,
                          shadowColor: AppTheme.primary.withOpacity(0.5)
                        ),
                      ),
                    ),
                  ]
                ],
              ),
            ),
          ),

          if (_isLoading)
            Positioned.fill(
              child: GlassPanel(
                color: Colors.black54,
                blur: 10,
                child: Center(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Lottie.asset('assets/lottie/tree.json', height: 150),
                      const Text('АНАЛИЗ ИИ...', style: TextStyle(color: AppTheme.primary, fontWeight: FontWeight.w900, letterSpacing: 2)),
                    ],
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildMainGlassButton() {
    return GestureDetector(
      onTap: _pickImageSource,
      child: GlassPanel(
        height: 320, 
        width: double.infinity,
        color: AppTheme.surface.withOpacity(0.4),
        padding: EdgeInsets.zero,
        child: _imageFile == null
            ? Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  AnimatedBuilder(
                    animation: _pulseController,
                    builder: (ctx, child) {
                      return Transform.scale(
                        scale: 1.0 + (_pulseController.value * 0.05),
                        child: Container(
                          padding: const EdgeInsets.all(20),
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            color: AppTheme.primary.withOpacity(0.1),
                            boxShadow: [BoxShadow(color: AppTheme.primary.withOpacity(0.3 * _pulseController.value), blurRadius: 30)],
                          ),
                          child: const Icon(Icons.document_scanner_outlined, size: 48, color: AppTheme.primary),
                        ),
                      );
                    }
                  ),
                  const SizedBox(height: 20),
                  const Text(
                    "НАЖМИТЕ ИЛИ ПЕРЕТАЩИТЕ ФОТО",
                    style: TextStyle(
                      color: AppTheme.primary2,
                      fontWeight: FontWeight.w900,
                      letterSpacing: 1.5,
                      shadows: [Shadow(color: AppTheme.primary2, blurRadius: 8)],
                    ),
                  ),
                ],
              )
            : Stack(
                fit: StackFit.expand,
                children: [
                  ClipRRect(
                    borderRadius: BorderRadius.circular(24),
                    child: _annotatedImageBytes != null
                        ? Image.memory(_annotatedImageBytes!, fit: BoxFit.cover)
                        : Image.file(_imageFile!, fit: BoxFit.cover),
                  ),
                  Positioned(
                    top: 10, right: 10,
                    child: IconButton(
                      icon: const Icon(Icons.refresh, color: Colors.white, shadows: [Shadow(color: Colors.black, blurRadius: 4)]),
                      onPressed: _pickImageSource,
                    ),
                  )
                ],
              ),
      ),
    );
  }

  Widget _buildWindSlidersCard() {
    return GlassPanel(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: const [
              Icon(Icons.storm, color: AppTheme.primary2),
              SizedBox(width: 8),
              Expanded(
                child: Text(
                  "СИМУЛЯТОР ШТОРМА (SIA)", 
                  style: TextStyle(color: AppTheme.text, fontWeight: FontWeight.w900, letterSpacing: 1.0)
                )
              ),
            ],
          ),
          const SizedBox(height: 12),
          const Text(
            "Проверка на прочность. По умолчанию риск рассчитывается для ураганного ветра (25 м/с). Вы можете изменить силу шторма для краш-теста.",
            style: TextStyle(color: AppTheme.muted, fontSize: 12, height: 1.3),
          ),
          const SizedBox(height: 20),
          _buildSingleHorizontalSlider('СКОРОСТЬ ВЕТРА', _manualWindSpeedMS, 35.0, (v) => setState(() => _manualWindSpeedMS = v)),
        ],
      ),
    );
  }

  Widget _buildSingleHorizontalSlider(String label, double value, double maxVal, ValueChanged<double> onChanged) {
    Color color = _getWindColor(value);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(label, style: const TextStyle(fontWeight: FontWeight.w800, color: AppTheme.muted, fontSize: 11, letterSpacing: 1.0)),
            Text("${value.toStringAsFixed(1)} м/с", style: TextStyle(fontWeight: FontWeight.w900, color: color, fontSize: 16, shadows: [Shadow(color: color, blurRadius: 8)])),
          ],
        ),
        SliderTheme(
          data: SliderThemeData(
            trackHeight: 6,
            activeTrackColor: color.withOpacity(0.8),
            inactiveTrackColor: AppTheme.surface3,
            thumbColor: color,
            overlayColor: color.withOpacity(0.2),
            thumbShape: const RoundSliderThumbShape(enabledThumbRadius: 10),
            overlayShape: const RoundSliderOverlayShape(overlayRadius: 20),
          ),
          child: Slider(
            value: value,
            max: maxVal,
            onChanged: onChanged,
          ),
        ),
      ],
    );
  }

  Color _getWindColor(double speed) {
    if (speed < 5) return AppTheme.primary2; 
    if (speed < 12) return AppTheme.primary; 
    if (speed < 20) return AppTheme.warning; 
    return AppTheme.danger; 
  }

  Widget _buildArMeasurementsCard() {
    final hasAny = _arHeightM != null || _arCrownWidthM != null || _arTrunkDiameterM != null || _manualScale != null;
    return GlassPanel(
      border: Border.all(color: hasAny ? AppTheme.primary : AppTheme.primary.withOpacity(0.2), width: hasAny ? 2 : 1),
      boxShadow: hasAny ? [BoxShadow(color: AppTheme.primary.withOpacity(0.2), blurRadius: 20)] : null,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.view_in_ar, color: AppTheme.primary, size: 28),
              const SizedBox(width: 12),
              const Expanded(child: Text('AR-ИЗМЕРЕНИЯ', style: TextStyle(fontWeight: FontWeight.w900, color: AppTheme.primary, letterSpacing: 1.5))),
              if (hasAny)
                IconButton(icon: const Icon(Icons.refresh, color: AppTheme.muted), onPressed: () => setState((){ _arHeightM=null; _arCrownWidthM=null; _arTrunkDiameterM=null; _manualScale=null; }))
            ],
          ),
          const SizedBox(height: 16),
          Row(
            children: [
              _ArStat('ВЫСОТА', _arHeightM),
              _ArStat('КРОНА', _arCrownWidthM),
              _ArStat('СТВОЛ', _arTrunkDiameterM),
            ],
          ),
          const SizedBox(height: 16),
          SizedBox(
            width: double.infinity,
            child: OutlinedButton.icon(
              onPressed: _openArMeasure,
              icon: const Icon(Icons.camera_rounded, color: AppTheme.primary),
              label: const Text('ЗАПУСТИТЬ AR', style: TextStyle(color: AppTheme.text, fontWeight: FontWeight.w900, letterSpacing: 1.0)),
              style: OutlinedButton.styleFrom(
                side: const BorderSide(color: AppTheme.primary, width: 1.5),
                padding: const EdgeInsets.symmetric(vertical: 16),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildResultCard() {
    final cat = _result!['risk']?['category'] ?? 'неизвестно';
    final isHigh = cat == 'высокий';
    return GlassPanel(
      color: isHigh ? AppTheme.danger.withOpacity(0.1) : AppTheme.surface2,
      border: Border.all(color: isHigh ? AppTheme.danger : AppTheme.primary, width: 2),
      boxShadow: [BoxShadow(color: isHigh ? AppTheme.danger.withOpacity(0.2) : AppTheme.primary.withOpacity(0.1), blurRadius: 20)],
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text("РЕЗУЛЬТАТ АНАЛИЗА", style: TextStyle(color: isHigh ? AppTheme.danger : AppTheme.primary, fontWeight: FontWeight.w900, letterSpacing: 1.5)),
          const SizedBox(height: 16),
          Text(_result!['species'] ?? 'Неизвестно', style: const TextStyle(fontSize: 24, fontWeight: FontWeight.w900, color: Colors.white)),
          const SizedBox(height: 8),
          Ui.badge(text: "РИСК: ${cat.toUpperCase()}", color: isHigh ? AppTheme.danger : AppTheme.primary2),
          const SizedBox(height: 16),
          SizedBox(
            width: double.infinity,
            child: OutlinedButton.icon(
              onPressed: _openFeedback,
              icon: Icon(
                _result!['feedback_submitted'] == true
                    ? Icons.verified_outlined
                    : Icons.fact_check_outlined,
                color: isHigh ? AppTheme.danger : AppTheme.primary,
              ),
              label: Text(
                _result!['feedback_submitted'] == true
                    ? 'ПРОВЕРЕНО'
                    : 'ПРОВЕРИТЬ И ИСПРАВИТЬ',
                style: const TextStyle(
                  color: AppTheme.text,
                  fontWeight: FontWeight.w900,
                  letterSpacing: 0.8,
                ),
              ),
              style: OutlinedButton.styleFrom(
                side: BorderSide(
                  color: isHigh ? AppTheme.danger : AppTheme.primary,
                ),
                padding: const EdgeInsets.symmetric(vertical: 14),
              ),
            ),
          ),
          const SizedBox(height: 10),
          SizedBox(
            width: double.infinity,
            child: FilledButton(
              onPressed: _openReport,
              style: FilledButton.styleFrom(
                backgroundColor: isHigh ? AppTheme.danger : AppTheme.primary,
                padding: const EdgeInsets.symmetric(vertical: 14),
              ),
              child: const Text(
                'ПОЛНЫЙ ОТЧЕТ',
                textAlign: TextAlign.center,
                style: TextStyle(
                  color: Colors.black,
                  fontWeight: FontWeight.w900,
                  letterSpacing: 0.8,
                ),
              ),
            ),
          )
        ],
      ),
    );
  }
}

class _ModalBtn extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback onTap;
  const _ModalBtn({required this.icon, required this.label, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Column(
        children: [
          Container(
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(shape: BoxShape.circle, color: AppTheme.surface3, border: Border.all(color: AppTheme.primary.withOpacity(0.5))),
            child: Icon(icon, size: 32, color: AppTheme.primary),
          ),
          const SizedBox(height: 8),
          Text(label, style: const TextStyle(fontWeight: FontWeight.w900, letterSpacing: 1.0)),
        ],
      ),
    );
  }
}

class _ArStat extends StatelessWidget {
  final String label;
  final double? val;
  const _ArStat(this.label, this.val);

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: Column(
        children: [
          Text(label, style: const TextStyle(color: AppTheme.muted, fontSize: 10, fontWeight: FontWeight.w900, letterSpacing: 1.0)),
          const SizedBox(height: 4),
          Text(val == null ? "—" : "${val!.toStringAsFixed(1)} м", style: const TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.w900)),
        ],
      ),
    );
  }
}

class _ScaleOptionBtn extends StatelessWidget {
  final IconData icon;
  final String title;
  final String subtitle;
  final VoidCallback onTap;

  const _ScaleOptionBtn({required this.icon, required this.title, required this.subtitle, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: AppTheme.surface3.withOpacity(0.5),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: AppTheme.primary.withOpacity(0.3)),
        ),
        child: Row(
          children: [
            Icon(icon, size: 32, color: AppTheme.primary),
            const SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(title, style: const TextStyle(fontWeight: FontWeight.w900, color: AppTheme.text, fontSize: 14)),
                  const SizedBox(height: 4),
                  Text(subtitle, style: const TextStyle(color: AppTheme.muted, fontSize: 11)),
                ],
              ),
            ),
            const Icon(Icons.chevron_right, color: AppTheme.muted),
          ],
        ),
      ),
    );
  }
}