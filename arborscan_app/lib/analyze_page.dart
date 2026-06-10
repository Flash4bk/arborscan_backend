import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:lottie/lottie.dart';
import 'package:geolocator/geolocator.dart';

import 'api_config.dart';
import 'admin_gate.dart';
import 'admin_panel_page.dart';
import 'feedback_page.dart';
import 'ar_measure_channel.dart';
import 'app_theme.dart';
import 'analysis_report_page.dart';
import 'location_service.dart';

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
  String? _gpsStatusText;
  bool _lastGpsOk = false;
  double? _lastArMeters;
  double? _arHeightM;
  double? _arCrownWidthM;
  double? _arTrunkDiameterM;
  double? _manualBetaKgS;
  double? _manualCrownStartHeightM;
  double? _manualCrownDensityFactor;
  double? _manualCrownShapeFactor;
  double? _manualWindSpeedMS;
  double? _manualWindGustMS;

  bool _isAdmin = false;
  static const String _adminFlagKey = 'arborscan_is_admin';

  static String get _apiUrl => '${ApiConfig.baseUrl}/analyze-tree';
  static String get _feedbackUrl => '${ApiConfig.baseUrl}/feedback';

  static const String _historyKey = 'arborscan_history';
  static const String _authTokenKey = 'arborscan_auth_token';
  final List<AnalysisResult> _history = [];

  @override
  void initState() {
    super.initState();
    _loadHistory().then((_) => _syncServerHistory());
    _loadAdminFlag();
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

      if (!mounted) return;
      setState(() {
        _history
          ..clear()
          ..addAll(loaded);
      });
    } catch (e) {
      debugPrint('History load skipped: $e');
    }
  }

  Future<void> _saveHistory() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final encoded = _history.take(30).map((e) {
        final map = e.toJson();
        map['imageBase64'] = '';
        return jsonEncode(map);
      }).toList();
      await prefs.setStringList(_historyKey, encoded);
    } catch (e) {
      debugPrint('History save skipped: $e');
    }
  }

  Future<void> _clearHistory() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_historyKey);
    setState(() {
      _history.clear();
    });
  }

  Future<void> _syncServerHistory() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final token = prefs.getString(_authTokenKey) ?? '';
      if (token.isEmpty) return;

      final uri = Uri.parse('${ApiConfig.baseUrl}/analyses/my').replace(
        queryParameters: {
          'token': token,
          'limit': '100',
        },
      );

      final res = await http.get(uri).timeout(const Duration(seconds: 12));
      if (res.statusCode != 200) return;

      final data = jsonDecode(utf8.decode(res.bodyBytes)) as Map<String, dynamic>;
      final items = (data['items'] as List? ?? const []);

      final serverHistory = <AnalysisResult>[];
      for (final raw in items) {
        if (raw is! Map) continue;
        final m = raw.cast<String, dynamic>();
        final createdRaw = m['created_at']?.toString();
        DateTime createdAt;
        try {
          createdAt = createdRaw != null
              ? DateTime.parse(createdRaw.replaceFirst('Z', ''))
              : DateTime.now();
        } catch (_) {
          createdAt = DateTime.now();
        }

        serverHistory.add(
          AnalysisResult(
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
            timestamp: createdAt,
            analysisId: m['analysis_id']?.toString() ?? '',
          ),
        );
      }

      if (!mounted || serverHistory.isEmpty) return;

      setState(() {
        final byId = <String, AnalysisResult>{
          for (final h in _history)
            if (h.analysisId.isNotEmpty) h.analysisId: h
        };
        for (final s in serverHistory) {
          if (s.analysisId.isNotEmpty) {
            byId[s.analysisId] = s;
          }
        }
        _history
          ..clear()
          ..addAll(byId.values.toList()
            ..sort((a, b) => b.timestamp.compareTo(a.timestamp)));
      });

      await _saveHistory();
    } catch (_) {
      // Серверная история не должна ломать основной анализ.
    }
  }

  Future<void> _loadAdminFlag() async {
    final prefs = await SharedPreferences.getInstance();
    final isAdmin = prefs.getBool(_adminFlagKey) ?? false;
    setState(() {
      _isAdmin = isAdmin;
    });
  }

  Future<void> _pickImage(ImageSource source) async {
    try {
      final picked = await _picker.pickImage(
        source: source,
        imageQuality: 72,
        maxWidth: 1600,
        maxHeight: 1600,
      );
      if (picked == null) return;

      setState(() {
        _imageFile = File(picked.path);
        _annotatedImageBytes = null;
        _result = null;
        _error = null;
        _lastArMeters = null;
        _arHeightM = null;
        _arCrownWidthM = null;
        _arTrunkDiameterM = null;
      });
    } catch (e) {
      setState(() {
        _error = 'Ошибка при выборе изображения: $e';
      });
    }
  }

  String _formatMeters(double? value) {
    if (value == null || value <= 0) return 'не измерено';
    return '${value.toStringAsFixed(2)} м';
  }

  Future<void> _openArMeasure() async {
    try {
      final result = await ArMeasureChannel.openArMeasure();
      if (!mounted) return;

      if (result == null) {
        // Пользователь просто закрыл AR без сохранения
        return;
      }

      setState(() {
        _lastArMeters = result.distanceMeters;
        _arHeightM = result.heightMeters ?? result.distanceMeters;
        _arCrownWidthM = result.crownWidthMeters;
        _arTrunkDiameterM = result.trunkDiameterMeters;
      });

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Row(
          children: const [
              Icon(Icons.check_circle, color: AppTheme.primary),
              SizedBox(width: 10),
              Expanded(child: Text('Точные AR-измерения успешно сохранены!')),
            ],
          ),
          behavior: SnackBarBehavior.floating,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
        ),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('AR ошибка: $e')),
      );
    }
  }

  Future<void> _analyze() async {
    if (_imageFile == null) return;

    setState(() {
      _isLoading = true;
      _error = null;
    });

    try {
      final locationResult = await LocationService.getCurrentPositionDetailed();
      final pos = locationResult.position;
      final double? deviceLat = pos?.latitude;
      final double? deviceLon = pos?.longitude;

      if (mounted) {
        setState(() {
          _lastGpsOk = pos != null;
          _gpsStatusText = locationResult.message;
        });
      }

      final uri = Uri.parse(_apiUrl);
      final request = http.MultipartRequest('POST', uri);

      final prefsForAuth = await SharedPreferences.getInstance();
      final authToken = prefsForAuth.getString(_authTokenKey) ?? '';
      if (authToken.isNotEmpty) {
        request.fields['auth_token'] = authToken;
      }

      if (deviceLat != null) request.fields['lat'] = deviceLat.toString();
      if (deviceLon != null) request.fields['lon'] = deviceLon.toString();

      if (_arHeightM != null) {
        request.fields['ar_height_m'] = _arHeightM!.toStringAsFixed(3);
      }
      if (_arCrownWidthM != null) {
        request.fields['ar_crown_width_m'] = _arCrownWidthM!.toStringAsFixed(3);
      }
      if (_arTrunkDiameterM != null) {
        request.fields['ar_trunk_diameter_m'] = _arTrunkDiameterM!.toStringAsFixed(3);
      }
      if (_manualBetaKgS != null && _manualBetaKgS! > 0) {
        request.fields['manual_beta_kg_s'] = _manualBetaKgS!.toStringAsFixed(3);
      }
      if (_manualCrownStartHeightM != null && _manualCrownStartHeightM! > 0) {
        request.fields['crown_start_height_m'] = _manualCrownStartHeightM!.toStringAsFixed(3);
      }
      if (_manualCrownDensityFactor != null && _manualCrownDensityFactor! > 0) {
        request.fields['crown_density_factor'] = _manualCrownDensityFactor!.toStringAsFixed(3);
      }
      if (_manualCrownShapeFactor != null && _manualCrownShapeFactor! > 0) {
        request.fields['crown_shape_factor'] = _manualCrownShapeFactor!.toStringAsFixed(3);
      }
      if (_manualWindSpeedMS != null && _manualWindSpeedMS! > 0) {
        request.fields['manual_wind_speed_m_s'] = _manualWindSpeedMS!.toStringAsFixed(3);
      }
      if (_manualWindGustMS != null && _manualWindGustMS! > 0) {
        request.fields['manual_wind_gust_m_s'] = _manualWindGustMS!.toStringAsFixed(3);
      }
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

      final String analysisId = data['analysis_id'] as String? ?? '';

      final historyItem = AnalysisResult(
        species: data['species'] as String? ?? 'Неизвестно',
        height: height,
        crown: crown,
        trunk: trunk,
        scale: scale,
        riskIndex: riskIndex,
        riskCategory: riskCategory,
        lat: ((gps?['lat'] as num?)?.toDouble()) ?? deviceLat,
        lon: ((gps?['lon'] as num?)?.toDouble()) ?? deviceLon,
        address: address,
        imageBase64: '',
        timestamp: DateTime.now(),
        analysisId: analysisId,
      );

      final resultForUi = Map<String, dynamic>.from(data);
      if (resultForUi['gps'] == null && deviceLat != null && deviceLon != null) {
        resultForUi['gps'] = {'lat': deviceLat, 'lon': deviceLon};
      }

      setState(() {
        _annotatedImageBytes = annotatedBytes;
        _result = resultForUi;
        _history.insert(0, historyItem);
      });

      await _saveHistory();
    } catch (e) {
      final rawError = e.toString();
      final friendlyError = rawError.contains('OutOfMemoryError') ||
              rawError.contains('Failed to allocate') ||
              rawError.contains('exceeds the limit')
          ? 'Не хватило памяти. Обычно это происходит из-за старой истории. Очистите данные приложения ArborScan.'
          : rawError;
      setState(() {
        _error = friendlyError;
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

  Widget _buildArMeasurementsCard() { 
    // Проверяем, есть ли хотя бы одно измерение
    final hasAny = _arHeightM != null || _arCrownWidthM != null || _arTrunkDiameterM != null;

    // Вспомогательный виджет для плитки с параметром (Высота/Крона/Ствол)
    Widget valueTile(String label, double? value, IconData icon) {
      final hasValue = value != null && value > 0;
      return Expanded(
        child: Container(
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: hasValue ? AppTheme.primary.withOpacity(0.12) : AppTheme.surface2,
            borderRadius: BorderRadius.circular(16),
            border: Border.all(
              color: hasValue ? AppTheme.primary.withOpacity(0.4) : AppTheme.border,
            ),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Icon(icon, size: 20, color: hasValue ? AppTheme.primary : AppTheme.muted),
              const SizedBox(height: 8),
              Text(
                label,
                style: TextStyle(
                  color: hasValue ? AppTheme.primary : AppTheme.muted,
                  fontSize: 12,
                  fontWeight: FontWeight.w700,
                ),
              ),
              const SizedBox(height: 3),
              Text(
                _formatMeters(value),
                style: TextStyle(
                  color: hasValue ? AppTheme.text : AppTheme.muted,
                  fontSize: 15,
                  fontWeight: FontWeight.w900,
                ),
              ),
            ],
          ),
        ),
      );
    }

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppTheme.surface,
        borderRadius: BorderRadius.circular(24),
        border: Border.all(
          // Если данные есть, подсвечиваем карточку зеленым контуром
          color: hasAny ? AppTheme.primary.withOpacity(0.4) : AppTheme.border,
          width: hasAny ? 1.5 : 1.0,
        ),
        boxShadow: hasAny
            ? [
                BoxShadow(
                  color: AppTheme.primary.withOpacity(0.08),
                  blurRadius: 20,
                  spreadRadius: 2,
                )
              ]
            : null,
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: AppTheme.primary.withOpacity(0.15),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: const Icon(Icons.view_in_ar, color: AppTheme.primary, size: 22),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Text(
                  'AR-Дальномер',
                  style: Theme.of(context).textTheme.titleMedium?.copyWith(
                        fontWeight: FontWeight.w900,
                        color: AppTheme.text,
                      ),
                ),
              ),
              if (hasAny)
                TextButton.icon(
                  onPressed: () {
                    setState(() {
                      _lastArMeters = null;
                      _arHeightM = null;
                      _arCrownWidthM = null;
                      _arTrunkDiameterM = null;
                    });
                  },
                  icon: const Icon(Icons.refresh, size: 16),
                  label: const Text('Сброс'),
                  style: TextButton.styleFrom(
                    foregroundColor: AppTheme.muted,
                    padding: const EdgeInsets.symmetric(horizontal: 8),
                    minimumSize: Size.zero,
                  ),
                ),
            ],
          ),
          const SizedBox(height: 12),
          const Text(
            'Умный замер с помощью AR и гироскопа. Наведите камеру на основание дерева, а высоту и крону телефон рассчитает по углу наклона.',
            style: TextStyle(
              color: AppTheme.muted,
              fontSize: 13,
              height: 1.3,
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: 16),
          Row(
            children: [
              valueTile('Высота', _arHeightM, Icons.height),
              const SizedBox(width: 10),
              valueTile('Крона', _arCrownWidthM, Icons.filter_hdr),
              const SizedBox(width: 10),
              valueTile('Ствол', _arTrunkDiameterM, Icons.circle_outlined),
            ],
          ),
          const SizedBox(height: 16),
          SizedBox(
            width: double.infinity,
            child: FilledButton.icon(
              onPressed: _openArMeasure,
              icon: const Icon(Icons.camera_rounded, color: Color(0xFF06140E)),
              label: const Text(
                'Запустить AR-дальномер',
                style: TextStyle(
                  color: Color(0xFF06140E),
                  fontWeight: FontWeight.w800,
                  fontSize: 15,
                ),
              ),
              style: FilledButton.styleFrom(
                backgroundColor: AppTheme.primary,
                padding: const EdgeInsets.symmetric(vertical: 16),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(16),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildBetaSettingsCard() {
    Widget numberField({
      required String label,
      required String hint,
      required String suffix,
      required IconData icon,
      required double? value,
      required ValueChanged<double?> onChanged,
    }) {
      return TextField(
        keyboardType: const TextInputType.numberWithOptions(decimal: true),
        decoration: InputDecoration(
          labelText: label,
          hintText: value == null ? hint : value.toStringAsFixed(2),
          prefixIcon: Icon(icon),
          suffixText: suffix,
        ),
        onChanged: (raw) {
          final normalized = raw.trim().replaceAll(',', '.');
          final parsed = double.tryParse(normalized);
          onChanged(parsed != null && parsed > 0 ? parsed : null);
        },
      );
    }

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: AppTheme.surface,
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: AppTheme.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.analytics_outlined, color: AppTheme.primary),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  'Аналитический центр β',
                  style: Theme.of(context).textTheme.titleMedium?.copyWith(
                        color: AppTheme.text,
                        fontWeight: FontWeight.w800,
                      ),
                ),
              ),
              TextButton.icon(
                onPressed: () => setState(() {
                  _manualBetaKgS = null;
                  _manualCrownStartHeightM = null;
                  _manualCrownDensityFactor = null;
                  _manualCrownShapeFactor = null;
                  _manualWindSpeedMS = null;
                  _manualWindGustMS = null;
                }),
                icon: const Icon(Icons.auto_fix_high, size: 18),
                label: const Text('авто'),
              ),
            ],
          ),
          const SizedBox(height: 6),
          const Text(
            'Сервер рассчитает β, ветровую силу, центр нагрузки и момент у основания.',
            style: TextStyle(
              color: AppTheme.muted,
              fontSize: 12,
              height: 1.25,
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: 12),
          numberField(
            label: 'β вручную',
            hint: 'автоматически',
            suffix: 'кг/с',
            icon: Icons.functions,
            value: _manualBetaKgS,
            onChanged: (v) => setState(() => _manualBetaKgS = v),
          ),
          const SizedBox(height: 10),
          numberField(
            label: 'Высота начала кроны',
            hint: 'авто: 0.55H',
            suffix: 'м',
            icon: Icons.forest_outlined,
            value: _manualCrownStartHeightM,
            onChanged: (v) => setState(() => _manualCrownStartHeightM = v),
          ),
          const SizedBox(height: 10),
          Row(
            children: [
              Expanded(
                child: numberField(
                  label: 'Плотность кроны',
                  hint: '1.0',
                  suffix: '×',
                  icon: Icons.grain,
                  value: _manualCrownDensityFactor,
                  onChanged: (v) => setState(() => _manualCrownDensityFactor = v),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: numberField(
                  label: 'Форма кроны',
                  hint: '1.0',
                  suffix: '×',
                  icon: Icons.filter_hdr,
                  value: _manualCrownShapeFactor,
                  onChanged: (v) => setState(() => _manualCrownShapeFactor = v),
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          Row(
            children: [
              Expanded(
                child: numberField(
                  label: 'Ветер',
                  hint: 'из GPS',
                  suffix: 'м/с',
                  icon: Icons.air,
                  value: _manualWindSpeedMS,
                  onChanged: (v) => setState(() => _manualWindSpeedMS = v),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: numberField(
                  label: 'Порыв',
                  hint: 'из GPS',
                  suffix: 'м/с',
                  icon: Icons.storm_outlined,
                  value: _manualWindGustMS,
                  onChanged: (v) => setState(() => _manualWindGustMS = v),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildGpsStatusCard() {
    final status = _gpsStatusText ??
        'GPS будет получен при запуске анализа. Если координаты появятся, запись попадёт на карту.';

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: AppTheme.surface,
        borderRadius: BorderRadius.circular(22),
        border: Border.all(
          color: _lastGpsOk ? AppTheme.primary.withOpacity(0.45) : AppTheme.border,
        ),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(
            _lastGpsOk ? Icons.location_on : Icons.location_searching,
            color: _lastGpsOk ? AppTheme.primary : AppTheme.muted,
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              status,
              style: const TextStyle(
                color: AppTheme.muted,
                fontSize: 12,
                height: 1.25,
                fontWeight: FontWeight.w700,
              ),
            ),
          ),
          IconButton(
            tooltip: 'Настройки геолокации',
            onPressed: () async {
              await LocationService.openLocationSettings();
            },
            icon: const Icon(Icons.settings_outlined),
          ),
        ],
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
              color: AppTheme.muted,
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
              const Icon(Icons.info_outline, color: AppTheme.muted),
              const SizedBox(width: 12),
              Expanded(
                child: Text(
                  'Результаты появятся после анализа.\n'
                  'Загрузите фото дерева и нажмите «Анализировать».',
                  style: theme.textTheme.bodyMedium?.copyWith(
                    color: AppTheme.muted,
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
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
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
                  color: AppTheme.muted,
                ),
              )
            else
              Text(
                'GPS-данные в фото не найдены.',
                style: theme.textTheme.bodySmall?.copyWith(
                  color: AppTheme.muted,
                ),
              ),
          ],
        ),
      ),
    );
  }

  void _openAnalysisReport() {
    if (_result == null) return;
    Navigator.of(context).push(
      MaterialPageRoute(
        builder: (_) => AnalysisReportPageV2.fromRawResult(
          raw: _result!,
          annotatedImageBytes: _annotatedImageBytes,
        ),
      ),
    );
  }

  Future<void> _sendFeedbackToServer(
    Map<String, dynamic> feedback,
    String analysisId,
  ) async {
    final body = {
      "corrected_height_m": feedback["height_m_corrected"],
      "corrected_crown_width_m": feedback["crown_width_m_corrected"],
      "corrected_trunk_diameter_m": feedback["trunk_diameter_m_corrected"],
      "corrected_scale_px_to_m": feedback["scale_px_to_m_corrected"],
      "verifier_role": _isAdmin ? "admin" : "user",
      "analysis_id": analysisId,
      "use_for_training": feedback["use_for_training"] ?? true,
      "tree_ok": feedback["tree_ok"],
      "stick_ok": feedback["stick_ok"],
      "params_ok": feedback["params_ok"],
      "species_ok": feedback["species_ok"],
      "correct_species": feedback["correct_species"],
      "user_mask_base64": feedback["user_mask_base64"],
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
            content: Text("Ошибка отправки: ${resp.statusCode}"),
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
    final analysisId = data['analysis_id'] as String?;
    if (analysisId == null || analysisId.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("analysis_id отсутствует")),
      );
      return;
    }

    final originalB64 = data['original_image_base64'] as String? ?? "";
    if (originalB64.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Оригинальное изображение недоступно")),
      );
      return;
    }

    final annotatedB64 = data['annotated_image_base64'] as String?;

    final feedback = await Navigator.push<Map<String, dynamic>?>(
      context,
      MaterialPageRoute(
        builder: (_) => FeedbackPage(
          analysisId: analysisId,
          originalImageBase64: originalB64,
          annotatedImageBase64: annotatedB64,
          species: data['species'] ?? 'Неизвестно',
          heightM: (data['height_m'] as num?)?.toDouble(),
          crownWidthM: (data['crown_width_m'] as num?)?.toDouble(),
          trunkDiameterM: (data['trunk_diameter_m'] as num?)?.toDouble(),
          scalePxToM: (data['scale_px_to_m'] as num?)?.toDouble(),
        ),
      ),
    );

    if (feedback != null) {
      await _sendFeedbackToServer(feedback, analysisId);
    }
  }

  Future<void> _openAdminPanel() async {
    await Navigator.of(context).push(
      MaterialPageRoute(
        builder: (_) => const AdminPanelPage(baseUrl: ApiConfig.baseUrl),
      ),
    );
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
                      color: AppTheme.muted,
                    ),
                  ),
                  const SizedBox(height: 12),

                  if (_isAdmin) ...[
                    AdminGate(
                      isAdmin: _isAdmin,
                      onOpenFeedback: _openFeedback,
                      onOpenAdminPanel: _openAdminPanel,
                    ),
                    const SizedBox(height: 16),
                  ],

                  _buildImageCard(),
                  const SizedBox(height: 16),

                  _buildResultCard(),
                  const SizedBox(height: 12),
                  _buildGpsStatusCard(),
                  if (_result != null) ...[
                    const SizedBox(height: 12),
                    SizedBox(
                      width: double.infinity,
                      child: OutlinedButton.icon(
                        onPressed: _openAnalysisReport,
                        icon: const Icon(Icons.description_outlined),
                        label: const Text('Открыть подробный отчёт'),
                        style: OutlinedButton.styleFrom(
                          padding: const EdgeInsets.symmetric(vertical: 14),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(999),
                          ),
                        ),
                      ),
                    ),
                  ],
                  const SizedBox(height: 12),

                  if (_isAdmin &&
                      _annotatedImageBytes != null &&
                      _result != null &&
                      _result?['analysis_id'] != null)
                    SizedBox(
                      width: double.infinity,
                      child: FilledButton.icon(
                        onPressed: _openFeedback,
                        icon: const Icon(Icons.check_circle_outline),
                        label: const Text('Подтвердить / исправить анализ'),
                        style: FilledButton.styleFrom(
                          padding: const EdgeInsets.symmetric(vertical: 16),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(999),
                          ),
                        ),
                      ),
                    ),

                  if (_isAdmin &&
                      _annotatedImageBytes != null &&
                      _result != null &&
                      _result?['analysis_id'] != null)
                    const SizedBox(height: 16),

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
                  _buildArMeasurementsCard(),
                  const SizedBox(height: 12),
                  _buildBetaSettingsCard(),
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
                                    _lastArMeters = null;
                                    _arHeightM = null;
                                    _arCrownWidthM = null;
                                    _arTrunkDiameterM = null;
                                  });
                                },
                      icon: const Icon(Icons.clear),
                      label: const Text('Очистить'),
                    ),
                  ),
                  const SizedBox(height: 8),

                  SizedBox(
                    width: double.infinity,
                    child: FilledButton.icon(
                      onPressed:
                          _imageFile == null || _isLoading ? null : _analyze,
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
                    color: AppTheme.mutedOnLight,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  value,
                  style: theme.textTheme.bodyMedium?.copyWith(
                    color: AppTheme.textOnLight,
                    fontWeight: FontWeight.w800,
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
                  color: AppTheme.muted,
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