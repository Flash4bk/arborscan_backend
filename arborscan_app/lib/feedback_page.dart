import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

import 'api_config.dart';
import 'app_theme.dart';
import 'mask_drawing_page.dart';
import 'stick_page.dart';

/// Проверка и корректировка одного анализа.
///
/// Экран самостоятельно выполняет единственный POST /feedback и возвращает
/// уже сохранённый backend-ответ. Это исключает двойную отправку данных.
class FeedbackPage extends StatefulWidget {
  final String baseUrl;
  final String? authToken;

  final String analysisId;
  final String originalImageBase64;
  final String? annotatedImageBase64;
  final String? maskImageBase64;

  final String species;
  final double? heightM;
  final double? crownWidthM;
  final double? trunkDiameterM;
  final double? scalePxToM;

  const FeedbackPage({
    super.key,
    this.baseUrl = ApiConfig.baseUrl,
    this.authToken,
    required this.analysisId,
    required this.originalImageBase64,
    this.annotatedImageBase64,
    this.maskImageBase64,
    required this.species,
    this.heightM,
    this.crownWidthM,
    this.trunkDiameterM,
    this.scalePxToM,
  });

  @override
  State<FeedbackPage> createState() => _FeedbackPageState();
}

class _FeedbackPageState extends State<FeedbackPage> {
  final _formKey = GlobalKey<FormState>();

  bool _treeOk = true;
  bool _stickOk = true;
  bool _useForTraining = true;
  bool _isSending = false;

  late final TextEditingController _speciesController;
  late final TextEditingController _heightController;
  late final TextEditingController _crownController;
  late final TextEditingController _trunkController;

  String? _userMaskBase64;
  double? _userScale;

  @override
  void initState() {
    super.initState();
    _speciesController = TextEditingController(text: widget.species);
    _heightController = TextEditingController(
      text: widget.heightM?.toStringAsFixed(2) ?? '',
    );
    _crownController = TextEditingController(
      text: widget.crownWidthM?.toStringAsFixed(2) ?? '',
    );
    _trunkController = TextEditingController(
      text: widget.trunkDiameterM?.toStringAsFixed(2) ?? '',
    );
    _userScale = widget.scalePxToM;
  }

  @override
  void dispose() {
    _speciesController.dispose();
    _heightController.dispose();
    _crownController.dispose();
    _trunkController.dispose();
    super.dispose();
  }

  String _stripDataUri(String value) {
    final text = value.trim();
    if (!text.startsWith('data:')) return text;
    final comma = text.indexOf(',');
    return comma >= 0 ? text.substring(comma + 1) : text;
  }

  Uint8List? _decodeImage(String? value) {
    if (value == null || value.trim().isEmpty) return null;
    try {
      return base64Decode(_stripDataUri(value));
    } catch (_) {
      return null;
    }
  }

  double? _parseNumber(String value) {
    final normalized = value.trim().replaceAll(',', '.');
    if (normalized.isEmpty) return null;
    return double.tryParse(normalized);
  }

  bool _sameNumber(double? first, double? second, {double epsilon = 1e-6}) {
    if (first == null && second == null) return true;
    if (first == null || second == null) return false;
    return (first - second).abs() <= epsilon;
  }

  bool _parametersUnchanged() {
    return _sameNumber(_parseNumber(_heightController.text), widget.heightM) &&
        _sameNumber(
          _parseNumber(_crownController.text),
          widget.crownWidthM,
        ) &&
        _sameNumber(
          _parseNumber(_trunkController.text),
          widget.trunkDiameterM,
        ) &&
        _sameNumber(_userScale, widget.scalePxToM, epsilon: 1e-10);
  }

  bool _speciesUnchanged() {
    return _speciesController.text.trim() == widget.species.trim();
  }

  String? _validateRange(
    String? value, {
    required String label,
    required double min,
    required double max,
  }) {
    final text = (value ?? '').trim();
    if (text.isEmpty) return null;
    final parsed = _parseNumber(text);
    if (parsed == null) return 'Введите корректное число';
    if (parsed < min || parsed > max) {
      return '$label: допустимо от $min до $max';
    }
    return null;
  }

  Future<void> _openMaskEditor() async {
    if (_isSending) return;

    final result = await Navigator.of(context).push<Map<String, dynamic>>(
      MaterialPageRoute(
        builder: (_) => MaskDrawingPage(
          originalImageBase64: widget.originalImageBase64,
          aiMaskBase64: widget.maskImageBase64,
          initialMaskBase64: _userMaskBase64,
        ),
      ),
    );

    if (result == null) return;
    final mask = (result['mask_png_base64'] ?? result['mask_b64'])?.toString();
    if (mask == null || mask.isEmpty) return;

    setState(() {
      _userMaskBase64 = mask;
      _treeOk = false;
    });
  }

  Future<void> _openStickEditor() async {
    if (_isSending) return;

    final scale = await Navigator.of(context).push<double>(
      MaterialPageRoute(
        builder: (_) => StickPage(
          originalImageBase64: widget.originalImageBase64,
          currentScalePxToM: _userScale ?? widget.scalePxToM ?? 0.0,
        ),
      ),
    );

    if (scale == null || scale <= 0) return;
    setState(() {
      _userScale = scale;
      _stickOk = false;
    });
  }

  String _responseMessage(Map<String, dynamic> response) {
    switch (response['status']?.toString()) {
      case 'verified':
        return 'Исправления сохранены. Пример доступен для обучения.';
      case 'saved_pending_review':
        return 'Исправления сохранены и ожидают проверки.';
      case 'saved_not_for_training':
        return 'Исправления сохранены без использования в обучении.';
      default:
        return 'Обратная связь сохранена.';
    }
  }

  String _extractServerError(http.Response response) {
    try {
      final decoded = jsonDecode(utf8.decode(response.bodyBytes));
      if (decoded is Map) {
        final detail =
            decoded['detail'] ?? decoded['error'] ?? decoded['message'];
        if (detail != null) return detail.toString();
      }
    } catch (_) {}
    return 'Ошибка сервера (${response.statusCode})';
  }

  Future<void> _sendFeedback() async {
    if (_isSending) return;
    if (!(_formKey.currentState?.validate() ?? false)) return;

    final species = _speciesController.text.trim();
    if (species.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Укажите вид дерева.')),
      );
      return;
    }

    if (_useForTraining && !_treeOk && _userMaskBase64 == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text(
            'Вы отметили маску как неверную. Нарисуйте корректную маску '
            'либо отключите использование примера в обучении.',
          ),
        ),
      );
      return;
    }

    final paramsOk = _parametersUnchanged();
    final speciesOk = _speciesUnchanged();

    final body = <String, dynamic>{
      'analysis_id': widget.analysisId,
      'use_for_training': _useForTraining,
      'tree_ok': _treeOk,
      'stick_ok': _stickOk,
      'params_ok': paramsOk,
      'species_ok': speciesOk,
      'correct_species': speciesOk ? null : species,
      'corrected_height_m': _parseNumber(_heightController.text),
      'corrected_crown_width_m': _parseNumber(_crownController.text),
      'corrected_trunk_diameter_m': _parseNumber(_trunkController.text),
      'corrected_scale_px_to_m': _userScale,
      'user_mask_base64': _userMaskBase64,
    };

    setState(() => _isSending = true);
    try {
      final headers = <String, String>{
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      };
      final token = widget.authToken?.trim();
      if (token != null && token.isNotEmpty) {
        headers['Authorization'] = 'Bearer $token';
      }

      final response = await http
          .post(
            Uri.parse('${widget.baseUrl}/feedback'),
            headers: headers,
            body: jsonEncode(body),
          )
          .timeout(const Duration(seconds: 45));

      if (!mounted) return;
      if (response.statusCode < 200 || response.statusCode >= 300) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text(_extractServerError(response))),
        );
        return;
      }

      final decoded = jsonDecode(utf8.decode(response.bodyBytes));
      final result = decoded is Map<String, dynamic>
          ? decoded
          : Map<String, dynamic>.from(decoded as Map);

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(_responseMessage(result))),
      );
      Navigator.of(context).pop(<String, dynamic>{
        ...result,
        'submitted': true,
      });
    } catch (error) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Ошибка сети: $error')),
      );
    } finally {
      if (mounted) setState(() => _isSending = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final paramsOk = _parametersUnchanged();
    final speciesOk = _speciesUnchanged();
    final previewBytes = _decodeImage(
      widget.annotatedImageBase64 ?? widget.originalImageBase64,
    );

    return Scaffold(
      appBar: AppBar(
        title: const Text('ПОДТВЕРЖДЕНИЕ'),
        actions: [
          if (_isSending)
            const Padding(
              padding: EdgeInsets.all(14),
              child: SizedBox(
                width: 18,
                height: 18,
                child: CircularProgressIndicator(strokeWidth: 2),
              ),
            )
          else
            IconButton(
              tooltip: 'Отправить',
              icon: const Icon(Icons.done_all),
              onPressed: _sendFeedback,
            ),
        ],
      ),
      body: Form(
        key: _formKey,
        child: ListView(
          padding: const EdgeInsets.fromLTRB(16, 16, 16, 100),
          children: [
            if (previewBytes != null) ...[
              GlassPanel(
                padding: EdgeInsets.zero,
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(24),
                  child: AspectRatio(
                    aspectRatio: 4 / 3,
                    child: Image.memory(previewBytes, fit: BoxFit.cover),
                  ),
                ),
              ),
              const SizedBox(height: 12),
            ],

            Ui.paddedCard(
              context,
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Icon(
                    Icons.verified_outlined,
                    color: AppTheme.primary,
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Text(
                      'Проверьте результат анализа и при необходимости '
                      'внесите корректировки. После отправки данные могут '
                      'использоваться для улучшения модели.',
                      style: Theme.of(context)
                          .textTheme
                          .bodyMedium
                          ?.copyWith(color: AppTheme.muted),
                    ),
                  ),
                ],
              ),
            ),

            Ui.sectionTitle(context, 'Инструменты'),
            Ui.paddedCard(
              context,
              child: Row(
                children: [
                  Expanded(
                    child: _ActionCard(
                      title: 'Маска',
                      icon: Icons.auto_fix_high,
                      statusOk: _treeOk,
                      onTap: _openMaskEditor,
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: _ActionCard(
                      title: 'Масштаб',
                      icon: Icons.straighten,
                      statusOk: _stickOk,
                      onTap: _openStickEditor,
                    ),
                  ),
                ],
              ),
            ),

            Ui.sectionTitle(context, 'Вид дерева'),
            Ui.paddedCard(
              context,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Ui.badge(
                        text: speciesOk ? 'Ок' : 'Изменено',
                        color: speciesOk
                            ? AppTheme.success
                            : AppTheme.warning,
                        icon: speciesOk ? Icons.check_circle : Icons.edit,
                      ),
                      const Spacer(),
                      Flexible(
                        child: Text(
                          'Исходно: ${widget.species}',
                          overflow: TextOverflow.ellipsis,
                          style: Theme.of(context)
                              .textTheme
                              .bodySmall
                              ?.copyWith(color: AppTheme.muted),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 10),
                  TextFormField(
                    controller: _speciesController,
                    enabled: !_isSending,
                    textCapitalization: TextCapitalization.sentences,
                    decoration: const InputDecoration(
                      labelText: 'Вид дерева',
                      hintText: 'Введите определённый вид',
                    ),
                    validator: (value) => (value ?? '').trim().isEmpty
                        ? 'Укажите вид дерева'
                        : null,
                    onChanged: (_) => setState(() {}),
                  ),
                ],
              ),
            ),

            Ui.sectionTitle(context, 'Параметры'),
            Ui.paddedCard(
              context,
              child: Column(
                children: [
                  Row(
                    children: [
                      Ui.badge(
                        text: paramsOk ? 'Ок' : 'Изменено',
                        color: paramsOk
                            ? AppTheme.success
                            : AppTheme.warning,
                        icon: paramsOk ? Icons.check_circle : Icons.tune,
                      ),
                      const Spacer(),
                      Text(
                        'Масштаб: ${_userScale?.toStringAsFixed(6) ?? '—'}',
                        style: Theme.of(context)
                            .textTheme
                            .bodySmall
                            ?.copyWith(color: AppTheme.muted),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  _numField(
                    controller: _heightController,
                    label: 'Высота, м',
                    min: 0.5,
                    max: 100,
                  ),
                  const SizedBox(height: 10),
                  _numField(
                    controller: _crownController,
                    label: 'Ширина кроны, м',
                    min: 0.1,
                    max: 100,
                  ),
                  const SizedBox(height: 10),
                  _numField(
                    controller: _trunkController,
                    label: 'Диаметр ствола, м',
                    min: 0.01,
                    max: 10,
                  ),
                ],
              ),
            ),

            Ui.sectionTitle(context, 'Датасет для обучения'),
            Ui.paddedCard(
              context,
              child: SwitchListTile(
                contentPadding: EdgeInsets.zero,
                activeColor: AppTheme.primary,
                title: const Text('Использовать для обучения'),
                subtitle: Text(
                  'Если выключить, данные сохранятся, но не попадут в '
                  'обучающий набор.',
                  style: Theme.of(context)
                      .textTheme
                      .bodySmall
                      ?.copyWith(color: AppTheme.muted),
                ),
                value: _useForTraining,
                onChanged: _isSending
                    ? null
                    : (value) => setState(() => _useForTraining = value),
              ),
            ),

            const SizedBox(height: 16),
            FilledButton.icon(
              onPressed: _isSending ? null : _sendFeedback,
              icon: _isSending
                  ? const SizedBox(
                      width: 18,
                      height: 18,
                      child: CircularProgressIndicator(
                        strokeWidth: 2,
                        color: Colors.black,
                      ),
                    )
                  : const Icon(Icons.done_all, color: Colors.black),
              label: Text(
                _isSending ? 'ОТПРАВКА...' : 'ПОДТВЕРДИТЬ И ОТПРАВИТЬ',
                style: const TextStyle(
                  color: Colors.black,
                  fontWeight: FontWeight.w900,
                  letterSpacing: 1,
                ),
              ),
              style: FilledButton.styleFrom(
                backgroundColor: AppTheme.primary,
                minimumSize: const Size.fromHeight(54),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(18),
                ),
              ),
            ),
            const SizedBox(height: 10),
            OutlinedButton(
              onPressed: _isSending
                  ? null
                  : () => Navigator.of(context).maybePop(),
              style: OutlinedButton.styleFrom(
                minimumSize: const Size.fromHeight(50),
              ),
              child: const Text('Отмена'),
            ),
          ],
        ),
      ),
    );
  }

  Widget _numField({
    required TextEditingController controller,
    required String label,
    required double min,
    required double max,
  }) {
    return TextFormField(
      controller: controller,
      enabled: !_isSending,
      keyboardType: const TextInputType.numberWithOptions(
        decimal: true,
        signed: false,
      ),
      decoration: InputDecoration(
        labelText: label,
        hintText: '—',
      ),
      validator: (value) => _validateRange(
        value,
        label: label,
        min: min,
        max: max,
      ),
      onChanged: (_) => setState(() {}),
    );
  }
}

class _ActionCard extends StatelessWidget {
  final String title;
  final IconData icon;
  final bool statusOk;
  final VoidCallback onTap;

  const _ActionCard({
    required this.title,
    required this.icon,
    required this.statusOk,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final color = statusOk ? AppTheme.success : AppTheme.warning;

    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(14),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 14),
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: color.withOpacity(0.25)),
          color: color.withOpacity(0.07),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, color: color),
            const SizedBox(height: 8),
            Text(
              title,
              style: Theme.of(context)
                  .textTheme
                  .titleSmall
                  ?.copyWith(fontWeight: FontWeight.w800),
            ),
            const SizedBox(height: 4),
            Text(
              statusOk ? 'Ок' : 'Изменено',
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                    color: color,
                    fontWeight: FontWeight.w800,
                  ),
            ),
          ],
        ),
      ),
    );
  }
}
