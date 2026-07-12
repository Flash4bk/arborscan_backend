import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

import 'app_theme.dart';
import 'mask_drawing_page.dart';
import 'stick_page.dart';

/// Экран проверки результата анализа.
///
/// Важное правило этапа 3: этот экран сам выполняет единственный POST /feedback
/// и возвращает на предыдущий экран уже подтверждённый ответ backend. Это
/// исключает прежнюю двойную отправку одного и того же feedback.
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
    required this.baseUrl,
    required this.analysisId,
    required this.originalImageBase64,
    required this.species,
    this.authToken,
    this.annotatedImageBase64,
    this.maskImageBase64,
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
        _sameNumber(_parseNumber(_crownController.text), widget.crownWidthM) &&
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
        ),
      ),
    );

    if (result == null) return;
    final mask = (result['mask_png_base64'] ?? result['mask_b64'])?.toString();
    if (mask == null || mask.isEmpty) return;

    setState(() {
      _userMaskBase64 = mask;
      // false означает: исходная AI-маска была исправлена. Backend отдельно
      // учитывает наличие корректирующей пользовательской маски.
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
        return 'Исправления сохранены. Пример доступен для обучающего набора.';
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
      final decoded = jsonDecode(response.body);
      if (decoded is Map) {
        final detail = decoded['detail'] ?? decoded['error'] ?? decoded['message'];
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
            'Вы отметили маску как неверную. Нарисуйте корректную маску либо '
            'отключите использование примера в обучении.',
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

      final decoded = jsonDecode(response.body);
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
        title: const Text('Проверка анализа'),
        actions: [
          if (_isSending)
            const Padding(
              padding: EdgeInsets.all(14),
              child: SizedBox(
                width: 20,
                height: 20,
                child: CircularProgressIndicator(strokeWidth: 2),
              ),
            )
          else
            IconButton(
              tooltip: 'Сохранить',
              icon: const Icon(Icons.done_all),
              onPressed: _sendFeedback,
            ),
        ],
      ),
      body: Form(
        key: _formKey,
        child: ListView(
          padding: const EdgeInsets.fromLTRB(16, 8, 16, 24),
          children: [
            Ui.paddedCard(
              context,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      const Icon(Icons.fact_check_outlined),
                      const SizedBox(width: 10),
                      Expanded(
                        child: Text(
                          'Проверьте результат. Исправления сохраняются в '
                          'Supabase и не зависят от временного кеша Railway.',
                          style: Theme.of(context)
                              .textTheme
                              .bodyMedium
                              ?.copyWith(color: AppTheme.muted),
                        ),
                      ),
                    ],
                  ),
                  if (previewBytes != null) ...[
                    const SizedBox(height: 14),
                    ClipRRect(
                      borderRadius: BorderRadius.circular(14),
                      child: AspectRatio(
                        aspectRatio: 4 / 3,
                        child: Image.memory(previewBytes, fit: BoxFit.cover),
                      ),
                    ),
                  ],
                ],
              ),
            ),

            Ui.sectionTitle(context, 'Контур и масштаб'),
            Ui.paddedCard(
              context,
              child: Column(
                children: [
                  Row(
                    children: [
                      Expanded(
                        child: _ActionCard(
                          title: 'Маска дерева',
                          icon: Icons.gesture,
                          statusText: _userMaskBase64 != null
                              ? 'Исправлена вручную'
                              : (_treeOk ? 'Подтверждена' : 'Требует правки'),
                          statusOk: _treeOk || _userMaskBase64 != null,
                          onTap: _openMaskEditor,
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: _ActionCard(
                          title: 'Масштаб',
                          icon: Icons.straighten,
                          statusText: _stickOk ? 'Подтверждён' : 'Исправлен',
                          statusOk: _stickOk,
                          onTap: _openStickEditor,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 10),
                  SwitchListTile.adaptive(
                    contentPadding: EdgeInsets.zero,
                    title: const Text('Исходная маска ИИ верна'),
                    subtitle: Text(
                      _userMaskBase64 != null
                          ? 'Сохранена новая пользовательская маска.'
                          : 'Выключите и нарисуйте новый контур, если ИИ ошибся.',
                    ),
                    value: _treeOk,
                    onChanged: _isSending
                        ? null
                        : (value) => setState(() => _treeOk = value),
                  ),
                  SwitchListTile.adaptive(
                    contentPadding: EdgeInsets.zero,
                    title: const Text('Масштаб определён верно'),
                    subtitle: Text(
                      _userScale == null
                          ? 'Масштаб отсутствует.'
                          : 'Текущее значение: '
                              '${_userScale!.toStringAsFixed(8)} м/px',
                    ),
                    value: _stickOk,
                    onChanged: _isSending
                        ? null
                        : (value) => setState(() => _stickOk = value),
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
                        text: speciesOk ? 'Без изменений' : 'Исправлен',
                        color: speciesOk ? AppTheme.success : AppTheme.warning,
                        icon: speciesOk ? Icons.check_circle : Icons.edit,
                      ),
                      const Spacer(),
                      Flexible(
                        child: Text(
                          'Исходно: ${widget.species}',
                          textAlign: TextAlign.end,
                          style: Theme.of(context)
                              .textTheme
                              .bodySmall
                              ?.copyWith(color: AppTheme.muted),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  TextFormField(
                    controller: _speciesController,
                    textCapitalization: TextCapitalization.sentences,
                    decoration: const InputDecoration(
                      labelText: 'Вид дерева',
                      helperText: 'Можно указать вид, отсутствующий в коротком списке.',
                    ),
                    validator: (value) => (value ?? '').trim().isEmpty
                        ? 'Укажите вид дерева'
                        : null,
                    onChanged: (_) => setState(() {}),
                  ),
                ],
              ),
            ),

            Ui.sectionTitle(context, 'Размеры дерева'),
            Ui.paddedCard(
              context,
              child: Column(
                children: [
                  Row(
                    children: [
                      Ui.badge(
                        text: paramsOk ? 'Без изменений' : 'Исправлены',
                        color: paramsOk ? AppTheme.success : AppTheme.warning,
                        icon: paramsOk ? Icons.check_circle : Icons.tune,
                      ),
                      const Spacer(),
                      Text(
                        'м',
                        style: Theme.of(context)
                            .textTheme
                            .bodySmall
                            ?.copyWith(color: AppTheme.muted),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  TextFormField(
                    controller: _heightController,
                    keyboardType: const TextInputType.numberWithOptions(
                      decimal: true,
                    ),
                    decoration: const InputDecoration(labelText: 'Высота, м'),
                    validator: (value) => _validateRange(
                      value,
                      label: 'Высота',
                      min: 0.5,
                      max: 100,
                    ),
                    onChanged: (_) => setState(() {}),
                  ),
                  const SizedBox(height: 10),
                  TextFormField(
                    controller: _crownController,
                    keyboardType: const TextInputType.numberWithOptions(
                      decimal: true,
                    ),
                    decoration: const InputDecoration(
                      labelText: 'Ширина кроны, м',
                    ),
                    validator: (value) => _validateRange(
                      value,
                      label: 'Ширина кроны',
                      min: 0.1,
                      max: 100,
                    ),
                    onChanged: (_) => setState(() {}),
                  ),
                  const SizedBox(height: 10),
                  TextFormField(
                    controller: _trunkController,
                    keyboardType: const TextInputType.numberWithOptions(
                      decimal: true,
                    ),
                    decoration: const InputDecoration(
                      labelText: 'Диаметр ствола, м',
                    ),
                    validator: (value) => _validateRange(
                      value,
                      label: 'Диаметр ствола',
                      min: 0.01,
                      max: 10,
                    ),
                    onChanged: (_) => setState(() {}),
                  ),
                ],
              ),
            ),

            Ui.sectionTitle(context, 'Использование данных'),
            Ui.paddedCard(
              context,
              child: SwitchListTile.adaptive(
                contentPadding: EdgeInsets.zero,
                title: const Text('Разрешить использовать для обучения'),
                subtitle: Text(
                  _useForTraining
                      ? 'После проверки пример сможет попасть в обучающий набор.'
                      : 'Исправления сохранятся, но пример будет исключён из обучения.',
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

            const SizedBox(height: 18),
            FilledButton.icon(
              onPressed: _isSending ? null : _sendFeedback,
              icon: const Icon(Icons.cloud_upload_outlined),
              label: Text(_isSending ? 'Сохранение…' : 'Сохранить проверку'),
              style: FilledButton.styleFrom(
                minimumSize: const Size.fromHeight(52),
              ),
            ),
            const SizedBox(height: 10),
            OutlinedButton(
              onPressed: _isSending ? null : () => Navigator.of(context).pop(),
              child: const Text('Отмена'),
            ),
          ],
        ),
      ),
    );
  }
}

class _ActionCard extends StatelessWidget {
  final String title;
  final String statusText;
  final IconData icon;
  final bool statusOk;
  final VoidCallback onTap;

  const _ActionCard({
    required this.title,
    required this.statusText,
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
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 14),
        decoration: BoxDecoration(
          color: color.withOpacity(0.07),
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: color.withOpacity(0.25)),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, color: color),
            const SizedBox(height: 8),
            Text(
              title,
              textAlign: TextAlign.center,
              style: Theme.of(context)
                  .textTheme
                  .titleSmall
                  ?.copyWith(fontWeight: FontWeight.w800),
            ),
            const SizedBox(height: 5),
            Text(
              statusText,
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
              textAlign: TextAlign.center,
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                    color: color,
                    fontWeight: FontWeight.w700,
                  ),
            ),
          ],
        ),
      ),
    );
  }
}
