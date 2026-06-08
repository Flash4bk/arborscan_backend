import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

import 'app_theme.dart';
import 'mask_drawing_page.dart';
import 'stick_page.dart';

class FeedbackPage extends StatefulWidget {
  final String analysisId;
  final String originalImageBase64;
  final String? annotatedImageBase64;

  final String species;
  final double? heightM;
  final double? crownWidthM;
  final double? trunkDiameterM;
  final double? scalePxToM;

  const FeedbackPage({
    super.key,
    required this.analysisId,
    required this.originalImageBase64,
    this.annotatedImageBase64,
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
  bool _treeOk = true;
  bool _stickOk = true;
  bool _useForTraining = true;
  bool _isSending = false;

  late String _selectedSpecies;
  late TextEditingController _heightController;
  late TextEditingController _crownController;
  late TextEditingController _trunkController;

  String? _userMaskBase64;
  double? _userScale;

  final List<String> _popularSpecies = ["Береза", "Дуб", "Ель", "Сосна", "Тополь"];

  @override
  void initState() {
    super.initState();
    _selectedSpecies = widget.species;
    _heightController = TextEditingController(text: widget.heightM?.toStringAsFixed(2) ?? '');
    _crownController = TextEditingController(text: widget.crownWidthM?.toStringAsFixed(2) ?? '');
    _trunkController = TextEditingController(text: widget.trunkDiameterM?.toStringAsFixed(2) ?? '');
    _userScale = widget.scalePxToM;
  }

  @override
  void dispose() {
    _heightController.dispose();
    _crownController.dispose();
    _trunkController.dispose();
    super.dispose();
  }

  bool _checkParamsOk() {
    final h = double.tryParse(_heightController.text);
    final c = double.tryParse(_crownController.text);
    final t = double.tryParse(_trunkController.text);

    final okH = (h == null && widget.heightM == null) ||
        (h != null && widget.heightM != null && (h - widget.heightM!).abs() < 1e-6);
    final okC = (c == null && widget.crownWidthM == null) ||
        (c != null && widget.crownWidthM != null && (c - widget.crownWidthM!).abs() < 1e-6);
    final okT = (t == null && widget.trunkDiameterM == null) ||
        (t != null && widget.trunkDiameterM != null && (t - widget.trunkDiameterM!).abs() < 1e-6);

    final okS = (_userScale == null && widget.scalePxToM == null) ||
        (_userScale != null &&
            widget.scalePxToM != null &&
            (_userScale! - widget.scalePxToM!).abs() < 1e-9);

    return okH && okC && okT && okS;
  }

  Future<void> _openMaskEditor() async {
    if (_isSending) return;

    final result = await Navigator.push<Map<String, dynamic>>(
      context,
      MaterialPageRoute(
        builder: (_) => MaskDrawingPage(
          originalImageBase64: widget.originalImageBase64,
          aiMaskBase64: widget.annotatedImageBase64,
        ),
      ),
    );

    if (result == null) return;

    final mask = (result['mask_b64'] ?? result['mask_png_base64']) as String?;
    if (mask != null && mask.isNotEmpty) {
      setState(() {
        _userMaskBase64 = mask;
        _treeOk = false;
      });
    }
  }

  Future<void> _openStickEditor() async {
    if (_isSending) return;

    // StickPage требует currentScalePxToM (double, required).
    final current = _userScale ?? widget.scalePxToM ?? 0.0;

    final scale = await Navigator.push<double>(
      context,
      MaterialPageRoute(
        builder: (_) => StickPage(
          originalImageBase64: widget.originalImageBase64,
          currentScalePxToM: current,
        ),
      ),
    );

    if (scale == null) return;

    setState(() {
      _userScale = scale;
      _stickOk = false;
    });
  }

  Future<void> _sendFeedback() async {
    if (_isSending) return;
    setState(() => _isSending = true);

    final body = {
      "analysis_id": widget.analysisId,
      "use_for_training": _useForTraining,

      // quality flags
      "tree_ok": _treeOk,
      "stick_ok": _stickOk,
      "params_ok": _checkParamsOk(),
      "species_ok": _selectedSpecies == widget.species,

      // corrected values (only send when user changed them)
      "correct_species": (_selectedSpecies == widget.species) ? null : _selectedSpecies,
      "corrected_height_m": _heightController.text.trim().isEmpty
          ? null
          : double.tryParse(_heightController.text.trim().replaceAll(',', '.')),
      "corrected_crown_width_m": _crownController.text.trim().isEmpty
          ? null
          : double.tryParse(_crownController.text.trim().replaceAll(',', '.')),
      "corrected_trunk_diameter_m": _trunkController.text.trim().isEmpty
          ? null
          : double.tryParse(_trunkController.text.trim().replaceAll(',', '.')),
      "corrected_scale_px_to_m": _userScale,

      // IMPORTANT: backend expects this key (and accepts camelCase too)
      "user_mask_base64": _userMaskBase64,
    };

    try {
      final response = await http.post(
        Uri.parse('https://arborscanbackend-production.up.railway.app/feedback'),
        headers: {"Content-Type": "application/json"},
        body: jsonEncode(body),
      );

      if (!mounted) return;

      if (response.statusCode >= 200 && response.statusCode < 300) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text("Данные успешно подтверждены")),
        );
        Navigator.of(context).maybePop({"ok": true});
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text("Ошибка отправки: ${response.statusCode}")),
        );
      }
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Ошибка сети: $e")),
      );
    } finally {
      if (mounted) setState(() => _isSending = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final paramsOk = _checkParamsOk();
    final speciesOk = _selectedSpecies == widget.species;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Подтверждение'),
        actions: [
          if (_isSending)
            const Padding(
              padding: EdgeInsets.all(14),
              child: SizedBox(width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2)),
            )
          else
            IconButton(
              tooltip: 'Отправить',
              icon: const Icon(Icons.done_all),
              onPressed: _sendFeedback,
            ),
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Ui.paddedCard(
            context,
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Icon(Icons.verified_outlined),
                const SizedBox(width: 12),
                Expanded(
                  child: Text(
                    'Проверьте результат анализа и, при необходимости, внесите корректировки. '
                    'После отправки данные могут быть использованы для улучшения модели.',
                    style: Theme.of(context).textTheme.bodyMedium?.copyWith(color: AppTheme.muted),
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
                      color: speciesOk ? AppTheme.success : AppTheme.warning,
                      icon: speciesOk ? Icons.check_circle : Icons.edit,
                    ),
                    const Spacer(),
                    Text(
                      'Исходно: ${widget.species}',
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(color: AppTheme.muted),
                    ),
                  ],
                ),
                const SizedBox(height: 10),
                DropdownButtonFormField<String>(
                  value: _selectedSpecies,
                  isExpanded: true,
                  items: _popularSpecies
                      .map((s) => DropdownMenuItem<String>(value: s, child: Text(s)))
                      .toList(),
                  onChanged: (v) => setState(() => _selectedSpecies = v ?? _selectedSpecies),
                  decoration: const InputDecoration(labelText: 'Вид'),
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
                      color: paramsOk ? AppTheme.success : AppTheme.warning,
                      icon: paramsOk ? Icons.check_circle : Icons.tune,
                    ),
                    const Spacer(),
                    Text(
                      'Масштаб: ${_userScale?.toStringAsFixed(6) ?? '—'}',
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(color: AppTheme.muted),
                    ),
                  ],
                ),
                const SizedBox(height: 12),
                _numField(controller: _heightController, label: 'Высота, м'),
                const SizedBox(height: 10),
                _numField(controller: _crownController, label: 'Ширина кроны, м'),
                const SizedBox(height: 10),
                _numField(controller: _trunkController, label: 'Диаметр ствола, м'),
              ],
            ),
          ),

          Ui.sectionTitle(context, 'Датасет для обучения'),
          Ui.paddedCard(
            context,
            child: SwitchListTile(
              contentPadding: EdgeInsets.zero,
              title: const Text('Использовать для обучения'),
              subtitle: Text(
                'Если выключить, данные будут сохранены, но не попадут в обучающий набор.',
                style: Theme.of(context).textTheme.bodySmall?.copyWith(color: AppTheme.muted),
              ),
              value: _useForTraining,
              onChanged: _isSending ? null : (v) => setState(() => _useForTraining = v),
            ),
          ),

          const SizedBox(height: 16),

          ElevatedButton.icon(
            onPressed: _isSending ? null : _sendFeedback,
            icon: const Icon(Icons.done_all),
            label: const Text('Подтвердить и отправить'),
            style: ElevatedButton.styleFrom(minimumSize: const Size.fromHeight(50)),
          ),
          const SizedBox(height: 10),
          OutlinedButton(
            onPressed: _isSending ? null : () => Navigator.of(context).maybePop(),
            child: const Text('Отмена'),
          ),
        ],
      ),
    );
  }

  Widget _numField({required TextEditingController controller, required String label}) {
    return TextField(
      controller: controller,
      keyboardType: const TextInputType.numberWithOptions(decimal: true, signed: false),
      decoration: InputDecoration(
        labelText: label,
        hintText: '—',
      ),
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
              style: Theme.of(context).textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w800),
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
