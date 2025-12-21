import 'dart:convert';
import 'package:flutter/material.dart';

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
  bool _paramsOk = true;
  bool _speciesOk = true;
  bool _useForTraining = true;

  String _selectedSpecies = '';
  final TextEditingController _customSpeciesController = TextEditingController();

  late TextEditingController _heightController;
  late TextEditingController _crownController;
  late TextEditingController _trunkController;
  late TextEditingController _scaleController;

  String? _userMaskBase64;

  @override
  void initState() {
    super.initState();
    _selectedSpecies = widget.species;

    _heightController =
        TextEditingController(text: widget.heightM?.toStringAsFixed(2) ?? '');
    _crownController =
        TextEditingController(text: widget.crownWidthM?.toStringAsFixed(2) ?? '');
    _trunkController =
        TextEditingController(text: widget.trunkDiameterM?.toStringAsFixed(2) ?? '');
    _scaleController =
        TextEditingController(text: widget.scalePxToM?.toStringAsFixed(6) ?? '');
  }

  @override
  void dispose() {
    _customSpeciesController.dispose();
    _heightController.dispose();
    _crownController.dispose();
    _trunkController.dispose();
    _scaleController.dispose();
    super.dispose();
  }

  double? _parseDouble(String text) {
    if (text.trim().isEmpty) return null;
    return double.tryParse(text.replaceAll(',', '.'));
  }

  String _displaySpecies() {
    if (!_speciesOk && _selectedSpecies.isNotEmpty) {
      return _selectedSpecies;
    }
    return widget.species;
  }

  Future<void> _openMaskPage() async {
    final resultMask = await Navigator.of(context).push<String?>(
      MaterialPageRoute(
        builder: (_) => MaskDrawingPage(
          originalImageBase64: widget.originalImageBase64,
          initialMaskBase64: _userMaskBase64,
        ),
      ),
    );

    if (resultMask != null) {
      setState(() {
        _userMaskBase64 = resultMask;
        _treeOk = true;
      });
    }
  }

  /// Коррекция палки: 2 точки => возвращается новый scale (м/px)
  Future<void> _openStickPage() async {
    final currentScalePxToM =
        _parseDouble(_scaleController.text) ?? widget.scalePxToM;

    if (currentScalePxToM == null || currentScalePxToM <= 0) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Сначала нужен найденный масштаб палки.')),
      );
      return;
    }

    final newScale = await Navigator.of(context).push<double?>(
      MaterialPageRoute(
        builder: (_) => StickPage(
          originalImageBase64: widget.originalImageBase64,
          currentScalePxToM: currentScalePxToM,
        ),
      ),
    );

    if (newScale == null) return;

    setState(() {
      final oldScale = currentScalePxToM;
      final factor = newScale / oldScale;

      _scaleController.text = newScale.toStringAsFixed(6);

      final h = _parseDouble(_heightController.text) ?? widget.heightM;
      final c = _parseDouble(_crownController.text) ?? widget.crownWidthM;
      final t = _parseDouble(_trunkController.text) ?? widget.trunkDiameterM;

      if (h != null) _heightController.text = (h * factor).toStringAsFixed(2);
      if (c != null) _crownController.text = (c * factor).toStringAsFixed(2);
      if (t != null) _trunkController.text = (t * factor).toStringAsFixed(2);

      _stickOk = true;
      _paramsOk = true;
    });
  }

  void _submit() {
    final feedback = <String, dynamic>{
      "tree_ok": _treeOk,
      "stick_ok": _stickOk,
      "params_ok": _paramsOk,
      "species_ok": _speciesOk,
      "correct_species": _speciesOk ? null : _selectedSpecies,
      "user_mask_base64": _userMaskBase64,
      "use_for_training": _useForTraining,

      "height_m_corrected": _parseDouble(_heightController.text),
      "crown_width_m_corrected": _parseDouble(_crownController.text),
      "trunk_diameter_m_corrected": _parseDouble(_trunkController.text),
      "scale_px_to_m_corrected": _parseDouble(_scaleController.text),
    };

    Navigator.of(context).pop(feedback);
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final bytes = base64Decode(
      widget.annotatedImageBase64 ?? widget.originalImageBase64,
    );

    return Scaffold(
      appBar: AppBar(
        title: const Text('Проверка анализа'),
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          ClipRRect(
            borderRadius: BorderRadius.circular(20),
            child: Image.memory(bytes),
          ),
          const SizedBox(height: 16),

          Card(
            child: Padding(
              padding: const EdgeInsets.all(12),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Общий итог',
                    style: theme.textTheme.titleMedium
                        ?.copyWith(fontWeight: FontWeight.w600),
                  ),
                  const SizedBox(height: 8),
                  Wrap(
                    spacing: 8,
                    runSpacing: 4,
                    children: [
                      _buildChip(
                        ok: _treeOk,
                        label: 'Дерево',
                        onTap: () => setState(() => _treeOk = !_treeOk),
                      ),
                      _buildChip(
                        ok: _stickOk,
                        label: 'Палка',
                        onTap: () => setState(() => _stickOk = !_stickOk),
                      ),
                      _buildChip(
                        ok: _paramsOk,
                        label: 'Параметры',
                        onTap: () => setState(() => _paramsOk = !_paramsOk),
                      ),
                      _buildChip(
                        ok: _speciesOk,
                        label: 'Порода',
                        onTap: () => setState(() => _speciesOk = !_speciesOk),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  Row(
                    children: [
                      Switch(
                        value: _useForTraining,
                        onChanged: (v) => setState(() => _useForTraining = v),
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          'Разрешить использовать этот пример для дообучения модели',
                          style: theme.textTheme.bodySmall,
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 16),

          _buildSectionHeader(
            icon: Icons.forest_outlined,
            title: 'Выделение дерева (маска)',
            trailing: TextButton.icon(
              icon: const Icon(Icons.brush_outlined),
              label: const Text('Редактировать'),
              onPressed: _openMaskPage,
            ),
          ),
          Card(
            child: SwitchListTile(
              title: const Text('Дерево выделено правильно'),
              value: _treeOk,
              onChanged: (v) => setState(() => _treeOk = v),
            ),
          ),
          const SizedBox(height: 16),

          _buildSectionHeader(
            icon: Icons.straighten,
            title: 'Палка (масштаб)',
            trailing: TextButton.icon(
              icon: const Icon(Icons.edit_outlined),
              label: const Text('Исправить'),
              onPressed: _openStickPage,
            ),
          ),
          Card(
            child: Column(
              children: [
                SwitchListTile(
                  title: const Text('Палка определена правильно'),
                  value: _stickOk,
                  onChanged: (v) => setState(() => _stickOk = v),
                ),
                ListTile(
                  title: const Text('Масштаб, м/px'),
                  subtitle: TextField(
                    controller: _scaleController,
                    keyboardType:
                        const TextInputType.numberWithOptions(decimal: true),
                    decoration: const InputDecoration(
                      hintText: 'Например: 0.004500',
                    ),
                    onChanged: (_) => setState(() => _paramsOk = false),
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 16),

          _buildSectionHeader(
            icon: Icons.analytics_outlined,
            title: 'Параметры дерева',
          ),
          Card(
            child: Column(
              children: [
                SwitchListTile(
                  title: const Text('Параметры рассчитаны верно'),
                  value: _paramsOk,
                  onChanged: (v) => setState(() => _paramsOk = v),
                ),
                _buildParamField(label: 'Высота, м', controller: _heightController),
                _buildParamField(label: 'Ширина кроны, м', controller: _crownController),
                _buildParamField(label: 'Диаметр ствола, м', controller: _trunkController),
              ],
            ),
          ),
          const SizedBox(height: 16),

          _buildSectionHeader(
            icon: Icons.park,
            title: 'Вид дерева',
          ),
          Card(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                SwitchListTile(
                  title: const Text('Вид определён верно'),
                  value: _speciesOk,
                  onChanged: (v) => setState(() => _speciesOk = v),
                ),
                if (!_speciesOk) ...[
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                    child: DropdownButtonFormField<String>(
                      value: _selectedSpecies.isEmpty ? null : _selectedSpecies,
                      items: const [
                        DropdownMenuItem(value: 'Береза', child: Text('Берёза')),
                        DropdownMenuItem(value: 'Дуб', child: Text('Дуб')),
                        DropdownMenuItem(value: 'Ель', child: Text('Ель')),
                        DropdownMenuItem(value: 'Сосна', child: Text('Сосна')),
                        DropdownMenuItem(value: 'Тополь', child: Text('Тополь')),
                        DropdownMenuItem(value: 'Другое', child: Text('Другое')),
                      ],
                      decoration: const InputDecoration(
                        labelText: 'Правильный вид',
                        border: OutlineInputBorder(),
                      ),
                      onChanged: (value) => setState(() => _selectedSpecies = value ?? ''),
                    ),
                  ),
                  if (_selectedSpecies == 'Другое')
                    Padding(
                      padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
                      child: TextField(
                        controller: _customSpeciesController,
                        decoration: const InputDecoration(
                          labelText: 'Введите вид дерева',
                          border: OutlineInputBorder(),
                        ),
                        onChanged: (text) => _selectedSpecies = text,
                      ),
                    ),
                ] else
                  Padding(
                    padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
                    child: Text(
                      'Текущий вид: ${_displaySpecies()}',
                      style: theme.textTheme.bodyMedium,
                    ),
                  ),
              ],
            ),
          ),
          const SizedBox(height: 24),

          FilledButton.icon(
            onPressed: _submit,
            icon: const Icon(Icons.send_rounded),
            label: const Padding(
              padding: EdgeInsets.symmetric(vertical: 12),
              child: Text('Сохранить и отправить'),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSectionHeader({
    required IconData icon,
    required String title,
    Widget? trailing,
  }) {
    final theme = Theme.of(context);
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Row(
        children: [
          Icon(icon),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              title,
              style: theme.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w600),
            ),
          ),
          if (trailing != null) trailing,
        ],
      ),
    );
  }

  Widget _buildParamField({
    required String label,
    required TextEditingController controller,
  }) {
    return ListTile(
      title: Text(label),
      subtitle: TextField(
        controller: controller,
        keyboardType: const TextInputType.numberWithOptions(decimal: true, signed: false),
        decoration: const InputDecoration(hintText: 'Измените при необходимости'),
        onChanged: (_) => setState(() => _paramsOk = false),
      ),
    );
  }

  Widget _buildChip({
    required bool ok,
    required String label,
    required VoidCallback onTap,
  }) {
    final colorBg = ok ? const Color(0xFFD9F5DC) : const Color(0xFFFFE1E1);
    final colorFg = ok ? const Color(0xFF1B5E20) : const Color(0xFFB71C1C);
    return GestureDetector(
      onTap: onTap,
      child: Chip(
        backgroundColor: colorBg,
        label: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(ok ? Icons.check_circle : Icons.error_outline, size: 16, color: colorFg),
            const SizedBox(width: 4),
            Text(label, style: TextStyle(color: colorFg, fontWeight: FontWeight.w600)),
          ],
        ),
      ),
    );
  }
}
