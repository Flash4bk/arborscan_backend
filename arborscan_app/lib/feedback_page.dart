import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/material.dart';

import 'mask_drawing_page.dart';
import 'stick_page.dart';

class FeedbackPage extends StatefulWidget {
  final String analysisId;
  final String originalImageBase64;

  final String species;
  final double? heightM;
  final double? crownWidthM;
  final double? trunkDiameterM;
  final double? scalePxToM;

  const FeedbackPage({
    super.key,
    required this.analysisId,
    required this.originalImageBase64,
    required this.species,
    required this.heightM,
    required this.crownWidthM,
    required this.trunkDiameterM,
    required this.scalePxToM,
  });

  @override
  State<FeedbackPage> createState() => _FeedbackPageState();
}

class _FeedbackPageState extends State<FeedbackPage> {
  // флаги корректности
  bool _treeOk = true;
  bool _stickOk = true;
  bool _paramsOk = true;
  bool _speciesOk = true;
  bool _useForTraining = true;

  // вид
  late String _selectedSpecies;
  final TextEditingController _customSpeciesController =
      TextEditingController();

  // редактируемые параметры
  double? _heightM;
  double? _crownWidthM;
  double? _trunkDiameterM;
  double? _scalePxToM;

  // пользовательская маска (дерево или палка — решим позже сервером)
  String? _userMaskBase64;

  @override
  void initState() {
    super.initState();
    _selectedSpecies = widget.species;
    _heightM = widget.heightM;
    _crownWidthM = widget.crownWidthM;
    _trunkDiameterM = widget.trunkDiameterM;
    _scalePxToM = widget.scalePxToM;
  }

  @override
  void dispose() {
    _customSpeciesController.dispose();
    super.dispose();
  }

  Uint8List get _originalBytes =>
      base64Decode(widget.originalImageBase64);

  void _onSavePressed() {
    // если пользователь выбрал "Другое", а строка пустая — возьмём исходный вид
    String? correctedSpecies;
    if (!_speciesOk) {
      if (_selectedSpecies == 'Другое') {
        correctedSpecies = _customSpeciesController.text.trim();
        if (correctedSpecies.isEmpty) {
          correctedSpecies = widget.species;
        }
      } else {
        correctedSpecies = _selectedSpecies;
      }
    }

    Navigator.pop<Map<String, dynamic>>(context, {
      "tree_ok": _treeOk,
      "stick_ok": _stickOk,
      "params_ok": _paramsOk,
      "species_ok": _speciesOk,
      "correct_species": correctedSpecies,
      "use_for_training": _useForTraining,
      "user_mask_base64": _userMaskBase64,
      // откорректированные параметры
      "height_m": _heightM,
      "crown_width_m": _crownWidthM,
      "trunk_diameter_m": _trunkDiameterM,
      "scale_px_to_m": _scalePxToM,
    });
  }

  Future<void> _openMaskDrawing() async {
    final maskBytes = await Navigator.push<Uint8List?>(
      context,
      MaterialPageRoute(
        builder: (_) => MaskDrawingPage(
          imageBytes: _originalBytes,
          title: "Обведите дерево / палку",
          hint:
              "Аккуратно обведите дерево и палку.\nЭта маска поможет дообучить модель.",
        ),
      ),
    );

    if (maskBytes != null) {
      setState(() {
        _userMaskBase64 = base64Encode(maskBytes);
      });
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Маска сохранена.")),
      );
    }
  }

  Future<void> _openStickPage() async {
    final result = await Navigator.push<Map<String, double?>?>(
      context,
      MaterialPageRoute(
        builder: (_) => StickPage(
          imageBytes: _originalBytes,
          heightM: _heightM,
          crownWidthM: _crownWidthM,
          trunkDiameterM: _trunkDiameterM,
          scalePxToM: _scalePxToM,
        ),
      ),
    );

    if (result != null) {
      setState(() {
        _heightM = result['height_m'];
        _crownWidthM = result['crown_width_m'];
        _trunkDiameterM = result['trunk_diameter_m'];
        _scalePxToM = result['scale_px_to_m'];
        _paramsOk = true; // раз пользователь поправил — считаем ок
        _stickOk = true;
      });
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Параметры обновлены.")),
      );
    }
  }

  String _formatMeters(double? v) {
    if (v == null) return "—";
    return "${v.toStringAsFixed(2)} м";
  }

  String _formatScale(double? v) {
    if (v == null) return "—";
    return "1 px ≈ ${v.toStringAsFixed(4)} м";
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(
        title: const Text("Проверка анализа"),
      ),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(16, 12, 16, 24),
        children: [
          // фото
          ClipRRect(
            borderRadius: BorderRadius.circular(24),
            child: Image.memory(
              _originalBytes,
              fit: BoxFit.cover,
            ),
          ),
          const SizedBox(height: 16),

          // карточка с текущими параметрами
          _buildMetricsCard(theme),
          const SizedBox(height: 16),

          // флаги корректности (только здесь, без дублей в StickPage)
          _buildSwitch(
            title: "Дерево выделено правильно",
            value: _treeOk,
            onChanged: (v) => setState(() => _treeOk = v),
          ),
          _buildSwitch(
            title: "Палка определена правильно",
            value: _stickOk,
            onChanged: (v) => setState(() => _stickOk = v),
          ),
          _buildSwitch(
            title: "Параметры рассчитаны верно",
            value: _paramsOk,
            onChanged: (v) => setState(() => _paramsOk = v),
          ),
          _buildSwitch(
            title: "Вид определён верно",
            value: _speciesOk,
            onChanged: (v) => setState(() => _speciesOk = v),
          ),

          const SizedBox(height: 8),

          if (!_speciesOk) _buildSpeciesSelector(),

          const SizedBox(height: 16),

          // две функции: маска и правка палки/параметров
          Row(
            children: [
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: _openMaskDrawing,
                  icon: const Icon(Icons.brush_outlined),
                  label: const Text("Нарисовать маску"),
                  style: OutlinedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(vertical: 12),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(999),
                    ),
                  ),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: _openStickPage,
                  icon: const Icon(Icons.straighten),
                  label: const Text("Палка / параметры"),
                  style: OutlinedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(vertical: 12),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(999),
                    ),
                  ),
                ),
              ),
            ],
          ),

          const SizedBox(height: 20),

          // использовать для обучения
          Card(
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(20),
            ),
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Icon(Icons.school_outlined),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          "Использовать этот пример для обучения",
                          style: theme.textTheme.titleSmall
                              ?.copyWith(fontWeight: FontWeight.w600),
                        ),
                        const SizedBox(height: 4),
                        Text(
                          "Если включено — изображение, ваши правки и маски "
                          "будут сохранены как доверенный пример.",
                          style: theme.textTheme.bodySmall
                              ?.copyWith(color: Colors.black54),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(width: 8),
                  Switch(
                    value: _useForTraining,
                    onChanged: (v) => setState(() => _useForTraining = v),
                  ),
                ],
              ),
            ),
          ),

          const SizedBox(height: 24),

          // кнопка сохранения
          SizedBox(
            width: double.infinity,
            child: FilledButton(
              onPressed: _onSavePressed,
              style: FilledButton.styleFrom(
                padding: const EdgeInsets.symmetric(vertical: 16),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(999),
                ),
              ),
              child: const Text(
                "Сохранить и отправить",
                style: TextStyle(fontWeight: FontWeight.w600),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildMetricsCard(ThemeData theme) {
    return Card(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
      child: Padding(
        padding: const EdgeInsets.fromLTRB(16, 16, 16, 12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              "Текущий результат",
              style: theme.textTheme.titleMedium
                  ?.copyWith(fontWeight: FontWeight.w700),
            ),
            const SizedBox(height: 12),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                _MetricChip(
                  icon: Icons.park_outlined,
                  label: "Вид",
                  value: widget.species,
                ),
                _MetricChip(
                  icon: Icons.height,
                  label: "Высота",
                  value: _formatMeters(_heightM),
                ),
                _MetricChip(
                  icon: Icons.landscape_outlined,
                  label: "Крона",
                  value: _formatMeters(_crownWidthM),
                ),
                _MetricChip(
                  icon: Icons.radio_button_unchecked,
                  label: "Ствол",
                  value: _formatMeters(_trunkDiameterM),
                ),
                _MetricChip(
                  icon: Icons.straighten,
                  label: "Масштаб",
                  value: _formatScale(_scalePxToM),
                  isMuted: true,
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSwitch({
    required String title,
    required bool value,
    required ValueChanged<bool> onChanged,
  }) {
    return Card(
      margin: const EdgeInsets.only(bottom: 8),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
      child: SwitchListTile(
        title: Text(title),
        value: value,
        onChanged: onChanged,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
      ),
    );
  }

  Widget _buildSpeciesSelector() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          "Выберите правильный вид:",
          style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
        ),
        const SizedBox(height: 8),
        DropdownButtonFormField<String>(
          decoration: const InputDecoration(
            border: OutlineInputBorder(),
          ),
          value: _selectedSpecies,
          items: const [
            DropdownMenuItem(value: "Береза", child: Text("Берёза")),
            DropdownMenuItem(value: "Дуб", child: Text("Дуб")),
            DropdownMenuItem(value: "Ель", child: Text("Ель")),
            DropdownMenuItem(value: "Сосна", child: Text("Сосна")),
            DropdownMenuItem(value: "Тополь", child: Text("Тополь")),
            DropdownMenuItem(value: "Другое", child: Text("Другое")),
          ],
          onChanged: (value) {
            setState(() {
              _selectedSpecies = value ?? "Другое";
            });
          },
        ),
        const SizedBox(height: 10),
        if (_selectedSpecies == "Другое")
          TextField(
            controller: _customSpeciesController,
            decoration: const InputDecoration(
              labelText: "Введите вид дерева",
              border: OutlineInputBorder(),
            ),
          ),
      ],
    );
  }
}

class _MetricChip extends StatelessWidget {
  final IconData icon;
  final String label;
  final String value;
  final bool isMuted;

  const _MetricChip({
    required this.icon,
    required this.label,
    required this.value,
    this.isMuted = false,
  });

  @override
  Widget build(BuildContext context) {
    final bg = isMuted ? const Color(0xFFF1F3F4) : const Color(0xFFE6F3EB);

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
      decoration: BoxDecoration(
        color: bg,
        borderRadius: BorderRadius.circular(999),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 18),
          const SizedBox(width: 6),
          Text(
            "$label: ",
            style: const TextStyle(fontWeight: FontWeight.w500),
          ),
          Text(value),
        ],
      ),
    );
  }
}
