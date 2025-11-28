import 'dart:convert';
import 'package:flutter/material.dart';
import 'mask_drawing_page.dart';

class FeedbackPage extends StatefulWidget {
  final String analysisId;
  final String species;
  final double? heightM;
  final double? crownWidthM;
  final double? trunkDiameterM;
  final String annotatedImageBase64;

  const FeedbackPage({
    super.key,
    required this.analysisId,
    required this.species,
    required this.heightM,
    required this.crownWidthM,
    required this.trunkDiameterM,
    required this.annotatedImageBase64,
  });

  @override
  State<FeedbackPage> createState() => _FeedbackPageState();
}

class _FeedbackPageState extends State<FeedbackPage> {
  bool _treeOk = true;
  bool _stickOk = true;
  bool _paramsOk = true;
  bool _speciesOk = true;

  String _selectedSpecies = "";
  final TextEditingController _customSpeciesController = TextEditingController();

  // --- Controllers for parameters ---
  final TextEditingController _heightController = TextEditingController();
  final TextEditingController _crownController = TextEditingController();
  final TextEditingController _trunkController = TextEditingController();

  // --- Mask ---
  String? _maskBase64;
  bool get _maskAdded => _maskBase64 != null;

  @override
  void initState() {
    super.initState();
    _selectedSpecies = widget.species;

    _heightController.text =
        widget.heightM != null ? widget.heightM!.toStringAsFixed(2) : "";
    _crownController.text =
        widget.crownWidthM != null ? widget.crownWidthM!.toStringAsFixed(2) : "";
    _trunkController.text =
        widget.trunkDiameterM != null ? widget.trunkDiameterM!.toStringAsFixed(2) : "";
  }

  Future<void> _openMaskDrawing() async {
    final originalBytes = base64Decode(widget.annotatedImageBase64);

    final mask = await Navigator.push<String?>(
      context,
      MaterialPageRoute(
        builder: (_) => MaskDrawingPage(
          originalImageBytes: originalBytes,
        ),
      ),
    );

    if (mask != null) {
      setState(() {
        _maskBase64 = mask;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final imgBytes = base64Decode(widget.annotatedImageBase64);

    return Scaffold(
      appBar: AppBar(
        title: const Text("Проверка анализа"),
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          ClipRRect(
            borderRadius: BorderRadius.circular(16),
            child: Image.memory(imgBytes),
          ),

          const SizedBox(height: 20),

          // TREE OK
          _buildSwitch(
            "Дерево выделено правильно",
            _treeOk,
            (v) => setState(() => _treeOk = v),
          ),

          // ==== Кнопка рисования маски, если дерево выделено неверно ====
          if (!_treeOk) _buildMaskButton(),

          // STICK OK
          _buildSwitch(
            "Палка определена правильно",
            _stickOk,
            (v) => setState(() => _stickOk = v),
          ),

          // PARAMS OK
          _buildSwitch(
            "Параметры рассчитаны верно",
            _paramsOk,
            (v) => setState(() => _paramsOk = v),
          ),

          if (!_paramsOk) _buildParamsCorrection(),

          // SPECIES OK
          _buildSwitch(
            "Вид определён верно",
            _speciesOk,
            (v) => setState(() => _speciesOk = v),
          ),

          if (!_speciesOk) _buildSpeciesSelector(),

          const SizedBox(height: 25),

          FilledButton(
            onPressed: () {
              Navigator.pop(context, {
                "tree_ok": _treeOk,
                "stick_ok": _stickOk,
                "params_ok": _paramsOk,
                "species_ok": _speciesOk,

                "correct_species":
                    _speciesOk ? null : _selectedSpecies.trim(),

                "correct_height":
                    _paramsOk ? null : double.tryParse(_heightController.text),
                "correct_crown":
                    _paramsOk ? null : double.tryParse(_crownController.text),
                "correct_trunk":
                    _paramsOk ? null : double.tryParse(_trunkController.text),

                // ← маска
                "user_mask_base64": _maskBase64,
              });
            },
            child: const Padding(
              padding: EdgeInsets.symmetric(vertical: 14),
              child: Text("Отправить"),
            ),
          ),
        ],
      ),
    );
  }

  // Mask button
  Widget _buildMaskButton() {
    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              "Исправьте контур дерева:",
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
            ),
            const SizedBox(height: 12),

            FilledButton.icon(
              onPressed: _openMaskDrawing,
              icon: const Icon(Icons.brush),
              label: const Text("Нарисовать маску"),
            ),

            if (_maskAdded) ...[
              const SizedBox(height: 8),
              Row(
                children: const [
                  Icon(Icons.check_circle, color: Colors.green),
                  SizedBox(width: 6),
                  Text(
                    "Маска добавлена",
                    style: TextStyle(color: Colors.green),
                  ),
                ],
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildSwitch(String title, bool value, Function(bool) onChanged) {
    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      child: SwitchListTile(
        title: Text(title),
        value: value,
        onChanged: onChanged,
      ),
    );
  }

  Widget _buildParamsCorrection() {
    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              "Исправьте параметры:",
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
            ),
            const SizedBox(height: 12),
            _buildNumberField("Высота (м)", _heightController),
            const SizedBox(height: 12),
            _buildNumberField("Ширина кроны (м)", _crownController),
            const SizedBox(height: 12),
            _buildNumberField("Диаметр ствола (м)", _trunkController),
          ],
        ),
      ),
    );
  }

  Widget _buildNumberField(String label, TextEditingController controller) {
    return TextField(
      controller: controller,
      keyboardType:
          const TextInputType.numberWithOptions(decimal: true, signed: false),
      decoration: InputDecoration(
        labelText: label,
        border: const OutlineInputBorder(),
      ),
    );
  }

  Widget _buildSpeciesSelector() {
    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
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
                onChanged: (text) {
                  _selectedSpecies = text;
                },
              ),
          ],
        ),
      ),
    );
  }
}
