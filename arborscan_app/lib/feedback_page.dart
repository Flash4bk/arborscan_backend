import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'mask_drawing_page.dart';

class FeedbackPage extends StatefulWidget {
  final String analysisId;
  final String species;
  final double? heightM;
  final double? crownWidthM;
  final double? trunkDiameterM;
  final String annotatedImageBase64;

  /// Обязательно — оригинальное фото
  final String originalImageBase64;

  const FeedbackPage({
    super.key,
    required this.analysisId,
    required this.species,
    required this.heightM,
    required this.crownWidthM,
    required this.trunkDiameterM,
    required this.annotatedImageBase64,
    required this.originalImageBase64,
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

  Uint8List? userMaskBytes;

  @override
  void initState() {
    super.initState();
    _selectedSpecies = widget.species;
  }

  @override
  Widget build(BuildContext context) {
    final annotatedBytes = base64Decode(widget.annotatedImageBase64);

    return Scaffold(
      appBar: AppBar(title: const Text("Проверка анализа")),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          ClipRRect(
            borderRadius: BorderRadius.circular(16),
            child: Image.memory(annotatedBytes),
          ),
          const SizedBox(height: 20),

          FilledButton.icon(
            onPressed: () async {
              final originalBytes =
                  base64Decode(widget.originalImageBase64);

              final mask = await Navigator.push<Uint8List?>(
                context,
                MaterialPageRoute(
                  builder: (_) => MaskDrawingPage(
                    originalImageBytes: originalBytes,
                  ),
                ),
              );

              if (mask != null) {
                setState(() => userMaskBytes = mask);
              }
            },
            icon: const Icon(Icons.brush),
            label: const Text("Нарисовать маску"),
          ),

          if (userMaskBytes != null) ...[
            const SizedBox(height: 12),
            const Text("Ваша маска:", style: TextStyle(fontWeight: FontWeight.w600)),
            const SizedBox(height: 8),
            ClipRRect(
              borderRadius: BorderRadius.circular(12),
              child: Image.memory(userMaskBytes!),
            ),
          ],

          const SizedBox(height: 20),

          _buildSwitch("Дерево выделено правильно", _treeOk, (v) => setState(() => _treeOk = v)),
          _buildSwitch("Палка определена правильно", _stickOk, (v) => setState(() => _stickOk = v)),
          _buildSwitch("Параметры рассчитаны верно", _paramsOk, (v) => setState(() => _paramsOk = v)),
          _buildSwitch("Вид определён верно", _speciesOk, (v) => setState(() => _speciesOk = v)),

          if (!_speciesOk) _buildSpeciesSelector(),

          const SizedBox(height: 25),

          FilledButton(
            onPressed: () {
              Navigator.pop(context, {
                "tree_ok": _treeOk,
                "stick_ok": _stickOk,
                "params_ok": _paramsOk,
                "species_ok": _speciesOk,
                "correct_species": _speciesOk ? null : _selectedSpecies,
                "user_mask_base64":
                    userMaskBytes != null ? base64Encode(userMaskBytes!) : null,
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

  Widget _buildSwitch(String title, bool v, Function(bool) onChanged) {
    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      child: SwitchListTile(
        value: v,
        onChanged: onChanged,
        title: Text(title),
      ),
    );
  }

  Widget _buildSpeciesSelector() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text("Выберите правильный вид:", style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600)),
        const SizedBox(height: 8),
        DropdownButtonFormField<String>(
          decoration: const InputDecoration(border: OutlineInputBorder()),
          value: _selectedSpecies,
          items: const [
            DropdownMenuItem(value: "Береза", child: Text("Берёза")),
            DropdownMenuItem(value: "Дуб", child: Text("Дуб")),
            DropdownMenuItem(value: "Ель", child: Text("Ель")),
            DropdownMenuItem(value: "Сосна", child: Text("Сосна")),
            DropdownMenuItem(value: "Тополь", child: Text("Тополь")),
            DropdownMenuItem(value: "Другое", child: Text("Другое")),
          ],
          onChanged: (v) => setState(() => _selectedSpecies = v ?? "Другое"),
        ),
        const SizedBox(height: 10),
        if (_selectedSpecies == "Другое")
          TextField(
            controller: _customSpeciesController,
            decoration: const InputDecoration(
              border: OutlineInputBorder(),
              labelText: "Введите вид",
            ),
            onChanged: (txt) => _selectedSpecies = txt,
          ),
      ],
    );
  }
}
