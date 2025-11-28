import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'mask_drawing_page.dart';

class FeedbackPage extends StatefulWidget {
  final String analysisId;

  /// Важно: оригинальное фото (НЕ аннотированное)
  final String originalImageBase64;

  final String species;
  final double? heightM;
  final double? crownWidthM;
  final double? trunkDiameterM;

  const FeedbackPage({
    super.key,
    required this.analysisId,
    required this.originalImageBase64,
    required this.species,
    required this.heightM,
    required this.crownWidthM,
    required this.trunkDiameterM,
  });

  @override
  State<FeedbackPage> createState() => _FeedbackPageState();
}

class _FeedbackPageState extends State<FeedbackPage> {
  bool _treeOk = true;
  bool _stickOk = true;
  bool _paramsOk = true;
  bool _speciesOk = true;

  Uint8List? _userMaskBytes;
  String _selectedSpecies = "";
  final TextEditingController _customSpeciesController =
      TextEditingController();

  @override
  void initState() {
    super.initState();
    _selectedSpecies = widget.species;
  }

  @override
  Widget build(BuildContext context) {
    final Uint8List originalBytes =
        base64Decode(widget.originalImageBase64);

    return Scaffold(
      appBar: AppBar(
        title: const Text("Проверка анализа"),
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          ClipRRect(
            borderRadius: BorderRadius.circular(16),
            child: Image.memory(originalBytes),
          ),

          const SizedBox(height: 20),

          _buildSwitch("Дерево выделено правильно", _treeOk,
              (v) => setState(() => _treeOk = v)),
          _buildSwitch("Палка определена правильно", _stickOk,
              (v) => setState(() => _stickOk = v)),
          _buildSwitch("Параметры рассчитаны верно", _paramsOk,
              (v) => setState(() => _paramsOk = v)),
          _buildSwitch("Вид определён верно", _speciesOk,
              (v) => setState(() => _speciesOk = v)),

          if (!_speciesOk) _buildSpeciesSelector(),

          const SizedBox(height: 20),

          OutlinedButton.icon(
            onPressed: () async {
              final mask = await Navigator.push<Uint8List?>(
                context,
                MaterialPageRoute(
                  builder: (_) => MaskDrawingPage(
                    imageBytes: originalBytes,
                  ),
                ),
              );

              if (mask != null) {
                setState(() {
                  _userMaskBytes = mask;
                });
              }
            },
            icon: const Icon(Icons.brush),
            label: const Text("Нарисовать маску вручную"),
          ),

          if (_userMaskBytes != null)
            Padding(
              padding: const EdgeInsets.only(top: 10),
              child: Text(
                "Маска добавлена ✓",
                style: TextStyle(
                  color: Colors.green.shade700,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),

          const SizedBox(height: 30),

          FilledButton(
            onPressed: () {
              Navigator.pop(context, {
                "tree_ok": _treeOk,
                "stick_ok": _stickOk,
                "params_ok": _paramsOk,
                "species_ok": _speciesOk,
                "correct_species":
                    _speciesOk ? null : _selectedSpecies,
                "user_mask_base64": _userMaskBytes == null
                    ? null
                    : base64Encode(_userMaskBytes!),
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

  Widget _buildSwitch(
      String title, bool value, Function(bool) onChanged) {
    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      child: SwitchListTile(
        title: Text(title),
        value: value,
        onChanged: onChanged,
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
          onChanged: (v) {
            setState(() => _selectedSpecies = v ?? "Другое");
          },
        ),
        if (_selectedSpecies == "Другое")
          Padding(
            padding: const EdgeInsets.only(top: 10),
            child: TextField(
              controller: _customSpeciesController,
              decoration: const InputDecoration(
                labelText: "Введите вид дерева",
                border: OutlineInputBorder(),
              ),
              onChanged: (v) => _selectedSpecies = v,
            ),
          ),
      ],
    );
  }
}
