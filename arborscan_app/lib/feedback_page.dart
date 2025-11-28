import 'dart:convert';
import 'package:flutter/material.dart';

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

  @override
  void initState() {
    super.initState();
    _selectedSpecies = widget.species;
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

          // --- TREE OK ---
          _buildSwitch(
            "Дерево выделено правильно",
            _treeOk,
            (v) => setState(() => _treeOk = v),
          ),

          // --- STICK OK ---
          _buildSwitch(
            "Палка определена правильно",
            _stickOk,
            (v) => setState(() => _stickOk = v),
          ),

          // --- PARAMS OK ---
          _buildSwitch(
            "Параметры рассчитаны верно",
            _paramsOk,
            (v) => setState(() => _paramsOk = v),
          ),

          // --- SPECIES OK ---
          _buildSwitch(
            "Вид определён верно",
            _speciesOk,
            (v) => setState(() => _speciesOk = v),
          ),

          const SizedBox(height: 10),

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
              });
            },
            child: const Padding(
              padding: EdgeInsets.symmetric(vertical: 14),
              child: Text("Продолжить"),
            ),
          ),
        ],
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
            onChanged: (text) {
              _selectedSpecies = text;
            },
          ),
      ],
    );
  }
}
