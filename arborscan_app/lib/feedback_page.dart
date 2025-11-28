import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';

import 'mask_drawing_page.dart';
import 'stick_page.dart'; // новый экран для выделения палки

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
  bool treeOk = true;
  bool stickOk = true;
  bool paramsOk = true;
  bool speciesOk = true;

  String selectedSpecies = "";
  final customSpeciesController = TextEditingController();

  Uint8List? userMask;
  double? correctedScale;

  late TextEditingController heightCtrl;
  late TextEditingController crownCtrl;
  late TextEditingController trunkCtrl;

  @override
  void initState() {
    super.initState();

    selectedSpecies = widget.species;

    heightCtrl = TextEditingController(
        text: (widget.heightM ?? 0).toStringAsFixed(2));
    crownCtrl = TextEditingController(
        text: (widget.crownWidthM ?? 0).toStringAsFixed(2));
    trunkCtrl = TextEditingController(
        text: (widget.trunkDiameterM ?? 0).toStringAsFixed(2));
  }

  @override
  Widget build(BuildContext context) {
    final originalBytes = base64Decode(widget.originalImageBase64);

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

          _buildSwitch("Дерево выделено правильно", treeOk,
              (v) => setState(() => treeOk = v)),
          _buildSwitch("Палка определена правильно", stickOk,
              (v) => setState(() => stickOk = v)),
          if (!stickOk)
            Padding(
              padding: const EdgeInsets.only(left: 12),
              child: FilledButton.icon(
                onPressed: _openStickFix,
                icon: const Icon(Icons.straighten_outlined),
                label: const Text("Перевыделить палку"),
              ),
            ),
          const SizedBox(height: 10),

          _buildSwitch("Параметры рассчитаны верно", paramsOk,
              (v) => setState(() => paramsOk = v)),
          if (!paramsOk) _buildParamsEditor(),

          const SizedBox(height: 10),

          _buildSwitch("Вид определён верно", speciesOk,
              (v) => setState(() => speciesOk = v)),
          if (!speciesOk) _buildSpeciesSelector(),

          const SizedBox(height: 20),

          FilledButton.icon(
            onPressed: _openMaskEditor,
            icon: const Icon(Icons.border_color),
            label: const Text("Нарисовать маску"),
          ),

          const SizedBox(height: 30),

          FilledButton(
            onPressed: _finish,
            child: const Padding(
              padding: EdgeInsets.symmetric(vertical: 14),
              child: Text("Отправить"),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSwitch(String title, bool value, Function(bool) onChanged) {
    return Card(
      child: SwitchListTile(
        title: Text(title),
        value: value,
        onChanged: onChanged,
      ),
    );
  }

  Widget _buildParamsEditor() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text("Исправьте параметры:", style: TextStyle(fontSize: 16)),
        const SizedBox(height: 8),
        _paramField("Высота (м)", heightCtrl),
        const SizedBox(height: 8),
        _paramField("Крона (м)", crownCtrl),
        const SizedBox(height: 8),
        _paramField("Ствол (м)", trunkCtrl),
      ],
    );
  }

  Widget _paramField(String label, TextEditingController c) {
    return TextField(
      controller: c,
      keyboardType: TextInputType.number,
      decoration: InputDecoration(
        border: const OutlineInputBorder(),
        labelText: label,
      ),
    );
  }

  Widget _buildSpeciesSelector() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          "Выберите правильный вид:",
          style: TextStyle(fontSize: 16),
        ),
        const SizedBox(height: 8),
        DropdownButtonFormField<String>(
          decoration: const InputDecoration(border: OutlineInputBorder()),
          value: selectedSpecies,
          items: const [
            DropdownMenuItem(value: "Береза", child: Text("Берёза")),
            DropdownMenuItem(value: "Дуб", child: Text("Дуб")),
            DropdownMenuItem(value: "Ель", child: Text("Ель")),
            DropdownMenuItem(value: "Сосна", child: Text("Сосна")),
            DropdownMenuItem(value: "Тополь", child: Text("Тополь")),
            DropdownMenuItem(value: "Другое", child: Text("Другое")),
          ],
          onChanged: (value) {
            setState(() => selectedSpecies = value ?? "Другое");
          },
        ),
        const SizedBox(height: 10),
        if (selectedSpecies == "Другое")
          TextField(
            controller: customSpeciesController,
            decoration: const InputDecoration(
              border: OutlineInputBorder(),
              labelText: "Введите вид",
            ),
          ),
      ],
    );
  }

  Future<void> _openMaskEditor() async {
    final mask = await Navigator.push<Uint8List?>(
      context,
      MaterialPageRoute(
        builder: (_) => MaskDrawingPage(imageBytes: base64Decode(widget.originalImageBase64)),
      ),
    );

    if (mask != null) {
      setState(() => userMask = mask);
    }
  }

  Future<void> _openStickFix() async {
    final newScale = await Navigator.push<double?>(
      context,
      MaterialPageRoute(
        builder: (_) => StickFixPage(
          imageBytes: base64Decode(widget.originalImageBase64),
        ),
      ),
    );

    if (newScale != null) {
      setState(() {
        correctedScale = newScale;
        // пересчитываем параметры
        heightCtrl.text =
            (newScale * (widget.heightM! / widget.scalePxToM!))
                .toStringAsFixed(2);
        crownCtrl.text =
            (newScale * (widget.crownWidthM! / widget.scalePxToM!))
                .toStringAsFixed(2);
        trunkCtrl.text =
            (newScale * (widget.trunkDiameterM! / widget.scalePxToM!))
                .toStringAsFixed(2);
      });
    }
  }

  void _finish() {
    Navigator.pop(context, {
      "tree_ok": treeOk,
      "stick_ok": stickOk,
      "params_ok": paramsOk,
      "species_ok": speciesOk,
      "correct_species":
          speciesOk ? null : (selectedSpecies == "Другое"
              ? customSpeciesController.text
              : selectedSpecies),
      "correct_height_m": paramsOk ? null : double.tryParse(heightCtrl.text),
      "correct_crown_width_m": paramsOk ? null : double.tryParse(crownCtrl.text),
      "correct_trunk_diameter_m": paramsOk ? null : double.tryParse(trunkCtrl.text),
      "correct_scale_px_to_m": correctedScale,
      "user_mask_base64": userMask != null ? base64Encode(userMask!) : null,
    });
  }
}
