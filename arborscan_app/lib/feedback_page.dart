import 'dart:convert';
import 'dart:typed_data';

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

  late String _currentSpecies;
  final TextEditingController _speciesController = TextEditingController();

  double? _heightM;
  double? _crownWidthM;
  double? _trunkDiameterM;
  double? _scalePxToM;

  String? _userMaskBase64;

  @override
  void initState() {
    super.initState();
    _currentSpecies = widget.species;
    _heightM = widget.heightM;
    _crownWidthM = widget.crownWidthM;
    _trunkDiameterM = widget.trunkDiameterM;
    _scalePxToM = widget.scalePxToM;
  }

  @override
  void dispose() {
    _speciesController.dispose();
    super.dispose();
  }

  String _formatDouble(double? v, {String suffix = "м"}) {
    if (v == null) return "—";
    return "${v.toStringAsFixed(2)} $suffix";
  }

  Future<void> _openMaskDrawing() async {
    final bytes = base64Decode(widget.originalImageBase64);

    final String? maskB64 = await Navigator.push<String?>(
      context,
      MaterialPageRoute(
        builder: (_) => MaskDrawingPage(
          imageBytes: bytes,
        ),
      ),
    );

    if (maskB64 != null) {
      setState(() {
        _userMaskBase64 = maskB64;
        _treeOk = false; // раз пользователь рисовал маску — значит ИИ был не идеален
      });
    }
  }

  Future<void> _openStickPage() async {
    final bytes = base64Decode(widget.originalImageBase64);

    final result = await Navigator.push<Map<String, dynamic>?>(
      context,
      MaterialPageRoute(
        builder: (_) => StickPage(
          imageBytes: bytes,
          initialHeightM: _heightM,
          initialCrownWidthM: _crownWidthM,
          initialTrunkDiameterM: _trunkDiameterM,
          initialScalePxToM: _scalePxToM,
        ),
      ),
    );

    if (result != null) {
      setState(() {
        _heightM = result["height_m"] as double?;
        _crownWidthM = result["crown_width_m"] as double?;
        _trunkDiameterM = result["trunk_diameter_m"] as double?;
        _scalePxToM = result["scale_px_to_m"] as double?;
        _stickOk = result["stick_ok"] as bool? ?? _stickOk;
        _paramsOk = result["params_ok"] as bool? ?? _paramsOk;
      });
    }
  }

  void _onSave() {
    String? correctSpecies;
    if (!_speciesOk) {
      if (_speciesController.text.trim().isNotEmpty) {
        correctSpecies = _speciesController.text.trim();
      } else {
        correctSpecies = _currentSpecies;
      }
    }

    Navigator.pop<Map<String, dynamic>>(context, {
      "tree_ok": _treeOk,
      "stick_ok": _stickOk,
      "params_ok": _paramsOk,
      "species_ok": _speciesOk,
      "correct_species": correctSpecies,
      "use_for_training": _useForTraining,
      "user_mask_base64": _userMaskBase64,
      // при желании сюда можно добавить исправленные параметры,
      // а затем расширить серверную модель FeedbackRequest
      // "fixed_height_m": _heightM,
      // ...
    });
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    Uint8List headerBytes;
    if (widget.annotatedImageBase64 != null &&
        widget.annotatedImageBase64!.isNotEmpty) {
      headerBytes = base64Decode(widget.annotatedImageBase64!);
    } else {
      headerBytes = base64Decode(widget.originalImageBase64);
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text("Проверка анализа"),
        actions: [
          IconButton(
            onPressed: _onSave,
            icon: const Icon(Icons.check),
          )
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          ClipRRect(
            borderRadius: BorderRadius.circular(20),
            child: Image.memory(headerBytes),
          ),
          const SizedBox(height: 16),

          // Блок текущих параметров
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
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
                    spacing: 12,
                    runSpacing: 8,
                    children: [
                      _metricChip(
                        icon: Icons.park,
                        label: "Вид",
                        value: _currentSpecies,
                      ),
                      _metricChip(
                        icon: Icons.height,
                        label: "Высота",
                        value: _formatDouble(_heightM),
                      ),
                      _metricChip(
                        icon: Icons.landscape_outlined,
                        label: "Крона",
                        value: _formatDouble(_crownWidthM),
                      ),
                      _metricChip(
                        icon: Icons.circle_outlined,
                        label: "Ствол",
                        value: _formatDouble(_trunkDiameterM),
                      ),
                      _metricChip(
                        icon: Icons.straighten,
                        label: "Масштаб",
                        value: _scalePxToM == null
                            ? "—"
                            : "1 px ≈ ${_scalePxToM!.toStringAsFixed(4)} м",
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 16),

          // Свитчи корректности
          _switchCard(
            title: "Дерево выделено правильно",
            value: _treeOk,
            onChanged: (v) => setState(() => _treeOk = v),
          ),
          _switchCard(
            title: "Палка определена правильно",
            value: _stickOk,
            onChanged: (v) => setState(() => _stickOk = v),
          ),
          _switchCard(
            title: "Параметры рассчитаны верно",
            value: _paramsOk,
            onChanged: (v) => setState(() => _paramsOk = v),
          ),
          _switchCard(
            title: "Вид определён верно",
            value: _speciesOk,
            onChanged: (v) => setState(() => _speciesOk = v),
          ),

          const SizedBox(height: 12),

          // Исправление вида
          if (!_speciesOk)
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      "Уточните вид дерева",
                      style: theme.textTheme.titleMedium,
                    ),
                    const SizedBox(height: 8),
                    TextField(
                      controller: _speciesController
                        ..text = _speciesController.text.isEmpty
                            ? _currentSpecies
                            : _speciesController.text,
                      decoration: const InputDecoration(
                        border: OutlineInputBorder(),
                        labelText: "Введите корректный вид",
                      ),
                    ),
                  ],
                ),
              ),
            ),

          const SizedBox(height: 12),

          // Кнопки «нарисовать маску» и «палка / параметры»
          Row(
            children: [
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: _openMaskDrawing,
                  icon: const Icon(Icons.brush),
                  label: const Text("Нарисовать маску"),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: _openStickPage,
                  icon: const Icon(Icons.straighten),
                  label: const Text("Палка / параметры"),
                ),
              ),
            ],
          ),

          const SizedBox(height: 16),

          // Флаг использования в обучении
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Row(
                children: [
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          "Использовать этот пример для обучения",
                          style: theme.textTheme.titleMedium
                              ?.copyWith(fontWeight: FontWeight.w600),
                        ),
                        const SizedBox(height: 4),
                        Text(
                          "При включении изображение и ваши правки будут сохранены как доверенный пример.",
                          style: theme.textTheme.bodySmall
                              ?.copyWith(color: Colors.black54),
                        ),
                      ],
                    ),
                  ),
                  Switch(
                    value: _useForTraining,
                    onChanged: (v) =>
                        setState(() => _useForTraining = v),
                  ),
                ],
              ),
            ),
          ),

          const SizedBox(height: 24),
          SizedBox(
            width: double.infinity,
            child: FilledButton(
              onPressed: _onSave,
              child: const Padding(
                padding: EdgeInsets.symmetric(vertical: 14),
                child: Text("Сохранить и отправить"),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _switchCard({
    required String title,
    required bool value,
    required ValueChanged<bool> onChanged,
  }) {
    return Card(
      child: SwitchListTile(
        title: Text(title),
        value: value,
        onChanged: onChanged,
      ),
    );
  }

  Widget _metricChip({
    required IconData icon,
    required String label,
    required String value,
  }) {
    return Chip(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      avatar: Icon(icon, size: 18),
      label: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(label, style: const TextStyle(fontSize: 11)),
          Text(
            value,
            style: const TextStyle(fontWeight: FontWeight.w600),
          ),
        ],
      ),
    );
  }
}
