import 'dart:typed_data';

import 'package:flutter/material.dart';

import 'mask_drawing_page.dart';

class StickPage extends StatefulWidget {
  final Uint8List imageBytes;

  final double? heightM;
  final double? crownWidthM;
  final double? trunkDiameterM;
  final double? scalePxToM;

  const StickPage({
    super.key,
    required this.imageBytes,
    required this.heightM,
    required this.crownWidthM,
    required this.trunkDiameterM,
    required this.scalePxToM,
  });

  @override
  State<StickPage> createState() => _StickPageState();
}

class _StickPageState extends State<StickPage> {
  late TextEditingController _heightCtrl;
  late TextEditingController _crownCtrl;
  late TextEditingController _trunkCtrl;
  late TextEditingController _scaleCtrl;

  String? _stickMaskBase64; // если захочешь отдельную маску палки

  @override
  void initState() {
    super.initState();
    _heightCtrl = TextEditingController(
      text: widget.heightM?.toStringAsFixed(2) ?? "",
    );
    _crownCtrl = TextEditingController(
      text: widget.crownWidthM?.toStringAsFixed(2) ?? "",
    );
    _trunkCtrl = TextEditingController(
      text: widget.trunkDiameterM?.toStringAsFixed(2) ?? "",
    );
    _scaleCtrl = TextEditingController(
      text: widget.scalePxToM?.toStringAsFixed(4) ?? "",
    );
  }

  @override
  void dispose() {
    _heightCtrl.dispose();
    _crownCtrl.dispose();
    _trunkCtrl.dispose();
    _scaleCtrl.dispose();
    super.dispose();
  }

  Future<void> _openStickMaskDrawing() async {
    final bytes = await Navigator.push<Uint8List?>(
      context,
      MaterialPageRoute(
        builder: (_) => MaskDrawingPage(
          imageBytes: widget.imageBytes,
          title: "Обведите палку",
          hint: "Аккуратно обведите мерную палку.\n"
              "Эта маска поможет точнее определять масштаб.",
        ),
      ),
    );

    if (bytes != null) {
      setState(() {
        _stickMaskBase64 = base64Encode(bytes);
      });
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Маска палки сохранена.")),
      );
    }
  }

  void _onSave() {
    double? parseD(String s) {
      s = s.replaceAll(",", ".").trim();
      if (s.isEmpty) return null;
      return double.tryParse(s);
    }

    final height = parseD(_heightCtrl.text);
    final crown = parseD(_crownCtrl.text);
    final trunk = parseD(_trunkCtrl.text);
    final scale = parseD(_scaleCtrl.text);

    Navigator.pop<Map<String, double?>>(context, {
      "height_m": height,
      "crown_width_m": crown,
      "trunk_diameter_m": trunk,
      "scale_px_to_m": scale,
      // маску палки сейчас НИЖЕ не возвращаем в main,
      // можно добавить через FeedbackPage при необходимости
    });
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(
        title: const Text("Коррекция палки и параметров"),
      ),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(16, 12, 16, 24),
        children: [
          ClipRRect(
            borderRadius: BorderRadius.circular(24),
            child: Image.memory(
              widget.imageBytes,
              fit: BoxFit.cover,
            ),
          ),
          const SizedBox(height: 16),
          Text(
            "Проверьте корректность параметров:",
            style: theme.textTheme.titleMedium
                ?.copyWith(fontWeight: FontWeight.w600),
          ),
          const SizedBox(height: 12),
          _buildNumberField(
            controller: _heightCtrl,
            label: "Высота дерева, м",
            helper: "Если измерения неверны — укажите правильную высоту.",
          ),
          const SizedBox(height: 10),
          _buildNumberField(
            controller: _crownCtrl,
            label: "Ширина кроны, м",
            helper: "Максимальная ширина кроны.",
          ),
          const SizedBox(height: 10),
          _buildNumberField(
            controller: _trunkCtrl,
            label: "Диаметр ствола, м",
            helper: "Диаметр ствола на уровне груди.",
          ),
          const SizedBox(height: 10),
          _buildNumberField(
            controller: _scaleCtrl,
            label: "Масштаб (1 px ≈ X м)",
            helper: "При необходимости скорректируйте масштаб.",
          ),
          const SizedBox(height: 16),
          OutlinedButton.icon(
            onPressed: _openStickMaskDrawing,
            icon: const Icon(Icons.brush_outlined),
            label: const Text("Нарисовать маску палки"),
            style: OutlinedButton.styleFrom(
              padding: const EdgeInsets.symmetric(vertical: 12),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(999),
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
                child: Text("Сохранить"),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildNumberField({
    required TextEditingController controller,
    required String label,
    required String helper,
  }) {
    return TextField(
      controller: controller,
      keyboardType:
          const TextInputType.numberWithOptions(decimal: true, signed: false),
      decoration: InputDecoration(
        labelText: label,
        helperText: helper,
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
        ),
      ),
    );
  }
}
