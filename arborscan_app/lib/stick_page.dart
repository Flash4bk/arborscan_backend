import 'dart:convert';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';

class StickPage extends StatefulWidget {
  final Uint8List imageBytes;

  final double? initialHeightM;
  final double? initialCrownWidthM;
  final double? initialTrunkDiameterM;
  final double? initialScalePxToM;

  const StickPage({
    super.key,
    required this.imageBytes,
    this.initialHeightM,
    this.initialCrownWidthM,
    this.initialTrunkDiameterM,
    this.initialScalePxToM,
  });

  @override
  State<StickPage> createState() => _StickPageState();
}

class _StickPageState extends State<StickPage> {
  final GlobalKey _paintKey = GlobalKey();

  Offset? _start;
  Offset? _end;

  late TextEditingController _heightCtrl;
  late TextEditingController _crownCtrl;
  late TextEditingController _trunkCtrl;
  late TextEditingController _scaleCtrl;

  bool _stickOk = true;
  bool _paramsOk = true;

  @override
  void initState() {
    super.initState();
    _heightCtrl = TextEditingController(
      text: widget.initialHeightM?.toStringAsFixed(2),
    );
    _crownCtrl = TextEditingController(
      text: widget.initialCrownWidthM?.toStringAsFixed(2),
    );
    _trunkCtrl = TextEditingController(
      text: widget.initialTrunkDiameterM?.toStringAsFixed(2),
    );
    _scaleCtrl = TextEditingController(
      text: widget.initialScalePxToM?.toStringAsFixed(4),
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

  double? _parseDouble(String text) {
    if (text.trim().isEmpty) return null;
    return double.tryParse(text.replaceAll(',', '.'));
  }

  void _onSave() {
    Navigator.pop<Map<String, dynamic>>(context, {
      "height_m": _parseDouble(_heightCtrl.text),
      "crown_width_m": _parseDouble(_crownCtrl.text),
      "trunk_diameter_m": _parseDouble(_trunkCtrl.text),
      "scale_px_to_m": _parseDouble(_scaleCtrl.text),
      "stick_ok": _stickOk,
      "params_ok": _paramsOk,
      // при необходимости можно добавить mask палки, как на дереве
    });
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(
        title: const Text("Коррекция палки и параметров"),
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
          RepaintBoundary(
            key: _paintKey,
            child: GestureDetector(
              onPanStart: (details) {
                setState(() {
                  _start = details.localPosition;
                  _end = details.localPosition;
                });
              },
              onPanUpdate: (details) {
                setState(() {
                  _end = details.localPosition;
                });
              },
              onPanEnd: (_) {
                // линия зафиксирована
              },
              child: AspectRatio(
                aspectRatio: 3 / 4,
                child: Stack(
                  fit: StackFit.expand,
                  children: [
                    Image.memory(
                      widget.imageBytes,
                      fit: BoxFit.cover,
                    ),
                    CustomPaint(
                      painter: _StickPainter(start: _start, end: _end),
                    ),
                  ],
                ),
              ),
            ),
          ),
          const SizedBox(height: 16),

          Text(
            "Проверьте корректность параметров:",
            style: theme.textTheme.titleMedium
                ?.copyWith(fontWeight: FontWeight.w600),
          ),
          const SizedBox(height: 8),

          Card(
            child: SwitchListTile(
              title: const Text("Палка определена правильно"),
              value: _stickOk,
              onChanged: (v) => setState(() => _stickOk = v),
            ),
          ),
          Card(
            child: SwitchListTile(
              title: const Text("Высота / крона / ствол рассчитаны верно"),
              value: _paramsOk,
              onChanged: (v) => setState(() => _paramsOk = v),
            ),
          ),

          const SizedBox(height: 8),

          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                children: [
                  _numberField(
                    label: "Высота, м",
                    controller: _heightCtrl,
                  ),
                  const SizedBox(height: 12),
                  _numberField(
                    label: "Крона, м",
                    controller: _crownCtrl,
                  ),
                  const SizedBox(height: 12),
                  _numberField(
                    label: "Диаметр ствола, м",
                    controller: _trunkCtrl,
                  ),
                  const SizedBox(height: 12),
                  _numberField(
                    label: "Масштаб (м за 1 пиксель)",
                    controller: _scaleCtrl,
                    hint: "например 0.0182",
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
                child: Text("Сохранить"),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _numberField({
    required String label,
    required TextEditingController controller,
    String? hint,
  }) {
    return TextField(
      controller: controller,
      keyboardType:
          const TextInputType.numberWithOptions(decimal: true, signed: false),
      decoration: InputDecoration(
        labelText: label,
        hintText: hint,
        border: const OutlineInputBorder(),
      ),
    );
  }
}

class _StickPainter extends CustomPainter {
  final Offset? start;
  final Offset? end;

  _StickPainter({this.start, this.end});

  @override
  void paint(Canvas canvas, Size size) {
    if (start == null || end == null) return;

    final paint = Paint()
      ..color = Colors.redAccent
      ..strokeWidth = 5
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;

    canvas.drawLine(start!, end!, paint);
  }

  @override
  bool shouldRepaint(covariant _StickPainter oldDelegate) =>
      oldDelegate.start != start || oldDelegate.end != end;
}
