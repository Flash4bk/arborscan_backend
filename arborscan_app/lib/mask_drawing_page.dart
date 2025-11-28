import 'dart:convert';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart'; // <<< ВАЖНО: даёт RenderRepaintBoundary

class MaskDrawingPage extends StatefulWidget {
  final Uint8List originalImageBytes;

  const MaskDrawingPage({
    super.key,
    required this.originalImageBytes,
  });

  @override
  State<MaskDrawingPage> createState() => _MaskDrawingPageState();
}

class _MaskDrawingPageState extends State<MaskDrawingPage> {
  final GlobalKey _canvasKey = GlobalKey();
  List<Offset?> _points = [];

  void _clear() {
    setState(() => _points = []);
  }

  Future<void> _saveMask() async {
    try {
      final boundary =
          _canvasKey.currentContext!.findRenderObject() as RenderRepaintBoundary;

      final ui.Image maskImage =
          await boundary.toImage(pixelRatio: 2.0); // можно 1.5–3.0

      final byteData =
          await maskImage.toByteData(format: ui.ImageByteFormat.png);

      final pngBytes = byteData!.buffer.asUint8List();
      final base64mask = base64Encode(pngBytes);

      Navigator.pop(context, base64mask);
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Ошибка сохранения маски: $e")),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final original = Image.memory(widget.originalImageBytes, fit: BoxFit.cover);

    return Scaffold(
      appBar: AppBar(
        title: const Text("Исправьте контур дерева"),
        actions: [
          IconButton(
            icon: const Icon(Icons.delete_outline),
            tooltip: "Очистить",
            onPressed: _clear,
          ),
          IconButton(
            icon: const Icon(Icons.check),
            tooltip: "Сохранить маску",
            onPressed: _saveMask,
          ),
        ],
      ),
      body: Center(
        child: InteractiveViewer(
          maxScale: 5,
          minScale: 1,
          child: AspectRatio(
            aspectRatio: 3 / 4,
            child: Stack(
              children: [
                Positioned.fill(child: original),

                // Рисуем поверх
                Positioned.fill(
                  child: RepaintBoundary(
                    key: _canvasKey,
                    child: CustomPaint(
                      painter: _MaskPainter(points: _points),
                      child: GestureDetector(
                        behavior: HitTestBehavior.translucent,
                        onPanUpdate: (details) {
                          setState(() {
                            final renderBox = _canvasKey.currentContext!
                                .findRenderObject() as RenderBox;
                            final localPos = renderBox
                                .globalToLocal(details.globalPosition);
                            _points.add(localPos);
                          });
                        },
                        onPanEnd: (_) {
                          setState(() => _points.add(null));
                        },
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _MaskPainter extends CustomPainter {
  final List<Offset?> points;

  _MaskPainter({required this.points});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.white
      ..strokeWidth = 12
      ..strokeCap = StrokeCap.round;

    for (int i = 0; i < points.length - 1; i++) {
      final p1 = points[i];
      final p2 = points[i + 1];
      if (p1 != null && p2 != null) {
        canvas.drawLine(p1, p2, paint);
      }
    }
  }

  @override
  bool shouldRepaint(_MaskPainter oldDelegate) => true;
}
