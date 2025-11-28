import 'dart:convert';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';

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
  List<Offset?> _points = [];
  final GlobalKey _canvasKey = GlobalKey();

  // Очистка холста
  void _clear() {
    setState(() => _points = []);
  }

  // Сохранение в PNG + Base64
  Future<void> _saveMask() async {
    try {
      final boundary = _canvasKey.currentContext!
          .findRenderObject() as RenderRepaintBoundary;

      final ui.Image maskImage =
          await boundary.toImage(pixelRatio: 1.5);

      final byteData =
          await maskImage.toByteData(format: ui.ImageByteFormat.png);

      final pngBytes = byteData!.buffer.asUint8List();
      final base64mask = base64Encode(pngBytes);

      Navigator.pop(context, base64mask);
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Ошибка сохранения маски: $e")),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final original = Image.memory(widget.originalImageBytes);

    return Scaffold(
      appBar: AppBar(
        title: const Text("Исправьте контур дерева"),
        actions: [
          IconButton(
            icon: const Icon(Icons.delete_outline),
            onPressed: _clear,
          ),
          IconButton(
            icon: const Icon(Icons.check),
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

                // Рисование поверх Canvas
                Positioned.fill(
                  child: RepaintBoundary(
                    key: _canvasKey,
                    child: CustomPaint(
                      painter: _MaskPainter(points: _points),
                      child: GestureDetector(
                        onPanUpdate: (details) {
                          setState(() {
                            final box = context.findRenderObject() as RenderBox;
                            final localPos =
                                box.globalToLocal(details.globalPosition);
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

// Painter — рисует маску
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
      if (points[i] != null && points[i + 1] != null) {
        canvas.drawLine(points[i]!, points[i + 1]!, paint);
      }
    }
  }

  @override
  bool shouldRepaint(_MaskPainter oldDelegate) => true;
}
