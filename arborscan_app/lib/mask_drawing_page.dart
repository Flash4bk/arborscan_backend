import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';

class MaskDrawingPage extends StatefulWidget {
  final Uint8List imageBytes;

  const MaskDrawingPage({
    super.key,
    required this.imageBytes,
  });

  @override
  State<MaskDrawingPage> createState() => _MaskDrawingPageState();
}

class _MaskDrawingPageState extends State<MaskDrawingPage> {
  final GlobalKey _paintKey = GlobalKey();

  List<Offset?> points = [];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Обведите дерево"),
        actions: [
          IconButton(
            icon: const Icon(Icons.check),
            onPressed: _saveMask,
          ),
        ],
      ),
      body: Center(
        child: RepaintBoundary(
          key: _paintKey,
          child: Stack(
            children: [
              Image.memory(widget.imageBytes), // <<< ИСХОДНОЕ ФОТО
              Positioned.fill(
                child: GestureDetector(
                  onPanUpdate: (details) {
                    final box = _paintKey.currentContext!.findRenderObject() as RenderBox;
                    final localPosition = box.globalToLocal(details.globalPosition);

                    setState(() {
                      points.add(localPosition);
                    });
                  },
                  onPanEnd: (_) {
                    points.add(null); // разрыв линии
                  },
                  child: CustomPaint(
                    painter: MaskPainter(points),
                    size: Size.infinite,
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Future<void> _saveMask() async {
    try {
      final boundary =
          _paintKey.currentContext!.findRenderObject() as RenderRepaintBoundary;

      final image = await boundary.toImage(pixelRatio: 1.0);

      final byteData = await image.toByteData(format: ui.ImageByteFormat.png);

      if (byteData == null) {
        throw Exception("Ошибка конвертации изображения");
      }

      final pngBytes = byteData.buffer.asUint8List();

      Navigator.pop(context, pngBytes); // вернуть mask PNG
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Ошибка: $e")),
      );
    }
  }
}

class MaskPainter extends CustomPainter {
  final List<Offset?> points;

  MaskPainter(this.points);

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = const Color(0xFF00FF00) // <<< КАК У ИИ
      ..strokeWidth = 3 // <<< КАК У ИИ
      ..strokeCap = StrokeCap.round
      ..style = PaintingStyle.stroke;

    for (int i = 0; i < points.length - 1; i++) {
      if (points[i] != null && points[i + 1] != null) {
        canvas.drawLine(points[i]!, points[i + 1]!, paint);
      }
    }
  }

  @override
  bool shouldRepaint(MaskPainter oldDelegate) => true;
}
