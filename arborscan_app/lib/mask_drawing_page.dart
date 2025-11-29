import 'dart:convert';
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

  final List<Offset?> _points = [];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Рисование маски"),
        actions: [
          IconButton(
            icon: const Icon(Icons.check),
            onPressed: _onDone,
          )
        ],
      ),
      body: LayoutBuilder(
        builder: (context, constraints) {
          return Center(
            child: AspectRatio(
              aspectRatio: 3 / 4,
              child: RepaintBoundary(
                key: _paintKey,
                child: GestureDetector(
                  onPanStart: (details) {
                    setState(() {
                      _points.add(details.localPosition);
                    });
                  },
                  onPanUpdate: (details) {
                    setState(() {
                      _points.add(details.localPosition);
                    });
                  },
                  onPanEnd: (_) {
                    setState(() {
                      _points.add(null); // разделитель между линиями
                    });
                  },
                  child: Stack(
                    fit: StackFit.expand,
                    children: [
                      Image.memory(
                        widget.imageBytes,
                        fit: BoxFit.cover,
                      ),
                      CustomPaint(
                        painter: _MaskPainter(points: _points),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          );
        },
      ),
    );
  }

  Future<void> _onDone() async {
    if (_points.isEmpty) {
      Navigator.pop<String?>(context, null);
      return;
    }

    try {
      final boundary = _paintKey.currentContext!
          .findRenderObject() as RenderRepaintBoundary;
      final ui.Image image = await boundary.toImage(pixelRatio: 1.5);
      final byteData =
          await image.toByteData(format: ui.ImageByteFormat.png);
      final bytes = byteData!.buffer.asUint8List();
      final b64 = base64Encode(bytes);
      Navigator.pop<String>(context, b64);
    } catch (e) {
      Navigator.pop<String?>(context, null);
    }
  }
}

class _MaskPainter extends CustomPainter {
  final List<Offset?> points;

  _MaskPainter({required this.points});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.greenAccent.withOpacity(0.8)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 5
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round;

    for (int i = 0; i < points.length - 1; i++) {
      final p1 = points[i];
      final p2 = points[i + 1];
      if (p1 != null && p2 != null) {
        canvas.drawLine(p1, p2, paint);
      }
    }
  }

  @override
  bool shouldRepaint(covariant _MaskPainter oldDelegate) =>
      oldDelegate.points != points;
}
