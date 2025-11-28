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

  List<Offset> _points = [];
  ui.Image? _image;

  @override
  void initState() {
    super.initState();
    _loadUiImage();
  }

  Future<void> _loadUiImage() async {
    final data = await decodeImageFromList(widget.imageBytes);
    setState(() => _image = data);
  }

  Future<Uint8List?> _exportMask() async {
    try {
      final boundary = _paintKey.currentContext!.findRenderObject()
          as RenderRepaintBoundary;

      final img = await boundary.toImage(pixelRatio: 1.0);
      final byteData = await img.toByteData(format: ui.ImageByteFormat.png);

      return byteData?.buffer.asUint8List();
    } catch (e) {
      print("Export mask error: $e");
      return null;
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_image == null) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text("Рисование маски"),
        actions: [
          IconButton(
            icon: const Icon(Icons.check),
            onPressed: () async {
              final maskBytes = await _exportMask();
              Navigator.pop(context, maskBytes);
            },
          )
        ],
      ),
      body: Center(
        child: RepaintBoundary(
          key: _paintKey,
          child: GestureDetector(
            onPanUpdate: (details) {
              final box = context.findRenderObject() as RenderBox;
              final local = box.globalToLocal(details.globalPosition);

              setState(() {
                _points = List.from(_points)..add(local);
              });
            },
            onPanEnd: (_) => _points.add(Offset.infinite),
            child: CustomPaint(
              painter: _MaskPainter(
                points: _points,
                image: _image!,
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class _MaskPainter extends CustomPainter {
  final List<Offset> points;
  final ui.Image image;

  _MaskPainter({required this.points, required this.image});

  @override
  void paint(Canvas canvas, Size size) {
    canvas.drawImage(image, Offset.zero, Paint());

    final paintLine = Paint()
      ..color = const Color.fromARGB(255, 0, 255, 0)
      ..strokeWidth = 6
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;

    for (int i = 1; i < points.length; i++) {
      if (points[i] != Offset.infinite &&
          points[i - 1] != Offset.infinite) {
        canvas.drawLine(points[i - 1], points[i], paintLine);
      }
    }
  }

  @override
  bool shouldRepaint(covariant _MaskPainter oldDelegate) => true;
}
