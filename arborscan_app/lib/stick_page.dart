import 'dart:typed_data';
import 'dart:ui' as ui;
import 'dart:math';
import 'package:flutter/material.dart';

class StickFixPage extends StatefulWidget {
  final Uint8List imageBytes;

  const StickFixPage({
    super.key,
    required this.imageBytes,
  });

  @override
  State<StickFixPage> createState() => _StickFixPageState();
}

class _StickFixPageState extends State<StickFixPage> {
  ui.Image? _image;

  Offset? pointA;
  Offset? pointB;

  double? scale;
  double? offX;
  double? offY;

  @override
  void initState() {
    super.initState();
    _loadImage();
  }

  Future<void> _loadImage() async {
    final codec = await ui.instantiateImageCodec(widget.imageBytes);
    final frame = await codec.getNextFrame();
    setState(() => _image = frame.image);
  }

  void _onTapDown(TapDownDetails d, double scale, double offX, double offY) {
    final local = Offset(
      (d.localPosition.dx - offX) / scale,
      (d.localPosition.dy - offY) / scale,
    );

    setState(() {
      if (pointA == null) {
        pointA = local;
      } else if (pointB == null) {
        pointB = local;
      } else {
        pointA = local;
        pointB = null;
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final img = _image;

    return Scaffold(
      appBar: AppBar(
        title: const Text("Перевыделение палки"),
        actions: [
          if (pointA != null && pointB != null)
            IconButton(
              icon: const Icon(Icons.check),
              onPressed: _finish,
            ),
        ],
      ),
      body: img == null
          ? const Center(child: CircularProgressIndicator())
          : LayoutBuilder(
              builder: (ctx, constraints) {
                final scaleX = constraints.maxWidth / img.width;
                final scaleY = constraints.maxHeight / img.height;
                final s = scaleX < scaleY ? scaleX : scaleY;

                final w = img.width * s;
                final h = img.height * s;

                final x = (constraints.maxWidth - w) / 2;
                final y = (constraints.maxHeight - h) / 2;

                scale = s;
                offX = x;
                offY = y;

                return GestureDetector(
                  onTapDown: (d) => _onTapDown(d, s, x, y),
                  child: CustomPaint(
                    size: Size(constraints.maxWidth, constraints.maxHeight),
                    painter: _StickPainter(
                      image: img,
                      pointA: pointA,
                      pointB: pointB,
                      scale: s,
                      offX: x,
                      offY: y,
                    ),
                  ),
                );
              },
            ),
    );
  }

  void _finish() {
    if (pointA == null || pointB == null) return;

    final dx = pointA!.dx - pointB!.dx;
    final dy = pointA!.dy - pointB!.dy;
    final distPx = sqrt(dx * dx + dy * dy);

    if (distPx <= 1) return;

    final newScale = 1.0 / distPx;

    Navigator.pop(context, newScale);
  }
}

class _StickPainter extends CustomPainter {
  final ui.Image image;
  final Offset? pointA;
  final Offset? pointB;

  final double scale;
  final double offX;
  final double offY;

  _StickPainter({
    required this.image,
    required this.pointA,
    required this.pointB,
    required this.scale,
    required this.offX,
    required this.offY,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final dstRect = Rect.fromLTWH(offX, offY, image.width * scale, image.height * scale);
    final srcRect = Rect.fromLTWH(0, 0, image.width.toDouble(), image.height.toDouble());

    canvas.drawImageRect(image, srcRect, dstRect, Paint());

    final paintPoint = Paint()
      ..color = Colors.red
      ..style = PaintingStyle.fill;

    final paintLine = Paint()
      ..color = Colors.green
      ..strokeWidth = 4
      ..style = PaintingStyle.stroke;

    if (pointA != null) {
      canvas.drawCircle(
        Offset(pointA!.dx * scale + offX, pointA!.dy * scale + offY),
        8,
        paintPoint,
      );
    }

    if (pointB != null) {
      canvas.drawCircle(
        Offset(pointB!.dx * scale + offX, pointB!.dy * scale + offY),
        8,
        paintPoint,
      );
    }

    if (pointA != null && pointB != null) {
      canvas.drawLine(
        Offset(pointA!.dx * scale + offX, pointA!.dy * scale + offY),
        Offset(pointB!.dx * scale + offX, pointB!.dy * scale + offY),
        paintLine,
      );
    }
  }

  @override
  bool shouldRepaint(covariant _StickPainter oldDelegate) {
    return pointA != oldDelegate.pointA || pointB != oldDelegate.pointB;
  }
}
