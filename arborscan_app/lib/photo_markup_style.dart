import 'package:flutter/material.dart';

/// Logical screen pixels: photo zoom changes the geometry, not the marks.
const photoMarkupLineWidth = 1.25;

void paintPhotoMarkupHandle(Canvas canvas, Offset point,
    {double scale = 1,
    Color color = Colors.cyanAccent,
    bool first = false,
    bool active = false}) {
  final radius = (active ? 4.5 : (first ? 3.5 : 2.5)) / scale;
  canvas.drawCircle(
      point,
      radius,
      Paint()
        ..color = Colors.black.withValues(alpha: .7)
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.5 / scale);
  canvas.drawCircle(
      point,
      radius,
      Paint()
        ..color = color
        ..style = PaintingStyle.stroke
        ..strokeWidth = photoMarkupLineWidth / scale);
}
