import 'package:flutter/material.dart';

class ContourEditorState {
  final int width;
  final int height;
  final bool closed;
  final List<Offset> points;
  const ContourEditorState({required this.width, required this.height,
    required this.closed, required this.points});

  factory ContourEditorState.fromJson(Map<String, dynamic> data) {
    if (data['version'] != 1 || data['coordinates'] != 'normalized_oriented_image' ||
        data['width'] is! int || data['height'] is! int ||
        data['width'] <= 0 || data['height'] <= 0 ||
        data['width'] * data['height'] > 25000000 || data['closed'] is! bool ||
        data['points'] is! List || (data['points'] as List).length > 4096) {
      throw const FormatException('Неподдерживаемое состояние редактора');
    }
    final points = (data['points'] as List).map((p) {
      if (p is! Map || p['x'] is! num || p['y'] is! num) throw const FormatException('Неверные точки');
      final x = (p['x'] as num).toDouble(), y = (p['y'] as num).toDouble();
      if (!x.isFinite || !y.isFinite || x < 0 || x > 1 || y < 0 || y > 1) {
        throw const FormatException('Неверные координаты');
      }
      return Offset(x, y);
    }).toList();
    if (data['closed'] == true && points.length < 3) throw const FormatException('Незамкнутый контур');
    return ContourEditorState(width:data['width'], height:data['height'], closed:data['closed'], points:points);
  }

  Map<String, dynamic> toJson() => {'version':1, 'coordinates':'normalized_oriented_image',
    'width':width, 'height':height, 'closed':closed,
    'points':points.map((p) => {'x':p.dx,'y':p.dy}).toList()};
}
