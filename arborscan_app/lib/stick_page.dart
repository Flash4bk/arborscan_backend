import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/material.dart';

class StickPage extends StatefulWidget {
  final String originalImageBase64;
  final double currentScalePxToM;

  const StickPage({
    super.key,
    required this.originalImageBase64,
    required this.currentScalePxToM,
  });

  @override
  State<StickPage> createState() => _StickPageState();
}

class _StickPageState extends State<StickPage> {
  late Uint8List _imageBytes;

  final TransformationController _controller = TransformationController();

  /// две точки палки (низ и верх)
  final List<Offset> _points = [];

  int? _dragIndex;
  bool _lockPan = false;

  Offset? _downScenePos;
  bool _moved = false;

  static const double _pointRadius = 3;
  static const double _hitRadius = 18;
  static const double _moveThreshold = 4;

  @override
  void initState() {
    super.initState();
    _imageBytes = base64Decode(widget.originalImageBase64);
  }

  /// перевод координат экрана → координаты сцены (с учётом зума)
  Offset _scene(Offset local) {
    return _controller.toScene(local);
  }

  int? _hitPoint(Offset scenePos) {
    for (int i = 0; i < _points.length; i++) {
      if ((scenePos - _points[i]).distance <= _hitRadius) {
        return i;
      }
    }
    return null;
  }

  // ================= POINTER EVENTS =================

  void _onPointerDown(PointerDownEvent e) {
    final scenePos = _scene(e.localPosition);
    _downScenePos = scenePos;
    _moved = false;

    _dragIndex = _hitPoint(scenePos);
    _lockPan = _dragIndex != null;
  }

  void _onPointerMove(PointerMoveEvent e) {
    if (_downScenePos == null) return;

    final scenePos = _scene(e.localPosition);
    final dist = (scenePos - _downScenePos!).distance;

    if (dist > _moveThreshold) {
      _moved = true;
    }

    if (_dragIndex != null) {
      setState(() {
        _points[_dragIndex!] = scenePos;
      });
    }
  }

  void _onPointerUp(PointerUpEvent e) {
    final scenePos = _scene(e.localPosition);

    if (!_moved) {
      final hit = _hitPoint(scenePos);

      setState(() {
        if (hit != null) {
          _dragIndex = hit;
        } else if (_points.length < 2) {
          _points.add(scenePos);
        } else {
          final d0 = (scenePos - _points[0]).distance;
          final d1 = (scenePos - _points[1]).distance;
          _points[d0 < d1 ? 0 : 1] = scenePos;
        }
      });
    }

    _dragIndex = null;
    _downScenePos = null;
    _lockPan = false;
  }

  // ================= LOGIC =================

  double? _lengthPx() {
    if (_points.length != 2) return null;
    return (_points[0] - _points[1]).distance;
  }

  double? _scalePxToM() {
    final len = _lengthPx();
    if (len == null || len <= 0) return null;
    return 1 / len;
  }

  void _apply() {
    final scale = _scalePxToM();
    if (scale == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Поставьте 2 точки на палке (1 м)')),
      );
      return;
    }
    Navigator.pop(context, scale);
  }

  void _clear() {
    setState(() {
      _points.clear();
      _dragIndex = null;
      _lockPan = false;
    });
  }

  // ================= UI =================

  @override
  Widget build(BuildContext context) {
    final length = _lengthPx();
    final scale = _scalePxToM();

    return Scaffold(
      appBar: AppBar(
        title: const Text('Коррекция палки'),
        actions: [
          IconButton(
            icon: const Icon(Icons.delete_outline),
            onPressed: _clear,
          ),
          IconButton(
            icon: const Icon(Icons.check),
            onPressed: _apply,
          ),
        ],
      ),
      body: Listener(
        onPointerDown: _onPointerDown,
        onPointerMove: _onPointerMove,
        onPointerUp: _onPointerUp,
        child: InteractiveViewer(
          transformationController: _controller,
          minScale: 1,
          maxScale: 8,
          panEnabled: !_lockPan,
          scaleEnabled: true,
          boundaryMargin: const EdgeInsets.all(200),
          child: Stack(
            children: [
              Image.memory(_imageBytes, fit: BoxFit.contain),
              Positioned.fill(
                child: CustomPaint(
                  painter: _StickPainter(points: _points),
                ),
              ),
            ],
          ),
        ),
      ),
      bottomNavigationBar: Padding(
        padding: const EdgeInsets.all(12),
        child: Text(
          length == null
              ? 'Поставьте 2 точки по палке (1 м)'
              : 'Длина: ${length.toStringAsFixed(1)} px\n'
                'Масштаб: ${scale!.toStringAsFixed(6)} м/px',
          textAlign: TextAlign.center,
        ),
      ),
    );
  }
}

class _StickPainter extends CustomPainter {
  final List<Offset> points;

  _StickPainter({required this.points});

  @override
  void paint(Canvas canvas, Size size) {
    final linePaint = Paint()
      ..color = Colors.red
      ..strokeWidth = 2.5
      ..style = PaintingStyle.stroke;

    final dotPaint = Paint()..color = Colors.red;

    if (points.length == 2) {
      canvas.drawLine(points[0], points[1], linePaint);
    }

    for (final pt in points) {
      canvas.drawCircle(pt, 3, dotPaint);
    }
  }

  @override
  bool shouldRepaint(covariant _StickPainter oldDelegate) => true;
}
