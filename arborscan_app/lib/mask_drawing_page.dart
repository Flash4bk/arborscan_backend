import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';

class MaskDrawingPage extends StatefulWidget {
  final String originalImageBase64;
  final String? initialMaskBase64;

  const MaskDrawingPage({
    super.key,
    required this.originalImageBase64,
    this.initialMaskBase64,
  });

  @override
  State<MaskDrawingPage> createState() => _MaskDrawingPageState();
}

class _MaskDrawingPageState extends State<MaskDrawingPage> {
  late final Uint8List _imageBytes;

  final TransformationController _tc = TransformationController();

  final List<Offset> _points = [];
  bool _closed = false;
  int? _draggingIndex;
  bool _isTransforming = false;

  static const double _handleRadius = 4;
  static const double _hitRadius = 14;

  @override
  void initState() {
    super.initState();
    _imageBytes = base64Decode(widget.originalImageBase64);

    if (widget.initialMaskBase64 != null &&
        widget.initialMaskBase64!.isNotEmpty) {
      try {
        final raw =
            utf8.decode(base64Decode(widget.initialMaskBase64!));
        final decoded = jsonDecode(raw);
        if (decoded is List) {
          for (final e in decoded) {
            _points.add(
              Offset(
                (e['x'] as num).toDouble(),
                (e['y'] as num).toDouble(),
              ),
            );
          }
          if (_points.length >= 3) _closed = true;
        }
      } catch (_) {}
    }
  }

  @override
  void dispose() {
    _tc.dispose();
    super.dispose();
  }

  // ---------- действия ----------

  void _clearAll() {
    setState(() {
      _points.clear();
      _closed = false;
    });
  }

  void _removeLast() {
    if (_points.isNotEmpty && !_closed) {
      setState(() => _points.removeLast());
    }
  }

  void _unlockPolygon() {
    if (_closed) {
      setState(() => _closed = false);
    }
  }

  void _finish() {
    if (_points.length < 3) {
      Navigator.pop(context, null);
      return;
    }

    final data = _points.map((p) => {'x': p.dx, 'y': p.dy}).toList();
    Navigator.pop(
      context,
      base64Encode(utf8.encode(jsonEncode(data))),
    );
  }

  void _onTapDown(TapDownDetails details) {
    if (_isTransforming || _closed) return;

    final scenePos = _tc.toScene(details.localPosition);

    if (_points.length >= 3 &&
        (scenePos - _points.first).distance <= _hitRadius) {
      setState(() => _closed = true);
      return;
    }

    setState(() => _points.add(scenePos));
  }

  // ---------- UI ----------

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Редактирование маски'),
        actions: [
          IconButton(
            icon: const Icon(Icons.check),
            onPressed: _finish,
          )
        ],
      ),
      body: GestureDetector(
        onTapDown: _onTapDown,
        child: InteractiveViewer(
          transformationController: _tc,
          minScale: 1,
          maxScale: 8,
          boundaryMargin: const EdgeInsets.all(200),
          onInteractionStart: (_) => _isTransforming = true,
          onInteractionEnd: (_) =>
              Future.delayed(const Duration(milliseconds: 80), () {
            _isTransforming = false;
          }),
          child: Stack(
            children: [
              Image.memory(_imageBytes, fit: BoxFit.contain),
              Positioned.fill(
                child: CustomPaint(
                  painter: _PolygonPainter(
                    points: _points,
                    closed: _closed,
                  ),
                ),
              ),
              ..._buildHandles(),
              Positioned(
                left: 12,
                top: 12,
                child: _hint(theme),
              ),
            ],
          ),
        ),
      ),

      // ---------- НИЖНЯЯ ПАНЕЛЬ ----------
      bottomNavigationBar: SafeArea(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(12, 8, 12, 12),
          child: Row(
            children: [
              IconButton(
                tooltip: 'Удалить последнюю точку',
                icon: const Icon(Icons.undo),
                onPressed: (!_closed && _points.isNotEmpty)
                    ? _removeLast
                    : null,
              ),
              IconButton(
                tooltip: 'Разомкнуть полигон',
                icon: const Icon(Icons.lock_open),
                onPressed: _closed ? _unlockPolygon : null,
              ),
              IconButton(
                tooltip: 'Очистить всё',
                icon: const Icon(Icons.delete_outline),
                onPressed: _points.isNotEmpty ? _clearAll : null,
              ),
              const Spacer(),
              Text(
                _closed
                    ? 'Полигон замкнут'
                    : 'Точки: ${_points.length}',
                style: theme.textTheme.bodySmall,
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _hint(ThemeData theme) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.6),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Text(
        _closed
            ? 'Полигон замкнут — можно двигать точки'
            : 'Тапайте по контуру · замыкание — по первой точке',
        style: theme.textTheme.bodySmall?.copyWith(color: Colors.white),
      ),
    );
  }

  List<Widget> _buildHandles() {
    return List.generate(_points.length, (i) {
      final p = _points[i];
      final isFirst = i == 0;

      return Positioned(
        left: p.dx - _handleRadius,
        top: p.dy - _handleRadius,
        child: GestureDetector(
          onPanStart: (_) => _draggingIndex = i,
          onPanUpdate: (d) {
            if (_draggingIndex == i) {
              setState(() => _points[i] += d.delta);
            }
          },
          onPanEnd: (_) => _draggingIndex = null,
          child: Container(
            width: _handleRadius * 2,
            height: _handleRadius * 2,
            decoration: BoxDecoration(
              color: isFirst ? Colors.orange : const Color(0xFF2F6B3A),
              shape: BoxShape.circle,
              border: Border.all(color: Colors.white, width: 1.5),
            ),
          ),
        ),
      );
    });
  }
}

class _PolygonPainter extends CustomPainter {
  final List<Offset> points;
  final bool closed;

  _PolygonPainter({required this.points, required this.closed});

  @override
  void paint(Canvas canvas, Size size) {
    if (points.isEmpty) return;

    final stroke = Paint()
      ..color = const Color(0xFF2F6B3A)
      ..strokeWidth = 3
      ..style = PaintingStyle.stroke;

    final fill = Paint()
      ..color = const Color(0x332F6B3A)
      ..style = PaintingStyle.fill;

    final path = Path()..moveTo(points.first.dx, points.first.dy);
    for (int i = 1; i < points.length; i++) {
      path.lineTo(points[i].dx, points[i].dy);
    }

    if (closed) {
      path.close();
      canvas.drawPath(path, fill);
    }

    canvas.drawPath(path, stroke);
  }

  @override
  bool shouldRepaint(covariant _PolygonPainter oldDelegate) =>
      oldDelegate.points != points || oldDelegate.closed != closed;
}
