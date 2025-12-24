import 'dart:convert';
import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';

/// Mask/segmentation editor that returns a **normalized polygon** (0..1)
/// suitable for YOLOv8 Segmentation.
///
/// Returned payload (base64(JSON)):
/// {
///   "type": "tree_segmentation",
///   "format": "yolo_poly_v1",
///   "points": [[x,y], ...]   // x,y normalized to [0,1]
/// }
class MaskDrawingPage extends StatefulWidget {
  final String originalImageBase64;

  /// Optional initial mask payload (base64(JSON)).
  /// Supported formats:
  /// 1) New format: { "points": [[x,y],...], ... } with normalized points
  /// 2) Legacy format: [[x,y], ...] (assumed normalized)
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

  int? _imgW;
  int? _imgH;

  /// Normalized points (0..1)
  final List<Offset> _points = <Offset>[];

  /// Which point is being dragged
  int? _dragIndex;

  bool _closed = false;

  @override
  void initState() {
    super.initState();
    _imageBytes = base64Decode(widget.originalImageBase64);
    _decodeImageSize();
    _tryLoadInitialMask();
  }

  Future<void> _decodeImageSize() async {
    try {
      final ui.Image img = await _decodeUiImage(_imageBytes);
      if (!mounted) return;
      setState(() {
        _imgW = img.width;
        _imgH = img.height;
      });
    } catch (_) {
      // If image size can't be decoded, UI will still show but drawing will be disabled.
    }
  }

  Future<ui.Image> _decodeUiImage(Uint8List bytes) {
    final completer = Completer<ui.Image>();
    ui.decodeImageFromList(bytes, (ui.Image img) => completer.complete(img));
    return completer.future;
  }

  void _tryLoadInitialMask() {
    final s = widget.initialMaskBase64;
    if (s == null || s.trim().isEmpty) return;

    try {
      final raw = utf8.decode(base64Decode(s));
      final decoded = jsonDecode(raw);

      List<dynamic>? ptsDyn;
      if (decoded is Map<String, dynamic>) {
        final p = decoded['points'];
        if (p is List) ptsDyn = p;
      } else if (decoded is List) {
        ptsDyn = decoded;
      }

      if (ptsDyn == null) return;

      for (final e in ptsDyn) {
        if (e is List && e.length >= 2) {
          final x = (e[0] as num).toDouble();
          final y = (e[1] as num).toDouble();
          _points.add(_clamp01(Offset(x, y)));
        }
      }
    } catch (_) {
      // ignore invalid legacy mask
    }
  }

  Offset _clamp01(Offset o) {
    final dx = o.dx.isNaN ? 0.0 : o.dx.clamp(0.0, 1.0);
    final dy = o.dy.isNaN ? 0.0 : o.dy.clamp(0.0, 1.0);
    return Offset(dx, dy);
  }

  bool get _canDraw => (_imgW != null && _imgH != null);

  void _addPoint(Offset norm) {
    if (_closed) return;
    setState(() {
      _points.add(_clamp01(norm));
    });
  }

  void _toggleClosed() {
    if (_points.length < 3) return;
    setState(() {
      _closed = !_closed;
      _dragIndex = null;
    });
  }

  void _undo() {
    if (_points.isEmpty) return;
    setState(() {
      _closed = false;
      _dragIndex = null;
      _points.removeLast();
    });
  }

  void _clear() {
    setState(() {
      _closed = false;
      _dragIndex = null;
      _points.clear();
    });
  }

  void _finish() {
    if (_points.length < 3 || !_closed) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Нужно минимум 3 точки и замкнуть контур.'),
        ),
      );
      return;
    }

    final payload = <String, dynamic>{
      'type': 'tree_segmentation',
      'format': 'yolo_poly_v1',
      'points': _points.map((p) => [p.dx, p.dy]).toList(),
    };

    final jsonStr = jsonEncode(payload);
    final b64 = base64Encode(utf8.encode(jsonStr));

    Navigator.of(context).pop(b64);
  }

  // Finds nearest point index within threshold pixels
  int? _findHitPoint(Offset localPx, Size boxSize, {double thresholdPx = 18}) {
    if (_points.isEmpty) return null;

    int? bestI;
    double bestD = double.infinity;

    for (var i = 0; i < _points.length; i++) {
      final p = _points[i];
      final px = Offset(p.dx * boxSize.width, p.dy * boxSize.height);
      final d = (px - localPx).distance;
      if (d < bestD) {
        bestD = d;
        bestI = i;
      }
    }

    if (bestI != null && bestD <= thresholdPx) return bestI;
    return null;
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Контур дерева'),
        actions: [
          IconButton(
            tooltip: 'Готово',
            icon: const Icon(Icons.check),
            onPressed: _finish,
          ),
        ],
      ),
      body: Column(
        children: [
          // Canvas
          Expanded(
            child: LayoutBuilder(
              builder: (context, c) {
                final canvasW = c.maxWidth;
                final canvasH = c.maxHeight;

                if (!_canDraw) {
                  return Center(
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        const CircularProgressIndicator(),
                        const SizedBox(height: 12),
                        Text(
                          'Загружаем изображение…',
                          style: theme.textTheme.bodyMedium,
                        ),
                      ],
                    ),
                  );
                }

                final imgW = _imgW!.toDouble();
                final imgH = _imgH!.toDouble();

                final scale = math.min(canvasW / imgW, canvasH / imgH);
                final boxW = imgW * scale;
                final boxH = imgH * scale;

                // We draw inside this box, so local coords map 1:1 to box pixels.
                return Center(
                  child: SizedBox(
                    width: boxW,
                    height: boxH,
                    child: GestureDetector(
                      behavior: HitTestBehavior.opaque,
                      onTapDown: (d) {
                        if (_closed) return;
                        final local = d.localPosition;
                        final nx = (local.dx / boxW).clamp(0.0, 1.0);
                        final ny = (local.dy / boxH).clamp(0.0, 1.0);
                        _addPoint(Offset(nx, ny));
                      },
                      onPanStart: (d) {
                        final idx = _findHitPoint(d.localPosition, Size(boxW, boxH));
                        if (idx != null) {
                          setState(() => _dragIndex = idx);
                        }
                      },
                      onPanUpdate: (d) {
                        if (_dragIndex == null) return;
                        final local = d.localPosition;
                        final nx = (local.dx / boxW).clamp(0.0, 1.0);
                        final ny = (local.dy / boxH).clamp(0.0, 1.0);
                        setState(() {
                          _points[_dragIndex!] = Offset(nx, ny);
                        });
                      },
                      onPanEnd: (_) {
                        if (_dragIndex != null) {
                          setState(() => _dragIndex = null);
                        }
                      },
                      child: Stack(
                        children: [
                          Positioned.fill(
                            child: Image.memory(
                              _imageBytes,
                              fit: BoxFit.fill,
                            ),
                          ),
                          Positioned.fill(
                            child: CustomPaint(
                              painter: _PolygonPainter(
                                points: _points,
                                closed: _closed,
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                );
              },
            ),
          ),

          // Controls
          Padding(
            padding: const EdgeInsets.fromLTRB(12, 10, 12, 16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Row(
                  children: [
                    Expanded(
                      child: FilledButton.icon(
                        onPressed: _points.length >= 3 ? _toggleClosed : null,
                        icon: Icon(_closed ? Icons.link_off : Icons.link),
                        label: Text(_closed ? 'Разомкнуть' : 'Замкнуть'),
                      ),
                    ),
                    const SizedBox(width: 10),
                    Expanded(
                      child: OutlinedButton.icon(
                        onPressed: _points.isNotEmpty ? _undo : null,
                        icon: const Icon(Icons.undo),
                        label: const Text('Отменить'),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 10),
                Row(
                  children: [
                    Expanded(
                      child: OutlinedButton.icon(
                        onPressed: _points.isNotEmpty ? _clear : null,
                        icon: const Icon(Icons.delete_outline),
                        label: const Text('Очистить'),
                      ),
                    ),
                    const SizedBox(width: 10),
                    Expanded(
                      child: FilledButton.icon(
                        onPressed: (_points.length >= 3 && _closed) ? _finish : null,
                        icon: const Icon(Icons.check),
                        label: const Text('Сохранить'),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                Text(
                  'Тап — добавить точку. Перетаскивай точки пальцем. Сначала поставь минимум 3 точки и нажми «Замкнуть».',
                  style: theme.textTheme.bodySmall?.copyWith(color: Colors.black54),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _PolygonPainter extends CustomPainter {
  final List<Offset> points; // normalized 0..1
  final bool closed;

  _PolygonPainter({
    required this.points,
    required this.closed,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (points.isEmpty) return;

    final paintLine = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.5
      ..color = Colors.greenAccent.shade400;

    final paintFill = Paint()
      ..style = PaintingStyle.fill
      ..color = Colors.greenAccent.withOpacity(0.18);

    final paintPoint = Paint()
      ..style = PaintingStyle.fill
      ..color = Colors.white;

    final paintPointBorder = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2
      ..color = Colors.greenAccent.shade700;

    final path = Path();

    Offset toPx(Offset n) => Offset(n.dx * size.width, n.dy * size.height);

    final p0 = toPx(points.first);
    path.moveTo(p0.dx, p0.dy);
    for (var i = 1; i < points.length; i++) {
      final p = toPx(points[i]);
      path.lineTo(p.dx, p.dy);
    }

    if (closed && points.length >= 3) {
      path.close();
      canvas.drawPath(path, paintFill);
    }

    canvas.drawPath(path, paintLine);

    // Draw vertices
    for (final n in points) {
      final p = toPx(n);
      canvas.drawCircle(p, 5.5, paintPoint);
      canvas.drawCircle(p, 5.5, paintPointBorder);
    }
  }

  @override
  bool shouldRepaint(covariant _PolygonPainter oldDelegate) {
    return oldDelegate.points != points || oldDelegate.closed != closed;
  }
}