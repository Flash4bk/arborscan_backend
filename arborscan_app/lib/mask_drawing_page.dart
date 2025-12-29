import 'dart:convert';
import 'dart:math';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';

class MaskDrawingPage extends StatefulWidget {
  final String? originalImageBase64;
  final String? initialMaskBase64;

  const MaskDrawingPage({
    Key? key,
    this.originalImageBase64,
    this.initialMaskBase64,
  }) : super(key: key);

  @override
  State<MaskDrawingPage> createState() => _MaskDrawingPageState();
}

class _MaskDrawingPageState extends State<MaskDrawingPage> {
  late final Uint8List _imageBytes;

  ui.Image? _image;
  Size? _imageSize; // оригинальный размер изображения

  // Важно: Listener будет СНАРУЖИ InteractiveViewer, поэтому toScene() корректен
  final TransformationController _controller = TransformationController();

  // Точки храним В scene-координатах child (px внутри CustomPaint размера drawSize)
  final List<Offset> _points = [];
  int? _dragIndex;
  bool _closed = false;

  // Размер child (CustomPaint). Нужен для экспорта и нормализации
  Size? _drawSize;

  // Gesture state
  final Set<int> _activePointers = <int>{};
  int? _primaryPointer;
  bool _inPinch = false;

  bool _tapCandidate = false;
  Offset? _tapDownLocal; // viewport local (Listener local)
  Offset? _tapDownScene; // scene position at down

  static const double _tapSlopPx = 8.0;

  @override
  void initState() {
    super.initState();
    if (widget.originalImageBase64 == null) {
      throw Exception('originalImageBase64 is required');
    }
    _imageBytes = base64Decode(widget.originalImageBase64!);
    _loadImage();
  }

  Future<void> _loadImage() async {
    final codec = await ui.instantiateImageCodec(_imageBytes);
    final frame = await codec.getNextFrame();
    setState(() {
      _image = frame.image;
      _imageSize =
          Size(frame.image.width.toDouble(), frame.image.height.toDouble());
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  double get _scale => _controller.value.getMaxScaleOnAxis().clamp(1.0, 10.0);

  Offset _toScene(Offset viewportLocal) => _controller.toScene(viewportLocal);

  int? _hitTest(Offset scenePx) {
    // Порог уменьшаем на зуме — чтобы "попадание" оставалось удобным
    final threshold = 14.0 / _scale;
    for (int i = 0; i < _points.length; i++) {
      if ((_points[i] - scenePx).distance <= threshold) return i;
    }
    return null;
  }

  Future<void> _finish() async {
    if (_imageSize == null || _drawSize == null) return;

    final recorder = ui.PictureRecorder();
    final canvas = Canvas(recorder);

    // black bg
    canvas.drawRect(
      Rect.fromLTWH(0, 0, _imageSize!.width, _imageSize!.height),
      Paint()..color = Colors.black,
    );

    if (_points.length >= 3) {
      final path = Path();

      final first = Offset(
        _points.first.dx / _drawSize!.width * _imageSize!.width,
        _points.first.dy / _drawSize!.height * _imageSize!.height,
      );
      path.moveTo(first.dx, first.dy);

      for (int i = 1; i < _points.length; i++) {
        path.lineTo(
          _points[i].dx / _drawSize!.width * _imageSize!.width,
          _points[i].dy / _drawSize!.height * _imageSize!.height,
        );
      }
      path.close();

      canvas.drawPath(path, Paint()..color = Colors.white);
    }

    final img = await recorder
        .endRecording()
        .toImage(_imageSize!.width.toInt(), _imageSize!.height.toInt());

    final bytes = await img.toByteData(format: ui.ImageByteFormat.png);

    Navigator.pop(context, {
      "mask_png_base64": base64Encode(bytes!.buffer.asUint8List()),
      // points normalized 0..1 (relative to drawSize) — как раньше для твоего пайплайна
      "points": _points
          .map((p) => {
                "x": p.dx / _drawSize!.width,
                "y": p.dy / _drawSize!.height,
              })
          .toList(),
      "closed": _closed,
    });
  }

  void _resetTapState() {
    _tapCandidate = false;
    _tapDownLocal = null;
    _tapDownScene = null;
  }

  @override
  Widget build(BuildContext context) {
    if (_image == null || _imageSize == null) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text('Исправление маски'),
        actions: [
          IconButton(
            tooltip: 'Замкнуть контур',
            icon: const Icon(Icons.link),
            onPressed: _points.length >= 3 ? () => setState(() => _closed = true) : null,
          ),
          IconButton(
            tooltip: 'Открыть контур',
            icon: const Icon(Icons.link_off),
            onPressed: _closed ? () => setState(() => _closed = false) : null,
          ),
          IconButton(
            tooltip: 'Undo',
            icon: const Icon(Icons.undo),
            onPressed: _points.isEmpty
                ? null
                : () => setState(() {
                      _points.removeLast();
                      if (_points.length < 3) _closed = false;
                    }),
          ),
          IconButton(
            tooltip: 'Сброс зума',
            icon: const Icon(Icons.zoom_out_map),
            onPressed: () => _controller.value = Matrix4.identity(),
          ),
        ],
      ),
      body: LayoutBuilder(
        builder: (_, constraints) {
          final fitScale = min(
            constraints.maxWidth / _imageSize!.width,
            constraints.maxHeight / _imageSize!.height,
          );

          final drawSize = Size(
            _imageSize!.width * fitScale,
            _imageSize!.height * fitScale,
          );

          // сохраняем актуальный drawSize
          _drawSize = drawSize;

          return Center(
            child: SizedBox(
              width: drawSize.width,
              height: drawSize.height,

              // ✅ Listener СНАРУЖИ InteractiveViewer — это ключевое исправление
              child: Listener(
                behavior: HitTestBehavior.translucent,

                onPointerDown: (e) {
                  _activePointers.add(e.pointer);

                  // Если стало 2+ пальца -> pinch, запрет на добавление точек
                  if (_activePointers.length >= 2) {
                    _inPinch = true;
                    _dragIndex = null;
                    _primaryPointer = null;
                    _resetTapState();
                    return;
                  }

                  // один палец
                  _primaryPointer = e.pointer;

                  final scene = _toScene(e.localPosition);

                  // drag существующей точки разрешен всегда (даже если closed)
                  final hit = _hitTest(scene);
                  if (hit != null) {
                    _dragIndex = hit;
                    _resetTapState();
                    return;
                  }

                  // новые точки — только если контур не замкнут
                  if (_closed) {
                    _resetTapState();
                    return;
                  }

                  // кандидат на tap: фиксируем координату НА DOWN (чтобы transform не менял точку)
                  _tapCandidate = true;
                  _tapDownLocal = e.localPosition;
                  _tapDownScene = scene;
                },

                onPointerMove: (e) {
                  if (_primaryPointer != e.pointer) return;

                  final scene = _toScene(e.localPosition);

                  // перетаскивание точки
                  if (_dragIndex != null) {
                    setState(() {
                      _points[_dragIndex!] = scene;
                    });
                    return;
                  }

                  // отменяем tap, если пользователь реально "повёл"
                  if (_tapCandidate && _tapDownLocal != null) {
                    final localDist = (e.localPosition - _tapDownLocal!).distance;
                    if (localDist > _tapSlopPx) {
                      _tapCandidate = false;
                    }
                  }
                },

                onPointerUp: (e) {
                  _activePointers.remove(e.pointer);

                  // как только все пальцы убраны — pinch закончился
                  if (_activePointers.isEmpty) {
                    _inPinch = false;
                  }

                  if (_primaryPointer == e.pointer) {
                    // Добавляем точку только если это был реальный tap (без pinch, без pan)
                    if (_tapCandidate && !_inPinch && !_closed && _tapDownScene != null) {
                      setState(() {
                        _points.add(_tapDownScene!);
                      });
                    }

                    _dragIndex = null;
                    _primaryPointer = null;
                    _resetTapState();
                  }
                },

                onPointerCancel: (e) {
                  _activePointers.remove(e.pointer);
                  if (_activePointers.isEmpty) _inPinch = false;
                  _dragIndex = null;
                  _primaryPointer = null;
                  _resetTapState();
                },

                child: InteractiveViewer(
                  transformationController: _controller,
                  minScale: 1,
                  maxScale: 6,
                  panEnabled: true,
                  scaleEnabled: true,
                  child: CustomPaint(
                    size: drawSize,
                    painter: _MaskPainter(
                      image: _image!,
                      points: _points,
                      closed: _closed,
                    ),
                  ),
                ),
              ),
            ),
          );
        },
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: _finish,
        icon: const Icon(Icons.check),
        label: const Text('Подтвердить'),
      ),
    );
  }
}

class _MaskPainter extends CustomPainter {
  final ui.Image image;
  final List<Offset> points;
  final bool closed;

  _MaskPainter({
    required this.image,
    required this.points,
    required this.closed,
  });

  @override
  void paint(Canvas canvas, Size size) {
    canvas.drawImageRect(
      image,
      Rect.fromLTWH(
        0,
        0,
        image.width.toDouble(),
        image.height.toDouble(),
      ),
      Rect.fromLTWH(0, 0, size.width, size.height),
      Paint(),
    );

    if (points.isEmpty) return;

    final path = Path()..moveTo(points.first.dx, points.first.dy);
    for (int i = 1; i < points.length; i++) {
      path.lineTo(points[i].dx, points[i].dy);
    }
    if (closed && points.length >= 3) path.close();

    canvas.drawPath(
      path,
      Paint()
        ..color = Colors.red
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2,
    );

    for (final p in points) {
      canvas.drawCircle(p, 4, Paint()..color = Colors.cyanAccent);
    }

    // подсветка первой точки
    canvas.drawCircle(
      points.first,
      6,
      Paint()..color = Colors.yellowAccent.withOpacity(0.85),
    );
  }

  @override
  bool shouldRepaint(covariant _MaskPainter oldDelegate) =>
      oldDelegate.points != points || oldDelegate.closed != closed;
}
