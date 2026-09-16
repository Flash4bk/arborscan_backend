import 'dart:convert';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'app_theme.dart';

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
  Size? _imageSize;
  Size _displaySize = Size.zero;
  String? _error;

  /// Две точки линии
  final List<Offset> _points = [];

  int? _dragIndex;
  bool _lockPan = false;
  Offset? _downScenePos;
  bool _moved = false;

  static const double _hitRadius = 18;
  static const double _moveThreshold = 4;

  @override
  void initState() {
    super.initState();
    _imageBytes = base64Decode(widget.originalImageBase64);
    _decode();
  }

  Future<void> _decode() async {
    try {
      final codec = await ui.instantiateImageCodec(_imageBytes);
      final frame = await codec.getNextFrame();
      final size =
          Size(frame.image.width.toDouble(), frame.image.height.toDouble());
      frame.image.dispose();
      codec.dispose();
      if (mounted) setState(() => _imageSize = size);
    } catch (_) {
      if (mounted) setState(() => _error = 'Не удалось открыть изображение.');
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  Offset _scene(Offset local) {
    // Listener is inside the transformed image: Flutter already inverses zoom.
    // Persist points in ORIGINAL oriented pixels, not viewport pixels.
    return Offset(
        (local.dx / _displaySize.width).clamp(0, 1) * _imageSize!.width,
        (local.dy / _displaySize.height).clamp(0, 1) * _imageSize!.height);
  }

  int? _hitPoint(Offset scenePos) {
    for (int i = 0; i < _points.length; i++) {
      if ((scenePos - _points[i]).distance <=
          _hitRadius * _imageSize!.width / _displaySize.width) {
        return i;
      }
    }
    return null;
  }

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

    if (dist > _moveThreshold * _imageSize!.width / _displaySize.width) {
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

  double? _lengthPx() {
    if (_points.length != 2) return null;
    return (_points[0] - _points[1]).distance;
  }

  void _clear() {
    setState(() {
      _points.clear();
      _dragIndex = null;
      _lockPan = false;
    });
  }

  // --- НОВЫЙ БЛОК: Запрашиваем реальную длину у пользователя ---
  Future<void> _apply() async {
    final lenPx = _lengthPx();
    if (lenPx == null || lenPx <= 0) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
            content: Text('Поставьте 2 точки, чтобы отметить объект')),
      );
      return;
    }

    final ctrl = TextEditingController();

    final realLengthM = await showDialog<double>(
      context: context,
      builder: (ctx) => Dialog(
        backgroundColor: Colors.transparent,
        child: GlassPanel(
          radius: 24,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Text(
                "РАЗМЕР ОБЪЕКТА",
                style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.w900,
                    color: AppTheme.primary,
                    letterSpacing: 2),
              ),
              const SizedBox(height: 12),
              const Text(
                "Какова реальная длина выделенного вами отрезка?",
                textAlign: TextAlign.center,
                style:
                    TextStyle(color: AppTheme.muted, fontSize: 13, height: 1.3),
              ),
              const SizedBox(height: 20),
              TextField(
                controller: ctrl,
                keyboardType:
                    const TextInputType.numberWithOptions(decimal: true),
                decoration: const InputDecoration(
                  labelText: 'Длина в метрах',
                  hintText: 'Например, 1.8 (рост человека)',
                ),
                autofocus: true,
              ),
              const SizedBox(height: 24),
              Row(
                children: [
                  Expanded(
                    child: OutlinedButton(
                      onPressed: () => Navigator.pop(ctx, null),
                      child: const Text('ОТМЕНА'),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: FilledButton(
                      style: FilledButton.styleFrom(
                          backgroundColor: AppTheme.primary),
                      onPressed: () {
                        final val =
                            double.tryParse(ctrl.text.replaceAll(',', '.'));
                        if (val != null && val.isFinite && val > 0) {
                          Navigator.pop(ctx, val);
                        } else {
                          ScaffoldMessenger.of(ctx).showSnackBar(
                            const SnackBar(
                                content: Text(
                                    'Введите корректное число больше нуля')),
                          );
                        }
                      },
                      child: const Text('ГОТОВО',
                          style: TextStyle(
                              color: Colors.black,
                              fontWeight: FontWeight.w900)),
                    ),
                  ),
                ],
              )
            ],
          ),
        ),
      ),
    );

    if (!mounted) return;
    if (realLengthM != null && realLengthM.isFinite && realLengthM > 0) {
      // Metres per original oriented pixel under the stated capture assumptions.
      final scale = realLengthM / lenPx;
      Navigator.pop(context, scale);
    }
  }

  @override
  Widget build(BuildContext context) {
    final length = _lengthPx();

    return Scaffold(
      backgroundColor: AppTheme.background,
      appBar: AppBar(
        title: const Text('Универсальная линейка',
            style: TextStyle(fontSize: 18, letterSpacing: 1.0)),
        actions: [
          IconButton(
            tooltip: 'Сбросить',
            icon: const Icon(Icons.delete_outline, color: AppTheme.danger),
            onPressed: _clear,
          ),
          IconButton(
            tooltip: 'Применить',
            icon: const Icon(Icons.check, color: AppTheme.primary),
            onPressed: _apply,
          ),
        ],
      ),
      body: Column(
        children: [
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
            color: AppTheme.surface3.withOpacity(0.3),
            child: Row(
              children: const [
                Icon(Icons.info_outline, color: AppTheme.primary2, size: 20),
                SizedBox(width: 10),
                Expanded(
                  child: Text(
                    'Отметьте известную высоту. Эталон должен быть рядом с деревом на той же глубине. Перспектива и наклон камеры ограничивают метод.',
                    style: TextStyle(
                        color: AppTheme.muted, fontSize: 12, height: 1.3),
                  ),
                ),
              ],
            ),
          ),
          Expanded(
            child: _imageSize == null
                ? Center(
                    child: _error == null
                        ? const CircularProgressIndicator()
                        : Text(_error!))
                : LayoutBuilder(builder: (context, constraints) {
                    _displaySize = applyBoxFit(
                            BoxFit.contain, _imageSize!, constraints.biggest)
                        .destination;
                    return Center(
                        child: InteractiveViewer(
                            transformationController: _controller,
                            minScale: 1,
                            maxScale: 8,
                            panEnabled: !_lockPan,
                            child: SizedBox(
                                width: _displaySize.width,
                                height: _displaySize.height,
                                child: Listener(
                                  key:
                                      const ValueKey('legacy-reference-canvas'),
                                  onPointerDown: _onPointerDown,
                                  onPointerMove: _onPointerMove,
                                  onPointerUp: _onPointerUp,
                                  child: Stack(
                                    fit: StackFit.expand,
                                    children: [
                                      Image.memory(_imageBytes,
                                          fit: BoxFit.contain),
                                      Positioned.fill(
                                        child: CustomPaint(
                                          painter: _StickPainter(
                                              points: _points
                                                  .map((p) => Offset(
                                                      p.dx /
                                                          _imageSize!.width *
                                                          _displaySize.width,
                                                      p.dy /
                                                          _imageSize!.height *
                                                          _displaySize.height))
                                                  .toList()),
                                        ),
                                      ),
                                    ],
                                  ),
                                ))));
                  }),
          ),
        ],
      ),
      bottomNavigationBar: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: FilledButton.icon(
            onPressed: length == null ? null : _apply,
            style: FilledButton.styleFrom(
              backgroundColor: AppTheme.primary,
              disabledBackgroundColor: AppTheme.surface3,
              padding: const EdgeInsets.symmetric(vertical: 16),
              shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(16)),
            ),
            icon: Icon(Icons.straighten,
                color: length == null ? AppTheme.muted : Colors.black),
            label: Text(
              length == null ? 'ОТМЕТЬТЕ 2 ТОЧКИ' : 'ВВЕСТИ РАЗМЕР И ПРИМЕНИТЬ',
              style: TextStyle(
                color: length == null ? AppTheme.muted : Colors.black,
                fontWeight: FontWeight.w900,
                letterSpacing: 1.0,
              ),
            ),
          ),
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
      ..color = AppTheme.primary
      ..strokeWidth = 3.0
      ..strokeCap = StrokeCap.round
      ..style = PaintingStyle.stroke;

    final dotPaint = Paint()..color = AppTheme.primary2;
    final shadowPaint = Paint()
      ..color = Colors.black.withOpacity(0.5)
      ..strokeWidth = 5.0
      ..strokeCap = StrokeCap.round
      ..style = PaintingStyle.stroke;

    if (points.length == 2) {
      canvas.drawLine(points[0], points[1], shadowPaint); // Тень для контраста
      canvas.drawLine(points[0], points[1], linePaint);
    }

    for (final pt in points) {
      canvas.drawCircle(pt, 5, Paint()..color = Colors.black);
      canvas.drawCircle(pt, 4, dotPaint);
    }
  }

  @override
  bool shouldRepaint(covariant _StickPainter oldDelegate) => true;
}
