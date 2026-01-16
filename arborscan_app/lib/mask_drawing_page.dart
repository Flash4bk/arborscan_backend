import 'dart:convert';
import 'dart:math';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';

class MaskDrawingPage extends StatefulWidget {
  final String? originalImageBase64;
  final String? initialMaskBase64;
  final String? aiMaskBase64;

  const MaskDrawingPage({
    Key? key,
    this.originalImageBase64,
    this.initialMaskBase64,
    this.aiMaskBase64,
  }) : super(key: key);

  @override
  State<MaskDrawingPage> createState() => _MaskDrawingPageState();
}

class _MaskDrawingPageState extends State<MaskDrawingPage> {
  late final Uint8List _imageBytes;

  ui.Image? _image;
  Size? _imageSize;
  ui.Image? _aiMaskImage;

  final TransformationController _controller = TransformationController();
  final List<Offset> _points = [];
  int? _dragIndex;
  bool _closed = false;
  bool _finishing = false;
  Size? _drawSize;

  void _setFinishing(bool v) {
    if (!mounted) return;
    setState(() => _finishing = v);
  }

  final Set<int> _activePointers = <int>{};
  int? _primaryPointer;
  bool _inPinch = false;

  bool _tapCandidate = false;
  Offset? _tapDownLocal;
  Offset? _tapDownScene;

  bool _popScheduled = false;
  bool _showingPreview = false;

  /// Safe wrapper around Navigator.pop to avoid Flutter assertion
  /// `!_debugLocked` when a pop is triggered while the Navigator is
  /// processing another pop/push.
  void _safePop<T>(BuildContext ctx, [T? result]) {
    if (_popScheduled) return;
    _popScheduled = true;
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _popScheduled = false;
      if (!mounted) return;
      final nav = Navigator.of(ctx);
      if (!nav.canPop()) return;
      try {
        nav.pop(result);
      } catch (_) {
        // Если Navigator всё ещё залочен, пробуем ещё раз на следующем кадре.
        WidgetsBinding.instance.addPostFrameCallback((_) {
          if (!mounted) return;
          final nav2 = Navigator.of(ctx);
          if (nav2.canPop()) {
            try {
              nav2.pop(result);
            } catch (_) {
              // no-op
            }
          }
        });
      }
    });
  }

  static const double _tapSlopPx = 8.0;
  static const double _closeHitSlopPx = 18.0;

  bool _isCloseTapToFirstPoint({required Offset tapLocal}) {
    if (_points.length < 3) return false;
    final firstLocal = MatrixUtils.transformPoint(_controller.value, _points.first);
    return (firstLocal - tapLocal).distance <= _closeHitSlopPx;
  }

  @override
  void initState() {
    super.initState();
    if (widget.originalImageBase64 == null) {
      throw Exception('originalImageBase64 is required');
    }
    _imageBytes = base64Decode(widget.originalImageBase64!);
    _loadImage();
    
    if (widget.aiMaskBase64 != null && widget.aiMaskBase64!.isNotEmpty) {
      _loadAiMask();
    }
    
    // Загрузка начальной маски пользователя если есть
    if (widget.initialMaskBase64 != null && widget.initialMaskBase64!.isNotEmpty) {
      _loadInitialMask();
    }
  }

  Future<void> _loadImage() async {
    final codec = await ui.instantiateImageCodec(_imageBytes);
    final frame = await codec.getNextFrame();
    setState(() {
      _image = frame.image;
      _imageSize = Size(frame.image.width.toDouble(), frame.image.height.toDouble());
    });
  }

  Future<void> _loadAiMask() async {
    try {
      final aiMaskBytes = base64Decode(widget.aiMaskBase64!);
      final codec = await ui.instantiateImageCodec(aiMaskBytes);
      final frame = await codec.getNextFrame();
      setState(() {
        _aiMaskImage = frame.image;
      });
    } catch (e) {
      print('Ошибка загрузки маски ИИ: $e');
    }
  }

  Future<void> _loadInitialMask() async {
    try {
      final maskBytes = base64Decode(widget.initialMaskBase64!);
      final codec = await ui.instantiateImageCodec(maskBytes);
      final frame = await codec.getNextFrame();
      // Для начальной маски пользователя можно создать изображение контура
      // или просто пропустить, так как у нас уже есть точки
    } catch (e) {
      print('Ошибка загрузки начальной маски: $e');
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  double get _scale => _controller.value.getMaxScaleOnAxis().clamp(1.0, 10.0);

  Offset _toScene(Offset viewportLocal) => _controller.toScene(viewportLocal);

  int? _hitTest(Offset scenePx) {
    final threshold = 14.0 / _scale;
    for (int i = 0; i < _points.length; i++) {
      if ((_points[i] - scenePx).distance <= threshold) return i;
    }
    return null;
  }

  Future<Uint8List?> _createPreviewImage() async {
    if (_image == null || _drawSize == null || _points.length < 3) return null;

    final recorder = ui.PictureRecorder();
    final canvas = Canvas(recorder);
    
    const previewWidth = 300.0;
    final previewHeight = previewWidth * (_imageSize!.height / _imageSize!.width);
    final previewSize = Size(previewWidth, previewHeight);
    
    final scaleX = previewWidth / _drawSize!.width;
    final scaleY = previewHeight / _drawSize!.height;
    
    final paint = Paint();
    canvas.drawImageRect(
      _image!,
      Rect.fromLTWH(0, 0, _image!.width.toDouble(), _image!.height.toDouble()),
      Rect.fromLTWH(0, 0, previewWidth, previewHeight),
      paint,
    );
    
    if (_aiMaskImage != null) {
      canvas.drawImageRect(
        _aiMaskImage!,
        Rect.fromLTWH(0, 0, _aiMaskImage!.width.toDouble(), _aiMaskImage!.height.toDouble()),
        Rect.fromLTWH(0, 0, previewWidth, previewHeight),
        Paint()
          ..colorFilter = ColorFilter.mode(
            Colors.green.withOpacity(0.3),
            BlendMode.srcATop,
          ),
      );
    }
    
    if (_points.length >= 3) {
      final path = Path();
      path.moveTo(_points.first.dx * scaleX, _points.first.dy * scaleY);
      
      for (int i = 1; i < _points.length; i++) {
        path.lineTo(_points[i].dx * scaleX, _points[i].dy * scaleY);
      }
      path.close();
      
      canvas.drawPath(
        path,
        Paint()
          ..color = Colors.blue.withOpacity(0.4)
          ..style = PaintingStyle.fill,
      );
      
      canvas.drawPath(
        path,
        Paint()
          ..color = Colors.blue
          ..style = PaintingStyle.stroke
          ..strokeWidth = 2,
      );
      
      for (final p in _points) {
        canvas.drawCircle(
          Offset(p.dx * scaleX, p.dy * scaleY),
          3,
          Paint()..color = Colors.cyanAccent,
        );
      }
      
      canvas.drawCircle(
        Offset(_points.first.dx * scaleX, _points.first.dy * scaleY),
        5,
        Paint()..color = Colors.yellowAccent,
      );
    }
    
    const legendHeight = 40.0;
    final legendRect = Rect.fromLTWH(0, previewHeight - legendHeight, previewWidth, legendHeight);
    canvas.drawRect(
      legendRect,
      Paint()..color = Colors.black.withOpacity(0.7),
    );
    
    final textStyle = TextStyle(
      color: Colors.white,
      fontSize: 12,
    );
    
    final textPainter = TextPainter(
      text: TextSpan(
        text: 'Зелёный: маска ИИ | Синий: моя маска',
        style: textStyle,
      ),
      textDirection: TextDirection.ltr,
    );
    textPainter.layout();
    textPainter.paint(
      canvas,
      Offset(
        (previewWidth - textPainter.width) / 2,
        previewHeight - legendHeight + 14,
      ),
    );
    
    final previewPicture = recorder.endRecording();
    final previewImage = await previewPicture.toImage(
      previewWidth.toInt(),
      previewHeight.toInt(),
    );
    
    final bytes = await previewImage.toByteData(format: ui.ImageByteFormat.png);
    return bytes?.buffer.asUint8List();
  }

  Future<void> _showPreviewDialog() async {
    Uint8List? previewBytes;
    try {
      previewBytes = await _createPreviewImage();
    } catch (_) {
      previewBytes = null;
    }

    // Если превью не удалось, не блокируем пользователя: сохраняем без диалога.
    if (previewBytes == null) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Не удалось создать превью. Сохраняю маску без просмотра.')),
      );
      await _saveMaskAndExit();
      return;
    }

    bool? shouldConfirm;
    try {
      shouldConfirm = await showDialog<bool>(
        context: context,
        barrierDismissible: false,
        builder: (ctx) {
          return AlertDialog(
            title: const Text('Превью маски'),
            content: SingleChildScrollView(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Image.memory(previewBytes!),
                  const SizedBox(height: 16),
                  const Text(
                    'Проверьте выделение дерева.\nЗелёный цвет — маска ИИ, синий — ваша маска.',
                    textAlign: TextAlign.center,
                    style: TextStyle(fontSize: 14),
                  ),
                  const SizedBox(height: 16),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    children: [
                      ElevatedButton.icon(
                        onPressed: () => _safePop<bool>(ctx, false),
                        icon: const Icon(Icons.edit, size: 18),
                        label: const Text('Исправить'),
                        style: ElevatedButton.styleFrom(backgroundColor: Colors.orange),
                      ),
                      ElevatedButton.icon(
                        onPressed: () => _safePop<bool>(ctx, true),
                        icon: const Icon(Icons.check, size: 18),
                        label: const Text('Подтвердить'),
                        style: ElevatedButton.styleFrom(backgroundColor: Colors.green),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          );
        },
      );
    } finally {
      // Если диалог закрыли не подтверждением, возвращаем управление пользователю.
      if (mounted && shouldConfirm != true) {
        _setFinishing(false);
      }
    }

    if (!mounted) return;

    if (shouldConfirm == true) {
      await _saveMaskAndExit();
    }
  }

  Future<void> _saveMaskAndExit() async {
    try {
      final recorder = ui.PictureRecorder();
      final canvas = Canvas(recorder);

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
      if (!mounted || bytes == null) {
        setState(() => _finishing = false);
        return;
      }

      final result = {
        "mask_png_base64": base64Encode(bytes.buffer.asUint8List()),
        "points": _points
            .map((p) => {
                  "x": p.dx / _drawSize!.width,
                  "y": p.dy / _drawSize!.height,
                })
            .toList(),
        "closed": _closed,
      };

      if (!mounted) return;
      setState(() => _finishing = false);
      _safePop<Map<String, dynamic>>(context, result);
    } catch (e) {
      if (mounted) {
        setState(() => _finishing = false);
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Ошибка при формировании маски: $e')),
        );
      }
    }
  }

  Future<void> _finish() async {
    if (_finishing) return;
    if (_imageSize == null || _drawSize == null) return;

    if (_points.length < 3) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Нужно минимум 3 точки для маски.')),
      );
      return;
    }

    if (!_closed) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Сначала замкните контур (кнопка "Замкнуть" или тап по первой точки).')),
      );
      return;
    }

    _setFinishing(true);

    try {
      // Небольшая задержка, чтобы UI успел отрисовать overlay "Подготовка...".
      await Future.delayed(const Duration(milliseconds: 80));
      await _showPreviewDialog();
    } catch (e) {
      // Важно: не оставлять экран в состоянии "залипшего" _finishing.
      _setFinishing(false);
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Не удалось подготовить маску: $e')),
      );
    }
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

    final canConfirm = _closed && _points.length >= 3 && !_finishing;

    return WillPopScope(
      onWillPop: () async {
        if (!_finishing) return true;

        // Чтобы не получить «залипание» UI при ошибке в завершении,
        // разрешаем пользователю отменить «обработку».
        final cancel = await showDialog<bool>(
          context: context,
          builder: (ctx) => AlertDialog(
            title: const Text('Завершение выполняется'),
            content: const Text('Подождать или отменить и выйти?'),
            actions: [
              TextButton(
                onPressed: () => _safePop<bool>(ctx, false),
                child: const Text('Подождать'),
              ),
              TextButton(
                onPressed: () => _safePop<bool>(ctx, true),
                child: const Text('Отменить'),
              ),
            ],
          ),
        );

        if (cancel == true) {
          _setFinishing(false);
          return true;
        }
        return false;
      },
      child: Scaffold(
        appBar: AppBar(
          title: const Text('Исправление маски'),
          actions: [
            if (_points.length >= 3 && !_finishing)
              IconButton(
                tooltip: 'Быстрый превью',
                icon: const Icon(Icons.visibility),
                onPressed: () async {
                  final previewBytes = await _createPreviewImage();
                  if (previewBytes != null && mounted) {
                    showDialog(
                      context: context,
                      builder: (ctx) => AlertDialog(
                        title: const Text('Превью маски'),
                        content: Image.memory(previewBytes),
                        actions: [
                          TextButton(
                            onPressed: () => _safePop(ctx),
                            child: const Text('Закрыть'),
                          ),
                        ],
                      ),
                    );
                  }
                },
              ),
            IconButton(
              tooltip: 'Замкнуть контур',
              icon: const Icon(Icons.link),
              onPressed: _points.length >= 3
                  ? () => setState(() => _closed = true)
                  : null,
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

            _drawSize = drawSize;

            return Stack(
              children: [
                Center(
                  child: SizedBox(
                    width: drawSize.width,
                    height: drawSize.height,
                    child: Listener(
                      behavior: HitTestBehavior.translucent,
                      onPointerDown: (e) {
                        _activePointers.add(e.pointer);

                        if (_activePointers.length >= 2) {
                          _inPinch = true;
                          _dragIndex = null;
                          _primaryPointer = null;
                          _resetTapState();
                          return;
                        }

                        _primaryPointer = e.pointer;
                        final scene = _toScene(e.localPosition);
                        final hit = _hitTest(scene);
                        if (hit != null) {
                          _dragIndex = hit;
                          _resetTapState();
                          return;
                        }

                        if (_closed) {
                          _resetTapState();
                          return;
                        }

                        _tapCandidate = true;
                        _tapDownLocal = e.localPosition;
                        _tapDownScene = scene;
                      },
                      onPointerMove: (e) {
                        if (_primaryPointer != e.pointer) return;

                        final scene = _toScene(e.localPosition);

                        if (_dragIndex != null) {
                          setState(() {
                            _points[_dragIndex!] = scene;
                          });
                          return;
                        }

                        if (_tapCandidate && _tapDownLocal != null) {
                          final localDist =
                              (e.localPosition - _tapDownLocal!).distance;
                          if (localDist > _tapSlopPx) {
                            _tapCandidate = false;
                          }
                        }
                      },
                      onPointerUp: (e) {
                        _activePointers.remove(e.pointer);

                        if (_activePointers.isEmpty) {
                          _inPinch = false;
                        }

                        if (_primaryPointer == e.pointer) {
                          if (_tapCandidate &&
                              !_inPinch &&
                              !_closed &&
                              _tapDownScene != null &&
                              _tapDownScene != null &&
                              _tapDownLocal != null) {
                            final tapLocal = _tapDownLocal!;

                            if (_isCloseTapToFirstPoint(tapLocal: tapLocal)) {
                              setState(() {
                                _closed = true;
                              });
                            } else {
                              setState(() {
                                _points.add(_tapDownScene!);
                              });
                            }
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
                        panEnabled: _dragIndex == null,
                        scaleEnabled: true,
                        child: CustomPaint(
                          size: drawSize,
                          painter: _EnhancedMaskPainter(
                            image: _image!,
                            points: _points,
                            closed: _closed,
                            aiMaskImage: _aiMaskImage,
                          ),
                        ),
                      ),
                    ),
                  ),
                ),
                if (_aiMaskImage != null)
                  Positioned(
                    top: 10,
                    left: 10,
                    child: Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 12,
                        vertical: 6,
                      ),
                      decoration: BoxDecoration(
                        color: Colors.black.withOpacity(0.7),
                        borderRadius: BorderRadius.circular(20),
                      ),
                      child: Row(
                        children: [
                          Container(
                            width: 12,
                            height: 12,
                            decoration: BoxDecoration(
                              color: Colors.green,
                              borderRadius: BorderRadius.circular(6),
                            ),
                          ),
                          const SizedBox(width: 8),
                          const Text(
                            'Маска ИИ',
                            style: TextStyle(
                              color: Colors.white,
                              fontSize: 12,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                if (_finishing)
                  Positioned.fill(
                    child: Container(
                      color: Colors.black.withOpacity(0.5),
                      child: const Center(
                        child: CircularProgressIndicator(
                          valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                        ),
                      ),
                    ),
                  ),
              ],
            );
          },
        ),
        floatingActionButton: _finishing
            ? FloatingActionButton.extended(
                onPressed: null,
                icon: const SizedBox(
                  width: 20,
                  height: 20,
                  child: CircularProgressIndicator(
                    strokeWidth: 2,
                    valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                  ),
                ),
                label: const Text('Подготовка…'),
              )
            : FloatingActionButton.extended(
                onPressed: canConfirm ? _finish : null,
                icon: const Icon(Icons.check),
                label: const Text('Подтвердить'),
              ),
      ),
    );
  }
}

class _EnhancedMaskPainter extends CustomPainter {
  final ui.Image image;
  final List<Offset> points;
  final bool closed;
  final ui.Image? aiMaskImage;

  _EnhancedMaskPainter({
    required this.image,
    required this.points,
    required this.closed,
    this.aiMaskImage,
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

    if (aiMaskImage != null) {
      canvas.drawImageRect(
        aiMaskImage!,
        Rect.fromLTWH(
          0,
          0,
          aiMaskImage!.width.toDouble(),
          aiMaskImage!.height.toDouble(),
        ),
        Rect.fromLTWH(0, 0, size.width, size.height),
        Paint()
          ..colorFilter = ColorFilter.mode(
            Colors.green.withOpacity(0.25),
            BlendMode.srcATop,
          ),
      );
    }

    if (points.isEmpty) return;

    final path = Path()..moveTo(points.first.dx, points.first.dy);
    for (int i = 1; i < points.length; i++) {
      path.lineTo(points[i].dx, points[i].dy);
    }
    if (closed && points.length >= 3) {
      path.close();
      
      canvas.drawPath(
        path,
        Paint()
          ..color = Colors.blue.withOpacity(0.15)
          ..style = PaintingStyle.fill,
      );
    }

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

    canvas.drawCircle(
      points.first,
      6,
      Paint()..color = Colors.yellowAccent.withOpacity(0.85),
    );
  }

  @override
  bool shouldRepaint(covariant _EnhancedMaskPainter oldDelegate) =>
      oldDelegate.points != points ||
      oldDelegate.closed != closed ||
      oldDelegate.aiMaskImage != aiMaskImage;
}