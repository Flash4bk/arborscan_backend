import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';

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
  ui.Image? _image;

  /// Список нарисованных линий (каждая линия — список точек)
  final List<List<Offset>> _paths = [];

  /// Стек для redo
  final List<List<Offset>> _redoPaths = [];

  /// Текущая линия во время рисования
  List<Offset> _currentPath = [];

  /// Параметры кисти
  double _brushSize = 18;
  bool _isEraser = false; // пока только визуальный режим

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

  /// Экспорт маски как PNG (белый фон, чёрные линии там, где маска)
  Future<Uint8List?> _exportMask() async {
    if (_image == null) return null;

    final recorder = ui.PictureRecorder();
    final canvas = Canvas(recorder);

    // Белый фон
    canvas.drawRect(
      Rect.fromLTWH(0, 0, _image!.width.toDouble(), _image!.height.toDouble()),
      Paint()..color = Colors.white,
    );

    // Чёрные линии – маска
    final paint = Paint()
      ..color = Colors.black
      ..strokeWidth = _brushSize
      ..strokeCap = StrokeCap.round
      ..style = PaintingStyle.stroke;

    for (final path in _paths) {
      for (int i = 1; i < path.length; i++) {
        canvas.drawLine(path[i - 1], path[i], paint);
      }
    }

    final picture = recorder.endRecording();
    final img =
        await picture.toImage(_image!.width, _image!.height);
    final data =
        await img.toByteData(format: ui.ImageByteFormat.png);
    return data?.buffer.asUint8List();
  }

  void _startPath(Offset pos) {
    setState(() {
      _currentPath = [pos];
      _paths.add(_currentPath);
      _redoPaths.clear(); // новый штрих — стираем историю redo
    });
  }

  void _updatePath(Offset pos) {
    setState(() => _currentPath.add(pos));
  }

  void _clearAll() {
    setState(() {
      _paths.clear();
      _redoPaths.clear();
    });
  }

  void _undo() {
    if (_paths.isEmpty) return;
    setState(() {
      _redoPaths.add(_paths.removeLast());
    });
  }

  void _redo() {
    if (_redoPaths.isEmpty) return;
    setState(() {
      _paths.add(_redoPaths.removeLast());
    });
  }

  @override
  Widget build(BuildContext context) {
    final img = _image;

    return Scaffold(
      backgroundColor: const Color(0xFFF5F6EC),
      appBar: AppBar(
        title: const Text(
          'Рисование маски',
          style: TextStyle(fontWeight: FontWeight.w700),
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.check_rounded),
            tooltip: 'Сохранить маску',
            onPressed: () async {
              final mask = await _exportMask();
              if (!mounted) return;
              Navigator.pop(context, mask);
            },
          ),
        ],
      ),
      body: img == null
          ? const Center(child: CircularProgressIndicator())
          : LayoutBuilder(
              builder: (ctx, constraints) {
                // Подбираем масштаб, чтобы картинка целиком влезла
                final scaleX = constraints.maxWidth / img.width;
                final scaleY = constraints.maxHeight / img.height;
                final scale = scaleX < scaleY ? scaleX : scaleY;

                final drawW = img.width * scale;
                final drawH = img.height * scale;

                final offX = (constraints.maxWidth - drawW) / 2;
                final offY = (constraints.maxHeight - drawH) / 2;

                return Stack(
                  children: [
                    // ==== Основное полотно ====
                    GestureDetector(
                      onPanStart: (details) {
                        final p = Offset(
                          (details.localPosition.dx - offX) / scale,
                          (details.localPosition.dy - offY) / scale,
                        );
                        _startPath(p);
                      },
                      onPanUpdate: (details) {
                        final p = Offset(
                          (details.localPosition.dx - offX) / scale,
                          (details.localPosition.dy - offY) / scale,
                        );
                        _updatePath(p);
                      },
                      child: CustomPaint(
                        size: Size(
                          constraints.maxWidth,
                          constraints.maxHeight,
                        ),
                        painter: _MaskPainter(
                          image: img,
                          paths: _paths,
                          brushSize: _brushSize,
                          scale: scale,
                          offsetX: offX,
                          offsetY: offY,
                        ),
                      ),
                    ),

                    // ==== Мини-превью маски ====
                    Positioned(
                      left: 16,
                      bottom: 16 + 80,
                      child: ClipRRect(
                        borderRadius: BorderRadius.circular(16),
                        child: Container(
                          width: 80,
                          height: 80,
                          color: Colors.white,
                          child: CustomPaint(
                            painter: _MiniMaskPreviewPainter(
                              image: img,
                              paths: _paths,
                            ),
                          ),
                        ),
                      ),
                    ),

                    // ==== Панель инструментов ====
                    Positioned(
                      left: 0,
                      right: 0,
                      bottom: 16,
                      child: Center(
                        child: AnimatedContainer(
                          duration:
                              const Duration(milliseconds: 200),
                          padding: const EdgeInsets.symmetric(
                            horizontal: 16,
                            vertical: 10,
                          ),
                          decoration: BoxDecoration(
                            color: Colors.white.withOpacity(0.94),
                            borderRadius: BorderRadius.circular(22),
                            boxShadow: const [
                              BoxShadow(
                                color: Colors.black12,
                                blurRadius: 10,
                                offset: Offset(0, 4),
                              ),
                            ],
                          ),
                          child: Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              _toolButton(
                                icon: Icons.brush_rounded,
                                tooltip: 'Кисть',
                                active: !_isEraser,
                                onTap: () =>
                                    setState(() => _isEraser = false),
                              ),
                              const SizedBox(width: 10),
                              _toolButton(
                                icon: Icons.auto_fix_off_rounded,
                                tooltip: 'Ластик (визуальный)',
                                active: _isEraser,
                                onTap: () =>
                                    setState(() => _isEraser = true),
                              ),
                              const SizedBox(width: 10),
                              _toolButton(
                                icon: Icons.undo_rounded,
                                tooltip: 'Отменить',
                                active: false,
                                onTap: _undo,
                              ),
                              const SizedBox(width: 10),
                              _toolButton(
                                icon: Icons.redo_rounded,
                                tooltip: 'Вернуть',
                                active: false,
                                onTap: _redo,
                              ),
                              const SizedBox(width: 10),
                              _toolButton(
                                icon: Icons.delete_outline_rounded,
                                tooltip: 'Очистить всё',
                                active: false,
                                onTap: _clearAll,
                              ),
                              const SizedBox(width: 10),
                              _toolButton(
                                icon: _brushSize <= 18
                                    ? Icons.format_size_rounded
                                    : Icons.format_bold_rounded,
                                tooltip: 'Толщина линии',
                                active: false,
                                onTap: () {
                                  setState(() {
                                    _brushSize =
                                        _brushSize <= 18 ? 28 : 18;
                                  });
                                },
                              ),
                            ],
                          ),
                        ),
                      ),
                    ),
                  ],
                );
              },
            ),
    );
  }

  Widget _toolButton({
    required IconData icon,
    required String tooltip,
    required bool active,
    required VoidCallback onTap,
  }) {
    return Tooltip(
      message: tooltip,
      child: GestureDetector(
        onTap: onTap,
        child: AnimatedScale(
          scale: active ? 1.05 : 1.0,
          duration: const Duration(milliseconds: 150),
          child: AnimatedContainer(
            duration: const Duration(milliseconds: 150),
            width: 44,
            height: 44,
            decoration: BoxDecoration(
              color: active
                  ? const Color(0xFF2F6B3A)
                  : const Color(0xFFF0F0F0),
              borderRadius: BorderRadius.circular(14),
            ),
            child: Icon(
              icon,
              color: active ? Colors.white : Colors.black87,
              size: 22,
            ),
          ),
        ),
      ),
    );
  }
}

/// Рисует фото + полупрозрачную маску поверх
class _MaskPainter extends CustomPainter {
  final ui.Image image;
  final List<List<Offset>> paths;
  final double brushSize;
  final double scale;
  final double offsetX;
  final double offsetY;

  _MaskPainter({
    required this.image,
    required this.paths,
    required this.brushSize,
    required this.scale,
    required this.offsetX,
    required this.offsetY,
  });

  @override
  void paint(Canvas canvas, Size size) {
    // Рисуем исходное изображение
    final dstRect = Rect.fromLTWH(
      offsetX,
      offsetY,
      image.width * scale,
      image.height * scale,
    );
    final srcRect = Rect.fromLTWH(
      0,
      0,
      image.width.toDouble(),
      image.height.toDouble(),
    );

    canvas.drawImageRect(image, srcRect, dstRect, Paint());

    // Полупрозрачная маска
    final maskPaint = Paint()
      ..color = const Color(0xFF2ECC71).withOpacity(0.55)
      ..strokeWidth = brushSize * scale * 0.6
      ..strokeCap = StrokeCap.round
      ..style = PaintingStyle.stroke;

    for (final path in paths) {
      for (int i = 1; i < path.length; i++) {
        final p1 = Offset(
          path[i - 1].dx * scale + offsetX,
          path[i - 1].dy * scale + offsetY,
        );
        final p2 = Offset(
          path[i].dx * scale + offsetX,
          path[i].dy * scale + offsetY,
        );
        canvas.drawLine(p1, p2, maskPaint);
      }
    }
  }

  @override
  bool shouldRepaint(covariant _MaskPainter oldDelegate) {
    return oldDelegate.paths != paths ||
        oldDelegate.brushSize != brushSize;
  }
}

/// Маленькое превью маски (без фото, только контуры)
class _MiniMaskPreviewPainter extends CustomPainter {
  final ui.Image image;
  final List<List<Offset>> paths;

  _MiniMaskPreviewPainter({
    required this.image,
    required this.paths,
  });

  @override
  void paint(Canvas canvas, Size size) {
    // Белый фон
    canvas.drawRect(
      Offset.zero & size,
      Paint()..color = Colors.white,
    );

    // Рамка
    canvas.drawRect(
      Offset.zero & size,
      Paint()
        ..color = Colors.black12
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1,
    );

    // Масштабируем пути в мини-окно
    final scaleX = size.width / image.width;
    final scaleY = size.height / image.height;
    final scale = scaleX < scaleY ? scaleX : scaleY;

    final offX = (size.width - image.width * scale) / 2;
    final offY = (size.height - image.height * scale) / 2;

    final paint = Paint()
      ..color = const Color(0xFF2ECC71)
      ..strokeWidth = 2
      ..strokeCap = StrokeCap.round
      ..style = PaintingStyle.stroke;

    for (final path in paths) {
      for (int i = 1; i < path.length; i++) {
        final p1 = Offset(
          path[i - 1].dx * scale + offX,
          path[i - 1].dy * scale + offY,
        );
        final p2 = Offset(
          path[i].dx * scale + offX,
          path[i].dy * scale + offY,
        );
        canvas.drawLine(p1, p2, paint);
      }
    }
  }

  @override
  bool shouldRepaint(covariant _MiniMaskPreviewPainter oldDelegate) {
    return oldDelegate.paths != paths;
  }
}
