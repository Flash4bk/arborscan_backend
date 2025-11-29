import 'dart:convert';
import 'dart:typed_data';
import 'dart:ui' as ui;
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
  ui.Image? _image;
  List<Offset> _points = [];
  final List<List<Offset>> _paths = [];
  bool _loaded = false;

  @override
  void initState() {
    super.initState();
    _loadOriginal();
  }

  Future<void> _loadOriginal() async {
    final bytes = base64Decode(widget.originalImageBase64);
    final codec = await ui.instantiateImageCodec(bytes);
    final frame = await codec.getNextFrame();
    setState(() {
      _image = frame.image;
      _loaded = true;
    });
  }

  void _onPanStart(DragStartDetails d, BoxConstraints c) {
    if (_image == null) return;

    final local = _getLocalPoint(d.localPosition, c);
    _points = [local];
    setState(() {});
  }

  void _onPanUpdate(DragUpdateDetails d, BoxConstraints c) {
    if (_image == null) return;

    final local = _getLocalPoint(d.localPosition, c);
    _points.add(local);
    setState(() {});
  }

  void _onPanEnd(DragEndDetails d) {
    if (_points.isNotEmpty) {
      _paths.add(List.of(_points));
    }
    _points.clear();
    setState(() {});
  }

  Offset _getLocalPoint(Offset global, BoxConstraints c) {
    final imgW = _image!.width.toDouble();
    final imgH = _image!.height.toDouble();

    final ratio = imgH / imgW;
    final containerRatio = c.maxHeight / c.maxWidth;

    double renderW, renderH;

    if (ratio > containerRatio) {
      renderH = c.maxHeight;
      renderW = renderH / ratio;
    } else {
      renderW = c.maxWidth;
      renderH = renderW * ratio;
    }

    final dx = (c.maxWidth - renderW) / 2;
    final dy = (c.maxHeight - renderH) / 2;

    final x = ((global.dx - dx) / renderW) * imgW;
    final y = ((global.dy - dy) / renderH) * imgH;

    return Offset(x.clamp(0, imgW), y.clamp(0, imgH));
  }

  Future<void> _saveMask() async {
    if (_image == null) return;

    final recorder = ui.PictureRecorder();
    final canvas = Canvas(recorder);
    final paint = Paint()
      ..color = Colors.white
      ..strokeWidth = 30
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;

    canvas.drawRect(
      Rect.fromLTWH(0, 0, _image!.width.toDouble(), _image!.height.toDouble()),
      Paint()..color = Colors.black,
    );

    for (final path in _paths) {
      for (int i = 0; i < path.length - 1; i++) {
        canvas.drawLine(path[i], path[i + 1], paint);
      }
    }

    if (_points.isNotEmpty) {
      for (int i = 0; i < _points.length - 1; i++) {
        canvas.drawLine(_points[i], _points[i + 1], paint);
      }
    }

    final pic = recorder.endRecording();
    final maskImage = await pic.toImage(_image!.width, _image!.height);

    final byteData = await maskImage.toByteData(format: ui.ImageByteFormat.png);
    if (byteData == null) return;

    final b64 = base64Encode(byteData.buffer.asUint8List());

    if (!mounted) return;
    Navigator.pop(context, b64);
  }

  @override
  Widget build(BuildContext context) {
    if (!_loaded || _image == null) {
      return Scaffold(
        appBar: AppBar(title: const Text("Маска")),
        body: const Center(child: CircularProgressIndicator()),
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text("Рисование маски"),
      ),
      body: LayoutBuilder(
        builder: (context, c) {
          return GestureDetector(
            onPanStart: (d) => _onPanStart(d, c),
            onPanUpdate: (d) => _onPanUpdate(d, c),
            onPanEnd: _onPanEnd,
            child: Stack(
              children: [
                Center(
                  child: CustomPaint(
                    size: Size(c.maxWidth, c.maxHeight),
                    painter: _ImagePainter(
                      image: _image!,
                      paths: _paths,
                      currentPoints: _points,
                    ),
                  ),
                ),
              ],
            ),
          );
        },
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: _saveMask,
        icon: const Icon(Icons.check),
        label: const Text("Сохранить"),
      ),
    );
  }
}

class _ImagePainter extends CustomPainter {
  final ui.Image image;
  final List<List<Offset>> paths;
  final List<Offset> currentPoints;

  _ImagePainter({
    required this.image,
    required this.paths,
    required this.currentPoints,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final imgW = image.width.toDouble();
    final imgH = image.height.toDouble();
    final ratio = imgH / imgW;
    final containerRatio = size.height / size.width;

    double renderW, renderH;

    if (ratio > containerRatio) {
      renderH = size.height;
      renderW = renderH / ratio;
    } else {
      renderW = size.width;
      renderH = renderW * ratio;
    }

    final dx = (size.width - renderW) / 2;
    final dy = (size.height - renderH) / 2;

    final rect = Rect.fromLTWH(dx, dy, renderW, renderH);
    final paintMask = Paint()
      ..color = Colors.greenAccent.withOpacity(0.7)
      ..strokeWidth = 6
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;

    canvas.drawImageRect(
      image,
      Rect.fromLTWH(0, 0, imgW, imgH),
      rect,
      Paint(),
    );

    // Scale factor
    final scaleX = renderW / imgW;
    final scaleY = renderH / imgH;

    for (final path in paths) {
      for (int i = 0; i < path.length - 1; i++) {
        canvas.drawLine(
          Offset(path[i].dx * scaleX + dx, path[i].dy * scaleY + dy),
          Offset(path[i + 1].dx * scaleX + dx, path[i + 1].dy * scaleY + dy),
          paintMask,
        );
      }
    }

    for (int i = 0; i < currentPoints.length - 1; i++) {
      canvas.drawLine(
        Offset(currentPoints[i].dx * scaleX + dx,
            currentPoints[i].dy * scaleY + dy),
        Offset(currentPoints[i + 1].dx * scaleX + dx,
            currentPoints[i + 1].dy * scaleY + dy),
        paintMask,
      );
    }
  }

  @override
  bool shouldRepaint(_) => true;
}
