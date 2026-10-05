import 'dart:ui' as ui;

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

/// Tints foreground luminance without painting the black PNG background.
///
/// This changes only display pixels: the source image and exported mask stay
/// untouched. Preserve the source alpha separately so transparent white RGB
/// cannot become foreground when luminance is converted to alpha.
void drawBinaryMaskOverlay(ui.Canvas canvas, ui.Image image, ui.Rect source,
    ui.Rect destination, ui.Color color) {
  canvas.saveLayer(destination, ui.Paint());
  canvas.drawImageRect(
      image,
      source,
      destination,
      ui.Paint()
        ..colorFilter = const ui.ColorFilter.matrix([
          0,
          0,
          0,
          0,
          255,
          0,
          0,
          0,
          0,
          255,
          0,
          0,
          0,
          0,
          255,
          .2126,
          .7152,
          .0722,
          0,
          0,
        ]));
  canvas.drawImageRect(
      image, source, destination, ui.Paint()..blendMode = ui.BlendMode.dstIn);
  canvas.drawRect(
      destination,
      ui.Paint()
        ..color = color
        ..blendMode = ui.BlendMode.srcIn);
  canvas.restore();
}

/// A binary PNG overlay fitted like the original photo in a shared viewport.
class BinaryMaskOverlay extends StatefulWidget {
  final Uint8List bytes;
  final Color color;
  final BoxFit fit;
  const BinaryMaskOverlay(
      {super.key,
      required this.bytes,
      required this.color,
      this.fit = BoxFit.contain});

  @override
  State<BinaryMaskOverlay> createState() => _BinaryMaskOverlayState();
}

class _BinaryMaskOverlayState extends State<BinaryMaskOverlay> {
  ui.Image? _image;
  var _generation = 0;
  var _failed = false;

  @override
  void initState() {
    super.initState();
    _decode();
  }

  @override
  void didUpdateWidget(covariant BinaryMaskOverlay oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (!listEquals(oldWidget.bytes, widget.bytes)) _decode();
  }

  Future<void> _decode() async {
    final generation = ++_generation;
    _image?.dispose();
    _image = null;
    _failed = false;
    ui.Codec? codec;
    try {
      codec = await ui.instantiateImageCodec(widget.bytes);
      final image = (await codec.getNextFrame()).image;
      if (!mounted || generation != _generation) {
        image.dispose();
        return;
      }
      setState(() => _image = image);
    } catch (_) {
      if (mounted && generation == _generation) {
        setState(() => _failed = true);
      }
    } finally {
      codec?.dispose();
    }
  }

  @override
  void dispose() {
    _generation++;
    _image?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => _failed
      ? const Center(
          child: Icon(Icons.broken_image_outlined,
              semanticLabel: 'PNG-маска недоступна'))
      : _image == null
          ? const SizedBox.expand()
          : CustomPaint(
              size: Size.infinite,
              painter: _BinaryMaskPainter(_image!, widget.color, widget.fit));
}

class _BinaryMaskPainter extends CustomPainter {
  final ui.Image image;
  final Color color;
  final BoxFit fit;
  _BinaryMaskPainter(this.image, this.color, this.fit);

  @override
  void paint(Canvas canvas, Size size) {
    final input = Size(image.width.toDouble(), image.height.toDouble());
    final fitted = applyBoxFit(fit, input, size);
    final source =
        Alignment.center.inscribe(fitted.source, Offset.zero & input);
    final destination =
        Alignment.center.inscribe(fitted.destination, Offset.zero & size);
    drawBinaryMaskOverlay(canvas, image, source, destination, color);
  }

  @override
  bool shouldRepaint(covariant _BinaryMaskPainter oldDelegate) =>
      oldDelegate.image != image ||
      oldDelegate.color != color ||
      oldDelegate.fit != fit;
}
