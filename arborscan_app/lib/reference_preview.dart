import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as im;
import 'combined_reference.dart';
import 'corrections_service.dart';
import 'photo_markup_style.dart';
import 'package:crypto/crypto.dart';

Uint8List? savedTreeMask(Map<String, dynamic>? report) {
  final images = report?['images'];
  final encoded = images is Map ? images['mask_image_base64'] : null;
  if (encoded is! String || encoded.isEmpty) return null;
  try {
    return base64Decode(encoded);
  } catch (_) {
    return null;
  }
}

/// A binary tree mask is a separate visual layer, never a crown/trunk mask.
Uint8List? treeMaskOverlay(Uint8List? bytes, int width, int height,
    {bool binary = false}) {
  if (bytes == null ||
      bytes.length > 15 * 1024 * 1024 ||
      width <= 0 ||
      height <= 0 ||
      width * height > 25000000) {
    return null;
  }
  try {
    final decoder = im.findDecoderForData(bytes);
    final header = decoder?.startDecode(bytes);
    final preview = modelPreviewSize(width, height);
    if (header == null ||
        !((header.width == width && header.height == height) ||
            (!binary &&
                header.width == preview[0] &&
                header.height == preview[1]))) {
      return null;
    }
    final mask = decoder!.decodeFrame(0)!;
    // Bound preview memory independently of the original coordinate system.
    final scaled = mask.width > 1000 || mask.height > 1000
        ? im.copyResize(mask,
            width: mask.width >= mask.height ? 1000 : null,
            height: mask.height > mask.width ? 1000 : null,
            interpolation: im.Interpolation.nearest)
        : mask;
    final overlay =
        im.Image(width: scaled.width, height: scaled.height, numChannels: 4);
    for (final p in scaled) {
      if (!binary && scaled.numChannels == 4
          ? p.a > 0
          : p.r > 127 || p.g > 127 || p.b > 127) {
        overlay.setPixelRgba(p.x, p.y, 32, 146, 88, 110);
      }
    }
    return Uint8List.fromList(im.encodePng(overlay));
  } catch (_) {
    return null;
  }
}

class ReferencePreview extends StatefulWidget {
  final Uint8List image;
  final int width, height;
  final Map<String, dynamic>? report, reference;
  final String? correctionId;
  final Offset? tap;
  final TransformationController? controller;
  final bool showHandles;
  final bool showLayerControls;
  final Set<String> hiddenLayers;
  final ValueChanged<Set<String>>? onHiddenLayersChanged;
  final double handleSizeFactor;
  const ReferencePreview(
      {super.key,
      required this.image,
      required this.width,
      required this.height,
      this.report,
      this.reference,
      this.correctionId,
      this.tap,
      this.controller,
      this.showHandles = true,
      this.showLayerControls = true,
      this.hiddenLayers = const {},
      this.onHiddenLayersChanged,
      this.handleSizeFactor = .75});
  @override
  State<ReferencePreview> createState() => _ReferencePreviewState();
}

class _ReferencePreviewState extends State<ReferencePreview> {
  Uint8List? _mask, _overlay;
  bool _invalidPin = false;
  late Set<String> _hiddenLayers = Set.of(widget.hiddenLayers);
  @override
  void initState() {
    super.initState();
    _refresh();
  }

  @override
  void didUpdateWidget(ReferencePreview oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.image != oldWidget.image ||
        widget.hiddenLayers != oldWidget.hiddenLayers) {
      _hiddenLayers = Set.of(widget.hiddenLayers);
    }
    if (oldWidget.report != widget.report ||
        oldWidget.width != widget.width ||
        oldWidget.height != widget.height ||
        oldWidget.image != widget.image ||
        oldWidget.correctionId != widget.correctionId) {
      _refresh();
    }
  }

  Map<String, String> get _layers => {
        if (_overlay != null) 'mask': 'Контур дерева',
        for (final entry in ReferencePreviewPainter.labels.entries)
          if (widget.reference?[entry.key] is List &&
              (widget.reference![entry.key] as List).isNotEmpty)
            entry.key: entry.value,
        if (widget.tap != null) 'selection': 'Точка выбора дерева',
      };

  Future<void> _chooseLayers() async {
    final epoch = CorrectionsService.authChanges.value;
    final layers = _layers;
    await showModalBottomSheet<void>(
        context: context,
        isScrollControlled: true,
        builder: (sheetContext) => StatefulBuilder(
            builder: (sheetContext, updateSheet) => SafeArea(
                    child: SingleChildScrollView(
                        child:
                            Column(mainAxisSize: MainAxisSize.min, children: [
                  const Padding(
                      padding: EdgeInsets.all(16),
                      child: Text('Слои фото',
                          style: TextStyle(
                              fontSize: 20, fontWeight: FontWeight.bold))),
                  for (final entry in layers.entries)
                    CheckboxListTile(
                        key: ValueKey('photo-layer-${entry.key}'),
                        title: Text(entry.value),
                        value: !_hiddenLayers.contains(entry.key),
                        onChanged: (visible) {
                          if (!mounted ||
                              epoch != CorrectionsService.authChanges.value) {
                            Navigator.pop(sheetContext);
                            return;
                          }
                          setState(() {
                            _hiddenLayers = Set.of(_hiddenLayers);
                            if (visible == true) {
                              _hiddenLayers.remove(entry.key);
                            } else {
                              _hiddenLayers.add(entry.key);
                            }
                          });
                          updateSheet(() {});
                          widget.onHiddenLayersChanged
                              ?.call(Set.of(_hiddenLayers));
                        }),
                  const Padding(
                      padding:
                          EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                      child: Text(
                          'Выбор действует в этом просмотре и PDF. Разметка и размеры сохраняются полностью.')),
                  TextButton(
                      onPressed: () => Navigator.pop(sheetContext),
                      child: const Text('Готово')),
                ])))));
  }

  void _refresh() {
    _mask = savedTreeMask(widget.report);
    var binary = false;
    _invalidPin = false;
    final pinned = widget.report?['contour_preview'];
    if (pinned is Map) {
      _mask = null;
      _invalidPin = true;
      binary = true;
    }
    if (pinned is Map &&
        widget.correctionId != null &&
        pinned['correction_id'] == widget.correctionId &&
        pinned['width'] == widget.width &&
        pinned['height'] == widget.height &&
        pinned['photo_sha256'] == sha256.convert(widget.image).toString()) {
      try {
        _mask = base64Decode(pinned['mask_png_base64']);
        binary = true;
        _invalidPin = false;
      } catch (_) {
        _mask = null;
      }
    }
    _overlay =
        treeMaskOverlay(_mask, widget.width, widget.height, binary: binary);
    if (binary && _overlay == null) {
      _invalidPin = true;
    }
  }

  @override
  Widget build(BuildContext context) => AspectRatio(
      aspectRatio: widget.width / widget.height,
      child: Stack(fit: StackFit.expand, children: [
        Image.memory(widget.image, fit: BoxFit.fill),
        if (_overlay != null && !_hiddenLayers.contains('mask'))
          Image.memory(_overlay!, fit: BoxFit.fill),
        CustomPaint(
            painter: ReferencePreviewPainter(widget.reference, widget.tap,
                controller: widget.controller,
                showHandles: widget.showHandles,
                hiddenLayers: _hiddenLayers,
                handleSizeFactor: widget.handleSizeFactor)),
        if (widget.showLayerControls && _layers.isNotEmpty)
          Positioned(
              top: 4,
              right: 4,
              child: Material(
                  color: Theme.of(context).colorScheme.surface,
                  borderRadius: BorderRadius.circular(24),
                  child: IconButton(
                      tooltip: 'Слои фото',
                      onPressed: _chooseLayers,
                      icon: const Icon(Icons.layers_outlined)))),
        if (_invalidPin)
          const Positioned(
              bottom: 0,
              left: 0,
              right: 0,
              child: ColoredBox(
                  color: Colors.white,
                  child: Text(
                      'Связанный контур не соответствует фото или выбранной ревизии.'))),
      ]));
}

class ReferencePreviewPainter extends CustomPainter {
  final Map<String, dynamic>? reference;
  final Offset? tap;
  final TransformationController? controller;
  final bool showHandles;
  final Set<String> hiddenLayers;
  final double handleSizeFactor;
  ReferencePreviewPainter(this.reference, this.tap,
      {this.controller,
      this.showHandles = true,
      this.hiddenLayers = const {},
      this.handleSizeFactor = 1})
      : super(repaint: controller);
  static const labels = <String, String>{
    'outline': 'Контур эталона',
    'reference': 'Высота эталона',
    'tree': 'Высота дерева',
    'crown': 'Ширина кроны',
    'crown_height': 'Высота живой кроны',
    'trunk': 'Края сечения ствола',
    'trunk_axis': 'Ось участка ствола',
  };
  static const colors = <String, Color>{
    'outline': Color(0xffffa000),
    'reference': Color(0xffffa000),
    'tree': Color(0xff00bcd4),
    'crown': Color(0xffd81b60),
    'crown_height': Color(0xff8e24aa),
    'trunk': Color(0xff1565c0),
    'trunk_axis': Color(0xffef6c00),
  };
  @override
  void paint(Canvas canvas, Size size) {
    final scale = controller?.value.getMaxScaleOnAxis().clamp(1.0, 12.0) ?? 1.0;
    for (final entry in colors.entries) {
      if (hiddenLayers.contains(entry.key)) continue;
      final points = reference?[entry.key];
      if (points is! List || points.isEmpty) continue;
      final offsets = <Offset>[];
      for (final point in points) {
        if (point is! Map || point['x'] is! num || point['y'] is! num) {
          return;
        }
        final x = (point['x'] as num).toDouble(),
            y = (point['y'] as num).toDouble();
        if (!x.isFinite || !y.isFinite || x < 0 || x > 1 || y < 0 || y > 1) {
          return;
        }
        offsets.add(Offset(x * size.width, y * size.height));
      }
      final paint = Paint()
        ..color = entry.value
        ..strokeWidth = photoMarkupLineWidth / scale
        ..style = PaintingStyle.stroke;
      final path = Path()..moveTo(offsets.first.dx, offsets.first.dy);
      for (final p in offsets.skip(1)) {
        path.lineTo(p.dx, p.dy);
      }
      if (entry.key == 'outline' && offsets.length >= 3) path.close();
      canvas.drawPath(path, paint);
      if (showHandles) {
        for (var i = 0; i < offsets.length; i++) {
          paintPhotoMarkupHandle(canvas, offsets[i],
              scale: scale,
              color: entry.value,
              first: i == 0,
              sizeFactor: handleSizeFactor);
        }
      }
    }
    if (tap != null && showHandles && !hiddenLayers.contains('selection')) {
      final p = Offset(tap!.dx * size.width, tap!.dy * size.height);
      paintPhotoMarkupHandle(canvas, p,
          scale: scale, first: true, sizeFactor: handleSizeFactor);
    }
  }

  @override
  bool shouldRepaint(ReferencePreviewPainter oldDelegate) =>
      oldDelegate.reference != reference ||
      oldDelegate.tap != tap ||
      oldDelegate.controller != controller ||
      oldDelegate.showHandles != showHandles ||
      oldDelegate.hiddenLayers != hiddenLayers ||
      oldDelegate.handleSizeFactor != handleSizeFactor;
}

/// Selection is expressed in the uncropped, oriented original even after zoom.
class SelectTreePointPage extends StatefulWidget {
  final Uint8List image;
  final int width, height;
  final Offset? initial;
  final String sessionToken;
  const SelectTreePointPage(
      {super.key,
      required this.image,
      required this.width,
      required this.height,
      required this.sessionToken,
      this.initial});
  @override
  State<SelectTreePointPage> createState() => _SelectTreePointPageState();
}

class _SelectTreePointPageState extends State<SelectTreePointPage> {
  late Offset? _point = widget.initial;
  final _transform = TransformationController();
  bool _invalid = false;
  bool _showHandles = true;
  @override
  void initState() {
    super.initState();
    CorrectionsService.authChanges.addListener(_invalidate);
  }

  void _invalidate() {
    if (mounted) setState(() => _invalid = true);
  }

  @override
  void dispose() {
    CorrectionsService.authChanges.removeListener(_invalidate);
    _transform.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => Scaffold(
      appBar: AppBar(title: const Text('Выбор дерева'), actions: [
        IconButton(
            tooltip: _showHandles ? 'Скрыть точки' : 'Показать точки',
            onPressed: () => setState(() => _showHandles = !_showHandles),
            icon: Icon(_showHandles ? Icons.adjust : Icons.panorama_fish_eye)),
        IconButton(
            tooltip: 'Сброс зума',
            onPressed: () => _transform.value = Matrix4.identity(),
            icon: const Icon(Icons.zoom_out_map)),
      ]),
      body: _invalid
          ? const Center(child: Text('Аккаунт изменился. Закройте выбор.'))
          : Column(children: [
              const Padding(
                  padding: EdgeInsets.all(16),
                  child: Text(
                      'Коснитесь именно измеряемого дерева. Для тонкого дерева выберите видимую крону. Два пальца - увеличение и перемещение. Это выбор цели для модели, а не обещание распознавания.')),
              Expanded(child: LayoutBuilder(builder: (context, constraints) {
                final size = applyBoxFit(
                        BoxFit.contain,
                        Size(widget.width.toDouble(), widget.height.toDouble()),
                        constraints.biggest)
                    .destination;
                return Center(
                    child: InteractiveViewer(
                        transformationController: _transform,
                        minScale: 1,
                        maxScale: 12,
                        child: SizedBox(
                            width: size.width,
                            height: size.height,
                            child: GestureDetector(
                                key: const ValueKey('tree-selection-canvas'),
                                onTapUp: (event) => setState(() => _point =
                                    Offset(
                                        (event.localPosition.dx / size.width)
                                            .clamp(0, 1),
                                        (event.localPosition.dy / size.height)
                                            .clamp(0, 1))),
                                child: ReferencePreview(
                                    image: widget.image,
                                    width: widget.width,
                                    height: widget.height,
                                    controller: _transform,
                                    showHandles: _showHandles,
                                    showLayerControls: false,
                                    handleSizeFactor: 1,
                                    tap: _point)))));
              })),
              SafeArea(
                  child: Padding(
                      padding: const EdgeInsets.all(16),
                      child: FilledButton(
                          onPressed: _point == null
                              ? null
                              : () async {
                                  try {
                                    await CorrectionsService()
                                        .checkSession(widget.sessionToken);
                                    if (context.mounted && !_invalid) {
                                      Navigator.pop(context, _point);
                                    }
                                  } catch (_) {
                                    if (mounted) {
                                      setState(() => _invalid = true);
                                    }
                                  }
                                },
                          child: const Text('Выбрать это дерево')))),
            ]));
}
