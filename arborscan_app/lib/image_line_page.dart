import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'corrections_service.dart';

/// The gesture detector lives INSIDE the transformed, image-sized child.
/// Flutter maps touches through the inverse transform before normalization.
class ImageLinePage extends StatefulWidget {
  final Uint8List image;
  final int width, height;
  final String title;
  final List<Offset> initial;
  final String? sessionToken;
  const ImageLinePage(
      {super.key,
      required this.image,
      required this.width,
      required this.height,
      required this.title,
      this.initial = const [],
      this.sessionToken});
  @override
  State<ImageLinePage> createState() => _ImageLinePageState();
}

class _ImageLinePageState extends State<ImageLinePage> {
  late final points = List<Offset>.from(widget.initial);
  int selected = 0;
  final transform = TransformationController();
  bool invalid = false;
  @override
  void initState() {
    super.initState();
    CorrectionsService.authChanges.addListener(_invalidate);
  }

  void _invalidate() {
    if (widget.sessionToken != null && mounted) setState(() => invalid = true);
  }

  @override
  void dispose() {
    CorrectionsService.authChanges.removeListener(_invalidate);
    transform.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => Scaffold(
        appBar: AppBar(title: Text(widget.title)),
        body: invalid
            ? const Center(child: Text('Аккаунт изменился. Закройте разметку.'))
            : Column(children: [
                const Padding(
                    padding: EdgeInsets.all(12),
                    child: Text(
                        'Выберите конец и коснитесь нужного места. Двумя пальцами увеличивайте и перемещайте фото. Можно переставлять точки сколько угодно.')),
                SegmentedButton<int>(
                    segments: const [
                      ButtonSegment(value: 0, label: Text('Основание / слева')),
                      ButtonSegment(value: 1, label: Text('Верх / справа'))
                    ],
                    selected: {
                      selected
                    },
                    onSelectionChanged: (v) =>
                        setState(() => selected = v.first)),
                Expanded(child: LayoutBuilder(builder: (context, c) {
                  final size = applyBoxFit(
                          BoxFit.contain,
                          Size(widget.width.toDouble(),
                              widget.height.toDouble()),
                          c.biggest)
                      .destination;
                  return Center(
                      child: InteractiveViewer(
                          transformationController: transform,
                          minScale: 1,
                          maxScale: 12,
                          child: SizedBox(
                              width: size.width,
                              height: size.height,
                              child: GestureDetector(
                                  key: const ValueKey('image-line-canvas'),
                                  onTapUp: (e) => setState(() {
                                        final p = Offset(
                                            (e.localPosition.dx / size.width)
                                                .clamp(0, 1),
                                            (e.localPosition.dy / size.height)
                                                .clamp(0, 1));
                                        if (points.isEmpty) {
                                          points.add(p);
                                          selected = 1;
                                        } else if (points.length == 1 &&
                                            selected == 1) {
                                          points.add(p);
                                        } else {
                                          points[selected.clamp(
                                              0, points.length - 1)] = p;
                                        }
                                      }),
                                  child: Stack(fit: StackFit.expand, children: [
                                    Image.memory(widget.image,
                                        fit: BoxFit.fill),
                                    CustomPaint(
                                        painter: _LinePainter(List.of(points)))
                                  ])))));
                })),
                SafeArea(
                    child: Padding(
                        padding: const EdgeInsets.all(12),
                        child: FilledButton(
                            onPressed:
                                points.length != 2 || points[0] == points[1]
                                    ? null
                                    : () => Navigator.pop(
                                        context, List<Offset>.from(points)),
                            child: const Text('Применить отрезок')))),
              ]),
      );
}

class _LinePainter extends CustomPainter {
  final List<Offset> points;
  _LinePainter(this.points);
  @override
  void paint(Canvas c, Size s) {
    final p =
        points.map((v) => Offset(v.dx * s.width, v.dy * s.height)).toList();
    final paint = Paint()
      ..color = Colors.orange
      ..strokeWidth = 3;
    if (p.length == 2) c.drawLine(p[0], p[1], paint);
    for (final v in p) {
      c.drawCircle(v, 5, paint);
    }
  }

  @override
  bool shouldRepaint(covariant _LinePainter old) => true;
}
