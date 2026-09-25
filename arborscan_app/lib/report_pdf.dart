import 'package:flutter/services.dart';
import 'package:image/image.dart' as im;
import 'package:pdf/pdf.dart';
import 'package:pdf/widgets.dart' as pw;
import 'report_export_data.dart';

Future<Uint8List> buildReportPdfInWorker(Map<String, dynamic> args) =>
    buildReportPdf(args['data'] as ReportExportData,
        partial: args['partial'] as bool, fontData: args['font'] as ByteData);

class ReportImages {
  final Uint8List? original, annotated, contour;
  final List<String> legend;
  const ReportImages(this.original, this.annotated, this.legend, this.contour);
}

const exportLineLabels = {
  'reference': 'Эталон',
  'tree': 'Высота дерева',
  'crown': 'Ширина кроны',
  'crown_height': 'Высота живой кроны',
  'trunk': 'Края сечения ствола',
  'trunk_axis': 'Ось участка ствола',
  'outline': 'Контур эталона',
};
const exportLineColors = [
  0x0059b3,
  0xc42b26,
  0x187d3d,
  0x973ca5,
  0xaa5500,
  0x006b70,
  0x454545
];
const exportColorNames = [
  'синий',
  'красный',
  'зелёный',
  'фиолетовый',
  'коричневый',
  'бирюзовый',
  'серый'
];

ReportImages prepareReportImages(ReportExportData data,
    {bool partial = false}) {
  im.Image? decode(Uint8List? bytes) {
    if (bytes == null) return null;
    if (bytes.length > 30 * 1024 * 1024) {
      throw const FormatException('Изображение слишком велико для PDF.');
    }
    final d = im.findDecoderForData(bytes);
    final info = d?.startDecode(bytes);
    if (info == null || info.width * info.height > 25000000) {
      throw const FormatException(
          'Фото повреждено или больше 25 МП. Повторите загрузку.');
    }
    return im.bakeOrientation(d!.decodeFrame(0)!);
  }

  final original = decode(data.photo);
  if (original == null && !partial) {
    throw const FormatException(
        'Исходное фото недоступно. Повторите загрузку или явно выберите неполный PDF.');
  }
  im.Image fit(im.Image image) => image.width > 1600 || image.height > 1600
      ? im.copyResize(image,
          width: image.width >= image.height ? 1600 : null,
          height: image.height > image.width ? 1600 : null)
      : image.clone();
  final ref = data.reference;
  final legend = <String>[];
  im.Image? overlay;
  if (original != null && ref.isNotEmpty) {
    if (ref['coordinates'] != 'normalized_oriented_image' ||
        ref['width'] != original.width ||
        ref['height'] != original.height) {
      throw const FormatException(
          'Разметка не соответствует размерам или ориентации исходного фото. PDF не создан.');
    }
    overlay = fit(original);
    var index = 0;
    for (final entry in exportLineLabels.entries) {
      final color = exportLineColors[index++];
      final points = ref[entry.key] as List? ?? [];
      if (points.isEmpty) continue;
      for (final p in points) {
        if (p is! Map ||
            p['x'] is! num ||
            p['y'] is! num ||
            !(p['x'] as num).isFinite ||
            !(p['y'] as num).isFinite ||
            p['x'] < 0 ||
            p['x'] > 1 ||
            p['y'] < 0 ||
            p['y'] > 1) {
          throw const FormatException('Точки разметки повреждены.');
        }
      }
      final c =
          im.ColorRgb8((color >> 16) & 255, (color >> 8) & 255, color & 255);
      int x(dynamic p) => ((p['x'] as num) * (overlay!.width - 1)).round();
      int y(dynamic p) => ((p['y'] as num) * (overlay!.height - 1)).round();
      for (var i = 1;
          i < points.length + (entry.key == 'outline' ? 1 : 0);
          i++) {
        final a = points[i - 1], b = points[i % points.length];
        im.drawLine(overlay,
            x1: x(a), y1: y(a), x2: x(b), y2: y(b), color: c, thickness: 4);
      }
      for (final p in points) {
        im.fillCircle(overlay, x: x(p), y: y(p), radius: 6, color: c);
      }
      legend.add(
          '${exportColorNames[index - 1]} — ${entry.value}; точки (нормированные, округлены): ${points.map((p) => '(${exportNumber(p['x'])}; ${exportNumber(p['y'])})').join(' → ')}');
    }
  } else {
    final a = decode(data.annotation);
    if (a != null) {
      overlay = fit(a);
      legend.add(
          'Сохранённое изображение с разметкой выбранной версии; исходные точки могут отсутствовать.');
    }
  }
  im.Image? contour;
  if (data.mask != null && original != null) {
    final mask = decode(data.mask)!;
    if (mask.width != original.width || mask.height != original.height) {
      throw const FormatException('Размер маски не соответствует оригиналу.');
    }
    contour = fit(original);
    final scaled = im.copyResize(mask,
        width: contour.width,
        height: contour.height,
        interpolation: im.Interpolation.nearest);
    for (final p in contour) {
      final m = scaled.getPixel(p.x, p.y);
      if (m.r > 127 || m.g > 127 || m.b > 127) {
        p.setRgb(p.r * 0.55, p.g * 0.55 + 110, p.b * 0.55);
      }
    }
  }
  return ReportImages(
      original == null
          ? null
          : Uint8List.fromList(im.encodeJpg(fit(original), quality: 90)),
      overlay == null ? null : Uint8List.fromList(im.encodePng(overlay)),
      legend,
      contour == null ? null : Uint8List.fromList(im.encodePng(contour)));
}

Future<Uint8List> buildReportPdf(ReportExportData data,
    {bool partial = false, DateTime? generatedAt, ByteData? fontData}) async {
  final font = pw.Font.ttf(
      fontData ?? await rootBundle.load('assets/fonts/DejaVuSans.ttf'));
  final images = prepareReportImages(data, partial: partial);
  final pdf = pw.Document(
      title: 'ArborScan — отчёт об обследовании дерева', author: 'ArborScan');
  final theme = pw.ThemeData.withFont(
      base: font, bold: font, italic: font, boldItalic: font);
  const green = PdfColor.fromInt(0xff185746);
  pw.Widget text(String value) => pw.Padding(
      padding: const pw.EdgeInsets.only(bottom: 6),
      child: pw.Text(exportText(value),
          style: const pw.TextStyle(fontSize: 11, lineSpacing: 3)));
  pw.Widget heading(String value) => pw.Header(
      level: 1,
      child: pw.Text(value,
          style: const pw.TextStyle(fontSize: 16, color: green)));
  pw.Widget image(Uint8List bytes) => pw.Center(
      child: pw.SizedBox(
          height: 340,
          width: 490,
          child: pw.Image(pw.MemoryImage(bytes), fit: pw.BoxFit.contain)));
  final s = data.snapshot;
  pdf.addPage(pw.MultiPage(
      pageFormat: PdfPageFormat.a4,
      theme: theme,
      margin: const pw.EdgeInsets.all(36),
      maxPages: 100,
      footer: (c) => pw.Row(
              mainAxisAlignment: pw.MainAxisAlignment.spaceBetween,
              children: [
                pw.Text('ArborScan • сохранённый результат',
                    style: const pw.TextStyle(fontSize: 8)),
                pw.Text('${c.pageNumber} / ${c.pagesCount}',
                    style: const pw.TextStyle(fontSize: 8)),
              ]),
      build: (_) => [
            pw.Text('ArborScan',
                style: const pw.TextStyle(fontSize: 24, color: green)),
            heading('Отчёт об обследовании дерева'),
            if (partial)
              text(
                  'НЕПОЛНЫЙ ОТЧЁТ: исходное фото недоступно; экспортированы только доступные данные.'),
            text(data.local
                ? 'Локальный снимок: сохранён только на этом устройстве. Экспорт не отправляет его на сервер.'
                : 'Снимок выбранной версии отчёта из аккаунта.'),
            text('Запись: ${data.id}\nВерсия: ${data.version}'),
            text(
                'Дата обследования / анализа: ${exportText(s['kind'] == 'v4' ? data.report['captured_at'] : s['captured_at'])}\nСохранение версии: ${exportText(data.record['created_at'])}\nФормирование PDF (UTC): ${(generatedAt ?? DateTime.now()).toUtc().toIso8601String()}'),
            if (images.original != null) ...[
              pw.Column(
                  children: [heading('Исходное фото'), image(images.original!)])
            ],
            ...data.speciesLines.map(text),
            pw.NewPage(),
            heading('Измерения выбранной версии'),
            text(data.method),
            for (final m in data.metrics) ...[
              pw.Table(
                  border: pw.TableBorder.all(color: PdfColors.grey300),
                  columnWidths: {
                    0: const pw.FlexColumnWidth(3),
                    1: const pw.FlexColumnWidth(1)
                  },
                  children: [
                    pw.TableRow(children: [
                      pw.Padding(
                          padding: const pw.EdgeInsets.all(7),
                          child: pw.Text(m.label)),
                      pw.Padding(
                          padding: const pw.EdgeInsets.all(7),
                          child: pw.Text(m.display)),
                    ])
                  ]),
              text('Источник / метод: ${m.source}'),
              text(m.limitation),
            ],
            if (images.annotated != null) ...[
              pw.NewPage(),
              heading('Разметка выбранной версии'),
              image(images.annotated!),
              ...images.legend.map(text),
            ] else
              text(
                  'Разметка отсутствует в доступных данных этой версии; автоматически не восстановлена.'),
            if (images.contour != null) ...[
              pw.NewPage(),
              heading('Связанная ревизия контура'),
              image(images.contour!),
              text(
                  'Зелёное наложение: PNG-маска ревизии ${exportText(s['correction_id'])}. Текущее решение модерации не подставляется в исторический снимок.')
            ],
            heading('Происхождение и ограничения'),
            ...data.provenance.map(text),
            heading('Сохранённые условия'),
            ...data.environment.map(text),
            heading('Границы применения'),
            text(
                'Полевая точность не подтверждена. Двумерные проекции не определяют полную пространственную геометрию.'),
            text(
                'β (кг/с): расчёт отсутствует. Необходимы динамический эксперимент и подтверждённая модель.'),
            text(
                'Этот документ не является экспертным заключением или подтверждением безопасности дерева. Экспорт не выполняет новый анализ и не обновляет внешние данные.'),
            text(
                'Сохранённый или переданный PDF является самостоятельным файлом. Выход из аккаунта не отзывает его копии.'),
          ]));
  return pdf.save();
}
