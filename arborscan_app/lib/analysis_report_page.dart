import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:pdf/pdf.dart';
import 'package:pdf/widgets.dart' as pw;
import 'package:printing/printing.dart';

import 'app_theme.dart';

Uint8List? _tryDecodeImageB64(String? b64) {
  if (b64 == null) return null;
  var s = b64.trim();
  if (s.isEmpty) return null;
  final comma = s.indexOf(',');
  if (s.startsWith('data:') && comma != -1) {
    s = s.substring(comma + 1);
  }
  try {
    return base64.decode(s);
  } catch (_) {
    return null;
  }
}

// Глобальная функция форматирования даты, чтобы её видел и PDF, и UI
String _formatDateTime(DateTime dt) {
  String two(int v) => v < 10 ? '0$v' : '$v';
  return '${two(dt.day)}.${two(dt.month)}.${dt.year} ${two(dt.hour)}:${two(dt.minute)}';
}

class AnalysisReportPageV2 extends StatelessWidget {
  final Map<String, dynamic>? raw;
  final Uint8List? annotatedImageBytes;

  final String? species;
  final double? heightM;
  final double? crownWidthM;
  final double? trunkDiameterM;
  final double? scalePxToM;
  final double? riskIndex;
  final String? riskCategory;
  final double? lat;
  final double? lon;
  final String? address;
  final DateTime? timestamp;

  const AnalysisReportPageV2._({
    required this.raw,
    required this.annotatedImageBytes,
    required this.species,
    required this.heightM,
    required this.crownWidthM,
    required this.trunkDiameterM,
    required this.scalePxToM,
    required this.riskIndex,
    required this.riskCategory,
    required this.lat,
    required this.lon,
    required this.address,
    required this.timestamp,
  });

  factory AnalysisReportPageV2.fromRawResult({
    required Map<String, dynamic> raw,
    Uint8List? annotatedImageBytes,
  }) {
    final risk = (raw['risk'] as Map?)?.cast<String, dynamic>();
    final gps = (raw['gps'] as Map?)?.cast<String, dynamic>();

    return AnalysisReportPageV2._(
      raw: raw,
      annotatedImageBytes: annotatedImageBytes,
      species: raw['species'] as String?,
      heightM: (raw['height_m'] as num?)?.toDouble(),
      crownWidthM: (raw['crown_width_m'] as num?)?.toDouble(),
      trunkDiameterM: (raw['trunk_diameter_m'] as num?)?.toDouble(),
      scalePxToM: (raw['scale_px_to_m'] as num?)?.toDouble(),
      riskIndex: (risk?['index'] as num?)?.toDouble(),
      riskCategory: risk?['category'] as String?,
      lat: (gps?['lat'] as num?)?.toDouble(),
      lon: (gps?['lon'] as num?)?.toDouble(),
      address: raw['address'] as String?,
      timestamp: DateTime.now(),
    );
  }

  factory AnalysisReportPageV2.fromHistory({
    required String species,
    double? heightM,
    double? crownWidthM,
    double? trunkDiameterM,
    double? scalePxToM,
    double? riskIndex,
    String? riskCategory,
    double? lat,
    double? lon,
    String? address,
    DateTime? timestamp,
    String? imageBase64,
    Uint8List? annotatedImageBytes,
  }) {
    final resolvedBytes = annotatedImageBytes ?? _tryDecodeImageB64(imageBase64);
    return AnalysisReportPageV2._(
      raw: null,
      annotatedImageBytes: resolvedBytes,
      species: species,
      heightM: heightM,
      crownWidthM: crownWidthM,
      trunkDiameterM: trunkDiameterM,
      scalePxToM: scalePxToM,
      riskIndex: riskIndex,
      riskCategory: riskCategory,
      lat: lat,
      lon: lon,
      address: address,
      timestamp: timestamp,
    );
  }

  /// --- ГЕНЕРАЦИЯ PDF ---
  Future<void> _exportToPdf(BuildContext context) async {
    try {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Генерация PDF отчета...')),
      );

      final pdf = pw.Document();
      final ttf = await PdfGoogleFonts.robotoRegular();
      final ttfBold = await PdfGoogleFonts.robotoBold();

      pw.ImageProvider? pdfImage;
      final imageBytes = annotatedImageBytes ?? _tryDecodeAnnotated(raw);
      if (imageBytes != null) {
        pdfImage = pw.MemoryImage(imageBytes);
      }

      final resolvedSpecies = species ?? 'Неизвестно';
      final risk = raw?['risk'] as Map<String, dynamic>?;
      final explanation = (risk?['explanation'] as List?)?.cast<String>() ?? const [];

      pdf.addPage(
        pw.MultiPage(
          pageFormat: PdfPageFormat.a4,
          margin: const pw.EdgeInsets.all(32),
          build: (pw.Context ctx) {
            return [
              pw.Header(
                level: 0,
                child: pw.Row(
                  mainAxisAlignment: pw.MainAxisAlignment.spaceBetween,
                  children: [
                    pw.Text('ОТЧЕТ ARBORSCAN', style: pw.TextStyle(font: ttfBold, fontSize: 24, color: PdfColors.teal800)),
                    pw.Text(_formatDateTime(timestamp ?? DateTime.now()), style: pw.TextStyle(font: ttf, fontSize: 12, color: PdfColors.grey600)),
                  ],
                ),
              ),
              pw.SizedBox(height: 10),
              
              pw.Row(
                crossAxisAlignment: pw.CrossAxisAlignment.start,
                children: [
                  pw.Expanded(
                    child: pw.Column(
                      crossAxisAlignment: pw.CrossAxisAlignment.start,
                      children: [
                        pw.Text('Вид дерева:', style: pw.TextStyle(font: ttf, fontSize: 14, color: PdfColors.grey700)),
                        pw.Text(resolvedSpecies, style: pw.TextStyle(font: ttfBold, fontSize: 18)),
                        pw.SizedBox(height: 10),
                        pw.Text('Оценка риска:', style: pw.TextStyle(font: ttf, fontSize: 14, color: PdfColors.grey700)),
                        pw.Text(
                          '${riskCategory?.toUpperCase() ?? "НЕИЗВЕСТНО"} (${riskIndex?.toStringAsFixed(2) ?? "—"})', 
                          style: pw.TextStyle(
                            font: ttfBold, 
                            fontSize: 16, 
                            color: riskCategory == 'высокий' ? PdfColors.red800 : PdfColors.green800
                          )
                        ),
                      ],
                    )
                  ),
                  if (pdfImage != null)
                    pw.Container(
                      height: 200,
                      width: 150,
                      child: pw.ClipRRect(
                        horizontalRadius: 10,
                        verticalRadius: 10,
                        child: pw.Image(pdfImage, fit: pw.BoxFit.cover),
                      ),
                    ),
                ]
              ),
              
              pw.SizedBox(height: 20),
              pw.Divider(color: PdfColors.grey300),
              pw.SizedBox(height: 10),

              pw.Text('ФИЗИЧЕСКИЕ ПАРАМЕТРЫ', style: pw.TextStyle(font: ttfBold, fontSize: 14, color: PdfColors.teal800)),
              pw.SizedBox(height: 10),
              pw.Row(
                mainAxisAlignment: pw.MainAxisAlignment.spaceBetween,
                children: [
                  _pdfStatBox('Высота', '${heightM?.toStringAsFixed(2) ?? "—"} м', ttf, ttfBold),
                  _pdfStatBox('Крона', '${crownWidthM?.toStringAsFixed(2) ?? "—"} м', ttf, ttfBold),
                  _pdfStatBox('Диаметр ствола', '${trunkDiameterM?.toStringAsFixed(2) ?? "—"} м', ttf, ttfBold),
                ]
              ),

              pw.SizedBox(height: 20),

              if (address != null || lat != null) ...[
                pw.Text('ЛОКАЦИЯ', style: pw.TextStyle(font: ttfBold, fontSize: 14, color: PdfColors.teal800)),
                pw.SizedBox(height: 5),
                if (address != null) pw.Text(address!, style: pw.TextStyle(font: ttf, fontSize: 12)),
                if (lat != null && lon != null) pw.Text('GPS: $lat, $lon', style: pw.TextStyle(font: ttf, fontSize: 12, color: PdfColors.grey600)),
                pw.SizedBox(height: 20),
              ],

              pw.Text('ФАКТОРЫ РИСКА (SIA METHOD)', style: pw.TextStyle(font: ttfBold, fontSize: 14, color: PdfColors.teal800)),
              pw.SizedBox(height: 10),
              
              ...explanation.map((line) => pw.Padding(
                padding: const pw.EdgeInsets.only(bottom: 6),
                child: pw.Row(
                  crossAxisAlignment: pw.CrossAxisAlignment.start,
                  children: [
                    pw.Text('• ', style: pw.TextStyle(font: ttfBold)),
                    pw.Expanded(child: pw.Text(line, style: pw.TextStyle(font: ttf, fontSize: 12))),
                  ]
                )
              )).toList(),
              
              pw.Spacer(),
              pw.Divider(color: PdfColors.grey300),
              pw.SizedBox(height: 10),
              pw.Text('Сгенерировано в профессиональном приложении ArborScan AI', style: pw.TextStyle(font: ttf, fontSize: 10, color: PdfColors.grey500)),
            ];
          },
        ),
      );

      await Printing.sharePdf(
        bytes: await pdf.save(), 
        filename: 'ArborScan_Report_${DateTime.now().millisecondsSinceEpoch}.pdf'
      );
    } catch (e) {
      if (!context.mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Ошибка создания PDF: $e')));
    }
  }

  pw.Widget _pdfStatBox(String label, String val, pw.Font ttf, pw.Font ttfBold) {
    return pw.Container(
      padding: const pw.EdgeInsets.all(10),
      decoration: const pw.BoxDecoration(
        color: PdfColors.grey100,
        borderRadius: pw.BorderRadius.all(pw.Radius.circular(8)),
      ),
      child: pw.Column(
        crossAxisAlignment: pw.CrossAxisAlignment.start,
        children: [
          pw.Text(label, style: pw.TextStyle(font: ttf, fontSize: 10, color: PdfColors.grey700)),
          pw.SizedBox(height: 4),
          pw.Text(val, style: pw.TextStyle(font: ttfBold, fontSize: 14)),
        ]
      )
    );
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    final resolvedSpecies = (species ?? 'Неизвестно').toString();
    final resolvedTimestamp = timestamp ?? DateTime.now();

    final risk = raw?['risk'] as Map<String, dynamic>?;
    final explanation = (risk?['explanation'] as List?)?.cast<String>() ?? const [];
    final beta = (raw?['beta'] as Map?)?.cast<String, dynamic>();
    final analyticWindModel = (raw?['analytic_wind_model'] as Map?)?.cast<String, dynamic>();

    final sourceMap = (raw?['measurement_sources'] as Map?)?.cast<String, dynamic>();
    final dimensionsSource = raw?['dimensions_source'] as String?;
    final hasArMeasurements = _sourceLabel(sourceMap?['height_m']) == 'AR' ||
        _sourceLabel(sourceMap?['crown_width_m']) == 'AR' ||
        _sourceLabel(sourceMap?['trunk_diameter_m']) == 'AR';

    final hero = _RiskHeroData.from(
      riskIndex: riskIndex,
      riskCategory: riskCategory,
      theme: theme,
    );

    return Scaffold(
      backgroundColor: AppTheme.background,
      appBar: AppBar(
        title: const Text('ОТЧЁТ ПО АНАЛИЗУ', style: TextStyle(letterSpacing: 1.5)),
      ),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.fromLTRB(16, 16, 16, 100),
          children: [
            _HeroCard(
              hero: hero,
              species: resolvedSpecies,
              timestamp: resolvedTimestamp,
              imageBytes: annotatedImageBytes ?? _tryDecodeAnnotated(raw),
            ),
            const SizedBox(height: 24),

            Ui.sectionTitle(context, 'КЛЮЧЕВЫЕ ПАРАМЕТРЫ'),
            _SourceSummaryCard(
              dimensionsSource: dimensionsSource,
              hasArMeasurements: hasArMeasurements,
            ),
            const SizedBox(height: 12),
            _MetricsGrid(
              heightM: heightM,
              crownWidthM: crownWidthM,
              trunkDiameterM: trunkDiameterM,
              scalePxToM: scalePxToM,
              heightSource: _sourceLabel(sourceMap?['height_m']),
              crownSource: _sourceLabel(sourceMap?['crown_width_m']),
              trunkSource: _sourceLabel(sourceMap?['trunk_diameter_m']),
            ),
            const SizedBox(height: 12),
            _BetaCard(beta: beta),
            const SizedBox(height: 12),
            _AnalyticWindModelCard(model: analyticWindModel),
            const SizedBox(height: 24),

            Ui.sectionTitle(context, 'ЛОКАЦИЯ'),
            _LocationCard(address: address, lat: lat, lon: lon),
            const SizedBox(height: 24),

            Ui.sectionTitle(context, 'ФАКТОРЫ РИСКА (SIA)'),
            _ExplanationCard(lines: explanation),
            const SizedBox(height: 12),
            _FootnoteCard(raw: raw),
            const SizedBox(height: 32),
          ],
        ),
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: () => _exportToPdf(context),
        backgroundColor: AppTheme.primary,
        icon: const Icon(Icons.picture_as_pdf_rounded, color: Colors.black),
        label: const Text(
          'ЭКСПОРТ В PDF', 
          style: TextStyle(color: Colors.black, fontWeight: FontWeight.w900, letterSpacing: 1.0)
        ),
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
    );
  }

  static String _sourceLabel(dynamic value) {
    final s = value?.toString().toLowerCase().trim();
    if (s == 'ar') return 'AR';
    if (s == 'image' || s == 'photo') return 'Фото + ИИ';
    return '—';
  }

  static Uint8List? _tryDecodeAnnotated(Map<String, dynamic>? raw) {
    if (raw == null) return null;
    final b64 = raw['annotated_image_base64'] as String?;
    if (b64 == null || b64.isEmpty) return null;
    try {
      return base64Decode(b64);
    } catch (_) {
      return null;
    }
  }
}

// ----------------------------------------------------
// UI COMPONENTS (Адаптированы под Glassmorphism)
// ----------------------------------------------------

class _HeroCard extends StatelessWidget {
  final _RiskHeroData hero;
  final String species;
  final DateTime timestamp;
  final Uint8List? imageBytes;

  const _HeroCard({
    required this.hero,
    required this.species,
    required this.timestamp,
    required this.imageBytes,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return GlassPanel(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'ИТОГОВАЯ ОЦЕНКА',
                      style: theme.textTheme.titleMedium?.copyWith(
                        fontWeight: FontWeight.w900,
                        color: AppTheme.text,
                        letterSpacing: 1.0,
                      ),
                    ),
                    const SizedBox(height: 12),
                    Text(
                      species,
                      style: const TextStyle(fontSize: 20, color: AppTheme.primary2, fontWeight: FontWeight.w900, shadows: [Shadow(color: AppTheme.primary2, blurRadius: 8)]),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      _formatDateTime(timestamp),
                      style: theme.textTheme.bodySmall?.copyWith(color: AppTheme.muted),
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 12),
              _RiskBadge(hero: hero),
            ],
          ),
          const SizedBox(height: 16),

          if (imageBytes != null)
            ClipRRect(
              borderRadius: BorderRadius.circular(20),
              child: AspectRatio(
                aspectRatio: 3 / 4,
                child: Image.memory(imageBytes!, fit: BoxFit.cover),
              ),
            )
          else
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(14),
              decoration: BoxDecoration(
                color: AppTheme.surface3.withOpacity(0.5),
                borderRadius: BorderRadius.circular(20),
              ),
              child: Row(
                children: [
                  const Icon(Icons.image_not_supported_outlined, color: AppTheme.muted),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Text(
                      'Аннотированное изображение недоступно.',
                      style: theme.textTheme.bodyMedium?.copyWith(color: AppTheme.muted),
                    ),
                  ),
                ],
              ),
            ),
        ],
      ),
    );
  }
}

class _RiskBadge extends StatelessWidget {
  final _RiskHeroData hero;
  const _RiskBadge({required this.hero});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: hero.background.withOpacity(0.15),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: hero.background, width: 2),
        boxShadow: [BoxShadow(color: hero.background.withOpacity(0.2), blurRadius: 12)],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.end,
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            hero.label.toUpperCase(),
            style: TextStyle(
              fontWeight: FontWeight.w900,
              color: hero.foreground,
              fontSize: 10,
              letterSpacing: 1.0,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            hero.valueText,
            style: TextStyle(
              fontWeight: FontWeight.w900,
              color: hero.foreground,
              fontSize: 24,
              shadows: [Shadow(color: hero.foreground, blurRadius: 10)]
            ),
          ),
        ],
      ),
    );
  }
}

class _BetaCard extends StatelessWidget {
  final Map<String, dynamic>? beta;

  const _BetaCard({required this.beta});

  String _fmt(dynamic value, {String suffix = ''}) {
    if (value == null) return '—';
    if (value is num) return '${value.toStringAsFixed(2)}$suffix';
    final parsed = double.tryParse(value.toString());
    if (parsed == null) return value.toString();
    return '${parsed.toStringAsFixed(2)}$suffix';
  }

  @override
  Widget build(BuildContext context) {
    final betaValue = beta?['beta_kg_s'];
    final betaMax = beta?['beta_max_scenario'];
    final method = beta?['method']?.toString() ?? '—';
    final source = beta?['source']?.toString() ?? '—';
    final force = beta?['wind_force_n'];

    String methodText;
    switch (method) {
      case 'manual': methodText = 'вручную'; break;
      case 'estimated_from_geometry': methodText = 'по геометрии AR'; break;
      case 'species_default': methodText = 'по породе'; break;
      case 'empirical_borisevich_2021': methodText = 'Borisevich (2021)'; break;
      default: methodText = method;
    }

    double? maxForce;
    if (betaMax != null && force != null && betaValue != null && betaValue > 0) {
      maxForce = (force / betaValue) * betaMax;
    }

    return GlassPanel(
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: AppTheme.primary.withOpacity(0.15),
              shape: BoxShape.circle,
            ),
            child: const Icon(Icons.air_outlined, color: AppTheme.primary, size: 24),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'АЭРОДИНАМИКА (β)',
                  style: TextStyle(color: AppTheme.text, fontWeight: FontWeight.w900, fontSize: 12, letterSpacing: 1.0),
                ),
                const SizedBox(height: 12),
                
                _buildRow('Ожидаемый β:', _fmt(betaValue, suffix: ' кг/с'), AppTheme.success),
                
                if (betaMax != null && (betaValue == null || betaMax > betaValue)) ...[
                  const SizedBox(height: 6),
                  _buildRow('Худший сценарий\n(жесткая крона):', _fmt(betaMax, suffix: ' кг/с'), AppTheme.warning),
                ],

                const SizedBox(height: 8),
                Text('$source · $methodText', style: const TextStyle(color: AppTheme.muted, fontSize: 10, fontWeight: FontWeight.w600, height: 1.3)),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildRow(String label, String value, Color valueColor) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Expanded(child: Text(label, style: const TextStyle(color: AppTheme.muted, fontSize: 12, fontWeight: FontWeight.w600, height: 1.2))),
        const SizedBox(width: 8),
        Text(value, style: TextStyle(color: valueColor, fontSize: 14, fontWeight: FontWeight.w900, shadows: [Shadow(color: valueColor, blurRadius: 4)])),
      ],
    );
  }
}

class _AnalyticWindModelCard extends StatelessWidget {
  final Map<String, dynamic>? model;
  const _AnalyticWindModelCard({required this.model});

  String _fmt(dynamic value, {String suffix = ''}) {
    if (value == null) return '—';
    if (value is num) return '${value.toStringAsFixed(2)}$suffix';
    final parsed = double.tryParse(value.toString());
    if (parsed == null) return value.toString();
    return '${parsed.toStringAsFixed(2)}$suffix';
  }

  @override
  Widget build(BuildContext context) {
    final available = model?['available'] == true;
    final outputs = (model?['outputs'] as Map?)?.cast<String, dynamic>() ?? {};

    return GlassPanel(
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: available ? AppTheme.primary2.withOpacity(0.15) : AppTheme.surface3,
              shape: BoxShape.circle,
            ),
            child: Icon(available ? Icons.science_outlined : Icons.info_outline, color: available ? AppTheme.primary2 : AppTheme.muted),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: available
                ? Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text('НАГРУЗКА ИЗЛОМА (SIA)', style: TextStyle(color: AppTheme.text, fontWeight: FontWeight.w900, fontSize: 12, letterSpacing: 1.0)),
                      const SizedBox(height: 12),
                      _miniRow('Сила ветра (F)', _fmt(outputs['total_force_n'], suffix: ' Н')),
                      _miniRow('Момент излома у основания', _fmt(outputs['base_moment_nm'], suffix: ' Н·м')),
                    ],
                  )
                : const Text('Аналитическая модель недоступна.', style: TextStyle(color: AppTheme.muted, fontWeight: FontWeight.w700)),
          ),
        ],
      ),
    );
  }

  Widget _miniRow(String title, String value) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 6),
      child: Row(
        children: [
          Expanded(child: Text(title, style: const TextStyle(color: AppTheme.muted, fontSize: 12, fontWeight: FontWeight.w600))),
          Text(value, style: const TextStyle(color: AppTheme.text, fontSize: 14, fontWeight: FontWeight.w900)),
        ],
      ),
    );
  }
}

class _SourceSummaryCard extends StatelessWidget {
  final String? dimensionsSource;
  final bool hasArMeasurements;

  const _SourceSummaryCard({required this.dimensionsSource, required this.hasArMeasurements});

  @override
  Widget build(BuildContext context) {
    return GlassPanel(
      border: Border.all(color: hasArMeasurements ? AppTheme.primary.withOpacity(0.5) : AppTheme.border),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(hasArMeasurements ? Icons.view_in_ar_outlined : Icons.image_search_outlined, color: hasArMeasurements ? AppTheme.primary : AppTheme.muted),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text('ИСТОЧНИК ДАННЫХ', style: TextStyle(color: AppTheme.text, fontWeight: FontWeight.w900, fontSize: 12, letterSpacing: 1.0)),
                const SizedBox(height: 4),
                Text(dimensionsSource ?? 'Фото + ИИ', style: const TextStyle(color: AppTheme.primary2, fontWeight: FontWeight.w900, fontSize: 14)),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _MetricsGrid extends StatelessWidget {
  final double? heightM;
  final double? crownWidthM;
  final double? trunkDiameterM;
  final double? scalePxToM;
  final String heightSource;
  final String crownSource;
  final String trunkSource;

  const _MetricsGrid({required this.heightM, required this.crownWidthM, required this.trunkDiameterM, required this.scalePxToM, required this.heightSource, required this.crownSource, required this.trunkSource});

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Row(
          children: [
            Expanded(child: _MetricCard(title: 'ВЫСОТА', icon: Icons.height, value: _fmt(heightM, suffix: 'м'))),
            const SizedBox(width: 10),
            Expanded(child: _MetricCard(title: 'КРОНА', icon: Icons.filter_hdr, value: _fmt(crownWidthM, suffix: 'м'))),
          ],
        ),
        const SizedBox(height: 10),
        Row(
          children: [
            Expanded(child: _MetricCard(title: 'СТВОЛ', icon: Icons.circle_outlined, value: _fmt(trunkDiameterM, suffix: 'м'))),
            const SizedBox(width: 10),
            Expanded(child: _MetricCard(title: 'МАСШТАБ', icon: Icons.straighten, value: scalePxToM == null ? 'Нет' : '1 px ≈ ${scalePxToM!.toStringAsFixed(4)} м', isSecondary: true)),
          ],
        ),
      ],
    );
  }

  static String _fmt(double? v, {required String suffix}) {
    if (v == null) return '—';
    return '${v.toStringAsFixed(2)} $suffix';
  }
}

class _MetricCard extends StatelessWidget {
  final String title;
  final IconData icon;
  final String value;
  final bool isSecondary;

  const _MetricCard({required this.title, required this.icon, required this.value, this.isSecondary = false});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: isSecondary ? AppTheme.surface3.withOpacity(0.3) : AppTheme.surface3,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: AppTheme.primary.withOpacity(0.2)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(icon, color: isSecondary ? AppTheme.muted : AppTheme.primary, size: 18),
              const SizedBox(width: 8),
              Text(title, style: TextStyle(color: isSecondary ? AppTheme.muted : AppTheme.text, fontSize: 10, fontWeight: FontWeight.w900, letterSpacing: 1.0)),
            ],
          ),
          const SizedBox(height: 12),
          Text(value, style: TextStyle(color: AppTheme.text, fontSize: 16, fontWeight: FontWeight.w900, shadows: isSecondary ? [] : [const Shadow(color: AppTheme.primary, blurRadius: 8)])),
        ],
      ),
    );
  }
}

class _LocationCard extends StatelessWidget {
  final String? address;
  final double? lat;
  final double? lon;
  const _LocationCard({required this.address, required this.lat, required this.lon});

  @override
  Widget build(BuildContext context) {
    return GlassPanel(
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(color: AppTheme.primary2.withOpacity(0.15), shape: BoxShape.circle),
            child: const Icon(Icons.location_on_outlined, color: AppTheme.primary2),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(address != null && address!.isNotEmpty ? address! : 'Адрес не найден', style: const TextStyle(fontWeight: FontWeight.w900, fontSize: 14)),
                const SizedBox(height: 6),
                Text(lat != null && lon != null ? 'GPS: ${lat!.toStringAsFixed(6)}, ${lon!.toStringAsFixed(6)}' : 'Координаты отсутствуют.', style: const TextStyle(color: AppTheme.muted, fontSize: 12)),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _ExplanationCard extends StatelessWidget {
  final List<String> lines;
  const _ExplanationCard({required this.lines});

  @override
  Widget build(BuildContext context) {
    if (lines.isEmpty) return const SizedBox.shrink();
    return GlassPanel(
      color: AppTheme.danger.withOpacity(0.05),
      border: Border.all(color: AppTheme.danger.withOpacity(0.5)),
      child: Column(
        children: lines.map((l) => Padding(
          padding: const EdgeInsets.symmetric(vertical: 6),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Icon(Icons.warning_amber_rounded, color: AppTheme.danger, size: 16),
              const SizedBox(width: 8),
              Expanded(child: Text(l, style: const TextStyle(color: AppTheme.text, height: 1.4))),
            ],
          ),
        )).toList(),
      ),
    );
  }
}

class _FootnoteCard extends StatelessWidget {
  final Map<String, dynamic>? raw;
  const _FootnoteCard({required this.raw});

  @override
  Widget build(BuildContext context) {
    final analysisId = raw?['analysis_id'] as String?;

    return GlassPanel(
      color: AppTheme.surface3.withOpacity(0.3),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(Icons.shield_outlined, color: AppTheme.muted),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'ПРИМЕЧАНИЕ',
                  style: TextStyle(fontWeight: FontWeight.w900, color: AppTheme.muted, letterSpacing: 1.0, fontSize: 11),
                ),
                const SizedBox(height: 4),
                const Text(
                  'Оценка является аналитической подсказкой и не заменяет осмотр специалистом.',
                  style: TextStyle(color: AppTheme.muted, fontSize: 11),
                ),
                if (analysisId != null && analysisId.isNotEmpty) ...[
                  const SizedBox(height: 8),
                  Text(
                    'ID: $analysisId',
                    style: const TextStyle(color: AppTheme.muted, fontSize: 9),
                  ),
                ]
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _RiskHeroData {
  final String label;
  final String valueText;
  final Color background;
  final Color foreground;

  const _RiskHeroData({
    required this.label,
    required this.valueText,
    required this.background,
    required this.foreground,
  });

  static _RiskHeroData from({
    required double? riskIndex,
    required String? riskCategory,
    required ThemeData theme,
  }) {
    final cat = (riskCategory ?? '').trim().toLowerCase();
    final idx = riskIndex;

    if (cat == 'низкий') {
      return _RiskHeroData(
        label: 'БЕЗОПАСНО',
        valueText: idx != null ? idx.toStringAsFixed(2) : '—',
        background: AppTheme.success,
        foreground: Colors.black,
      );
    }
    if (cat == 'средний') {
      return _RiskHeroData(
        label: 'ВНИМАНИЕ',
        valueText: idx != null ? idx.toStringAsFixed(2) : '—',
        background: AppTheme.warning,
        foreground: Colors.black,
      );
    }
    if (cat == 'высокий' || cat.isNotEmpty) {
      return _RiskHeroData(
        label: 'КРИТИЧНО',
        valueText: idx != null ? idx.toStringAsFixed(2) : '—',
        background: AppTheme.danger,
        foreground: Colors.white,
      );
    }

    return _RiskHeroData(
      label: 'РИСК',
      valueText: idx != null ? idx.toStringAsFixed(2) : '—',
      background: AppTheme.surface3,
      foreground: AppTheme.muted,
    );
  }
}