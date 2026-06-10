import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/material.dart';

import 'app_theme.dart';

Uint8List? _tryDecodeImageB64(String? b64) {
  if (b64 == null) return null;
  var s = b64.trim();
  if (s.isEmpty) return null;
  // Support data URIs like: data:image/png;base64,....
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

/// Экран "Результат анализа v2".
/// Поддерживает 2 сценария:
/// 1) fromRawResult: сразу после /analyze-tree
/// 2) fromHistory: из локальной истории
class AnalysisReportPageV2 extends StatelessWidget {
  final Map<String, dynamic>? raw;
  final Uint8List? annotatedImageBytes;

  // History fallback
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
      appBar: AppBar(
        title: const Text('Отчёт по анализу'),
      ),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.fromLTRB(16, 16, 16, 24),
          children: [
            _HeroCard(
              hero: hero,
              species: resolvedSpecies,
              timestamp: resolvedTimestamp,
              imageBytes: annotatedImageBytes ?? _tryDecodeAnnotated(raw),
            ),
            const SizedBox(height: 12),

            _SectionTitle(
              title: 'Ключевые параметры',
              subtitle: hasArMeasurements
                  ? 'Размеры получены через AR и использованы при расчёте риска.'
                  : 'Размеры рассчитаны по фото/масштабу, если масштаб доступен.',
            ),
            const SizedBox(height: 8),
            _SourceSummaryCard(
              dimensionsSource: dimensionsSource,
              hasArMeasurements: hasArMeasurements,
            ),
            const SizedBox(height: 10),
            _MetricsGrid(
              heightM: heightM,
              crownWidthM: crownWidthM,
              trunkDiameterM: trunkDiameterM,
              scalePxToM: scalePxToM,
              heightSource: _sourceLabel(sourceMap?['height_m']),
              crownSource: _sourceLabel(sourceMap?['crown_width_m']),
              trunkSource: _sourceLabel(sourceMap?['trunk_diameter_m']),
            ),
            const SizedBox(height: 10),
            _BetaCard(beta: beta),
            const SizedBox(height: 10),
            _AnalyticWindModelCard(model: analyticWindModel),
            const SizedBox(height: 16),

            _SectionTitle(
              title: 'Локация',
              subtitle: raw?['gps'] != null ? 'Источник: GPS телефона или метаданные фото.' : 'Координаты недоступны для этого анализа.',
            ),
            const SizedBox(height: 8),
            _LocationCard(address: address, lat: lat, lon: lon),
            const SizedBox(height: 16),

            _SectionTitle(
              title: 'Факторы риска',
              subtitle: explanation.isNotEmpty
                  ? 'Пояснение модели к итоговой оценке.'
                  : 'Пояснение модели недоступно для этого анализа.',
            ),
            const SizedBox(height: 8),
            _ExplanationCard(lines: explanation),

            const SizedBox(height: 24),
            _FootnoteCard(raw: raw),
          ],
        ),
      ),
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

class _SectionTitle extends StatelessWidget {
  final String title;
  final String subtitle;

  const _SectionTitle({required this.title, required this.subtitle});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          title,
          style: theme.textTheme.titleMedium?.copyWith(
            fontWeight: FontWeight.w800,
          ),
        ),
        const SizedBox(height: 4),
        Text(
          subtitle,
          style: theme.textTheme.bodySmall?.copyWith(
            color: AppTheme.muted,
          ),
        ),
      ],
    );
  }
}

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

    return Card(
      margin: EdgeInsets.zero,
      child: Padding(
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
                        'Итоговая оценка',
                        style: theme.textTheme.titleMedium?.copyWith(
                          fontWeight: FontWeight.w800,
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        'Вид: $species',
                        style: theme.textTheme.bodyMedium,
                      ),
                      const SizedBox(height: 2),
                      Text(
                        _formatDateTime(timestamp),
                        style: theme.textTheme.bodySmall?.copyWith(
                          color: AppTheme.muted,
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(width: 12),
                _RiskBadge(hero: hero),
              ],
            ),
            const SizedBox(height: 12),

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
                  color: AppTheme.surface2,
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Row(
                  children: [
                    const Icon(Icons.image_not_supported_outlined,
                        color: AppTheme.muted),
                    const SizedBox(width: 10),
                    Expanded(
                      child: Text(
                        'Аннотированное изображение недоступно.',
                        style: theme.textTheme.bodyMedium?.copyWith(
                          color: AppTheme.muted,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
          ],
        ),
      ),
    );
  }

  static String _formatDateTime(DateTime dt) {
    String two(int v) => v < 10 ? '0$v' : '$v';
    return '${two(dt.day)}.${two(dt.month)}.${dt.year} ${two(dt.hour)}:${two(dt.minute)}';
  }
}

class _RiskBadge extends StatelessWidget {
  final _RiskHeroData hero;
  const _RiskBadge({required this.hero});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: hero.background,
        borderRadius: BorderRadius.circular(18),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.end,
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            hero.label,
            style: theme.textTheme.labelMedium?.copyWith(
              fontWeight: FontWeight.w800,
              color: hero.foreground,
            ),
          ),
          const SizedBox(height: 6),
          Text(
            hero.valueText,
            style: theme.textTheme.titleMedium?.copyWith(
              fontWeight: FontWeight.w900,
              color: hero.foreground,
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
      case 'manual':
        methodText = 'вручную';
        break;
      case 'estimated_from_geometry':
        methodText = 'по геометрии AR';
        break;
      case 'species_default':
        methodText = 'по породе';
        break;
      case 'empirical_borisevich_2021':
        methodText = 'Borisevich (2021)';
        break;
      default:
        methodText = method;
    }

    // Вычисляем худшую силу ветра пропорционально (если есть betaMax)
    double? maxForce;
    if (betaMax != null && force != null && betaValue != null && betaValue > 0) {
      maxForce = (force / betaValue) * betaMax;
    }

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppTheme.surface2,
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: AppTheme.border),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: AppTheme.primary.withOpacity(0.15),
              borderRadius: BorderRadius.circular(14),
            ),
            child: const Icon(Icons.air_outlined, color: AppTheme.primary, size: 24),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Коэффициент β (аэродинамика)',
                  style: TextStyle(
                    color: AppTheme.text,
                    fontWeight: FontWeight.w900,
                    fontSize: 15,
                  ),
                ),
                const SizedBox(height: 10),
                
                _buildRow('Ожидаемый β:', _fmt(betaValue, suffix: ' кг/с'), AppTheme.success),
                
                if (betaMax != null && (betaValue == null || betaMax > betaValue)) ...[
                  const SizedBox(height: 6),
                  _buildRow('Худший сценарий\n(жесткая крона):', _fmt(betaMax, suffix: ' кг/с'), AppTheme.warning),
                ],

                const SizedBox(height: 8),
                Text(
                  '$source · $methodText',
                  style: const TextStyle(
                    color: AppTheme.muted,
                    fontSize: 11,
                    fontWeight: FontWeight.w600,
                    height: 1.3,
                  ),
                ),

                if (force != null) ...[
                  const Padding(
                    padding: EdgeInsets.symmetric(vertical: 12),
                    child: Divider(color: AppTheme.border, height: 1),
                  ),
                  const Text(
                    'Ветровая сила (F = β · v)',
                    style: TextStyle(
                      color: AppTheme.text,
                      fontWeight: FontWeight.w900,
                      fontSize: 13,
                    ),
                  ),
                  const SizedBox(height: 8),
                  _buildRow('Расчетная сила:', _fmt(force, suffix: ' Н'), AppTheme.success),
                  
                  if (maxForce != null && maxForce > force) ...[
                    const SizedBox(height: 6),
                    _buildRow('При порыве и\nжесткой кроне:', 'до ${_fmt(maxForce, suffix: ' Н')}', AppTheme.warning),
                  ],
                ],
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
        Expanded(
          child: Text(
            label,
            style: const TextStyle(
              color: AppTheme.muted,
              fontSize: 13,
              fontWeight: FontWeight.w600,
              height: 1.2,
            ),
          ),
        ),
        const SizedBox(width: 8),
        Text(
          value,
          style: TextStyle(
            color: valueColor,
            fontSize: 14,
            fontWeight: FontWeight.w900,
          ),
        ),
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
    final inputs = (model?['inputs'] as Map?)?.cast<String, dynamic>() ?? {};

    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: AppTheme.surface2,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: AppTheme.border),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(
            available ? Icons.science_outlined : Icons.info_outline,
            color: available ? AppTheme.primary : AppTheme.muted,
          ),
          const SizedBox(width: 10),
          Expanded(
            child: available
                ? Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        'Аналитическая модель ветровой нагрузки',
                        style: TextStyle(
                          color: AppTheme.text,
                          fontWeight: FontWeight.w900,
                        ),
                      ),
                      const SizedBox(height: 8),
                      _miniRow('Суммарная сила', _fmt(outputs['total_force_n'], suffix: ' Н')),
                      _miniRow('Центр нагрузки', _fmt(outputs['center_of_load_m'], suffix: ' м')),
                      _miniRow('Момент у основания', _fmt(outputs['base_moment_nm'], suffix: ' Н·м')),
                      _miniRow('Индекс аналитики', _fmt(outputs['analytical_score'])),
                      const SizedBox(height: 6),
                      Text(
                        'Крона начинается: ${_fmt(inputs['crown_start_height_m'], suffix: ' м')} · элементов: ${inputs['n_elements'] ?? '—'}',
                        style: const TextStyle(
                          color: AppTheme.muted,
                          fontSize: 12,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ],
                  )
                : Text(
                    model?['reason']?.toString() ?? 'Аналитическая модель недоступна.',
                    style: const TextStyle(
                      color: AppTheme.muted,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
          ),
        ],
      ),
    );
  }

  Widget _miniRow(String title, String value) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 4),
      child: Row(
        children: [
          Expanded(
            child: Text(
              title,
              style: const TextStyle(
                color: AppTheme.muted,
                fontSize: 12,
                fontWeight: FontWeight.w700,
              ),
            ),
          ),
          Text(
            value,
            style: const TextStyle(
              color: AppTheme.text,
              fontSize: 13,
              fontWeight: FontWeight.w900,
            ),
          ),
        ],
      ),
    );
  }
}

class _SourceSummaryCard extends StatelessWidget {
  final String? dimensionsSource;
  final bool hasArMeasurements;

  const _SourceSummaryCard({
    required this.dimensionsSource,
    required this.hasArMeasurements,
  });

  @override
  Widget build(BuildContext context) {
    final title = hasArMeasurements
        ? 'Источник размеров: AR-измерение'
        : 'Источник размеров: ${dimensionsSource ?? 'Фото + ИИ'}';
    final subtitle = hasArMeasurements
        ? 'Высота, крона и диаметр переданы в анализ как реальные AR-значения.'
        : 'Если рядом с деревом нет рейки 1 м, часть размеров может быть недоступна.';

    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: hasArMeasurements
            ? AppTheme.primary.withOpacity(0.12)
            : AppTheme.surface2,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(
          color: hasArMeasurements
              ? AppTheme.primary.withOpacity(0.35)
              : AppTheme.border,
        ),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(
            hasArMeasurements ? Icons.view_in_ar_outlined : Icons.image_search_outlined,
            color: hasArMeasurements ? AppTheme.primary : AppTheme.muted,
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: const TextStyle(
                    color: AppTheme.text,
                    fontWeight: FontWeight.w900,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  subtitle,
                  style: const TextStyle(
                    color: AppTheme.muted,
                    fontSize: 12,
                    height: 1.25,
                    fontWeight: FontWeight.w600,
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

class _MetricsGrid extends StatelessWidget {
  final double? heightM;
  final double? crownWidthM;
  final double? trunkDiameterM;
  final double? scalePxToM;
  final String heightSource;
  final String crownSource;
  final String trunkSource;

  const _MetricsGrid({
    required this.heightM,
    required this.crownWidthM,
    required this.trunkDiameterM,
    required this.scalePxToM,
    required this.heightSource,
    required this.crownSource,
    required this.trunkSource,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Row(
          children: [
            Expanded(
              child: _MetricCard(
                title: 'Высота',
                icon: Icons.height,
                value: _fmt(heightM, suffix: 'м'),
                source: heightSource,
              ),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: _MetricCard(
                title: 'Крона',
                icon: Icons.filter_hdr,
                value: _fmt(crownWidthM, suffix: 'м'),
                source: crownSource,
              ),
            ),
          ],
        ),
        const SizedBox(height: 10),
        Row(
          children: [
            Expanded(
              child: _MetricCard(
                title: 'Диаметр ствола',
                icon: Icons.circle_outlined,
                value: _fmt(trunkDiameterM, suffix: 'м'),
                source: trunkSource,
              ),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: _MetricCard(
                title: 'Масштаб',
                icon: Icons.straighten,
                value: scalePxToM == null
                    ? 'Не найден'
                    : '1 px ≈ ${scalePxToM!.toStringAsFixed(4)} м',
                isSecondary: true,
              ),
            ),
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
  final String? source;

  const _MetricCard({
    required this.title,
    required this.icon,
    required this.value,
    this.isSecondary = false,
    this.source,
  });

  @override
  Widget build(BuildContext context) {
    final tileColor = isSecondary ? AppTheme.surface2 : const Color(0xFFEFFBF4);
    final titleColor = isSecondary ? AppTheme.muted : AppTheme.mutedOnLight;
    final valueColor = isSecondary ? AppTheme.text : AppTheme.textOnLight;
    final iconColor = isSecondary ? AppTheme.muted : AppTheme.primary2;

    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: tileColor,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: isSecondary ? AppTheme.border : const Color(0xFFD6E2DA)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, color: iconColor),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: TextStyle(
                    color: titleColor,
                    fontSize: 12,
                    fontWeight: FontWeight.w800,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  value,
                  style: TextStyle(
                    color: valueColor,
                    fontSize: 16,
                    fontWeight: FontWeight.w900,
                  ),
                ),
                if (source != null && source!.trim().isNotEmpty && source != '—') ...[
                  const SizedBox(height: 6),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                    decoration: BoxDecoration(
                      color: source == 'AR'
                          ? AppTheme.primary.withOpacity(0.16)
                          : AppTheme.surface.withOpacity(0.25),
                      borderRadius: BorderRadius.circular(999),
                      border: Border.all(
                        color: source == 'AR'
                            ? AppTheme.primary.withOpacity(0.35)
                            : AppTheme.border,
                      ),
                    ),
                    child: Text(
                      source!,
                      style: TextStyle(
                        color: source == 'AR' ? AppTheme.primary : titleColor,
                        fontSize: 11,
                        fontWeight: FontWeight.w800,
                      ),
                    ),
                  ),
                ],
              ],
            ),
          )
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
    final theme = Theme.of(context);

    final hasAddress = address != null && address!.trim().isNotEmpty;
    final hasCoords = lat != null && lon != null;

    return Card(
      margin: EdgeInsets.zero,
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Icon(Icons.location_on_outlined, color: Color(0xFF1565C0)),
            const SizedBox(width: 10),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    hasAddress ? address! : 'Адрес не найден',
                    style: theme.textTheme.bodyMedium?.copyWith(
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  const SizedBox(height: 6),
                  Text(
                    hasCoords
                        ? 'Координаты: ${lat!.toStringAsFixed(6)}, ${lon!.toStringAsFixed(6)}'
                        : 'Координаты отсутствуют.',
                    style: theme.textTheme.bodySmall?.copyWith(
                      color: AppTheme.muted,
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _ExplanationCard extends StatelessWidget {
  final List<String> lines;
  const _ExplanationCard({required this.lines});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    if (lines.isEmpty) {
      return Card(
        margin: EdgeInsets.zero,
        child: Padding(
          padding: const EdgeInsets.all(14),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Icon(Icons.info_outline, color: AppTheme.muted),
              const SizedBox(width: 10),
              Expanded(
                child: Text(
                  'Для этого анализа сервер не вернул текстовое объяснение факторов риска.',
                  style: theme.textTheme.bodyMedium?.copyWith(
                    color: AppTheme.muted,
                  ),
                ),
              ),
            ],
          ),
        ),
      );
    }

    return Card(
      margin: EdgeInsets.zero,
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          children: [
            for (final l in lines)
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 6),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text('•  '),
                    Expanded(
                      child: Text(
                        l,
                        style: theme.textTheme.bodyMedium,
                      ),
                    ),
                  ],
                ),
              ),
          ],
        ),
      ),
    );
  }
}

class _FootnoteCard extends StatelessWidget {
  final Map<String, dynamic>? raw;
  const _FootnoteCard({required this.raw});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final analysisId = raw?['analysis_id'] as String?;

    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: AppTheme.surface2,
        borderRadius: BorderRadius.circular(20),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(Icons.shield_outlined, color: AppTheme.muted),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Примечание',
                  style: theme.textTheme.bodyMedium?.copyWith(
                    fontWeight: FontWeight.w800,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  'Оценка является аналитической подсказкой и не заменяет осмотр специалистом. При сомнениях используйте подтверждение/коррекцию и зафиксируйте дополнительные фото.',
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: AppTheme.muted,
                  ),
                ),
                if (analysisId != null && analysisId.isNotEmpty) ...[
                  const SizedBox(height: 8),
                  Text(
                    'ID анализа: $analysisId',
                    style: theme.textTheme.labelSmall?.copyWith(
                      color: AppTheme.muted,
                    ),
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
        label: 'Низкий риск',
        valueText: idx != null ? idx.toStringAsFixed(2) : '—',
        background: const Color(0xFFD9F5DC),
        foreground: const Color(0xFF1B5E20),
      );
    }
    if (cat == 'средний') {
      return _RiskHeroData(
        label: 'Средний риск',
        valueText: idx != null ? idx.toStringAsFixed(2) : '—',
        background: const Color(0xFFFFF4D1),
        foreground: const Color(0xFF8D6E00),
      );
    }
    if (cat == 'высокий' || cat.isNotEmpty) {
      return _RiskHeroData(
        label: cat.isNotEmpty ? 'Высокий риск' : 'Риск',
        valueText: idx != null ? idx.toStringAsFixed(2) : '—',
        background: const Color(0xFFFFE1E1),
        foreground: const Color(0xFFB71C1C),
      );
    }

    final fallbackFg = theme.colorScheme.primary;
    return _RiskHeroData(
      label: 'Риск',
      valueText: idx != null ? idx.toStringAsFixed(2) : '—',
      background: const Color(0xFFE0E0E0),
      foreground: fallbackFg,
    );
  }
}