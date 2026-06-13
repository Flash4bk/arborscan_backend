import 'dart:ui';
import 'package:flutter/material.dart';

class AppTheme {
  // Eco-Futurism Palette
  static const Color background = Color(0xFF041217); // Глубокий сине-зеленый
  static const Color bg = background;

  static const Color surface = Color(0xFF082229); // Стекло 1
  static const Color surface2 = Color(0xFF0B2E36); // Стекло 2
  static const Color surface3 = Color(0xFF103A43);

  static const Color primary = Color(0xFF00FFA3); // Neon Mint
  static const Color primary2 = Color(0xFF00E5FF); // Cyan Glow
  static const Color accent = primary;

  static const Color text = Color(0xFFF8FAFC);
  static const Color muted = Color(0xFF6B929B); // Мягкий лесной тон
  static const Color border = Color(0x5500FFA3); // Полупрозрачный мятный бордер

  static const Color danger = Color(0xFFFF4B6B);
  static const Color warning = Color(0xFFFFB800);
  static const Color success = Color(0xFF00FFA3);

  static const Color textOnLight = Color(0xFF041217);
  static const Color mutedOnLight = Color(0xFF385E66);

  static ThemeData light() {
    final base = ThemeData(
      useMaterial3: true,
      brightness: Brightness.dark,
      scaffoldBackgroundColor: background,
      canvasColor: background,
      colorScheme: const ColorScheme.dark(
        primary: primary,
        secondary: primary2,
        surface: surface,
        background: background,
        error: danger,
      ),
    );

    final radius = BorderRadius.circular(24);

    return base.copyWith(
      textTheme: base.textTheme.apply(
        bodyColor: text,
        displayColor: text,
        fontFamily: 'Roboto',
      ).copyWith(
        headlineSmall: base.textTheme.headlineSmall?.copyWith(
          fontWeight: FontWeight.w900,
          color: text,
          letterSpacing: 1.2,
          shadows: [const Shadow(color: primary2, blurRadius: 12)],
        ),
        titleLarge: base.textTheme.titleLarge?.copyWith(
          fontWeight: FontWeight.w800,
          color: text,
          shadows: [const Shadow(color: primary, blurRadius: 8)],
        ),
        titleMedium: base.textTheme.titleMedium?.copyWith(
          fontWeight: FontWeight.w700,
          color: text,
        ),
      ),
      appBarTheme: const AppBarTheme(
        backgroundColor: Colors.transparent,
        elevation: 0,
        centerTitle: true,
        titleTextStyle: TextStyle(
          fontSize: 20,
          fontWeight: FontWeight.w900,
          color: text,
          letterSpacing: 2.0,
          shadows: [Shadow(color: primary2, blurRadius: 10)],
        ),
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: surface.withOpacity(0.5),
        labelStyle: const TextStyle(color: muted, fontWeight: FontWeight.w600),
        contentPadding: const EdgeInsets.symmetric(horizontal: 18, vertical: 16),
        border: OutlineInputBorder(borderRadius: radius, borderSide: const BorderSide(color: border)),
        enabledBorder: OutlineInputBorder(borderRadius: radius, borderSide: const BorderSide(color: border)),
        focusedBorder: OutlineInputBorder(borderRadius: radius, borderSide: const BorderSide(color: primary, width: 2)),
      ),
    );
  }
}

class GlassPanel extends StatelessWidget {
  final Widget child;
  final EdgeInsetsGeometry? padding;
  final EdgeInsetsGeometry? margin;
  final double radius;
  final Color? color;
  final Gradient? gradient;
  final Border? border;
  final List<BoxShadow>? boxShadow;
  final double blur;
  final double? width;
  final double? height;
  final VoidCallback? onTap;

  const GlassPanel({
    super.key,
    required this.child,
    this.padding,
    this.margin,
    this.radius = 24,
    this.color,
    this.gradient,
    this.border,
    this.boxShadow,
    this.blur = 16,
    this.width,
    this.height,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    Widget content = Container(
      width: width,
      height: height,
      margin: margin,
      child: ClipRRect(
        borderRadius: BorderRadius.circular(radius),
        child: BackdropFilter(
          filter: ImageFilter.blur(sigmaX: blur, sigmaY: blur),
          child: Container(
            padding: padding ?? const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: color ?? AppTheme.surface.withOpacity(0.3),
              gradient: gradient,
              borderRadius: BorderRadius.circular(radius),
              border: border ?? Border.all(color: AppTheme.primary.withOpacity(0.2), width: 1.5),
              boxShadow: boxShadow ?? [
                BoxShadow(
                  color: AppTheme.primary2.withOpacity(0.05),
                  blurRadius: 20,
                  spreadRadius: -5,
                )
              ],
            ),
            child: child,
          ),
        ),
      ),
    );

    if (onTap != null) {
      return GestureDetector(onTap: onTap, child: content);
    }
    return content;
  }
}

class Ui {
  static Widget sectionTitle(BuildContext context, String text) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(4, 16, 4, 12),
      child: Text(
        text.toUpperCase(),
        style: Theme.of(context).textTheme.titleMedium?.copyWith(
          color: AppTheme.primary,
          letterSpacing: 1.5,
          shadows: [const Shadow(color: AppTheme.primary, blurRadius: 10)],
        ),
      ),
    );
  }

  static Widget paddedCard(
    BuildContext context, {
    required Widget child,
    EdgeInsetsGeometry? padding,
    EdgeInsetsGeometry? margin,
  }) {
    return GlassPanel(
      margin: margin ?? const EdgeInsets.only(bottom: 12),
      padding: padding,
      child: child,
    );
  }

  static Widget badge({required String text, required Color color, IconData? icon}) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: color.withOpacity(0.15),
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: color.withOpacity(0.5), width: 1.5),
        boxShadow: [BoxShadow(color: color.withOpacity(0.2), blurRadius: 8)],
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (icon != null) ...[
            Icon(icon, size: 14, color: color),
            const SizedBox(width: 6),
          ],
          Text(text, style: TextStyle(fontWeight: FontWeight.w900, color: color, fontSize: 12, letterSpacing: 0.5)),
        ],
      ),
    );
  }
}

// Восстановленные виджеты, которые нужны для history_tab_page.dart
class AppActionButton extends StatelessWidget {
  final String? label;
  final String? title;
  final String? text;
  final String? subtitle;
  final IconData? icon;
  final VoidCallback? onPressed;
  final VoidCallback? onTap;
  final bool primary;
  final bool danger;
  final bool expanded;
  final bool compact;
  final bool loading;
  final bool enabled;
  final EdgeInsetsGeometry? padding;
  final Color? color;
  final Color? foregroundColor;

  const AppActionButton({
    super.key,
    this.label,
    this.title,
    this.text,
    this.subtitle,
    this.icon,
    this.onPressed,
    this.onTap,
    this.primary = false,
    this.danger = false,
    this.expanded = false,
    this.compact = false,
    this.loading = false,
    this.enabled = true,
    this.padding,
    this.color,
    this.foregroundColor,
  });

  @override
  Widget build(BuildContext context) {
    final caption = label ?? title ?? text ?? '';
    final callback = onPressed ?? onTap;
    final isEnabled = enabled && !loading && callback != null;
    final bg = color ?? (danger ? AppTheme.danger.withOpacity(0.2) : primary ? AppTheme.primary.withOpacity(0.2) : AppTheme.surface2);
    final fg = foregroundColor ?? (danger ? AppTheme.danger : primary ? AppTheme.primary : AppTheme.text);
    final borderColor = danger ? AppTheme.danger : primary ? AppTheme.primary : AppTheme.border;

    final hasSubtitle = subtitle != null && subtitle!.isNotEmpty;

    final child = Row(
      mainAxisSize: expanded ? MainAxisSize.max : MainAxisSize.min,
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        if (loading)
          SizedBox(
            width: 18,
            height: 18,
            child: CircularProgressIndicator(strokeWidth: 2, valueColor: AlwaysStoppedAnimation<Color>(fg)),
          )
        else if (icon != null)
          Icon(icon, size: 18, color: fg),
        if ((loading || icon != null) && caption.isNotEmpty) const SizedBox(width: 8),
        if (caption.isNotEmpty || hasSubtitle)
          Flexible(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: expanded ? CrossAxisAlignment.start : CrossAxisAlignment.center,
              children: [
                if (caption.isNotEmpty)
                  Text(
                    caption,
                    overflow: TextOverflow.ellipsis,
                    style: TextStyle(color: fg, fontWeight: FontWeight.w900, letterSpacing: 1.0),
                  ),
                if (hasSubtitle) ...[
                  const SizedBox(height: 2),
                  Text(
                    subtitle!,
                    overflow: TextOverflow.ellipsis,
                    style: TextStyle(
                      color: fg.withOpacity(0.78),
                      fontSize: 10,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ],
              ],
            ),
          ),
      ],
    );

    return SizedBox(
      width: expanded ? double.infinity : null,
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          onTap: isEnabled ? callback : null,
          borderRadius: BorderRadius.circular(compact ? 16 : 20),
          child: Container(
            padding: padding ?? EdgeInsets.symmetric(
              horizontal: compact ? 12 : 16,
              vertical: compact ? 10 : 14,
            ),
            decoration: BoxDecoration(
              color: isEnabled ? bg : AppTheme.surface2.withOpacity(0.3),
              borderRadius: BorderRadius.circular(compact ? 16 : 20),
              border: Border.all(color: isEnabled ? borderColor.withOpacity(0.5) : AppTheme.border),
            ),
            child: child,
          ),
        ),
      ),
    );
  }
}

class AppStatCard extends StatelessWidget {
  final String? title;
  final String? label;
  final String value;
  final String? subtitle;
  final IconData? icon;
  final Color? color;

  const AppStatCard({
    super.key,
    this.title,
    this.label,
    required this.value,
    this.subtitle,
    this.icon,
    this.color,
  });

  @override
  Widget build(BuildContext context) {
    final c = color ?? AppTheme.primary;
    final t = title ?? label ?? '';

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppTheme.surface2.withOpacity(0.5),
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: c.withOpacity(0.3)),
        boxShadow: [BoxShadow(color: c.withOpacity(0.05), blurRadius: 10)],
      ),
      child: Row(
        children: [
          if (icon != null) ...[
            Container(
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                color: c.withOpacity(0.15),
                shape: BoxShape.circle,
              ),
              child: Icon(icon, color: c, size: 24),
            ),
            const SizedBox(width: 12),
          ],
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                if (t.isNotEmpty)
                  Text(t.toUpperCase(), style: const TextStyle(color: AppTheme.muted, fontSize: 10, fontWeight: FontWeight.w900, letterSpacing: 1.0)),
                if (t.isNotEmpty) const SizedBox(height: 4),
                Text(value, style: TextStyle(color: AppTheme.text, fontSize: 20, fontWeight: FontWeight.w900, shadows: [Shadow(color: c, blurRadius: 8)])),
                if (subtitle != null && subtitle!.isNotEmpty) ...[
                  const SizedBox(height: 2),
                  Text(subtitle!, style: const TextStyle(color: AppTheme.muted, fontSize: 11, fontWeight: FontWeight.w600)),
                ],
              ],
            ),
          ),
        ],
      ),
    );
  }
}