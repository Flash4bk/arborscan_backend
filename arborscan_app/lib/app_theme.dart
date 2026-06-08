import 'dart:ui';

import 'package:flutter/material.dart';

class AppTheme {
  static const Color background = Color(0xFF071414);
  static const Color bg = background;

  static const Color surface = Color(0xFF0D1B2A);
  static const Color surface2 = Color(0xFF10243A);
  static const Color surface3 = Color(0xFF172B43);

  static const Color primary = Color(0xFF37D88B);
  static const Color primary2 = Color(0xFF1E6F5C);
  static const Color accent = primary;

  static const Color text = Color(0xFFF8FAFC);
  static const Color muted = Color(0xFFB8C4CC);
  static const Color border = Color(0xFF24364A);

  static const Color danger = Color(0xFFFF6B6B);
  static const Color warning = Color(0xFFFBBF24);
  static const Color success = Color(0xFF37D88B);

  static const Color textOnLight = Color(0xFF101418);
  static const Color mutedOnLight = Color(0xFF4B5563);

  static ThemeData light() {
    final base = ThemeData(
      useMaterial3: true,
      brightness: Brightness.dark,
      colorScheme: const ColorScheme.dark(
        primary: primary,
        secondary: primary2,
        surface: surface,
        background: background,
        error: danger,
        onPrimary: Color(0xFF06140E),
        onSecondary: text,
        onSurface: text,
        onBackground: text,
        onError: Color(0xFF2A0505),
        onSurfaceVariant: muted,
        outline: border,
      ),
      scaffoldBackgroundColor: background,
      canvasColor: background,
    );

    final radius = BorderRadius.circular(18);

    return base.copyWith(
      textTheme: base.textTheme.apply(
        bodyColor: text,
        displayColor: text,
      ).copyWith(
        headlineSmall: base.textTheme.headlineSmall?.copyWith(
          fontWeight: FontWeight.w800,
          color: text,
          letterSpacing: 0.1,
        ),
        titleLarge: base.textTheme.titleLarge?.copyWith(
          fontWeight: FontWeight.w800,
          color: text,
        ),
        titleMedium: base.textTheme.titleMedium?.copyWith(
          fontWeight: FontWeight.w700,
          color: text,
        ),
        titleSmall: base.textTheme.titleSmall?.copyWith(
          fontWeight: FontWeight.w700,
          color: text,
        ),
        bodyLarge: base.textTheme.bodyLarge?.copyWith(color: text, height: 1.25),
        bodyMedium: base.textTheme.bodyMedium?.copyWith(color: text, height: 1.25),
        bodySmall: base.textTheme.bodySmall?.copyWith(color: muted, height: 1.25),
        labelLarge: base.textTheme.labelLarge?.copyWith(color: text, fontWeight: FontWeight.w800),
        labelMedium: base.textTheme.labelMedium?.copyWith(color: text, fontWeight: FontWeight.w700),
        labelSmall: base.textTheme.labelSmall?.copyWith(color: muted, fontWeight: FontWeight.w700),
      ),
      appBarTheme: const AppBarTheme(
        backgroundColor: background,
        surfaceTintColor: background,
        elevation: 0,
        centerTitle: false,
        titleTextStyle: TextStyle(
          fontSize: 22,
          fontWeight: FontWeight.w800,
          color: text,
          letterSpacing: 0.2,
        ),
        iconTheme: IconThemeData(color: text),
        actionsIconTheme: IconThemeData(color: text),
      ),
      cardTheme: CardThemeData(
        color: surface,
        surfaceTintColor: surface,
        elevation: 0,
        margin: EdgeInsets.zero,
        shape: RoundedRectangleBorder(
          borderRadius: radius,
          side: const BorderSide(color: border, width: 1),
        ),
      ),
      dividerTheme: const DividerThemeData(color: border, thickness: 1, space: 1),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: surface2,
        labelStyle: const TextStyle(color: muted, fontWeight: FontWeight.w600),
        hintStyle: const TextStyle(color: muted),
        contentPadding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
        border: OutlineInputBorder(borderRadius: radius, borderSide: const BorderSide(color: border)),
        enabledBorder: OutlineInputBorder(borderRadius: radius, borderSide: const BorderSide(color: border)),
        focusedBorder: OutlineInputBorder(borderRadius: radius, borderSide: const BorderSide(color: primary, width: 1.4)),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: primary,
          foregroundColor: const Color(0xFF06140E),
          disabledBackgroundColor: const Color(0xFF1A2B3A),
          disabledForegroundColor: const Color(0xFF7D8B96),
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
          textStyle: const TextStyle(fontWeight: FontWeight.w800),
        ),
      ),
      filledButtonTheme: FilledButtonThemeData(
        style: FilledButton.styleFrom(
          backgroundColor: primary,
          foregroundColor: const Color(0xFF06140E),
          disabledBackgroundColor: const Color(0xFF1A2B3A),
          disabledForegroundColor: const Color(0xFF7D8B96),
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
          textStyle: const TextStyle(fontWeight: FontWeight.w800),
        ),
      ),
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: OutlinedButton.styleFrom(
          foregroundColor: text,
          disabledForegroundColor: const Color(0xFF7D8B96),
          side: const BorderSide(color: border),
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
          textStyle: const TextStyle(fontWeight: FontWeight.w700),
        ),
      ),
      textButtonTheme: TextButtonThemeData(
        style: TextButton.styleFrom(
          foregroundColor: primary,
          disabledForegroundColor: const Color(0xFF7D8B96),
          textStyle: const TextStyle(fontWeight: FontWeight.w700),
        ),
      ),
      floatingActionButtonTheme: FloatingActionButtonThemeData(
        backgroundColor: primary,
        foregroundColor: const Color(0xFF06140E),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
      ),
      chipTheme: base.chipTheme.copyWith(
        backgroundColor: surface2,
        selectedColor: primary.withOpacity(0.22),
        disabledColor: const Color(0xFF172333),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(999)),
        side: const BorderSide(color: border),
        labelStyle: const TextStyle(color: text, fontWeight: FontWeight.w700),
        secondaryLabelStyle: const TextStyle(color: text, fontWeight: FontWeight.w700),
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      ),
      bottomNavigationBarTheme: const BottomNavigationBarThemeData(
        backgroundColor: Color(0xFF0A1824),
        selectedItemColor: primary,
        unselectedItemColor: Color(0xFFA8B3BC),
        selectedLabelStyle: TextStyle(fontWeight: FontWeight.w800),
        unselectedLabelStyle: TextStyle(fontWeight: FontWeight.w700),
        type: BottomNavigationBarType.fixed,
        elevation: 0,
      ),
      bottomSheetTheme: const BottomSheetThemeData(
        backgroundColor: surface,
        surfaceTintColor: surface,
        showDragHandle: true,
        dragHandleColor: muted,
      ),
      dialogTheme: DialogThemeData(
        backgroundColor: surface,
        surfaceTintColor: surface,
        titleTextStyle: const TextStyle(color: text, fontSize: 18, fontWeight: FontWeight.w800),
        contentTextStyle: const TextStyle(color: text),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
      ),
      snackBarTheme: SnackBarThemeData(
        backgroundColor: const Color(0xFF111827),
        contentTextStyle: const TextStyle(color: Colors.white, fontWeight: FontWeight.w600),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
        behavior: SnackBarBehavior.floating,
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

  const GlassPanel({
    super.key,
    required this.child,
    this.padding,
    this.margin,
    this.radius = 22,
    this.color,
    this.gradient,
    this.border,
    this.boxShadow,
    this.blur = 12,
    this.width,
    this.height,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      width: width,
      height: height,
      margin: margin,
      child: ClipRRect(
        borderRadius: BorderRadius.circular(radius),
        child: BackdropFilter(
          filter: ImageFilter.blur(sigmaX: blur, sigmaY: blur),
          child: Container(
            padding: padding ?? const EdgeInsets.all(14),
            decoration: BoxDecoration(
              color: color ?? AppTheme.surface.withOpacity(0.92),
              gradient: gradient,
              borderRadius: BorderRadius.circular(radius),
              border: border ?? Border.all(color: AppTheme.border.withOpacity(0.85), width: 1),
              boxShadow: boxShadow,
            ),
            child: child,
          ),
        ),
      ),
    );
  }
}

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
    final bg = color ?? (danger ? AppTheme.danger : primary ? AppTheme.primary : AppTheme.surface2);
    final fg = foregroundColor ?? (primary || danger ? const Color(0xFF06140E) : AppTheme.text);

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
                    style: TextStyle(color: fg, fontWeight: FontWeight.w800),
                  ),
                if (hasSubtitle) ...[
                  const SizedBox(height: 2),
                  Text(
                    subtitle!,
                    overflow: TextOverflow.ellipsis,
                    style: TextStyle(
                      color: fg.withOpacity(0.78),
                      fontSize: 12,
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
        color: isEnabled ? bg : AppTheme.surface2.withOpacity(0.55),
        borderRadius: BorderRadius.circular(compact ? 13 : 16),
        child: InkWell(
          onTap: isEnabled ? callback : null,
          borderRadius: BorderRadius.circular(compact ? 13 : 16),
          child: Container(
            padding: padding ?? EdgeInsets.symmetric(
              horizontal: compact ? 10 : 14,
              vertical: compact ? 8 : 12,
            ),
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(compact ? 13 : 16),
              border: Border.all(color: AppTheme.border),
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
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: AppTheme.surface2,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: AppTheme.border),
      ),
      child: Row(
        children: [
          if (icon != null) ...[
            Icon(icon, color: c, size: 24),
            const SizedBox(width: 10),
          ],
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                if (t.isNotEmpty)
                  Text(t, style: const TextStyle(color: AppTheme.muted, fontSize: 12, fontWeight: FontWeight.w700)),
                if (t.isNotEmpty) const SizedBox(height: 3),
                Text(value, style: const TextStyle(color: AppTheme.text, fontSize: 18, fontWeight: FontWeight.w900)),
                if (subtitle != null && subtitle!.isNotEmpty) ...[
                  const SizedBox(height: 3),
                  Text(subtitle!, style: const TextStyle(color: AppTheme.muted, fontSize: 12, fontWeight: FontWeight.w600)),
                ],
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class Ui {
  static Widget sectionTitle(BuildContext context, String text) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(2, 12, 2, 8),
      child: Text(text, style: Theme.of(context).textTheme.titleMedium),
    );
  }

  static Widget paddedCard(
    BuildContext context, {
    required Widget child,
    EdgeInsetsGeometry? padding,
    EdgeInsetsGeometry? margin,
    Color? color,
    double radius = 18,
  }) {
    return Container(
      margin: margin,
      child: Card(
        color: color ?? AppTheme.surface,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(radius),
          side: const BorderSide(color: AppTheme.border),
        ),
        child: Padding(
          padding: padding ?? const EdgeInsets.all(14),
          child: child,
        ),
      ),
    );
  }

  static Widget badge({
    required String text,
    required Color color,
    IconData? icon,
  }) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: color.withOpacity(0.16),
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: color.withOpacity(0.32)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (icon != null) ...[
            Icon(icon, size: 14, color: color),
            const SizedBox(width: 6),
          ],
          Text(text, style: TextStyle(fontWeight: FontWeight.w800, color: color, fontSize: 12)),
        ],
      ),
    );
  }
}
