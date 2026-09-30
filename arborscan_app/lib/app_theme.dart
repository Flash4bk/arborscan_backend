import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

/// AS-15: one palette for screens, controls and status messages.
class AppTheme {
  static const background = Color(0xFFF5F5EF);
  static const bg = background;
  static const surface = Color(0xFFFFFFFF);
  static const surface2 = Color(0xFFECEEE3);
  static const surface3 = Color(0xFFDDE2CE);
  static const primary = Color(0xFF526238);
  static const primary2 = Color(0xFF476456);
  static const accent = primary;
  static const text = Color(0xFF242A20);
  static const muted = Color(0xFF5C6456);
  static const border = Color(0xFFCBD0C0);
  static const danger = Color(0xFFA33132);
  static const warning = Color(0xFF805716);
  static const success = Color(0xFF386445);
  static const textOnLight = text;
  static const mutedOnLight = muted;

  static ThemeData light() {
    final scheme =
        ColorScheme.fromSeed(seedColor: primary, brightness: Brightness.light)
            .copyWith(
                primary: primary,
                secondary: primary2,
                surface: surface,
                onSurface: text,
                onSurfaceVariant: muted,
                outline: border,
                error: danger,
                onPrimary: Colors.white,
                onSecondary: Colors.white);
    final base = ThemeData(
        useMaterial3: true,
        colorScheme: scheme,
        scaffoldBackgroundColor: background,
        canvasColor: surface);
    final shape =
        RoundedRectangleBorder(borderRadius: BorderRadius.circular(16));
    final button = ButtonStyle(
      minimumSize: const WidgetStatePropertyAll(Size(48, 52)),
      padding: const WidgetStatePropertyAll(
          EdgeInsets.symmetric(horizontal: 16, vertical: 14)),
      shape: WidgetStatePropertyAll(shape),
      textStyle: const WidgetStatePropertyAll(
          TextStyle(fontSize: 15, fontWeight: FontWeight.w600)),
    );
    return base.copyWith(
      textTheme:
          base.textTheme.apply(bodyColor: text, displayColor: text).copyWith(
                headlineSmall: const TextStyle(
                    fontSize: 24,
                    height: 1.25,
                    fontWeight: FontWeight.w700,
                    color: text),
                titleLarge: const TextStyle(
                    fontSize: 21,
                    height: 1.3,
                    fontWeight: FontWeight.w700,
                    color: text),
                titleMedium: const TextStyle(
                    fontSize: 17,
                    height: 1.35,
                    fontWeight: FontWeight.w600,
                    color: text),
              ),
      appBarTheme: const AppBarTheme(
          backgroundColor: background,
          foregroundColor: text,
          surfaceTintColor: Colors.transparent,
          elevation: 0,
          centerTitle: false,
          systemOverlayStyle: SystemUiOverlayStyle.dark,
          titleTextStyle: TextStyle(
              fontSize: 20, fontWeight: FontWeight.w700, color: text)),
      cardTheme: CardThemeData(
          color: surface,
          surfaceTintColor: Colors.transparent,
          elevation: 0,
          margin: const EdgeInsets.only(bottom: 12),
          shape: shape.copyWith(side: const BorderSide(color: border))),
      elevatedButtonTheme: ElevatedButtonThemeData(
          style: button.copyWith(
              backgroundColor: WidgetStateProperty.resolveWith(
                  (s) => s.contains(WidgetState.disabled) ? surface2 : primary),
              foregroundColor: WidgetStateProperty.resolveWith((s) =>
                  s.contains(WidgetState.disabled) ? muted : Colors.white),
              elevation: const WidgetStatePropertyAll(0))),
      filledButtonTheme: FilledButtonThemeData(style: button),
      outlinedButtonTheme: OutlinedButtonThemeData(
          style: button.copyWith(
              side: const WidgetStatePropertyAll(BorderSide(color: border)))),
      textButtonTheme: TextButtonThemeData(style: button),
      iconButtonTheme: const IconButtonThemeData(
          style:
              ButtonStyle(minimumSize: WidgetStatePropertyAll(Size(48, 48)))),
      dialogTheme: DialogThemeData(
          backgroundColor: surface,
          surfaceTintColor: Colors.transparent,
          shape: shape),
      bottomSheetTheme: const BottomSheetThemeData(
          backgroundColor: surface, showDragHandle: true),
      snackBarTheme: SnackBarThemeData(
          backgroundColor: text,
          contentTextStyle: const TextStyle(color: Colors.white),
          behavior: SnackBarBehavior.floating,
          shape: shape),
      navigationBarTheme: const NavigationBarThemeData(
          backgroundColor: surface,
          indicatorColor: surface3,
          labelTextStyle: WidgetStatePropertyAll(TextStyle(
              fontSize: 12, fontWeight: FontWeight.w600, color: text))),
      dividerTheme: const DividerThemeData(color: border, thickness: 1),
      inputDecorationTheme: InputDecorationTheme(
          filled: true,
          fillColor: surface,
          labelStyle: const TextStyle(color: muted),
          contentPadding:
              const EdgeInsets.symmetric(horizontal: 16, vertical: 16),
          border: OutlineInputBorder(
              borderRadius: BorderRadius.circular(12),
              borderSide: const BorderSide(color: border)),
          enabledBorder: OutlineInputBorder(
              borderRadius: BorderRadius.circular(12),
              borderSide: const BorderSide(color: border)),
          focusedBorder: OutlineInputBorder(
              borderRadius: BorderRadius.circular(12),
              borderSide: const BorderSide(color: primary, width: 2))),
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
    return Container(
      width: width,
      height: height,
      margin: margin,
      child: Material(
        color: color ?? AppTheme.surface,
        shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(radius),
            side: const BorderSide(color: AppTheme.border)),
        clipBehavior: Clip.antiAlias,
        child: InkWell(
            onTap: onTap,
            child: Padding(
                padding: padding ?? const EdgeInsets.all(16), child: child)),
      ),
    );
  }
}

class Ui {
  static Widget sectionTitle(BuildContext context, String text) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(4, 16, 4, 12),
      child: Text(
        text,
        style: Theme.of(context).textTheme.titleMedium?.copyWith(
              color: AppTheme.primary,
              letterSpacing: 0,
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

  static Widget badge(
      {required String text, required Color color, IconData? icon}) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: color.withOpacity(0.15),
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: color.withOpacity(0.5), width: 1.5),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (icon != null) ...[
            Icon(icon, size: 14, color: color),
            const SizedBox(width: 6),
          ],
          Flexible(
              child: Text(text,
                  style: TextStyle(
                      fontWeight: FontWeight.w600,
                      color: color,
                      fontSize: 13,
                      letterSpacing: 0))),
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
    final bg = color ??
        (danger
            ? AppTheme.danger.withOpacity(0.2)
            : primary
                ? AppTheme.primary
                : AppTheme.surface2);
    final fg = foregroundColor ??
        (danger
            ? AppTheme.danger
            : primary
                ? Colors.white
                : AppTheme.text);
    final borderColor = danger
        ? AppTheme.danger
        : primary
            ? AppTheme.primary
            : AppTheme.border;

    final hasSubtitle = subtitle != null && subtitle!.isNotEmpty;

    final child = Row(
      mainAxisSize: expanded ? MainAxisSize.max : MainAxisSize.min,
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        if (loading)
          SizedBox(
            width: 18,
            height: 18,
            child: CircularProgressIndicator(
                strokeWidth: 2, valueColor: AlwaysStoppedAnimation<Color>(fg)),
          )
        else if (icon != null)
          Icon(icon, size: 18, color: fg),
        if ((loading || icon != null) && caption.isNotEmpty)
          const SizedBox(width: 8),
        if (caption.isNotEmpty || hasSubtitle)
          Flexible(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: expanded
                  ? CrossAxisAlignment.start
                  : CrossAxisAlignment.center,
              children: [
                if (caption.isNotEmpty)
                  Text(
                    caption,
                    softWrap: true,
                    style: TextStyle(
                        color: fg,
                        fontWeight: FontWeight.w600,
                        letterSpacing: 0),
                  ),
                if (hasSubtitle) ...[
                  const SizedBox(height: 2),
                  Text(
                    subtitle!,
                    softWrap: true,
                    style: TextStyle(
                      color: fg.withOpacity(0.78),
                      fontSize: 13,
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
            constraints: const BoxConstraints(minHeight: 48),
            padding: padding ??
                EdgeInsets.symmetric(
                  horizontal: compact ? 12 : 16,
                  vertical: compact ? 10 : 14,
                ),
            decoration: BoxDecoration(
              color: isEnabled ? bg : AppTheme.surface2.withOpacity(0.3),
              borderRadius: BorderRadius.circular(compact ? 16 : 20),
              border: Border.all(
                  color: isEnabled
                      ? borderColor.withOpacity(0.5)
                      : AppTheme.border),
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
                  Text(t,
                      style: const TextStyle(
                          color: AppTheme.muted,
                          fontSize: 13,
                          fontWeight: FontWeight.w600,
                          letterSpacing: 0)),
                if (t.isNotEmpty) const SizedBox(height: 4),
                Text(value,
                    style: const TextStyle(
                      color: AppTheme.text,
                      fontSize: 20,
                      fontWeight: FontWeight.w600,
                    )),
                if (subtitle != null && subtitle!.isNotEmpty) ...[
                  const SizedBox(height: 2),
                  Text(subtitle!,
                      style: const TextStyle(
                          color: AppTheme.muted,
                          fontSize: 13,
                          fontWeight: FontWeight.w600)),
                ],
              ],
            ),
          ),
        ],
      ),
    );
  }
}

/// Consistent breathing room for task screens; never imposes fixed text heights.
class AppContentList extends StatelessWidget {
  final List<Widget> children;
  final EdgeInsetsGeometry? padding;
  const AppContentList({super.key, required this.children, this.padding});
  @override
  Widget build(BuildContext context) => ListView.separated(
    padding: padding ?? const EdgeInsets.all(16),
    itemCount: children.length,
    separatorBuilder: (_, __) => const SizedBox(height: 12),
    itemBuilder: (_, index) => children[index],
  );
}
