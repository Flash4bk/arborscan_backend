import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

/// Forest Atlas tokens. Static colors describe the light palette; [color]
/// resolves a semantic token against the nearest theme (including dialogs).
class AppTheme {
  static const background = Color(0xFFF6F3EA);
  static const bg = background;
  static const surface = Color(0xFFFFFDF7);
  static const surface2 = Color(0xFFECEEE2);
  static const surface3 = Color(0xFFDAE7A7);
  static const primary = Color(0xFF173E32);
  static const primary2 = Color(0xFF53645C);
  static const accent = primary2;
  static const text = Color(0xFF173E32);
  static const muted = Color(0xFF53645C);
  static const border = Color(0xFF788B7C);
  static const danger = Color(0xFF922D22);
  static const warning = Color(0xFF78501C);
  static const success = Color(0xFF386445);
  static const textOnLight = text;
  static const mutedOnLight = muted;
  // A filled brand surface keeps white text legible in both modes.
  static const brand = primary;

  static Color color(BuildContext context, Color token) =>
      resolve(token, Theme.of(context).brightness);

  static Color resolve(Color token, Brightness brightness) {
    if (brightness == Brightness.light) return token;
    if (token == background) return const Color(0xFF122B25);
    if (token == surface) return const Color(0xFF1A3930);
    if (token == surface2) return const Color(0xFF24483B);
    if (token == surface3) return const Color(0xFF3B5540);
    if (token == primary || token == text) return const Color(0xFFEFF3E6);
    if (token == muted || token == primary2) return const Color(0xFFC0CDBD);
    if (token == border) return const Color(0xFF8C9E8F);
    if (token == danger) return const Color(0xFFFFB4A8);
    if (token == warning) return const Color(0xFFF1CB89);
    if (token == success) return const Color(0xFFB1D6AA);
    return token;
  }

  static ThemeData light() => _theme(Brightness.light);
  static ThemeData dark() => _theme(Brightness.dark);
  static ThemeData _theme(Brightness brightness) {
    final dark = brightness == Brightness.dark;
    Color c(Color token) => resolve(token, brightness);
    final scheme =
        ColorScheme.fromSeed(seedColor: primary, brightness: brightness)
            .copyWith(
                primary: dark ? surface3 : primary,
                onPrimary: dark ? primary : surface,
                secondary: c(muted),
                onSecondary: c(surface),
                surface: c(surface),
                onSurface: c(text),
                onSurfaceVariant: c(muted),
                outline: c(border),
                error: c(danger),
                onError: dark ? primary : surface);
    final base = ThemeData(
        useMaterial3: true,
        colorScheme: scheme,
        fontFamily: 'Atlas',
        scaffoldBackgroundColor: c(background),
        canvasColor: c(surface));
    final shape =
        RoundedRectangleBorder(borderRadius: BorderRadius.circular(13));
    final button = ButtonStyle(
        minimumSize: const WidgetStatePropertyAll(Size(48, 52)),
        padding: const WidgetStatePropertyAll(
            EdgeInsets.symmetric(horizontal: 16, vertical: 14)),
        shape: WidgetStatePropertyAll(
            RoundedRectangleBorder(borderRadius: BorderRadius.circular(11))),
        textStyle: const WidgetStatePropertyAll(TextStyle(
            fontFamily: 'Atlas', fontSize: 15, fontWeight: FontWeight.w600)),
        animationDuration: const Duration(milliseconds: 120));
    final textTheme = base.textTheme.copyWith(
        headlineLarge: TextStyle(
            fontFamily: 'Literata',
            fontSize: 34,
            height: 1.16,
            color: c(text),
            fontWeight: FontWeight.w600),
        headlineMedium: TextStyle(
            fontFamily: 'Literata',
            fontSize: 30,
            height: 1.2,
            color: c(text),
            fontWeight: FontWeight.w600),
        headlineSmall: TextStyle(
            fontFamily: 'Literata',
            fontSize: 24,
            height: 1.22,
            color: c(text),
            fontWeight: FontWeight.w600),
        titleLarge: TextStyle(
            fontFamily: 'Literata',
            fontSize: 22,
            height: 1.3,
            color: c(text),
            fontWeight: FontWeight.w600),
        titleMedium: TextStyle(
            fontFamily: 'Atlas',
            fontSize: 17,
            height: 1.35,
            color: c(text),
            fontWeight: FontWeight.w600));
    return base.copyWith(
      textTheme: textTheme,
      appBarTheme: AppBarTheme(
          backgroundColor: c(background),
          foregroundColor: c(text),
          surfaceTintColor: Colors.transparent,
          elevation: 0,
          centerTitle: false,
          systemOverlayStyle:
              dark ? SystemUiOverlayStyle.light : SystemUiOverlayStyle.dark,
          titleTextStyle: TextStyle(
              fontFamily: 'Atlas',
              fontSize: 20,
              fontWeight: FontWeight.w600,
              color: c(text))),
      cardTheme: CardThemeData(
          color: c(surface),
          surfaceTintColor: Colors.transparent,
          elevation: 0,
          margin: const EdgeInsets.only(bottom: 12),
          shape: shape.copyWith(
              side: BorderSide(color: c(border).withValues(alpha: .55)))),
      elevatedButtonTheme: ElevatedButtonThemeData(
          style: button.copyWith(
              backgroundColor: WidgetStateProperty.resolveWith((s) =>
                  s.contains(WidgetState.disabled)
                      ? c(surface2)
                      : scheme.primary),
              foregroundColor: WidgetStateProperty.resolveWith((s) =>
                  s.contains(WidgetState.disabled)
                      ? c(muted)
                      : scheme.onPrimary),
              elevation: const WidgetStatePropertyAll(0))),
      filledButtonTheme: FilledButtonThemeData(style: button),
      outlinedButtonTheme: OutlinedButtonThemeData(
          style: button.copyWith(
              side: WidgetStatePropertyAll(BorderSide(color: c(border))))),
      textButtonTheme: TextButtonThemeData(style: button),
      iconButtonTheme: const IconButtonThemeData(
          style:
              ButtonStyle(minimumSize: WidgetStatePropertyAll(Size(48, 48)))),
      dialogTheme: DialogThemeData(
          backgroundColor: c(surface),
          surfaceTintColor: Colors.transparent,
          shape: shape),
      bottomSheetTheme: BottomSheetThemeData(
          backgroundColor: c(surface), showDragHandle: true),
      snackBarTheme: SnackBarThemeData(
          backgroundColor: c(text),
          contentTextStyle: TextStyle(color: c(background)),
          behavior: SnackBarBehavior.floating,
          shape: shape),
      navigationBarTheme: NavigationBarThemeData(
          backgroundColor: c(background),
          indicatorColor: surface3,
          iconTheme: WidgetStateProperty.resolveWith((s) => IconThemeData(
              color: s.contains(WidgetState.selected) ? primary : c(muted))),
          labelTextStyle: WidgetStatePropertyAll(TextStyle(
              fontFamily: 'Atlas',
              fontSize: 12,
              fontWeight: FontWeight.w600,
              color: c(text)))),
      dividerTheme: DividerThemeData(
          color: c(border).withValues(alpha: .55), thickness: 1),
      inputDecorationTheme: InputDecorationTheme(
          filled: true,
          fillColor: c(surface),
          labelStyle: TextStyle(color: c(muted)),
          contentPadding:
              const EdgeInsets.symmetric(horizontal: 16, vertical: 16),
          border: OutlineInputBorder(
              borderRadius: BorderRadius.circular(11),
              borderSide: BorderSide(color: c(border))),
          enabledBorder: OutlineInputBorder(
              borderRadius: BorderRadius.circular(11),
              borderSide: BorderSide(color: c(border))),
          focusedBorder: OutlineInputBorder(
              borderRadius: BorderRadius.circular(11),
              borderSide: BorderSide(color: scheme.primary, width: 2))),
    );
  }
}

/// The approved SVG path, painted without an extra runtime dependency.
class AtlasMark extends StatelessWidget {
  final double size;
  const AtlasMark({super.key, this.size = 32});
  @override
  Widget build(BuildContext context) => Semantics(
      label: 'ArborScan — ось и крона',
      image: true,
      child: SizedBox.square(
          dimension: size,
          child: CustomPaint(
              painter:
                  _AtlasMarkPainter(AppTheme.color(context, AppTheme.text)))));
}

class _AtlasMarkPainter extends CustomPainter {
  final Color color;
  _AtlasMarkPainter(this.color);
  @override
  void paint(Canvas canvas, Size size) {
    canvas.save();
    canvas.scale(size.width / 100, size.height / 100);
    final p = Path()
      ..moveTo(20, 84)
      ..lineTo(20, 39)
      ..lineTo(50, 13)
      ..lineTo(80, 39)
      ..lineTo(80, 84)
      ..moveTo(34, 84)
      ..lineTo(34, 47)
      ..lineTo(50, 32)
      ..lineTo(66, 47)
      ..lineTo(66, 84)
      ..moveTo(50, 32)
      ..lineTo(50, 84)
      ..moveTo(14, 58)
      ..lineTo(86, 58);
    canvas.drawPath(
        p,
        Paint()
          ..color = color
          ..style = PaintingStyle.stroke
          ..strokeWidth = 5
          ..strokeCap = StrokeCap.round
          ..strokeJoin = StrokeJoin.round);
    canvas.restore();
  }

  @override
  bool shouldRepaint(_AtlasMarkPainter oldDelegate) =>
      oldDelegate.color != color;
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
    this.radius = 13,
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
        color: color ?? AppTheme.color(context, AppTheme.surface),
        shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(radius),
            side: BorderSide(color: AppTheme.color(context, AppTheme.border))),
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
              color: AppTheme.color(context, AppTheme.primary),
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
        color: color.withValues(alpha: 0.15),
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: color.withValues(alpha: 0.5), width: 1.5),
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
            ? AppTheme.color(context, AppTheme.danger).withValues(alpha: 0.2)
            : primary
                ? AppTheme.brand
                : AppTheme.color(context, AppTheme.surface2));
    final fg = !isEnabled
        ? AppTheme.color(context, AppTheme.muted)
        : foregroundColor ??
            (danger
                ? AppTheme.color(context, AppTheme.danger)
                : primary
                    ? Colors.white
                    : AppTheme.color(context, AppTheme.text));
    final borderColor = danger
        ? AppTheme.color(context, AppTheme.danger)
        : primary
            ? AppTheme.brand
            : AppTheme.color(context, AppTheme.border);

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
                      color: fg.withValues(alpha: 0.78),
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
          borderRadius: BorderRadius.circular(11),
          child: Container(
            constraints: const BoxConstraints(minHeight: 48),
            padding: padding ??
                EdgeInsets.symmetric(
                  horizontal: compact ? 12 : 16,
                  vertical: compact ? 10 : 14,
                ),
            decoration: BoxDecoration(
              color: isEnabled
                  ? bg
                  : AppTheme.color(context, AppTheme.surface2)
                      .withValues(alpha: 0.3),
              borderRadius: BorderRadius.circular(11),
              border: Border.all(
                  color: isEnabled
                      ? borderColor.withValues(alpha: 0.5)
                      : AppTheme.color(context, AppTheme.border)),
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
    final c = color ?? AppTheme.color(context, AppTheme.primary);
    final t = title ?? label ?? '';

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color:
            AppTheme.color(context, AppTheme.surface2).withValues(alpha: 0.5),
        borderRadius: BorderRadius.circular(13),
        border: Border.all(color: c.withValues(alpha: 0.3)),
      ),
      child: Row(
        children: [
          if (icon != null) ...[
            Container(
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                color: c.withValues(alpha: 0.15),
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
                      style: TextStyle(
                          color: AppTheme.color(context, AppTheme.muted),
                          fontSize: 13,
                          fontWeight: FontWeight.w600,
                          letterSpacing: 0)),
                if (t.isNotEmpty) const SizedBox(height: 4),
                Text(value,
                    style: TextStyle(
                      color: AppTheme.color(context, AppTheme.text),
                      fontSize: 20,
                      fontWeight: FontWeight.w600,
                    )),
                if (subtitle != null && subtitle!.isNotEmpty) ...[
                  const SizedBox(height: 2),
                  Text(subtitle!,
                      style: TextStyle(
                          color: AppTheme.color(context, AppTheme.muted),
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
