import 'dart:ui';
import 'package:flutter/material.dart';

class AppTheme {
  static const Color primary = Color(0xFF46E0A1);
  static const Color accent = Color(0xFF16C784);

  static const Color background = Color(0xFF08111F);
  static const Color surface = Color(0xFF101A2B);
  static const Color surface2 = Color(0xFF132238);
  static const Color surface3 = Color(0xFF1A2E49);

  static const Color text = Color(0xFFF6FAFF);
  static const Color muted = Color(0xFF9BAABD);
  static const Color border = Color(0xFF243B59);

  static const Color success = Color(0xFF2BD576);
  static const Color warning = Color(0xFFF4B03E);
  static const Color danger = Color(0xFFFF6B6B);

  static ThemeData light() => darkTheme();

  static ThemeData darkTheme() {
    const scheme = ColorScheme.dark(
      primary: primary,
      secondary: accent,
      surface: surface,
      background: background,
      error: danger,
      onPrimary: Colors.black,
      onSecondary: Colors.black,
      onSurface: text,
      onBackground: text,
      onError: Colors.white,
    );

    final radius = BorderRadius.circular(22);

    return ThemeData(
      useMaterial3: true,
      brightness: Brightness.dark,
      scaffoldBackgroundColor: background,
      colorScheme: scheme,
      dividerColor: border,
      splashColor: Colors.white10,
      highlightColor: Colors.transparent,
      appBarTheme: const AppBarTheme(
        backgroundColor: background,
        foregroundColor: text,
        elevation: 0,
        centerTitle: false,
        iconTheme: IconThemeData(color: text),
        titleTextStyle: TextStyle(
          color: text,
          fontSize: 20,
          fontWeight: FontWeight.w800,
        ),
      ),
      textTheme: const TextTheme(
        headlineLarge: TextStyle(color: text, fontWeight: FontWeight.w800),
        headlineMedium: TextStyle(color: text, fontWeight: FontWeight.w800),
        headlineSmall: TextStyle(color: text, fontWeight: FontWeight.w800, height: 1.15),
        titleLarge: TextStyle(color: text, fontWeight: FontWeight.w800),
        titleMedium: TextStyle(color: text, fontWeight: FontWeight.w700),
        bodyLarge: TextStyle(color: text, height: 1.35),
        bodyMedium: TextStyle(color: text, height: 1.35),
        bodySmall: TextStyle(color: muted, height: 1.35),
        labelLarge: TextStyle(color: text, fontWeight: FontWeight.w700),
      ),
      cardTheme: CardThemeData(
        color: surface,
        elevation: 0,
        margin: EdgeInsets.zero,
        shape: RoundedRectangleBorder(
          borderRadius: radius,
          side: const BorderSide(color: border),
        ),
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: surface,
        hintStyle: const TextStyle(color: muted),
        labelStyle: const TextStyle(color: muted),
        prefixIconColor: muted,
        suffixIconColor: muted,
        contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 15),
        border: OutlineInputBorder(
          borderRadius: radius,
          borderSide: const BorderSide(color: border),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: radius,
          borderSide: const BorderSide(color: border),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: radius,
          borderSide: const BorderSide(color: primary, width: 1.4),
        ),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: primary,
          foregroundColor: Colors.black,
          padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 14),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
          textStyle: const TextStyle(fontWeight: FontWeight.w800),
          elevation: 0,
        ),
      ),
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: OutlinedButton.styleFrom(
          foregroundColor: text,
          side: const BorderSide(color: border),
          padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 14),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
          textStyle: const TextStyle(fontWeight: FontWeight.w700),
        ),
      ),
      textButtonTheme: TextButtonThemeData(
        style: TextButton.styleFrom(
          foregroundColor: primary,
          textStyle: const TextStyle(fontWeight: FontWeight.w700),
        ),
      ),
      chipTheme: ChipThemeData(
        backgroundColor: surface2,
        selectedColor: primary.withOpacity(0.16),
        disabledColor: surface2,
        secondarySelectedColor: primary.withOpacity(0.16),
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
        labelStyle: const TextStyle(color: text, fontWeight: FontWeight.w700),
        secondaryLabelStyle: const TextStyle(color: text, fontWeight: FontWeight.w700),
        brightness: Brightness.dark,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(999),
          side: const BorderSide(color: border),
        ),
      ),
      bottomNavigationBarTheme: const BottomNavigationBarThemeData(
        backgroundColor: Colors.transparent,
        selectedItemColor: primary,
        unselectedItemColor: muted,
        type: BottomNavigationBarType.fixed,
        elevation: 0,
      ),
      snackBarTheme: SnackBarThemeData(
        backgroundColor: surface3,
        contentTextStyle: const TextStyle(color: text, fontWeight: FontWeight.w600),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        behavior: SnackBarBehavior.floating,
      ),
      bottomSheetTheme: const BottomSheetThemeData(
        backgroundColor: surface,
        showDragHandle: true,
      ),
    );
  }
}

class Ui {
  static Widget sectionTitle(BuildContext context, String text, {Widget? trailing}) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(2, 8, 2, 10),
      child: Row(
        children: [
          Expanded(
            child: Text(
              text,
              style: Theme.of(context).textTheme.titleLarge?.copyWith(
                    color: AppTheme.text,
                    fontWeight: FontWeight.w800,
                  ),
            ),
          ),
          if (trailing != null) trailing,
        ],
      ),
    );
  }

  static Widget paddedCard(
    BuildContext context, {
    required Widget child,
    EdgeInsetsGeometry padding = const EdgeInsets.all(14),
  }) {
    return GlassPanel(
      padding: padding,
      radius: 22,
      child: child,
    );
  }

  static Widget badge({
    required String text,
    required Color color,
    IconData? icon,
  }) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 7),
      decoration: BoxDecoration(
        color: color.withOpacity(0.14),
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: color.withOpacity(0.28)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (icon != null) ...[
            Icon(icon, size: 14, color: color),
            const SizedBox(width: 6),
          ],
          Text(
            text,
            style: TextStyle(
              fontWeight: FontWeight.w800,
              color: color,
              fontSize: 12,
            ),
          ),
        ],
      ),
    );
  }

  static Widget animatedEntrance({
    required Widget child,
    int index = 0,
  }) {
    return TweenAnimationBuilder<double>(
      duration: Duration(milliseconds: 360 + index * 90),
      tween: Tween(begin: 0, end: 1),
      curve: Curves.easeOutCubic,
      builder: (context, value, _) {
        return Opacity(
          opacity: value,
          child: Transform.translate(
            offset: Offset(0, 16 * (1 - value)),
            child: child,
          ),
        );
      },
    );
  }
}

class GlassPanel extends StatelessWidget {
  final Widget child;
  final EdgeInsetsGeometry? padding;
  final double radius;

  const GlassPanel({
    super.key,
    required this.child,
    this.padding,
    this.radius = 22,
  });

  @override
  Widget build(BuildContext context) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(radius),
      child: BackdropFilter(
        filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
        child: Container(
          decoration: BoxDecoration(
            gradient: LinearGradient(
              colors: [
                Colors.white.withOpacity(0.07),
                Colors.white.withOpacity(0.04),
              ],
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
            ),
            borderRadius: BorderRadius.circular(radius),
            border: Border.all(color: AppTheme.border.withOpacity(0.9)),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withOpacity(0.20),
                blurRadius: 22,
                offset: const Offset(0, 12),
              ),
            ],
          ),
          child: Padding(
            padding: padding ?? const EdgeInsets.all(14),
            child: child,
          ),
        ),
      ),
    );
  }
}

class AppActionButton extends StatefulWidget {
  final VoidCallback? onTap;
  final IconData icon;
  final String title;
  final String? subtitle;
  final bool primary;
  final bool compact;

  const AppActionButton({
    super.key,
    required this.onTap,
    required this.icon,
    required this.title,
    this.subtitle,
    this.primary = false,
    this.compact = false,
  });

  @override
  State<AppActionButton> createState() => _AppActionButtonState();
}

class _AppActionButtonState extends State<AppActionButton> {
  bool _pressed = false;

  @override
  Widget build(BuildContext context) {
    final radius = widget.compact ? 18.0 : 22.0;
    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: widget.onTap,
        onHighlightChanged: (v) => setState(() => _pressed = v),
        borderRadius: BorderRadius.circular(radius),
        child: AnimatedOpacity(
          duration: const Duration(milliseconds: 180),
          opacity: widget.onTap == null ? 0.55 : 1,
          child: AnimatedScale(
            scale: _pressed ? 0.985 : 1,
            duration: const Duration(milliseconds: 120),
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 220),
              curve: Curves.easeOut,
              padding: EdgeInsets.symmetric(
                horizontal: widget.compact ? 14 : 16,
                vertical: widget.compact ? 14 : 16,
              ),
              decoration: BoxDecoration(
                gradient: widget.primary
                    ? const LinearGradient(
                        colors: [Color(0xFF46E0A1), Color(0xFF26C281)],
                        begin: Alignment.topLeft,
                        end: Alignment.bottomRight,
                      )
                    : LinearGradient(
                        colors: [
                          AppTheme.surface2.withOpacity(0.88),
                          AppTheme.surface3.withOpacity(0.72),
                        ],
                        begin: Alignment.topLeft,
                        end: Alignment.bottomRight,
                      ),
                borderRadius: BorderRadius.circular(radius),
                border: Border.all(
                  color: widget.primary ? AppTheme.primary.withOpacity(0.35) : AppTheme.border,
                ),
                boxShadow: [
                  BoxShadow(
                    color: widget.primary
                        ? AppTheme.primary.withOpacity(_pressed ? 0.14 : 0.24)
                        : Colors.black.withOpacity(0.16),
                    blurRadius: widget.primary ? (_pressed ? 16 : 28) : 16,
                    offset: const Offset(0, 10),
                  ),
                ],
              ),
              child: Row(
                children: [
                  Container(
                    width: widget.compact ? 40 : 46,
                    height: widget.compact ? 40 : 46,
                    decoration: BoxDecoration(
                      color: widget.primary ? Colors.black.withOpacity(0.16) : Colors.white.withOpacity(0.07),
                      borderRadius: BorderRadius.circular(14),
                    ),
                    child: Icon(
                      widget.icon,
                      color: widget.primary ? Colors.black : AppTheme.text,
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Text(
                          widget.title,
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: Theme.of(context).textTheme.titleMedium?.copyWith(
                                color: widget.primary ? Colors.black : AppTheme.text,
                                fontWeight: FontWeight.w800,
                              ),
                        ),
                        if (widget.subtitle != null) ...[
                          const SizedBox(height: 2),
                          Text(
                            widget.subtitle!,
                            maxLines: 2,
                            overflow: TextOverflow.ellipsis,
                            style: Theme.of(context).textTheme.bodySmall?.copyWith(
                                  color: widget.primary ? Colors.black.withOpacity(0.72) : AppTheme.muted,
                                ),
                          ),
                        ],
                      ],
                    ),
                  ),
                  const SizedBox(width: 8),
                  Icon(
                    Icons.arrow_forward_rounded,
                    color: widget.primary ? Colors.black : AppTheme.text,
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class AppStatCard extends StatelessWidget {
  final String value;
  final String label;
  final IconData icon;
  final Color color;

  const AppStatCard({
    super.key,
    required this.value,
    required this.label,
    required this.icon,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Ui.paddedCard(
      context,
      padding: const EdgeInsets.all(14),
      child: Row(
        children: [
          Container(
            width: 42,
            height: 42,
            decoration: BoxDecoration(
              color: color.withOpacity(0.14),
              borderRadius: BorderRadius.circular(14),
            ),
            child: Icon(icon, color: color),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(value, style: Theme.of(context).textTheme.titleLarge),
                const SizedBox(height: 2),
                Text(label, style: Theme.of(context).textTheme.bodySmall),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
