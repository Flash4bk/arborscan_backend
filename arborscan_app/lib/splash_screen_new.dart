import 'dart:async';
import 'package:flutter/material.dart';

import 'app_root.dart';
import 'app_theme.dart';

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller;
  late final Animation<double> _fade;
  late final Animation<double> _scale;
  late final Animation<double> _glow;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1350),
    )..forward();

    _fade = CurvedAnimation(
      parent: _controller,
      curve: Curves.easeOutCubic,
    );

    _scale = Tween<double>(begin: 0.94, end: 1.0).animate(
      CurvedAnimation(parent: _controller, curve: Curves.easeOutBack),
    );

    _glow = Tween<double>(begin: 0.25, end: 1.0).animate(
      CurvedAnimation(parent: _controller, curve: Curves.easeInOut),
    );

    Timer(const Duration(milliseconds: 1650), () {
      if (!mounted) return;
      Navigator.of(context).pushReplacement(
        PageRouteBuilder(
          transitionDuration: const Duration(milliseconds: 420),
          pageBuilder: (_, __, ___) => const AppRoot(),
          transitionsBuilder: (_, animation, __, child) {
            final curved = CurvedAnimation(
              parent: animation,
              curve: Curves.easeOutCubic,
            );
            return FadeTransition(
              opacity: curved,
              child: child,
            );
          },
        ),
      );
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.background,
      body: Stack(
        children: [
          Positioned.fill(
            child: DecoratedBox(
              decoration: const BoxDecoration(
                gradient: LinearGradient(
                  colors: [
                    Color(0xFF08111F),
                    Color(0xFF0B1830),
                    Color(0xFF08111F),
                  ],
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                ),
              ),
            ),
          ),
          Positioned(
            top: -120,
            left: -40,
            child: AnimatedBuilder(
              animation: _glow,
              builder: (context, _) {
                return Container(
                  width: 240,
                  height: 240,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: AppTheme.primary.withOpacity(0.08 * _glow.value),
                    boxShadow: [
                      BoxShadow(
                        color: AppTheme.primary.withOpacity(0.10 * _glow.value),
                        blurRadius: 80,
                        spreadRadius: 14,
                      ),
                    ],
                  ),
                );
              },
            ),
          ),
          Positioned(
            bottom: -120,
            right: -30,
            child: AnimatedBuilder(
              animation: _glow,
              builder: (context, _) {
                return Container(
                  width: 220,
                  height: 220,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: AppTheme.accent.withOpacity(0.07 * _glow.value),
                    boxShadow: [
                      BoxShadow(
                        color: AppTheme.accent.withOpacity(0.09 * _glow.value),
                        blurRadius: 70,
                        spreadRadius: 10,
                      ),
                    ],
                  ),
                );
              },
            ),
          ),
          Center(
            child: FadeTransition(
              opacity: _fade,
              child: ScaleTransition(
                scale: _scale,
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Container(
                      width: 94,
                      height: 94,
                      decoration: BoxDecoration(
                        borderRadius: BorderRadius.circular(28),
                        gradient: const LinearGradient(
                          colors: [Color(0xFF46E0A1), Color(0xFF26C281)],
                          begin: Alignment.topLeft,
                          end: Alignment.bottomRight,
                        ),
                        boxShadow: [
                          BoxShadow(
                            color: AppTheme.primary.withOpacity(0.28),
                            blurRadius: 28,
                            offset: const Offset(0, 14),
                          ),
                        ],
                      ),
                      child: const Icon(
                        Icons.park_rounded,
                        color: Colors.black,
                        size: 46,
                      ),
                    ),
                    const SizedBox(height: 20),
                    Text(
                      'ArborScan',
                      style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                            fontWeight: FontWeight.w800,
                            color: AppTheme.text,
                            letterSpacing: 0.3,
                          ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      'Загрузка модулей анализа',
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                            color: AppTheme.muted,
                          ),
                    ),
                    const SizedBox(height: 22),
                    SizedBox(
                      width: 28,
                      height: 28,
                      child: CircularProgressIndicator(
                        strokeWidth: 2.6,
                        valueColor: const AlwaysStoppedAnimation<Color>(AppTheme.primary),
                        backgroundColor: Colors.white10,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
