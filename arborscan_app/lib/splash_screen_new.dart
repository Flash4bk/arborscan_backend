import 'dart:async';
import 'package:flutter/material.dart';

import 'app_root.dart';
import 'app_theme.dart';

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> with SingleTickerProviderStateMixin {
  late final AnimationController _controller;
  late final Animation<double> _fade;
  late final Animation<double> _scale;
  late final Animation<double> _glow;

  @override
  void initState() {
    super.initState();
    // Делаем анимацию чуть медленнее и плавнее
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 2000), 
    )..forward();

    _fade = CurvedAnimation(parent: _controller, curve: Curves.easeIn);

    _scale = Tween<double>(begin: 0.90, end: 1.0).animate(
      CurvedAnimation(parent: _controller, curve: Curves.easeOutCubic),
    );

    _glow = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _controller, curve: Curves.easeInOutSine),
    );

    Timer(const Duration(milliseconds: 2800), () {
      if (!mounted) return;
      Navigator.of(context).pushReplacement(
        PageRouteBuilder(
          transitionDuration: const Duration(milliseconds: 800), // Плавное исчезновение
          pageBuilder: (_, __, ___) => const AppRoot(),
          transitionsBuilder: (_, animation, __, child) {
            return FadeTransition(
              opacity: CurvedAnimation(parent: animation, curve: Curves.easeOut),
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
          Positioned(
            top: -120,
            left: -40,
            child: AnimatedBuilder(
              animation: _glow,
              builder: (context, _) {
                return Container(
                  width: 240, height: 240,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: AppTheme.primary.withOpacity(0.08 * _glow.value),
                    boxShadow: [
                      BoxShadow(color: AppTheme.primary.withOpacity(0.10 * _glow.value), blurRadius: 80, spreadRadius: 14),
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
                  width: 220, height: 220,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: AppTheme.primary2.withOpacity(0.07 * _glow.value),
                    boxShadow: [
                      BoxShadow(color: AppTheme.primary2.withOpacity(0.09 * _glow.value), blurRadius: 70, spreadRadius: 10),
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
                      width: 94, height: 94,
                      decoration: BoxDecoration(
                        borderRadius: BorderRadius.circular(28),
                        color: AppTheme.surface3,
                        border: Border.all(color: AppTheme.primary.withOpacity(0.5)),
                        boxShadow: [
                          BoxShadow(color: AppTheme.primary.withOpacity(0.3), blurRadius: 28, offset: const Offset(0, 14)),
                        ],
                      ),
                      child: const Icon(Icons.park_rounded, color: AppTheme.primary, size: 46),
                    ),
                    const SizedBox(height: 24),
                    Text(
                      'ARBORSCAN',
                      style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                            fontWeight: FontWeight.w900,
                            color: AppTheme.text,
                            letterSpacing: 3.0,
                          ),
                    ),
                    const SizedBox(height: 8),
                    Text('AI & AR Tree Analytics', style: TextStyle(color: AppTheme.primary.withOpacity(0.7), fontWeight: FontWeight.w700, letterSpacing: 1.5)),
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