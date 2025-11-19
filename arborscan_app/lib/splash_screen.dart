import 'package:flutter/material.dart';
import 'package:lottie/lottie.dart';
import 'main.dart';

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen>
    with TickerProviderStateMixin {
  late AnimationController _fadeController;
  late AnimationController _scaleController;

  late Animation<double> _fadeAnimation;
  late Animation<double> _scaleAnimation;

  @override
  void initState() {
    super.initState();

    // Fade-in controller
    _fadeController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 900),
    );

    // Scale animation (slight zoom-in)
    _scaleController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1100),
      lowerBound: 0.92,
      upperBound: 1.0,
    );

    _fadeAnimation = CurvedAnimation(
      parent: _fadeController,
      curve: Curves.easeOut,
    );

    _scaleAnimation = CurvedAnimation(
      parent: _scaleController,
      curve: Curves.easeOut,
    );

    _fadeController.forward();
    _scaleController.forward();

    // После задержки перейти в приложение
    Future.delayed(const Duration(milliseconds: 2500), () {
      Navigator.pushReplacement(
        context,
        PageRouteBuilder(
          transitionDuration: const Duration(milliseconds: 500),
          pageBuilder: (_, __, ___) => const ArborScanPage(),
          transitionsBuilder: (context, animation, secAnim, child) {
            return FadeTransition(
              opacity: animation,
              child: child,
            );
          },
        ),
      );
    });
  }

  @override
  void dispose() {
    _fadeController.dispose();
    _scaleController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: FadeTransition(
        opacity: _fadeAnimation,
        child: Center(
          child: ScaleTransition(
            scale: _scaleAnimation,
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                // Lottie анимация листа
                Lottie.asset(
                  'assets/lottie/Leaf scanning.json',
                  width: 160,
                  repeat: true,
                ),

                const SizedBox(height: 10),

                // Логотип ArborScan
                Image.asset(
                  'assets/icons/arborscan_icon.png',
                  width: 140,
                ),

                const SizedBox(height: 14),

                const Text(
                  "ArborScan",
                  style: TextStyle(
                    fontSize: 28,
                    fontWeight: FontWeight.bold,
                    letterSpacing: 0.5,
                    color: Colors.black87,
                  ),
                ),

                const SizedBox(height: 6),

                Text(
                  "Система оценки состояния деревьев",
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    fontSize: 14,
                    color: Colors.grey.shade700,
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
