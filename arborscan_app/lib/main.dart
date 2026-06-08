import 'package:flutter/material.dart';

import 'app_root.dart';
import 'app_theme.dart';
import 'splash_screen_new.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const ArborScanApp());
}

class ArborScanApp extends StatelessWidget {
  const ArborScanApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'ArborScan',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.light(),
      home: const SplashScreen(),
    );
  }
}
