import 'package:flutter/material.dart';

import 'app_root.dart';
import 'app_theme.dart';
import 'splash_screen_new.dart';
import 'platform_tls_trust.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await initializePlatformTlsTrust();
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
