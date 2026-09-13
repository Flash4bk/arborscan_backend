class ApiConfig {
  /// Legacy v3 backend. Kept for auth/admin/history until those parts are
  /// migrated to Unified Analysis v4.
  static const String baseUrl =
      'https://arborscanbackend-production.up.railway.app';

  /// Unified Analysis v4 backend.
  ///
  /// Alpha default points to localhost so a debug APK can be tested through:
  ///   1) ssh -L 8001:127.0.0.1:8001 arborscan@<VPS>
  ///   2) adb reverse tcp:8001 tcp:8001
  ///
  /// Production builds should override it with an HTTPS endpoint:
  /// flutter build apk --release \
  ///   --dart-define=ARBORSCAN_V4_BASE_URL=https://api.example.com
  static const String v4BaseUrl = String.fromEnvironment(
    'ARBORSCAN_V4_BASE_URL',
    defaultValue: 'http://127.0.0.1:8001',
  );

  /// Google OAuth client used by the existing account flow.
  static const String googleWebClientId =
      '946297507051-33c4msb91harv7rqppf2f31qn10n1m2m.apps.googleusercontent.com';
}
