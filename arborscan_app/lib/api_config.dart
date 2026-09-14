class ApiConfig {
  /// Вход, профиль, история и администрирование.
  static const String baseUrl = String.fromEnvironment(
    'ARBORSCAN_V3_BASE_URL',
    defaultValue: 'http://127.0.0.1:8000',
  );

  /// Анализ v4 и сохранение исправленных контуров.
  static const String v4BaseUrl = String.fromEnvironment(
    'ARBORSCAN_V4_BASE_URL',
    defaultValue: 'http://127.0.0.1:8001',
  );

  static const String googleWebClientId =
      '946297507051-33c4msb91harv7rqppf2f31qn10n1m2m.apps.googleusercontent.com';
}