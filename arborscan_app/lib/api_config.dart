class ApiConfig {
  /// Вход, профиль, история и администрирование.
  static const String baseUrl = String.fromEnvironment(
    'ARBORSCAN_V3_BASE_URL',
    defaultValue: 'https://31.57.170.88/api/v3',
  );

  /// Анализ v4 и сохранение исправленных контуров.
  static const String v4BaseUrl = String.fromEnvironment(
    'ARBORSCAN_V4_BASE_URL',
    defaultValue: 'https://31.57.170.88/api/v4',
  );

  /// Append backend paths without Uri.resolve dropping the proxy prefix.
  /// /api/v4/v4/... is intentional: nginx strips only /api/v4/.
  static Uri endpoint(String base, String path) => Uri.parse(
      '${base.replaceFirst(RegExp(r'/+$'), '')}/${path.replaceFirst(RegExp(r'^/+'), '')}');

  static Uri v3(String path) => endpoint(baseUrl, path);
  static Uri v4(String path) => endpoint(v4BaseUrl, path);

  /// Backend-relative images belong under the same external API prefix.
  /// Absolute URLs (e.g. Google avatars) and already-prefixed paths survive.
  static String imageUrl(String value, {String base = baseUrl}) {
    final uri = Uri.parse(value);
    if (uri.hasScheme) return value;
    final baseUri = Uri.parse(base);
    if (uri.hasAuthority) return uri.replace(scheme: baseUri.scheme).toString();
    final prefix = baseUri.path.replaceFirst(RegExp(r'/+$'), '');
    if (prefix.isNotEmpty &&
        (uri.path == prefix || uri.path.startsWith('$prefix/'))) {
      return baseUri.resolveUri(uri).toString();
    }
    return endpoint(base, value).toString();
  }

  static bool isLoopback(String url) {
    final host = Uri.parse(url).host;
    return host == 'localhost' || host == '::1' || host == '[::1]' ||
        host.startsWith('127.');
  }

  static String connectionHint(String url) {
    if (isLoopback(url)) {
      final port = Uri.parse(url).port;
      return 'Локальный адрес: для USB используйте adb reverse tcp:$port tcp:$port. '
          'Если сервер доступен на ПК через SSH-туннель, оставьте его открытым.';
    }
    if (Uri.parse(url).scheme == 'https') {
      return 'Прямое подключение по HTTPS. Достаточно доступа к интернету.';
    }
    return 'Проверьте доступность настроенного сервера из сети телефона.';
  }

  static const String googleWebClientId =
      '946297507051-33c4msb91harv7rqppf2f31qn10n1m2m.apps.googleusercontent.com';
}
