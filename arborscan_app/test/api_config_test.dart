import 'package:flutter_test/flutter_test.dart';
import 'package:arborscan_app/api_config.dart';

void main() {
  test('defaults and dart-define overrides are used', () {
    expect(ApiConfig.baseUrl, const String.fromEnvironment(
      'ARBORSCAN_V3_BASE_URL', defaultValue: 'https://31.57.170.88/api/v3'));
    expect(ApiConfig.v4BaseUrl, const String.fromEnvironment(
      'ARBORSCAN_V4_BASE_URL', defaultValue: 'https://31.57.170.88/api/v4'));
  });

  test('endpoint paths retain nginx prefixes and intentional repeated v4', () {
    for (final base in ['https://31.57.170.88/api/v4', 'https://31.57.170.88/api/v4/']) {
      for (final path in ['/v4/analyze-tree', 'v4/corrections', '/v4/corrections/id']) {
        expect(ApiConfig.endpoint(base, path).toString(),
          'https://31.57.170.88/api/v4/${path.replaceFirst(RegExp(r"^/"), "")}');
      }
    }
    for (final path in ['/auth/login', '/auth/me', '/analyses/my', '/feedback', '/analyze-tree']) {
      expect(ApiConfig.endpoint('https://31.57.170.88/api/v3/', path).path, '/api/v3$path');
    }
    expect(ApiConfig.endpoint('http://127.0.0.1:8001/', '/v4/corrections').toString(),
      'http://127.0.0.1:8001/v4/corrections');
    final uri = ApiConfig.v3('/analyses/my').replace(queryParameters: {'token': 'a+b'});
    expect(uri.queryParameters['token'], 'a+b');
    expect(uri.path, endsWith('/analyses/my'));
  });

  test('relative images retain prefix, absolute URLs are preserved', () {
    const base = 'https://31.57.170.88/api/v3';
    for (final path in ['images/a.png', '/images/a.png', '/api/v3/images/a.png']) {
      expect(ApiConfig.imageUrl(path, base: base), '$base/images/a.png');
    }
    expect(ApiConfig.imageUrl('https://example.com/avatar.png', base: base),
      'https://example.com/avatar.png');
    expect(ApiConfig.imageUrl('//example.com/avatar.png', base: base),
      'https://example.com/avatar.png');
    expect(ApiConfig.imageUrl('/images/a.png?x=1', base: base), '$base/images/a.png?x=1');
    expect(ApiConfig.imageUrl('/images/a.png', base: 'https://31.57.170.88/api/v4'),
      'https://31.57.170.88/api/v4/images/a.png');
  });

  test('tunnel hints appear only for loopback configurations', () {
    expect(ApiConfig.connectionHint('https://31.57.170.88/api/v4'), isNot(contains('adb')));
    expect(ApiConfig.connectionHint('https://31.57.170.88/api/v4'), isNot(contains('SSH')));
    expect(ApiConfig.connectionHint('http://127.0.0.1:8001'), contains('tcp:8001'));
    expect(ApiConfig.connectionHint('http://localhost:8000'), contains('tcp:8000'));
    expect(ApiConfig.connectionHint('http://192.168.1.20:8001'), isNot(contains('adb')));
  });
}
