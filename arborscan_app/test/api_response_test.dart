import 'package:arborscan_app/api_response.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;

void main() {
  for (final body in [
    'Internal Server Error',
    '<html>private gateway diagnostic</html>',
    '{"detail":"Contour workflow migration is not ready"}',
  ]) {
    for (final status in [500, 503]) {
      test('HTTP $status is a server error regardless of body $body', () {
        expect(
          () => decodeApiObject(http.Response(body, status),
              service: 'Сервис анализа'),
          throwsA(isA<ApiResponseException>()
              .having((e) => e.statusCode, 'status', status)
              .having((e) => e.message, 'message',
                  'Сервис анализа временно недоступен (HTTP $status). Повторите попытку позже.')),
        );
      });
    }
  }

  for (final body in ['private malformed response', '[]', '']) {
    test('invalid HTTP200 $body cannot become a successful object', () {
      expect(
        () => decodeApiObject(http.Response(body, 200),
            service: 'Сервис авторизации'),
        throwsA(
            isA<FormatException>().having((e) => e.source, 'source', isNull)),
      );
    });
  }

  test('successful JSON object and 401/409 retain their meaning', () {
    expect(
        decodeApiObject(http.Response('{"ok":true}', 200),
            service: 'Сервис авторизации'),
        {'ok': true});
    for (final status in [401, 409]) {
      expect(
        () => decodeApiObject(
            http.Response('{"detail":"synthetic public error"}', status),
            service: 'Сервис авторизации'),
        throwsA(isA<ApiResponseException>()
            .having((e) => e.statusCode, 'status', status)),
      );
    }
  });
}
