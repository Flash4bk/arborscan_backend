import 'dart:convert';

import 'package:http/http.dart' as http;

class ApiResponseException implements Exception {
  const ApiResponseException(this.message, this.statusCode);

  final String message;
  final int statusCode;

  @override
  String toString() => message;
}

/// Checks the HTTP status before parsing a successful JSON response. Gateway
/// and server failures can be plain text/HTML and must not expose their body.
Map<String, dynamic> decodeApiObject(
  http.Response response, {
  required String service,
}) {
  final status = response.statusCode;
  if (status >= 500) {
    throw ApiResponseException(
      '$service временно недоступен (HTTP $status). '
      'Повторите попытку позже.',
      status,
    );
  }
  if (status < 200 || status >= 300) {
    String? detail;
    try {
      final decoded = jsonDecode(utf8.decode(response.bodyBytes));
      if (decoded is Map && decoded['detail'] is String) {
        detail = decoded['detail'] as String;
      }
    } on FormatException {
      // An invalid error body does not change the HTTP failure into a parser
      // failure or make the response eligible for authentication.
    }
    throw ApiResponseException(
      'HTTP $status: ${detail ?? 'Запрос не выполнен. Повторите попытку.'}',
      status,
    );
  }
  try {
    final decoded = jsonDecode(utf8.decode(response.bodyBytes));
    if (decoded is Map<String, dynamic>) return decoded;
  } on FormatException {
    // Do not attach the raw response to a user-visible FormatException.
  }
  throw const FormatException('Сервер вернул некорректный ответ.');
}
