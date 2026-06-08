import 'package:geolocator/geolocator.dart';

class LocationResult {
  final Position? position;
  final String status;
  final String message;

  const LocationResult({
    required this.position,
    required this.status,
    required this.message,
  });

  bool get ok => position != null;
}

class LocationService {
  /// Надёжное получение координат.
  ///
  /// Почему так:
  /// - getCurrentPosition часто не успевает за 6 секунд в помещении;
  /// - lastKnownPosition часто уже есть и подходит для анализа;
  /// - Android может вернуть null без явной ошибки.
  static Future<LocationResult> getCurrentPositionDetailed() async {
    try {
      final serviceEnabled = await Geolocator.isLocationServiceEnabled();
      if (!serviceEnabled) {
        return const LocationResult(
          position: null,
          status: 'service_disabled',
          message: 'Геолокация выключена в системе.',
        );
      }

      LocationPermission permission = await Geolocator.checkPermission();
      if (permission == LocationPermission.denied) {
        permission = await Geolocator.requestPermission();
      }

      if (permission == LocationPermission.denied) {
        return const LocationResult(
          position: null,
          status: 'permission_denied',
          message: 'Разрешение на геолокацию не выдано.',
        );
      }

      if (permission == LocationPermission.deniedForever) {
        return const LocationResult(
          position: null,
          status: 'permission_denied_forever',
          message: 'Геолокация запрещена навсегда. Разрешите её в настройках приложения.',
        );
      }

      // 1) Сначала пробуем актуальные координаты.
      try {
        final current = await Geolocator.getCurrentPosition(
          desiredAccuracy: LocationAccuracy.best,
          timeLimit: const Duration(seconds: 15),
        );
        return LocationResult(
          position: current,
          status: 'current',
          message: 'Получены текущие GPS-координаты.',
        );
      } catch (_) {
        // 2) Если GPS не успел, берём последнее известное местоположение.
        final last = await Geolocator.getLastKnownPosition();
        if (last != null) {
          return LocationResult(
            position: last,
            status: 'last_known',
            message: 'Использованы последние известные GPS-координаты.',
          );
        }

        return const LocationResult(
          position: null,
          status: 'timeout',
          message: 'Не удалось получить GPS за отведённое время.',
        );
      }
    } catch (e) {
      return LocationResult(
        position: null,
        status: 'error',
        message: 'Ошибка геолокации: $e',
      );
    }
  }

  /// Старый совместимый метод.
  static Future<Position?> tryGetCurrentPosition() async {
    final result = await getCurrentPositionDetailed();
    return result.position;
  }

  static Future<void> openLocationSettings() => Geolocator.openLocationSettings();

  static Future<void> openAppSettings() => Geolocator.openAppSettings();
}
