import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';

class ImageSelectionException implements Exception {
  final String message;
  const ImageSelectionException(this.message);

  @override
  String toString() => message;
}

/// Checks the Android gallery intent before image_picker owns a pending reply.
/// The locked image_picker_android uses GET_CONTENT/image/* with PhotoPicker
/// disabled. Keep the native query in sync if that selection mode changes.
class ImageSelectionService {
  ImageSelectionService({ImagePicker? picker})
      : _picker = picker ?? ImagePicker();

  final ImagePicker _picker;
  static const _channel = MethodChannel('arborscan/image_selection');
  static const unavailableMessage =
      'На устройстве нет приложения для выбора фото. '
      'Установите галерею или файловый менеджер и попробуйте снова.';

  Future<XFile?> pick(ImageSource source) async {
    if (!kIsWeb &&
        defaultTargetPlatform == TargetPlatform.android &&
        source == ImageSource.gallery) {
      bool available;
      try {
        available =
            await _channel.invokeMethod<bool>('canPickGalleryImage') == true;
      } on PlatformException {
        throw const ImageSelectionException(
            'Не удалось проверить приложение для выбора фото. '
            'Попробуйте снова.');
      } on MissingPluginException {
        throw const ImageSelectionException(
            'Проверка выбора фото недоступна. Обновите ArborScan.');
      }
      if (!available) {
        throw const ImageSelectionException(unavailableMessage);
      }
    }
    // Preserve the plugin's existing options, original bytes and cancel result.
    return _picker.pickImage(source: source);
  }
}
