import 'package:arborscan_app/image_selection_service.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image_picker/image_picker.dart';

class _Picker extends ImagePicker {
  final calls = <ImageSource>[];
  XFile? selected;
  Object? failure;

  @override
  Future<XFile?> pickImage({
    required ImageSource source,
    double? maxWidth,
    double? maxHeight,
    int? imageQuality,
    CameraDevice preferredCameraDevice = CameraDevice.rear,
    bool requestFullMetadata = true,
  }) async {
    calls.add(source);
    expect(maxWidth, isNull);
    expect(maxHeight, isNull);
    expect(imageQuality, isNull);
    expect(requestFullMetadata, isTrue);
    if (failure != null) throw failure!;
    return selected;
  }
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  const channel = MethodChannel('arborscan/image_selection');
  final messenger =
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger;

  setUp(() => debugDefaultTargetPlatformOverride = TargetPlatform.android);
  tearDown(() {
    debugDefaultTargetPlatformOverride = null;
    messenger.setMockMethodCallHandler(channel, null);
  });

  for (final unavailable in [false, null]) {
    test('Android gallery $unavailable stops before plugin owns a reply',
        () async {
      final picker = _Picker();
      final calls = <MethodCall>[];
      messenger.setMockMethodCallHandler(channel, (call) async {
        calls.add(call);
        return unavailable;
      });
      await expectLater(
          ImageSelectionService(picker: picker).pick(ImageSource.gallery),
          throwsA(isA<ImageSelectionException>().having((e) => e.message,
              'message', ImageSelectionService.unavailableMessage)));
      expect(calls, hasLength(1));
      expect(calls.single.method, 'canPickGalleryImage');
      expect(calls.single.arguments, isNull);
      expect(picker.calls, isEmpty);
    });
  }

  test('available Android gallery returns original plugin file unchanged',
      () async {
    final original = XFile('synthetic-original-exif.jpg');
    final picker = _Picker()..selected = original;
    messenger.setMockMethodCallHandler(channel, (_) async => true);
    expect(
        await ImageSelectionService(picker: picker).pick(ImageSource.gallery),
        same(original));
    expect(picker.calls, [ImageSource.gallery]);
  });

  test('normal gallery cancellation remains null', () async {
    final picker = _Picker();
    messenger.setMockMethodCallHandler(channel, (_) async => true);
    expect(
        await ImageSelectionService(picker: picker).pick(ImageSource.gallery),
        isNull);
    expect(picker.calls, [ImageSource.gallery]);
  });

  test('native query failure is safe and never starts gallery', () async {
    final picker = _Picker();
    messenger.setMockMethodCallHandler(channel, (_) async {
      throw PlatformException(code: 'synthetic', message: 'private diagnostic');
    });
    await expectLater(
        ImageSelectionService(picker: picker).pick(ImageSource.gallery),
        throwsA(isA<ImageSelectionException>().having((e) => e.message,
            'no internal diagnostic', isNot(contains('private')))));
    expect(picker.calls, isEmpty);
  });

  test('missing native guard fails clearly without unsafe gallery fallback',
      () async {
    final picker = _Picker();
    await expectLater(
        ImageSelectionService(picker: picker).pick(ImageSource.gallery),
        throwsA(isA<ImageSelectionException>().having(
            (e) => e.message, 'update instruction', contains('Обновите'))));
    expect(picker.calls, isEmpty);
  });

  test('Android camera does not use gallery availability', () async {
    final picker = _Picker();
    messenger.setMockMethodCallHandler(channel, (_) async {
      fail('Camera must not query the gallery provider');
    });
    expect(await ImageSelectionService(picker: picker).pick(ImageSource.camera),
        isNull);
    expect(picker.calls, [ImageSource.camera]);
  });

  for (final platform in [
    TargetPlatform.iOS,
    TargetPlatform.windows,
    TargetPlatform.macOS,
    TargetPlatform.linux
  ]) {
    test('$platform gallery bypasses Android guard', () async {
      debugDefaultTargetPlatformOverride = platform;
      final original = XFile('synthetic-cross-platform.jpg');
      final picker = _Picker()..selected = original;
      messenger.setMockMethodCallHandler(channel, (_) async {
        fail('Android query must not disable another platform');
      });
      expect(
          await ImageSelectionService(picker: picker).pick(ImageSource.gallery),
          same(original));
      expect(picker.calls, [ImageSource.gallery]);
    });
  }

  test('plugin failure passes to existing caller error handling', () async {
    final failure = PlatformException(code: 'synthetic-picker-error');
    final picker = _Picker()..failure = failure;
    messenger.setMockMethodCallHandler(channel, (_) async => true);
    await expectLater(
        ImageSelectionService(picker: picker).pick(ImageSource.gallery),
        throwsA(same(failure)));
    expect(picker.calls, [ImageSource.gallery]);
  });
}
