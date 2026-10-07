import 'dart:async';
import 'dart:io';

import 'package:arborscan_app/analyze_page.dart';
import 'package:arborscan_app/app_theme.dart';
import 'package:arborscan_app/contour_drafts.dart';
import 'package:arborscan_app/contour_editor_state.dart';
import 'package:arborscan_app/reference_measurement.dart';
import 'package:arborscan_app/reference_measurement_page.dart';
import 'package:crypto/crypto.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

const _unavailable = 'На устройстве нет приложения для выбора фото.';

void main() {
  const availability = MethodChannel('arborscan/image_selection');
  const picker = MethodChannel('plugins.flutter.io/image_picker');
  const ar = MethodChannel('arborscan/ar_measure');
  const owner = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa';
  final photo = File('test/fixtures/reference_exif6.jpg').absolute;

  setUp(() {
    SharedPreferences.setMockInitialValues({
      'arborscan_auth_token': 'synthetic-gallery-token',
      'arborscan_user_id': owner,
    });
  });

  tearDown(() {
    final messenger =
        TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger;
    for (final channel in [availability, picker, ar]) {
      messenger.setMockMethodCallHandler(channel, null);
    }
  });

  Future<void> settle(WidgetTester tester) async {
    for (var i = 0; i < 8; i++) {
      await tester.runAsync(
          () async => Future<void>.delayed(const Duration(milliseconds: 25)));
      await tester.pump(const Duration(milliseconds: 100));
    }
  }

  Future<void> tap(WidgetTester tester, String text) async {
    await tester.ensureVisible(find.text(text).first);
    await tester.pump(const Duration(milliseconds: 350));
    await tester.runAsync(() => tester.tap(find.text(text).first));
    await settle(tester);
  }

  testWidgets('missing gallery never starts plugin and retry is available',
      (tester) async {
    var availabilityCalls = 0;
    var pickerCalls = 0;
    final messenger = tester.binding.defaultBinaryMessenger;
    messenger.setMockMethodCallHandler(availability, (call) async {
      expect(call.method, 'canPickGalleryImage');
      availabilityCalls++;
      return false;
    });
    messenger.setMockMethodCallHandler(picker, (_) async {
      pickerCalls++;
      return null;
    });
    await tester.pumpWidget(
        MaterialApp(theme: AppTheme.light(), home: const ArborScanPage()));
    await tap(tester, 'Галерея');
    expect(availabilityCalls, 1);
    expect(pickerCalls, 0);
    expect(find.textContaining(_unavailable), findsOneWidget);
    await tap(tester, 'Галерея');
    expect(availabilityCalls, 2);
    expect(pickerCalls, 0);
    expect(tester.takeException(), isNull);
  });

  testWidgets('pending gallery preflight blocks repeat taps', (tester) async {
    final pending = Completer<bool>();
    var availabilityCalls = 0;
    var pickerCalls = 0;
    final messenger = tester.binding.defaultBinaryMessenger;
    messenger.setMockMethodCallHandler(availability, (_) {
      availabilityCalls++;
      return pending.future;
    });
    messenger.setMockMethodCallHandler(picker, (_) async {
      pickerCalls++;
      return null;
    });
    await tester.pumpWidget(
        MaterialApp(theme: AppTheme.light(), home: const ArborScanPage()));
    await tester.tap(find.text('Галерея'));
    await tester.pump();
    await tester.tap(find.text('Галерея'));
    await tester.pump();
    expect(availabilityCalls, 1);
    expect(pickerCalls, 0);
    pending.complete(false);
    await settle(tester);
    await tester.scrollUntilVisible(find.textContaining(_unavailable), 200,
        scrollable: find.byType(Scrollable).first);
    expect(find.textContaining(_unavailable), findsOneWidget);
    expect(tester.takeException(), isNull);
  });

  testWidgets('unavailable replacement preserves existing photo and AR binding',
      (tester) async {
    tester.view.physicalSize = const Size(360, 1200);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    var available = true;
    var pickerCalls = 0;
    final messenger = tester.binding.defaultBinaryMessenger;
    messenger.setMockMethodCallHandler(availability, (_) async => available);
    messenger.setMockMethodCallHandler(picker, (_) async {
      pickerCalls++;
      return photo.path;
    });
    messenger.setMockMethodCallHandler(
        ar,
        (_) async => {
              'height_m': 12.0,
              'trunk_diameter_m': .3,
              'trunk_measurement_height_m': 1.3,
              'distance_m': 4.0,
              'quality': .8,
              'overall_status': 'usable',
            });
    await tester.pumpWidget(
        MaterialApp(theme: AppTheme.light(), home: const ArborScanPage()));
    await tap(tester, 'Галерея');
    await tap(tester, 'AR');
    await tap(tester, 'Это то же дерево');
    expect(find.text('Связано с этим фото'), findsOneWidget);
    available = false;
    await tap(tester, 'Галерея');
    expect(pickerCalls, 1);
    expect(find.textContaining(_unavailable), findsOneWidget);
    expect(find.text('Фото добавлено'), findsOneWidget);
    expect(find.text('Связано с этим фото'), findsOneWidget);
    final image = tester.widget<Image>(find.byType(Image).first);
    expect((image.image as FileImage).file.path, photo.path);
    expect(tester.takeException(), isNull);
  });

  testWidgets(
      'reference draft retains bytes, coordinates and result on failure',
      (tester) async {
    final folder = (await tester.runAsync(
        () => Directory.systemTemp.createTemp('gallery-reference-fixture-')))!;
    addTearDown(() async {
      await tester.pumpWidget(const SizedBox());
      await tester.runAsync(() => folder.delete(recursive: true));
    });
    final store = ContourDrafts(directory: () async => folder);
    final image = (await tester.runAsync(() => photo.readAsBytes()))!;
    const id = 'synthetic-gallery-draft';
    final data = {
      'version': 1,
      'width': 100,
      'height': 200,
      'photo_sha256': sha256.convert(image).toString(),
      'length_text': '1',
      'unit': 'm',
      'same_plane': true,
      'measurement_version': 2,
      'outline': const ContourEditorState(
          width: 100,
          height: 200,
          closed: true,
          points: [
            Offset(.1, .8),
            Offset(.3, .8),
            Offset(.3, .5),
            Offset(.1, .5)
          ]).toJson(),
      'reference': ReferenceMeasurement.encode(
          [const Offset(.2, .8), const Offset(.2, .6)]),
      'tree': ReferenceMeasurement.encode(
          [const Offset(.5, .9), const Offset(.5, .1)]),
      'crown': ReferenceMeasurement.encode(
          [const Offset(.2, .3), const Offset(.8, .3)]),
    };
    await tester.runAsync(() => store.save(owner, id, data, image));
    var pickerCalls = 0;
    final messenger = tester.binding.defaultBinaryMessenger;
    messenger.setMockMethodCallHandler(availability, (_) async => false);
    messenger.setMockMethodCallHandler(picker, (_) async {
      pickerCalls++;
      return null;
    });
    await tester.runAsync(() => tester.pumpWidget(MaterialApp(
        theme: AppTheme.light(),
        home: ReferenceMeasurementPage(draftId: id, drafts: store))));
    await settle(tester);
    expect(
        tester
            .widget<OutlinedButton>(
                find.widgetWithText(OutlinedButton, 'Выбрать фото'))
            .onPressed,
        isNotNull);
    await tap(tester, 'Выбрать фото');
    expect(pickerCalls, 0);
    await tester.scrollUntilVisible(find.textContaining(_unavailable), -250,
        scrollable: find.byType(Scrollable).first);
    expect(find.textContaining(_unavailable), findsOneWidget);
    await tester.scrollUntilVisible(find.text('Высота: 4.00 м'), 250,
        scrollable: find.byType(Scrollable).first);
    expect(find.text('Высота: 4.00 м'), findsOneWidget);
    final restored = (await tester.runAsync(() => store.load(owner, id)))!;
    expect(restored['image'], image);
    for (final field in data.keys) {
      expect(restored[field], data[field], reason: field);
    }
    expect(tester.takeException(), isNull);
  });
}
