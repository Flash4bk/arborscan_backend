import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:ui' as ui;

import 'package:arborscan_app/app_theme.dart';
import 'package:arborscan_app/corrections_service.dart';
import 'package:arborscan_app/model_comparison_page.dart';
import 'package:arborscan_app/model_comparison_service.dart';
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:image/image.dart' as im;
import 'package:shared_preferences/shared_preferences.dart';

const _aSha =
    '2222222222222222222222222222222222222222222222222222222222222222';
const _bSha =
    '4444444444444444444444444444444444444444444444444444444444444444';
const _a = ComparisonModel(2, _aSha), _b = ComparisonModel(4, _bSha);

Uint8List _image() {
  final image = im.Image(width: 40, height: 60);
  im.fill(image, color: im.ColorRgb8(180, 200, 220));
  return Uint8List.fromList(im.encodePng(image));
}

Map<String, dynamic> _catalog() => {
      'schema': 'arborscan.model-comparison.v1',
      'supported': true,
      'default_version': 4,
      'available': [
        {
          'version': 2,
          'weights_sha256': _aSha,
          'source': 'local_saved_weights'
        },
        {
          'version': 4,
          'weights_sha256': _bSha,
          'source': 'local_saved_weights'
        },
      ],
    };

http.Response _json(Object body, [int code = 200]) =>
    http.Response(jsonEncode(body), code,
        headers: {'content-type': 'application/json; charset=utf-8'});

Map<String, dynamic> _result(ComparisonPhoto photo) {
  Map<String, dynamic> model(ComparisonModel model, bool detected) {
    final mask =
        im.Image(width: photo.width, height: photo.height, numChannels: 1);
    if (detected) {
      im.fillRect(mask,
          x1: 12, y1: 8, x2: 28, y2: 52, color: im.ColorRgb8(255, 255, 255));
    }
    return {
      'version': model.version,
      'weights_sha256': model.weightsSha256,
      'tree_class_id': 0,
      'detected': detected,
      'confidence': detected ? .8 : null,
      'mask_png_base64': base64Encode(im.encodePng(mask)),
      'inference_timing': {
        'mean_seconds': .022,
        'median_seconds': .022,
        'samples_seconds': [.02, .022, .024]
      },
      'warnings': <String>[],
    };
  }

  return {
    'schema': 'arborscan.model-comparison.v1',
    'persisted': false,
    'image': {
      'upload_sha256': photo.uploadSha256,
      'width': photo.width,
      'height': photo.height,
      'original_width': photo.originalWidth,
      'original_height': photo.originalHeight,
      'preview_transform': 'resize_only',
      'processing_scale_x': photo.width / photo.originalWidth,
      'processing_scale_y': photo.height / photo.originalHeight
    },
    'settings': {
      'imgsz': 1024,
      'confidence': .25,
      'device': 'cpu',
      'threads': 1,
      'warmup_runs': 1,
      'timing_repeats': 3,
      'processing_max_side': comparisonProcessingMaxSide,
      'max_detections': 16
    },
    'models': [model(_a, true), model(_b, false)],
    'quality_metrics': <String, dynamic>{
      'precision': null,
      'recall': null,
      'map50': null,
      'map50_95': null,
      'mean_iou': null,
      'reason': 'verified_reference_masks_required'
    },
  };
}

ModelComparisonService _service(
        Future<http.Response> Function(http.Request) handler) =>
    ModelComparisonService(clientFactory: () => MockClient(handler));

Future<void> _tap(WidgetTester tester, String label) async {
  final target = find.text(label).hitTestable();
  for (var n = 0; n < 30 && target.evaluate().isEmpty; n++) {
    await tester.drag(find.byType(Scrollable).first, const Offset(0, -220));
    await tester.pumpAndSettle();
  }
  expect(target, findsOneWidget);
  await tester.tap(target);
  await tester.pump();
}

Future<void> _pumpPage(WidgetTester tester, ModelComparisonService service,
    {Future<Uint8List?> Function()? picker,
    GlobalKey? capture,
    double scale = 1}) async {
  Widget app = MaterialApp(
      theme: AppTheme.light(),
      debugShowCheckedModeBanner: false,
      builder: (context, child) => MediaQuery(
          data: MediaQuery.of(context)
              .copyWith(textScaler: TextScaler.linear(scale)),
          child: child!),
      home: ModelComparisonPage(
          service: service, pickImage: picker ?? () async => _image()));
  if (capture != null) app = RepaintBoundary(key: capture, child: app);
  await tester.pumpWidget(app);
  await tester.pumpAndSettle();
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  setUp(() => SharedPreferences.setMockInitialValues({
        'arborscan_auth_token': 'first',
        'arborscan_user_id': 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa',
      }));

  test('multipart binds original bytes, versions and hashes without activation',
      () async {
    final photo = ComparisonPhoto.decode(_image());
    final requests = <http.Request>[];
    final service = _service((request) async {
      requests.add(request);
      expect(request.headers['Authorization'], 'Bearer first');
      expect(
          request.url.path.startsWith('/api/v4/v4/model-comparison'), isTrue);
      if (request.method == 'GET') return _json(_catalog());
      final multipart = latin1.decode(request.bodyBytes);
      for (final value in [
        'baseline_version',
        'candidate_version',
        'baseline_sha256',
        'candidate_sha256',
        _aSha,
        _bSha
      ]) {
        expect(multipart, contains(value));
      }
      expect(multipart, contains(latin1.decode(photo.original)));
      return _json(_result(photo));
    });
    final models = await service.models('first');
    expect(models.available.map((m) => m.version), [2, 4]);
    final result = await service.compare('first', photo, _a, _b);
    final overlay = im.decodePng(result.models.first.overlayPng)!;
    expect(overlay.getPixel(0, 0).a, 0);
    expect(overlay.getPixel(20, 20).a, 112);
    expect(im.decodePng(result.models.last.overlayPng)!.getPixel(20, 20).a, 0);
    expect(requests.map((r) => r.method), ['GET', 'POST']);
    expect(requests.any((r) => r.url.path.contains('activate')), isFalse);
  });

  test('photo preview respects EXIF while retaining original upload and SHA',
      () {
    final bytes = File('test/fixtures/reference_exif6.jpg').readAsBytesSync();
    final source = im.decodeJpg(bytes)!;
    final photo = ComparisonPhoto.decode(bytes);
    final oriented = im.bakeOrientation(source);
    expect(photo.originalWidth, oriented.width);
    expect(photo.originalHeight, oriented.height);
    expect(photo.original, bytes);
    expect(im.decodePng(photo.previewPng)!.width, photo.width);
  });

  test('extreme aspect ratios preserve a one-pixel processing dimension', () {
    for (final size in [const Size(6000, 1), const Size(1, 6000)]) {
      final source =
          im.Image(width: size.width.toInt(), height: size.height.toInt());
      final photo =
          ComparisonPhoto.decode(Uint8List.fromList(im.encodePng(source)));
      expect(photo.width, size.width > 1 ? comparisonProcessingMaxSide : 1);
      expect(photo.height, size.height > 1 ? comparisonProcessingMaxSide : 1);
      final preview = im.decodePng(photo.previewPng)!;
      expect(preview.width, photo.width);
      expect(preview.height, photo.height);
    }
  });

  final smokeResponse =
      Platform.environment['ARBORSCAN_COMPARISON_SMOKE_RESPONSE'];
  final smokePhoto = Platform.environment['ARBORSCAN_COMPARISON_SMOKE_PHOTO'];
  if (smokeResponse != null && smokePhoto != null) {
    test('actual local CPU child response matches the Flutter upload and masks',
        () async {
      final data = jsonDecode(File(smokeResponse).readAsStringSync())
          as Map<String, dynamic>;
      final photo = ComparisonPhoto.decode(File(smokePhoto).readAsBytesSync());
      final selected = [
        for (final row in data['models'])
          ComparisonModel(row['version'], row['weights_sha256'])
      ];
      final service = _service((request) async => _json(data));
      final result =
          await service.compare('first', photo, selected[0], selected[1]);
      expect(result.models.length, 2);
      for (final model in result.models) {
        final overlay = im.decodePng(model.overlayPng)!;
        expect(overlay.width, photo.width);
        expect(overlay.height, photo.height);
        expect(overlay.any((pixel) => pixel.a > 0), model.detected);
      }
    });
  }

  for (final corrupt in [
    'photo',
    'version',
    'weights',
    'transform',
    'mask',
    'quality',
    'persisted'
  ]) {
    test('rejects mismatched comparison $corrupt rather than displaying it',
        () async {
      final photo = ComparisonPhoto.decode(_image());
      final body = _result(photo);
      switch (corrupt) {
        case 'photo':
          body['image']['upload_sha256'] = _aSha;
        case 'version':
          body['models'][0]['version'] = 1;
        case 'weights':
          body['models'][0]['weights_sha256'] = _bSha;
        case 'transform':
          body['image']['preview_transform'] = 'crop';
        case 'mask':
          final mask = im.Image(
              width: photo.width, height: photo.height, numChannels: 1);
          mask.setPixelR(20, 20, 127);
          body['models'][0]['mask_png_base64'] =
              base64Encode(im.encodePng(mask));
        case 'quality':
          body['quality_metrics']['precision'] = .99;
        case 'persisted':
          body['persisted'] = true;
      }
      final service = _service((r) async => _json(body));
      await expectLater(service.compare('first', photo, _a, _b),
          throwsA(isA<CorrectionException>()));
    });
  }

  testWidgets('picker cancellation leaves the selected photo', (tester) async {
    var picks = 0;
    final service = _service((r) async => _json(_catalog()));
    await _pumpPage(tester, service,
        picker: () async => picks++ == 0 ? _image() : null);
    await _tap(tester, 'Выбрать фото для сравнения');
    await tester.pumpAndSettle();
    expect(find.byKey(const ValueKey('comparison-selected-photo')),
        findsOneWidget);
    await _tap(tester, 'Выбрать фото для сравнения');
    await tester.pumpAndSettle();
    expect(find.byKey(const ValueKey('comparison-selected-photo')),
        findsOneWidget);
    await tester.pumpWidget(const SizedBox());
  });

  testWidgets(
      'network failure preserves photo and retry returns two masks and the six-row table',
      (tester) async {
    final photo = ComparisonPhoto.decode(_image());
    var posts = 0;
    final service = _service((r) async {
      if (r.method == 'GET') return _json(_catalog());
      if (++posts == 1) throw http.ClientException('test offline');
      return _json(_result(photo));
    });
    await _pumpPage(tester, service);
    await _tap(tester, 'Выбрать фото для сравнения');
    await tester.pumpAndSettle();
    await _tap(tester, 'Сравнить на этом фото');
    await tester.pumpAndSettle();
    expect(find.byKey(const ValueKey('comparison-selected-photo')),
        findsOneWidget);
    expect(find.textContaining('Нет ответа сервиса сравнения'), findsOneWidget);
    await _tap(tester, 'Сравнить на этом фото');
    await tester.pumpAndSettle();
    expect(posts, 2);
    expect(find.text('Маска получена'), findsOneWidget);
    await _tap(tester, 'Маска не получена');
    expect(find.text('Маска не получена'), findsOneWidget);
    await _tap(tester, 'Precision');
    expect(
        find.byKey(const ValueKey('model-comparison-table')), findsOneWidget);
    expect(find.text('Не оценено'), findsNWidgets(10));
    expect(find.textContaining('нужна проверенная разметка'), findsOneWidget);
    expect(find.text('0,022'), findsNWidgets(2));
    expect(find.text('Активировать'), findsNothing);
    await tester.pumpWidget(const SizedBox());
  });

  testWidgets('duplicate taps send one POST; cancel discards a late result',
      (tester) async {
    final pending = Completer<http.Response>();
    final photo = ComparisonPhoto.decode(_image());
    var posts = 0;
    final service = _service((r) async {
      if (r.method == 'GET') return _json(_catalog());
      posts++;
      return pending.future;
    });
    await _pumpPage(tester, service);
    await _tap(tester, 'Выбрать фото для сравнения');
    await tester.pumpAndSettle();
    await _tap(tester, 'Сравнить на этом фото');
    await tester.tap(find.text('Сравнить на этом фото'));
    await tester.pump();
    expect(posts, 1);
    final cancel = find.text('Отменить запрос').hitTestable();
    for (var i = 0; i < 12 && cancel.evaluate().isEmpty; i++) {
      await tester.drag(find.byType(Scrollable).first, const Offset(0, -100));
      await tester.pump(const Duration(milliseconds: 40));
    }
    expect(cancel, findsOneWidget);
    await tester.tap(cancel);
    await tester.pumpAndSettle();
    pending.complete(_json(_result(photo)));
    await tester.pumpAndSettle();
    expect(find.byKey(const ValueKey('model-comparison-table')), findsNothing);
    expect(find.byKey(const ValueKey('comparison-selected-photo')),
        findsOneWidget);
    expect(find.textContaining('Запрос отменён'), findsOneWidget);
    await tester.pumpWidget(const SizedBox());
  });

  testWidgets('account change clears photo and rejects a late comparison',
      (tester) async {
    final pending = Completer<http.Response>();
    final photo = ComparisonPhoto.decode(_image());
    final service = _service((r) async =>
        r.method == 'GET' ? _json(_catalog()) : await pending.future);
    await _pumpPage(tester, service);
    await _tap(tester, 'Выбрать фото для сравнения');
    await tester.pumpAndSettle();
    await _tap(tester, 'Сравнить на этом фото');
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('arborscan_auth_token', 'second');
    CorrectionsService.authChanges.value++;
    await tester.pumpAndSettle();
    pending.complete(_json(_result(photo)));
    await tester.pumpAndSettle();
    expect(find.byType(Image), findsNothing);
    expect(find.textContaining('Аккаунт изменился'), findsOneWidget);
    expect(find.byKey(const ValueKey('model-comparison-table')), findsNothing);
    await tester.pumpWidget(const SizedBox());
  });

  for (final code in [404, 405, 501]) {
    testWidgets(
        'old API $code is explicit and returning leaves the application usable',
        (tester) async {
      final service = _service((r) async => _json({}, code));
      await tester.pumpWidget(MaterialApp(
          home: Builder(
              builder: (context) => Scaffold(
                  body: TextButton(
                      onPressed: () => Navigator.push(
                          context,
                          MaterialPageRoute(
                              builder: (_) =>
                                  ModelComparisonPage(service: service))),
                      child: const Text('Открыть сравнение'))))));
      await tester.tap(find.text('Открыть сравнение'));
      await tester.pumpAndSettle();
      expect(find.textContaining('пока недоступно на сервере'), findsOneWidget);
      await tester.tap(find.byType(BackButton));
      await tester.pumpAndSettle();
      expect(find.text('Открыть сравнение'), findsOneWidget);
      await tester.pumpWidget(const SizedBox());
    });
  }

  testWidgets(
      'comparison previews and table render with the real theme and large text',
      (tester) async {
    await tester.binding.setSurfaceSize(const Size(411, 891));
    addTearDown(() => tester.binding.setSurfaceSize(null));
    for (final entry in {
      'Atlas': 'assets/fonts/DejaVuSans.ttf',
      'Literata': 'assets/fonts/Literata.ttf',
      'MaterialIcons': 'fonts/MaterialIcons-Regular.otf'
    }.entries) {
      await (FontLoader(entry.key)..addFont(rootBundle.load(entry.value)))
          .load();
    }
    final photo = ComparisonPhoto.decode(_image());
    final capture = GlobalKey();
    final service = _service(
        (r) async => _json(r.method == 'GET' ? _catalog() : _result(photo)));
    await _pumpPage(tester, service, capture: capture);
    await _tap(tester, 'Выбрать фото для сравнения');
    await tester.pumpAndSettle();
    await _tap(tester, 'Сравнить на этом фото');
    await tester.pumpAndSettle();
    final folder = Directory(
        Platform.environment['ARBORSCAN_MODEL_UI_OUTPUT'] ??
            '../output/model-comparison-20261104/ui')
      ..createSync(recursive: true);
    Future<void> snapshot(String name) async {
      // Image decoding runs outside the fake animation clock. Wait for the
      // actual providers so a successful layout cannot hide blank previews.
      await tester.runAsync(() async {
        final context = capture.currentContext!;
        for (final image in tester.widgetList<Image>(find.byType(Image))) {
          await precacheImage(image.image, context);
        }
      });
      await tester.pumpAndSettle();
      final boundary =
          capture.currentContext!.findRenderObject() as RenderRepaintBoundary;
      await tester.runAsync(() async {
        final picture = await boundary.toImage(pixelRatio: 2);
        final png = await picture.toByteData(format: ui.ImageByteFormat.png);
        await File('${folder.path}/$name.png')
            .writeAsBytes(png!.buffer.asUint8List());
        picture.dispose();
      });
    }

    await _tap(tester, 'Маска получена');
    await Scrollable.ensureVisible(tester.element(find.text('Модель v2')),
        alignment: 0);
    await tester.pumpAndSettle();
    await snapshot('comparison-masks');
    final rendered = im.decodePng(
        File('${folder.path}/comparison-masks.png').readAsBytesSync())!;
    final bluePixels =
        rendered.where((p) => p.r == 180 && p.g == 200 && p.b == 220).length;
    final overlayPixels = rendered
        .where((p) =>
            p.r >= 110 &&
            p.r <= 130 &&
            p.g >= 170 &&
            p.g <= 185 &&
            p.b >= 150 &&
            p.b <= 170)
        .length;
    expect(bluePixels, greaterThan(1000));
    expect(overlayPixels, greaterThan(1000));
    await _tap(tester, 'Precision');
    await Scrollable.ensureVisible(
        tester.element(find.text('Сравнение результатов')),
        alignment: 0);
    await snapshot('comparison-table');
    expect(tester.takeException(), isNull);
    await tester.pumpWidget(const SizedBox());
    await tester.binding.setSurfaceSize(const Size(280, 700));
    await _pumpPage(tester, service, scale: 2);
    await _tap(tester, 'Выбрать фото для сравнения');
    await tester.pumpAndSettle();
    await _tap(tester, 'Сравнить на этом фото');
    await tester.pumpAndSettle();
    await _tap(tester, 'Precision');
    expect(tester.takeException(), isNull);
    await tester.pumpWidget(const SizedBox());
  });
}
