import 'dart:async';

import 'package:arborscan_app/app_theme.dart';
import 'package:arborscan_app/corrections_service.dart';
import 'package:arborscan_app/map_page.dart';
import 'package:arborscan_app/survey_map_repository.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:google_maps_flutter_platform_interface/google_maps_flutter_platform_interface.dart';
import 'package:shared_preferences/shared_preferences.dart';

const _unavailable =
    'Подложка карты недоступна. Записи можно открыть в списке.';

class _PositionedRepository extends SurveyMapRepository {
  @override
  Future<SurveyMapResult> load(
          {void Function(SurveyMapResult)? onLocal}) async =>
      SurveyMapResult([
        SurveyMapEntry.version({
          'version_id': 'demo-version',
          'draft_id': 'demo-local',
          'snapshot': {
            'report': {'species': 'DEMO lifecycle', 'height_m': 12},
            'environment': {
              'gps': {
                'value': {'lat': 50.3, 'lon': 10.3},
                'source': 'manual',
                'position_kind': 'tree'
              }
            }
          }
        }, 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa', server: false)
      ], []);

  @override
  Future<Map<String, dynamic>> detail(SurveyMapEntry entry) async =>
      {'record': entry.metadata, 'snapshot': entry.metadata['snapshot']};
}

// Use the real GoogleMap widget/controller and event streams. Only the native
// view/channel is fake, so controller use after widget disposal still throws.
class _NativeMaps extends MethodChannelGoogleMapsFlutter {
  final initialized = <int>[];
  final animations = <int>[];
  final disposals = <int, int>{};
  Completer<void>? pendingAnimation;
  bool failAnimation = false;

  @override
  Future<void> init(int mapId) async {
    initialized.add(mapId);
    final mapChannel = ensureChannelInitialized(mapId);
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(mapChannel, (call) async {
      if (call.method == 'camera#animate') {
        if (disposals.containsKey(mapId)) {
          throw StateError('Native map $mapId has been disposed');
        }
        animations.add(mapId);
        if (failAnimation) {
          throw PlatformException(code: 'map_unavailable');
        }
        await pendingAnimation?.future;
      }
      return null;
    });
    await super.init(mapId);
  }

  @override
  void dispose({required int mapId}) {
    disposals.update(mapId, (count) => count + 1, ifAbsent: () => 1);
    super.dispose(mapId: mapId);
  }

  @override
  Future<void> updateGroundOverlays(GroundOverlayUpdates updates,
      {required int mapId}) async {}

  @override
  Widget buildViewWithConfiguration(
      int creationId, PlatformViewCreatedCallback onPlatformViewCreated,
      {required MapWidgetConfiguration widgetConfiguration,
      MapConfiguration mapConfiguration = const MapConfiguration(),
      MapObjects mapObjects = const MapObjects()}) {
    return _NativeView(
        key: ValueKey(creationId),
        onCreated: () => onPlatformViewCreated(creationId));
  }

  void clearHandlers() {
    for (final id in initialized) {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel(id), null);
    }
  }
}

class _NativeView extends StatefulWidget {
  final VoidCallback onCreated;
  const _NativeView({super.key, required this.onCreated});
  @override
  State<_NativeView> createState() => _NativeViewState();
}

class _NativeViewState extends State<_NativeView> {
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted) widget.onCreated();
    });
  }

  @override
  Widget build(BuildContext context) => const SizedBox.expand();
}

void main() {
  late GoogleMapsFlutterPlatform previous;
  late _NativeMaps native;

  setUp(() {
    SharedPreferences.setMockInitialValues({
      'arborscan_auth_token': 'synthetic-lifecycle',
      'arborscan_user_id': 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa'
    });
    previous = GoogleMapsFlutterPlatform.instance;
    native = _NativeMaps();
    GoogleMapsFlutterPlatform.instance = native;
  });

  tearDown(() {
    native.clearHandlers();
    GoogleMapsFlutterPlatform.instance = previous;
  });

  Future<void> openMap(WidgetTester tester) async {
    await tester.pumpWidget(MaterialApp(
        theme: AppTheme.light(),
        home: MapPage(
            initialFocus: const LatLngFocus(50.3, 10.3),
            repository: _PositionedRepository())));
    await tester.pumpAndSettle();
    expect(native.initialized, hasLength(1));
    expect(native.animations, [native.initialized.single]);
    expect(tester.takeException(), isNull);
  }

  testWidgets(
      'Map -> list -> card avoids disposed controller; recreation uses current map and disposes once',
      (tester) async {
    await openMap(tester);
    final first = native.initialized.single;
    await tester.tap(find.byTooltip('Показать список'));
    await tester.pumpAndSettle();
    expect(native.disposals[first], 1);
    await tester.tap(find.text('DEMO lifecycle'));
    await tester.pumpAndSettle();
    expect(find.text('Открыть эту версию'), findsOneWidget);
    expect(find.text(_unavailable), findsNothing);
    expect(native.animations, [first]);
    Navigator.of(tester.element(find.byType(MapPage))).pop();
    await tester.pumpAndSettle();

    await tester.tap(find.byTooltip('Показать карту'));
    await tester.pumpAndSettle();
    final second = native.initialized.last;
    expect(second, isNot(first));
    await tester.tap(find.byTooltip('Наклонный обзор 3D'));
    await tester.pumpAndSettle();
    expect(native.animations.last, second);
    expect(find.text(_unavailable), findsNothing);

    // An empty positioned filter also removes the native map.
    await tester.tap(find.text('Аккаунт'));
    await tester.pumpAndSettle();
    expect(native.disposals[second], 1);
    await tester.tap(find.text('На устройстве'));
    await tester.pumpAndSettle();
    final third = native.initialized.last;
    expect(third, isNot(second));
    await tester.tap(find.byTooltip('Плоский обзор 2D'));
    await tester.pumpAndSettle();
    expect(native.animations.last, third);

    // Session reset must not reuse the previous session's native controller.
    CorrectionsService.authChanges.value++;
    await tester.pumpAndSettle();
    expect(native.disposals[third], 1);
    final fourth = native.initialized.last;
    expect(fourth, isNot(third));
    await tester.tap(find.byTooltip('Наклонный обзор 3D'));
    await tester.pumpAndSettle();
    expect(native.animations.last, fourth);

    await tester.pumpWidget(const SizedBox());
    await tester.pumpAndSettle();
    expect(native.initialized, hasLength(4));
    expect(native.disposals.values, everyElement(1));
    expect(tester.takeException(), isNull);
  });

  testWidgets('Late camera failure after map removal does not poison list',
      (tester) async {
    await openMap(tester);
    native.pendingAnimation = Completer<void>();
    await tester.tap(find.byTooltip('Наклонный обзор 3D'));
    await tester.pump();
    await tester.tap(find.byTooltip('Показать список'));
    await tester.pumpAndSettle();
    native.pendingAnimation!
        .completeError(PlatformException(code: 'disposed_map'));
    await tester.pumpAndSettle();
    expect(find.text(_unavailable), findsNothing);
    await tester.tap(find.text('DEMO lifecycle'));
    await tester.pumpAndSettle();
    expect(find.text('Открыть эту версию'), findsOneWidget);
    expect(find.text(_unavailable), findsNothing);
    expect(tester.takeException(), isNull);
  });

  testWidgets('Current native camera failure remains an explicit map error',
      (tester) async {
    await openMap(tester);
    native.failAnimation = true;
    await tester.tap(find.byTooltip('Наклонный обзор 3D'));
    await tester.pumpAndSettle();
    expect(find.text(_unavailable), findsOneWidget);
    expect(tester.takeException(), isNull);
  });
}
