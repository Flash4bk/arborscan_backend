import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:arborscan_app/reference_measurement.dart';
import 'package:arborscan_app/reference_measurement_page.dart';
import 'package:arborscan_app/contour_drafts.dart';
import 'package:arborscan_app/image_line_page.dart';
import 'package:arborscan_app/stick_page.dart';
import 'package:arborscan_app/history_tab_page.dart';

const owner = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa';
const other = 'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb';
final photo = base64Decode(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNImXYCAAMkAcMgVWSjAAAAAElFTkSuQmCC');
ReferenceMeasurement measure(double length) => ReferenceMeasurement(
    width: 1000,
    height: 2000,
    lengthM: length,
    samePlane: true,
    reference: const [Offset(.1, .8), Offset(.1, .7)],
    tree: const [Offset(.5, .9), Offset(.5, .1)],
    crown: const [Offset(.2, .4), Offset(.8, .4)],
    outline: const [Offset(0, .9), Offset(.2, .9), Offset(.1, .6)]);

void main() {
  test('optional geometry rotates with reference and old versions remain old', () {
    final base = measure(2).toJson();
    final r = ReferenceMeasurement.fromJson({...base, 'version':2, 'method':'known_object_segment_v2',
      'crown_height':[{'x':.5,'y':.5},{'x':.5,'y':.1}],
      'trunk':[{'x':.45,'y':.7},{'x':.55,'y':.7}],
      'trunk_axis':[{'x':.5,'y':.8},{'x':.5,'y':.6}]});
    expect(r.crownHeightM, closeTo(8,1e-8));
    expect(r.trunkM, closeTo(1,1e-8));
    expect(r.leanDeg, 0);
    expect(() => ReferenceMeasurement.fromJson({...r.toJson(), 'trunk':[{'x':.45,'y':.95},{'x':.55,'y':.95}]}),throwsFormatException);
    expect(r.geometry['dbh']['value'],isNull);
    expect(r.geometry['crown_porosity']['value'],isNull);
    expect(ReferenceMeasurement.fromJson(r.toJson()).geometry, r.geometry);
    expect(ReferenceMeasurement.fromJson(base).report.containsKey('geometry'),false);
    final leaning = ReferenceMeasurement.fromJson({...r.toJson(), 'trunk_axis':[{'x':.2,'y':.8},{'x':.8,'y':.5}]});
    expect(leaning.leanDeg,closeTo(45,1e-8));
    expect(() => ReferenceMeasurement.fromJson({...r.toJson(), 'trunk_axis':[{'x':.5,'y':.5},{'x':.5,'y':.5}]}),throwsFormatException);
  });

  TestWidgetsFlutterBinding.ensureInitialized();
  testWidgets('reference journal is accessible directly from History toolbar', (tester) async {
    SharedPreferences.setMockInitialValues({});
    await tester.pumpWidget(const MaterialApp(home: HistoryTabPage()));
    await tester.pumpAndSettle();
    await tester.tap(find.byTooltip('Измерения по эталону'));
    await tester.pumpAndSettle();
    expect(find.byType(ReferenceHistoryPage), findsOneWidget);
  });
  testWidgets('legacy ruler returns scale per original pixel, preserving its controls', (tester) async {
    double? scale;
    await tester.pumpWidget(MaterialApp(home: Builder(builder: (context) => TextButton(
      onPressed: () async {scale = await Navigator.push<double>(context, MaterialPageRoute(builder: (_) =>
        StickPage(originalImageBase64: base64Encode(photo), currentScalePxToM: 0)));}, child: const Text('Open')))));
    await tester.tap(find.text('Open'));await tester.pump();
    await tester.runAsync(() async => Future<void>.delayed(const Duration(milliseconds:100)));
    await tester.pumpAndSettle();
    final rect = tester.getRect(find.byKey(const ValueKey('legacy-reference-canvas')));
    await tester.tapAt(rect.topLeft + Offset(rect.width*.25, rect.height*.5));await tester.pump();
    await tester.tapAt(rect.topLeft + Offset(rect.width*.75, rect.height*.5));await tester.pump();
    await tester.tap(find.byTooltip('Применить'));await tester.pumpAndSettle();
    await tester.enterText(find.byType(TextField), '2');
    await tester.tap(find.text('ГОТОВО'));await tester.pumpAndSettle();
    expect(scale, closeTo(4, 1e-8)); // 2 metres / half of the original 1px image.
  });
  test(
      'anisotropic normalized coordinates use original pixels, units and rescaling',
      () {
    final r = measure(ReferenceMeasurement.parseLength('200', 'cm'));
    expect(r.heightM, closeTo(16, 1e-10));
    expect(r.crownM, closeTo(6, 1e-10));
    expect(measure(4).heightM, closeTo(32, 1e-10));
    expect(
        ReferenceMeasurement.fromJson(jsonDecode(jsonEncode(r.toJson())))
            .toJson(),
        r.toJson());
    for (final input in ['0', '-1', 'NaN', 'Infinity', 'bad', '']) {
      expect(() => ReferenceMeasurement.parseLength(input, 'm'),
          throwsFormatException);
    }
    expect(ReferenceMeasurement.parseLength('1,5', 'm'), 1.5);
    expect(
        () => ReferenceMeasurement.fromJson({
              ...r.toJson(),
              'reference': [
                {'x': .1, 'y': .5},
                {'x': .1, 'y': .5}
              ]
            }),
        throwsFormatException);
    expect(
        () =>
            ReferenceMeasurement.fromJson({...r.toJson(), 'same_plane': false}),
        throwsFormatException);
  });
  test(
      'height is a vertical projection, crown width is perpendicular to reference',
      () {
    final r = measure(2);
    final leaning = ReferenceMeasurement.fromJson({
      ...r.toJson(),
      'tree': [
        {'x': .2, 'y': .9},
        {'x': .8, 'y': .1}
      ]
    });
    expect(leaning.heightM, closeTo(16, 1e-10));
    // A rolled photograph rotates all vectors together without changing metres.
    List<Offset> rotate(List<Offset> points) =>
        points.map((p) => Offset(1 - p.dy, p.dx)).toList();
    final rolled = ReferenceMeasurement(
        width: r.height,
        height: r.width,
        lengthM: r.lengthM,
        reference: rotate(r.reference),
        tree: rotate(r.tree),
        crown: rotate(r.crown),
        outline: rotate(r.outline),
        samePlane: true);
    expect(rolled.heightM, closeTo(r.heightM, 1e-10));
    expect(rolled.crownM, closeTo(r.crownM, 1e-10));
  });
  testWidgets('line endpoints remain normalized across viewport changes',
      (tester) async {
    List<Offset>? returned;
    await tester.pumpWidget(MaterialApp(
        home: Builder(
            builder: (context) => TextButton(
                onPressed: () async {
                  returned = await Navigator.push<List<Offset>>(
                      context,
                      MaterialPageRoute(
                          builder: (_) => ImageLinePage(
                                  image: photo,
                                  width: 1000,
                                  height: 2000,
                                  title: 'test',
                                  initial: const [
                                    Offset(.1, .8),
                                    Offset(.1, .7)
                                  ])));
                },
                child: const Text('Open')))));
    await tester.tap(find.text('Open'));
    await tester.pumpAndSettle();
    final canvas = find.byKey(const ValueKey('image-line-canvas'));
    final rect = tester.getRect(canvas);
    expect(rect.height / rect.width, closeTo(2, 1e-10));
    await tester
        .tapAt(rect.topLeft + Offset(rect.width * .25, rect.height * .75));
    await tester.pump();
    // Re-layout must not re-interpret stored points as display pixels.
    await tester.binding.setSurfaceSize(const Size(1000, 900));
    await tester.pumpAndSettle();
    final controller = tester
        .widget<InteractiveViewer>(find.byType(InteractiveViewer))
        .transformationController!;
    controller.value = Matrix4.diagonal3Values(2, 2, 1);
    await tester.pump();
    final box = tester.renderObject<RenderBox>(canvas);
    await tester.tapAt(
        box.localToGlobal(Offset(box.size.width * .2, box.size.height * .2)));
    await tester.pump();
    await tester.tap(find.text('Применить отрезок'));
    await tester.pumpAndSettle();
    expect(returned![0].dx, closeTo(.2, 1e-8));
    expect(returned![0].dy, closeTo(.2, 1e-8));
    expect(returned![1], const Offset(.1, .7));
    expect(tester.takeException(), isNull);
    await tester.binding.setSurfaceSize(null);
  });
  testWidgets(
      'reference journal restores report after screen recreation and isolates owner',
      (tester) async {
    SharedPreferences.setMockInitialValues(
        {'arborscan_auth_token': 'a', 'arborscan_user_id': owner});
    final folder = (await tester
        .runAsync(() => Directory.systemTemp.createTemp('reference-test-')))!;
    final store = ContourDrafts(directory: () async => folder);
    final r = ReferenceMeasurement.fromJson({...measure(2).toJson(),'width':100,'height':200});
    final original = (await tester.runAsync(() => File('test/fixtures/reference_exif6.jpg').readAsBytes()))!;
    final data = {
      'version': 1,
      'width': r.width,
      'height': r.height,
      'length_text': '200',
      'unit': 'cm',
      'same_plane': true,
      'outline': {
        'version': 1,
        'coordinates': 'normalized_oriented_image',
        'width': r.width,
        'height': r.height,
        'closed': true,
        'points': ReferenceMeasurement.encode(r.outline)
      },
      'reference': ReferenceMeasurement.encode(r.reference),
      'tree': ReferenceMeasurement.encode(r.tree),
      'crown': ReferenceMeasurement.encode(r.crown)
    };
    await tester.runAsync(() async => store.save(owner, 'sample', data, original));
    Future<void> open() async {
      await tester.pumpWidget(MaterialApp(
          home: ReferenceMeasurementPage(
              draftId: 'sample',
              drafts: ContourDrafts(directory: () async => folder))));
      for (var i = 0; i < 12; i++) {
        await tester.runAsync(
            () async => Future<void>.delayed(const Duration(milliseconds: 20)));
        await tester.pump();
      }
      await tester.pumpAndSettle();
    }

    await open();
    await tester.scrollUntilVisible(find.text('Высота: 16.00 м'), 300,
        scrollable: find.byType(Scrollable).first);
    await tester.scrollUntilVisible(find.text('Ширина кроны: 6.00 м'), 100,
        scrollable: find.byType(Scrollable).first);
    expect(find.text('Ширина кроны: 6.00 м'), findsOneWidget);
    await tester.pumpWidget(const SizedBox());
    await open();
    await tester.scrollUntilVisible(find.text('Высота: 16.00 м'), 300,
        scrollable: find.byType(Scrollable).first);
    expect(find.text('Высота: 16.00 м'), findsOneWidget);
    await tester.runAsync(() async {
      expect(await store.load(other, 'sample'), isNull);
    });
    await tester.pumpWidget(const SizedBox());
    await tester.runAsync(() => store.save(owner,'sample',{...data,'width':999},original));
    await open();
    expect(find.textContaining('Размеры фото не соответствуют разметке'),findsOneWidget);
    expect(find.text('Сохранить отчёт на устройстве'),findsNothing);
    await tester.pumpWidget(const SizedBox());
    await tester.runAsync(() async => folder.delete(recursive: true));
  });
}
