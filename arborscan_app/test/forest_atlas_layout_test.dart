import 'dart:convert';
import 'dart:io';
import 'package:arborscan_app/analyze_page.dart';
import 'package:arborscan_app/app_theme.dart';
import 'package:arborscan_app/contour_editor_state.dart';
import 'package:arborscan_app/image_line_page.dart';
import 'package:arborscan_app/mask_drawing_page.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as im;
import 'package:shared_preferences/shared_preferences.dart';

void main() {
  setUp(() => SharedPreferences.setMockInitialValues({}));
  Future<void> viewport(WidgetTester t, {double width = 360, double height = 800}) async {
    t.view.physicalSize = Size(width, height);
    t.view.devicePixelRatio = 1;
    addTearDown(t.view.resetPhysicalSize);
    addTearDown(t.view.resetDevicePixelRatio);
  }
  for (final dark in [false, true]) {
    for (final scale in [1.0, 2.0]) {
      testWidgets('pinned analyze stays visible with photo, scrolling and keyboard $dark/$scale', (t) async {
        await viewport(t);
        const picker = MethodChannel('plugins.flutter.io/image_picker');
        const availability = MethodChannel('arborscan/image_selection');
        final messenger = t.binding.defaultBinaryMessenger;
        messenger.setMockMethodCallHandler(picker, (_) async => File('test/fixtures/reference_exif6.jpg').absolute.path);
        messenger.setMockMethodCallHandler(availability, (_) async => true);
        addTearDown(() {messenger.setMockMethodCallHandler(picker, null);messenger.setMockMethodCallHandler(availability, null);});
        await t.pumpWidget(MaterialApp(theme: dark ? AppTheme.dark() : AppTheme.light(),
          builder: (c, child) => MediaQuery(data: MediaQuery.of(c).copyWith(textScaler: TextScaler.linear(scale)), child: child!),
          home: const Scaffold(body: ArborScanPage(), bottomNavigationBar: SizedBox(height:80))));
        await t.pumpAndSettle();
        final action = find.byKey(const ValueKey('analyze-photo'));
        expect(t.getRect(action).bottom, lessThanOrEqualTo(720));
        expect(t.widget<FilledButton>(action).onPressed, isNull);
        final gallery = find.text('Галерея');
        await t.ensureVisible(gallery);
        await t.pumpAndSettle();
        await t.runAsync(() async {await t.tap(gallery);await Future<void>.delayed(const Duration(milliseconds:200));});
        await t.pumpAndSettle();
        expect(t.widget<FilledButton>(action).onPressed, isNotNull);
        final before=t.getRect(action);
        await t.drag(find.byType(ListView).first,const Offset(0,-400));await t.pumpAndSettle();
        expect(t.getRect(action),before);
        t.view.viewInsets=const FakeViewPadding(bottom:280);await t.pumpAndSettle();
        expect(t.getRect(action).bottom,lessThanOrEqualTo(520));
        expect(t.takeException(),isNull);
        t.view.resetViewInsets();
      });
    }
  }

  final photo=Uint8List.fromList(im.encodePng(im.Image(width:100,height:200)));
  testWidgets('line taps retain normalized original coordinates after zoom and 200% layout', (t) async {
    await viewport(t);
    List<Offset>? result;
    await t.pumpWidget(MaterialApp(theme:AppTheme.dark(),
      builder:(c,child)=>MediaQuery(data:MediaQuery.of(c).copyWith(textScaler:const TextScaler.linear(2)),child:child!),
      home:Builder(builder:(c)=>Scaffold(body:TextButton(onPressed:() async {
        result=await Navigator.push<List<Offset>>(c,MaterialPageRoute(builder:(_)=>ImageLinePage(image:photo,width:100,height:200,title:'Основание и вершина дерева')));
      },child:const Text('Открыть'))))));
    await t.tap(find.text('Открыть'));await t.pumpAndSettle();
    final viewer=t.widget<InteractiveViewer>(find.byType(InteractiveViewer));
    viewer.transformationController!.value=Matrix4.diagonal3Values(1.5,1.5,1)..setTranslationRaw(-30,-40,0);
    await t.pump();
    final canvas=find.byKey(const ValueKey('image-line-canvas'));
    final box=t.renderObject<RenderBox>(canvas);
    for(final p in [const Offset(.3,.4),const Offset(.6,.6)]) {
      await t.tapAt(box.localToGlobal(Offset(p.dx*box.size.width,p.dy*box.size.height)));await t.pump();
    }
    expect(t.getRect(find.text('Применить отрезок')).bottom,lessThan(800));
    await t.tap(find.text('Применить отрезок'));await t.pumpAndSettle();
    expect(result,hasLength(2));
    expect(result![0].dx,closeTo(.3,1e-6));expect(result![0].dy,closeTo(.4,1e-6));
    expect(result![1].dx,closeTo(.6,1e-6));expect(result![1].dy,closeTo(.6,1e-6));
    expect(t.takeException(),isNull);
  });

  testWidgets('mask drag and undo survive resized canvas without changing original coordinates', (t) async {
    await viewport(t);
    const points=[Offset(.2,.2),Offset(.8,.2),Offset(.5,.8)];
    Map<String,dynamic>? draft;
    await t.pumpWidget(MaterialApp(theme:AppTheme.light(),home:MaskDrawingPage(
      originalImageBase64:base64Encode(photo),
      editorState:const ContourEditorState(width:100,height:200,closed:true,points:points),
      onDraftChanged:(v)=>draft=v)));
    await t.runAsync(() async=>Future<void>.delayed(const Duration(milliseconds:150)));await t.pumpAndSettle();
    final canvas=find.byKey(const ValueKey('mask-image-canvas'));
    final rect=t.getRect(canvas);
    final start=rect.topLeft+Offset(rect.width*.2,rect.height*.2);
    await t.dragFrom(start,Offset(rect.width*.1,rect.height*.1));await t.pump();
    expect((draft!['points'][0]['x'] as num).toDouble(),closeTo(.3,1e-5));
    t.view.physicalSize=const Size(412,720);await t.pumpAndSettle();
    await t.tap(find.byTooltip('Отменить последнее действие'));await t.pumpAndSettle();
    final restored=ContourEditorState.fromJson(draft!);
    expect(restored.width,100);expect(restored.height,200);
    for(var i=0;i<points.length;i++){
      expect(restored.points[i].dx,closeTo(points[i].dx,1e-6));
      expect(restored.points[i].dy,closeTo(points[i].dy,1e-6));
    }
    expect(t.getRect(canvas).height,greaterThan(200));
    expect(t.getRect(find.text('Подтвердить')).bottom,lessThan(720));
    expect(t.takeException(),isNull);
  });
}
