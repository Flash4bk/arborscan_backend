import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:arborscan_app/contour_drafts.dart';
import 'package:arborscan_app/contour_editor_state.dart';
import 'package:arborscan_app/contour_workspace_page.dart';
import 'package:arborscan_app/corrections_service.dart';
import 'package:arborscan_app/mask_drawing_page.dart';

const owner='aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa';
const other='bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb';
const aid='cccccccc-cccc-4ccc-8ccc-cccccccccccc';
final image = base64Decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNImXYCAAMkAcMgVWSjAAAAAElFTkSuQmCC');
Map<String,dynamic> editor({bool closed=true}) => ContourEditorState(width:1,height:1,
  closed:closed,points:const [Offset(.1,.1),Offset(.8,.1),Offset(.5,.9)]).toJson();

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  setUp(() => SharedPreferences.setMockInitialValues({'arborscan_auth_token':'alice','arborscan_user_id':owner}));

  test('editor state restores exact normalized points and open/closed state', () {
    for (final closed in [true,false]) {
      expect(ContourEditorState.fromJson(editor(closed:closed)).toJson(),editor(closed:closed));
    }
    expect(() => ContourEditorState.fromJson({...editor(),'version':2}),throwsFormatException);
    expect(() => ContourEditorState.fromJson({...editor(),'points':[{'x':double.nan,'y':0}]}),throwsFormatException);
  });

  testWidgets('mismatched editor dimensions stay blocked across rebuilds', (tester) async {
    Widget page() => MaterialApp(home: MaskDrawingPage(
      originalImageBase64: base64Encode(image),
      editorState: const ContourEditorState(width:2, height:1, closed:false, points:[])));
    await tester.pumpWidget(page());
    await tester.runAsync(() async => Future<void>.delayed(const Duration(milliseconds:100)));
    await tester.pumpAndSettle();
    expect(find.textContaining('Размеры состояния не совпадают'), findsOneWidget);
    await tester.binding.setSurfaceSize(const Size(700,900));
    await tester.pumpWidget(page());
    await tester.pumpAndSettle();
    expect(find.textContaining('Размеры состояния не совпадают'), findsOneWidget);
    expect(find.byType(InteractiveViewer), findsNothing);
    await tester.binding.setSurfaceSize(null);
  });

  test('disk draft survives service recreation and is owner isolated', () async {
    final folder=await Directory.systemTemp.createTemp('arbor-drafts-test-');
    try {
      final store=ContourDrafts(directory:() async=>folder);
      final data={'analysis_id':aid,'parent_id':'parent','editor_state':editor(closed:false)};
      await store.save(owner,'draft',data,image);
      final recreated=ContourDrafts(directory:() async=>folder);
      final loaded=await recreated.load(owner,'draft');
      expect(loaded!['editor_state'],data['editor_state']);
      expect(loaded['image'],image);
      expect(await recreated.load(other,'draft'),isNull);
      await recreated.save(other,'draft',{'analysis_id':'different'},image);
      expect((await recreated.load(owner,'draft'))!['analysis_id'],aid);
      await recreated.remove(other,'draft');
      expect(await recreated.load(owner,'draft'),isNotNull);
    } finally { await folder.delete(recursive:true); }
  });

  test('old API capabilities retain legacy save and return clear conflict', () async {
    final service=CorrectionsService(clientFactory:()=>MockClient((req) async {
      if(req.url.path.endsWith('/capabilities')) return http.Response('',404);
      if(req.url.path.endsWith('/workflow')) return http.Response('',409);
      return http.Response('{"saved":true}',200);
    }));
    expect(await service.workflowAvailable('alice'),false);
    await service.save(token:'alice',analysisId:aid,image:image,mask:image);
    await expectLater(service.saveRevision(token:'alice',analysisId:aid,image:image,mask:image,
      editorState:editor()),throwsA(isA<CorrectionException>().having((e)=>e.statusCode,'conflict',409)));
  });

  testWidgets('saved draft restores after screen recreation, retry is idempotent and send is blocked', (tester) async {
    await tester.binding.setSurfaceSize(const Size(800,1200));
    addTearDown(() => tester.binding.setSurfaceSize(null));
    final drafts=MemoryDrafts();
    await drafts.save(owner,'${aid}_root',{'analysis_id':aid,'editor_state':editor(),
      'mask_png_base64':base64Encode(image)},image);
    final waiting=Completer<http.Response>();
    var saves=0;
    final bodies=<String>[];
    final service=CorrectionsService(clientFactory:()=>MockClient((req) async {
      if(req.url.path.endsWith('/capabilities')) return http.Response('{"workflow_version":1}',200);
      saves++; bodies.add(latin1.decode(req.bodyBytes));
      if(saves==1) return waiting.future;
      return http.Response('{"saved":true,"workflow_version":1,"correction_id":"revision","review_status":"draft"}',200);
    }));
    Widget page() => MaterialApp(home:ContourWorkspacePage(analysisId:aid,drafts:drafts,service:service));
    await tester.pumpWidget(page()); await tester.pumpAndSettle();
    expect(find.text('Продолжить редактирование'),findsOneWidget);
    await tester.ensureVisible(find.text('Сохранить контур'));
    await tester.tap(find.text('Сохранить контур')); await tester.pump(); await tester.pump();
    expect(tester.widget<FilledButton>(find.widgetWithText(FilledButton,'Сохранить контур')).onPressed,isNull);
    waiting.complete(http.Response('{"saved":false}',200)); await tester.pumpAndSettle();
    expect(find.text('Сохранено'),findsNothing);
    await tester.pumpWidget(const SizedBox());
    await tester.pumpWidget(page()); await tester.pumpAndSettle();
    await tester.ensureVisible(find.text('Сохранить контур'));
    await tester.tap(find.text('Сохранить контур')); await tester.pumpAndSettle();
    expect(find.text('Сохранено'),findsOneWidget); expect(saves,2);
    // Multipart boundaries may differ; the logical fields and payload must not.
    for(final body in bodies) { expect(body,contains(aid)); expect(body,contains('normalized_oriented_image')); }
    expect((await drafts.load(owner,'${aid}_root'))!['saved_id'],'revision');
    CorrectionsService.authChanges.value++;
    await tester.pumpAndSettle();
    expect(find.byType(Image),findsNothing);
  });

  testWidgets('legacy PNG uses explicit new contour, accepted state restores editor and new edit loses acceptance', (tester) async {
    final drafts=MemoryDrafts();
    final service=CorrectionsService(clientFactory:()=>MockClient((_) async=>http.Response('{"workflow_version":1}',200)));
    final legacy={'analysis_id':aid,'correction_id':'legacy','mask_png_base64':base64Encode(image),
      'review_status':'pending_review'};
    await tester.pumpWidget(MaterialApp(home:ContourWorkspacePage(analysisId:aid,image:image,
      record:legacy,drafts:drafts,service:service)));
    await tester.pumpAndSettle();
    expect(find.text('Создать новый контур по фото'),findsOneWidget);
    await tester.pumpWidget(const SizedBox());
    final record={...legacy,'correction_id':'accepted','editor_state':editor(),'review_status':'accepted'};
    await tester.pumpWidget(MaterialApp(home:ContourWorkspacePage(analysisId:aid,image:image,
      record:record,drafts:drafts,service:service)));
    await tester.pumpAndSettle();
    await tester.ensureVisible(find.text('Продолжить редактирование'));
    await tester.tap(find.text('Продолжить редактирование'));
    await tester.pump(); await tester.pump(const Duration(seconds:1));
    final page=tester.widget<MaskDrawingPage>(find.byType(MaskDrawingPage));
    expect(page.editorState!.toJson(),editor());
    final changed={...editor(),'points':[{'x':.2,'y':.1},{'x':.8,'y':.1},{'x':.5,'y':.9}]};
    page.onDraftChanged!(changed);
    Navigator.of(tester.element(find.byType(MaskDrawingPage))).pop({'editor_state':changed,'mask_png_base64':base64Encode(image)});
    await tester.pumpAndSettle();
    expect(find.text('Статус: черновик'),findsOneWidget);
    final draft=await drafts.load(owner,'${aid}_accepted');
    expect(draft!['parent_id'],'accepted'); expect(draft['saved_id'],isNull);
  });
}

class MemoryDrafts extends ContourDrafts {
  final data=<String,Map<String,dynamic>>{};
  @override
  Future<Map<String,dynamic>?> load(String owner,String id) async => data['$owner/$id'];
  @override
  Future<void> save(String owner,String id,Map<String,dynamic> value,Uint8List image) async {
    data['$owner/$id']={...jsonDecode(jsonEncode(value)) as Map<String,dynamic>,'image':image};
  }
}
