import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:arborscan_app/contour_drafts.dart';
import 'package:arborscan_app/corrections_service.dart';
import 'package:arborscan_app/report_history_service.dart';

const owner='aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa';
void main(){
 TestWidgetsFlutterBinding.ensureInitialized();
 late Directory temp;
 late ContourDrafts journal;
 setUp(() async {
  SharedPreferences.setMockInitialValues({'arborscan_auth_token':'first','arborscan_user_id':owner});
  temp=await Directory.systemTemp.createTemp('report-test-');
  journal=ContourDrafts(directory:()async=>temp);
 });
 tearDown(()async{await temp.delete(recursive:true);});
 Future<Map<String,dynamic>> stage(ReportHistoryService service,{int value=1,String? parent})=>service.stage(
   token:'first',localId:'local',analysisId:owner,image:Uint8List.fromList([1,2,3]),parentId:parent,
   snapshot:{'version':1,'report':{'height_m':value},'captured_at':'2026-09-17T00:00:00Z'});
 test('restart retains operation ID; uncertain reply retries exactly; new edit keeps parent',()async{
  final service=ReportHistoryService(journal:journal,clientFactory:()=>MockClient((r)async=>http.Response('{"saved":false}',200)));
  final first=await stage(service);
  await expectLater(service.upload('first','local'),throwsA(isA<CorrectionException>()));
  final reopened=ReportHistoryService(journal:journal,clientFactory:()=>MockClient((r)async=>http.Response(
    jsonEncode({'saved':true,'persisted':true,'record':{'version_id':first['version_id']}}),200)));
  expect((await stage(reopened))['version_id'],first['version_id']);
  await reopened.upload('first','local');
  final child=await stage(reopened,value:2);
  expect(child['parent_id'],first['version_id']);
  expect(child['version_id'],isNot(first['version_id']));
  expect((await journal.load(owner,'local'))!['saved'],false);
 });
 test('account switch rejects in-flight response and does not mark operation saved',()async{
  final response=Completer<http.Response>();
  final sent=Completer<void>();
  final service=ReportHistoryService(journal:journal,clientFactory:()=>MockClient((r){sent.complete();return response.future;}));
  final d=await stage(service);
  final uploading=service.upload('first','local');
  final assertion=expectLater(uploading,throwsA(isA<CorrectionException>()));
  await sent.future;
  final prefs=await SharedPreferences.getInstance();await prefs.setString('arborscan_auth_token','second');
  CorrectionsService.authChanges.value++;
  response.complete(http.Response(jsonEncode({'saved':true,'persisted':true,'record':{'version_id':d['version_id']}}),200));
  await assertion;
  expect((await journal.load(owner,'local'))!['saved'],false);
  expect(await journal.list('bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb'),isEmpty);
 });
 test('network and conflict preserve complete local snapshot and photo',()async{
  for(final status in [409,503,401]){
   final service=ReportHistoryService(journal:journal,clientFactory:()=>MockClient((r)async=>http.Response('{}',status)));
   await stage(service);
   await expectLater(service.upload('first','local'),throwsA(isA<CorrectionException>()));
   final d=(await journal.load(owner,'local'))!;
   expect(d['image'],[1,2,3]);expect(d['snapshot']['report']['height_m'],1);expect(d['saved'],false);
  }
 });
}
