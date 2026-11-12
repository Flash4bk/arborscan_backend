import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:arborscan_app/corrections_service.dart';
import 'package:arborscan_app/report_history_service.dart';
import 'package:arborscan_app/science_section.dart';
import 'package:arborscan_app/science_section_service.dart';

http.Response scienceJson(dynamic data) => http.Response(jsonEncode(data), 200,
    headers: {'content-type': 'application/json; charset=utf-8'});

const scienceOwner = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa';
const scienceSource = 'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb';
Map<String, dynamic> scienceForm() => {
      'diameter_text': '200',
      'diameter_unit': 'mm',
      'mor_mpa_text': '40',
      'diameter_origin': 'measured',
      'mor_origin': 'scenario_assumption',
      'diameter_source': 'Рулетка, тестовое сечение',
      'mor_source': 'Учебный сценарий',
      'strength_definition':
          'Заданная эффективная прочность; поправки не применялись',
      'assumptions_confirmed': true,
    };
Map<String, dynamic> scienceResult(String version, Map inputs) => {
      'schema_version': 1,
      'method_id': sectionMethodId,
      'method_version': '1.0.0',
      'quantity': 'single_section_bending_capacity',
      'unit': 'N m',
      'capacity_nm': 31415.9265358979,
      'validation_status': 'scenario_software_only',
      'field_validated': false,
      'input_snapshot': inputs,
      'result_id': version,
      'source_version_id': scienceSource,
      'source_doi': '10.1016/j.envsoft.2015.01.016',
      for (final key in [
        'input_sha256',
        'result_sha256',
        'implementation_sha256',
        'source_payload_sha256',
        'image_sha256'
      ])
        key: List.filled(64, 'a').join(),
    };
Map<String, dynamic> scienceReply(Map request) => {
      'saved': true,
      'persisted': true,
      'science_version': 1,
      'record': {'version_id': request['version_id']},
      'science_result': scienceResult(request['version_id'], request['inputs']),
    };

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  setUp(() => SharedPreferences.setMockInitialValues(
      {'arborscan_auth_token': 'first', 'arborscan_user_id': scienceOwner}));
  ScienceSectionService service(
          Future<http.Response> Function(http.Request) handler) =>
      ScienceSectionService(
          history:
              ReportHistoryService(clientFactory: () => MockClient(handler)));
  test('m/mm inputs and MPa conversion produce the same explicit SI scenario',
      () {
    final mm = sectionInputs(scienceForm());
    final metres = sectionInputs(
        {...scienceForm(), 'diameter_text': '0,2', 'diameter_unit': 'm'});
    expect(sameScienceInputs(mm, metres), true);
    expect(mm['diameter']['value'], .2);
    expect(mm['effective_mor']['value'], 40000000);
    for (final delta in [
      {'diameter_text': '0'},
      {'mor_mpa_text': 'NaN'},
      {'assumptions_confirmed': false},
      {'diameter_origin': null},
      {'mor_source': ''},
      {'strength_definition': ''}
    ]) {
      expect(() => sectionInputs({...scienceForm(), ...delta}),
          throwsFormatException);
    }
  });
  test('fresh service restores partial form; accounts have separate journals',
      () async {
    final a = service((_) async => http.Response('{}', 200));
    await a.saveForm(
        'first', scienceSource, {'diameter_text': '12', 'diameter_unit': 'mm'});
    expect(
        (await service((_) async => http.Response('{}', 200))
            .draft('first', scienceSource))!['form']['diameter_text'],
        '12');
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('arborscan_auth_token', 'second');
    await prefs.setString(
        'arborscan_user_id', 'cccccccc-cccc-4ccc-8ccc-cccccccccccc');
    expect(await a.draft('second', scienceSource), isNull);
  });
  test(
      'lost reply restores exact operation and inputs after service recreation',
      () async {
    final requests = <Map>[];
    var lost = true;
    Future<http.Response> handler(http.Request r) async {
      expect(r.headers['Authorization'], 'Bearer first');
      expect(r.url.queryParameters.containsKey('token'), false);
      final body = jsonDecode(r.body) as Map;
      requests.add(body);
      if (lost) throw const SocketException('lost after commit');
      return scienceJson(scienceReply(body));
    }

    final a = service(handler);
    await a.saveForm('first', scienceSource, scienceForm());
    final inputs = sectionInputs(scienceForm());
    await expectLater(a.calculate('first', scienceSource, inputs),
        throwsA(isA<CorrectionException>()));
    final b = service(handler);
    final draft = (await b.draft('first', scienceSource))!;
    expect(draft['attempted'], true);
    expect(draft['saved'], false);
    await expectLater(
        b.saveForm(
            'first', scienceSource, {...scienceForm(), 'diameter_text': '220'}),
        throwsA(isA<CorrectionException>()));
    lost = false;
    await b.calculate('first', scienceSource, inputs);
    expect(requests.length, 2);
    expect(requests[0], requests[1]);
    expect((await b.draft('first', scienceSource))!['saved'], true);
  });
  for (final status in [401, 403, 409, 503]) {
    test('HTTP $status retains form and operation, does not confirm save',
        () async {
      final s = service((_) async => http.Response('{}', status));
      await s.saveForm('first', scienceSource, scienceForm());
      await expectLater(
          s.calculate('first', scienceSource, sectionInputs(scienceForm())),
          throwsA(isA<CorrectionException>()));
      final d = (await s.draft('first', scienceSource))!;
      expect(d['attempted'], true);
      expect(d['saved'], false);
      expect(d['form'], scienceForm());
    });
  }
  for (final status in [404, 405]) {
    test('old API $status is unavailable without destroying the draft',
        () async {
      final s = service((_) async => http.Response('{}', status));
      expect(await s.available('first'), false);
      await s.saveForm('first', scienceSource, scienceForm());
      await expectLater(
          s.calculate('first', scienceSource, sectionInputs(scienceForm())),
          throwsA(isA<CorrectionException>().having(
              (e) => e.message, 'message', contains('пока не поддерживает'))));
      expect((await s.draft('first', scienceSource))!['saved'], false);
    });
  }
  test(
      'saved false, mismatched inputs and malformed scientific response are never success',
      () async {
    for (final change in ['saved', 'inputs', 'unit', 'strength']) {
      SharedPreferences.setMockInitialValues(
          {'arborscan_auth_token': 'first', 'arborscan_user_id': scienceOwner});
      final s = service((r) async {
        final body = jsonDecode(r.body) as Map;
        final response = scienceReply(body);
        if (change == 'saved') response['saved'] = false;
        if (change == 'inputs') {
          response['science_result']['input_snapshot']['diameter']['value'] =
              .3;
        }
        if (change == 'unit') response['science_result']['unit'] = 'kg/s';
        if (change == 'strength') {
          response['science_result']['input_snapshot']
              .remove('strength_definition');
        }
        return scienceJson(response);
      });
      await s.saveForm('first', scienceSource, scienceForm());
      await expectLater(
          s.calculate('first', scienceSource, sectionInputs(scienceForm())),
          throwsA(isA<CorrectionException>()));
      expect((await s.draft('first', scienceSource))!['saved'], false);
    }
  });
  test(
      'late reply after account change cannot save into either new owner journal',
      () async {
    final sent = Completer<void>(), response = Completer<http.Response>();
    late Map body;
    final s = service((r) {
      body = jsonDecode(r.body);
      sent.complete();
      return response.future;
    });
    await s.saveForm('first', scienceSource, scienceForm());
    final calculation =
        s.calculate('first', scienceSource, sectionInputs(scienceForm()));
    final check = expectLater(calculation, throwsA(isA<CorrectionException>()));
    await sent.future;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('arborscan_auth_token', 'second');
    await prefs.setString(
        'arborscan_user_id', 'cccccccc-cccc-4ccc-8ccc-cccccccccccc');
    CorrectionsService.authChanges.value++;
    response.complete(scienceJson(scienceReply(body)));
    await check;
    expect(await s.draft('second', scienceSource), isNull);
    await prefs.setString('arborscan_auth_token', 'first');
    await prefs.setString('arborscan_user_id', scienceOwner);
    expect((await s.draft('first', scienceSource))!['saved'], false);
  });
  test(
      'definitive validation rejection survives restart and permits explicit corrected operation',
      () async {
    final requests = <Map>[];
    var rejected = true;
    Future<http.Response> handler(http.Request r) async {
      final body = jsonDecode(r.body) as Map;
      requests.add(body);
      return rejected
          ? http.Response('{"detail":"numerical_domain"}', 422)
          : scienceJson(scienceReply(body));
    }

    final large = {...scienceForm(), 'diameter_text': '1e100'};
    final first = service(handler);
    await first.saveForm('first', scienceSource, large);
    await expectLater(
        first.calculate('first', scienceSource, sectionInputs(large)),
        throwsA(isA<CorrectionException>()));
    final reopened = service(handler);
    expect(
        (await reopened.draft('first', scienceSource))!['validation_rejected'],
        true);
    await reopened.editRejected('first', scienceSource);
    await reopened.saveForm('first', scienceSource, scienceForm());
    rejected = false;
    await reopened.calculate(
        'first', scienceSource, sectionInputs(scienceForm()));
    expect(requests.length, 2);
    expect(requests[1]['version_id'], isNot(requests[0]['version_id']));
    expect(requests[0]['inputs']['diameter']['value'], greaterThan(1e90));
    expect(requests[1]['inputs']['diameter']['value'], .2);
    expect((await reopened.draft('first', scienceSource))!['saved'], true);
  });
  test('unknown outcome cannot be unlocked as a validation rejection',
      () async {
    final s =
        service((_) async => throw const SocketException('unknown outcome'));
    await s.saveForm('first', scienceSource, scienceForm());
    await expectLater(
        s.calculate('first', scienceSource, sectionInputs(scienceForm())),
        throwsA(isA<CorrectionException>()));
    final before = await s.draft('first', scienceSource);
    await expectLater(s.editRejected('first', scienceSource),
        throwsA(isA<CorrectionException>()));
    expect(await s.draft('first', scienceSource), before);
  });
}
