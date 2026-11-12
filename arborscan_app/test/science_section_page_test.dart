import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:arborscan_app/report_history_service.dart';
import 'package:arborscan_app/report_export_data.dart';
import 'package:arborscan_app/report_pdf.dart';
import 'package:arborscan_app/science_section.dart';
import 'package:arborscan_app/science_section_service.dart';
import 'package:arborscan_app/science_section_page.dart';
import 'package:arborscan_app/science_result_card.dart';
import 'package:arborscan_app/server_report_page.dart';
import 'science_section_service_test.dart'
    show
        scienceForm,
        scienceResult,
        scienceReply,
        scienceOwner,
        scienceSource,
        scienceJson;
import 'report_export_test.dart' show fixturePhoto;

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  setUp(() => SharedPreferences.setMockInitialValues(
      {'arborscan_auth_token': 'first', 'arborscan_user_id': scienceOwner}));
  ScienceSectionService service(
          Future<http.Response> Function(http.Request) handler) =>
      ScienceSectionService(
          history:
              ReportHistoryService(clientFactory: () => MockClient(handler)));
  Future<http.Response> capability(http.Request r) async => http.Response(
      jsonEncode({
        'science_version': 1,
        'methods': [sectionMethodId]
      }),
      200);
  Widget app(ScienceSectionService service, {double scale = 1}) => MaterialApp(
      builder: (c, w) => MediaQuery(
          data: MediaQuery.of(c).copyWith(textScaler: TextScaler.linear(scale)),
          child: w!),
      home:
          ScienceSectionPage(sourceVersionId: scienceSource, service: service));
  testWidgets(
      'partial form survives screen recreation even offline; unsupported server does not post',
      (tester) async {
    final s = service((_) async => http.Response('{}', 404));
    await tester.pumpWidget(app(s));
    await tester
        .runAsync(() => Future<void>.delayed(const Duration(milliseconds: 10)));
    await tester.pumpAndSettle();
    await tester.enterText(find.byKey(const ValueKey('diameter_text')), '145');
    await tester
        .runAsync(() => Future<void>.delayed(const Duration(milliseconds: 10)));
    await tester.pumpAndSettle();
    await tester.pumpWidget(const MaterialApp(home: SizedBox()));
    await tester
        .runAsync(() => Future<void>.delayed(const Duration(milliseconds: 10)));
    await tester.pumpAndSettle();
    final offline =
        service((_) async => throw const SocketException('offline'));
    await tester.pumpWidget(app(offline));
    await tester
        .runAsync(() => Future<void>.delayed(const Duration(milliseconds: 10)));
    await tester.pumpAndSettle();
    expect(
        tester
            .widget<TextField>(find.byKey(const ValueKey('diameter_text')))
            .controller!
            .text,
        '145');
    await tester.scrollUntilVisible(
        find.byKey(const ValueKey('calculate-section')), 250,
        scrollable: find.byType(Scrollable).first);
    expect(
        tester
            .widget<FilledButton>(
                find.byKey(const ValueKey('calculate-section')))
            .onPressed,
        isNull);
    expect(tester.takeException(), isNull);
  });
  testWidgets(
      'large font restores complete inputs, submits once and shows server result separately',
      (tester) async {
    var posts = 0;
    final response = Completer<http.Response>();
    late Map request;
    final s = service((r) {
      if (r.method == 'GET') return capability(r);
      posts++;
      request = jsonDecode(r.body);
      return response.future;
    });
    SharedPreferences.setMockInitialValues({
      'arborscan_auth_token': 'first',
      'arborscan_user_id': scienceOwner,
      'arborscan_section_draft_v1_$scienceOwner/$scienceSource': jsonEncode({
        'schema_version': 1,
        'source_version_id': scienceSource,
        'form': scienceForm(),
        'attempted': false,
        'saved': false
      })
    });
    await tester.pumpWidget(app(s, scale: 1.6));
    await tester
        .runAsync(() => Future<void>.delayed(const Duration(milliseconds: 10)));
    await tester.pumpAndSettle();
    await tester.scrollUntilVisible(
        find.byKey(const ValueKey('calculate-section')), 250,
        scrollable: find.byType(Scrollable).first);
    await tester
        .runAsync(() => Future<void>.delayed(const Duration(milliseconds: 10)));
    await tester.pumpAndSettle();
    await tester.tap(find.byKey(const ValueKey('calculate-section')));
    await tester.pump();
    await tester.pump();
    expect(
        tester
            .widget<FilledButton>(
                find.byKey(const ValueKey('calculate-section')))
            .onPressed,
        isNull);
    await tester
        .runAsync(() => Future<void>.delayed(const Duration(milliseconds: 20)));
    await tester.pump();
    expect(posts, 1);
    response.complete(scienceJson(scienceReply(request)));
    await tester
        .runAsync(() => Future<void>.delayed(const Duration(milliseconds: 10)));
    await tester.pumpAndSettle();
    await tester.scrollUntilVisible(find.byType(ScienceResultCard), 200,
        scrollable: find.byType(Scrollable).first);
    expect(find.byType(ScienceResultCard), findsOneWidget);
    await tester.scrollUntilVisible(
        find.text('Открыть сохранённую версию'), 250,
        scrollable: find.byType(Scrollable).first);
    await tester
        .runAsync(() => Future<void>.delayed(const Duration(milliseconds: 10)));
    await tester.pumpAndSettle();
    expect(find.text('Открыть сохранённую версию'), findsOneWidget);
    expect(
        (await tester
                .runAsync(() => s.draft('first', scienceSource)))!['inputs']
            ['diameter']['value'],
        .2);
    expect(tester.takeException(), isNull);
  });
  testWidgets(
      'pending operation restores locked fields and exact retry after process recreation',
      (tester) async {
    const version = 'dddddddd-dddd-4ddd-8ddd-dddddddddddd';
    SharedPreferences.setMockInitialValues({
      'arborscan_auth_token': 'first',
      'arborscan_user_id': scienceOwner,
      'arborscan_section_draft_v1_$scienceOwner/$scienceSource': jsonEncode({
        'schema_version': 1,
        'source_version_id': scienceSource,
        'form': scienceForm(),
        'attempted': true,
        'saved': false,
        'version_id': version,
        'inputs': sectionInputs(scienceForm())
      })
    });
    var posts = 0;
    final reopened = service((r) async {
      if (r.method == 'GET') return capability(r);
      posts++;
      final body = jsonDecode(r.body);
      expect(body['version_id'], version);
      return scienceJson(scienceReply(body));
    });
    await tester.pumpWidget(app(reopened));
    await tester
        .runAsync(() => Future<void>.delayed(const Duration(milliseconds: 10)));
    await tester.pumpAndSettle();
    expect(
        tester
            .widget<TextField>(find.byKey(const ValueKey('diameter_text')))
            .enabled,
        false);
    await tester.scrollUntilVisible(
        find.byKey(const ValueKey('calculate-section')), 250,
        scrollable: find.byType(Scrollable).first);
    await tester
        .runAsync(() => Future<void>.delayed(const Duration(milliseconds: 10)));
    await tester.pumpAndSettle();
    expect(find.text('Повторить тот же запрос'), findsOneWidget);
    await tester.tap(find.byKey(const ValueKey('calculate-section')));
    await tester
        .runAsync(() => Future<void>.delayed(const Duration(milliseconds: 10)));
    await tester.pumpAndSettle();
    await tester
        .runAsync(() => Future<void>.delayed(const Duration(milliseconds: 20)));
    await tester.pump();
    expect(posts, 1);
    await tester.scrollUntilVisible(find.byType(ScienceResultCard), 200,
        scrollable: find.byType(Scrollable).first);
    expect(find.byType(ScienceResultCard), findsOneWidget);
  });
  testWidgets(
      'saved server snapshot exposes calculation entry and a persisted scenario',
      (tester) async {
    final result = scienceResult(
        'dddddddd-dddd-4ddd-8ddd-dddddddddddd', sectionInputs(scienceForm()));
    final history = ReportHistoryService(
        clientFactory: () => MockClient((_) async => scienceJson({
              'record': {
                'version_id': scienceSource,
                'analysis_id': scienceOwner
              },
              'snapshot': {
                'version': 1,
                'kind': 'legacy',
                'report': {},
                'science_result': result,
                'image': {'original_base64': base64Encode(fixturePhoto())}
              }
            })));
    await tester.pumpWidget(MaterialApp(
        home: ServerReportPage(versionId: scienceSource, service: history)));
    await tester
        .runAsync(() => Future<void>.delayed(const Duration(milliseconds: 10)));
    await tester.pumpAndSettle();
    await tester.scrollUntilVisible(find.byType(ScienceResultCard), 200,
        scrollable: find.byType(Scrollable).first);
    expect(find.byType(ScienceResultCard), findsOneWidget);
    await tester.scrollUntilVisible(find.text('Расчёты: сценарий сечения'), 250,
        scrollable: find.byType(Scrollable).first);
    expect(find.text('Расчёты: сценарий сечения'), findsOneWidget);
    expect(tester.takeException(), isNull);
  });
  testWidgets(
      'known validation rejection offers explicit correction after reopening',
      (tester) async {
    const oldVersion = 'eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee';
    final large = {...scienceForm(), 'diameter_text': '1e100'};
    SharedPreferences.setMockInitialValues({
      'arborscan_auth_token': 'first',
      'arborscan_user_id': scienceOwner,
      'arborscan_section_draft_v1_$scienceOwner/$scienceSource': jsonEncode({
        'schema_version': 1,
        'source_version_id': scienceSource,
        'form': large,
        'attempted': true,
        'saved': false,
        'validation_rejected': true,
        'version_id': oldVersion,
        'inputs': sectionInputs(large)
      })
    });
    final s = service((r) async {
      if (r.method == 'GET') return capability(r);
      final body = jsonDecode(r.body) as Map;
      expect(body['version_id'], isNot(oldVersion));
      expect(body['inputs']['diameter']['value'], .2);
      return scienceJson(scienceReply(body));
    });
    await tester.pumpWidget(app(s));
    await tester.pumpAndSettle();
    expect(
        tester
            .widget<TextField>(find.byKey(const ValueKey('diameter_text')))
            .enabled,
        false);
    await tester.scrollUntilVisible(
        find.byKey(const ValueKey('edit-rejected-section')), 250,
        scrollable: find.byType(Scrollable).first);
    await tester.tap(find.byKey(const ValueKey('edit-rejected-section')));
    await tester.pumpAndSettle();
    await tester.scrollUntilVisible(
        find.byKey(const ValueKey('diameter_text')), -250,
        scrollable: find.byType(Scrollable).first);
    expect(
        tester
            .widget<TextField>(find.byKey(const ValueKey('diameter_text')))
            .enabled,
        true);
    await tester.enterText(find.byKey(const ValueKey('diameter_text')), '200');
    await tester.pumpAndSettle();
    await tester.scrollUntilVisible(
        find.byKey(const ValueKey('calculate-section')), 250,
        scrollable: find.byType(Scrollable).first);
    await tester.tap(find.byKey(const ValueKey('calculate-section')));
    await tester.pumpAndSettle();
    await tester.scrollUntilVisible(find.byType(ScienceResultCard), 200,
        scrollable: find.byType(Scrollable).first);
    expect(find.byType(ScienceResultCard), findsOneWidget);
    expect(tester.takeException(), isNull);
  });
  test('science PDF preserves original metrics and exact scenario provenance',
      () async {
    final result = scienceResult(
        'dddddddd-dddd-4ddd-8ddd-dddddddddddd', sectionInputs(scienceForm()));
    final data = ReportExportData(local: true, photo: fixturePhoto(), record: {
      'analysis_id': scienceOwner,
      'version_id': result['result_id'],
      'parent_id': scienceSource
    }, snapshot: {
      'version': 1,
      'kind': 'legacy',
      'captured_at': '2026-11-12T10:00:00Z',
      'report': {
        'height_m': 12.3,
        'crown_width_m': 4.5,
        'species': 'ДЕМОНСТРАЦИОННЫЕ ДАННЫЕ'
      },
      'science_result': result,
    });
    final before = jsonEncode(data.snapshot);
    final pdf =
        await buildReportPdf(data, generatedAt: DateTime.utc(2026, 11, 12, 10));
    expect(jsonEncode(data.snapshot), before);
    expect(pdf.length, greaterThan(10000));
    final output = File(
        '../output/remediation-20261112/flutter/science/ArborScan_DEMO_section.pdf');
    await output.parent.create(recursive: true);
    await output.writeAsBytes(pdf);
  });
}
