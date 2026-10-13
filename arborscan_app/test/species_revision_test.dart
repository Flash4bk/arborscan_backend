import 'dart:convert';
import 'dart:io';
import 'package:arborscan_app/contour_drafts.dart';
import 'package:arborscan_app/report_history_service.dart';
import 'package:arborscan_app/server_report_page.dart';
import 'package:arborscan_app/species_annotation.dart';
import 'package:arborscan_app/unified_analysis_report_page.dart';
import 'package:arborscan_app/unified_analysis_models.dart';
import 'package:arborscan_app/report_export_button.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import 'combined_reference_service_test.dart' show InspectClient, response;

void main() {
  testWidgets(
      'fresh analysis saves and exports a manual species without changing inference',
      (t) async {
    const owner = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa';
    SharedPreferences.setMockInitialValues(
        {'arborscan_auth_token': 'test', 'arborscan_user_id': owner});
    final folder = (await t
        .runAsync(() => Directory.systemTemp.createTemp('species-analysis-')))!;
    final photo = (await t.runAsync(
        () => File('test/fixtures/reference_exif6.jpg').readAsBytes()))!;
    var posts = 0;
    final journal = ContourDrafts(directory: () async => folder);
    final service = ReportHistoryService(
        journal: journal,
        clientFactory: () => InspectClient((r) async {
              posts++;
              final m = r as http.MultipartRequest;
              final snapshot = jsonDecode(m.fields['snapshot']!);
              expect(userSpeciesName(snapshot['report']), 'Сосна');
              expect(
                  snapshot['report']['measurements']['height']['value_m'], 4.0);
              expect(
                  snapshot['report']['species']['display_name'], 'Неизвестно');
              expect(snapshot['change_source'], 'manual');
              return response({
                'saved': true,
                'persisted': true,
                'record': {'version_id': m.fields['version_id']}
              });
            }));
    final result = UnifiedAnalysisResult.fromJson({
      'analysis_id': owner,
      'species': {'display_name': 'Неизвестно'},
      'tree': {'detected': true},
      'measurements': {
        'height': {'value_m': 4.0, 'source': 'ar'}
      }
    });
    Future<void> settle({bool allowBusy = false}) async {
      for (var i = 0; i < 150; i++) {
        await t.runAsync(
            () async => Future<void>.delayed(const Duration(milliseconds: 25)));
        await t.pump(const Duration(milliseconds: 100));
        if (i > 10 &&
            (allowBusy ||
                find.byType(LinearProgressIndicator).evaluate().isEmpty)) {
          return;
        }
      }
      fail('Screen remained busy');
    }

    await t.runAsync(() => t.pumpWidget(MaterialApp(
        home: UnifiedAnalysisReportPage(
            result: result, fallbackImageBytes: photo, history: service))));
    await settle();
    await t.scrollUntilVisible(find.text('Указать вид'), 150,
        scrollable: find.byType(Scrollable).first);
    await t.ensureVisible(find.text('Указать вид'));
    await t.pump();
    await t.runAsync(() => t.tap(find.text('Указать вид')));
    await settle(allowBusy: true);
    await t.enterText(
        find.descendant(
            of: find.byType(AlertDialog), matching: find.byType(TextField)),
        'Сосна');
    await t.pump();
    await t.runAsync(() => t.tap(find.text('Применить')));
    await settle();
    final saved = (await t.runAsync(() => journal.list(owner)))!.single;
    expect(userSpeciesName(saved['snapshot']['report']), 'Сосна');
    expect(result.raw.containsKey('species_annotation'), false);
    await t.scrollUntilVisible(find.byType(ReportExportButton), -150,
        scrollable: find.byType(Scrollable).first);
    final export = await t
        .widget<ReportExportButton>(find.byType(ReportExportButton))
        .load();
    expect(export.speciesLines.join('\n'), contains('Сосна'));
    await t.ensureVisible(find.text('Сохранить отчёт в аккаунте'));
    await t.pump();
    await t.runAsync(() => t.tap(find.text('Сохранить отчёт в аккаунте')));
    await settle();
    expect(posts, 1);
    expect(t.takeException(), isNull);
    await t.pumpWidget(const SizedBox());
    await t.runAsync(() => folder.delete(recursive: true));
  });

  testWidgets(
      'species edit makes a durable child and retries the same request after failure',
      (t) async {
    const owner = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa';
    const analysis = 'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb';
    const parent = 'cccccccc-cccc-4ccc-8ccc-cccccccccccc';
    SharedPreferences.setMockInitialValues(
        {'arborscan_auth_token': 'test', 'arborscan_user_id': owner});
    final folder = (await t
        .runAsync(() => Directory.systemTemp.createTemp('species-revision-')))!;
    final photo = (await t.runAsync(
        () => File('test/fixtures/reference_exif6.jpg').readAsBytes()))!;
    final journal = ContourDrafts(directory: () async => folder);
    final original = <String, dynamic>{
      'version': 1,
      'kind': 'v4',
      'captured_at': '2026-10-13T00:00:00Z',
      'change_source': 'analysis',
      'report': {
        'species': {'display_name': 'Предсказание'},
        'height_m': 4.0,
        'images': {'width': 100, 'height': 200}
      },
      'image': {'original_base64': base64Encode(photo)}
    };
    final frozen = jsonEncode(original);
    final requests = <Map<String, String>>[];
    final service = ReportHistoryService(
        journal: journal,
        cache: ContourDrafts(
            directory: () async => Directory('${folder.path}/cache')),
        clientFactory: () => InspectClient((request) async {
              if (request.method == 'GET') {
                return response({
                  'record': {'analysis_id': analysis, 'version_id': parent},
                  'snapshot': original
                });
              }
              final multipart = request as http.MultipartRequest;
              requests.add(Map.of(multipart.fields));
              expect(multipart.fields['parent_id'], parent);
              final snapshot = jsonDecode(multipart.fields['snapshot']!) as Map;
              expect(userSpeciesName(snapshot['report']), 'Сосна обыкновенная');
              expect(snapshot['report']['height_m'], 4.0);
              expect(snapshot['report']['species'],
                  original['report']!['species']);
              expect(snapshot['change_source'], 'manual');
              expect(await multipart.files.single.finalize().toBytes(), photo);
              if (requests.length == 1) {
                return response({'detail': 'temporary test failure'}, 503);
              }
              return response({
                'saved': true,
                'persisted': true,
                'record': {
                  'version_id': multipart.fields['version_id'],
                  'analysis_id': analysis
                }
              });
            }));
    Future<void> settle({bool allowBusy = false}) async {
      for (var i = 0; i < 150; i++) {
        await t.runAsync(
            () async => Future<void>.delayed(const Duration(milliseconds: 25)));
        await t.pump(const Duration(milliseconds: 100));
        if (i > 10 &&
            (allowBusy ||
                find.byType(LinearProgressIndicator).evaluate().isEmpty)) {
          return;
        }
      }
      fail('Screen remained busy');
    }

    Future<void> tap(String label) async {
      await settle();
      t
          .state<ScrollableState>(find.byType(Scrollable).first)
          .position
          .jumpTo(0);
      await t.pump();
      await t.scrollUntilVisible(find.text(label), 150,
          scrollable: find.byType(Scrollable).first);
      await t.ensureVisible(find.text(label));
      await t.pump();
      await t.runAsync(() => t.tap(find.text(label)));
      await settle(allowBusy: label == 'Указать вид');
    }

    await t.runAsync(() => t.pumpWidget(MaterialApp(
        home: ServerReportPage(versionId: parent, service: service))));
    await settle();
    await tap('Указать вид');
    await t.enterText(
        find.descendant(
            of: find.byType(AlertDialog), matching: find.byType(TextField)),
        'Сосна обыкновенная');
    await t.pump();
    await t.runAsync(() => t.tap(find.text('Применить')));
    await settle();
    final rows = (await t.runAsync(
        () => ContourDrafts(directory: () async => folder).list(owner)))!;
    expect(rows, hasLength(1));
    expect(rows.single['parent_id'], parent);
    expect(userSpeciesName(rows.single['snapshot']['report']),
        'Сосна обыкновенная');
    expect(jsonEncode(original), frozen);
    await tap('Сохранить на сервере / повторить');
    await tap('Сохранить на сервере / повторить');
    expect(requests, hasLength(2));
    expect(requests[0], requests[1]);
    expect(
        (await t.runAsync(() => journal.list(owner)))!.single['saved'], true);
    expect(jsonEncode(original), frozen);
    expect(t.takeException(), isNull);
    await t.pumpWidget(const SizedBox());
    await t.runAsync(() => folder.delete(recursive: true));
  });
}
