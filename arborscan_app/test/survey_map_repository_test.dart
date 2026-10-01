import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:arborscan_app/contour_drafts.dart';
import 'package:arborscan_app/corrections_service.dart';
import 'package:arborscan_app/report_history_service.dart';
import 'package:arborscan_app/survey_map_repository.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:shared_preferences/shared_preferences.dart';

const ownerA = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa';
const ownerB = 'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb';
Map<String, dynamic> environment(double lat, double lon) => {
      'version': 1,
      'gps': {
        'value': {'lat': lat, 'lon': lon, 'crs': 'EPSG:4326'},
        'source': 'manual',
        'position_kind': 'tree',
        'retrieved_at': '2026-10-01T10:00:00Z'
      }
    };
Map<String, dynamic> row(String version,
        {String analysis = 'same-image', double? lat = 0, double? lon = 30}) =>
    {
      'version_id': version,
      'analysis_id': analysis,
      'created_at': '2026-10-01T10:00:00Z',
      'summary': {
        'species': 'DEMO tree',
        'height_m': 12,
        'environment': lat == null || lon == null
            ? <String, dynamic>{}
            : environment(lat, lon)
      }
    };
Map<String, dynamic> snapshot(double lat, double lon) => {
      'version': 1,
      'kind': 'v4',
      'environment': environment(lat, lon),
      'report': {'species': 'DEMO local', 'height_m': 12}
    };

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  late Directory temp;
  late ContourDrafts journal, cache;
  setUp(() async {
    temp = await Directory.systemTemp.createTemp('arborscan-map-test-');
    journal =
        ContourDrafts(directory: () async => Directory('${temp.path}/journal'));
    cache =
        ContourDrafts(directory: () async => Directory('${temp.path}/cache'));
    SharedPreferences.setMockInitialValues(
        {'arborscan_auth_token': 'token-a', 'arborscan_user_id': ownerA});
  });
  tearDown(() async {
    await temp.delete(recursive: true);
  });
  SurveyMapRepository repository(
      Future<http.Response> Function(http.Request) respond,
      {int legacyDetailLimit = 20}) {
    http.Client client() => MockClient(respond);
    return SurveyMapRepository(
        history: ReportHistoryService(
            journal: journal, cache: cache, clientFactory: client),
        clientFactory: client,
        legacyDetailLimit: legacyDetailLimit,
        timeout: const Duration(seconds: 1));
  }

  http.Response jsonResponse(Object value) =>
      http.Response(jsonEncode(value), 200,
          headers: {'content-type': 'application/json; charset=utf-8'});

  test(
      'Version-only dedup preserves distinct versions and overlapping observations, including zero coordinates',
      () {
    final a = SurveyMapEntry.version(
        {...row('v1'), 'draft_id': 'local-1'}, ownerA,
        server: false);
    final synced = SurveyMapEntry.version(row('v1'), ownerA, server: true);
    final version2 = SurveyMapEntry.version(row('v2'), ownerA, server: true);
    final another = SurveyMapEntry.version(
        row('v3', analysis: 'another-image'), ownerA,
        server: true);
    final noGps =
        SurveyMapEntry.version(row('v4', lat: null), ownerA, server: true);
    final zeroLon = SurveyMapEntry.version(row('v5', lat: 52, lon: 0), ownerA,
        server: true);
    final result =
        mergeSurveyVersions([a, synced, version2, another, noGps, zeroLon]);
    expect(result, hasLength(5));
    expect(result.singleWhere((e) => e.versionId == 'v1').localId, 'local-1');
    expect(result.singleWhere((e) => e.versionId == 'v1').server, isTrue);
    expect(result.singleWhere((e) => e.versionId == 'v4').point, isNull);
    expect(result.singleWhere((e) => e.versionId == 'v5').point!.lon, 0);
    final groups = groupSurveyPoints(result);
    expect(groups['0.0,30.0'], hasLength(3));
    expect(groups, hasLength(2));
    expect(
        SurveyMapEntry.version(row('invalid', lat: 91), ownerA, server: true)
            .point,
        isNull);
    expect(
        SurveyMapEntry.version(row('infinite', lon: double.infinity), ownerA,
                server: true)
            .point,
        isNull);
  });

  test(
      'Local/server versions merge by exact version and metadata remains owner-isolated offline',
      () async {
    await journal.save(
        ownerA,
        'my-local',
        {'version_id': 'v1', 'analysis_id': 'a', 'snapshot': snapshot(0, 30)},
        Uint8List.fromList([1, 2, 3]));
    await journal.save(
        ownerB,
        'other-local',
        {
          'version_id': 'foreign',
          'analysis_id': 'b',
          'snapshot': snapshot(42, 30)
        },
        Uint8List.fromList([4]));
    final prefs = await SharedPreferences.getInstance();
    await prefs.setStringList('arborscan_history', [
      jsonEncode({
        'owner_id': ownerA,
        'analysisId': 'old-a',
        'lat': 53,
        'lon': 0,
        'species': 'DEMO old'
      }),
      jsonEncode({
        'owner_id': ownerB,
        'analysisId': 'old-b',
        'lat': 53,
        'lon': 28,
        'species': 'PRIVATE'
      }),
    ]);
    var offline = false;
    final requests = <http.Request>[];
    final repo = repository((r) async {
      requests.add(r);
      if (offline) throw const SocketException('synthetic offline');
      if (r.url.path.endsWith('/v4/reports')) {
        return jsonResponse({
          'items': [row('v1'), row('v2')],
          'next_offset': null
        });
      }
      return jsonResponse({'items': []});
    });
    final online = await repo.load();
    expect(online.entries.map((e) => e.versionId).whereType<String>().toSet(),
        {'v1', 'v2'});
    expect(online.entries, hasLength(3));
    expect(online.entries.every((e) => e.owner == ownerA), isTrue);
    expect(
        requests.every((r) => r.headers['Authorization'] == 'Bearer token-a'),
        isTrue);
    expect(requests.every((r) => !r.url.queryParameters.containsKey('token')),
        isTrue);
    offline = true;
    final reopened =
        await repository((_) async => throw const SocketException('offline'))
            .load();
    expect(reopened.entries, hasLength(3));
    expect(reopened.notices.join(), contains('недоступны'));
    final detail = await repo
        .detail(reopened.entries.singleWhere((e) => e.versionId == 'v1'));
    expect(detail['photo'], [1, 2, 3]);
    await prefs.setString('arborscan_auth_token', 'token-b');
    await prefs.setString('arborscan_user_id', ownerB);
    CorrectionsService.authChanges.value++;
    final switched = await repo.load();
    expect(switched.entries.map((e) => e.versionId).whereType<String>(),
        ['foreign']);
    expect(switched.entries.any((e) => e.species == 'DEMO old'), isFalse);
    await expectLater(repo.detail(reopened.entries.first),
        throwsA(isA<CorrectionException>()));
  });

  test(
      'Late response after account change never becomes map data or caches into the new owner',
      () async {
    final started = Completer<void>(), delayed = Completer<http.Response>();
    final repo = repository((r) async {
      if (r.url.path.endsWith('/v4/reports')) {
        started.complete();
        return delayed.future;
      }
      return jsonResponse({'items': []});
    });
    final operation = repo.load();
    final expectation =
        expectLater(operation, throwsA(isA<CorrectionException>()));
    await started.future;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('arborscan_auth_token', 'token-b');
    await prefs.setString('arborscan_user_id', ownerB);
    CorrectionsService.authChanges.value++;
    delayed.complete(jsonResponse({
      'items': [row('secret')],
      'next_offset': null
    }));
    await expectation;
    expect(prefs.getString('arborscan_map_index_v1_$ownerB'), isNull);
  });

  test(
      'Old v4 details are bounded, zero latitude restored, and immutable version IDs requested',
      () async {
    final ids = <String>[];
    final repo = repository((r) async {
      if (r.url.path.endsWith('/v4/reports')) {
        return jsonResponse({
          'items': [
            {
              'version_id': 'old-1',
              'analysis_id': 'same',
              'summary': {'species': 'DEMO'}
            },
            {
              'version_id': 'old-2',
              'analysis_id': 'same',
              'summary': {'species': 'DEMO'}
            },
            {
              'version_id': 'old-3',
              'analysis_id': 'same',
              'summary': {'species': 'DEMO'}
            }
          ],
          'next_offset': null
        });
      }
      if (r.url.path.contains('/v4/reports/')) {
        final id = r.url.pathSegments.last;
        ids.add(id);
        return jsonResponse({
          'record': {'version_id': id, 'analysis_id': 'same'},
          'snapshot': {
            'kind': 'legacy',
            'report': {
              'gps': {'lat': 0, 'lon': 30}
            },
            'image': {
              'original_base64': base64Encode([1, 2, 3])
            }
          }
        });
      }
      return jsonResponse({'items': []});
    }, legacyDetailLimit: 2);
    final result = await repo.load();
    expect(ids, containsAll(['old-1', 'old-2']));
    expect(ids, hasLength(2));
    expect(result.positioned, hasLength(2));
    expect(result.notices.join(), contains('первых 2'));
    final focused = await repo.loadVersion('old-3');
    expect(focused.versionId, 'old-3');
    expect(focused.point!.lat, 0);
    expect(ids.last, 'old-3');
  });

  test(
      'Previously opened owner cache is visible when all server endpoints are offline',
      () async {
    await cache.save(
        ownerA,
        'cached-v',
        {'record': row('cached-v'), 'snapshot': snapshot(48, 0)},
        Uint8List.fromList([1, 2, 3]));
    final result =
        await repository((_) async => throw const SocketException('offline'))
            .load();
    expect(result.positioned, hasLength(1));
    expect(result.entries.single.versionId, 'cached-v');
    expect(result.entries.single.cached, isTrue);
  });
}
