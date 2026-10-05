import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

import 'api_config.dart';
import 'corrections_service.dart';
import 'report_history_service.dart';
import 'survey_environment.dart';

Map<String, dynamic> _map(dynamic value) =>
    value is Map ? Map<String, dynamic>.from(value) : <String, dynamic>{};
double? _number(dynamic value) =>
    value is num && value.isFinite ? value.toDouble() : null;
Uint8List? surveyPhoto(dynamic value) {
  if (value is Uint8List) return value;
  if (value is! String || value.isEmpty) return null;
  try {
    return base64Decode(value.split(',').last);
  } catch (_) {
    return null;
  }
}

/// Identifies a saved report version, not a permanent real-world tree.
class SurveyMapEntry {
  final String key, owner, analysisId, species;
  final String? versionId, localId;
  final SurveyPoint? point;
  final DateTime? capturedAt;
  final bool server, cached;
  final Map<String, dynamic> metadata, environment;
  final double? heightM, crownM, trunkM;
  final String speciesStatus;

  const SurveyMapEntry(
      {required this.key,
      required this.owner,
      required this.analysisId,
      required this.species,
      required this.server,
      required this.metadata,
      required this.environment,
      required this.speciesStatus,
      this.versionId,
      this.localId,
      this.point,
      this.capturedAt,
      this.cached = false,
      this.heightM,
      this.crownM,
      this.trunkM});

  factory SurveyMapEntry.version(Map<String, dynamic> row, String owner,
      {required bool server, bool cached = false, String? localId}) {
    final snapshot = _map(row['snapshot']);
    final report = _map(snapshot['report']);
    final summary = _map(row['summary']);
    final environment = _map(snapshot['environment'] ??
        summary['environment'] ??
        report['environment_snapshot']);
    final gps = environment['gps'] ?? report['gps'];
    final measures = _map(report['measurements']);
    final speciesRaw = report['species'] ?? summary['species'];
    final species = speciesRaw is Map
        ? '${speciesRaw['display_name'] ?? speciesRaw['scientific_name'] ?? 'Вид не определён'}'
        : '${speciesRaw ?? 'Вид не определён'}';
    final point = SurveyPoint.fromJson(gps) ??
        SurveyPoint.fromJson({'lat': report['lat'], 'lon': report['lon']});
    final version = row['version_id']?.toString();
    final local = localId ?? row['draft_id']?.toString();
    // A display name alone is not a confirmed taxonomic label.
    final speciesStatus =
        speciesRaw is Map && speciesRaw['status'] == 'confirmed'
            ? 'Подтверждено в этой версии'
            : 'Без подтверждения в этой версии';
    return SurveyMapEntry(
      key: version == null || version.isEmpty
          ? 'local:$local'
          : 'version:$version',
      owner: owner,
      analysisId: '${row['analysis_id'] ?? ''}',
      species: species,
      versionId: version,
      localId: local,
      server: server,
      cached: cached,
      point: point,
      capturedAt: DateTime.tryParse(
          '${snapshot['captured_at'] ?? summary['captured_at'] ?? row['created_at'] ?? ''}'),
      metadata: Map<String, dynamic>.from(row),
      environment: environment,
      speciesStatus: speciesStatus,
      heightM: _number(summary['height_m'] ??
          report['height_m'] ??
          _map(measures['height'])['value_m']),
      crownM: _number(summary['crown_width_m'] ??
          report['crown_width_m'] ??
          _map(measures['crown_width'])['value_m']),
      trunkM: _number(summary['trunk_diameter_m'] ??
          report['trunk_diameter_m'] ??
          _map(measures['trunk_diameter'])['value_m']),
    );
  }

  factory SurveyMapEntry.legacy(Map<String, dynamic> row, String owner,
          {required bool server,
          required String identity,
          bool cached = false}) =>
      SurveyMapEntry(
        key: '${server ? 'v3' : 'legacy-local'}:$identity',
        owner: owner,
        analysisId: '${row['analysis_id'] ?? row['analysisId'] ?? ''}',
        species: '${row['species'] ?? 'Вид не определён'}',
        server: server,
        cached: cached,
        metadata: Map<String, dynamic>.from(row),
        environment: _map(row['environment'] ?? row['environment_snapshot']),
        point: SurveyPoint.fromJson(_map(
                row['environment'] ?? row['environment_snapshot'])['gps']) ??
            SurveyPoint.fromJson(row['gps']) ??
            SurveyPoint.fromJson(row),
        capturedAt: DateTime.tryParse(
            '${row['captured_at'] ?? row['created_at'] ?? row['timestamp'] ?? ''}'),
        heightM: _number(row['height_m'] ?? row['height']),
        crownM: _number(row['crown_width_m'] ?? row['crown']),
        trunkM: _number(row['trunk_diameter_m'] ?? row['trunk']),
        speciesStatus: 'Историческое определение; подтверждение не записано',
      );

  String get storageLabel => server
      ? (cached ? 'Аккаунт · сохранённый список' : 'Аккаунт')
      : 'На устройстве';
  String get formattedDate {
    final date = capturedAt?.toLocal();
    if (date == null) return 'Дата неизвестна';
    return '${date.day.toString().padLeft(2, '0')}.${date.month.toString().padLeft(2, '0')}.${date.year}';
  }

  SurveyMapEntry withLocalFallback(String? id) => SurveyMapEntry(
        key: key,
        owner: owner,
        analysisId: analysisId,
        species: species,
        server: server,
        metadata: metadata,
        environment: environment,
        speciesStatus: speciesStatus,
        versionId: versionId,
        localId: id,
        point: point,
        capturedAt: capturedAt,
        cached: cached,
        heightM: heightM,
        crownM: crownM,
        trunkM: trunkM,
      );
}

List<SurveyMapEntry> mergeSurveyVersions(Iterable<SurveyMapEntry> entries) {
  final values = entries.toList();
  // The automatic preferences row is a fallback summary of the initial
  // analysis, not another saved version. Suppress it only when that exact
  // owner's full initial analysis is available; all real revisions remain.
  final structuredAnalyses = {
    for (final entry in values)
      if (entry.versionId?.isNotEmpty == true &&
          entry.analysisId.isNotEmpty &&
          _map(entry.metadata['snapshot'])['kind'] == 'v4' &&
          _map(entry.metadata['snapshot'])['change_source'] == 'analysis')
        (entry.owner, entry.analysisId)
  };
  final result = <String, SurveyMapEntry>{};
  for (final entry in values) {
    if (entry.key.startsWith('legacy-local:') &&
        entry.metadata.containsKey('environment_snapshot') &&
        entry.metadata['analysisId'] == entry.analysisId &&
        structuredAnalyses.contains((entry.owner, entry.analysisId))) {
      continue;
    }
    final old = result[entry.key];
    if (old == null) {
      result[entry.key] = entry;
      continue;
    }
    if (entry.versionId != null) {
      final preferred = entry.server ? entry : old;
      result[entry.key] =
          preferred.withLocalFallback(entry.localId ?? old.localId);
    }
  }
  return result.values.toList()
    ..sort((a, b) => (b.capturedAt?.millisecondsSinceEpoch ?? 0)
        .compareTo(a.capturedAt?.millisecondsSinceEpoch ?? 0));
}

/// Exact coordinate overlap is a chooser, never a merge of observations.
Map<String, List<SurveyMapEntry>> groupSurveyPoints(
    Iterable<SurveyMapEntry> entries) {
  final groups = <String, List<SurveyMapEntry>>{};
  for (final entry in entries) {
    final point = entry.point;
    if (point != null) (groups[point.identity] ??= []).add(entry);
  }
  return groups;
}

class SurveyMapResult {
  final List<SurveyMapEntry> entries;
  final List<String> notices;
  const SurveyMapResult(this.entries, this.notices);
  List<SurveyMapEntry> get positioned =>
      entries.where((e) => e.point != null).toList();
}

/// Loads account-owned metadata. Photos are fetched only when a card is selected.
/// Cached metadata cannot authorize writes or imply an offline map tile package.
class SurveyMapRepository {
  final ReportHistoryService history;
  final http.Client Function() clientFactory;
  final Duration timeout;
  final int legacyDetailLimit, maxPages;
  SurveyMapRepository(
      {ReportHistoryService? history,
      http.Client Function()? clientFactory,
      this.timeout = const Duration(seconds: 10),
      this.legacyDetailLimit = 20,
      this.maxPages = 20})
      : history = history ?? ReportHistoryService(),
        clientFactory = clientFactory ?? http.Client.new;

  Future<void> _guard(String token, String owner, int epoch) async {
    await history.auth.checkSession(token);
    final currentOwner = await history.auth.owner(token);
    if (owner != currentOwner ||
        epoch != CorrectionsService.authChanges.value) {
      throw const CorrectionException(
          'Аккаунт изменился. Откройте карту заново.');
    }
  }

  Future<Map<String, dynamic>> _v3(String token, String path,
      {Map<String, String>? query}) async {
    final client = clientFactory();
    try {
      final r = await client.get(
          ApiConfig.v3(path).replace(queryParameters: query),
          headers: {'Authorization': 'Bearer $token'}).timeout(timeout);
      if (r.statusCode != 200) {
        throw const CorrectionException('Старая история сейчас недоступна.');
      }
      return _map(jsonDecode(utf8.decode(r.bodyBytes)));
    } finally {
      client.close();
    }
  }

  Future<SurveyMapResult> load(
      {void Function(SurveyMapResult)? onLocal}) async {
    final token = await CorrectionsService.currentToken();
    final epoch = CorrectionsService.authChanges.value;
    if (token.isEmpty) {
      return const SurveyMapResult(
          [], ['Войдите в профиль, чтобы увидеть свои обследования.']);
    }
    final owner = await history.auth.owner(token);
    final prefs = await SharedPreferences.getInstance();
    final notices = <String>[];
    final local = <SurveyMapEntry>[];
    try {
      for (final row in await history.journal.list(owner)) {
        local.add(SurveyMapEntry.version(row, owner, server: false));
      }
    } catch (_) {
      notices.add('Локальный журнал сейчас недоступен.');
    }
    final openedVersions = <SurveyMapEntry>[];
    try {
      for (final cached in await history.cache.list(owner)) {
        final record = _map(cached['record']);
        if (record['version_id'] is String) {
          openedVersions.add(SurveyMapEntry.version(
              {...record, 'snapshot': cached['snapshot']}, owner,
              server: true, cached: true));
        }
      }
    } catch (_) {
      /* Previously opened versions are an optional offline cache. */
    }
    var legacyIndex = 0;
    for (final encoded
        in prefs.getStringList('arborscan_history') ?? <String>[]) {
      try {
        final row = _map(jsonDecode(encoded));
        if (row['owner_id'] == owner) {
          local.add(SurveyMapEntry.legacy(row, owner,
              server: false,
              identity: '${row['analysisId'] ?? 'unknown'}:${legacyIndex++}'));
        }
      } catch (_) {/* One malformed local row must not hide valid records. */}
    }
    final cacheKey = 'arborscan_map_index_v1_$owner';
    Map<String, dynamic> cache = {};
    try {
      cache = _map(jsonDecode(prefs.getString(cacheKey) ?? '{}'));
    } catch (_) {}
    var v4Rows = (cache['v4'] as List? ?? [])
        .whereType<Map>()
        .map((e) => Map<String, dynamic>.from(e))
        .toList();
    var v3Rows = (cache['v3'] as List? ?? [])
        .whereType<Map>()
        .map((e) => Map<String, dynamic>.from(e))
        .toList();
    var cachedV4 = true, cachedV3 = true;
    List<SurveyMapEntry> combined() => mergeSurveyVersions([
          ...local,
          ...openedVersions,
          ...v4Rows.map((r) =>
              SurveyMapEntry.version(r, owner, server: true, cached: cachedV4)),
          for (var i = 0; i < v3Rows.length; i++)
            SurveyMapEntry.legacy(v3Rows[i], owner,
                server: true,
                cached: cachedV3,
                identity: '${v3Rows[i]['analysis_id'] ?? i}'),
        ]);
    await _guard(token, owner, epoch);
    onLocal?.call(SurveyMapResult(
        combined(), ['Загружаем сохранённые версии…', ...notices]));
    try {
      final rows = <Map<String, dynamic>>[];
      int? offset = 0;
      var pages = 0;
      while (offset != null && pages++ < maxPages) {
        final page = await history.list(token, offset).timeout(timeout);
        await _guard(token, owner, epoch);
        rows.addAll((page['items'] as List? ?? [])
            .whereType<Map>()
            .map((e) => Map<String, dynamic>.from(e)));
        final next = page['next_offset'];
        if (next != null && (next is! int || next <= offset)) {
          throw const FormatException('Неверная страница истории.');
        }
        offset = next as int?;
      }
      if (offset != null) {
        notices.add(
            'Показаны первые ${rows.length} версий; остальные доступны в истории.');
      }
      final oldRows = rows
          .where((r) => !_map(r['summary']).containsKey('environment'))
          .toList();
      if (oldRows.length > legacyDetailLimit) {
        notices.add(
            'Координаты старого формата проверены у первых $legacyDetailLimit версий. Остальные доступны в истории.');
      }
      // The previous API did not index coordinates. Discovery is bounded and
      // immutable: it loads specific versions, never recalculates a report.
      for (var start = 0;
          start < oldRows.length && start < legacyDetailLimit;
          start += 4) {
        await Future.wait(oldRows
            .skip(start)
            .take((legacyDetailLimit - start).clamp(0, 4))
            .map((row) async {
          final id = row['version_id'];
          if (id is! String || id.isEmpty) return;
          try {
            final detail = await history.record(token, id).timeout(timeout);
            await _guard(token, owner, epoch);
            if (_map(detail['record'])['version_id'] != id) return;
            final snapshot = _map(detail['snapshot']);
            final report = _map(snapshot['report']);
            final env =
                _map(snapshot['environment'] ?? report['environment_snapshot']);
            if (env.isEmpty) {
              final legacy = SurveyPoint.fromJson(report['gps']) ??
                  SurveyPoint.fromJson(report);
              if (legacy != null) {
                env.addAll({'version': 1, 'gps': legacy.toJson()});
              }
            }
            row['summary'] = {..._map(row['summary']), 'environment': env};
          } catch (_) {
            /* Keep version visible even if detail/photo is offline. */
          }
        }));
      }
      await _guard(token, owner, epoch);
      v4Rows = rows;
      cachedV4 = false;
    } catch (_) {
      await _guard(token, owner, epoch);
      notices.add(
          'Серверные версии недоступны. Показаны доступные локальные данные и сохранённый список.');
    }
    try {
      final page = await _v3(token, '/analyses/my', query: {'limit': '200'});
      await _guard(token, owner, epoch);
      v3Rows = (page['items'] as List? ?? [])
          .whereType<Map>()
          .map((e) => Map<String, dynamic>.from(e))
          .toList();
      cachedV3 = false;
    } catch (_) {
      await _guard(token, owner, epoch);
      if (v3Rows.isNotEmpty) {
        notices.add('Старая история показана из сохранённого списка.');
      }
    }
    await _guard(token, owner, epoch);
    await prefs.setString(
        cacheKey,
        jsonEncode({
          'v4': v4Rows,
          'v3': v3Rows,
          'retrieved_at': DateTime.now().toUtc().toIso8601String()
        }));
    await _guard(token, owner, epoch);
    return SurveyMapResult(combined(), notices);
  }

  Future<Map<String, dynamic>> detail(SurveyMapEntry entry) async {
    final token = await CorrectionsService.currentToken();
    final epoch = CorrectionsService.authChanges.value;
    await _guard(token, entry.owner, epoch);
    if (entry.localId != null) {
      final local = await history.journal.load(entry.owner, entry.localId!);
      await _guard(token, entry.owner, epoch);
      if (local != null &&
          (entry.versionId == null || local['version_id'] == entry.versionId)) {
        return {
          'local_copy': true,
          'record': local,
          'snapshot': local['snapshot'],
          'photo': local['image']
        };
      }
    }
    if (entry.versionId != null && entry.server) {
      final value =
          await history.record(token, entry.versionId!).timeout(timeout);
      await _guard(token, entry.owner, epoch);
      if (_map(value['record'])['version_id'] != entry.versionId) {
        throw const FormatException('Получена другая версия.');
      }
      return {
        ...value,
        'photo': surveyPhoto(
            _map(_map(value['snapshot'])['image'])['original_base64'])
      };
    }
    if (entry.server && entry.analysisId.isNotEmpty) {
      final value = await _v3(
          token, '/analyses/${Uri.encodeComponent(entry.analysisId)}');
      await _guard(token, entry.owner, epoch);
      final raw = _map(value['analysis']);
      return {
        'legacy': raw,
        'photo': surveyPhoto(
            raw['original_image_base64'] ?? raw['annotated_image_base64'])
      };
    }
    return {
      'legacy': entry.metadata,
      'photo': surveyPhoto(entry.metadata['imageBase64'])
    };
  }

  Future<SurveyMapEntry> loadVersion(String versionId) async {
    final token = await CorrectionsService.currentToken();
    final owner = await history.auth.owner(token);
    final epoch = CorrectionsService.authChanges.value;
    final detail = await history.record(token, versionId).timeout(timeout);
    await _guard(token, owner, epoch);
    final record = _map(detail['record']);
    if (record['version_id'] != versionId) {
      throw const FormatException('Получена другая версия.');
    }
    return SurveyMapEntry.version(
        {...record, 'snapshot': detail['snapshot']}, owner,
        server: true, cached: detail['cached'] == true);
  }
}
