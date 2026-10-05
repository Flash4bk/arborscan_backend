import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:google_maps_flutter/google_maps_flutter.dart' as gmaps;
import 'package:url_launcher/url_launcher.dart';

import 'analysis_report_page.dart';
import 'app_navigation.dart';
import 'app_theme.dart';
import 'corrections_service.dart';
import 'location_service.dart';
import 'server_report_page.dart';
import 'survey_environment.dart';
import 'survey_map_repository.dart';

class LatLngFocus {
  final double lat, lon, zoom;
  const LatLngFocus(this.lat, this.lon, {this.zoom = 16});
}

class MapPage extends StatefulWidget {
  final LatLngFocus? initialFocus;
  final String? focusVersionId;
  final SurveyMapRepository? repository;
  const MapPage(
      {super.key, this.initialFocus, this.focusVersionId, this.repository});
  @override
  State<MapPage> createState() => _MapPageState();
}

class _MapPageState extends State<MapPage> {
  late final SurveyMapRepository _repository;
  SurveyMapResult _result = const SurveyMapResult([], []);
  gmaps.GoogleMapController? _controller;
  gmaps.MapType _mapType = gmaps.MapType.normal;
  bool _loading = true, _list = false, _threeD = false, _moving = false;
  String _filter = 'all';
  String? _selectedKey, _error;
  int _epoch = 0;
  int _mapGeneration = 0;
  bool _focusApplied = false;
  // This is an overview camera, never coordinates assigned to an observation.
  static const _overview = gmaps.LatLng(20, 0);

  @override
  void initState() {
    super.initState();
    _repository = widget.repository ?? SurveyMapRepository();
    AppNavigation.mapVisits.addListener(_refresh);
    CorrectionsService.authChanges.addListener(_accountChanged);
    _refresh();
  }

  @override
  void dispose() {
    _epoch++;
    AppNavigation.mapVisits.removeListener(_refresh);
    CorrectionsService.authChanges.removeListener(_accountChanged);
    // GoogleMap owns and disposes its controller when its widget is removed.
    _controller = null;
    super.dispose();
  }

  void _accountChanged() {
    _epoch++;
    setState(() {
      _result = const SurveyMapResult([], []);
      _forgetMapController();
      _selectedKey = null;
      _error = null;
    });
    _refresh();
  }

  Future<void> _refresh() async {
    final epoch = ++_epoch;
    setState(() {
      _loading = true;
      _error = null;
    });
    void accept(SurveyMapResult result) {
      if (!mounted || epoch != _epoch) return;
      setState(() {
        _result = result;
        if (!_mapVisible) _forgetMapController();
      });
    }

    try {
      var result = await _repository.load(onLocal: accept);
      if (widget.focusVersionId != null &&
          !result.positioned.any((e) => e.versionId == widget.focusVersionId)) {
        try {
          final focused = await _repository.loadVersion(widget.focusVersionId!);
          result = SurveyMapResult(
              mergeSurveyVersions([...result.entries, focused]),
              result.notices);
        } catch (_) {/* Keep the remaining account map accessible. */}
      }
      if (!mounted || epoch != _epoch) return;
      accept(result);
      await _applyFocus();
    } catch (_) {
      if (mounted && epoch == _epoch) {
        setState(() => _error =
            'Не удалось обновить карту. Сохранённые данные остаются доступны.');
      }
    } finally {
      if (mounted && epoch == _epoch) setState(() => _loading = false);
    }
  }

  List<SurveyMapEntry> get _visible => _result.entries
      .where((e) =>
          _filter == 'all' ||
          (_filter == 'server' && e.server) ||
          (_filter == 'local' && !e.server))
      .toList();

  bool get _mapVisible =>
      !_list && _visible.any((entry) => entry.point != null);

  void _forgetMapController() {
    _controller = null;
    // Also reject an onMapCreated callback from a previous native map.
    _mapGeneration++;
  }

  Future<void> _applyFocus() async {
    if (_focusApplied) return;
    SurveyMapEntry? entry;
    for (final candidate in _result.entries) {
      if (candidate.versionId == widget.focusVersionId &&
          widget.focusVersionId != null) {
        entry = candidate;
        break;
      }
    }
    if (entry != null) {
      _selectedKey = entry.key;
      if (entry.point == null) {
        _focusApplied = true;
        if (mounted) {
          setState(() {
            _list = true;
            _forgetMapController();
            _error =
                'Нет GPS данных, карта недоступна для этой версии. Откройте отчёт и добавьте точку.';
          });
        }
        return;
      }
      if (_controller == null) return;
      await _move(entry.point!.lat, entry.point!.lon);
      _focusApplied = true;
    } else if (widget.initialFocus != null && _controller != null) {
      final focus = widget.initialFocus!;
      if (SurveyPoint.valid(focus.lat, focus.lon)) {
        await _move(focus.lat, focus.lon, zoom: focus.zoom);
      }
      _focusApplied = true;
    }
  }

  Future<void> _move(double lat, double lon, {double zoom = 16}) async {
    final controller = _controller;
    if (!mounted ||
        !_mapVisible ||
        controller == null ||
        !SurveyPoint.valid(lat, lon)) {
      return;
    }
    try {
      await controller.animateCamera(gmaps.CameraUpdate.newCameraPosition(
          gmaps.CameraPosition(
              target: gmaps.LatLng(lat, lon),
              zoom: zoom,
              tilt: _threeD ? 60 : 0,
              bearing: _threeD ? 35 : 0)));
    } catch (_) {
      if (mounted && _mapVisible && identical(_controller, controller)) {
        setState(() => _error =
            'Подложка карты недоступна. Записи можно открыть в списке.');
      }
    }
  }

  Future<void> _locate() async {
    if (_moving) return;
    setState(() => _moving = true);
    final result = await LocationService.getCurrentPositionDetailed();
    if (!mounted) return;
    setState(() => _moving = false);
    final point = result.position;
    if (point == null) {
      _message(result.message);
      return;
    }
    await _move(point.latitude, point.longitude, zoom: 14);
    if (!mounted) return;
    _message(result.status == 'last_known'
        ? 'Обзор по последнему известному положению от ${point.timestamp.toLocal()}. Координаты обследований не изменены.'
        : 'Камера карты перемещена к устройству. Координаты обследований не изменены.');
  }

  void _message(String message) => ScaffoldMessenger.of(context)
      .showSnackBar(SnackBar(content: Text(message)));
  gmaps.LatLng get _target {
    if (widget.initialFocus case final focus?
        when SurveyPoint.valid(focus.lat, focus.lon)) {
      return gmaps.LatLng(focus.lat, focus.lon);
    }
    final matching = _result.positioned
        .where((e) => e.versionId == widget.focusVersionId)
        .firstOrNull;
    final point = matching?.point ?? _result.positioned.firstOrNull?.point;
    return point == null ? _overview : gmaps.LatLng(point.lat, point.lon);
  }

  Future<void> _choose(List<SurveyMapEntry> entries) async {
    final epoch = CorrectionsService.authChanges.value;
    SurveyMapEntry? entry = entries.first;
    if (entries.length > 1) {
      entry = await showModalBottomSheet<SurveyMapEntry>(
          context: context,
          isScrollControlled: true,
          builder: (context) => _sessionView(
              epoch,
              SafeArea(
                  child: SizedBox(
                      height: MediaQuery.sizeOf(context).height * .65,
                      child: ListView(
                          padding: const EdgeInsets.all(16),
                          children: [
                            Text('В этой точке: ${entries.length}',
                                style: Theme.of(context).textTheme.titleLarge),
                            const Text(
                                'Выберите обследование или версию. Совпадение координат не означает одно дерево.'),
                            for (final e in entries)
                              ListTile(
                                  title: Text(e.species),
                                  subtitle: Text(
                                      '${e.formattedDate} · ${e.storageLabel}\n${e.versionId == null ? 'Старая запись' : 'Версия ${e.versionId!.substring(0, e.versionId!.length.clamp(0, 8))}'}'),
                                  trailing: const Icon(Icons.chevron_right),
                                  onTap: () => Navigator.pop(context, e)),
                          ])))));
    }
    if (entry == null ||
        !mounted ||
        epoch != CorrectionsService.authChanges.value) {
      return;
    }
    setState(() => _selectedKey = entry!.key);
    if (entry.point != null) await _move(entry.point!.lat, entry.point!.lon);
    if (!mounted || epoch != CorrectionsService.authChanges.value) return;
    await showModalBottomSheet<void>(
        context: context,
        isScrollControlled: true,
        builder: (context) => ValueListenableBuilder<int>(
            valueListenable: CorrectionsService.authChanges,
            builder: (context, current, _) => current == epoch
                ? _SurveyCard(
                    entry: entry!, repository: _repository, onOpen: _openReport)
                : const SafeArea(
                    child: Padding(
                        padding: EdgeInsets.all(24),
                        child:
                            Text('Аккаунт изменился. Закройте карточку.')))));
  }

  Widget _sessionView(int epoch, Widget child) => ValueListenableBuilder<int>(
      valueListenable: CorrectionsService.authChanges,
      builder: (_, current, __) => current == epoch
          ? child
          : const SafeArea(
              child: Padding(
                  padding: EdgeInsets.all(24),
                  child: Text('Аккаунт изменился. Закройте карточку.'))));

  Future<void> _openReport(
      SurveyMapEntry entry, Map<String, dynamic>? detail) async {
    final token = await CorrectionsService.currentToken();
    await _repository.history.auth.checkSession(token);
    if (await _repository.history.auth.owner(token) != entry.owner ||
        !mounted) {
      return;
    }
    Navigator.pop(context); // Close only the current map card.
    final epoch = CorrectionsService.authChanges.value;
    Widget page;
    if (entry.versionId != null) {
      final local = detail?['local_copy'] == true || !entry.server;
      page = ServerReportPage(
          versionId: local ? null : entry.versionId,
          localId: local ? entry.localId : null,
          expectedLocalVersionId: local ? entry.versionId : null);
    } else {
      final raw = surveyMap(detail?['legacy']);
      page = raw.containsKey('height_m') || raw.containsKey('gps')
          ? AnalysisReportPageV2.fromRawResult(
              raw: raw, annotatedImageBytes: detail?['photo'] as Uint8List?)
          : AnalysisReportPageV2.fromHistory(
              species: entry.species,
              heightM: entry.heightM,
              crownWidthM: entry.crownM,
              trunkDiameterM: entry.trunkM,
              lat: entry.point?.lat,
              lon: entry.point?.lon,
              timestamp: entry.capturedAt,
              imageBase64: entry.metadata['imageBase64']?.toString());
    }
    await Navigator.push(
        context,
        MaterialPageRoute(
            builder: (_) => ValueListenableBuilder<int>(
                valueListenable: CorrectionsService.authChanges,
                builder: (_, current, __) => current == epoch
                    ? page
                    : const Scaffold(
                        body: Center(child: Text('Аккаунт изменился.'))))));
    if (mounted) _refresh();
  }

  @override
  Widget build(BuildContext context) {
    final positioned = _visible.where((e) => e.point != null).toList();
    final groups = groupSurveyPoints(positioned);
    final mapGeneration = _mapGeneration;
    return Scaffold(
      appBar: AppBar(title: const Text('Карта обследований'), actions: [
        IconButton(
            tooltip: _list ? 'Показать карту' : 'Показать список',
            onPressed: () => setState(() {
                  _list = !_list;
                  if (_list) _forgetMapController();
                }),
            icon: Icon(_list ? Icons.map_outlined : Icons.list_alt)),
        IconButton(
            tooltip: 'Обновить карту',
            onPressed: _loading ? null : _refresh,
            icon: const Icon(Icons.refresh)),
      ]),
      body: SafeArea(
          child: Column(children: [
        if (_loading) const LinearProgressIndicator(minHeight: 2),
        ConstrainedBox(
            constraints: BoxConstraints(
                maxHeight: MediaQuery.sizeOf(context).height * .38),
            child: SingleChildScrollView(
                child: Padding(
                    padding: const EdgeInsets.fromLTRB(16, 8, 16, 8),
                    child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                              '${positioned.length} с координатами · ${_visible.length - positioned.length} без точки',
                              style: Theme.of(context).textTheme.titleSmall),
                          const SizedBox(height: 8),
                          Wrap(spacing: 8, runSpacing: 4, children: [
                            for (final option in const [
                              ('all', 'Все'),
                              ('server', 'Аккаунт'),
                              ('local', 'На устройстве')
                            ])
                              ChoiceChip(
                                  label: Text(option.$2),
                                  selected: _filter == option.$1,
                                  onSelected: (_) => setState(() {
                                        _filter = option.$1;
                                        if (!_mapVisible) {
                                          _forgetMapController();
                                        }
                                      })),
                          ]),
                          if (_error != null)
                            Text(_error!,
                                style: const TextStyle(color: AppTheme.danger)),
                          if (_result.notices.isNotEmpty)
                            ExpansionTile(
                                tilePadding: EdgeInsets.zero,
                                title: Text(_loading
                                    ? 'Загрузка списка…'
                                    : 'Доступность данных'),
                                children: [
                                  for (final notice in _result.notices)
                                    Padding(
                                        padding:
                                            const EdgeInsets.only(bottom: 8),
                                        child: Text(notice))
                                ]),
                        ])))),
        Expanded(
            child: _list || positioned.isEmpty
                ? _entriesList()
                : Stack(children: [
                    gmaps.GoogleMap(
                      key: ValueKey(mapGeneration),
                      initialCameraPosition: gmaps.CameraPosition(
                          target: _target,
                          zoom: widget.initialFocus?.zoom ?? 13),
                      mapType: _mapType,
                      myLocationButtonEnabled: false,
                      myLocationEnabled: false,
                      mapToolbarEnabled: false,
                      zoomControlsEnabled: false,
                      compassEnabled: true,
                      padding: const EdgeInsets.only(top: 52, bottom: 65),
                      onMapCreated: (controller) {
                        if (!mounted ||
                            !_mapVisible ||
                            mapGeneration != _mapGeneration) {
                          return;
                        }
                        _controller = controller;
                        _applyFocus();
                      },
                      markers: groups.entries
                          .map((group) => gmaps.Marker(
                                markerId: gmaps.MarkerId(group.key),
                                position: gmaps.LatLng(
                                    group.value.first.point!.lat,
                                    group.value.first.point!.lon),
                                // The icon distinguishes selection only; it never encodes risk.
                                icon:
                                    gmaps.BitmapDescriptor.defaultMarkerWithHue(
                                        group.value.any(
                                                (e) => e.key == _selectedKey)
                                            ? gmaps.BitmapDescriptor.hueOrange
                                            : gmaps.BitmapDescriptor.hueGreen),
                                infoWindow: gmaps.InfoWindow(
                                    title: group.value.length > 1
                                        ? '${group.value.length} обследования / версии'
                                        : group.value.first.species),
                                onTap: () => _choose(group.value),
                              ))
                          .toSet(),
                    ),
                    Positioned(
                        left: 12,
                        right: 12,
                        top: 8,
                        child: Wrap(spacing: 8, runSpacing: 6, children: [
                          IconButton.filledTonal(
                              tooltip: 'Переместить обзор к устройству',
                              onPressed: _moving ? null : _locate,
                              icon: const Icon(Icons.my_location)),
                          IconButton.filledTonal(
                              tooltip: _mapType == gmaps.MapType.normal
                                  ? 'Спутниковый слой'
                                  : 'Обычная карта',
                              onPressed: () => setState(() => _mapType =
                                  _mapType == gmaps.MapType.normal
                                      ? gmaps.MapType.hybrid
                                      : gmaps.MapType.normal),
                              icon: const Icon(Icons.layers_outlined)),
                          IconButton.filledTonal(
                              tooltip: _threeD
                                  ? 'Плоский обзор 2D'
                                  : 'Наклонный обзор 3D',
                              onPressed: () {
                                setState(() => _threeD = !_threeD);
                                _move(_target.latitude, _target.longitude);
                              },
                              icon: Icon(_threeD
                                  ? Icons.map_outlined
                                  : Icons.threed_rotation)),
                        ])),
                    Positioned(
                        left: 12,
                        right: 12,
                        bottom: 8,
                        child: Material(
                            color: AppTheme.surface,
                            borderRadius: BorderRadius.circular(16),
                            child: Padding(
                                padding: const EdgeInsets.all(10),
                                child: Text(
                                    'Подложке Google Maps нужна сеть. Если она не загрузилась, откройте список.',
                                    style: Theme.of(context)
                                        .textTheme
                                        .bodySmall)))),
                  ])),
      ])),
    );
  }

  Widget _entriesList() =>
      ListView(padding: const EdgeInsets.all(16), children: [
        if (_visible.isEmpty)
          const Padding(
              padding: EdgeInsets.symmetric(vertical: 24),
              child: Column(children: [
                Icon(Icons.location_on_outlined, size: 48),
                SizedBox(height: 12),
                Text('Пока нет сохранённых обследований'),
                Text(
                    'Добавьте точку во время исследования или в новой версии отчёта. GPS не обязателен.'),
              ])),
        if (_visible.isNotEmpty && !_list)
          const Padding(
              padding: EdgeInsets.only(bottom: 16),
              child: Text(
                  'Нет GPS данных, карта недоступна для этих записей. Откройте отчёт, чтобы добавить точку.')),
        for (final entry in _visible)
          Card(
              child: ListTile(
            selected: entry.key == _selectedKey,
            leading: Icon(entry.point == null
                ? Icons.location_off_outlined
                : Icons.location_on_outlined),
            title: Text(entry.species),
            subtitle: Text(
                '${entry.formattedDate} · ${entry.storageLabel}\n${entry.point?.coordinates ?? 'Нет координат'}'),
            trailing: const Icon(Icons.chevron_right),
            onTap: () => _choose([entry]),
          )),
      ]);
}

class _SurveyCard extends StatefulWidget {
  final SurveyMapEntry entry;
  final SurveyMapRepository repository;
  final Future<void> Function(SurveyMapEntry, Map<String, dynamic>?) onOpen;
  const _SurveyCard(
      {required this.entry, required this.repository, required this.onOpen});
  @override
  State<_SurveyCard> createState() => _SurveyCardState();
}

class _SurveyCardState extends State<_SurveyCard> {
  Map<String, dynamic>? _detail;
  String? _error;
  bool _busy = true;
  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    try {
      final data = await widget.repository.detail(widget.entry);
      if (mounted) setState(() => _detail = data);
    } catch (_) {
      if (mounted) {
        setState(() => _error =
            'Фото или полный отчёт сейчас недоступны. Сохранённые сведения показаны ниже.');
      }
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _external(Uri uri,
      {LaunchMode mode = LaunchMode.externalApplication}) async {
    try {
      if (!await launchUrl(uri, mode: mode)) throw StateError('unavailable');
    } catch (_) {
      if (mounted) setState(() => _error = 'Не удалось открыть внешнюю карту.');
    }
  }

  Future<void> _streetView(SurveyPoint point) async {
    final native = Uri.parse(
        'google.streetview:cbll=${point.lat},${point.lon}&cbp=0,0,0,0,0');
    try {
      if (await canLaunchUrl(native) &&
          await launchUrl(native, mode: LaunchMode.externalApplication)) {
        return;
      }
    } catch (_) {/* Fall through to the existing Google Maps web view. */}
    await _external(Uri.parse(
        'https://www.google.com/maps?layer=c&cbll=${point.lat},${point.lon}'));
  }

  @override
  Widget build(BuildContext context) {
    final entry = widget.entry.versionId != null && _detail?['snapshot'] != null
        ? SurveyMapEntry.version({
            ...widget.entry.metadata,
            ...surveyMap(_detail?['record']),
            'snapshot': _detail!['snapshot']
          }, widget.entry.owner,
            server: widget.entry.server,
            cached: widget.entry.cached,
            localId: widget.entry.localId)
        : widget.entry;
    final point = entry.point;
    final photo = _detail?['photo'] as Uint8List?;
    return DraggableScrollableSheet(
        expand: false,
        initialChildSize: .58,
        minChildSize: .35,
        maxChildSize: .94,
        builder: (context, scroll) => SafeArea(
                child: ListView(
                    controller: scroll,
                    padding: const EdgeInsets.fromLTRB(20, 0, 20, 24),
                    children: [
                  if (_busy) const LinearProgressIndicator(minHeight: 2),
                  Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
                    if (photo != null)
                      Padding(
                          padding: const EdgeInsets.only(right: 12),
                          child: ClipRRect(
                              borderRadius: BorderRadius.circular(14),
                              child: Image.memory(photo,
                                  width: 88,
                                  height: 96,
                                  fit: BoxFit.cover,
                                  errorBuilder: (_, __, ___) => const SizedBox(
                                      width: 88,
                                      height: 96,
                                      child: Icon(Icons
                                          .image_not_supported_outlined))))),
                    Expanded(
                        child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                          Text(entry.species,
                              style: Theme.of(context).textTheme.titleLarge),
                          Text(entry.speciesStatus,
                              style: Theme.of(context).textTheme.bodySmall),
                          const SizedBox(height: 6),
                          Text(
                              '${entry.formattedDate} · ${entry.storageLabel}'),
                        ])),
                  ]),
                  const SizedBox(height: 14),
                  Wrap(spacing: 8, runSpacing: 8, children: [
                    if (entry.heightM != null)
                      Ui.badge(
                          text: 'Высота ${entry.heightM!.toStringAsFixed(2)} м',
                          color: AppTheme.primary),
                    if (entry.crownM != null)
                      Ui.badge(
                          text: 'Крона ${entry.crownM!.toStringAsFixed(2)} м',
                          color: AppTheme.primary),
                    if (entry.trunkM != null)
                      Ui.badge(
                          text:
                              'Ствол ${(entry.trunkM! * 100).toStringAsFixed(1)} см',
                          color: AppTheme.primary),
                  ]),
                  const SizedBox(height: 14),
                  Text(
                      point?.coordinates ?? 'Нет GPS данных, карта недоступна'),
                  if (point != null)
                    Text(point.label,
                        style: const TextStyle(color: AppTheme.muted)),
                  if (_error != null)
                    Padding(
                        padding: const EdgeInsets.symmetric(vertical: 8),
                        child: Text(_error!,
                            style: const TextStyle(color: AppTheme.warning))),
                  const SizedBox(height: 14),
                  FilledButton.icon(
                      onPressed: () async {
                        try {
                          await widget.onOpen(entry, _detail);
                        } catch (_) {
                          if (mounted) {
                            setState(() => _error =
                                'Не удалось открыть отчёт. Обновите список и повторите.');
                          }
                        }
                      },
                      icon: const Icon(Icons.description_outlined),
                      label: Text(point == null
                          ? 'Открыть отчёт / добавить точку'
                          : 'Открыть эту версию')),
                  ExpansionTile(
                      title: const Text('Источники и условия'),
                      children: [
                        if (point != null) ...[
                          Text(
                              'Система координат: WGS84. ${point.positionKind == 'tree' ? 'Точка дерева' : 'Положение камеры или устройства'}. '
                              '${point.accuracyM == null ? 'Точность не записана.' : 'Записанная точность: ${point.accuracyM} м.'}'),
                          if (point.isLastKnown)
                            const Text(
                                'Использовано последнее известное положение.'),
                          if (point.retrievedAt.isNotEmpty)
                            Text('Координаты получены: ${point.retrievedAt}'),
                        ],
                        if (entry.environment['weather'] != null)
                          const Text(
                              'В этой версии есть снимок погоды. Параметры и дата доступны в отчёте.'),
                        if (entry.environment['soil'] != null)
                          const Text(
                              'В этой версии есть снимок почвенных данных. Параметры и глубина доступны в отчёте.'),
                        if (entry.environment['weather'] == null &&
                            entry.environment['soil'] == null)
                          const Text(
                              'Условия среды в этой версии не записаны.'),
                        if (entry.metadata['riskIndex'] != null ||
                            entry.metadata['risk_index'] != null)
                          const Text(
                              'Старая запись содержит историческую эвристику риска. Она не подтверждает безопасность дерева.'),
                        if (entry.versionId != null)
                          SelectableText('Версия: ${entry.versionId}'),
                      ]),
                  if (point != null)
                    Wrap(spacing: 8, runSpacing: 8, children: [
                      OutlinedButton.icon(
                          onPressed: () => _streetView(point),
                          icon: const Icon(Icons.threesixty),
                          label: const Text('Street View')),
                      OutlinedButton.icon(
                          onPressed: () => _external(
                              Uri.parse(
                                  'https://www.google.com/maps?layer=c&cbll=${point.lat},${point.lon}'),
                              mode: LaunchMode.inAppBrowserView),
                          icon: const Icon(Icons.public),
                          label: const Text('В браузере')),
                      OutlinedButton.icon(
                          onPressed: () => _external(Uri.parse(
                              'https://www.google.com/maps/dir/?api=1&destination=${point.lat},${point.lon}')),
                          icon: const Icon(Icons.directions_outlined),
                          label: const Text('Маршрут')),
                    ]),
                ])));
  }
}
