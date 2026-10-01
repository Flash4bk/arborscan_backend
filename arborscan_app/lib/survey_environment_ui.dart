import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:geolocator/geolocator.dart';
import 'package:google_maps_flutter/google_maps_flutter.dart';
import 'corrections_service.dart';
import 'location_service.dart';
import 'survey_environment.dart';

/// The saved snapshot is rendered without calling a location or weather service.
class EnvironmentSummary extends StatelessWidget {
  final Map<String, dynamic> snapshot;
  const EnvironmentSummary({super.key, required this.snapshot});
  @override
  Widget build(BuildContext context) {
    final point = SurveyPoint.fromJson(snapshot['gps']);
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Text(point?.coordinates ?? 'Место не указано',
          style: Theme.of(context).textTheme.titleMedium),
      if (point != null) ...[
        Text(point.label),
        Text(point.positionKind == 'tree'
            ? 'Отмечено положение дерева'
            : 'Это положение камеры / устройства, не точная точка дерева'),
        if (point.accuracyM != null)
          Text(
              'Точность устройства: ±${point.accuracyM!.toStringAsFixed(0)} м'),
        if (point.isLastKnown)
          const Text('Последнее известное положение, подтверждено при выборе'),
        if (point.isApproximate == true)
          const Text('Приблизительная геолокация'),
        if (point.observedAt != null)
          Text('Время положения: ${point.observedAt}'),
        if (point.capturedLocal != null)
          Text('Время EXIF: ${point.capturedLocal} (часовой пояс неизвестен)'),
      ],
      for (final key in ['weather', 'soil'])
        if (snapshot[key] is Map)
          _Conditions(keyName: key, data: surveyMap(snapshot[key])),
    ]);
  }
}

class _Conditions extends StatelessWidget {
  final String keyName;
  final Map<String, dynamic> data;
  const _Conditions({required this.keyName, required this.data});
  @override
  Widget build(BuildContext context) {
    final weather = keyName == 'weather';
    final value = surveyMap(data['value']);
    final fields = <String, (String, String)>{
      'temperature_c': ('Температура', '°C'),
      'wind_speed_m_s': ('Ветер', 'м/с'),
      'wind_gust_m_s': ('Порывы', 'м/с'),
      'wind_direction_deg': ('Направление ветра', '°'),
      'pressure_hpa': ('Давление', 'гПа'),
      'relative_humidity_pct': ('Влажность', '%'),
    };
    const names = {
      'clay': 'Глина',
      'sand': 'Песок',
      'silt': 'Ил',
      'soc': 'Органический углерод',
      'phh2o': 'pH (вода)'
    };
    return ExpansionTile(
        tilePadding: EdgeInsets.zero,
        title: Text(weather ? 'Погода' : 'Почва'),
        subtitle: Text(
            '${data['source'] ?? 'Источник неизвестен'} · ${data['status'] == 'ok' ? 'снимок получен' : environmentReason(data['reason'])}'),
        children: [
          Align(
              alignment: Alignment.centerLeft,
              child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    if (weather) ...[
                      const Text(
                          'Текущая погода на момент запроса. Это не архив погоды на дату съёмки.'),
                      for (final entry in fields.entries)
                        Text(
                            '${entry.value.$1}: ${value[entry.key] ?? 'нет данных'}${value[entry.key] == null ? '' : ' ${entry.value.$2}'}'),
                    ] else ...[
                      const Text(
                          'Оценка модели сетки, не измерение у корней дерева.'),
                      if (value['properties'] is List)
                        for (final raw in value['properties'])
                          Text(
                              '${names[surveyMap(raw)['name']] ?? surveyMap(raw)['name']}: ${surveyMap(raw)['value'] ?? 'нет данных'} ${surveyMap(raw)['unit'] ?? ''} · слой ${(surveyMap(raw)['depth_cm'] as List?)?.join('–') ?? '?'} см'),
                      Text(
                          'Разрешение: ${data['resolution_m'] ?? 'не указано'} м · версия ${data['dataset_version'] ?? 'не указана'}'),
                      const Text(
                          'Неопределённость отдельных свойств может быть недоступна.'),
                    ],
                    Text(
                        'Данные на: ${data['data_at'] ?? 'дата не указана источником'}'),
                    Text(
                        'Получено: ${data['retrieved_at'] ?? 'не указано'}${data['cached'] == true ? ' · из кеша' : ''}'),
                    if (data['request_point'] is Map)
                      Text(
                          'Точка запроса: ${surveyMap(data['request_point'])['lat']}, ${surveyMap(data['request_point'])['lon']}'),
                    if (data['attribution'] != null)
                      Text('${data['attribution']}'),
                    const SizedBox(height: 12),
                  ]))
        ]);
  }
}

class SurveyEnvironmentEditor extends StatefulWidget {
  final SurveyEnvironmentController controller;
  final Uint8List? original;
  final bool enabled;
  const SurveyEnvironmentEditor(
      {super.key,
      required this.controller,
      this.original,
      this.enabled = true});
  @override
  State<SurveyEnvironmentEditor> createState() =>
      _SurveyEnvironmentEditorState();
}

class _SurveyEnvironmentEditorState extends State<SurveyEnvironmentEditor> {
  bool _locating = false;
  String? _message;
  Future<void> _device() async {
    if (_locating) return;
    setState(() {
      _locating = true;
      _message = null;
    });
    final generation = CorrectionsService.authChanges.value;
    try {
      final result = await LocationService.getCurrentPositionDetailed();
      if (!mounted || generation != CorrectionsService.authChanges.value) {
        return;
      }
      final p = result.position;
      if (p == null) {
        setState(() => _message = result.message);
        return;
      }
      bool? approximate;
      try {
        approximate = await Geolocator.getLocationAccuracy() ==
            LocationAccuracyStatus.reduced;
      } catch (_) {}
      if (!mounted || generation != CorrectionsService.authChanges.value) {
        return;
      }
      final point = SurveyPoint(
          lat: p.latitude,
          lon: p.longitude,
          source: 'device',
          retrievedAt: DateTime.now().toUtc().toIso8601String(),
          observedAt: p.timestamp.toUtc().toIso8601String(),
          accuracyM: p.accuracy.isFinite && p.accuracy >= 0 ? p.accuracy : null,
          isLastKnown: result.status == 'last_known',
          isApproximate: approximate);
      if (!SurveyPoint.valid(point.lat, point.lon)) {
        throw const FormatException(
            'Устройство вернуло некорректные координаты.');
      }
      final confirmed = await showDialog<bool>(
          context: context,
          builder: (c) => AlertDialog(
                  title: const Text('Использовать положение устройства?'),
                  content: SingleChildScrollView(
                      child: Column(
                          mainAxisSize: MainAxisSize.min,
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                        const Text(
                            'Для старого фото текущая точка может не совпадать с местом съёмки. Положение телефона не задаёт точную точку дерева.'),
                        Text(point.coordinates),
                        Text('Время: ${point.observedAt}'),
                        Text(
                            'Возраст положения: ${DateTime.now().toUtc().difference(p.timestamp.toUtc()).inMinutes} мин'),
                        Text(
                            'Точность: ${point.accuracyM == null ? 'не указана' : '±${point.accuracyM!.toStringAsFixed(0)} м'}'),
                        if (point.isLastKnown)
                          const Text(
                              'GPS не успел: предлагается последнее известное положение.'),
                        if (point.isApproximate == true)
                          const Text(
                              'Android выдал приблизительное положение.'),
                      ])),
                  actions: [
                    TextButton(
                        onPressed: () => Navigator.pop(c, false),
                        child: const Text('Отмена')),
                    FilledButton(
                        onPressed: () => Navigator.pop(c, true),
                        child: const Text('Использовать эту точку'))
                  ]));
      if (mounted &&
          confirmed == true &&
          generation == CorrectionsService.authChanges.value) {
        widget.controller.setPoint(point);
      }
    } catch (e) {
      if (mounted) setState(() => _message = '$e');
    } finally {
      if (mounted) setState(() => _locating = false);
    }
  }

  Future<void> _manual() async {
    final generation = CorrectionsService.authChanges.value;
    final point = await Navigator.push<SurveyPoint>(
        context,
        MaterialPageRoute(
            builder: (_) =>
                SurveyPointPicker(initial: widget.controller.point)));
    if (mounted &&
        point != null &&
        generation == CorrectionsService.authChanges.value) {
      widget.controller.setPoint(point);
    }
  }

  @override
  Widget build(BuildContext context) => ListenableBuilder(
      listenable: widget.controller,
      builder: (context, _) {
        final c = widget.controller;
        final active = widget.enabled && !_locating && !c.busy;
        return Card(
            child: ExpansionTile(
                leading: const Icon(Icons.location_on_outlined),
                title: const Text('Место и условия'),
                subtitle:
                    Text(c.point?.coordinates ?? 'Добавить к обследованию'),
                childrenPadding: const EdgeInsets.all(16),
                children: [
              EnvironmentSummary(snapshot: c.snapshot),
              Wrap(spacing: 8, runSpacing: 8, children: [
                OutlinedButton.icon(
                    onPressed: active ? _manual : null,
                    icon: const Icon(Icons.map_outlined),
                    label: const Text('Указать дерево')),
                OutlinedButton.icon(
                    onPressed: active ? _device : null,
                    icon: const Icon(Icons.my_location),
                    label: Text(_locating ? 'Поиск GPS…' : 'GPS устройства')),
                if (widget.original != null)
                  TextButton(
                      onPressed: active
                          ? () {
                              final point =
                                  surveyPointFromExif(widget.original!);
                              if (point == null) {
                                setState(() => _message =
                                    'В оригинале нет доступных GPS-данных JPEG EXIF. Можно указать точку вручную.');
                              } else {
                                c.setPoint(point);
                                setState(() => _message = null);
                              }
                            }
                          : null,
                      child: const Text('Взять из EXIF')),
                if (c.point != null)
                  TextButton(
                      onPressed: active ? () => c.setPoint(null) : null,
                      child: const Text('Убрать место')),
              ]),
              if (c.point != null)
                OutlinedButton.icon(
                    onPressed: active
                        ? () async {
                            try {
                              await c.refresh(
                                  await CorrectionsService.currentToken());
                            } catch (e) {
                              if (mounted) setState(() => _message = '$e');
                            }
                          }
                        : null,
                    icon: c.busy
                        ? const SizedBox(
                            width: 18,
                            height: 18,
                            child: CircularProgressIndicator(strokeWidth: 2))
                        : const Icon(Icons.cloud_outlined),
                    label: Text(c.busy
                        ? 'Получение условий…'
                        : 'Получить текущие условия')),
              if (_message != null) Text(_message!),
              if (c.error != null) Text(c.error!),
              const Text(
                  'Условия сохраняются снимком. Получение погоды и почвы необязательно для анализа.'),
            ]));
      });
}

class SurveyPointPicker extends StatefulWidget {
  final SurveyPoint? initial;
  const SurveyPointPicker({super.key, this.initial});
  @override
  State<SurveyPointPicker> createState() => _SurveyPointPickerState();
}

class _SurveyPointPickerState extends State<SurveyPointPicker> {
  late final _lat =
      TextEditingController(text: widget.initial?.lat.toString() ?? '');
  late final _lon =
      TextEditingController(text: widget.initial?.lon.toString() ?? '');
  LatLng? _selected;
  String? _error;
  @override
  void initState() {
    super.initState();
    if (widget.initial != null) {
      _selected = LatLng(widget.initial!.lat, widget.initial!.lon);
    }
  }

  @override
  void dispose() {
    _lat.dispose();
    _lon.dispose();
    super.dispose();
  }

  void _save() {
    final lat = double.tryParse(_lat.text.replaceAll(',', '.')),
        lon = double.tryParse(_lon.text.replaceAll(',', '.'));
    if (lat == null || lon == null || !SurveyPoint.valid(lat, lon)) {
      setState(() =>
          _error = 'Введите широту от −90 до 90 и долготу от −180 до 180.');
      return;
    }
    Navigator.pop(
        context,
        SurveyPoint(
            lat: lat,
            lon: lon,
            source: 'manual',
            positionKind: 'tree',
            retrievedAt: DateTime.now().toUtc().toIso8601String()));
  }

  @override
  Widget build(BuildContext context) => Scaffold(
      appBar: AppBar(title: const Text('Точка дерева')),
      body: ListView(padding: const EdgeInsets.all(16), children: [
        const Text(
            'Нажмите на карту или введите координаты EPSG:4326. Центр карты не становится точкой автоматически.'),
        const SizedBox(height: 8),
        SizedBox(
            height: 300,
            child: ClipRRect(
                borderRadius: BorderRadius.circular(16),
                child: GoogleMap(
                    initialCameraPosition: CameraPosition(
                        target: _selected ?? const LatLng(0, 0),
                        zoom: _selected == null ? 1 : 17),
                    myLocationButtonEnabled: false,
                    zoomControlsEnabled: true,
                    mapToolbarEnabled: false,
                    markers: {
                      if (_selected != null)
                        Marker(
                            markerId: const MarkerId('selected'),
                            position: _selected!)
                    },
                    onTap: (point) => setState(() {
                          _selected = point;
                          _lat.text = point.latitude.toString();
                          _lon.text = point.longitude.toString();
                          _error = null;
                        })))),
        const Text(
            'Google Maps. Без сети подложка может не загрузиться; ввод координат доступен.'),
        TextField(
            controller: _lat,
            decoration: const InputDecoration(labelText: 'Широта'),
            keyboardType: const TextInputType.numberWithOptions(
                decimal: true, signed: true)),
        const SizedBox(height: 8),
        TextField(
            controller: _lon,
            decoration: const InputDecoration(labelText: 'Долгота'),
            keyboardType: const TextInputType.numberWithOptions(
                decimal: true, signed: true)),
        if (_error != null) Text(_error!),
        const SizedBox(height: 16),
        FilledButton(
            onPressed: _save, child: const Text('Выбрать точку дерева')),
      ]));
}
