import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:google_maps_flutter/google_maps_flutter.dart' as gmaps;
import 'package:shared_preferences/shared_preferences.dart';
import 'package:geolocator/geolocator.dart';
import 'package:url_launcher/url_launcher.dart';

import 'app_theme.dart';
import 'location_service.dart';


String _normalizeAddressRu(String? address) {
  if (address == null || address.trim().isEmpty) return '';
  final replacements = <String, String>{
    'Інтэрнат': 'Интернат',
    'вуліца': 'улица',
    'вул.': 'ул.',
    'Машынабудаўнікоў': 'Машиностроителей',
    'Аўтазаводскі': 'Автозаводский',
    'раён': 'район',
    'Пасёлак': 'Посёлок',
    'Заводскі': 'Заводской',
    'Мінск': 'Минск',
  };

  var out = address;
  replacements.forEach((from, to) {
    out = out.replaceAll(from, to);
  });
  return out;
}

class LatLngFocus {
  final double lat;
  final double lon;
  final double zoom;
  const LatLngFocus(this.lat, this.lon, {this.zoom = 16});
}

class MapPage extends StatefulWidget {
  final LatLngFocus? initialFocus;
  const MapPage({super.key, this.initialFocus});

  @override
  State<MapPage> createState() => _MapPageState();
}

class _MapPageState extends State<MapPage> {
  static const String _historyKey = 'arborscan_history';
  static const String _authTokenKey = 'arborscan_auth_token';
  static const String _baseUrl = 'https://arborscanbackend-production.up.railway.app';

  bool _loading = true;
  String? _error;

  List<_HistoryItem> _items = const [];
  gmaps.GoogleMapController? _mapController;

  gmaps.MapType _mapType = gmaps.MapType.normal;
  bool _is3dMode = false;

  _HistoryItem? _selected;

  // If controller not ready yet, keep focus to apply later
  LatLngFocus? _pendingFocus;

  // Default fallback (neutral)
  static const gmaps.LatLng _fallbackCenter = gmaps.LatLng(55.751244, 37.618423);

  @override
  void initState() {
    super.initState();
    _pendingFocus = widget.initialFocus;
    _load();
  }

  @override
  void dispose() {
    _mapController?.dispose();
    super.dispose();
  }

  Future<void> _safeRefresh() async {
    try {
      await _load();
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = 'Ошибка обновления карты: $e';
        _loading = false;
      });
    }
  }

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _error = null;
      _selected = null;
    });

    try {
      final prefs = await SharedPreferences.getInstance();
      final list = prefs.getStringList(_historyKey) ?? const [];

      final parsed = <_HistoryItem>[];
      for (final s in list) {
        try {
          final m = jsonDecode(s) as Map<String, dynamic>;
          parsed.add(_HistoryItem.fromJson(m));
        } catch (_) {
          // ignore broken record
        }
      }

      final serverItems = await _loadServerHistoryItems();
      final byId = <String, _HistoryItem>{
        for (final item in parsed)
          if (item.analysisId.isNotEmpty) item.analysisId: item
      };
      for (final item in serverItems) {
        if (item.analysisId.isNotEmpty) {
          byId[item.analysisId] = item;
        }
      }
      final merged = byId.isNotEmpty ? byId.values.toList() : parsed;

      // Keep only valid geographic coordinates.
      final withGeo = merged.where((e) {
        final lat = e.lat;
        final lon = e.lon;
        if (lat == null || lon == null) return false;
        if (lat.isNaN || lon.isNaN) return false;
        return lat >= -90 && lat <= 90 && lon >= -180 && lon <= 180;
      }).toList();

      if (!mounted) return;
      setState(() {
        _items = withGeo;
        _loading = false;
      });

      // Apply initial focus if requested
      if (_pendingFocus != null) {
        await _tryApplyPendingFocus();
        return;
      }

      // Otherwise move to first point. If there are no points, keep fallback center.
      // Current location is available via the built-in Google Maps location button.
      // This avoids crashes on some Android devices when refresh triggers camera animation
      // before the map controller is fully ready.
      if (withGeo.isNotEmpty && _mapController != null) {
        await _animateTo(withGeo.first.lat!, withGeo.first.lon!, zoom: 16);
      }
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = e.toString();
        _loading = false;
      });
    }
  }

  Future<List<_HistoryItem>> _loadServerHistoryItems() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final token = prefs.getString(_authTokenKey) ?? '';
      if (token.isEmpty) return const [];

      final uri = Uri.parse('$_baseUrl/analyses/my').replace(
        queryParameters: {'token': token, 'limit': '200'},
      );
      final res = await http.get(uri).timeout(const Duration(seconds: 12));
      if (res.statusCode != 200) return const [];

      final data = jsonDecode(utf8.decode(res.bodyBytes)) as Map<String, dynamic>;
      final items = (data['items'] as List? ?? const []);

      final out = <_HistoryItem>[];
      for (final raw in items) {
        if (raw is! Map) continue;
        final m = raw.cast<String, dynamic>();
        final createdRaw = m['created_at']?.toString();
        DateTime ts;
        try {
          ts = createdRaw != null
              ? DateTime.parse(createdRaw.replaceFirst('Z', ''))
              : DateTime.now();
        } catch (_) {
          ts = DateTime.now();
        }

        out.add(_HistoryItem(
          analysisId: m['analysis_id']?.toString() ?? '',
          species: m['species']?.toString() ?? 'Неизвестно',
          height: (m['height_m'] as num?)?.toDouble(),
          crown: (m['crown_width_m'] as num?)?.toDouble(),
          trunk: (m['trunk_diameter_m'] as num?)?.toDouble(),
          scale: null,
          riskIndex: (m['risk_index'] as num?)?.toDouble(),
          riskCategory: m['risk_category']?.toString(),
          lat: (m['lat'] as num?)?.toDouble(),
          lon: (m['lon'] as num?)?.toDouble(),
          address: m['address']?.toString(),
          imageBase64: '',
          timestamp: ts,
        ));
      }
      return out;
    } catch (_) {
      return const [];
    }
  }

  Future<void> _tryApplyPendingFocus() async {
    final f = _pendingFocus;
    if (f == null) return;

    // If controller not created yet, wait until onMapCreated
    if (_mapController == null) return;

    await _animateTo(f.lat, f.lon, zoom: f.zoom);
    _pendingFocus = null;
  }

  Future<void> _tryMoveToCurrentLocation() async {
    try {
      final result = await LocationService.getCurrentPositionDetailed();
      final pos = result.position;
      if (pos == null) return;
      await _animateTo(pos.latitude, pos.longitude, zoom: 14);
    } catch (_) {
      // ignore
    }
  }

  Future<void> _animateTo(
    double lat,
    double lon, {
    double zoom = 15,
    double? tilt,
    double? bearing,
  }) async {
    final c = _mapController;
    if (c == null) return;

    try {
      await c.animateCamera(
        gmaps.CameraUpdate.newCameraPosition(
          gmaps.CameraPosition(
            target: gmaps.LatLng(lat, lon),
            zoom: zoom,
            tilt: tilt ?? (_is3dMode ? 60 : 0),
            bearing: bearing ?? (_is3dMode ? 35 : 0),
          ),
        ),
      );
    } catch (_) {
      // GoogleMapController can throw during refresh/dispose. Ignore to keep UI alive.
    }
  }

  gmaps.LatLng _bestMapTarget() {
    if (_selected?.lat != null && _selected?.lon != null) {
      return gmaps.LatLng(_selected!.lat!, _selected!.lon!);
    }
    if (_items.isNotEmpty && _items.first.lat != null && _items.first.lon != null) {
      return gmaps.LatLng(_items.first.lat!, _items.first.lon!);
    }
    return _fallbackCenter;
  }

  Future<void> _toggleMapType() async {
    setState(() {
      _mapType = _mapType == gmaps.MapType.normal
          ? gmaps.MapType.hybrid
          : gmaps.MapType.normal;
    });
  }

  Future<void> _toggle3dMode() async {
    setState(() => _is3dMode = !_is3dMode);
    final target = _bestMapTarget();
    await _animateTo(
      target.latitude,
      target.longitude,
      zoom: _is3dMode ? 19 : 16,
      tilt: _is3dMode ? 65 : 0,
      bearing: _is3dMode ? 35 : 0,
    );
  }

  Set<gmaps.Marker> _buildMarkers() {
    return _items.map((e) {
      final id = e.analysisId.isNotEmpty ? e.analysisId : e.timestampIso;
      return gmaps.Marker(
        markerId: gmaps.MarkerId(id),
        position: gmaps.LatLng(e.lat!, e.lon!),
        onTap: () async {
          setState(() => _selected = e);
          await _animateTo(
            e.lat!,
            e.lon!,
            zoom: _is3dMode ? 19 : 17,
            tilt: _is3dMode ? 65 : 0,
            bearing: _is3dMode ? 35 : 0,
          );
          _showDetailsSheet(e);
        },
      );
    }).toSet();
  }

  Future<void> _openStreetView(_HistoryItem item) async {
    final lat = item.lat;
    final lon = item.lon;
    if (lat == null || lon == null) return;

    // More reliable Street View variants for Android.
    // 1) Native Google Maps Street View scheme.
    // 2) Google Maps "layer=c" panorama URL.
    // 3) Browser panorama URL. If Google has no panorama nearby, it will open normal Maps.
    final nativeStreetView = Uri.parse('google.streetview:cbll=$lat,$lon&cbp=0,0,0,0,0');
    final mapsLayerStreetView = Uri.parse('https://www.google.com/maps?layer=c&cbll=$lat,$lon');
    final webStreetView = Uri.parse(
      'https://www.google.com/maps/@?api=1&map_action=pano&viewpoint=$lat,$lon',
    );

    try {
      if (await canLaunchUrl(nativeStreetView)) {
        await launchUrl(nativeStreetView, mode: LaunchMode.externalApplication);
        return;
      }
    } catch (_) {
      // try next
    }

    try {
      await launchUrl(mapsLayerStreetView, mode: LaunchMode.externalApplication);
      return;
    } catch (_) {
      // try next
    }

    await launchUrl(webStreetView, mode: LaunchMode.inAppBrowserView);
  }

  Future<void> _openStreetViewInBrowser(_HistoryItem item) async {
    final lat = item.lat;
    final lon = item.lon;
    if (lat == null || lon == null) return;

    final uri = Uri.parse('https://www.google.com/maps?layer=c&cbll=$lat,$lon');
    await launchUrl(uri, mode: LaunchMode.inAppBrowserView);
  }

  void _showDetailsSheet(_HistoryItem item) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      showDragHandle: true,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(18)),
      ),
      builder: (_) {
        final bytes = item.imageBase64.isNotEmpty ? _safeB64(item.imageBase64) : null;

        return DraggableScrollableSheet(
          expand: false,
          initialChildSize: 0.58,
          minChildSize: 0.32,
          maxChildSize: 0.92,
          builder: (context, scrollController) {
            return SafeArea(
              child: SingleChildScrollView(
                controller: scrollController,
                padding: EdgeInsets.fromLTRB(
                  16,
                  8,
                  16,
                  16 + MediaQuery.of(context).padding.bottom,
                ),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                Row(
                  children: [
                    Ui.badge(
                      text: item.riskCategory?.isNotEmpty == true ? item.riskCategory! : 'Анализ',
                      color: AppTheme.primary,
                      icon: Icons.location_on,
                    ),
                    const Spacer(),
                    Text(
                      item.formattedTs,
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(color: AppTheme.muted),
                    ),
                  ],
                ),
                const SizedBox(height: 12),

                Text(
                  item.species.isNotEmpty ? item.species : 'Неизвестно',
                  style: Theme.of(context).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w800),
                ),
                const SizedBox(height: 8),

                if (bytes != null) ...[
                  ClipRRect(
                    borderRadius: BorderRadius.circular(14),
                    child: Image.memory(
                      bytes,
                      height: 140,
                      width: double.infinity,
                      fit: BoxFit.cover,
                      errorBuilder: (_, __, ___) => Container(
                        height: 140,
                        alignment: Alignment.center,
                        decoration: BoxDecoration(
                          color: Colors.black.withOpacity(0.04),
                          borderRadius: BorderRadius.circular(14),
                          border: Border.all(color: AppTheme.border),
                        ),
                        child: const Icon(Icons.image_not_supported),
                      ),
                    ),
                  ),
                  const SizedBox(height: 12),
                ],

                Wrap(
                  spacing: 10,
                  runSpacing: 10,
                  children: [
                    if (item.height != null)
                      Ui.badge(
                        text: 'H: ${item.height!.toStringAsFixed(2)} м',
                        color: AppTheme.success,
                        icon: Icons.height,
                      ),
                    if (item.crown != null)
                      Ui.badge(
                        text: 'Крона: ${item.crown!.toStringAsFixed(2)} м',
                        color: AppTheme.success,
                        icon: Icons.nature,
                      ),
                    if (item.trunk != null)
                      Ui.badge(
                        text: 'Ствол: ${item.trunk!.toStringAsFixed(2)} м',
                        color: AppTheme.success,
                        icon: Icons.circle,
                      ),
                    if (item.riskIndex != null)
                      Ui.badge(
                        text: 'Риск: ${item.riskIndex!.toStringAsFixed(2)}',
                        color: AppTheme.warning,
                        icon: Icons.shield,
                      ),
                  ],
                ),

                const SizedBox(height: 12),
                Wrap(
                  spacing: 10,
                  runSpacing: 10,
                  children: [
                    ElevatedButton.icon(
                      onPressed: () => _openStreetView(item),
                      icon: const Icon(Icons.threesixty),
                      label: const Text('Street View'),
                    ),
                    OutlinedButton.icon(
                      onPressed: () => _openStreetViewInBrowser(item),
                      icon: const Icon(Icons.public),
                      label: const Text('В браузере'),
                    ),
                    OutlinedButton.icon(
                      onPressed: () async {
                        final lat = item.lat;
                        final lon = item.lon;
                        if (lat == null || lon == null) return;
                        final uri = Uri.parse(
                          'https://www.google.com/maps/search/?api=1&query=$lat,$lon',
                        );
                        await launchUrl(uri, mode: LaunchMode.externalApplication);
                      },
                      icon: const Icon(Icons.near_me_outlined),
                      label: const Text('Точка'),
                    ),
                  ],
                ),
                const Text(
                  'Просмотр улиц доступен не везде. Если Google пишет, что он недоступен, значит рядом с этой точкой нет панорамы Street View.',
                  style: TextStyle(
                    color: AppTheme.muted,
                    fontSize: 12,
                    height: 1.25,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                const SizedBox(height: 10),

                if (item.address != null && _normalizeAddressRu(item.address).isNotEmpty) ...[
                  const SizedBox(height: 12),
                  Text(_normalizeAddressRu(item.address), style: Theme.of(context).textTheme.bodyMedium),
                ],

                const SizedBox(height: 14),
                Row(
                  children: [
                    Expanded(
                      child: OutlinedButton.icon(
                        onPressed: () async {
                          Navigator.of(context).pop();
                          await _animateTo(item.lat!, item.lon!, zoom: 17);
                        },
                        icon: const Icon(Icons.center_focus_strong),
                        label: const Text('Показать'),
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: ElevatedButton.icon(
                        onPressed: () => Navigator.of(context).pop(),
                        icon: const Icon(Icons.check),
                        label: const Text('Ок'),
                      ),
                    ),
                  ],
                ),
                  ],
                ),
              ),
            );
          },
        );
      },
    );
  }

  Uint8List? _safeB64(String b64) {
    try {
      return base64Decode(b64);
    } catch (_) {
      return null;
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_loading) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text('Карта'),
        actions: [
          IconButton(
            tooltip: 'Обновить',
            icon: const Icon(Icons.refresh),
            onPressed: _safeRefresh,
          ),
          IconButton(
            tooltip: 'Настройки геолокации',
            icon: const Icon(Icons.settings),
            onPressed: () async {
              await Geolocator.openLocationSettings();
            },
          ),
        ],
      ),
      body: _error != null
          ? _ErrorState(message: _error!, onRetry: _load)
          : _items.isEmpty
              ? _EmptyState(onRetry: _load)
              : Stack(
                  children: [
                    gmaps.GoogleMap(
                      initialCameraPosition: const gmaps.CameraPosition(
                        target: _fallbackCenter,
                        zoom: 10,
                      ),
                      mapType: _mapType,
                      buildingsEnabled: true,
                      compassEnabled: true,
                      rotateGesturesEnabled: true,
                      tiltGesturesEnabled: true,
                      zoomControlsEnabled: false,
                      myLocationEnabled: true,
                      myLocationButtonEnabled: true,
                      markers: _buildMarkers(),
                      onMapCreated: (c) async {
                        _mapController = c;
                        // If there is pending focus (from history), apply now
                        await _tryApplyPendingFocus();
                      },
                      onTap: (_) => setState(() => _selected = null),
                    ),

                    Positioned(
                      top: 12,
                      right: 12,
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.end,
                        children: [
                          _MapControlButton(
                            icon: _mapType == gmaps.MapType.normal
                                ? Icons.satellite_alt_outlined
                                : Icons.map_outlined,
                            label: _mapType == gmaps.MapType.normal
                                ? 'Спутник'
                                : 'Карта',
                            onTap: _toggleMapType,
                          ),
                          const SizedBox(height: 8),
                          _MapControlButton(
                            icon: Icons.threed_rotation,
                            label: _is3dMode ? '2D' : '3D',
                            active: _is3dMode,
                            onTap: _toggle3dMode,
                          ),
                        ],
                      ),
                    ),

                    Positioned(
                      left: 12,
                      right: 12,
                      bottom: 12,
                      child: Card(
                        child: Padding(
                          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
                          child: Row(
                            children: [
                              const Icon(Icons.info_outline, size: 18),
                              const SizedBox(width: 10),
                              Expanded(
                                child: Text(
                                  'Тап по маркеру — детали анализа',
                                  style: Theme.of(context).textTheme.bodySmall?.copyWith(color: AppTheme.muted),
                                ),
                              ),
                              TextButton(
                                onPressed: _safeRefresh,
                                child: const Text('Обновить'),
                              ),
                            ],
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
    );
  }
}


class _MapControlButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback onTap;
  final bool active;

  const _MapControlButton({
    required this.icon,
    required this.label,
    required this.onTap,
    this.active = false,
  });

  @override
  Widget build(BuildContext context) {
    return Material(
      color: active
          ? AppTheme.primary.withOpacity(0.92)
          : AppTheme.surface.withOpacity(0.92),
      borderRadius: BorderRadius.circular(999),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(999),
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 9),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(999),
            border: Border.all(
              color: active ? AppTheme.primary : AppTheme.border,
            ),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(
                icon,
                size: 18,
                color: active ? const Color(0xFF06140E) : AppTheme.text,
              ),
              const SizedBox(width: 6),
              Text(
                label,
                style: TextStyle(
                  color: active ? const Color(0xFF06140E) : AppTheme.text,
                  fontSize: 12,
                  fontWeight: FontWeight.w900,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _EmptyState extends StatelessWidget {
  final VoidCallback onRetry;

  const _EmptyState({required this.onRetry});

  @override
  Widget build(BuildContext context) {
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Ui.paddedCard(
          context,
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Icon(Icons.map_outlined),
              const SizedBox(width: 12),
              Expanded(
                child: Text(
                  'Пока нет точек на карте.\n\n'
                  'Сделай анализ с включённой геолокацией — и результаты появятся здесь.',
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(color: AppTheme.muted),
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 12),
        ElevatedButton.icon(
          onPressed: onRetry,
          icon: const Icon(Icons.refresh),
          label: const Text('Обновить'),
        ),
        const SizedBox(height: 10),
        OutlinedButton.icon(
          onPressed: () async => Geolocator.openLocationSettings(),
          icon: const Icon(Icons.settings),
          label: const Text('Настройки геолокации'),
        ),
      ],
    );
  }
}

class _ErrorState extends StatelessWidget {
  final String message;
  final VoidCallback onRetry;

  const _ErrorState({required this.message, required this.onRetry});

  @override
  Widget build(BuildContext context) {
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Ui.paddedCard(
          context,
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Icon(Icons.error_outline, color: AppTheme.danger),
              const SizedBox(width: 12),
              Expanded(
                child: Text(
                  message,
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(color: AppTheme.danger),
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 12),
        ElevatedButton.icon(
          onPressed: onRetry,
          icon: const Icon(Icons.refresh),
          label: const Text('Повторить'),
        ),
      ],
    );
  }
}

class _HistoryItem {
  final String analysisId;
  final String species;
  final double? height;
  final double? crown;
  final double? trunk;
  final double? scale;
  final double? riskIndex;
  final String? riskCategory;

  final double? lat;
  final double? lon;
  final String? address;

  final String imageBase64;
  final DateTime timestamp;

  _HistoryItem({
    required this.analysisId,
    required this.species,
    required this.imageBase64,
    required this.timestamp,
    this.height,
    this.crown,
    this.trunk,
    this.scale,
    this.riskIndex,
    this.riskCategory,
    this.lat,
    this.lon,
    this.address,
  });

  String get timestampIso => timestamp.toIso8601String();

  String get formattedTs {
    String two(int v) => v.toString().padLeft(2, '0');
    return '${two(timestamp.day)}.${two(timestamp.month)}.${timestamp.year} '
        '${two(timestamp.hour)}:${two(timestamp.minute)}';
  }

  factory _HistoryItem.fromJson(Map<String, dynamic> json) {
    return _HistoryItem(
      analysisId: (json['analysisId'] ?? '') as String,
      species: (json['species'] ?? 'Неизвестно') as String,
      height: (json['height'] as num?)?.toDouble(),
      crown: (json['crown'] as num?)?.toDouble(),
      trunk: (json['trunk'] as num?)?.toDouble(),
      scale: (json['scale'] as num?)?.toDouble(),
      riskIndex: (json['riskIndex'] as num?)?.toDouble(),
      riskCategory: json['riskCategory'] as String?,
      lat: (json['lat'] as num?)?.toDouble(),
      lon: (json['lon'] as num?)?.toDouble(),
      address: json['address'] as String?,
      imageBase64: (json['imageBase64'] ?? '') as String,
      timestamp: DateTime.tryParse((json['timestamp'] ?? '') as String) ?? DateTime.now(),
    );
  }
}
