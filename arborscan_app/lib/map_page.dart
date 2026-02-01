import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:google_maps_flutter/google_maps_flutter.dart' as gmaps;
import 'package:shared_preferences/shared_preferences.dart';
import 'package:geolocator/geolocator.dart';

import 'app_theme.dart';

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

  bool _loading = true;
  String? _error;

  List<_HistoryItem> _items = const [];
  gmaps.GoogleMapController? _mapController;

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

      // Keep only those with coordinates
      final withGeo = parsed.where((e) => e.lat != null && e.lon != null).toList();

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

      // Otherwise move to first point or current location
      if (withGeo.isNotEmpty) {
        await _animateTo(withGeo.first.lat!, withGeo.first.lon!, zoom: 16);
      } else {
        await _tryMoveToCurrentLocation();
      }
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = e.toString();
        _loading = false;
      });
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
      final serviceEnabled = await Geolocator.isLocationServiceEnabled();
      if (!serviceEnabled) return;

      var perm = await Geolocator.checkPermission();
      if (perm == LocationPermission.denied) {
        perm = await Geolocator.requestPermission();
      }
      if (perm == LocationPermission.denied || perm == LocationPermission.deniedForever) return;

      final pos = await Geolocator.getCurrentPosition(
        desiredAccuracy: LocationAccuracy.high,
        timeLimit: const Duration(seconds: 6),
      );
      await _animateTo(pos.latitude, pos.longitude, zoom: 14);
    } catch (_) {
      // ignore
    }
  }

  Future<void> _animateTo(double lat, double lon, {double zoom = 15}) async {
    final c = _mapController;
    if (c == null) return;

    await c.animateCamera(
      gmaps.CameraUpdate.newCameraPosition(
        gmaps.CameraPosition(target: gmaps.LatLng(lat, lon), zoom: zoom),
      ),
    );
  }

  Set<gmaps.Marker> _buildMarkers() {
    return _items.map((e) {
      final id = e.analysisId.isNotEmpty ? e.analysisId : e.timestampIso;
      return gmaps.Marker(
        markerId: gmaps.MarkerId(id),
        position: gmaps.LatLng(e.lat!, e.lon!),
        onTap: () {
          setState(() => _selected = e);
          _showDetailsSheet(e);
        },
      );
    }).toSet();
  }

  void _showDetailsSheet(_HistoryItem item) {
    showModalBottomSheet(
      context: context,
      showDragHandle: true,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(18)),
      ),
      builder: (_) {
        final bytes = item.imageBase64.isNotEmpty ? _safeB64(item.imageBase64) : null;

        return SafeArea(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
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

                if (item.address != null && item.address!.isNotEmpty) ...[
                  const SizedBox(height: 12),
                  Text(item.address!, style: Theme.of(context).textTheme.bodyMedium),
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
            onPressed: _load,
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
                                onPressed: _load,
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
