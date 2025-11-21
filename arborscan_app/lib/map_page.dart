import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:google_maps_flutter/google_maps_flutter.dart' as gmaps;

class MapPage extends StatefulWidget {
  final List<Map<String, dynamic>> points;

  const MapPage({super.key, required this.points});

  @override
  State<MapPage> createState() => _MapPageState();
}

class _MapPageState extends State<MapPage> {
  gmaps.GoogleMapController? _controller;

  List<Map<String, dynamic>> get _itemsWithGps =>
      widget.points.where((e) => e['lat'] != null && e['lon'] != null).toList();

  double _markerHue(String? category) {
    switch (category) {
      case 'низкий':
        return gmaps.BitmapDescriptor.hueGreen;
      case 'средний':
        return gmaps.BitmapDescriptor.hueYellow;
      case 'высокий':
        return gmaps.BitmapDescriptor.hueRed;
      default:
        return gmaps.BitmapDescriptor.hueAzure;
    }
  }

  void _showDetails(Map<String, dynamic> item) {
    final imageBase64 = item['imageBase64'] as String? ?? '';
    Uint8List? imgBytes;
    if (imageBase64.isNotEmpty) {
      try {
        imgBytes = base64Decode(imageBase64);
      } catch (_) {}
    }

    final tsStr = item['timestamp'] as String?;
    DateTime? ts;
    if (tsStr != null) {
      try {
        ts = DateTime.parse(tsStr);
      } catch (_) {}
    }

    final dateStr = ts != null
        ? '${ts.day.toString().padLeft(2, '0')}.'
          '${ts.month.toString().padLeft(2, '0')}.'
          '${ts.year}'
        : '';

    final riskCategory = item['riskCategory'] as String?;
    final riskIndex = (item['riskIndex'] as num?)?.toDouble();
    final address = item['address'] as String?;

    showModalBottomSheet(
      context: context,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
      ),
      builder: (ctx) {
        return Padding(
          padding: const EdgeInsets.fromLTRB(16, 12, 16, 24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Center(
                child: Container(
                  width: 40,
                  height: 4,
                  margin: const EdgeInsets.only(bottom: 12),
                  decoration: BoxDecoration(
                    color: Colors.grey.shade400,
                    borderRadius: BorderRadius.circular(999),
                  ),
                ),
              ),
              if (imgBytes != null)
                ClipRRect(
                  borderRadius: BorderRadius.circular(16),
                  child: Image.memory(
                    imgBytes,
                    fit: BoxFit.cover,
                    height: 160,
                    width: double.infinity,
                  ),
                ),
              const SizedBox(height: 12),
              Text(
                (item['species'] as String?) ?? 'Дерево',
                style: const TextStyle(
                    fontSize: 18, fontWeight: FontWeight.w600),
              ),
              if (dateStr.isNotEmpty) ...[
                const SizedBox(height: 4),
                Text(
                  'Дата измерения: $dateStr',
                  style: const TextStyle(color: Colors.grey, fontSize: 12),
                ),
              ],
              const SizedBox(height: 8),
              Text('Высота: ${(item['height'] ?? '-').toString()} м'),
              Text('Крона: ${(item['crown'] ?? '-').toString()} м'),
              Text('Диаметр ствола: ${(item['trunk'] ?? '-').toString()} м'),
              if (address != null && address.isNotEmpty) ...[
                const SizedBox(height: 8),
                Text(
                  address,
                  style: const TextStyle(fontSize: 12),
                ),
              ],
              if (riskCategory != null && riskIndex != null) ...[
                const SizedBox(height: 8),
                Text(
                  'Риск: $riskCategory (${riskIndex.toStringAsFixed(2)})',
                  style: const TextStyle(fontWeight: FontWeight.w500),
                ),
              ],
            ],
          ),
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    final items = _itemsWithGps;
    if (items.isEmpty) {
      return Scaffold(
        appBar: AppBar(title: const Text('Карта измерений')),
        body: const Center(
          child: Text('Нет данных для отображения карты'),
        ),
      );
    }

    final first = items.first;
    final lat = (first['lat'] as num).toDouble();
    final lon = (first['lon'] as num).toDouble();

    final initialCameraPosition = gmaps.CameraPosition(
      target: gmaps.LatLng(lat, lon),
      zoom: 16,
    );

    final markers = <gmaps.Marker>{};
    for (var i = 0; i < items.length; i++) {
      final item = items[i];
      final lt = (item['lat'] as num?)?.toDouble();
      final ln = (item['lon'] as num?)?.toDouble();
      if (lt == null || ln == null) continue;

      markers.add(
        gmaps.Marker(
          markerId: gmaps.MarkerId('tree_$i'),
          position: gmaps.LatLng(lt, ln),
          icon: gmaps.BitmapDescriptor.defaultMarkerWithHue(
            _markerHue(item['riskCategory'] as String?),
          ),
          onTap: () => _showDetails(item),
        ),
      );
    }

    return Scaffold(
      appBar: AppBar(title: const Text('Карта измерений')),
      body: gmaps.GoogleMap(
        initialCameraPosition: initialCameraPosition,
        markers: markers,
        onMapCreated: (controller) => _controller = controller,
        myLocationEnabled: false,
        myLocationButtonEnabled: false,
        compassEnabled: true,
        zoomControlsEnabled: false,
      ),
    );
  }
}
