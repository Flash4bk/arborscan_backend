import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:path_provider/path_provider.dart';

/// Account-owned local journal. Photo is written once, editor snapshots atomically.
/// Never stores authentication tokens. Caller must validate the current session.
class ContourDrafts {
  final Future<Directory> Function() directory;
  static Future<void> _writes = Future.value();
  ContourDrafts({Future<Directory> Function()? directory})
      : directory = directory ?? getApplicationSupportDirectory;

  Future<Directory> _folder(String owner) async {
    if (!RegExp(r'^[a-fA-F0-9-]{36}$').hasMatch(owner)) throw const FormatException('Invalid owner');
    final root = await directory();
    return Directory('${root.path}/contour-drafts-v1/$owner')..createSync(recursive: true);
  }

  String _id(String id) {
    if (!RegExp(r'^[a-zA-Z0-9_.-]{1,240}$').hasMatch(id) || id.contains('..')) {
      throw const FormatException('Invalid draft id');
    }
    return id;
  }

  Future<void> save(String owner, String id, Map<String, dynamic> data, Uint8List image) {
    // Snapshot before enqueueing so subsequent editor mutations cannot alter it.
    final encoded = jsonEncode(data);
    final task = _writes.catchError((_) {}).then((_) async {
      final folder = await _folder(owner);
      final stem = '${folder.path}/${_id(id)}';
      final photo = File('$stem.photo');
      if (!await photo.exists()) {
        await File('$stem.photo.tmp').writeAsBytes(image, flush: true);
        await File('$stem.photo.tmp').rename(photo.path);
      }
      await File('$stem.json.tmp').writeAsString(encoded, flush: true);
      await File('$stem.json.tmp').rename('$stem.json');
    });
    _writes = task;
    return task;
  }

  Future<Map<String, dynamic>?> load(String owner, String id) async {
    await _writes.catchError((_) {});
    final folder = await _folder(owner);
    final file = File('${folder.path}/${_id(id)}.json');
    if (!await file.exists()) return null;
    final data = jsonDecode(await file.readAsString()) as Map<String, dynamic>;
    data['image'] = await File('${folder.path}/$id.photo').readAsBytes();
    data['draft_id'] = id;
    return data;
  }

  Future<List<Map<String, dynamic>>> list(String owner) async {
    await _writes.catchError((_) {});
    final folder = await _folder(owner);
    final result = <Map<String, dynamic>>[];
    await for (final file in folder.list()) {
      if (file is File && file.path.endsWith('.json')) {
        final id = file.uri.pathSegments.last.replaceFirst(RegExp(r'\.json$'), '');
        try {
          // List metadata only; don't decode every original photo into memory.
          final row = jsonDecode(await file.readAsString()) as Map<String, dynamic>;
          result.add({...row, 'draft_id':id});
        } catch (_) { /* A corrupt draft must not hide the other drafts. */ }
      }
    }
    return result;
  }

  Future<void> remove(String owner, String id) {
    final task = _writes.catchError((_) {}).then((_) async {
      final folder = await _folder(owner);
      for (final suffix in ['.json', '.photo', '.json.tmp', '.photo.tmp']) {
        final file = File('${folder.path}/${_id(id)}$suffix');
        if (await file.exists()) await file.delete();
      }
    });
    _writes = task;
    return task;
  }
}
