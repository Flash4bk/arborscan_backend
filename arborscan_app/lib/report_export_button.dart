import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'corrections_service.dart';
import 'report_export_data.dart';
import 'report_pdf.dart';

typedef ExportLoader = Future<ReportExportData> Function();

class ReportExportButton extends StatefulWidget {
  final ExportLoader load;
  final bool enabled;
  const ReportExportButton(
      {super.key, required this.load, this.enabled = true});
  @override
  State<ReportExportButton> createState() => _ReportExportButtonState();
}

class _ReportExportButtonState extends State<ReportExportButton> {
  static const channel = MethodChannel('arborscan/report_export');
  late final Future<String> _token = CorrectionsService.currentToken();
  late final int _epoch = CorrectionsService.authChanges.value;
  bool _busy = false, _invalid = false;
  String? _message;
  @override
  void initState() {
    super.initState();
    // Bind the screen to the session before the first asynchronous operation.
    _token;
    _epoch;
    CorrectionsService.authChanges.addListener(_changed);
  }

  void _changed() {
    if (mounted) {
      setState(() {
        _invalid = true;
        _message = 'Аккаунт изменился. Откройте свой отчёт заново.';
      });
    }
  }

  @override
  void dispose() {
    CorrectionsService.authChanges.removeListener(_changed);
    super.dispose();
  }

  Future<void> _guard() async {
    if (_invalid || _epoch != CorrectionsService.authChanges.value) {
      throw const CorrectionException('Аккаунт изменился.');
    }
    await CorrectionsService().checkSession(await _token);
  }

  Future<void> _export() async {
    if (_busy || _invalid || !widget.enabled) return;
    setState(() {
      _busy = true;
      _message = 'Формирование PDF…';
    });
    try {
      await _guard();
      final frozen = await widget.load();
      await _guard();
      if (!mounted) return;
      var partial = false;
      if (frozen.photo == null) {
        partial = await showDialog<bool>(
                context: context,
                builder: (c) => AlertDialog(
                        title: const Text('Исходное фото недоступно'),
                        content: const Text(
                            'Вернитесь к записи и повторите загрузку. Можно явно создать неполный PDF только с доступными данными.'),
                        actions: [
                          TextButton(
                              onPressed: () => Navigator.pop(c, false),
                              child: const Text('Отмена')),
                          TextButton(
                              onPressed: () => Navigator.pop(c, true),
                              child: const Text('Неполный PDF'))
                        ])) ??
            false;
        if (!partial) {
          _message = 'Экспорт отменён';
          return;
        }
      }
      final font = await rootBundle.load('assets/fonts/DejaVuSans.ttf');
      final bytes = await compute(buildReportPdfInWorker,
          {'data': frozen, 'partial': partial, 'font': font});
      await _guard();
      final dir =
          Directory('${(await getTemporaryDirectory()).path}/report-export');
      await dir.create(recursive: true);
      // Grant recipients time to consume files. Clean only our PDFs older than 7 days.
      await for (final f in dir.list()) {
        if (f is File &&
            f.path.endsWith('.pdf') &&
            DateTime.now().difference((await f.stat()).modified).inDays >= 7) {
          await f.delete();
        }
      }
      final file = File(
          '${dir.path}/${DateTime.now().microsecondsSinceEpoch}_${frozen.filename}');
      await file.writeAsBytes(bytes, flush: true);
      await _guard();
      if (!mounted) return;
      // One operation holds the lock through the platform result, including SAF cancellation.
      final action = await showDialog<String>(
          context: context,
          builder: (c) =>
              SimpleDialog(title: const Text('PDF готов'), children: [
                for (final entry in {
                  'open': 'Открыть PDF',
                  'save': 'Сохранить PDF…',
                  'share': 'Поделиться PDF…'
                }.entries)
                  SimpleDialogOption(
                      onPressed: () => Navigator.pop(c, entry.key),
                      child: Text(entry.value)),
                SimpleDialogOption(
                    onPressed: () => Navigator.pop(c),
                    child: const Text('Закрыть')),
              ]));
      await _guard();
      if (action == null) {
        _message = 'PDF сформирован; сохранение не выполнялось.';
        return;
      }
      final result = await channel.invokeMethod<String>(
          action, {'path': file.path, 'name': frozen.filename});
      await _guard();
      _message = result == 'saved'
          ? 'PDF сохранён в выбранное место'
          : result == 'cancelled'
              ? 'Сохранение отменено'
              : action == 'share'
                  ? 'Системное меню передачи открыто; отправка не подтверждается приложением.'
                  : 'PDF передан приложению просмотра';
    } on CorrectionException catch (e) {
      _message = e.message;
    } on FormatException catch (e) {
      _message = e.message;
    } on PlatformException catch (_) {
      _message =
          'Не удалось открыть системное действие. Проверьте наличие приложения для PDF и повторите.';
    } catch (_) {
      _message =
          'Не удалось создать PDF. Проверьте доступность файлов и свободное место, затем повторите.';
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) =>
      Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        OutlinedButton.icon(
            key: const ValueKey('export-pdf'),
            onPressed: _busy || _invalid || !widget.enabled ? null : _export,
            icon: const Icon(Icons.picture_as_pdf_outlined),
            label: Text(_busy ? 'Формирование PDF…' : 'Экспорт PDF')),
        if (_busy) const LinearProgressIndicator(),
        if (_message != null) Text(_message!),
      ]);
}
