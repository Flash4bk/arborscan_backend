import 'package:flutter/material.dart';
import 'corrections_service.dart';

/// A user's name is separate from the unchanged model prediction and taxonomy.
String? userSpeciesName(Map? report) {
  final annotation = report?['species_annotation'];
  final name = annotation is Map ? annotation['display_name'] : null;
  return name is String && name.trim().isNotEmpty ? name.trim() : null;
}

String speciesDisplayName(Map? report) {
  final manual = userSpeciesName(report);
  if (manual != null) return manual;
  final predicted = report?['species'];
  final name = predicted is Map
      ? predicted['display_name'] ?? predicted['scientific_name']
      : predicted;
  return name is String && name.trim().isNotEmpty
      ? name.trim()
      : 'Вид не указан';
}

/// Older immutable list indexes contain the original inference, not necessarily
/// the later manual name. Keep that limitation explicit without fetching photos.
String summarySpeciesName(Map? summary) {
  final name = summary?['species']?.toString() ?? 'Вид не определён';
  return summary?['presentation_version'] == 1
      ? name
      : 'Исходное определение: $name';
}

Map<String, dynamic> withSpeciesAnnotation(Map report, String name) {
  final text = name.trim();
  if (text.length > 120) {
    throw const FormatException('Название длиннее 120 символов.');
  }
  final copy = Map<String, dynamic>.from(report);
  if (text.isEmpty) {
    copy.remove('species_annotation');
  } else {
    copy['species_annotation'] = {
      'version': 1,
      'display_name': text,
      'source': 'user',
      'confirmed': false,
    };
  }
  return copy;
}

/// null = cancel; empty string = remove only the user's annotation.
Future<String?> editSpeciesName(BuildContext context, Map? report) async {
  final controller = TextEditingController(text: userSpeciesName(report) ?? '');
  final navigator = Navigator.of(context);
  DialogRoute<String>? activeRoute;
  void cancelOnAccountChange() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (activeRoute?.isCurrent == true) navigator.pop();
    });
  }

  CorrectionsService.authChanges.addListener(cancelOnAccountChange);
  try {
    final route = DialogRoute<String>(
        context: context,
        builder: (c) => AlertDialog(
              scrollable: true,
              title: const Text('Вид дерева'),
              content: Column(mainAxisSize: MainAxisSize.min, children: [
                TextField(
                    controller: controller,
                    autofocus: true,
                    maxLength: 120,
                    decoration: const InputDecoration(
                        labelText: 'Название',
                        hintText: 'Например, сосна обыкновенная')),
                const Text(
                    'Сохранится как ваша запись, без подтверждения определения.'),
              ]),
              actions: [
                if (userSpeciesName(report) != null)
                  TextButton(
                      onPressed: () => Navigator.pop(c, ''),
                      child: const Text('Убрать мою запись')),
                TextButton(
                    onPressed: () => Navigator.pop(c),
                    child: const Text('Отмена')),
                ValueListenableBuilder<TextEditingValue>(
                    valueListenable: controller,
                    builder: (context, value, _) => FilledButton(
                        onPressed: value.text.trim().isEmpty
                            ? null
                            : () => Navigator.pop(c, value.text.trim()),
                        child: const Text('Применить'))),
              ],
            ));
    activeRoute = route;
    final result = await navigator.push(route);
    await route.completed;
    return result;
  } finally {
    CorrectionsService.authChanges.removeListener(cancelOnAccountChange);
    controller.dispose();
  }
}

class SpeciesAnnotationCard extends StatelessWidget {
  final Map? report;
  final VoidCallback? onEdit;
  const SpeciesAnnotationCard({super.key, required this.report, this.onEdit});
  @override
  Widget build(BuildContext context) => Card(
      child: Padding(
          padding: const EdgeInsets.all(16),
          child:
              Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Text('Вид дерева', style: Theme.of(context).textTheme.titleSmall),
            const SizedBox(height: 4),
            Text(speciesDisplayName(report),
                style: Theme.of(context).textTheme.titleLarge),
            if (userSpeciesName(report) != null) const Text('Указано вами'),
            if (onEdit != null)
              TextButton.icon(
                  onPressed: onEdit,
                  icon: const Icon(Icons.edit_outlined),
                  label: Text(userSpeciesName(report) == null
                      ? 'Указать вид'
                      : 'Изменить вид')),
          ])));
}
