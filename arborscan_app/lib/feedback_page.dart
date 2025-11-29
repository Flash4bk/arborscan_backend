import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/material.dart';

import 'mask_drawing_page.dart';
import 'stick_page.dart';

class FeedbackPage extends StatefulWidget {
  final String analysisId;
  final String originalImageBase64;
  final String species;
  final double? heightM;
  final double? crownWidthM;
  final double? trunkDiameterM;
  final double? scalePxToM;

  const FeedbackPage({
    super.key,
    required this.analysisId,
    required this.originalImageBase64,
    required this.species,
    required this.heightM,
    required this.crownWidthM,
    required this.trunkDiameterM,
    required this.scalePxToM,
  });

  @override
  State<FeedbackPage> createState() => _FeedbackPageState();
}

class _FeedbackPageState extends State<FeedbackPage> {
  bool _treeOk = true;
  bool _stickOk = true;
  bool _paramsOk = true;
  bool _speciesOk = true;
  bool _useForTraining = true;

  String _selectedSpecies = "";
  final TextEditingController _customSpeciesController =
      TextEditingController();

  String? _userMaskBase64;

  @override
  void initState() {
    super.initState();
    _selectedSpecies = widget.species;
  }

  @override
  void dispose() {
    _customSpeciesController.dispose();
    super.dispose();
  }

  Future<void> _openMaskDrawing() async {
    final maskBase64 = await Navigator.push<String?>(
      context,
      MaterialPageRoute(
        builder: (_) => MaskDrawingPage(
          originalImageBase64: widget.originalImageBase64,
          initialMaskBase64: _userMaskBase64,
        ),
      ),
    );

    if (maskBase64 != null && mounted) {
      setState(() {
        _userMaskBase64 = maskBase64;
      });
    }
  }

  Future<void> _openStickPage() async {
    final result = await Navigator.push<Map<String, dynamic>?>(
      context,
      MaterialPageRoute(
        builder: (_) => StickPage(
          originalImageBase64: widget.originalImageBase64,
          heightM: widget.heightM,
          crownWidthM: widget.crownWidthM,
          trunkDiameterM: widget.trunkDiameterM,
          scalePxToM: widget.scalePxToM,
        ),
      ),
    );

    if (result != null && mounted) {
      setState(() {
        if (result.containsKey('stick_ok')) {
          _stickOk = result['stick_ok'] as bool;
        }
        if (result.containsKey('params_ok')) {
          _paramsOk = result['params_ok'] as bool;
        }
      });
    }
  }

  void _onSubmit() {
    String? finalSpecies;
    if (_speciesOk) {
      finalSpecies = null; // вид принимаем как есть
    } else {
      if (_selectedSpecies == 'Другое') {
        finalSpecies = _customSpeciesController.text.trim().isEmpty
            ? null
            : _customSpeciesController.text.trim();
      } else {
        finalSpecies = _selectedSpecies;
      }
    }

    Navigator.pop<Map<String, dynamic>>(context, {
      "use_for_training": _useForTraining,
      "tree_ok": _treeOk,
      "stick_ok": _stickOk,
      "params_ok": _paramsOk,
      "species_ok": _speciesOk,
      "correct_species": finalSpecies,
      "user_mask_base64": _userMaskBase64,
    });
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final Uint8List imgBytes = base64Decode(widget.originalImageBase64);

    String _formatVal(double? v, {String suffix = 'м'}) {
      if (v == null) return '—';
      return '${v.toStringAsFixed(2)} $suffix';
    }

    String scaleText;
    if (widget.scalePxToM == null) {
      scaleText = 'Масштаб не найден (нет палки 1 м).';
    } else {
      scaleText = '1 px ≈ ${widget.scalePxToM!.toStringAsFixed(4)} м';
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text('Проверка анализа'),
      ),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(16, 16, 16, 24),
        children: [
          // Картинка
          ClipRRect(
            borderRadius: BorderRadius.circular(20),
            child: AspectRatio(
              aspectRatio: 3 / 4,
              child: Image.memory(
                imgBytes,
                fit: BoxFit.cover,
              ),
            ),
          ),
          const SizedBox(height: 16),

          // Краткие параметры
          Card(
            child: Padding(
              padding: const EdgeInsets.all(14),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Текущий результат',
                    style: theme.textTheme.titleMedium?.copyWith(
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Wrap(
                    spacing: 8,
                    runSpacing: 8,
                    children: [
                      _MiniChip(
                        label: 'Вид',
                        value: widget.species,
                        icon: Icons.park,
                      ),
                      _MiniChip(
                        label: 'Высота',
                        value: _formatVal(widget.heightM),
                        icon: Icons.height,
                      ),
                      _MiniChip(
                        label: 'Крона',
                        value: _formatVal(widget.crownWidthM),
                        icon: Icons.filter_hdr,
                      ),
                      _MiniChip(
                        label: 'Ствол',
                        value: _formatVal(widget.trunkDiameterM),
                        icon: Icons.circle_outlined,
                      ),
                      _MiniChip(
                        label: 'Масштаб',
                        value: scaleText,
                        icon: Icons.straighten,
                        isWide: true,
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 16),

          // Переключатели
          _buildSwitch(
            'Дерево выделено правильно',
            _treeOk,
            (v) => setState(() => _treeOk = v),
          ),
          _buildSwitch(
            'Палка определена правильно',
            _stickOk,
            (v) => setState(() => _stickOk = v),
          ),
          _buildSwitch(
            'Параметры рассчитаны верно',
            _paramsOk,
            (v) => setState(() => _paramsOk = v),
          ),
          _buildSwitch(
            'Вид определён верно',
            _speciesOk,
            (v) => setState(() => _speciesOk = v),
          ),

          const SizedBox(height: 8),

          // Кнопки маски и палки
          Row(
            children: [
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: _openMaskDrawing,
                  icon: const Icon(Icons.brush_outlined),
                  label: Text(
                    _userMaskBase64 == null
                        ? 'Нарисовать маску'
                        : 'Изменить маску',
                  ),
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: _openStickPage,
                  icon: const Icon(Icons.straighten),
                  label: const Text('Палка / параметры'),
                ),
              ),
            ],
          ),

          if (_userMaskBase64 != null) ...[
            const SizedBox(height: 8),
            Row(
              children: const [
                Icon(Icons.check_circle, size: 18, color: Colors.green),
                SizedBox(width: 6),
                Text('Маска пользователя добавлена'),
              ],
            ),
          ],

          const SizedBox(height: 16),

          if (!_speciesOk) _buildSpeciesSelector(),

          const SizedBox(height: 16),

          Card(
            child: SwitchListTile(
              title: const Text('Использовать этот пример для обучения'),
              subtitle: const Text(
                'При включении изображение и ваши правки\n'
                'будут сохранены как доверенный пример.',
              ),
              value: _useForTraining,
              onChanged: (v) => setState(() => _useForTraining = v),
            ),
          ),

          const SizedBox(height: 20),

          FilledButton(
            onPressed: _onSubmit,
            child: const Padding(
              padding: EdgeInsets.symmetric(vertical: 14),
              child: Text('Отправить отзыв'),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSwitch(
    String title,
    bool value,
    ValueChanged<bool> onChanged,
  ) {
    return Card(
      margin: const EdgeInsets.only(bottom: 8),
      child: SwitchListTile(
        title: Text(title),
        value: value,
        onChanged: onChanged,
      ),
    );
  }

  Widget _buildSpeciesSelector() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Выберите правильный вид:',
          style: TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w600,
          ),
        ),
        const SizedBox(height: 8),
        DropdownButtonFormField<String>(
          decoration: const InputDecoration(
            border: OutlineInputBorder(),
          ),
          value: _selectedSpecies.isEmpty ? widget.species : _selectedSpecies,
          items: const [
            DropdownMenuItem(value: 'Береза', child: Text('Берёза')),
            DropdownMenuItem(value: 'Дуб', child: Text('Дуб')),
            DropdownMenuItem(value: 'Ель', child: Text('Ель')),
            DropdownMenuItem(value: 'Сосна', child: Text('Сосна')),
            DropdownMenuItem(value: 'Тополь', child: Text('Тополь')),
            DropdownMenuItem(value: 'Другое', child: Text('Другое')),
          ],
          onChanged: (value) {
            setState(() {
              _selectedSpecies = value ?? 'Другое';
              if (_selectedSpecies != 'Другое') {
                _customSpeciesController.clear();
              }
            });
          },
        ),
        const SizedBox(height: 10),
        if (_selectedSpecies == 'Другое')
          TextField(
            controller: _customSpeciesController,
            decoration: const InputDecoration(
              labelText: 'Введите вид дерева',
              border: OutlineInputBorder(),
            ),
          ),
      ],
    );
  }
}

class _MiniChip extends StatelessWidget {
  final String label;
  final String value;
  final IconData icon;
  final bool isWide;

  const _MiniChip({
    required this.label,
    required this.value,
    required this.icon,
    this.isWide = false,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return ConstrainedBox(
      constraints: BoxConstraints(
        minWidth: isWide ? 180 : 0,
      ),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
        decoration: BoxDecoration(
          color: const Color(0xFFF3F7F4),
          borderRadius: BorderRadius.circular(999),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 18, color: Colors.black54),
            const SizedBox(width: 6),
            Flexible(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    label,
                    style: theme.textTheme.labelSmall?.copyWith(
                      color: Colors.black54,
                    ),
                  ),
                  Text(
                    value,
                    style: theme.textTheme.bodySmall?.copyWith(
                      fontWeight: FontWeight.w600,
                    ),
                    overflow: TextOverflow.ellipsis,
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
