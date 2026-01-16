import 'dart:convert';
import 'package:flutter/material.dart';
import 'mask_drawing_page.dart';
import 'stick_page.dart';
import 'package:http/http.dart' as http;

class FeedbackPage extends StatefulWidget {
  final String analysisId;
  final String originalImageBase64;
  final String? annotatedImageBase64;

  final String species;
  final double? heightM;
  final double? crownWidthM;
  final double? trunkDiameterM;
  final double? scalePxToM;

  const FeedbackPage({
    super.key,
    required this.analysisId,
    required this.originalImageBase64,
    this.annotatedImageBase64,
    required this.species,
    this.heightM,
    this.crownWidthM,
    this.trunkDiameterM,
    this.scalePxToM,
  });

  @override
  State<FeedbackPage> createState() => _FeedbackPageState();
}

class _FeedbackPageState extends State<FeedbackPage> {
  // Статусы того, были ли внесены правки
  bool _treeOk = true;
  bool _stickOk = true;
  bool _useForTraining = true;
  bool _isSending = false;

  void _popWithResult(Map<String, dynamic> result) {
    if (!mounted) return;
    // Чтобы не ловить navigator "_debugLocked" при pop во время build.
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted) return;
      final nav = Navigator.of(context);
      if (nav.canPop()) {
        nav.pop(result);
      }
    });
  }

  // Некоторые экраны открывают FeedbackPage как `push<Map<String, dynamic>>()`.
  // Поэтому этот экран должен возвращать Map, а не bool.

  // Данные для редактирования
  late String _selectedSpecies;
  late TextEditingController _heightController;
  late TextEditingController _crownController;
  late TextEditingController _trunkController;
  
  // Данные после правок в других экранах
  String? _userMaskBase64;
  double? _userScale;

  final List<String> _popularSpecies = ["Береза", "Дуб", "Ель", "Сосна", "Тополь"];

  @override
  void initState() {
    super.initState();
    _selectedSpecies = widget.species;
    _heightController = TextEditingController(text: widget.heightM?.toStringAsFixed(2) ?? '');
    _crownController = TextEditingController(text: widget.crownWidthM?.toStringAsFixed(2) ?? '');
    _trunkController = TextEditingController(text: widget.trunkDiameterM?.toStringAsFixed(2) ?? '');
    _userScale = widget.scalePxToM;
  }

  @override
  void dispose() {
    _heightController.dispose();
    _crownController.dispose();
    _trunkController.dispose();
    super.dispose();
  }

  // --- ЛОГИКА ОТПРАВКИ ---
  Future<void> _sendFeedback() async {
    setState(() => _isSending = true);
    
    final body = {
      "analysis_id": widget.analysisId,
      "tree_ok": _treeOk,
      "stick_ok": _stickOk,
      "species_ok": _selectedSpecies == widget.species,
      "params_ok": _checkParamsOk(),
      "use_for_training": _useForTraining,
      "user_mask_b64": _userMaskBase64,
      "user_species": _selectedSpecies,
      "user_params": {
        "height_m": double.tryParse(_heightController.text),
        "crown_width_m": double.tryParse(_crownController.text),
        "trunk_diameter_m": double.tryParse(_trunkController.text),
        "scale_px_to_m": _userScale,
      }
    };

    try {
      // Здесь должен быть твой URL из config
      final response = await http.post(
        Uri.parse('https://arborscanbackend-production.up.railway.app/feedback'),
        headers: {"Content-Type": "application/json"},
        body: jsonEncode(body),
      );

      if (response.statusCode == 200) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('Данные успешно подтверждены'), backgroundColor: Colors.green),
          );
          _popWithResult({
            "ok": true,
            "analysisId": widget.analysisId,
            "useForTraining": _useForTraining,
            "treeOk": _treeOk,
            "stickOk": _stickOk,
          });
        }
      } else {
        throw Exception('Ошибка сервера: ${response.statusCode}');
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Ошибка: $e'), backgroundColor: Colors.red),
        );
      }
    } finally {
      if (mounted) setState(() => _isSending = false);
    }
  }

  bool _checkParamsOk() {
    return _heightController.text == widget.heightM?.toStringAsFixed(2) &&
           _crownController.text == widget.crownWidthM?.toStringAsFixed(2) &&
           _trunkController.text == widget.trunkDiameterM?.toStringAsFixed(2);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Подтверждение'),
        actions: [
          _isSending 
            ? const Center(child: Padding(padding: EdgeInsets.all(16), child: CircularProgressIndicator(strokeWidth: 2)))
            : IconButton(
                icon: const Icon(Icons.done_all, color: Colors.green, size: 28),
                onPressed: _sendFeedback,
              ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _header("Проверка инструментов"),
            Row(
              children: [
                _actionTile(
                  "Маска", 
                  Icons.gesture, 
                  _treeOk, 
                  () async {
                    final res = await Navigator.push(context, MaterialPageRoute(
                      builder: (_) => MaskDrawingPage(
                        originalImageBase64: widget.originalImageBase64,
                        aiMaskBase64: widget.annotatedImageBase64,
                      ),
                    ));
                    if (res != null && res is Map) {
                      setState(() {
                        _treeOk = false;
                        _userMaskBase64 = res['mask_b64'];
                      });
                    }
                  }
                ),
                const SizedBox(width: 12),
                _actionTile(
                  "Масштаб", 
                  Icons.straighten, 
                  _stickOk, 
                  () async {
                    final res = await Navigator.push(context, MaterialPageRoute(
                      builder: (_) => StickPage(
                        originalImageBase64: widget.originalImageBase64,
                        currentScalePxToM: _userScale ?? 0.0,
                      ),
                    ));
                    if (res != null && res is double) {
                      setState(() {
                        _stickOk = false;
                        _userScale = res;
                      });
                    }
                  }
                ),
              ],
            ),
            const SizedBox(height: 24),
            _header("Вид дерева"),
            Wrap(
              spacing: 8,
              children: _popularSpecies.map((s) => ChoiceChip(
                label: Text(s),
                selected: _selectedSpecies == s,
                onSelected: (val) => setState(() => _selectedSpecies = s),
              )).toList(),
            ),
            const SizedBox(height: 8),
            TextField(
              decoration: const InputDecoration(
                hintText: "Если нет в списке, введите вручную",
                prefixIcon: Icon(Icons.edit_outlined),
              ),
              onChanged: (v) => setState(() => _selectedSpecies = v),
            ),
            const SizedBox(height: 24),
            _header("Физические параметры"),
            _inputField("Высота (м)", _heightController, Icons.height),
            _inputField("Ширина кроны (м)", _crownController, Icons.park_outlined),
            _inputField("Диаметр ствола (м)", _trunkController, Icons.radio_button_unchecked),
            const SizedBox(height: 24),
            Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: Colors.blue.withOpacity(0.05),
                borderRadius: BorderRadius.circular(12),
              ),
              child: SwitchListTile(
                title: const Text("В датасет для обучения"),
                subtitle: const Text("Помогает AI лучше распознавать деревья"),
                value: _useForTraining,
                onChanged: (v) => setState(() => _useForTraining = v),
              ),
            ),
            const SizedBox(height: 40),
          ],
        ),
      ),
    );
  }

  Widget _header(String text) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Text(text, style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: Colors.blueGrey)),
    );
  }

  Widget _actionTile(String title, IconData icon, bool isOk, VoidCallback onTap) {
    return Expanded(
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(12),
        child: Container(
          padding: const EdgeInsets.symmetric(vertical: 16),
          decoration: BoxDecoration(
            border: Border.all(color: isOk ? Colors.green.withOpacity(0.3) : Colors.orange),
            borderRadius: BorderRadius.circular(12),
            color: isOk ? Colors.green.withOpacity(0.02) : Colors.orange.withOpacity(0.05),
          ),
          child: Column(
            children: [
              Icon(icon, color: isOk ? Colors.green : Colors.orange),
              const SizedBox(height: 8),
              Text(title, style: const TextStyle(fontWeight: FontWeight.bold)),
              Text(isOk ? "Ок" : "Изменено", style: TextStyle(fontSize: 11, color: isOk ? Colors.green : Colors.orange)),
            ],
          ),
        ),
      ),
    );
  }

  Widget _inputField(String label, TextEditingController controller, IconData icon) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: TextFormField(
        controller: controller,
        keyboardType: const TextInputType.numberWithOptions(decimal: true),
        decoration: InputDecoration(
          labelText: label,
          prefixIcon: Icon(icon, size: 20),
          border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
          contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        ),
      ),
    );
  }
}