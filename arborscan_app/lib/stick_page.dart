import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';

class StickPage extends StatefulWidget {
  final String originalImageBase64;
  final double? heightM;
  final double? crownWidthM;
  final double? trunkDiameterM;
  final double? scalePxToM;

  const StickPage({
    super.key,
    required this.originalImageBase64,
    this.heightM,
    this.crownWidthM,
    this.trunkDiameterM,
    this.scalePxToM,
  });

  @override
  State<StickPage> createState() => _StickPageState();
}

class _StickPageState extends State<StickPage> {
  bool _stickOk = true;
  bool _paramsOk = true;

  @override
  Widget build(BuildContext context) {
    final img = base64Decode(widget.originalImageBase64);
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(title: const Text("Коррекция палки и параметров")),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(16, 16, 16, 24),
        children: [
          ClipRRect(
            borderRadius: BorderRadius.circular(20),
            child: Image.memory(img, fit: BoxFit.cover),
          ),
          const SizedBox(height: 16),

          Text(
            'Проверьте корректность параметров:',
            style: theme.textTheme.titleMedium?.copyWith(
              fontWeight: FontWeight.w700,
            ),
          ),
          const SizedBox(height: 12),

          _buildSwitch(
            "Палка определена правильно",
            _stickOk,
            (v) => setState(() => _stickOk = v),
          ),
          _buildSwitch(
            "Высота / крона / ствол рассчитаны верно",
            _paramsOk,
            (v) => setState(() => _paramsOk = v),
          ),

          const SizedBox(height: 20),

          FilledButton(
            onPressed: () {
              Navigator.pop(context, {
                "stick_ok": _stickOk,
                "params_ok": _paramsOk,
              });
            },
            child: const Padding(
              padding: EdgeInsets.symmetric(vertical: 14),
              child: Text("Сохранить"),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSwitch(String title, bool value, ValueChanged<bool> onChanged) {
    return Card(
      child: SwitchListTile(
        title: Text(title),
        value: value,
        onChanged: onChanged,
      ),
    );
  }
}
