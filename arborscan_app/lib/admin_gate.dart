import 'package:flutter/material.dart';
import 'admin_list_page.dart';

/// Safe UI gate: this is a single widget you can place into any `children: []`
/// without using collection-if / spreads in your main.dart.
class AdminGate extends StatelessWidget {
  final bool isAdmin;
  final VoidCallback? onOpenFeedback;

  const AdminGate({
    super.key,
    required this.isAdmin,
    this.onOpenFeedback,
  });

  @override
  Widget build(BuildContext context) {
    if (!isAdmin) return const SizedBox.shrink();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        FilledButton.icon(
          icon: const Icon(Icons.admin_panel_settings),
          label: const Text('Admin Panel'),
          onPressed: () {
            Navigator.push(
              context,
              MaterialPageRoute(builder: (_) => const AdminListPage()),
            );
          },
        ),
        if (onOpenFeedback != null) ...[
          const SizedBox(height: 8),
          FilledButton.icon(
            icon: const Icon(Icons.check_circle_outline),
            label: const Text('Подтвердить / исправить анализ'),
            onPressed: onOpenFeedback,
          ),
        ],
      ],
    );
  }
}
