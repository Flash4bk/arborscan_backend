import 'package:flutter/material.dart';

/// Compact admin entry block shown on the Home screen.
/// This widget does NOT navigate by itself (no page imports) to avoid build issues.
/// Navigation is handled by callbacks from main.dart.
class AdminGate extends StatelessWidget {
  final bool isAdmin;

  /// Open admin tools panel (models, training, etc.)
  final VoidCallback onOpenAdminPanel;

  /// Open feedback / correction flow for the last analysis.
  final VoidCallback onOpenFeedback;

  const AdminGate({
    super.key,
    required this.isAdmin,
    required this.onOpenAdminPanel,
    required this.onOpenFeedback,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;

    return Card(
      margin: EdgeInsets.zero,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(
                  isAdmin ? Icons.admin_panel_settings : Icons.lock_outline,
                  color: isAdmin ? const Color(0xFF1565C0) : cs.onSurfaceVariant,
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: Text(
                    isAdmin ? 'Admin Mode' : 'Admin tools',
                    style: theme.textTheme.titleSmall?.copyWith(
                      fontWeight: FontWeight.w800,
                    ),
                  ),
                ),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                  decoration: BoxDecoration(
                    color: isAdmin ? const Color(0xFFE8F3FF) : const Color(0xFFEFEFEF),
                    borderRadius: BorderRadius.circular(999),
                  ),
                  child: Text(
                    isAdmin ? 'ADMIN' : 'LOCKED',
                    style: theme.textTheme.labelMedium?.copyWith(
                      fontWeight: FontWeight.w800,
                      color: isAdmin ? const Color(0xFF0D47A1) : cs.onSurfaceVariant,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 10),
            Text(
              isAdmin
                  ? 'Доступны инструменты администратора и правка анализов.'
                  : 'Включите режим администратора в Настройках, чтобы открыть инструменты.',
              style: theme.textTheme.bodySmall?.copyWith(
                color: cs.onSurfaceVariant,
              ),
            ),
            const SizedBox(height: 14),

            // Buttons
            Row(
              children: [
                Expanded(
                  child: FilledButton.icon(
                    onPressed: isAdmin ? onOpenAdminPanel : null,
                    icon: const Icon(Icons.tune),
                    label: const Text('Admin panel'),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: isAdmin ? onOpenFeedback : null,
                    icon: const Icon(Icons.edit_note),
                    label: const Text('Правка'),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
