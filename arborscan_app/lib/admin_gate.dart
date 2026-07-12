import 'package:flutter/material.dart';

/// Блок входа в административные инструменты.
///
/// Флаг [isAdmin] используется только для отображения интерфейса. Реальные
/// права при открытии панели повторно проверяет backend по Bearer-токену.
class AdminGate extends StatelessWidget {
  final bool isAdmin;
  final VoidCallback onOpenAdminPanel;
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
    final colors = theme.colorScheme;

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
                  isAdmin
                      ? Icons.admin_panel_settings
                      : Icons.lock_outline,
                  color: isAdmin
                      ? const Color(0xFF1565C0)
                      : colors.onSurfaceVariant,
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: Text(
                    'Административные инструменты',
                    style: theme.textTheme.titleSmall?.copyWith(
                      fontWeight: FontWeight.w800,
                    ),
                  ),
                ),
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 10,
                    vertical: 6,
                  ),
                  decoration: BoxDecoration(
                    color: isAdmin
                        ? const Color(0xFFE8F3FF)
                        : const Color(0xFFEFEFEF),
                    borderRadius: BorderRadius.circular(999),
                  ),
                  child: Text(
                    isAdmin ? 'ADMIN' : 'LOCKED',
                    style: theme.textTheme.labelMedium?.copyWith(
                      fontWeight: FontWeight.w800,
                      color: isAdmin
                          ? const Color(0xFF0D47A1)
                          : colors.onSurfaceVariant,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 10),
            Text(
              isAdmin
                  ? 'Роль получена из серверного профиля. При открытии панели '
                      'backend ещё раз проверит токен и права.'
                  : 'Административный доступ отсутствует.',
              style: theme.textTheme.bodySmall?.copyWith(
                color: colors.onSurfaceVariant,
              ),
            ),
            const SizedBox(height: 14),
            Row(
              children: [
                Expanded(
                  child: FilledButton.icon(
                    onPressed: isAdmin ? onOpenAdminPanel : null,
                    icon: const Icon(Icons.tune),
                    label: const Text('Админ-панель'),
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
