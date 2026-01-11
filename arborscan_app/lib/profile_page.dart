import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'core/admin_state.dart';

class ProfilePage extends StatelessWidget {
  const ProfilePage({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    final adminState = context.watch<AdminState>();

    return Scaffold(
      appBar: AppBar(
        title: const Text('Профиль'),
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          const Text(
            'ArborScan',
            style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 4),
          const Text(
            'Анализ состояния деревьев',
            style: TextStyle(color: Colors.grey),
          ),
          const SizedBox(height: 24),

          // --- ADMIN MODE ---
          SwitchListTile(
            title: const Text('Режим администратора'),
            subtitle: const Text('Включить расширенные функции'),
            value: adminState.isAdmin,
            onChanged: (_) => adminState.toggle(),
          ),

          if (adminState.isAdmin) ...[
            const SizedBox(height: 16),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.blue.withOpacity(0.1),
                borderRadius: BorderRadius.circular(8),
              ),
              child: const Row(
                children: [
                  Icon(Icons.admin_panel_settings, color: Colors.blue),
                  SizedBox(width: 12),
                  Expanded(
                    child: Text(
                      'ADMIN MODE включён.\n'
                      'Теперь доступны инструменты проверки и исправления анализа.',
                      style: TextStyle(color: Colors.blue),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ],
      ),
    );
  }
}
