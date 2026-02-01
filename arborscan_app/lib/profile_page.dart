import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'app_theme.dart';

class ProfilePage extends StatefulWidget {
  const ProfilePage({super.key});

  @override
  State<ProfilePage> createState() => _ProfilePageState();
}

class _ProfilePageState extends State<ProfilePage> {
  static const String _adminFlagKey = 'arborscan_is_admin';

  bool _loading = true;
  bool _isAdmin = false;

  @override
  void initState() {
    super.initState();
    _loadAdminFlag();
  }

  Future<void> _loadAdminFlag() async {
    final prefs = await SharedPreferences.getInstance();
    final v = prefs.getBool(_adminFlagKey) ?? false;
    if (!mounted) return;
    setState(() {
      _isAdmin = v;
      _loading = false;
    });
  }

  Future<void> _setAdmin(bool v) async {
    setState(() => _isAdmin = v);
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_adminFlagKey, v);

    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(v ? 'Режим администратора включён' : 'Режим администратора выключен')),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Профиль')),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : ListView(
              padding: const EdgeInsets.all(16),
              children: [
                Ui.paddedCard(
                  context,
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Icon(Icons.account_circle_outlined, size: 34),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text('ArborScan', style: Theme.of(context).textTheme.titleLarge),
                            const SizedBox(height: 4),
                            Text(
                              'Анализ состояния деревьев',
                              style: Theme.of(context)
                                  .textTheme
                                  .bodyMedium
                                  ?.copyWith(color: AppTheme.muted),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),

                Ui.sectionTitle(context, 'Доступ'),
                Ui.paddedCard(
                  context,
                  child: SwitchListTile(
                    contentPadding: EdgeInsets.zero,
                    title: const Text('Режим администратора'),
                    subtitle: Text(
                      'Включает расширенные функции проверки и управления обучением.',
                      style: Theme.of(context)
                          .textTheme
                          .bodySmall
                          ?.copyWith(color: AppTheme.muted),
                    ),
                    value: _isAdmin,
                    onChanged: _setAdmin,
                  ),
                ),

                if (_isAdmin) ...[
                  const SizedBox(height: 12),
                  Ui.paddedCard(
                    context,
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Icon(Icons.admin_panel_settings, color: AppTheme.primary),
                        const SizedBox(width: 12),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Row(
                                children: [
                                  Ui.badge(
                                    text: 'ADMIN MODE',
                                    color: AppTheme.primary,
                                    icon: Icons.verified_user,
                                  ),
                                  const SizedBox(width: 10),
                                  Text(
                                    'включён',
                                    style: Theme.of(context)
                                        .textTheme
                                        .bodyMedium
                                        ?.copyWith(color: AppTheme.muted),
                                  ),
                                ],
                              ),
                              const SizedBox(height: 10),
                              const Text(
                                'Теперь доступна админ-панель и подтверждение/исправление анализа.',
                              ),
                            ],
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
