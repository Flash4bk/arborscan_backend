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
    if (_loading) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    return Scaffold(
      appBar: AppBar(title: const Text('Профиль')),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(16, 12, 16, 18),
        children: [
          Container(
            padding: const EdgeInsets.all(18),
            decoration: BoxDecoration(
              gradient: const LinearGradient(
                colors: [AppTheme.surface2, AppTheme.surface3],
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
              ),
              borderRadius: BorderRadius.circular(26),
              border: Border.all(color: AppTheme.border),
            ),
            child: Row(
              children: [
                Container(
                  width: 58,
                  height: 58,
                  decoration: BoxDecoration(
                    color: AppTheme.primary.withOpacity(0.14),
                    borderRadius: BorderRadius.circular(18),
                  ),
                  child: const Icon(Icons.account_circle_rounded, color: AppTheme.primary, size: 34),
                ),
                const SizedBox(width: 14),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('ArborScan', style: Theme.of(context).textTheme.titleLarge),
                      const SizedBox(height: 4),
                      Text(
                        'Полевая система анализа состояния деревьев',
                        style: Theme.of(context).textTheme.bodySmall,
                      ),
                    ],
                  ),
                ),
                Ui.badge(
                  text: _isAdmin ? 'ADMIN' : 'USER',
                  color: _isAdmin ? AppTheme.primary : AppTheme.muted,
                  icon: _isAdmin ? Icons.verified_user : Icons.person_outline,
                ),
              ],
            ),
          ),
          const SizedBox(height: 16),
          Row(
            children: const [
              Expanded(
                child: AppStatCard(
                  value: 'AI',
                  label: 'анализ породы',
                  icon: Icons.auto_awesome_rounded,
                  color: AppTheme.primary,
                ),
              ),
              SizedBox(width: 12),
              Expanded(
                child: AppStatCard(
                  value: 'AR',
                  label: 'измерение',
                  icon: Icons.view_in_ar_rounded,
                  color: AppTheme.warning,
                ),
              ),
            ],
          ),
          const SizedBox(height: 18),
          Ui.sectionTitle(context, 'Режим и доступ'),
          Ui.paddedCard(
            context,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Container(
                      width: 42,
                      height: 42,
                      decoration: BoxDecoration(
                        color: AppTheme.primary.withOpacity(0.12),
                        borderRadius: BorderRadius.circular(14),
                      ),
                      child: const Icon(Icons.admin_panel_settings_rounded, color: AppTheme.primary),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Text(
                        'Режим администратора',
                        style: Theme.of(context).textTheme.titleMedium,
                      ),
                    ),
                    Switch(
                      value: _isAdmin,
                      onChanged: _setAdmin,
                      activeColor: Colors.black,
                      activeTrackColor: AppTheme.primary,
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                Text(
                  _isAdmin
                      ? 'Включён доступ к расширенным функциям проверки и управления обучением.'
                      : 'В обычном режиме доступны просмотр и запуск анализа.',
                  style: Theme.of(context).textTheme.bodySmall,
                ),
              ],
            ),
          ),
          const SizedBox(height: 16),
          Ui.sectionTitle(context, 'Быстрые действия'),
          AppActionButton(
            onTap: null,
            icon: Icons.security_rounded,
            title: _isAdmin ? 'Расширенные функции активны' : 'Обычный пользовательский режим',
            subtitle: _isAdmin
                ? 'Можно подтверждать и исправлять результаты анализа.'
                : 'Включи режим администратора, чтобы получить расширенные инструменты.',
            compact: true,
          ),
        ],
      ),
    );
  }
}
