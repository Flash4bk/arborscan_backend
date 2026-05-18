import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import 'package:url_launcher/url_launcher.dart';

import 'app_theme.dart';

class ProfilePage extends StatefulWidget {
  const ProfilePage({super.key});

  @override
  State<ProfilePage> createState() => _ProfilePageState();
}

class _ProfilePageState extends State<ProfilePage> {
  static const String _baseUrl =
      'https://arborscanbackend-production.up.railway.app';

  static const String _adminFlagKey = 'arborscan_is_admin';

  static const String _nameKey = 'arborscan_profile_name';
  static const String _emailKey = 'arborscan_profile_email';
  static const String _loggedInKey = 'arborscan_profile_logged_in';
  static const String _roleKey = 'arborscan_profile_role';
  static const String _tokenKey = 'arborscan_auth_token';
  static const String _expiresAtKey = 'arborscan_auth_expires_at';

  static const String _appName = 'ArborScan';
  static const String _appVersion = '1.0.0 beta';
  static const String _developerEmail = 'danik.alshkevich@gmail.com';
  static const String _adminPasscode = '8426';

  final _nameController = TextEditingController();
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();
  final _adminCodeController = TextEditingController();

  bool _loading = true;
  bool _busy = false;
  bool _isRegisterMode = true;
  bool _loggedIn = false;
  bool _isAdmin = false;
  bool _serverOnline = false;

  String _name = '';
  String _email = '';
  String _role = 'user';
  String _token = '';
  String _statusText = 'Проверка профиля...';

  @override
  void initState() {
    super.initState();
    _loadProfile();
  }

  @override
  void dispose() {
    _nameController.dispose();
    _emailController.dispose();
    _passwordController.dispose();
    _adminCodeController.dispose();
    super.dispose();
  }

  Uri _uri(String path, [Map<String, String>? query]) {
    return Uri.parse('$_baseUrl$path').replace(queryParameters: query);
  }

  Future<Map<String, dynamic>> _postJson(
    String path,
    Map<String, dynamic> body,
  ) async {
    final res = await http
        .post(
          _uri(path),
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode(body),
        )
        .timeout(const Duration(seconds: 15));

    final data = jsonDecode(utf8.decode(res.bodyBytes)) as Map<String, dynamic>;
    if (res.statusCode < 200 || res.statusCode >= 300) {
      throw Exception(data['detail']?.toString() ?? 'Ошибка сервера');
    }
    return data;
  }

  Future<Map<String, dynamic>> _getJson(
    String path,
    Map<String, String> query,
  ) async {
    final res = await http
        .get(_uri(path, query))
        .timeout(const Duration(seconds: 15));

    final data = jsonDecode(utf8.decode(res.bodyBytes)) as Map<String, dynamic>;
    if (res.statusCode < 200 || res.statusCode >= 300) {
      throw Exception(data['detail']?.toString() ?? 'Ошибка сервера');
    }
    return data;
  }

  Future<void> _loadProfile() async {
    final prefs = await SharedPreferences.getInstance();
    final token = prefs.getString(_tokenKey) ?? '';
    final storedName = prefs.getString(_nameKey) ?? '';
    final storedEmail = prefs.getString(_emailKey) ?? '';
    final storedRole = prefs.getString(_roleKey) ?? 'user';

    setState(() {
      _name = storedName;
      _email = storedEmail;
      _role = storedRole;
      _token = token;
      _loggedIn = (prefs.getBool(_loggedInKey) ?? false) && token.isNotEmpty;
      _isAdmin = prefs.getBool(_adminFlagKey) ?? storedRole == 'admin';
      _nameController.text = storedName;
      _emailController.text = storedEmail;
    });

    if (token.isNotEmpty) {
      try {
        final data = await _getJson('/auth/me', {'token': token});
        await _applyAuthData(data, tokenFromResponse: token);
        if (!mounted) return;
        setState(() {
          _serverOnline = true;
          _statusText = 'Сессия подтверждена сервером.';
        });
      } catch (_) {
        if (!mounted) return;
        setState(() {
          _serverOnline = false;
          _statusText = _loggedIn
              ? 'Сервер недоступен. Используется сохранённый профиль.'
              : 'Войдите или зарегистрируйтесь.';
        });
      }
    } else {
      setState(() {
        _statusText = 'Войдите или зарегистрируйтесь.';
      });
    }

    if (!mounted) return;
    setState(() => _loading = false);
  }

  Future<void> _applyAuthData(
    Map<String, dynamic> data, {
    String? tokenFromResponse,
  }) async {
    final user = (data['user'] as Map?)?.cast<String, dynamic>() ?? {};
    final token = (data['token'] ?? tokenFromResponse ?? _token).toString();
    final expiresAt = data['expires_at']?.toString() ?? '';

    final name = user['name']?.toString() ?? '';
    final email = user['email']?.toString() ?? '';
    final role = user['role']?.toString() ?? 'user';
    final isAdmin = role == 'admin';

    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_nameKey, name);
    await prefs.setString(_emailKey, email);
    await prefs.setString(_roleKey, role);
    await prefs.setString(_tokenKey, token);
    await prefs.setString(_expiresAtKey, expiresAt);
    await prefs.setBool(_loggedInKey, true);
    await prefs.setBool(_adminFlagKey, isAdmin);

    if (!mounted) return;
    setState(() {
      _name = name;
      _email = email;
      _role = role;
      _token = token;
      _loggedIn = true;
      _isAdmin = isAdmin;
      _nameController.text = name;
      _emailController.text = email;
      _passwordController.clear();
    });
  }

  String? _validateEmail(String value) {
    final email = value.trim();
    final ok = RegExp(r'^[^@\s]+@[^@\s]+\.[^@\s]+$').hasMatch(email);
    if (!ok) return 'Введите корректную почту.';
    return null;
  }

  Future<void> _register() async {
    final name = _nameController.text.trim();
    final email = _emailController.text.trim();
    final password = _passwordController.text;

    if (name.length < 2) {
      _snack('Введите имя не короче 2 символов.');
      return;
    }
    final emailError = _validateEmail(email);
    if (emailError != null) {
      _snack(emailError);
      return;
    }
    if (password.length < 4) {
      _snack('Пароль должен быть не короче 4 символов.');
      return;
    }

    setState(() => _busy = true);
    try {
      final data = await _postJson('/auth/register', {
        'name': name,
        'email': email,
        'password': password,
      });
      await _applyAuthData(data);
      if (!mounted) return;
      setState(() {
        _serverOnline = true;
        _statusText = 'Профиль создан на сервере.';
      });
      _snack('Профиль создан. Вы вошли как пользователь.');
    } catch (e) {
      _snack(e.toString().replaceFirst('Exception: ', ''));
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _login() async {
    final email = _emailController.text.trim();
    final password = _passwordController.text;

    final emailError = _validateEmail(email);
    if (emailError != null) {
      _snack(emailError);
      return;
    }
    if (password.length < 4) {
      _snack('Введите пароль.');
      return;
    }

    setState(() => _busy = true);
    try {
      final data = await _postJson('/auth/login', {
        'email': email,
        'password': password,
      });
      await _applyAuthData(data);
      if (!mounted) return;
      setState(() {
        _serverOnline = true;
        _statusText = 'Вход выполнен через сервер.';
      });
      _snack(_isAdmin ? 'Вы вошли как администратор.' : 'Вы вошли как пользователь.');
    } catch (e) {
      _snack(e.toString().replaceFirst('Exception: ', ''));
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _logout() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_loggedInKey, false);
    await prefs.setBool(_adminFlagKey, false);
    await prefs.remove(_tokenKey);

    if (!mounted) return;
    setState(() {
      _loggedIn = false;
      _isAdmin = false;
      _role = 'user';
      _token = '';
      _passwordController.clear();
      _adminCodeController.clear();
      _statusText = 'Вы вышли из профиля.';
    });
    _snack('Вы вышли из профиля.');
  }

  Future<void> _deleteLocalSession() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_tokenKey);
    await prefs.remove(_expiresAtKey);
    await prefs.setBool(_loggedInKey, false);
    await prefs.setBool(_adminFlagKey, false);

    if (!mounted) return;
    setState(() {
      _loggedIn = false;
      _isAdmin = false;
      _token = '';
      _statusText = 'Локальная сессия очищена.';
    });
    _snack('Локальная сессия очищена. Аккаунт на сервере не удалён.');
  }

  Future<void> _setRole(String role) async {
    if (!_loggedIn || _token.isEmpty) {
      _snack('Сначала войдите или зарегистрируйтесь.');
      return;
    }

    final adminCode = _adminCodeController.text.trim();
    if (role == 'admin' && adminCode.isEmpty) {
      _snack('Введите код администратора.');
      return;
    }

    setState(() => _busy = true);
    try {
      final data = await _postJson('/auth/set-role', {
        'token': _token,
        'role': role,
        'admin_code': role == 'admin' ? adminCode : null,
      });
      await _applyAuthData(data, tokenFromResponse: _token);
      if (!mounted) return;
      _snack(role == 'admin'
          ? 'Роль администратора включена.'
          : 'Роль пользователя включена.');
    } catch (e) {
      _snack(e.toString().replaceFirst('Exception: ', ''));
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _openDeveloperEmail() async {
    final uri = Uri(
      scheme: 'mailto',
      path: _developerEmail,
      queryParameters: {'subject': 'ArborScan: обратная связь'},
    );

    try {
      await launchUrl(uri, mode: LaunchMode.externalApplication);
    } catch (_) {
      _snack('Почтовое приложение не найдено. Почта: $_developerEmail');
    }
  }

  void _snack(String text) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(text)));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Профиль'),
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : Stack(
              children: [
                ListView(
                  padding: const EdgeInsets.all(16),
                  children: [
                    _buildAppHeader(context),
                    const SizedBox(height: 14),
                    _buildDeveloperCard(context),
                    const SizedBox(height: 14),
                    _loggedIn ? _buildAccountCard(context) : _buildAuthCard(context),
                    const SizedBox(height: 14),
                    _buildRoleCard(context),
                    const SizedBox(height: 14),
                    _buildInfoCard(context),
                  ],
                ),
                if (_busy)
                  Container(
                    color: Colors.black.withOpacity(0.20),
                    child: const Center(child: CircularProgressIndicator()),
                  ),
              ],
            ),
    );
  }

  Widget _buildAppHeader(BuildContext context) {
    return Ui.paddedCard(
      context,
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 58,
            height: 58,
            decoration: BoxDecoration(
              color: AppTheme.primary.withOpacity(0.12),
              borderRadius: BorderRadius.circular(18),
              border: Border.all(color: AppTheme.primary.withOpacity(0.22)),
            ),
            child: const Icon(Icons.forest, color: AppTheme.primary, size: 32),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  _appName,
                  style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                        fontWeight: FontWeight.w900,
                      ),
                ),
                const SizedBox(height: 3),
                Text(
                  'Мобильная оценка состояния и риска повреждения деревьев',
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                        color: AppTheme.muted,
                      ),
                ),
                const SizedBox(height: 10),
                Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  children: [
                    Ui.badge(
                      text: 'Версия $_appVersion',
                      color: AppTheme.primary,
                      icon: Icons.info_outline,
                    ),
                    Ui.badge(
                      text: _serverOnline ? 'Сервер' : 'Локально',
                      color: _serverOnline ? AppTheme.success : AppTheme.warning,
                      icon: _serverOnline ? Icons.cloud_done : Icons.storage,
                    ),
                    Ui.badge(
                      text: _loggedIn ? 'Профиль активен' : 'Гость',
                      color: _loggedIn ? AppTheme.success : AppTheme.warning,
                      icon: _loggedIn ? Icons.verified_user : Icons.person_outline,
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                Text(
                  _statusText,
                  style: Theme.of(context).textTheme.bodySmall?.copyWith(
                        color: AppTheme.muted,
                        fontWeight: FontWeight.w700,
                      ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildDeveloperCard(BuildContext context) {
    return Ui.paddedCard(
      context,
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(Icons.support_agent, color: AppTheme.primary),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('Связь с разработчиком', style: Theme.of(context).textTheme.titleMedium),
                const SizedBox(height: 6),
                SelectableText(
                  _developerEmail,
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                        color: AppTheme.primary,
                        fontWeight: FontWeight.w800,
                      ),
                ),
                const SizedBox(height: 10),
                OutlinedButton.icon(
                  onPressed: _openDeveloperEmail,
                  icon: const Icon(Icons.email_outlined),
                  label: const Text('Написать разработчику'),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildAuthCard(BuildContext context) {
    return Ui.paddedCard(
      context,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.login, color: AppTheme.primary),
              const SizedBox(width: 10),
              Expanded(
                child: Text(
                  _isRegisterMode ? 'Регистрация' : 'Вход',
                  style: Theme.of(context).textTheme.titleMedium,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          SegmentedButton<bool>(
            segments: const [
              ButtonSegment<bool>(
                value: true,
                label: Text('Регистрация'),
                icon: Icon(Icons.person_add_alt),
              ),
              ButtonSegment<bool>(
                value: false,
                label: Text('Вход'),
                icon: Icon(Icons.lock_open),
              ),
            ],
            selected: {_isRegisterMode},
            onSelectionChanged: (v) => setState(() => _isRegisterMode = v.first),
          ),
          const SizedBox(height: 14),
          if (_isRegisterMode) ...[
            TextField(
              controller: _nameController,
              decoration: const InputDecoration(
                labelText: 'Имя',
                prefixIcon: Icon(Icons.person_outline),
              ),
              textInputAction: TextInputAction.next,
            ),
            const SizedBox(height: 10),
          ],
          TextField(
            controller: _emailController,
            decoration: const InputDecoration(
              labelText: 'Почта',
              prefixIcon: Icon(Icons.alternate_email),
            ),
            keyboardType: TextInputType.emailAddress,
            textInputAction: TextInputAction.next,
          ),
          const SizedBox(height: 10),
          TextField(
            controller: _passwordController,
            decoration: const InputDecoration(
              labelText: 'Пароль',
              prefixIcon: Icon(Icons.password),
            ),
            obscureText: true,
            onSubmitted: (_) => _isRegisterMode ? _register() : _login(),
          ),
          const SizedBox(height: 14),
          SizedBox(
            width: double.infinity,
            child: FilledButton.icon(
              onPressed: _busy ? null : (_isRegisterMode ? _register : _login),
              icon: Icon(_isRegisterMode ? Icons.person_add_alt : Icons.login),
              label: Text(_isRegisterMode ? 'Создать профиль' : 'Войти'),
            ),
          ),
          const SizedBox(height: 8),
          Text(
            'Регистрация выполняется через сервер ArborScan. Пароль хранится на сервере в виде хэша.',
            style: Theme.of(context).textTheme.bodySmall?.copyWith(color: AppTheme.muted),
          ),
        ],
      ),
    );
  }

  Widget _buildAccountCard(BuildContext context) {
    return Ui.paddedCard(
      context,
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(Icons.account_circle, color: AppTheme.primary, size: 34),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  _name.isNotEmpty ? _name : 'Пользователь ArborScan',
                  style: Theme.of(context).textTheme.titleLarge,
                ),
                const SizedBox(height: 4),
                SelectableText(
                  _email,
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                        color: AppTheme.muted,
                        fontWeight: FontWeight.w700,
                      ),
                ),
                const SizedBox(height: 10),
                Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  children: [
                    Ui.badge(
                      text: _isAdmin ? 'Администратор' : 'Пользователь',
                      color: _isAdmin ? AppTheme.primary : AppTheme.success,
                      icon: _isAdmin ? Icons.admin_panel_settings : Icons.person,
                    ),
                    Ui.badge(
                      text: _serverOnline ? 'Серверная сессия' : 'Локальная копия',
                      color: _serverOnline ? AppTheme.success : AppTheme.warning,
                      icon: _serverOnline ? Icons.cloud_done : Icons.storage,
                    ),
                  ],
                ),
                const SizedBox(height: 14),
                Row(
                  children: [
                    Expanded(
                      child: OutlinedButton.icon(
                        onPressed: _logout,
                        icon: const Icon(Icons.logout),
                        label: const Text('Выйти'),
                      ),
                    ),
                    const SizedBox(width: 10),
                    Expanded(
                      child: TextButton.icon(
                        onPressed: _deleteLocalSession,
                        icon: const Icon(Icons.cleaning_services_outlined),
                        label: const Text('Очистить'),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildRoleCard(BuildContext context) {
    return Ui.paddedCard(
      context,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.manage_accounts_outlined, color: AppTheme.primary),
              const SizedBox(width: 10),
              Expanded(
                child: Text('Роль в приложении', style: Theme.of(context).textTheme.titleMedium),
              ),
            ],
          ),
          const SizedBox(height: 10),
          Text(
            'Роль хранится на сервере и управляет доступом к админ-функциям.',
            style: Theme.of(context).textTheme.bodySmall?.copyWith(color: AppTheme.muted),
          ),
          const SizedBox(height: 14),
          SegmentedButton<String>(
            segments: const [
              ButtonSegment<String>(
                value: 'user',
                label: Text('Пользователь'),
                icon: Icon(Icons.person_outline),
              ),
              ButtonSegment<String>(
                value: 'admin',
                label: Text('Администратор'),
                icon: Icon(Icons.admin_panel_settings_outlined),
              ),
            ],
            selected: {_isAdmin ? 'admin' : 'user'},
            onSelectionChanged: _loggedIn
                ? (v) {
                    final role = v.first;
                    if (role == 'user') {
                      _setRole('user');
                    }
                  }
                : null,
          ),
          const SizedBox(height: 12),
          TextField(
            controller: _adminCodeController,
            enabled: _loggedIn,
            obscureText: true,
            keyboardType: TextInputType.number,
            decoration: const InputDecoration(
              labelText: 'Код администратора',
              prefixIcon: Icon(Icons.lock_outline),
              helperText: 'Введите код и нажмите кнопку ниже, чтобы включить роль администратора.',
            ),
          ),
          const SizedBox(height: 10),
          SizedBox(
            width: double.infinity,
            child: FilledButton.icon(
              onPressed: _loggedIn && !_busy ? () => _setRole('admin') : null,
              icon: const Icon(Icons.admin_panel_settings),
              label: const Text('Включить администратора'),
            ),
          ),
          const SizedBox(height: 10),
          Text(
            _loggedIn
                ? (_isAdmin ? 'Текущий доступ: администратор.' : 'Текущий доступ: пользователь.')
                : 'Для выбора роли сначала создайте профиль или войдите.',
            style: Theme.of(context).textTheme.bodySmall?.copyWith(
                  color: _loggedIn ? AppTheme.muted : AppTheme.warning,
                  fontWeight: FontWeight.w700,
                ),
          ),
        ],
      ),
    );
  }

  Widget _buildInfoCard(BuildContext context) {
    return Ui.paddedCard(
      context,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('О приложении', style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: 10),
          _infoRow(Icons.analytics_outlined, 'AI-анализ', 'Параметры дерева и факторы риска.'),
          _infoRow(Icons.view_in_ar_outlined, 'AR-измерения', 'Высота, крона и диаметр по 6 точкам.'),
          _infoRow(Icons.map_outlined, 'Карта', 'GPS-точки анализов, спутник, 3D и Street View.'),
          _infoRow(Icons.science_outlined, 'β-аналитика', 'Ветровая нагрузка и момент у основания.'),
        ],
      ),
    );
  }

  Widget _infoRow(IconData icon, String title, String text) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, color: AppTheme.primary, size: 20),
          const SizedBox(width: 10),
          Expanded(
            child: RichText(
              text: TextSpan(
                style: Theme.of(context).textTheme.bodyMedium,
                children: [
                  TextSpan(text: '$title: ', style: const TextStyle(fontWeight: FontWeight.w900)),
                  TextSpan(text: text, style: const TextStyle(color: AppTheme.muted)),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
