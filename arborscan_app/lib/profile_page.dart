import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:google_sign_in/google_sign_in.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:url_launcher/url_launcher.dart';
import 'onboarding_page.dart';

import 'app_theme.dart';
import 'api_config.dart';

class ProfilePage extends StatefulWidget {
  final VoidCallback? onAuthChanged;

  const ProfilePage({
    super.key,
    this.onAuthChanged,
  });

  @override
  State<ProfilePage> createState() => _ProfilePageState();
}

class _ProfilePageState extends State<ProfilePage> {
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

  final _nameController = TextEditingController();
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();

  late final GoogleSignIn _googleSignIn = GoogleSignIn(
    scopes: ['email', 'profile'],
    serverClientId: ApiConfig.googleWebClientId,
  );

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
  String _avatarUrl = '';

  int _totalAnalyses = 0;
  int _geoAnalyses = 0;
  int _highRiskAnalyses = 0;
  double? _avgRisk;
  Map<String, dynamic>? _lastAnalysis;

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
    super.dispose();
  }

  Uri _uri(String path, [Map<String, String>? query]) {
    return Uri.parse('${ApiConfig.baseUrl}$path').replace(queryParameters: query);
  }

  Future<Map<String, String>> _requestHeaders({
    bool jsonBody = false,
    bool useAuth = false,
  }) async {
    final headers = <String, String>{
      'Accept': 'application/json',
      if (jsonBody) 'Content-Type': 'application/json',
    };

    if (useAuth) {
      final prefs = await SharedPreferences.getInstance();
      final token = prefs.getString(_tokenKey)?.trim() ?? _token.trim();
      if (token.isEmpty) {
        throw Exception('Сессия отсутствует. Войдите снова.');
      }
      headers['Authorization'] = 'Bearer $token';
    }
    return headers;
  }

  Future<Map<String, dynamic>> _postJson(
    String path,
    Map<String, dynamic>? body, {
    bool useAuth = false,
  }) async {
    final res = await http
        .post(
          _uri(path),
          headers: await _requestHeaders(
            jsonBody: body != null,
            useAuth: useAuth,
          ),
          body: body == null ? null : jsonEncode(body),
        )
        .timeout(const Duration(seconds: 15));

    final decoded = res.bodyBytes.isEmpty
        ? <String, dynamic>{}
        : jsonDecode(utf8.decode(res.bodyBytes));
    final data = decoded is Map<String, dynamic>
        ? decoded
        : (decoded is Map ? decoded.cast<String, dynamic>() : <String, dynamic>{});

    if (res.statusCode < 200 || res.statusCode >= 300) {
      throw Exception(
        'HTTP ${res.statusCode}: '
        '${data['detail']?.toString() ?? 'Ошибка сервера'}',
      );
    }
    return data;
  }

  Future<Map<String, dynamic>> _getJson(
    String path, {
    Map<String, String>? query,
    bool useAuth = false,
  }) async {
    final res = await http
        .get(
          _uri(path, query),
          headers: await _requestHeaders(useAuth: useAuth),
        )
        .timeout(const Duration(seconds: 15));

    final decoded = res.bodyBytes.isEmpty
        ? <String, dynamic>{}
        : jsonDecode(utf8.decode(res.bodyBytes));
    final data = decoded is Map<String, dynamic>
        ? decoded
        : (decoded is Map ? decoded.cast<String, dynamic>() : <String, dynamic>{});

    if (res.statusCode < 200 || res.statusCode >= 300) {
      throw Exception(
        'HTTP ${res.statusCode}: '
        '${data['detail']?.toString() ?? 'Ошибка сервера'}',
      );
    }
    return data;
  }

  Future<void> _clearInvalidSession(String message) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_tokenKey);
    await prefs.remove(_expiresAtKey);
    await prefs.setBool(_loggedInKey, false);
    await prefs.setBool(_adminFlagKey, false);

    if (!mounted) return;
    setState(() {
      _loggedIn = false;
      _isAdmin = false;
      _serverOnline = false;
      _role = 'user';
      _token = '';
      _avatarUrl = '';
      _totalAnalyses = 0;
      _geoAnalyses = 0;
      _highRiskAnalyses = 0;
      _avgRisk = null;
      _lastAnalysis = null;
      _statusText = message;
    });
    widget.onAuthChanged?.call();
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
        final data = await _getJson('/auth/me', useAuth: true);
        await _applyAuthData(data, tokenFromResponse: token);
        await _loadStats();
        if (!mounted) return;
        setState(() {
          _serverOnline = true;
          _statusText = 'Сессия подтверждена сервером.';
        });
      } catch (e) {
        final msg = e.toString();
        if (msg.contains('401') || msg.contains('Сессия') || msg.contains('Unauthorized')) {
          await _clearInvalidSession(
            'Сессия истекла или была сброшена после обновления сервера. Войдите снова.',
          );
        } else {
          if (!mounted) return;
          setState(() {
            _serverOnline = false;
            _statusText = _loggedIn
                ? 'Сервер недоступен. Используется сохранённый профиль.'
                : 'Войдите или зарегистрируйтесь.';
          });
        }
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
    final avatarUrl = user['avatar_url']?.toString() ?? '';
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
      _avatarUrl = avatarUrl;
      _nameController.text = name;
      _emailController.text = email;
      _passwordController.clear();
    });

    widget.onAuthChanged?.call();
    await _loadStats();
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
      _snack('Профиль создан. Добро пожаловать!');
      
      // ПОКАЗЫВАЕМ ОБУЧЕНИЕ ПОСЛЕ РЕГИСТРАЦИИ!
      if (mounted) {
        Navigator.of(context).push(MaterialPageRoute(builder: (_) => const OnboardingPage()));
      }
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
    try {
      if (_token.isNotEmpty) {
        await _postJson('/auth/logout', null, useAuth: true);
      }
    } catch (_) {
      // Локальный выход должен работать даже при недоступном сервере.
    }

    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_loggedInKey, false);
    await prefs.setBool(_adminFlagKey, false);
    await prefs.setString(_roleKey, 'user');
    await prefs.remove(_tokenKey);
    await prefs.remove(_expiresAtKey);

    if (!mounted) return;
    setState(() {
      _loggedIn = false;
      _isAdmin = false;
      _role = 'user';
      _token = '';
      _passwordController.clear();
      _statusText = 'Вы вышли из профиля.';
      _avatarUrl = '';
      _totalAnalyses = 0;
      _geoAnalyses = 0;
      _highRiskAnalyses = 0;
      _avgRisk = null;
      _lastAnalysis = null;
    });
    widget.onAuthChanged?.call();
    _snack('Вы вышли из профиля.');
  }

  Future<void> _deleteLocalSession() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_tokenKey);
    await prefs.remove(_expiresAtKey);
    await prefs.setBool(_loggedInKey, false);
    await prefs.setBool(_adminFlagKey, false);
    await prefs.setString(_roleKey, 'user');

    if (!mounted) return;
    setState(() {
      _loggedIn = false;
      _isAdmin = false;
      _role = 'user';
      _token = '';
      _statusText = 'Локальная сессия очищена.';
      _avatarUrl = '';
      _totalAnalyses = 0;
      _geoAnalyses = 0;
      _highRiskAnalyses = 0;
      _avgRisk = null;
      _lastAnalysis = null;
    });
    widget.onAuthChanged?.call();
    _snack('Локальная сессия очищена. Аккаунт на сервере не удалён.');
  }

  Future<void> _loginWithGoogle() async {
    setState(() => _busy = true);
    try {
      await _clearInvalidSession('Выполняется вход через Google...');
      await _googleSignIn.signOut(); 
      final account = await _googleSignIn.signIn();
      if (account == null) {
        _snack('Вход через Google отменён.');
        return;
      }

      final auth = await account.authentication;
      final idToken = auth.idToken;
      if (idToken == null || idToken.isEmpty) {
        _snack('Google не вернул idToken. Проверьте OAuth Client ID.');
        return;
      }

      final data = await _postJson('/auth/google', {
        'id_token': idToken,
        'email': account.email,
        'name': account.displayName ?? account.email,
        'photo_url': account.photoUrl,
      });

      await _applyAuthData(data);
      if (!mounted) return;
      setState(() {
        _serverOnline = true;
        _statusText = 'Вход выполнен через Google.';
      });
      _snack('Вы вошли через Google.');
    } catch (e) {
      _snack(e.toString().replaceFirst('Exception: ', ''));
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _loadStats() async {
    if (_token.isEmpty) return;
    try {
      final data = await _getJson('/profile/stats', useAuth: true);
      final stats = (data['stats'] as Map?)?.cast<String, dynamic>() ?? {};
      final user = (data['user'] as Map?)?.cast<String, dynamic>() ?? {};

      if (!mounted) return;
      setState(() {
        _avatarUrl = user['avatar_url']?.toString() ?? _avatarUrl;
        _totalAnalyses = (stats['total_analyses'] as num?)?.toInt() ?? 0;
        _geoAnalyses = (stats['with_geo'] as num?)?.toInt() ?? 0;
        _highRiskAnalyses = (stats['high_risk_count'] as num?)?.toInt() ?? 0;
        _avgRisk = (stats['avg_risk'] as num?)?.toDouble();
        _lastAnalysis = (stats['last_analysis'] as Map?)?.cast<String, dynamic>();
      });
    } catch (_) {
      // Статистика профиля не должна ломать вход.
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
                    if (_loggedIn) ...[
                      const SizedBox(height: 14),
                      _buildStatsCard(context),
                    ],
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

  Widget _buildAvatar({double size = 58}) {
    final hasAvatar = _avatarUrl.trim().isNotEmpty;
    return Container(
      width: size,
      height: size,
      decoration: BoxDecoration(
        color: AppTheme.primary.withOpacity(0.12),
        shape: BoxShape.circle,
        border: Border.all(color: AppTheme.primary.withOpacity(0.30)),
      ),
      clipBehavior: Clip.antiAlias,
      child: hasAvatar
          ? Image.network(
              _avatarUrl,
              fit: BoxFit.cover,
              errorBuilder: (_, __, ___) => const Icon(
                Icons.account_circle,
                color: AppTheme.primary,
                size: 34,
              ),
            )
          : const Icon(
              Icons.account_circle,
              color: AppTheme.primary,
              size: 34,
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
          SizedBox(
            width: double.infinity,
            child: OutlinedButton.icon(
              onPressed: _busy ? null : _loginWithGoogle,
              icon: const Icon(Icons.g_mobiledata, size: 28),
              label: const Text('Войти через Google'),
            ),
          ),
          const SizedBox(height: 10),
          Row(
            children: [
              Expanded(child: Divider(color: AppTheme.border)),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 10),
                child: Text(
                  'или',
                  style: TextStyle(color: AppTheme.muted, fontWeight: FontWeight.w700),
                ),
              ),
              Expanded(child: Divider(color: AppTheme.border)),
            ],
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
          _buildAvatar(size: 48),
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
                    if (_isAdmin)
                      Ui.badge(
                        text: 'Администратор',
                        color: AppTheme.primary,
                        icon: Icons.admin_panel_settings,
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

  Widget _buildStatsCard(BuildContext context) {
    final avgRiskText = _avgRisk == null ? '—' : _avgRisk!.toStringAsFixed(2);
    final last = _lastAnalysis;
    final lastSpecies = last?['species']?.toString() ?? '—';
    final lastRisk = (last?['risk_index'] as num?)?.toDouble();
    final lastRiskCategory = last?['risk_category']?.toString();

    return Ui.paddedCard(
      context,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.insights_outlined, color: AppTheme.primary),
              const SizedBox(width: 10),
              Expanded(
                child: Text(
                  'Мои анализы',
                  style: Theme.of(context).textTheme.titleMedium,
                ),
              ),
              IconButton(
                tooltip: 'Обновить статистику',
                onPressed: _loadStats,
                icon: const Icon(Icons.refresh),
              ),
            ],
          ),
          const SizedBox(height: 12),
          LayoutBuilder(
            builder: (context, constraints) {
              final twoCols = constraints.maxWidth > 430;
              final cards = [
                _statTile('Всего', _totalAnalyses.toString(), Icons.analytics_outlined),
                _statTile('На карте', _geoAnalyses.toString(), Icons.location_on_outlined),
                _statTile('Высокий риск', _highRiskAnalyses.toString(), Icons.warning_amber),
                _statTile('Средний риск', avgRiskText, Icons.speed_outlined),
              ];

              if (twoCols) {
                return GridView.count(
                  crossAxisCount: 2,
                  mainAxisSpacing: 10,
                  crossAxisSpacing: 10,
                  childAspectRatio: 2.6,
                  shrinkWrap: true,
                  physics: const NeverScrollableScrollPhysics(),
                  children: cards,
                );
              }

              return Column(
                children: [
                  for (final c in cards) ...[
                    c,
                    const SizedBox(height: 8),
                  ],
                ],
              );
            },
          ),
          const SizedBox(height: 12),
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: AppTheme.surface2,
              borderRadius: BorderRadius.circular(18),
              border: Border.all(color: AppTheme.border),
            ),
            child: Row(
              children: [
                const Icon(Icons.history, color: AppTheme.primary),
                const SizedBox(width: 10),
                Expanded(
                  child: last == null
                      ? const Text(
                          'Пока нет серверных анализов.',
                          style: TextStyle(
                            color: AppTheme.muted,
                            fontWeight: FontWeight.w700,
                          ),
                        )
                      : Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              'Последний анализ: $lastSpecies',
                              style: const TextStyle(
                                color: AppTheme.text,
                                fontWeight: FontWeight.w900,
                              ),
                            ),
                            const SizedBox(height: 4),
                            Text(
                              'Риск: ${lastRisk?.toStringAsFixed(2) ?? '—'}'
                              '${lastRiskCategory == null ? '' : ' · $lastRiskCategory'}',
                              style: const TextStyle(
                                color: AppTheme.muted,
                                fontWeight: FontWeight.w700,
                              ),
                            ),
                          ],
                        ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _statTile(String title, String value, IconData icon) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: AppTheme.surface2,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: AppTheme.border),
      ),
      child: Row(
        children: [
          Icon(icon, color: AppTheme.primary),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              title,
              style: const TextStyle(
                color: AppTheme.muted,
                fontSize: 12,
                fontWeight: FontWeight.w700,
              ),
            ),
          ),
          Text(
            value,
            style: const TextStyle(
              color: AppTheme.text,
              fontSize: 18,
              fontWeight: FontWeight.w900,
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