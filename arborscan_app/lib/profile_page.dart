import 'app_navigation.dart';
import 'dart:async';
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;
import 'package:google_sign_in/google_sign_in.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:url_launcher/url_launcher.dart';
import 'onboarding_page.dart';

import 'app_theme.dart';
import 'model_quality_page.dart';
import 'api_config.dart';
import 'admin_panel_page.dart';
import 'saved_corrections_page.dart';
import 'profile_session_guard.dart';

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
  static const String _appVersion = '1.3.0';
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
    AppNavigation.profileVisits.addListener(_loadStats);
    _loadProfile();
  }

  @override
  void dispose() {
    AppNavigation.profileVisits.removeListener(_loadStats);
    _nameController.dispose();
    _emailController.dispose();
    _passwordController.dispose();
    super.dispose();
  }

  Uri _uri(String path, [Map<String, String>? query]) {
    return ApiConfig.v3(path).replace(queryParameters: query);
  }

  Future<Map<String, String>> _requestHeaders({
    bool jsonBody = false,
    bool useAuth = false,
    String? authToken,
  }) async {
    final headers = <String, String>{
      'Accept': 'application/json',
      if (jsonBody) 'Content-Type': 'application/json',
    };

    if (useAuth) {
      final prefs = await SharedPreferences.getInstance();
      final token = authToken?.trim() ??
          prefs.getString(_tokenKey)?.trim() ??
          _token.trim();
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
    String? authToken,
  }) async {
    final res = await http
        .post(
          _uri(path),
          headers: await _requestHeaders(
            jsonBody: body != null,
            useAuth: useAuth,
            authToken: authToken,
          ),
          body: body == null ? null : jsonEncode(body),
        )
        .timeout(const Duration(seconds: 15));

    final decoded = res.bodyBytes.isEmpty
        ? <String, dynamic>{}
        : jsonDecode(utf8.decode(res.bodyBytes));
    final data = decoded is Map<String, dynamic>
        ? decoded
        : (decoded is Map
            ? decoded.cast<String, dynamic>()
            : <String, dynamic>{});

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
    String? authToken,
  }) async {
    final res = await http
        .get(
          _uri(path, query),
          headers:
              await _requestHeaders(useAuth: useAuth, authToken: authToken),
        )
        .timeout(const Duration(seconds: 15));

    final decoded = res.bodyBytes.isEmpty
        ? <String, dynamic>{}
        : jsonDecode(utf8.decode(res.bodyBytes));
    final data = decoded is Map<String, dynamic>
        ? decoded
        : (decoded is Map
            ? decoded.cast<String, dynamic>()
            : <String, dynamic>{});

    if (res.statusCode < 200 || res.statusCode >= 300) {
      throw Exception(
        'HTTP ${res.statusCode}: '
        '${data['detail']?.toString() ?? 'Ошибка сервера'}',
      );
    }
    return data;
  }

  Future<bool> _clearInvalidSession(
      String message, ProfileSessionTicket ticket) async {
    final cleared = await ProfileSessionGuard.write(ticket, (prefs) async {
      await prefs.remove(_tokenKey);
      await prefs.remove('arborscan_user_id');
      await prefs.remove(_expiresAtKey);
      await prefs.setBool(_loggedInKey, false);
      await prefs.setBool(_adminFlagKey, false);
      await prefs.setString(_roleKey, 'user');
    }, active: () => mounted);

    if (!cleared || !mounted) return false;
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
    return true;
  }

  Future<void> _loadProfile() async {
    final ticket = await ProfileSessionGuard.capture();
    final prefs = await SharedPreferences.getInstance();
    if (!mounted) return;
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
        final data =
            await _getJson('/auth/me', useAuth: true, authToken: token);
        if (!await _applyAuthData(data,
            ticket: ticket, tokenFromResponse: token)) {
          _discardStaleProfile();
          return;
        }
        if (!mounted) return;
        setState(() {
          _serverOnline = true;
          _statusText = 'Сессия подтверждена сервером.';
        });
      } catch (e) {
        final msg = e.toString();
        if (msg.contains('401') ||
            msg.contains('Сессия') ||
            msg.contains('Unauthorized')) {
          final cleared = await _clearInvalidSession(
            'Сессия истекла или была сброшена после обновления сервера. Войдите снова.',
            ticket,
          );
          if (!cleared) _discardStaleProfile();
        } else {
          if (!await ProfileSessionGuard.current(ticket,
              active: () => mounted)) {
            _discardStaleProfile();
            return;
          }
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

  void _discardStaleProfile() {
    if (!mounted) return;
    setState(() {
      _loading = false;
      _loggedIn = false;
      _isAdmin = false;
      _serverOnline = false;
      _name = '';
      _email = '';
      _role = 'user';
      _token = '';
      _avatarUrl = '';
      _nameController.clear();
      _emailController.clear();
      _passwordController.clear();
      _totalAnalyses = 0;
      _geoAnalyses = 0;
      _highRiskAnalyses = 0;
      _avgRisk = null;
      _lastAnalysis = null;
      _statusText = 'Профиль изменился. Откройте вкладку снова.';
    });
  }

  Future<bool> _applyAuthData(
    Map<String, dynamic> data, {
    required ProfileSessionTicket ticket,
    String? tokenFromResponse,
  }) async {
    if (data['user'] is! Map) {
      throw const FormatException('Некорректный ответ авторизации.');
    }
    final user = (data['user'] as Map).cast<String, dynamic>();
    final rawToken = data['token'] ?? tokenFromResponse;
    if (rawToken is! String ||
        rawToken.trim().isEmpty ||
        user['id'] is! String ||
        (user['id'] as String).trim().isEmpty) {
      throw const FormatException('Некорректный ответ авторизации.');
    }
    final token = rawToken;
    if (tokenFromResponse != null &&
        ticket.owner.isNotEmpty &&
        user['id'] != ticket.owner) {
      throw const FormatException('Ответ относится к другому профилю.');
    }

    final name = user['name']?.toString() ?? '';
    final email = user['email']?.toString() ?? '';
    final role = user['role']?.toString() ?? 'user';
    final avatarUrl = user['avatar_url']?.toString() ?? '';
    final isAdmin = role == 'admin';

    final saved = await ProfileSessionGuard.write(ticket, (prefs) async {
      final expiresAt = data['expires_at']?.toString() ??
          (tokenFromResponse != null
              ? prefs.getString(_expiresAtKey) ?? ''
              : '');
      await prefs.remove(_tokenKey);
      await prefs.remove('arborscan_user_id');
      await prefs.setString(_nameKey, name);
      await prefs.setString(_emailKey, email);
      await prefs.setString(_roleKey, role);
      await prefs.setString('arborscan_user_id', user['id'] as String);
      await prefs.setString(_expiresAtKey, expiresAt);
      await prefs.setString(_tokenKey, token);
      await prefs.setBool(_loggedInKey, true);
      await prefs.setBool(_adminFlagKey, isAdmin);
    }, active: () => mounted);

    if (!saved || !mounted) return false;
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
    unawaited(_loadStats());
    return true;
  }

  String? _validateEmail(String value) {
    final email = value.trim();
    final ok = RegExp(r'^[^@\s]+@[^@\s]+\.[^@\s]+$').hasMatch(email);
    if (!ok) return 'Введите корректную почту.';
    return null;
  }

  Future<void> _register() async {
    if (_busy) return;
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
      final ticket = await ProfileSessionGuard.capture(supersede: true);
      final data = await _postJson('/auth/register', {
        'name': name,
        'email': email,
        'password': password,
      });
      if (!await _applyAuthData(data, ticket: ticket)) return;
      setState(() {
        _serverOnline = true;
        _statusText = 'Профиль создан на сервере.';
      });
      _snack('Профиль создан. Добро пожаловать!');

      // ПОКАЗЫВАЕМ ОБУЧЕНИЕ ПОСЛЕ РЕГИСТРАЦИИ!
      if (mounted) {
        Navigator.of(context)
            .push(MaterialPageRoute(builder: (_) => const OnboardingPage()));
      }
    } catch (e) {
      _snack(e.toString().replaceFirst('Exception: ', ''));
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _login() async {
    if (_busy) return;
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
      final ticket = await ProfileSessionGuard.capture(supersede: true);
      final data = await _postJson('/auth/login', {
        'email': email,
        'password': password,
      });
      if (!await _applyAuthData(data, ticket: ticket)) return;
      setState(() {
        _serverOnline = true;
        _statusText = 'Вход выполнен через сервер.';
      });
      _snack(_isAdmin
          ? 'Вы вошли как администратор.'
          : 'Вы вошли как пользователь.');
    } catch (e) {
      _snack(e.toString().replaceFirst('Exception: ', ''));
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _logout() async {
    if (_busy) return;
    setState(() => _busy = true);
    try {
      final ticket = await ProfileSessionGuard.capture(supersede: true);
      try {
        if (ticket.token.isNotEmpty) {
          await _postJson('/auth/logout', null,
              useAuth: true, authToken: ticket.token);
        }
      } catch (_) {
        // A failed remote revoke must still allow a local logout.
      }
      if (await _clearInvalidSession('Вы вышли из профиля.', ticket)) {
        _passwordController.clear();
        _snack('Вы вышли из профиля.');
      }
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _deleteLocalSession() async {
    if (_busy) return;
    setState(() => _busy = true);
    try {
      final ticket = await ProfileSessionGuard.capture(supersede: true);
      if (await _clearInvalidSession('Локальная сессия очищена.', ticket)) {
        _passwordController.clear();
        _snack('Локальная сессия очищена. Аккаунт на сервере не удалён.');
      }
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _loginWithGoogle() async {
    if (_busy) return;
    setState(() => _busy = true);
    try {
      final ticket = await ProfileSessionGuard.capture(supersede: true);
      await _googleSignIn.signOut();
      final account = await _googleSignIn.signIn();
      if (account == null) {
        _snack('Вход через Google отменён.');
        return;
      }

      final auth = await account.authentication;
      final idToken = auth.idToken;
      if (idToken == null || idToken.isEmpty) {
        _snack(
            'Не удалось получить подтверждение Google. Повторите вход или используйте почту.');
        return;
      }

      final data = await _postJson('/auth/google', {
        'id_token': idToken,
      });

      if (!await _applyAuthData(data, ticket: ticket)) return;
      setState(() {
        _serverOnline = true;
        _statusText = 'Вход выполнен через Google.';
      });
      _snack('Вы вошли через Google.');
    } catch (e) {
      _snack(_googleErrorMessage(e));
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  String _googleErrorMessage(Object error) {
    if (error is PlatformException) {
      if (error.code == 'sign_in_canceled' || error.code == 'canceled') {
        return 'Вход через Google отменён.';
      }
      if (error.code == 'network_error') {
        return 'Нет связи с Google. Проверьте интернет и повторите вход.';
      }
      return 'Вход Google сейчас недоступен. Повторите попытку или используйте почту.';
    }
    if (error is TimeoutException || error is http.ClientException) {
      return 'Не удалось связаться с сервером. Проверьте интернет и повторите вход.';
    }
    if (error is FormatException) {
      return 'Сервер вернул некорректный ответ. Вход не изменён; повторите попытку.';
    }
    final message = error.toString();
    if (message.contains('HTTP 409:')) {
      return 'Google не удалось связать с этим профилем. Войдите прежним способом и обратитесь к разработчику.';
    }
    if (message.contains('HTTP 401:')) {
      return 'Сервер не подтвердил вход Google. Повторите попытку или используйте почту.';
    }
    return 'Вход Google сейчас недоступен. Повторите попытку или используйте почту.';
  }

  Future<void> _loadStats() async {
    if (_token.isEmpty) return;
    try {
      final session = _token;
      final ticket = await ProfileSessionGuard.capture();
      if (ticket.token != session) return;
      final data =
          await _getJson('/profile/stats', useAuth: true, authToken: session);
      if (session != _token ||
          !await ProfileSessionGuard.current(ticket, active: () => mounted)) {
        return;
      }
      final stats = (data['stats'] as Map?)?.cast<String, dynamic>() ?? {};
      final user = (data['user'] as Map?)?.cast<String, dynamic>() ?? {};

      if (!mounted) return;
      setState(() {
        _avatarUrl = user['avatar_url']?.toString() ?? _avatarUrl;
        _totalAnalyses = (stats['total_analyses'] as num?)?.toInt() ?? 0;
        _geoAnalyses = (stats['with_geo'] as num?)?.toInt() ?? 0;
        _highRiskAnalyses = (stats['high_risk_count'] as num?)?.toInt() ?? 0;
        _avgRisk = (stats['avg_risk'] as num?)?.toDouble();
        _lastAnalysis =
            (stats['last_analysis'] as Map?)?.cast<String, dynamic>();
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
                    _loggedIn
                        ? _buildAccountCard(context)
                        : _buildAuthCard(context),
                    if (_loggedIn) ...[
                      const SizedBox(height: 14),
                      _buildStatsCard(context),
                    ],
                    const SizedBox(height: 14),
                    _buildDeveloperCard(context),
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
              ApiConfig.imageUrl(_avatarUrl),
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
                  'Фото, измерения и история обследований',
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                        color: AppTheme.muted,
                      ),
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

  void _showRecoveryHelp() {
    showDialog<void>(
        context: context,
        builder: (context) => AlertDialog(
              title: const Text('Восстановление доступа'),
              content: const SingleChildScrollView(
                  child: Text(
                      'Если вы входили через Google, используйте тот же аккаунт Google. '
                      'Для входа по почте автоматический сброс пароля пока недоступен на сервере. '
                      'Обратитесь к разработчику и укажите почту профиля. Пароль присылать не нужно. '
                      'Новую учётную запись вместо прежней создавать не требуется.')),
              actions: [
                TextButton(
                    onPressed: () => Navigator.pop(context),
                    child: const Text('Закрыть')),
                TextButton.icon(
                    onPressed: _openDeveloperEmail,
                    icon: const Icon(Icons.email_outlined),
                    label: const Text('Связаться')),
              ],
            ));
  }

  Widget _buildDeveloperCard(BuildContext context) => Card(
          child: Column(children: [
        ListTile(
            leading: const Icon(Icons.help_outline),
            title: const Text('Как пользоваться'),
            trailing: const Icon(Icons.chevron_right),
            onTap: () => Navigator.push(context,
                MaterialPageRoute(builder: (_) => const OnboardingPage()))),
        const Divider(height: 1),
        ExpansionTile(
            leading: const Icon(Icons.support_agent),
            title: const Text('Связь с разработчиком'),
            children: [
              Padding(
                  padding: const EdgeInsets.all(12),
                  child: Column(
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        const SelectableText(_developerEmail),
                        TextButton.icon(
                            onPressed: _openDeveloperEmail,
                            icon: const Icon(Icons.email_outlined),
                            label: const Text('Написать разработчику')),
                      ])),
            ]),
      ]));

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
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              ChoiceChip(
                avatar: const Icon(Icons.person_add_alt),
                label: const Text('Регистрация'),
                selected: _isRegisterMode,
                materialTapTargetSize: MaterialTapTargetSize.padded,
                onSelected: _busy
                    ? null
                    : (_) => setState(() => _isRegisterMode = true),
              ),
              ChoiceChip(
                avatar: const Icon(Icons.lock_open),
                label: const Text('Вход'),
                selected: !_isRegisterMode,
                materialTapTargetSize: MaterialTapTargetSize.padded,
                onSelected: _busy
                    ? null
                    : (_) => setState(() => _isRegisterMode = false),
              ),
            ],
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
          const Row(
            children: [
              Expanded(child: Divider(color: AppTheme.border)),
              Padding(
                padding: EdgeInsets.symmetric(horizontal: 10),
                child: Text(
                  'или',
                  style: TextStyle(
                      color: AppTheme.muted, fontWeight: FontWeight.w700),
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
          if (!_isRegisterMode)
            TextButton(
                onPressed: _busy ? null : _showRecoveryHelp,
                child: const Text('Не помню пароль')),
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

  Widget _buildAccountCard(BuildContext context) => Ui.paddedCard(context,
      child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
        Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
          _buildAvatar(size: 48),
          const SizedBox(width: 12),
          Expanded(
              child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                Text(_name.isNotEmpty ? _name : 'Пользователь ArborScan',
                    style: Theme.of(context).textTheme.titleLarge),
                const SizedBox(height: 4),
                SelectableText(_email,
                    style: const TextStyle(color: AppTheme.muted)),
              ])),
        ]),
        const SizedBox(height: 12),
        Wrap(spacing: 8, runSpacing: 8, children: [
          if (_isAdmin)
            Ui.badge(
                text: 'Администратор',
                color: AppTheme.primary,
                icon: Icons.admin_panel_settings),
          Ui.badge(
              text: _serverOnline ? 'Серверная сессия' : 'Локальная копия',
              color: _serverOnline ? AppTheme.success : AppTheme.warning,
              icon: _serverOnline ? Icons.cloud_done : Icons.storage),
        ]),
        const SizedBox(height: 16),
        if (_isAdmin && _loggedIn) ...[
          FilledButton.icon(
              onPressed: _busy
                  ? null
                  : () => Navigator.push(
                      context,
                      MaterialPageRoute(
                          builder: (_) => const ModelQualityPage())),
              icon: const Icon(Icons.model_training),
              label: const Text('Модели и данные')),
          const SizedBox(height: 8),
          OutlinedButton.icon(
              onPressed: _busy
                  ? null
                  : () => Navigator.push(
                      context,
                      MaterialPageRoute(
                          builder: (_) => const AdminPanelPage(
                              baseUrl: ApiConfig.baseUrl))),
              icon: const Icon(Icons.admin_panel_settings_outlined),
              label: const Text('Админ-панель')),
          const SizedBox(height: 8),
          FilledButton.icon(
              onPressed: _busy
                  ? null
                  : () => Navigator.push(
                      context,
                      MaterialPageRoute(
                          builder: (_) =>
                              const SavedCorrectionsPage(adminQueue: true))),
              icon: const Icon(Icons.fact_check_outlined),
              label: const Text('Проверка контуров')),
          const Divider(height: 32),
        ],
        OutlinedButton.icon(
            onPressed: _busy ? null : _logout,
            icon: const Icon(Icons.logout),
            label: const Text('Выйти')),
        TextButton.icon(
            onPressed: _busy ? null : _deleteLocalSession,
            icon: const Icon(Icons.cleaning_services_outlined),
            label: const Text('Очистить локальную сессию')),
      ]));

  Widget _buildStatsCard(BuildContext context) => Ui.paddedCard(context,
      child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
        Row(children: [
          Expanded(
              child: Text('Мои анализы',
                  style: Theme.of(context).textTheme.titleMedium)),
          IconButton(
              tooltip: 'Обновить статистику',
              onPressed: _loadStats,
              icon: const Icon(Icons.refresh)),
        ]),
        _statTile(
            'В архиве анализов', '$_totalAnalyses', Icons.analytics_outlined),
        const SizedBox(height: 8),
        _statTile(
            'С координатами', '$_geoAnalyses', Icons.location_on_outlined),
        if (_lastAnalysis == null)
          const Padding(
              padding: EdgeInsets.symmetric(vertical: 12),
              child: Text('Пока нет серверных анализов.')),
        if (_lastAnalysis != null)
          Padding(
              padding: const EdgeInsets.symmetric(vertical: 12),
              child: Text(
                  'Последний анализ: ${_lastAnalysis!['species'] ?? 'Порода не определена'}')),
        ExpansionTile(title: const Text('Исторические показатели'), children: [
          const Text(
              'Прежняя эвристическая оценка риска не подтверждает безопасность дерева. Эти числа относятся к архиву анализов.'),
          _statTile(
              'Высокая прежняя оценка', '$_highRiskAnalyses', Icons.history),
          _statTile('Средняя прежняя оценка',
              _avgRisk?.toStringAsFixed(2) ?? '—', Icons.history),
        ]),
      ]));

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

  Widget _buildInfoCard(BuildContext context) => Card(
          child: ExpansionTile(
        title: const Text('О приложении и диагностика'),
        childrenPadding: const EdgeInsets.all(16),
        children: [
          _infoRow(Icons.info_outline, 'Версия', _appVersion),
          _infoRow(Icons.analytics_outlined, 'Анализ фото',
              'Контуры и предположение о породе. Подтверждение вида хранится отдельно.'),
          _infoRow(Icons.view_in_ar_outlined, 'Измерения',
              'AR и известный объект. Полевую точность необходимо проверять контрольными измерениями.'),
          _infoRow(Icons.map_outlined, 'Карта',
              'Обследования с сохранёнными координатами.'),
          _infoRow(Icons.science_outlined, 'Ограничения',
              'β в кг/с и научно подтверждённая устойчивость пока не рассчитываются.'),
          const SelectableText(
              'API v3: ${ApiConfig.baseUrl}\nAPI v4: ${ApiConfig.v4BaseUrl}'),
        ],
      ));

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
                  TextSpan(
                      text: '$title: ',
                      style: const TextStyle(fontWeight: FontWeight.w900)),
                  TextSpan(
                      text: text,
                      style: const TextStyle(color: AppTheme.muted)),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
