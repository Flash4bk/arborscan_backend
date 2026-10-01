import 'package:flutter/material.dart';

import 'analyze_page.dart';
import 'app_navigation.dart';
import 'corrections_service.dart';
import 'history_tab_page.dart';
import 'map_page.dart';
import 'profile_page.dart';

class AppRoot extends StatefulWidget {
  const AppRoot({super.key});

  @override
  State<AppRoot> createState() => _AppRootState();
}

class _AppRootState extends State<AppRoot> {
  int _index = 0;
  final _visited = <int>{0};

  late Widget _analyzePage;
  late final Widget _historyPage;
  late final Widget _mapPage;
  late final Widget _profilePage;

  List<Widget> get _pages => [
        _analyzePage,
        _historyPage,
        _mapPage,
        _profilePage,
      ];

  @override
  void initState() {
    super.initState();
    _analyzePage = ArborScanPage(key: UniqueKey());
    _historyPage = const HistoryTabPage();
    _mapPage = const MapPage();
    _profilePage = ProfilePage(onAuthChanged: _handleAuthChanged);
  }

  void _handleAuthChanged() {
    CorrectionsService.authChanges.value++;
    if (!mounted) return;
    setState(() {
      // Пересоздаём только экран анализа, чтобы он перечитал роль и токен,
      // не затрагивая дизайн и состояние остальных вкладок.
      _analyzePage = ArborScanPage(key: UniqueKey());
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IndexedStack(index: _index, children: [
        for (var i = 0; i < _pages.length; i++)
          _visited.contains(i) ? _pages[i] : const SizedBox.shrink()
      ]),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _index,
        onDestinationSelected: (value) {
          setState(() {
            _visited.add(value);
            _index = value;
          });
          AppNavigation.activate(value);
        },
        destinations: const [
          NavigationDestination(
              icon: Icon(Icons.add_a_photo_outlined), label: 'Анализ'),
          NavigationDestination(
              icon: Icon(Icons.history_rounded), label: 'История'),
          NavigationDestination(icon: Icon(Icons.map_outlined), label: 'Карта'),
          NavigationDestination(
              icon: Icon(Icons.person_outline), label: 'Профиль'),
        ],
      ),
    );
  }
}
