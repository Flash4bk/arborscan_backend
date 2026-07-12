import 'dart:ui';
import 'package:flutter/material.dart';

import 'analyze_page.dart';
import 'app_theme.dart';
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
    if (!mounted) return;
    setState(() {
      // Пересоздаём только экран анализа, чтобы он перечитал роль и токен.
      _analyzePage = ArborScanPage(key: UniqueKey());
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBody: true,
      body: Stack(
        children: [
          Positioned(
            top: -100,
            left: -100,
            child: Container(
              width: 300,
              height: 300,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: AppTheme.primary2.withOpacity(0.15),
              ),
              child: BackdropFilter(
                filter: ImageFilter.blur(sigmaX: 80, sigmaY: 80),
                child: const SizedBox(),
              ),
            ),
          ),
          
          // Плавный эффект переходов между вкладками (Fade + Scale)
          AnimatedSwitcher(
            duration: const Duration(milliseconds: 500),
            switchInCurve: Curves.easeOutCubic,
            switchOutCurve: Curves.easeInCubic,
            transitionBuilder: (Widget child, Animation<double> animation) {
              return FadeTransition(
                opacity: animation,
                child: ScaleTransition(
                  scale: Tween<double>(begin: 0.96, end: 1.0).animate(animation),
                  child: child,
                ),
              );
            },
            child: KeyedSubtree(
              key: ValueKey<int>(_index),
              child: _KeepAlivePage(child: _pages[_index]),
            ),
          ),
        ],
      ),
      bottomNavigationBar: SafeArea(
        minimum: const EdgeInsets.fromLTRB(20, 0, 20, 24),
        child: GlassPanel(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 12),
          radius: 30,
          border: Border.all(color: AppTheme.primary.withOpacity(0.3), width: 1),
          boxShadow: [
            BoxShadow(color: AppTheme.primary2.withOpacity(0.1), blurRadius: 30, spreadRadius: 5)
          ],
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: [
              _NavItem(icon: Icons.radar_rounded, label: 'АНАЛИЗ', selected: _index == 0, onTap: () => setState(() => _index = 0)),
              _NavItem(icon: Icons.history_rounded, label: 'ИСТОРИЯ', selected: _index == 1, onTap: () => setState(() => _index = 1)),
              _NavItem(icon: Icons.map_rounded, label: 'КАРТА', selected: _index == 2, onTap: () => setState(() => _index = 2)),
              _NavItem(icon: Icons.person_rounded, label: 'ПРОФИЛЬ', selected: _index == 3, onTap: () => setState(() => _index = 3)),
            ],
          ),
        ),
      ),
    );
  }
}

class _KeepAlivePage extends StatefulWidget {
  final Widget child;
  const _KeepAlivePage({required this.child});

  @override
  State<_KeepAlivePage> createState() => _KeepAlivePageState();
}

class _KeepAlivePageState extends State<_KeepAlivePage> with AutomaticKeepAliveClientMixin {
  @override
  bool get wantKeepAlive => true;

  @override
  Widget build(BuildContext context) {
    super.build(context);
    return widget.child;
  }
}

class _NavItem extends StatelessWidget {
  final IconData icon;
  final String label;
  final bool selected;
  final VoidCallback onTap;

  const _NavItem({required this.icon, required this.label, required this.selected, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      behavior: HitTestBehavior.opaque,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 300),
        curve: Curves.easeOutExpo,
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        decoration: BoxDecoration(
          color: selected ? AppTheme.primary.withOpacity(0.15) : Colors.transparent,
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: selected ? AppTheme.primary.withOpacity(0.5) : Colors.transparent),
          boxShadow: selected ? [BoxShadow(color: AppTheme.primary.withOpacity(0.2), blurRadius: 12)] : [],
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              icon,
              color: selected ? AppTheme.primary : AppTheme.muted,
              size: selected ? 28 : 24,
              shadows: selected ? [const Shadow(color: AppTheme.primary, blurRadius: 8)] : [],
            ),
            if (selected) ...[
              const SizedBox(height: 4),
              Text(
                label,
                style: const TextStyle(
                  color: AppTheme.primary,
                  fontSize: 10,
                  fontWeight: FontWeight.w900,
                  letterSpacing: 1.0,
                  shadows: [Shadow(color: AppTheme.primary2, blurRadius: 6)],
                ),
              ),
            ]
          ],
        ),
      ),
    );
  }
}