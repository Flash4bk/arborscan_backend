import 'package:flutter/material.dart';
import 'package:lottie/lottie.dart';
import 'app_theme.dart';

class OnboardingPage extends StatefulWidget {
  const OnboardingPage({super.key});

  @override
  State<OnboardingPage> createState() => _OnboardingPageState();
}

class _OnboardingPageState extends State<OnboardingPage> {
  final PageController _pageController = PageController();
  int _currentPage = 0;

  final List<Map<String, String>> _pages = [
    {
      "title": "УМНЫЙ АНАЛИЗ",
      "subtitle": "Сфотографируйте дерево. Нейросеть сама определит породу, найдет ствол и оценит риск падения.",
      "icon": "park_rounded",
    },
    {
      "title": "AR-ИЗМЕРЕНИЯ",
      "subtitle": "Забудьте про рулетку. Используйте камеру и гироскоп телефона, чтобы мгновенно измерить высоту и крону.",
      "icon": "view_in_ar_rounded",
    },
    {
      "title": "ВЕТРОВАЯ НАГРУЗКА",
      "subtitle": "Введите силу ветра, и мы рассчитаем коэффициент β и аэродинамический момент у основания ствола.",
      "icon": "storm_rounded",
    }
  ];

  IconData _getIcon(String name) {
    if (name == 'park_rounded') return Icons.park_rounded;
    if (name == 'view_in_ar_rounded') return Icons.view_in_ar_rounded;
    return Icons.storm_rounded;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          PageView.builder(
            controller: _pageController,
            onPageChanged: (i) => setState(() => _currentPage = i),
            itemCount: _pages.length,
            itemBuilder: (context, index) {
              return Padding(
                padding: const EdgeInsets.all(32.0),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    GlassPanel(
                      padding: const EdgeInsets.all(40),
                      radius: 100,
                      child: Icon(_getIcon(_pages[index]["icon"]!), size: 80, color: AppTheme.primary),
                    ),
                    const SizedBox(height: 48),
                    Text(
                      _pages[index]["title"]!,
                      style: const TextStyle(fontSize: 24, fontWeight: FontWeight.w900, color: AppTheme.primary2, letterSpacing: 2.0),
                      textAlign: TextAlign.center,
                    ),
                    const SizedBox(height: 16),
                    Text(
                      _pages[index]["subtitle"]!,
                      style: const TextStyle(fontSize: 16, color: AppTheme.muted, height: 1.5),
                      textAlign: TextAlign.center,
                    ),
                  ],
                ),
              );
            },
          ),
          Positioned(
            bottom: 40,
            left: 20,
            right: 20,
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Row(
                  children: List.generate(
                    _pages.length,
                    (index) => AnimatedContainer(
                      duration: const Duration(milliseconds: 300),
                      margin: const EdgeInsets.only(right: 8),
                      height: 8,
                      width: _currentPage == index ? 24 : 8,
                      decoration: BoxDecoration(
                        color: _currentPage == index ? AppTheme.primary : AppTheme.surface3,
                        borderRadius: BorderRadius.circular(4),
                      ),
                    ),
                  ),
                ),
                FilledButton(
                  onPressed: () {
                    if (_currentPage == _pages.length - 1) {
                      Navigator.of(context).pop(); // Закрываем обучение
                    } else {
                      _pageController.nextPage(duration: const Duration(milliseconds: 400), curve: Curves.easeInOut);
                    }
                  },
                  style: FilledButton.styleFrom(
                    backgroundColor: AppTheme.primary,
                    foregroundColor: Colors.black,
                    padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
                  ),
                  child: Text(_currentPage == _pages.length - 1 ? 'НАЧАТЬ' : 'ДАЛЕЕ', style: const TextStyle(fontWeight: FontWeight.w900, letterSpacing: 1.0)),
                )
              ],
            ),
          )
        ],
      ),
    );
  }
}