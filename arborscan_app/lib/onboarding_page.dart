import 'package:flutter/material.dart';
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
      "title": "Анализ фотографии",
      "subtitle":
          "Сфотографируйте дерево или выберите снимок. Распознавание породы — предположение классификатора; подтверждение таксона хранится отдельно.",
      "icon": "park_rounded",
    },
    {
      "title": "Измерения и эталон",
      "subtitle":
          "Измерьте дерево в AR или разметьте фото с известным объектом на той же глубине. Источник каждого размера сохраняется. Полевая точность требует контрольных измерений.",
      "icon": "view_in_ar_rounded",
    },
    {
      "title": "Границы результата",
      "subtitle":
          "β в кг/с пока не рассчитывается: нужны динамический эксперимент и проверенная модель. Правка и принятие маски не подтверждают физические измерения или устойчивость дерева.",
      "icon": "storm_rounded",
    }
  ];

  IconData _getIcon(String name) {
    if (name == 'park_rounded') return Icons.park_rounded;
    if (name == 'view_in_ar_rounded') return Icons.view_in_ar_rounded;
    return Icons.storm_rounded;
  }

  @override
  void dispose() {
    _pageController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => Scaffold(
        appBar: AppBar(title: const Text('Как пользоваться')),
        body: PageView.builder(
            controller: _pageController,
            onPageChanged: (i) => setState(() => _currentPage = i),
            itemCount: _pages.length,
            itemBuilder: (context, index) =>
                ListView(padding: const EdgeInsets.all(24), children: [
                  Icon(_getIcon(_pages[index]['icon']!),
                      size: 64, color: AppTheme.primary),
                  const SizedBox(height: 24),
                  Text(_pages[index]['title']!,
                      style: Theme.of(context).textTheme.headlineSmall),
                  const SizedBox(height: 16),
                  Text(_pages[index]['subtitle']!,
                      style: Theme.of(context).textTheme.bodyLarge),
                ])),
        bottomNavigationBar: SafeArea(
            child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(mainAxisSize: MainAxisSize.min, children: [
                  Text('${_currentPage + 1} из ${_pages.length}'),
                  const SizedBox(height: 8),
                  SizedBox(
                      width: double.infinity,
                      child: FilledButton(
                          onPressed: () {
                            if (_currentPage == _pages.length - 1) {
                              Navigator.pop(context);
                            } else {
                              if (MediaQuery.disableAnimationsOf(context)) {
                                _pageController.jumpToPage(_currentPage + 1);
                              } else {
                                _pageController.nextPage(
                                    duration: const Duration(milliseconds: 200),
                                    curve: Curves.easeOut);
                              }
                            }
                          },
                          child: Text(_currentPage == _pages.length - 1
                              ? 'Готово'
                              : 'Далее'))),
                ]))),
      );
}
