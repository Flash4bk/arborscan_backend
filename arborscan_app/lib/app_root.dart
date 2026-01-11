import 'package:flutter/material.dart';

// существующие экраны
import 'map_page.dart';
import 'profile_page.dart'; // если нет — скажи, дам заглушку
import 'analyze_page.dart'; // это твой текущий Home/Analyze
// если AnalyzePage называется иначе — просто поправь импорт

class AppRoot extends StatefulWidget {
  const AppRoot({Key? key}) : super(key: key);

  @override
  State<AppRoot> createState() => _AppRootState();
}

class _AppRootState extends State<AppRoot> {
  int _index = 0;

  late final List<Widget> _pages;

  @override
  void initState() {
    super.initState();
    _pages = const [
      AnalyzePage(),
      _HistoryStubPage(),
      MapPage(),
      ProfilePage(),
    ];
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IndexedStack(
        index: _index,
        children: _pages,
      ),
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _index,
        onTap: (i) => setState(() => _index = i),
        type: BottomNavigationBarType.fixed,
        items: const [
          BottomNavigationBarItem(
            icon: Icon(Icons.camera_alt),
            label: 'Анализ',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.history),
            label: 'История',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.map),
            label: 'Карта',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.person),
            label: 'Профиль',
          ),
        ],
      ),
    );
  }
}

class _HistoryStubPage extends StatelessWidget {
  const _HistoryStubPage();

  @override
  Widget build(BuildContext context) {
    return const Center(
      child: Text(
        'История появится позже',
        style: TextStyle(fontSize: 16, color: Colors.grey),
      ),
    );
  }
}
