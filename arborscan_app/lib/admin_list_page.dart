import 'package:flutter/material.dart';
import 'admin_service.dart';
import 'admin_analysis_item.dart';
import 'admin_analysis_detail_page.dart';

class AdminListPage extends StatefulWidget {
  const AdminListPage({super.key});

  @override
  State<AdminListPage> createState() => _AdminListPageState();
}

class _AdminListPageState extends State<AdminListPage> {
  late Future<List<AdminAnalysisItem>> _future;

  @override
  void initState() {
    super.initState();
    _future = AdminService.fetchVerifiedAnalyses();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Verified analyses'),
      ),
      body: FutureBuilder<List<AdminAnalysisItem>>(
        future: _future,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Center(child: CircularProgressIndicator());
          }

          if (snapshot.hasError) {
            return Center(child: Text('Ошибка: ${snapshot.error}'));
          }

          final items = snapshot.data!;
          if (items.isEmpty) {
            return const Center(child: Text('Нет данных'));
          }

          return ListView.separated(
            itemCount: items.length,
            separatorBuilder: (_, __) => const Divider(height: 1),
            itemBuilder: (context, index) {
              final item = items[index];
              return ListTile(
                title: Text(item.species ?? '—'),
                subtitle: Text(
                  'Risk: ${item.riskCategory ?? '-'} | Trust: ${item.trustScore.toStringAsFixed(2)}',
                ),
                trailing: const Icon(Icons.chevron_right),
                onTap: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (_) => AdminAnalysisDetailPage(
                        analysisId: item.analysisId,
                      ),
                    ),
                  );
                },
              );
            },
          );
        },
      ),
    );
  }
}
