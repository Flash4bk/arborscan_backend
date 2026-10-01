import 'package:arborscan_app/app_theme.dart';
import 'package:arborscan_app/corrections_service.dart';
import 'package:arborscan_app/map_page.dart';
import 'package:arborscan_app/survey_map_repository.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

class _LocalMap extends SurveyMapRepository {
  int detailLoads = 0;
  @override
  Future<SurveyMapResult> load(
      {void Function(SurveyMapResult)? onLocal}) async {
    final preferences = await SharedPreferences.getInstance();
    if (preferences.getString('arborscan_user_id') !=
        'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa') {
      return const SurveyMapResult([], ['Для этого аккаунта записей нет.']);
    }
    return SurveyMapResult([
      SurveyMapEntry.version({
        'version_id': 'demoversion',
        'draft_id': 'demo-local',
        'snapshot': {
          'report': {
            'species': 'DEMO обследование без координат',
            'height_m': 12
          }
        },
      }, 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa', server: false)
    ], [
      'Нет сети. Локальный отчёт доступен.'
    ]);
  }

  @override
  Future<Map<String, dynamic>> detail(SurveyMapEntry entry) async {
    detailLoads++;
    return {'record': entry.metadata, 'snapshot': entry.metadata['snapshot']};
  }
}

void main() {
  testWidgets(
      'No GPS does not invent a marker; large-font offline list and account switch clear selected card',
      (tester) async {
    tester.view.physicalSize = const Size(320, 800);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    SharedPreferences.setMockInitialValues({
      'arborscan_auth_token': 'synthetic-a',
      'arborscan_user_id': 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa'
    });
    final repository = _LocalMap();
    await tester.pumpWidget(MaterialApp(
        theme: AppTheme.light(),
        home: MediaQuery(
            data: const MediaQueryData(textScaler: TextScaler.linear(2)),
            child: MapPage(repository: repository))));
    await tester.pumpAndSettle();
    expect(find.text('0 с координатами · 1 без точки'), findsOneWidget);
    expect(find.textContaining('Нет GPS данных, карта недоступна'),
        findsOneWidget);
    expect(repository.detailLoads, 0);
    expect(tester.takeException(), isNull);
    await tester.ensureVisible(find.text('DEMO обследование без координат'));
    await tester.tap(find.text('DEMO обследование без координат'));
    await tester.pumpAndSettle();
    expect(repository.detailLoads, 1);
    expect(find.text('Открыть отчёт / добавить точку'), findsOneWidget);
    expect(find.textContaining('Высота 12.00 м'), findsOneWidget);
    expect(tester.takeException(), isNull);
    final preferences = await SharedPreferences.getInstance();
    await preferences.setString('arborscan_auth_token', 'synthetic-b');
    await preferences.setString(
        'arborscan_user_id', 'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb');
    CorrectionsService.authChanges.value++;
    await tester.pumpAndSettle();
    expect(find.text('DEMO обследование без координат'), findsNothing);
    expect(find.text('Аккаунт изменился. Закройте карточку.'), findsOneWidget);
    expect(tester.takeException(), isNull);
  });
}
