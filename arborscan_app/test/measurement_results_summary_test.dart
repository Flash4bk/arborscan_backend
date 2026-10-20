import 'package:arborscan_app/measurement_results_summary.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

Map<String, dynamic> referenceSnapshot() => {
      'kind': 'reference',
      'reference': {'method': 'known_object_segment_v2'},
      'report': {
        'height_m': 21.7,
        'crown_width_m': 5.07,
        'measurements': {
          'height': {'value_m': 99, 'source': 'ar'},
          'crown_width': {'value_m': 98, 'source': 'ar'},
          'trunk_diameter': {'value_m': 97, 'source': 'ar'},
        },
        'geometry': {
          'trunk_diameter': {
            'value': 0.172,
            'unit': 'm',
            'source': 'reference'
          },
          'trunk_lean': {'value': 0, 'unit': 'deg', 'source': 'reference'},
        },
      },
    };

Future<void> pumpSummary(
  WidgetTester tester,
  Map<String, dynamic> snapshot, {
  double width = 390,
  double scale = 1,
}) =>
    tester.pumpWidget(
      MaterialApp(
        home: MediaQuery(
          data: MediaQueryData(textScaler: TextScaler.linear(scale)),
          child: Scaffold(
            body: SingleChildScrollView(
              child: SizedBox(
                width: width,
                child: MeasurementResultsSummary(snapshot: snapshot),
              ),
            ),
          ),
        ),
      ),
    );

void main() {
  testWidgets('irrelevant image payloads are not serialized by the summary',
      (tester) async {
    final snapshot = referenceSnapshot();
    (snapshot['report'] as Map)['images'] = {'original_image_base64': Object()};
    snapshot['image'] = {'original_base64': Object()};
    await pumpSummary(tester, snapshot);
    expect(tester.takeException(), isNull);
    expect(find.text('21,70 м'), findsOneWidget);
    expect(find.text('0,172 м'), findsOneWidget);
  });

  testWidgets(
    'reference snapshot keeps projection precedence and trunk precision',
    (tester) async {
      final snapshot = referenceSnapshot();
      await pumpSummary(tester, snapshot);

      expect(find.text('Результат измерения и расчётов'), findsOneWidget);
      expect(find.text('Эталон · фотооценка проекции'), findsOneWidget);
      expect(find.text('21,70 м'), findsOneWidget);
      expect(find.text('5,07 м'), findsOneWidget);
      expect(find.text('0,172 м'), findsOneWidget);
      expect(find.text('0,00 °'), findsNothing);
      await tester.ensureVisible(find.text('Другие измеренные параметры'));
      await tester.tap(find.text('Другие измеренные параметры'));
      await tester.pumpAndSettle();
      expect(find.text('0,00 °'), findsOneWidget);
      expect(find.textContaining('99,00'), findsNothing);
      expect(find.text('AR-измерение'), findsNothing);
      expect(find.text('Диаметр отмеченного сечения ствола'), findsOneWidget);
      expect(
        find.text('Ширина сечения ствола (фотооценка диаметра)'),
        findsNothing,
      );
      expect((snapshot['report'] as Map)['height_m'], 21.7);
      expect(((snapshot['report'] as Map)['measurements'] as Map)['height'], {
        'value_m': 99,
        'source': 'ar',
      });
    },
  );

  testWidgets(
    'AR values and per-metric sources precede top-level legacy fields',
    (tester) async {
      await pumpSummary(tester, {
        'kind': 'v4',
        'report': {
          'measurement_method_version': 2,
          'height_m': 88,
          'crown_width_m': 87,
          'trunk_diameter_m': 86,
          'measurements': {
            'height': {'value_m': 23.456, 'source': 'ar', 'unit': 'm'},
            'crown_width': {'value_m': 4.567, 'source': 'ar', 'unit': 'm'},
            'trunk_diameter': {'value_m': 0.1719, 'source': 'ar', 'unit': 'm'},
          },
          'geometry': {
            'trunk_diameter': {'value': 0.999, 'unit': 'm'},
          },
        },
      });

      expect(find.text('Высота дерева'), findsOneWidget);
      expect(find.text('23,46 м'), findsOneWidget);
      expect(find.text('4,57 м'), findsOneWidget);
      expect(find.text('0,172 м'), findsOneWidget);
      expect(find.text('AR-измерение'), findsNWidgets(3));
      expect(find.text('0,999 м'), findsNothing);
      expect(find.textContaining('88,00'), findsNothing);
      expect(find.text('Эталон · фотооценка проекции'), findsNothing);
    },
  );

  testWidgets('photo mask height preserves its definition and stored source', (
    tester,
  ) async {
    await pumpSummary(tester, {
      'kind': 'v4',
      'report': {
        'measurement_method_version': 2,
        'measurements': {
          'height': {'value_m': 12.3, 'source': 'reference+vision'},
          'crown_width': {'value_px': 100, 'source': 'vision'},
        },
      },
    });

    expect(find.text('Вертикальный размер маски на фото'), findsOneWidget);
    expect(find.text('12,30 м'), findsOneWidget);
    expect(find.text('Фото с масштабом эталона'), findsOneWidget);
    expect(find.text('нет данных · нужна разметка'), findsNWidgets(2));
    expect(find.text('100,00 м'), findsNothing);
  });

  testWidgets(
    'explicit missing or unavailable metrics do not revive old values',
    (tester) async {
      await pumpSummary(tester, {
        'kind': 'v4',
        'report': {
          'height_m': 88,
          'crown_width_m': 87,
          'trunk_diameter_m': 0.86,
          'measurements': {
            'height': {'value_m': null, 'source': 'unavailable'},
            'crown_width': {'source': 'ar'},
            'trunk_diameter': {'value_m': 0.172, 'source': 'unavailable'},
          },
          'geometry': {
            'trunk_lean': {'value': 5, 'unit': 'deg', 'source': 'unavailable'},
          },
        },
      });

      expect(
        find.text('нет данных · нужен масштаб и измерение'),
        findsOneWidget,
      );
      expect(find.text('нет данных · нужна разметка'), findsNWidgets(2));
      expect(find.text('88,00 м'), findsNothing);
      expect(find.text('87,00 м'), findsNothing);
      expect(find.text('0,860 м'), findsNothing);
      expect(find.text('0,172 м'), findsNothing);
      expect(find.text('5,00 °'), findsNothing);
    },
  );

  testWidgets('legacy-only dimensions remain labelled as saved values', (
    tester,
  ) async {
    await pumpSummary(tester, {
      'kind': 'legacy',
      'report': {
        'height_m': 12.3,
        'crown_width_m': 4.56,
        'trunk_diameter_m': 0.172,
        'measurement_sources': {
          'height_m': 'ar',
          'crown_width_m': 'manual',
          'trunk_diameter_m': 'ar',
        },
      },
    });

    expect(find.text('Сохранённые размеры прежнего формата'), findsOneWidget);
    expect(find.text('12,30 м'), findsOneWidget);
    expect(find.text('4,56 м'), findsOneWidget);
    expect(find.text('0,172 м'), findsOneWidget);
    expect(find.text('Сохранённое значение · AR-измерение'), findsNWidgets(2));
    expect(find.text('Сохранённое значение · Введено вручную'), findsOneWidget);
  });

  testWidgets(
    'missing geometry remains visible and calculation help is collapsed',
    (tester) async {
      await pumpSummary(tester, {'kind': 'v4', 'report': <String, dynamic>{}});

      expect(
        find.text('нет данных · нужен масштаб и измерение'),
        findsOneWidget,
      );
      expect(find.text('нет данных · нужна разметка'), findsNWidgets(2));
      expect(find.text('Коэффициент β (кг/с)'), findsOneWidget);
      expect(find.text('Критическая скорость ветра (м/с)'), findsOneWidget);
      expect(find.text('Вероятность повреждения'), findsOneWidget);
      expect(find.text('не рассчитан'), findsOneWidget);
      expect(find.text('не рассчитана'), findsNWidgets(2));
      expect(find.textContaining('измеренные траектории'), findsNothing);

      await tester.ensureVisible(find.text('Что нужно для расчётов'));
      await tester.tap(find.text('Что нужно для расчётов'));
      await tester.pumpAndSettle();
      expect(find.textContaining('измеренные траектории'), findsOneWidget);
      expect(find.textContaining('корней и основания'), findsOneWidget);
      expect(find.textContaining('вероятностной модели'), findsOneWidget);
    },
  );

  testWidgets(
    'nonfinite and nonpositive geometry never becomes a numeric result',
    (tester) async {
      await pumpSummary(tester, {
        'kind': 'reference',
        'reference': {'method': 'known_object_segment_v2'},
        'report': {
          'height_m': double.nan,
          'crown_width_m': double.infinity,
          'geometry': {
            'trunk_diameter': {'value': 'NaN', 'unit': 'm'},
            'crown_height': {'value': -1, 'unit': 'm'},
            'tree_segment_length': {'value': 0, 'unit': 'm'},
            'trunk_lean': {'value': double.negativeInfinity, 'unit': 'deg'},
            'crown_porosity': {'value': 1.2, 'unit': '1'},
          },
        },
      });

      expect(
        find.text('нет данных · нужен масштаб и измерение'),
        findsOneWidget,
      );
      expect(find.text('нет данных · нужна разметка'), findsNWidgets(2));
      expect(find.textContaining('NaN'), findsNothing);
      expect(find.textContaining('Infinity'), findsNothing);
      expect(find.textContaining('-1,00'), findsNothing);
      expect(find.text('0,00 м'), findsNothing);
      expect(find.text('1,20 1'), findsNothing);
    },
  );

  testWidgets('explicit incompatible units are not relabelled as metres', (
    tester,
  ) async {
    await pumpSummary(tester, {
      'kind': 'v4',
      'report': {
        'measurements': {
          'height': {'value_m': 100, 'unit': 'cm', 'source': 'ar'},
          'crown_width': {'value_m': 80, 'unit': 'cm', 'source': 'ar'},
          'trunk_diameter': {'value_m': 17.2, 'unit': 'cm', 'source': 'ar'},
        },
        'geometry': {
          'crown_height': {'value': 90, 'unit': 'cm'},
          'trunk_lean': {'value': 0.5, 'unit': 'rad'},
        },
      },
    });

    expect(find.text('100,00 м'), findsNothing);
    expect(find.text('80,00 м'), findsNothing);
    expect(find.text('17,200 м'), findsNothing);
    expect(find.text('90,00 м'), findsNothing);
    expect(find.text('0,50 °'), findsNothing);
    expect(find.text('нет данных · нужна разметка'), findsNWidgets(2));
  });

  testWidgets(
    'legacy beta mechanical and low-risk numbers are never fresh results',
    (tester) async {
      await pumpSummary(tester, {
        'kind': 'legacy',
        'report': {
          'beta': {'beta_kg_s': 44.4, 'available': true},
          'beta_kg_s': 44.4,
          'critical_wind_m_s': 27.3,
          'damage_probability': 0.01,
          'risk': {'available': true, 'index': 0.01, 'category': 'низкий'},
          'risk_category': 'низкий',
          'analytic_wind_model': {
            'outputs': {'analytical_score': 0.01, 'total_force_n': 44.4},
          },
        },
      });

      expect(find.text('не рассчитан'), findsOneWidget);
      expect(find.text('не рассчитана'), findsNWidgets(2));
      expect(find.textContaining('44,4'), findsNothing);
      expect(find.textContaining('27,3'), findsNothing);
      expect(find.textContaining('низкий'), findsNothing);
      expect(find.textContaining('0,01'), findsNothing);
    },
  );

  testWidgets('summary wraps on a narrow screen with large system text', (
    tester,
  ) async {
    await pumpSummary(tester, referenceSnapshot(), width: 280, scale: 3);
    expect(tester.takeException(), isNull);
    expect(find.text('0,172 м'), findsOneWidget);
    await tester.ensureVisible(find.text('Что нужно для расчётов'));
    await tester.tap(find.text('Что нужно для расчётов'));
    await tester.pumpAndSettle();
    expect(tester.takeException(), isNull);
    expect(find.textContaining('измеренные траектории'), findsOneWidget);
  });
}
