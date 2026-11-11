// Audit probes assert current defects, not approved product behaviour.
// Synthetic legacy records only; no account, server or file mutations.
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:arborscan_app/analysis_report_page.dart';

void main() {
  testWidgets('audit reproduces safe verdict for an old unvalidated category',
      (tester) async {
    await tester.pumpWidget(MaterialApp(
      home: AnalysisReportPageV2.fromHistory(
        species: 'DEMO tree',
        riskCategory: 'низкий',
      ),
    ));
    expect(find.text('БЕЗОПАСНО'), findsOneWidget);
  });

  testWidgets('audit reproduces critical verdict for an unknown old category',
      (tester) async {
    await tester.pumpWidget(MaterialApp(
      home: AnalysisReportPageV2.fromHistory(
        species: 'DEMO tree',
        riskCategory: 'DEMO unsupported category',
      ),
    ));
    expect(find.text('КРИТИЧНО'), findsOneWidget);
  });
}
