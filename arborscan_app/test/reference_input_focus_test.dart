import 'dart:convert';
import 'dart:io';

import 'package:arborscan_app/app_theme.dart';
import 'package:arborscan_app/contour_drafts.dart';
import 'package:arborscan_app/contour_editor_state.dart';
import 'package:arborscan_app/reference_measurement_page.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

void main() {
  testWidgets('clearing a valid reference keeps focus while export disappears',
      (tester) async {
    const owner = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa';
    const id = 'reference-focus';
    SharedPreferences.setMockInitialValues(
        {'arborscan_auth_token': 'test', 'arborscan_user_id': owner});
    final folder = (await tester
        .runAsync(() => Directory.systemTemp.createTemp('reference-focus-')))!;
    final store = ContourDrafts(directory: () async => folder);
    final image = base64Decode(
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNImXYCAAMkAcMgVWSjAAAAAElFTkSuQmCC');
    const outline = ContourEditorState(
        width: 1,
        height: 1,
        closed: true,
        points: [Offset(0, .9), Offset(.2, .9), Offset(.1, .6)]);
    final reference = [
      {'x': .1, 'y': .8},
      {'x': .1, 'y': .7}
    ];
    final tree = [
      {'x': .5, 'y': .9},
      {'x': .5, 'y': .1}
    ];
    await tester.runAsync(() => store.save(
        owner,
        id,
        {
          'version': 1,
          'width': 1,
          'height': 1,
          'length_text': '100',
          'unit': 'cm',
          'same_plane': true,
          'outline': outline.toJson(),
          'reference': reference,
          'tree': tree,
          'crown': [
            {'x': .2, 'y': .4},
            {'x': .8, 'y': .4}
          ],
          'measurement_version': 2,
        },
        image));
    Future<void> settle() async {
      for (var i = 0; i < 8; i++) {
        await tester.runAsync(
            () async => Future<void>.delayed(const Duration(milliseconds: 25)));
        await tester.pump(const Duration(milliseconds: 100));
      }
    }

    Future<void> open() async {
      await tester.runAsync(() => tester.pumpWidget(MaterialApp(
          theme: AppTheme.light(),
          home: ReferenceMeasurementPage(
              draftId: id,
              drafts: ContourDrafts(directory: () async => folder)))));
      await settle();
      await tester.scrollUntilVisible(find.byType(TextField), 150,
          scrollable: find.byType(Scrollable).first);
      await tester.pumpAndSettle();
    }

    await open();
    await tester.tap(find.byType(TextField));
    await tester.pump();
    final editable = tester.state<EditableTextState>(find.byType(EditableText));
    expect(editable.widget.focusNode.hasFocus, isTrue);
    expect(editable.widget.controller.text, '100');

    // Unlike tester.enterText, this sends to the existing input connection
    // without refocusing the field. It reproduces clearing before replacement.
    tester.testTextInput.enterText('');
    await settle();
    expect(find.byType(EditableText), findsOneWidget);
    expect(
        identical(tester.state(find.byType(EditableText)), editable), isTrue);
    expect(editable.widget.focusNode.hasFocus, isTrue);
    expect(tester.testTextInput.hasAnyClients, isTrue);
    tester.testTextInput.enterText('2');
    await settle();
    expect(editable.widget.focusNode.hasFocus, isTrue);
    tester.testTextInput.enterText('200');
    await settle();
    expect(editable.widget.controller.text, '200');
    final saved = (await tester.runAsync(() => store.load(owner, id)))!;
    expect(saved['length_text'], '200');
    expect(saved['unit'], 'cm');
    expect(saved['report']['height_m'], closeTo(16, 1e-8));
    expect(saved['reference'], reference);
    expect(saved['tree'], tree);
    expect(saved['outline'], outline.toJson());
    await tester.pumpWidget(const SizedBox());
    await open();
    expect(tester.widget<TextField>(find.byType(TextField)).controller!.text,
        '200');
    await tester.pumpWidget(const SizedBox());
    await tester.runAsync(() => folder.delete(recursive: true));
  });
}
