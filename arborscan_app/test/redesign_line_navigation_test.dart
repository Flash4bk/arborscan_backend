import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:arborscan_app/app_theme.dart';
import 'package:arborscan_app/image_line_page.dart';

void main() {
  testWidgets(
      'Back warns about unapplied points; continue keeps them, explicit discard exits',
      (t) async {
    final photo = base64Decode(
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNImXYCAAMkAcMgVWSjAAAAAElFTkSuQmCC');
    await t.pumpWidget(MaterialApp(
        theme: AppTheme.light(),
        home: Builder(
            builder: (c) => Scaffold(
                body: FilledButton(
                    onPressed: () => Navigator.push(
                        c,
                        MaterialPageRoute(
                            builder: (_) => ImageLinePage(
                                image: photo,
                                width: 1,
                                height: 1,
                                title: 'Тестовая разметка'))),
                    child: const Text('Начать'))))));
    await t.tap(find.text('Начать'));
    await t.pumpAndSettle();
    await t.tapAt(t.getCenter(find.byKey(const ValueKey('image-line-canvas'))));
    await t.pump();
    await t.pageBack();
    await t.pumpAndSettle();
    expect(find.text('Отрезок не применён'), findsOneWidget);
    await t.tap(find.text('Продолжить'));
    await t.pumpAndSettle();
    expect(find.byType(ImageLinePage), findsOneWidget);
    await t.pageBack();
    await t.pumpAndSettle();
    await t.tap(find.text('Выйти без правок'));
    await t.pumpAndSettle();
    expect(find.byType(ImageLinePage), findsNothing);
  });
}
