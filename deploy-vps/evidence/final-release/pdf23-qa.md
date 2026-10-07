# Контрольные PDF окончательного APK23

Source `bbcd5b539427e62da58739fc2dbeff8550ee72c6`, release1.3.0+23,
APK SHA71f38189ee62a7825029ec7c25bf635891d3adcd0fa3eea3d9b618a223e4a31c.
Это настоящие файлы DEMO, созданные через системный Save и выгруженные с
совпавшим SHA устройства. Все15 страниц отдельно отрендерены Poppler и
визуально просмотрены root agent; сообщения об успешном экспорте не заменяли файлы.

| Файл | SHA-256 / страницы | Сверка |
|---|---|---|
| [AVD выбранная v2](demo23-reference-v2.pdf) | 7eddee6d284f2958d7e1ef47a0d42a6019a02db2deb837ca079adf945059c887 /5 | 32/32: свежий version snapshot, ref2м,8.10м/5.19м, фото/7paths/единицы/источники/privacy |
| [AVD выбранная v3](demo23-reference-v3.pdf) | f91c5e5756d67e9d0bca4ed808ad95aa45fa4465f2446670905dd90c857611b2 /5 | 32/32: ref3м,12.15м/7.79м, правильные version/parent; ещё8/8 cross-version checks |
| [S24 DEMO](s24-demo23.pdf) | e9f0beea94d83b32e054129c7d63ff56f0d1abccc58a9f41fa12f8cd0414e78c /5 | 12/12: содержимое, ref1м/height4.8м/crown2.1м/image400×600, privacy; отдельное server-original byte comparison не выполнялось |

Подробные checks/байты/PSNR/annotation bounds/units/provider dates/limitations:
[AVD QA](pdf23-qa.json), [S24 QA](s24-pdf23-qa.json).
AVD originals256×384 относятся к явным DEMO, GPS50.2/10.2 контролируемый.
S24 PDF содержит только ранее разрешённую DEMO точку50.2001/10.2,
не полученные сейчас реальные координаты телефона. Отсутствующие β,
DBH или неопределённости не подставлялись: недоступность указана в PDF.

## Рендеры фактически просмотренных страниц

AVD v2: [1–2](demo23-v2-pages-1-2.png), [3–4](demo23-v2-pages-3-4.png),
[5](demo23-v2-pages-5-5.png). AVD v3:
[1–2](demo23-v3-pages-1-2.png), [3–4](demo23-v3-pages-3-4.png),
[5](demo23-v3-pages-5-5.png).
S24: [1](s24-demo23-page-1.png), [2](s24-demo23-page-2.png),
[3](s24-demo23-page-3.png), [4](s24-demo23-page-4.png), [5](s24-demo23-page-5.png).

Native S24 доказательства: [сохранение](s24-pdf23-save.json),
[viewer](s24-pdf23-open.json), [share без отправки](s24-pdf23-share.json).
Viewer открыт из app cache; Download-файл отдельно выгружен и проверен.
Произвольное ручное открытие Download-файла через файловый менеджер не заявлено.
PDF-значения не доказывают полевую точность или native Google login.
