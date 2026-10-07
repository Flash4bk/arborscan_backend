# AS-16 — подписанный кандидат 1.3.0 и остаток AS-14

Работа от `origin/main` **10b54e77ae77fa7a51621bc1902889ce4d2a8ef6**,
в отдельном managed worktree `codex/final-release`, по
`AS16_FINAL_RELEASE_PROMPT.md`. Исходный `D:\arborscan_backend`, его пользовательские
PDF и кеши сохранены. Это технические результаты агента. Пользовательская,
визуальная и экспериментальная приёмка не назначены.

**Стабильный release/tag не создан:** настоящий Google-вход с постоянным
сертификатом на API36 и с прежним сертификатом на API24 проверен раздельно. Обнаруженное серверное security исправление подготовлено, но
его rollout запрещён текущим заданием и не выполнен. Кандидат можно
доставлять для проверенных функций. AS-14 не закрыт отсутствующим independent
receiver и новой VM/HTTPS. Молчание владельца не означает принятия этих ограничений.

## Точный артефакт и исходник

| Поле | Фактическое значение |
|---|---|
| Пользовательская версия / versionCode | **1.3.0 / 23** |
| Исходник APK | **bbcd5b539427e62da58739fc2dbeff8550ee72c6** |
| Файл вне Git | `D:\arborscan_backend\output\final-release\ArborScan-1.3.0-23.apk` |
| Размер | 115594211 байт |
| SHA-256 APK | `71f38189ee62a7825029ec7c25bf635891d3adcd0fa3eea3d9b618a223e4a31c` |
| applicationId | `com.example.arborscan_app` |
| min / target / compile SDK | 24 / 36 / 36 |
| Release / debuggable | release / false |
| Rotation min SDK / lineage SHA | 33 / `93d10d0e6b813bf9053debf1cc879378a4e90e2d0e07ac755eba50eb57a87dfe` |

API24–32 сохраняют прежний сертификат SHA-256
`68ff9861f52fdb22744b30ee27092008097f9eff4ead7eb38f80e662115f5b97`;
API33+ применяют постоянный
`7fb94ede4099199db9166609c34d6278ebe51c7d80bd6ce1cd0208bf863863b8`.
Первый — прежний debug certificate совместимости. Его сохранение не называется
ротацией всех платформ или готовностью Google Play/RuStore. Поля SHA-1 OAuth и
точный native-контракт находятся в [FINAL_RELEASE_GOOGLE.md](FINAL_RELEASE_GOOGLE.md).

Подпись проверена `apksigner` для диапазонов24–27/28–32/33–36, embedded lineage
и её capabilities, manifest/nondebuggable/INTERNET и16KiB zip alignment проверены
независимо от подписывающего инструмента. Постоянный ключ не регенерировался;
прошлое фактическое восстановление escrow/signing сохранено в
[RELEASE_READINESS.md](RELEASE_READINESS.md), без необоснованного повторения.

## Исправления и программные проверки

- **6284f93** — подготовленная verified Google identity policy, защищённый subject
  linking и аддитивная SQL-предпосылка. Это server source, не production rollout.
- **91f17b5** — ProfilePage сохраняет действующую сессию при отмене/ошибке Google;
  сериализация записей и owner/revision guards не дают запоздавшему ответу другого
  аккаунта менять профиль, токен или черновик. Busy блокирует повторные операции.
  Nullable idle queue корректно работает при пересоздании экрана/async zone.
  Клиент проверяет форму ответа и передаёт только Google ID token.
- **750e9b6** — промежуточный catch renderer/inflation errors устранил process crash,
  но чистый API24 обнаружил ANR в AndroidX Fragment teardown после частичного inflation.
  APK20 не доставлялся на S24 и не считается рабочим итоговым кандидатом.
- **495ea37** — OpenGL ES3.0 preflight и создание общего Filament engine до Fragment
  inflation. Wrapped ARCore failures сохраняют recovery/cancel semantics. Измерительные
  формулы, library versions и данные не менялись; фактическая проверка возврата21
  прошла на clean24 (два responsive возврата/то же фото/cold restart).
- **36c44aa + e78c29d** — verified owner hydration старой сессии не скрывает профиль,
  когда история одновременно заполняет ID того же owner. Red15pass/1fail, затем26pass;
  foreign owner/token/revision и отсутствие authenticated response остаются fail-closed.
  Единственный новый lint исправлен, final sourcee78c29d получил full163/analyze53old.

- **8d675a4** — проверка точного `GET_CONTENT image/*` перед Android gallery,
  согласованная с locked image_picker_android0.8.13+9/PhotoPicker=false и package
  visibility queries. При отсутствии provider plugin не вызывается, фото/AR/
  разметка сохраняются. Camera/iOS/desktop не менялись. Новые16 tests и целевые25
  прошли; автоматические тесты не подменяют реальную native33 проверку.
- **bbcd5b5** — запоздавшая gallery error после смены владельца не заменяет
  сообщение нового аккаунта/invalid-session. Red4pass/2fail, green18/18; оба
  новых widget cases входят в full181. Это узкое исправление catch, не смена
  геометрии. Native preflight не обещает защиту от удаления provider между
  query и start; при смене режима PhotoPicker/lock нужно согласовать exact intent.
  Источники: [locked plugin](https://github.com/flutter/packages/blob/image_picker_android-v0.8.13%2B9/packages/image_picker/image_picker_android/android/src/main/java/io/flutter/plugins/imagepicker/ImagePickerDelegate.java),
  [Android package visibility](https://developer.android.com/training/package-visibility/use-cases).

| Проверка | Среда / результат | Граница |
|---|---|---|
| Полный Flutter suite | Windows Flutter3.41.1/Dart3.11, **181 passed**, exact sourcebbcd5b5 | После profile, gallery availability и late-error owner fixes; native Google transport в widget-тестах заменён mock |
| Новые auth/guard тесты | Первоначальные15 и новые11, целевые **26 passed**, все включены в final181 | Отмена, сеть, double tap, late replies, owner switch, logout queue, draft/service recreation |
| Flutter analyze | **0errors,5warnings,48info**, exit1 | Все53 существовали; первоначальные3 новых test-only lint исправлены |
| Android release unit | JDK21/Gradle8.11.1/AGP8.9.1/Kotlin2.1, **9 passed** на sourcebbcd5b5 | После build23, test task действительно выполнен; прежние5 на19 сохранены |
| Release APK23 | **163.0с**, default pub, без dart-define | APK19 ранее121с; первый `--no-pub` fresh-worktree build отказал stale dev registrant и не считается успехом |
| APK secret-pattern scan | 0private key/env/pg filenames, PEM markers, service-role JWT / PostgreSQL URL constants | Не доказательство отсутствия любых произвольных секретов; private значения не читались |
| Google policy | **17 passed** в текущем pinned Python3.11 image; Windows3.14 также17 | Exact handler AST + synthetic verifier/REST; не настоящий Google OAuth |
| SQL prerequisite | **16/16** настоящий PostgreSQL17.6, network-none/tmpfs lab | Реальная миграция synthetic users, не текстовый анализ и не production SQL |
| Runner safety | **8/8 Windows +8/8 Linux3.11** | Host option injection/cleanup guards; новые guards не выданы за повторPG16 |

При fresh checkout `flutter pub get` включает dev-only Android registrants,
а release Gradle исключает их. Во Flutter3.41 `build --no-pub` пропускает нужное
platform tooling regeneration. Проверенная команда release — штатная сборка с pub,
затем Android unit tests; рабочая runtime dependency не изменялась ради обхода.
Подробные cache/AVD команды и границы —
[FINAL_RELEASE_ANDROID.md](FINAL_RELEASE_ANDROID.md).

## Устройства, интерфейс и контрольные файлы

Промежуточный19/source91f17b5 SHA `c0fd15a4c2e9280984df3b4b21d58440ce26530388710858fb2efee0f66df30b`:
AVD24/33/36 обновление18→19 и repeat `-r` сохранили73/73/71 файлов, без изменений,
до UI. S2418→19 сохранил62/62; profile/cold history/DEMO read-only/AR activity/back
прошли, экран сообщил tracking lost. Не проверена физическая точность/tracking.
Чистый24 подтвердил startup/cold restart, но выявил renderer crash: это **failed**,
не успех AR. Его stack и отдельные безопасные screenshots сохранены.

На isolated DEMO AVD36 **19** проверены реальный email UI login/logout/cold restore,
native Google «Checking info» → Back/cancel → retry/cancel (без нового login),
несохранённый эталон2→3м/force-stop/восстановление12.15м/7.79м, две выбранные
server versions, настоящий SAF save/open/cancel и share-sheet/cancel без отправки.
Два PDF19 по5страниц выгружены с совпавшими device SHA; все10 страниц отдельно
отрендерены и визуально просмотрены. 22 content checks для каждого +8 cross-version
прошли: координаты/единицы, photo256×384/original SHA, семь annotation paths,
разные размеры при1/2м и unavailableβ/DBH. Saved weather/soil являются реальными
ответами06.10, не новым запросом QA; photo/geometry/GPS явно DEMO.
Ранний pull v1 увидел пустой SAF placeholder до окончания copy; harness дождался
stable nonempty файла. Нулевой diagnostic не выдаётся за контрольный PDF.

APK20 clean24 вернул MainActivity, но UI не отвечал; реальный ANR и main-thread
stack сохранены. Это failed regression, а не успешный fallback. API24 объявляет
OpenGL ES2.0, недостаточный для Sceneform. На21 реальный red→green прошёл: два responsive возврата, сохранённый photo region,
холодный запуск, no fatal/ANR. После этого независимый code review и synthetic test
нашли гонку legacy profile owner. Кандидат22 исправил её: 163Flutter,9Android, clean24/33/36 и обновление24/36
прошли. Отдельный ATD33 выявил gallery failure без системного handler; это
подтверждённый failed22, а не успех интерфейса. Кандидат23 содержит исправление;
его фактическая матрица и финальный UI-цикл ведутся отдельно.
Доказательства19/20 не переименовываются в21.
Private inventories/account/history screenshots не публикуются.

Финальный **23/sourcebbcd5b5** проверен отдельно:

| Проверка | Реальное окружение / результат |
|---|---|
| Чистая установка, startup/cold restart | Отдельные пустые AVD API24/33/36, точные code23/сертификат/nondebuggable |
| Сохранное обновление и повтор | Valid API24:73/73 SHA; новый valid API33 lab:2/2; API36 lab:3/3. Небольшие fixtures не называются представительскими73 файлами |
| Прямое18→23/repeat | Собственный API36:8/8 SHA; normalUI сохранённый отчёт5.00м/2.93м и незавершённый ввод3м восстановлены после guestOS reboot. Это отдельная проверка от22→23 |
| Gallery без provider | API33: понятная inline error, повтор/навигация responsive, тот же process, без нового fatal/ANR. Проверка22 действительно падала; результаты23 — red→green |
| Нет ARCore | API36: системный Google Play запрос входа, Back/cancel/retry сохраняют фото/process; сервис не установлен. Это не успешная установка/tracking |
| S24 установка | Android16/API36,22→23 и repeat,62/62 SHA в пяти папках идентичны до UI; данные/аккаунт не очищались |
| S24 функции | Тот же профиль и история после force-stop, разрешённый DEMO отчёт read-only, native Gallery/Back, ArMeasureActivity/Back и responsive MainActivity |
| S24 PDF | Реальный SAF save, Download/pull SHA совпадают; native Drive PDF viewer открыт из export dialog; share-sheet/Back и SAF cancel возвращают к рабочему отчёту. Адресат не выбран, ничего не отправлено |
| Модерация23 | Автор увидел отклонение/причину, восстановил черновик после force-stop, сохранил/отправил ровно одну дочернюю ревизию после двух быстрых нажатий. S24 принял только эту allowlisted DEMO ревизию; автор увидел принятие. Новая правка создала draft, родитель остался accepted. Stale-parent save/retry показали конфликт и сохранили черновик без изменения цепочки. Старый PNG открыт с явным созданием нового контура |
| Две серверные версии23 | UI эталон2→3м, несохранённый ввод восстановлен после force-stop; новая серверная v3 создана один раз. PDF выбранных v2/v3 по5страниц:32/32 проверок каждый и8/8 сравнений; все10 страниц отдельно отрендерены и просмотрены |
| PDF S24 — реальные байты | 55774байт/5страниц, SHAe9f0beea…:12/12 проверки содержимого/приватности и просмотр всех5страниц. Системные save/open/share-cancel/cancel проверены; отдельное сравнение фото с серверными байтами здесь не выполнялось |
| Карта/источники/GPS23 | AVD: реальные Google Maps tiles/DEMO marker, UI-запрос OpenWeather/SoilGrids07.10 21:24UTC; partial soil явно обозначена. S24: свежий GPS-result dialog/0мин/±17м, отменён без server-save; точность не подтверждена |
| Анализ23 | Native picker выбрал собственный synthetic JPEG, настоящий Analyze завершился «Дерево не найдено» без искусственной маски; локальная запись сохранена. Это проверка доступности, не качества модели |
| Offline23 | Только собственный AVD: сеть выключена, force-stop, локальная запись/экспорт/реальный PDF viewer; сеть восстановлена и выбранная v3 загружена из серверной истории |

Границы и подробная матрица: [FINAL_RELEASE_ANDROID.md](FINAL_RELEASE_ANDROID.md).
Прямой переход/авторский цикл: [release23-ui.md](evidence/final-release/release23-ui.md).
Проверки S24: [SHA установки](evidence/final-release/s24-update23.json),
[UI](evidence/final-release/s24-ui23.json),
[native gallery/AR](evidence/final-release/s24-native23.json).
Новый23 не наследует proof tracking19 или старые PDF19. Native viewer открывает
файл из app cache; Download-файл отдельно выгружен с совпавшим device SHA.
Все15 страниц трёх PDF23 отрендерены и отдельно визуально просмотрены агентом.
[Контрольные PDF и сравнения](evidence/final-release/pdf23-qa.md),
[полный DEMO цикл](evidence/final-release/release23-ui.md),
[модерация](evidence/final-release/moderation23-workflow.json),
[offline](evidence/final-release/offline23.json),
[источники](evidence/final-release/provider23.json),
[GPS S24 без координат](evidence/final-release/s24-gps23.json). AR tracking/физическая точность, Google login и интерфейсное принятие
пользователем из этих проверок не следуют. Сторонние уведомления и личные экраны
S24 в публичные screenshots не включаются.


В конце проверки исключены из будущего exporter только три новые allowlisted
синтетические DEMO ревизии: существующий marker
`purpose=export_smoke_test_only_not_model_evaluation`. До записи сохранена и
проверена приватная копия JSON/SQL metadata. Через существующий Storage API
изменено только ancillary поле purpose; оригинал, PNG, editor state, revision,
parent, SQL status/решения полностью совпали до/после. Публичный авторизованный
HTTPS GET подтвердил все три записи. Сам действующий exporter отверг их для
segmentation и classification; принятый DEMO без marker ранее проходил exporter.
Первое немедленное чтение Storage было запоздалым, ограниченный readback/retry
подтвердил durable marker. Датасет не фиксировался, jobs/training не запускались.
Это служебная изоляция собственных тестовых данных, не новый пользовательский API
и не автоматическое наследование purpose будущими правками. Если продолжать
модерировать этот DEMO, каждую новую тестовую ревизию нужно отдельно исключить
до создания датасета. [Точный proof](evidence/final-release/moderation23-test-quarantine.json).

## Фактический production manifest

Read-only срез07.10, после всего UI и изоляции DEMO дополнительно **22:30UTC** на final23: оба публичных HTTPS health200/statusok со штатной TLS-проверкой;
`/api/v3/auth/me`, `/api/v4/v4/reports`, `/api/v4/v4/corrections` без токена дают401.
[Итоговый runtime proof](evidence/final-release/runtime-identity-after-ui23.json);
[предыдущий срез18:38](evidence/final-release/runtime-identity-final23.json)
подтвердил те же images/start times и отсутствие prepared Google policy на VPS.
V3 handler и весь canonical AST совпадают с `main10b54e7` (сравнение нормализовано
для Python3.11/3.14; различие их AST dump не выдаётся за другой серверный код).

| Компонент | Запущенная версия / образ |
|---|---|
| API v3 | `sha256:5a0b1d63e6c5eb13d14af84590dafebf900f59156613b987000ae8b6938f563b`, старт07.09 |
| API v4 | код **0437e98**, `sha256:034a9d67b49adb5565a1af9454706a104721e21160d725db32a31f796ff20193`, старт06.10 |
| Quality worker | `sha256:3fe28a5b7eb5716a71e984908520c31fff9ac325a9c5885c5e102e93996ac14a`, старт29.09 |
| API/schema advertised v3 / build metadata | `3.0.3` / `1.0.0` / `a34400f`; build metadata не заменяет сравнение фактического server.py |
| API/schema advertised v4 | `4.0.0-alpha.1` / `4.0.0-alpha.1` (не номер APK) |
| SQL contour/history/model quality | **1 /1 /1**, read-only SQL в этом пакете |
| PostgreSQL | **17.6**, настоящий read-only `current_setting('server_version')` в существующем Supabase проекте |
| Active segmentation | **model_v4**, segment, conf0.25/imgsz1024,26626466байт |
| SHA модели | `626f62b97b05fd65275dc3f909c3aec090f6c9957c4ef9f12c4e743a1f75a2bb` |
| Геометрия / эталон | measurement_method_version2 / `known_object_segment_v2` |
| AR | `ar_measurement_v4`; height `fixed_vertical_tree_plane_ray_intersection_v1`, DBH `cylindrical_tangent_rays_median_v2` |

Production runtime API/worker, схема БД, HTTPS и модели не менялись; containers
не перезапускались, обучение не запускалось. Разрешённые UI-проверки могут создавать
версии отдельных тестовых DEMO-записей на сервере; это не утверждение неизменности
всех данных БД. Реальные пользовательские отчёты и решения не изменяются.
Лабораторные PG containers не монтировали production данные/секреты и удалены
по ownership guards.

## Google и эксплуатационные блокеры

Native Google ID token завершается **собственной ArborScan-сессией**, не Supabase
Auth. При повторном открытии консоль существующего проекта оказалась
доступна: прежние Web/Android credentials проверены, новая Android-регистрация
постоянного signer добавлена19:00:57UTC. Прежняя пара сохранена, API keys/scopes/
публикация audience/сервер не менялись. Позднее пользователь добавил разрешённый
существующий test Google account; агент проверил2testusers без публикации emails.
После приватного ввода владельцем credentials/MFA агент проверил настоящий
Google-вход на отдельном API36 со signer7fb94e…: возврат в ArborScan/серверная
сессия/пустая собственная история, force-stop/revalidation того же аккаунта,
logout/relogin, native chooser cancellation, offline error и network-retry.
Read-only проверка только разрешённого test identity подтвердила одну строку
и неизменные owner/subject/role/created_at после повторов. Это legitimate flow
старого production handler, не проверка развёрнутой новой security policy.
[Actual native36](evidence/final-release/google23-native36.json),
[offline](evidence/final-release/google23-offline.json),
[identity continuity](evidence/final-release/google23-identity-continuity.json).
S24/его Google-аккаунт не затрагивались. Прежний signer API24 затем проверен
отдельно: exact APK/source, тот же test account/canonical owner, actual Google
login/session/history/restart/logout/relogin/chooser-cancel/offline-error/retry.
[Native24](evidence/final-release/google23-native24.json),
[offline24](evidence/final-release/google24-offline.json),
[тот же owner](evidence/final-release/google24-identity-continuity.json).
Protected ADB network commands получили Permission Denial, root/override не
применялись. Offline проверен штатным Settings UI; после airplane-off persisted
wifi flag3 исправлен на первоначальный1 через разрешённый Settings provider
(отдельный Wi-Fi Activity отсутствует в этом image). Первичный failed restore
сохранён, итоговые0/1/1 и успешный retry проверены. Сохранённая сессия S24 и
mock не выдаются за отдельный old-signer login; native API28–32/каждая Android
версия и OTA32→33 здесь не проверялись. Фактические поля и
доказательства — [FINAL_RELEASE_GOOGLE.md](FINAL_RELEASE_GOOGLE.md).

Найден concrete server defect: старый handler может использовать присланный
клиентом email при отсутствующем verified email, не проверяет email_verified и
допускает неоднозначный subject/email linking. Воспроизведено только синтетически.
Защищённая реализация/DB contract готовы и проверены, но deployment v3/SQL
этим заданием запрещён. Нужны отдельное разрешение и exact rollout по
[FINAL_RELEASE_GOOGLE.md](FINAL_RELEASE_GOOGLE.md). Открытый безопасный вход с
новой подписью до этих предпосылок не объявляется готовым.

AS-14: read-only срез подтвердил прежний calendar service success, состояние timer
и наличие 14 COMPLETE-наборов; прежние full restore/key tests сохранены. Этот срез
не является новым восстановлением всех 14 наборов. Нет назначенного независимого
always-on receiver, clean VM и отдельного test HTTPS имени/прав. Конкретные объёмы,
подготовленные команды, автоматизация и её недоказанные части —
[FINAL_RELEASE_AS14.md](FINAL_RELEASE_AS14.md). Новых платных ресурсов не создано.

## Сохранный клиентский откат — подготовлен, не выполнен

Android не допускает обычное сохранное18/19/20/21/22 поверх23. Не использовать uninstall,
clear data, `-d` или подпись другим ключом. Если нужен откат приложения, от проверенного
старого исходника собрать **новый code24 или строго выше установленного**, прежний applicationId,
ту же lineage/платформенные signer и API-compatible форматы. Сначала проверить
restore/update на isolated AVD, затем `adb -s DEVICE install -r APK` и SHA inventory
до UI. Серверные данные не восстанавливать поверх production ради клиентского отката.
Это план, фактический rollback +23 не выполнялся. Старые локальные/серверные форматы
сохраняются; несовместимые будущие изменения потребуют отдельного migration/backup.

Backend security rollback также только подготовлен: claims=false на всехinstances,
точный preimage/config при необходимости, additive unique index сохранить.
Возврат старого Google handler возвращает найденный дефект и не является постоянным
безопасным решением. Current production image этим пакетом не заменён.

## Короткая инструкция по функциям и данным

В «Анализе» добавьте фото; измерения доступны через AR или известный вертикальный
эталон. Эталон/дерево должны быть примерно на одной глубине; проекция, EXIF и масштаб
не дают автоматически исправленной перспективы. DBH без нужного измерения остаётся
недоступным. Field accuracy/AR tracking не выводятся из запуска экрана.

«История» открывает выбранные серверные версии отчётов. Исправление маски создаёт
отдельную ревизию, не переписывает физические измерения; отправка на проверку и
администраторские решения доступны по серверным правам. Принятие маски не подтверждает
высоту/DBH/механику и не запускает обучение. PDF отражает выбранную версию; «Сохранить»
использует системный dialog. «Поделиться» открывает системное меню: выбор приложения
передаёт ему PDF, а отправку конкретному адресату пользователь выполняет в нём.

Серверные отчёты/контуры и подтверждённые ревизии доступны аккаунту после входа на
другом устройстве. Локальные кеши, несохранённые drafts и локальные измерения изолированы
по аккаунту **на устройстве**, это не межустройственная синхронизация. Force-stop/
перезапуск сохраняет уже записанные на диск данные; выход скрывает их от другого
пользователя, не является upload. Очистка/удаление приложения может удалить локальные
drafts; собственного переноса на новый телефон нет. Восстановление средствами Android
backup/переноса телефона здесь не проверялось, поэтому сохранность этих данных не
обещается. Не загруженный на сервер материал нужно заранее экспортировать. Сессия может
истечь/быть отозвана, тогда требуется новый вход. Для offline нужны уже скачанные
фото/разметка/данные; погода/почва показывают фактическую доступность/время источника,
не вымышленные значения.

β в кг/с по фото/AR сейчас не вычисляется. Исследовательский solver синтетически
проверен отдельно, не подключён в production; экспериментальные AS-10/AS-13,
устойчивость AS-11, независимое качество моделей и визуальная приёмкаAS-15 остаются
в [общем плане](ARBORSCAN_MASTER_PLAN.md).
