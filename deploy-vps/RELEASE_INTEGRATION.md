# AS-16 — интеграция и фактический остаток, 06.10.2026

Отчёт текущего пакета по `AS16_INTEGRATION_AND_AUDIT_PROMPT.md`. Канонический
план — [ARBORSCAN_MASTER_PLAN.md](ARBORSCAN_MASTER_PLAN.md). Основной ID AS-16;
затронуты AS-08/AS-09, воспроизводимость AS-10 и регрессия истории AS-02, ввода AS-03 и эксплуатационный остаток AS-14.
Это debug-кандидат, а не выпуск с постоянной release-подписью. Результаты
агента, прежняя приёмка пользователя и научная валидация разделены.

## История и конкретные исправления

От `origin/main` `823ebc46f5133c2fb0e3bc00cf12827f84091206` создан отдельный
managed worktree и `codex/release-integration`. `codex/app-redesign`
`ab25e39836ffa68af5f00abbe724eb2af80ce66a` уже предок `codex/beta-dynamics`
`0fe95eb9a6d5c115b8c094bff68addaa86d7d14a`: интеграция выполнена fast-forward,
без повторных cherry-pick. Изучен полный объединённый diff (239 файлов),
а не только последний документ. Оригинальный `D:\arborscan_backend` остался
на прежней beta-ветке с пользовательскими PDF/кешами; они не включены в коммиты.
Рабочий checkout:
`C:\Users\danik\.codex\worktrees\release-integration\arborscan_backend`.

Готовые до пакета возможности сохранены: редактор/ревизии/модерация,
серверная история, reference v1/v2 и AR, геометрия, погодные/почвенные snapshots,
новый дизайн, версионный PDF и самостоятельный research CLI.

Исправления:

1. `7d05e5133170a49151d577c227538dc9bfddb9da`: историческая метка DBH не
   объявляет измерение подтверждённым; отчёт выводит frozen geometry выбранной
   версии; NaN/Infinity показываются недоступными, допустимый нулевой угол
   сохраняется. Формулы/разметка/измерительный протокол не изменены.
   Добавлена отсутствовавшая в чистом checkout пустая declared assets directory.
2. `0437e98e0570e0b74a1705b5d3095d32f038ddda`: реальный конфликт двух сохранений
   оставлял в Storage immutable blob до commit ревизии. Listing теперь
   проверяет owner-scoped metadata одним запросом на страницу и исключает
   незавершённый schema-2 blob; committed фото повторно не скачиваются.
   Старые PNG/schema-1 доступны; сбой БД/Storage не превращается в ложную пустую
   историю. `next_offset` относится к сырой странице Storage. Flutter позволяет
   загрузить следующую страницу даже после пустой отфильтрованной первой.
   Никакие прежние записи не удалялись и не переиндексировались.

3. `9010886489d8201a0a7c8310c1010e5f2d229515` и `8ef27b82a89876e63690302836020db2674d6583`: guarded prepare/deploy/rollback только v4, lock, exact-image/code/env/model/config invariants, свежий backup и отдельная проверка native PostgreSQL manifest. 29 новых ops safety tests плюс8 прежних weather; реальный rollout выполнен.
4. `1ab174304335f4bd93fe04418d0b3a77f70c63d4`: найденный на AVD14 дефект очистки длины эталона — исчезновение Export сдвигало lazy list и уничтожало focused input. Стабильный slot/key сохраняет существующее подключение ввода. Red→green widget проверяет100→пусто→200 без refocus и пересоздание store/screen; фактический AVD15 подтвердил исправление. Формулы/состояние разметки не изменены.

## Геометрия, среда и хранение

Подробные матрицы «код → доказательство → остаток»:
[геометрия](evidence/as16-integration/geometry-audit.md),
[среда](evidence/as16-integration/environment-audit.md),
[GEO_ENVIRONMENT.md](GEO_ENVIRONMENT.md).

- Высота/живые размеры кроны/сечение/наклон reference используют координаты
  полного EXIF-oriented оригинала, его W/H и вертикальную ось эталона.
  Zoom/pan не меняет координаты и масштаб. 1 м/100 см эквивалентны.
- Произвольное сечение не называется DBH. Уровень 1,30 м и историческая метка
  сами не доказывают соблюдение полевого протокола. Перспектива и разные
  расстояния до камеры остаются ограничениями проекции.
- Общая маска дерева не является отдельной gap-preserving маской кроны;
  физическая пористость недоступна без необходимых входов и проверенного метода.
- AR — самостоятельное измерение с привязкой SHA фото. AVD не доказывает
  tracking/полевую точность. Из фото/AR не выдаётся β реального дерева.
- Почва уже реализована: ISRIC SoilGrids WCS, 0–5 см mean, сетка 250 м,
  clay/sand/silt %, SOC г/кг, pH безразмерный; это модель сетки, не измерение
  корней/полевого профиля. Прямая проверка пяти TIFF прошла. Погода сохраняет
  прежнего провайдера OpenWeather и штатную TLS-проверку.
- Геоточка содержит provenance GPS/EXIF/manual; нет фиктивного fallback.
  Snapshot среды принадлежит конкретной версии; открытие старой версии
  не обновляет её текущей погодой. Покрытие Street View отсутствует для нужных
  мест: принятое ограничение, не дефект и не повод менять провайдера.
- Несохранённый черновик/локальный reference/кеш фото и отчётов хранятся
  только на устройстве, изолированы по владельцу. Переживают force-stop и
  возвращение того же аккаунта, но не очистку данных/удаление приложения.
  Это не синхронизация. Серверные версии явно сохраняют оригинал, SHA,
  reference/report snapshot, автора/родителя; доступны на другом устройстве
  после входа. Offline требует заранее полученных локальных данных.

## Автоматические проверки точного кода

| Проверка | Среда / результат | Доказательство |
|---|---|---|
| Полный backend после добавления rollout helper | Фактически собранный candidate image `034a9d…`; Git sources `9010886`, API/config совпадают с `0437e98`; сеть отключена, rootfs RO; **216 passed, 3 subtests, 1 Windows-only skip**, 12,36 с | [backend-tests-candidate901.json](evidence/as16-integration/backend-tests-candidate901.json) |
| Последняя правка backup-контракта helper `8ef27b8` | После full suite изменились только helper/tests: **37 targeted passed** (29 rollout + 8 прежних weather). Реальная prepare/deploy также выполнена | [v4-rollout-safety-tests.json](evidence/as16-integration/v4-rollout-safety-tests.json) |
| Windows Job Object | Настоящая Windows, отдельный свежий venv; **1 passed** | [windows-job-tests.json](evidence/as16-integration/windows-job-tests.json) |
| Flutter финальный код | Windows Flutter 3.41.1/Dart 3.11.0; **130 passed**, 25 с | [apk15-provenance.json](evidence/as16-integration/apk14-provenance.json) |
| Android геометрия/AR failure handling | JDK 21.0.8, Gradle 8.11.1, compile/target SDK36; **5 passed**; native source после прогона не менялся | [apk15-provenance.json](evidence/as16-integration/apk15-provenance.json) |
| `flutter analyze` | **exit1**, **0 errors, 5 warnings, 48 info**; все 53 существовали в +12, новых нет | [analyzer-comparison.json](evidence/as16-integration/analyzer-comparison.json) |
| Research AS-10 | Чистый Python3.14.6 venv, только pinned `requirements-beta.txt`; **70 passed**, 230,42 с, pip check | [research/README.md](evidence/as16-integration/research/README.md) |

Backend guards/moderation/revisions/history/reference/environment проверены
на синтетических данных. Native skip выполнен отдельно на Windows. Два
Starlette/AnyIO warning существовали ранее; это предупреждения, а не errors.
Новый listing покрыт 12 backend cases и реальным widget-переходом пустая
страница → следующая → открытие PNG. Исходники/тесты full backend архива
(98 файлов) сверены с Git objects `0437e98`; последующий прогон в реальном candidate-образе
сверил все 100 файлов `9010886`. Для последней tool-only правки повторены
затронутые 37 tests. После найденного UI-дефекта Flutter проверен новым полным
прогоном130 на1ab1743; предыдущий129 относится к0437/APK14. API/native/research
после соответствующих прогонов не менялись.

Начальные технические отказы сохранены в provenance: Windows mmap-lock PDF
в тесте (только task-generated PDF сохранены отдельно, затем полный suite прошёл),
неподходящая Java8/ошибка пути JAVA_HOME исправлены использованием существующего
JDK21. Промежуточный APK+13 после переноса generated build на D: не содержал
Dart kernel и застревал на native splash; отклонён. Task-owned кеш сохранён
в отдельном архиве, +14 пересобран, kernel 61 674 608 bytes и холодный запуск
фактически проверены. Никаких пользовательских данных/кешей не очищали.

AS-10 отдельно: реальные CLI и replay восстановили синтетические 2,4 кг/с как
**2,400001788258296 кг/с**, CSV совпал побайтно. Полный численный протокол
126,512 с; n=20/короткое окно оставляет **β=null**, кандидат 80 не результат:
сигнал 5,386×10⁻⁷ м меньше заданного 10⁻⁵ м. Входы, единицы, версии и семь
SHA COMPLETE проверены; все восемь диагностических PNG просмотрены.
Git CRLF/LF отражён явно: raw SHA checkout может отличаться от прежнего LF,
canonical Git blobs совпадают. Research не подключён к production API/worker;
настоящие экспериментальные AS-10/AS-13 и научные вопросы AS-11 открыты.

## APK и доступные интерфейсные проверки

Финальный клиент: **1.3.0-integration.1ab1743**, **versionCode15**, source
`1ab174304335f4bd93fe04418d0b3a77f70c63d4`,
`com.example.arborscan_app`, minSDK24, targetSDK36. Без dart-define,
публичные штатные HTTPS defaults. APK SHA-256:
`b0ddc1292fc894ae8c39c5e41d7c14a1fdf906a4b51aa67c7e4d0b3e45d813fe`.
Прежний debug certificate SHA-256:
`68ff9861f52fdb22744b30ee27092008097f9eff4ead7eb38f80e662115f5b97`.
Артефакт оставлен вне Git:
`arborscan_app/build/app/outputs/flutter-apk/app-debug.apk` managed checkout
(build junction на task-owned D: output).

S24 отсутствует в `adb devices -l` и доступном mDNS inventory. +15 на S24
**не установлен**, его текущая версия/локальные файлы сейчас недоступны.
Прежняя пользовательская приёмка и +12 установка остаются историческими
доказательствами; это не проверка новой сборки на телефоне.

AVD `Codex_AS12_20260925`, emulator-5582, Android16/API36: `adb install -r`
успешен; все **61/61** файлов files/shared_prefs до/после+15 сохранены по SHA.
Приватная tar-копия488960B подготовлена, восстановление не выполнялось.
Холодный запуск Statusok,2962мс. Прежний+14 сохранял49/49, затем отдельный
DEMO UI-цикл добавил данные; это не сравнение разных наборов. Реальный
пользователь не затронут. [APK15 provenance](evidence/as16-integration/apk15-provenance.json).

Самостоятельные UI/file результаты с точными APK — [UI evidence](evidence/as16-integration/ui/README.md):

- +14: picker/исходный DEMO256×384, замкнутый контур и6отрезков;1м=100см;
  локальный save/force-stop; реальные SoilGrids/OpenWeather snapshots;
  account save→новая версия2м; два быстрых Save создали1исходную версию.
- +15: input100см→пусто→200см без касания; force-stop сохранил значение/точки.
  Вновь открыты обе выбранные серверные версии:4.050847→8.101695м. Их полные
  snapshots повторно прочитаны по реальному HTTPS и остались структурно
  прежними; UI-open не делает новый inference/lookup среды.
- Четыре настоящих контрольных PDF выгружены: два локальных (+14,4+4страницы),
  две серверные версии (+15,5+5страниц). **18страниц** отрендерены/просмотрены;
  дополнительно geometry fixtures8страниц. Фото/разметка/числа/единицы/версии
  и историческая среда совпали. SAF-save/Android Viewer/Android Files,
  отмена, настоящее Share sheet без отправки проверены; не только сообщение.
- +15 offline после force-stop открывает старый cached server-report;
  карта сохраняет5версий, после восстановления сети refresh вновь их загрузил.
  Состояние сети возвращено0/1/1. Подложка Google Maps не обещается offline.
- +14 AR photo-bind cancel/камера denied/AVD unavailable-session возвращают фото
  без падения; ARCore1.56.262080393 установлен. Native/AR-код после проверки
  не менялся. AVD недоступная сессия не является проверкой tracking на S24.
- +15 все6DEMOконтуров открылись: legacyPNG/overlay без придуманных точек,
  rejected reason, accepted mask и новые draft. Серверная schema2 ревизия
  восстановила3исходные точки в настоящем редакторе. Решения UI не назначались:
  права/decision writes/retry/conflict проверены отдельными DEMO HTTPS/backend.
  Crash buffer:0AndroidRuntimeFATAL. Приватные токены/аккаунты/rawbackups внеGit.

Небольшие ошибки селекторов проверочного скрипта исправлены под существующие
local-cache timestamps и GET`/v4/reports/{version_id}`; это не изменения
приложения/API. Screenshot transition frames повторно сняты после settle.

## VPS, резервные копии и откат

Исходный `/opt/arborscan` чистый, HEAD
`5d8176a22fddd6634c266a1ed09f1845019065c5`; checkout не перезаписан.
Фактический старый v4 — код `4483d48`, image
`sha256:9dc0ad620110a347b7bbd5a050ad84a6d2453e16165abc15aee47550ff8e017b`.
Worker image `sha256:3fe28a5b7eb5716a71e984908520c31fff9ac325a9c5885c5e102e93996ac14a`,
StartedAt `2026-09-29T21:03:49.935343028Z`.
v3 image `sha256:5a0b1d63e6c5eb13d14af84590dafebf900f59156613b987000ae8b6938f563b`,
StartedAt `2026-09-07T17:35:02.146108577Z`.
Три RPC версии БД фактически прочитаны: contour_workflow/server_history/
model_quality **1/1/1**. Новых SQL/migrations нет.

Кандидат v4 собран из отдельного detached checkout
`/home/arborscan/as16-integration-0437e98`, exact commit `0437e98`, child
прежнего runtime с прежними dependencies/models. Все API-source SHA сверены
с Git, сборка `--network none --pull=false`. Image:
`sha256:034a9d67b49adb5565a1af9454706a104721e21160d725db32a31f796ff20193`.
[v4-candidate-image.json](evidence/as16-integration/v4-candidate-image.json).

Проверка выявила настоящий AS-14 остаток: timer active/waiting, но service
06.10 завершился exit1 по защитному лимиту **14 COMPLETE**. Все 14 outer
manifests проверены; latest 05.10 дополнительно 4 PostgreSQL и 2240 embedded
application entries проверены. Ровно старая 26.09 копия (19 authoritative
файлов, 927004164 bytes) уже независимо проверена в `D:\ArborScanBackups`;
атомарно перенесена под существующим lock в `/home/arborscan/ops-archive`,
все SHA сохранены, ни одного файла не удалено. После release lock запущена
штатная процедура новой копии. Последующий prepare выявил ссылку живых
v4/worker на `reliability.yml` в этом каталоге. Весь 26.09 каталог возвращён
под lock, все SHA и файл восстановлены, сервисы не перезапускались. Вместо
него архивирован `20260927T032112Z`: все 22 SHA и независимая D-копия проверены,
проверены 29 label/mount путей всех работающих контейнеров — зависимостей нет.
Ни одной копии не удалено. Guard/timer/service не изменены. Политика
дальнейшего архивирования с исключением активных runtime dependencies и
самостоятельный очередной запуск service относятся к следующему AS-14;
ручная копия не является успешным systemd-run.

Свежая **`20261006T131325Z`** создана обычным установленным `ops_backup.sh`,
exit0: **30 outer SHA, 4 native PostgreSQL SHA, 2261 embedded SHA**; PostgreSQL
17.6/custom dump 52 102 608 B. Реальный offline file restore: 2261 файлов,
16 report links; **новый SQL restore не выполнялся**. Свежая копия сама не
содержала `reliability.yml` из-за первого архивирования. Этот пробел явно
компенсирован перед deploy отдельным **проверенным current runtime/config
snapshot**: 10 Compose source-файлов, env, inspect и exact old image archive
(13 private files, mode600), включая восстановленный reliability.yml с SHA
`8219012d566fef3328aef3cde1a75842cd8927060dca1f89680af3284596f701`.
Это не объявление fresh backup самодостаточным full-config rollback.
[BACKUP_RETENTION.md](evidence/as16-integration/BACKUP_RETENTION.md),
[fresh-deploy-backup.json](evidence/as16-integration/fresh-deploy-backup.json).

**Фактически обновлён только API v4**: exact API commit `0437e98`, image
`034a9d…`, StartedAt **2026-10-06T13:35:21.059827568Z**, healthy.
Guarded helper — commit `8ef27b8`, private plan
`/home/arborscan/as16-v4-20261006T132500Z`. Все environment values, действующая
погода, model SHA, nginx configuration SHA, v3 и worker identity/StartedAt
остались прежними. `/opt/arborscan` не изменён. Обрыв SSH потерял итоговый
stdout (transport exit1), но private operation marker записал healthy;
последующее независимое чтение состояния, invariants и HTTPS подтвердило
успех. Повторный deploy не выполнялся.

После обновления: **оба HTTPS health200**, no-auth401, обычному автору
moderation403, чужому владельцу404; **41 real assertions** на DEMO проверили
старый PNG/точное состояние редактора/новую ревизию/retry409 conflict/все
перечисленные контуры200/frozen высоты4→8 м/неизменённые решения и исходные
измерения. Initial verification harness исправлен под существующий `record`
wrapper и `actor_id`/`at`, production contract для этого не менялся.
Startup logs: 0 traceback/ERROR/FATAL. DB RPC повторно **1/1/1**.
[live-api-after-deploy.json](evidence/as16-integration/live-api-after-deploy.json),
[vps-state-after.json](evidence/as16-integration/vps-state-after.json),
[v4-deployment-result.json](evidence/as16-integration/v4-deployment-result.json)
(полный manifest неизменённых model/config SHA).

Подготовленный **откат приложения** (не выполнялся, schema/data не восстанавливать):

```sh
ssh arborscan@31.57.170.88 \
  'python3 /home/arborscan/as16-integration-v4-rollout.py rollback --directory /home/arborscan/as16-v4-20261006T132500Z'
```

Оператор возвращает exact9dc image и полный captured environment, включая
погоду. При отсутствии image cache загружает verified private archive
608738816 B/SHA `99b54a5999bc332ff0fd710dde7390c25341b60db43bd016f1c70930f92f0299`.
Старый API читает новые записи: схема/форматы/revision endpoints не менялись.
После rollback проверить оба health и no-auth401; известный старый listing
дефект вернётся. Private plan/snapshot/active Compose sources сохранять.

APK установить без удаления/очистки данных (при доступном авторизованном S24):

```powershell
$adb = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe"
& $adb devices -l
& $adb -s SERIAL install -r arborscan_app/build/app/outputs/flutter-apk/app-debug.apk
& $adb -s SERIAL shell am start -n com.example.arborscan_app/.MainActivity
```

Для клиентского rollback собрать прежний `ecb1df11b4ea4a1d0d8db066b1f94db8b7ff19f2`
в **отдельном checkout**, с прежним debug key и versionCode выше фактически
установленного (например16 после15), затем `install -r`. Не удалять приложение
и не обходить signature/version check; последующая новая версия тоже требует
более высокого code. Старый +12 APK/SHA сохранён в оригинальном D: checkout;
его установка поверх +15 без повышения версии здесь не выполнена.

## Остаток и следующий пакет

Текущий пакет не даёт нового научного/полевого подтверждения. S24 установка
и реальные AR tracking/точность недоступны; AVD, API и unit-тесты разделены.
Визуальное одобрение AS-15 пользователя не заявлено. Пористость, полноценный
профиль корней и научные методики не заменены придуманными значениями.

Следующий самостоятельный пакет по согласованной очереди: **AS-14 доступные
эксплуатационные остатки (архивирование ежедневных копий/проверка service),
AS-16 постоянная подпись и безопасный перенос данных**. Затем release-сборка
и самостоятельная доставка функционального выпуска. AS-05…07 реальные
кандидаты/качество только при достаточных данных и разрешении на обучение;
AS-11 источники/методика; AS-10/AS-13 реальная экспериментальная валидация
при появлении оборудования/данных. Ни один следующий этап автоматически
не запущен текущей интеграцией.
