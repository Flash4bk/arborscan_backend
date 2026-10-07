# AS-14 / AS-16 — готовность резервных копий и подписанного клиента

Пакет начат от `origin/main` **946a632** в отдельном managed worktree
`codex/release-readiness`. Исходный `D:\arborscan_backend` с пользовательскими
PDF, кешами и веткой beta-dynamics сохранён. Этот документ фиксирует технические
проверки агента; пользовательская и экспериментальная приёмка не назначаются.
Окончательный release/tag и публикация в магазине не выполнялись.

## Сервер и ежедневные копии

Приложение VPS осталось на API v4 **0437e98**, image
`sha256:034a9d67b49adb5565a1af9454706a104721e21160d725db32a31f796ff20193`.
V3, worker, HTTPS, модели, SQL-схема и пользовательские записи не менялись.
Установлены только backup tools; сервисы приложения не перезапускались.
`/opt/arborscan` сохранён на **5d8176a**. Перед заменой инструменты сохранены в
`/home/arborscan/ops-tools-readiness-before-20261006T152944Z`, reader отдельно в
`/home/arborscan/ops-snapshot-before-20261006T202156Z`.

Причина прежней остановки — guard, который отказывал при 14 COMPLETE до создания
следующей копии. Новая политика: lock → private staging → полные native
PostgreSQL/Storage/config/runtime SHA и файловое восстановление → атомарная
публикация COMPLETE → удаление только старейшего проверенного автоматического
набора до лимита 14. Ручные, закреплённые, действующие Compose и зависимости
образов защищены. До удаления доступен dry-run. Повтор прерванной операции
продолжает её; попытки не считаются отдельными успешными наборами. При недостатке
места процедура отказывает без предварительного удаления. Внешний отказ не
удаляет единственную пригодную копию. [Контракт и команды](BACKUP_POLICY.md).

Первый реальный запуск systemd 06.10 19:52:58–20:03:15 UTC завершился exit 1:
PostgREST `analyses`, limit 500, HTTP 500 `statement timeout`. Все 2247 Storage
объекта успели выгрузиться; COMPLETE/ротации не было, 14 старых копий сохранились.
Независимый диагностический HTTP 429 Storage был отдельной ошибкой нагрузки,
не причиной исходного отказа. Reader исправлен: страницы 100 → 25 → 5 → 1
уменьшаются только при явном SQL timeout, с PK order, точным count и запретом
повторов/пропусков. Все 108 полных строк прочитаны дважды, SHA совпал; отдельный
PK inventory подтвердил полноту. Прежние три bounded transient retry сохранены.
Логи содержат только allow-listed stage/status/reason, без URL/данных/секретов.

**Автоматический календарный запуск 07.10 03:21:52–03:34:25 UTC прошёл**, 753 с,
systemd success/exit 0. Это подтверждено journal и LastTrigger таймера, а не
выведено из enabled/active. Опубликован `20261006T195518Z`: имя унаследовано от
прерванной операции daily-2026-10-06, actual snapshot выполнен 07.10/attempt 2.
Manifest SHA:
`d850deae43556e2dcb69980cd3619b7013b428a66a7d5860a7310657fae17eb2`.
После проверки новой копии удалён допустимый автоматический `20260929T031924Z`;
осталось 14 COMPLETE. `20260926T202023Z/reliability.yml` с действующей
конфигурацией сохранён. Повторная команда sudo больше не нужна для доказательства
штатного service-run. Агент отдельно перепроверил все **29 outer SHA / 979545362
байта**, четыре native PostgreSQL файла, 2247 Storage объектов, 15 таблиц и 12
фактических Compose sources. Восстановлены 2265 файлов и проверены 18 оригиналов
отчётов / 11 parent links; это file-only restore, без записи в production БД.
[Фактическая служба и состав](evidence/release-readiness/service-result.json).

Независимый review затем выявил отдельный сбой durable metadata при прерывании
после публикации/ротации. Final policy2aa70c5… установлен07.10 06:52:03UTC под lock
с private preimage; повтор `daily-2026-10-06` прошёл **485,47с/verified_existing**,
0create/0delete,14→14, все архивы/manifest/COMPLETE/staging неизменны.
Первоначальные completed_at03:29:58 и verified_at03:34:25 сохранены;
last_integrity_check07:01:09 записан отдельно. Production fault не имитировался.
Ночной systemd-run был на предшествующей policy, narrow final branch проверен
реальным replay, не выдан за новый полный systemd backup.
[Установка и regression](evidence/release-readiness/backup-policy-idempotency-fix.json),
[фактический replay](evidence/release-readiness/backup-policy-existing-replay.json).

Новый runtime-контракт включает SHA-covered `RUNTIME_DEPENDENCIES.json` с
точными OCI image IDs и приватными путями архивов. Релокационная карта внешней
копии не переписывает оригинальный manifest. Точные full archives current v4 и
worker сохранены отдельно от неизменённого base, их полный SHA проверен.
Новый SQL restore не выполняется: формат native dump не менялся, предыдущие
изолированные восстановления сохраняют доказательства. Проверяется затронутая
часть — текущие runtime archives в пустом Docker engine без production сокетов,
сети, конфигурации или запуска HTTP/обучения. **Реальный restore прошёл за184,33с**:
исходно ноль images/containers, Docker29.8, обе точные OCI image IDs восстановлены,
Python и SHA исходников внутри обоих образов проверены. OCI index/manifest ID
отделён от config digest. Lab остановлен. Предыдущая незавершённая попытка06.10
не засчитана; неизменённый six-image base повторно не разворачивался.
[Результат](evidence/release-readiness/runtime-restore-result.json).

## Внешняя копия

Windows-задача `ArborScan Offsite Backup` сохраняет закрытую копию на
`D:\ArborScanBackups`, ежедневно 08:00 Minsk и при входе. Реальные ручные запуски
06.10 успешны: 795,010 с с одним успешным resume и 160,104 с для повторной
проверки всех 14 наборов. Подтверждён Windows-дефект CRLF manifests; исправлен
с red→green тестом и побайтным совпадением SHA с VPS. File restore: 2261 файл,
16 report links, 3,857 с; это не новый SQL restore.

Календарный Windows-run 07.10 05:00 UTC завершился `ssh_inventory_failed`/exit 1;
точная причина его SSH-отказа неизвестна: прежний stderr не сохранён. Добавлены
ограниченные повторы inventory с безопасной диагностикой, строгая проверка SSH
сохранена. Два последующих ручных запуска прервались при connection closed/reset;
они не получили COMPLETE и сохранили пригодные старые копии.

Фактический ручной запуск существующей Scheduler-задачи **07.10 09:53:32–09:57:14
UTC прошёл,222,485с/LastTaskResult=0**:13 ранее проверенных наборов и новый
`20261006T195518Z`; все14 текущих серверных наборов проверены,16 исторических
COMPLETE на Windows сохранены. Новый application.tar получен прежним block reuse:
из проверенного старого архива переиспользованы только payload с совпадающим SHA,
недостающие байты прочитаны с VPS; результат830341120байт побайтно совпал с trusted
source SHA. Это не полная повторная передача830MB по сети.

После передачи проверены **29 outer SHA**,4 native PostgreSQL файла и3 полных
runtime архива. Отдельный закрытый relocation package содержит **34файла /
4760932219байт**, все скопированы и прочитаны обратно по SHA; исходные manifests
не переписывались. Настоящий file-only restore2265файлов/18report-links прошёл
за6,354с; общая проверка38,789с, завершение09:59:12UTC. Нового SQL restore не было.
[Фактическая Windows-копия](evidence/release-readiness/windows-offsite-fresh-20261007.json).
Следующий календарный Windows-run08.10 05:00UTC ещё не наблюдался; ручной success
не заменяет его. При выключенном ПК новый перенос не выполняется.

Проверены sender/receiver с private SSH, строгим known_hosts, resumable stage,
immutable manifest, readback всех байтов и SHA, отказами/повтором/конкуренцией.
Это isolated CLI/RPC tests, **не фактический независимый SSH receiver**.
Автоматический always-on перенос не включён: подходящий отдельный хост не
назначен. Требуется один разрешённый независимый SSH/Python3 receiver, закрытый
каталог, dedicated key и проверенный known_hosts в приватном конфиге VPS.
Пароли/ключи не присылать в чат. [Настройка и фактические проверки](OFFSITE_READINESS.md).

Windows независим от VPS, но зависит от включённого ПК. AS-14 целиком не закрыт:
always-on receiver и новая публичная VM/HTTPS остаются ограничениями. Допустимую
потерю данных и RPO/RTO от имени владельца не назначаем.

## Подпись и данные

ApplicationId `com.example.arborscan_app`, minSdk 24, targetSdk 36 сохранены.
Gradle release unsigned и non-debuggable; окончательный артефакт создаёт
`ops_android_signing.py` после zipalign и фактической проверки подписей/manifest.

| Платформа | Действующий сертификат | Проверка |
|---|---|---|
| API 24–27 | Прежний debug, v2 | APK verify; реальная установка на API 24 |
| API 28–32 | Прежний debug, v3 | APK verify по диапазону; это не ротация на новый ключ |
| API 33+ | Новый постоянный, v3.1 + lineage | Реальные обновления API 33 и 36 |

Публичный SHA-256 нового сертификата:
`7fb94ede4099199db9166609c34d6278ebe51c7d80bd6ce1cd0208bf863863b8`.
SHA-1: `e4dfd76dc3ec5f96be6b740949c678aebf1f110e`.
Прежний SHA-256:
`68ff9861f52fdb22744b30ee27092008097f9eff4ead7eb38f80e662115f5b97`;
SHA-1: `2c08b4d637574c16be5cba1a595a4baac8541a4c`.
Lineage old → new сохраняет installed-data, old rollback запрещён. Старые
Android сохраняют прежний signer; APK не называется окончательным store-release.
[Официальные источники, команды и границы](ANDROID_SIGNING.md).

Приватные ключи/lineage/DPAPI находятся вне Git в `D:\ArborScanSigning` с закрытым
ACL. AES-GCM escrow независимо от ПК сохранён на существующем VPS в
`/home/arborscan/signing-private`, mode 700/600; recovery secret отдельным файлом.
Это отдельный хост для ключей, но не независимый от VPS data-backup receiver;
компрометация доступа к обоим серверным файлам раскрывает escrow. Материал
скачан назад по проверенному SSH, восстановлен в новом приватном каталоге и
побайтно сопоставлен. Восстановленным ключом реально подписан/проверен APK с
тем же новым отпечатком и lineage. Формат и автономная процедура восстановления
после потери ПК — `ops_signing_key_recovery.py`; plaintext не пишется до AEAD/
archive validation, существующие материалы не заменяются.

Окончательный кандидат **1.3.0-readiness.1cf6ce3+18**, SHA-256
`de4b262e6983b1d6941383f97e1c30bd4e1388c9d3e23c7714e784095f398033`:
release без debuggable/с INTERNET, min24/target36, все signature ranges и 16KiB
alignment проверены,115577827байт. App/source точка `1cf6ce3` включает TLS fix
`eed7963`; последующие ops/docs коммиты не меняют APK. Промежуточный+17 SHA:
`08eba2754ea14febab83235fc66752abd88e13c14d47f131fb4638df2276ae5b`; +16 SHA:
`66a1d8aa37af9ba598923fa245b4bb83c1df6514d6ae677d6d6324fcac0183fa`.
APK вне Git: `D:\arborscan_backend\output\release-readiness`.

На настоящих AVD Android 7/API24, Android13/API33 и Android16/API36 пройдена
цепочка +15 → +16 → +17 без uninstall/clear: 70/70 файлов первого перехода и
71/71 второго совпали по SHA. Фактический signer считан PackageManager, а не
имя APK. Последующее+17→+18 сохранило71/71файлов API24,73/73 API33 и71/71 API36
до первого UI launch; probeV3 проверяет также databases. API24 остаётся на прежнем
сертификате, API33/36 — на постоянном.

Отдельный чистый AVD API24 со старым debug15 воспроизвёл прежний
`CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate`; это не
следствие смены подписи или просроченного токена. В+18 добавлен только официальный
public ISRG Root X1 в существующий Dart SecurityContext до network initialization,
с проверкой DER SHA25696bcec… . Платформенные CA, hostname/date/chain verification
сохранены; bypass/callback/HttpOverrides нет. Никаких изменений VPS/renewal.
На реальном API24 UI-вход, серверная история/отчёт и force-stop→reopen прошли.
[Public root/первичные источники](../arborscan_app/assets/certificates/README.md).

На API36 release17 самостоятельно проверены: account confirmation; 200см/
шесть исходных отрезков; исходно пустой unsaved draft (пустота подтверждена
baseline, не потеря обновления); contour draft/editor; legacy PNG без придуманных
точек; accepted/rejected status/reason; обе серверные версии; force-stop/offline
cached report/PDF; восстановление сети; DEMO logout → скрытая история → UI login
→ те же пять серверных отчётов. Google Maps реально показывает подложку под
новым сертификатом; cloud restrictions не изменялись. Google Sign-In/новая
Android OAuth pair пока отдельно не проверены, Bearer email/password UI прошёл.

Google Cloud должен сохранять прежнюю Android pair и разрешать новую
`com.example.arborscan_app` + SHA-1 `e4dfd76dc3ec5f96be6b740949c678aebf1f110e`
для используемых Android Maps restrictions/OAuth credentials. Наличие плиток
не подтверждает конкретную настройку консоли или весь Google OAuth flow.
Проверка TLS и авторизации не отключалась.

Три PDF реально сохранены через Android SAF, открыты viewer, Share sheet
вызван/отменён без отправки. Файлы выгружены, pypdf проверил значения, единицы,
координаты/разметку, историческую среду и размеры фото; JPEG отличие пикселей
ожидаемо и измерено. Все **15 страниц** отрендерены Poppler/визуально просмотрены.
[Контрольные PDF и hashes](evidence/release-readiness/pdf/checks.json),
[UI доказательства](evidence/release-readiness/avd36-ui-results.json).

На окончательном+18 дополнительно выгружены реальные PDF из DEMO AVD и S24:
ещё **10страниц** отрендерены и визуально проверены, values/units/photo/overlay
совпали с выбранными версиями. AVD PDFv2 по тексту отличается от+17 только
временем генерации; decoded photo/overlay pixels идентичны. S24 PDF подтверждает
4,8м/2,1м/эталон1м и историческую почву29,5%. Его исходник55774байт, SHA
`5288eead31f0c7e6364cc7db2c1e7b78865096e8a628ea2f1d615d7a5ef188b2`,
сохранён только в закрытом `D:\arborscan_backend\output\release-readiness`:
синтетическое происхождение GPS не доказано, поэтому координаты не публикуются.
Публичны только safe hash/checks и previews с закрытыми UUID/GPS.
[Проверки реальных PDF18](evidence/release-readiness/pdf18-file-checks.json).

S24 реально имел **1.3.0-geo.ecb1df1+12**, old debug/API36. Агент сверил подпись
выгруженного установленного APK, сохранил private full-data tar91,6MB вне Git
и установил **+18 поверх+12**. До первого запуска все **61/61файла пяти папок**
побайтно SHA-идентичны, включая preferences/secure storage/database; Android
PackageManager подтвердил новый сертификат и non-debuggable18. [Сравнение](evidence/release-readiness/s24-after18-comparison.json).
Самостоятельно открыты профиль с подтверждённой сервером сессией, история,
существующий отчёт/оригинал, список контуров и настоящая Google Maps подложка.
Аккаунт не менялся, реальные отчёты/решения не правились. После первого USB-обрыва
связь восстановилась: **force-stop→launch→история→существующий DEMO отчёт** на S24
подтверждены. Из этого тестового отчёта PDF реально сохранён в Download через
SAF, выгружен и открыт Google Drive local PdfViewerActivity. Системное Share меню
`com.android.intentresolver/.ChooserActivityLauncher` вызвано и отменено без
отправки или загрузки в Drive. Offline/reconnect остаются проверками отдельного
DEMO AVD, сеть пользовательского телефона не переключалась. Private screenshots
с данными/местоположением и chat-head avatar не
публикуются; сохранён безопасный [home screenshot](evidence/release-readiness/ui/s24-release18-home.png).
AVD не подтверждает AR tracking или полевую точность. Экспериментальные AS-10/11/13
и визуальная приёмка AS-15 не закрываются.

На API36+18 повторено force-stop с сетью off: кеш выбранной версии и настоящая
генерация/открытие локального PDF доступны, фото и8,1м/5,19м показаны viewer.
После возврата сети0/1/1 и повторного обновления список серверных версий восстановился.
Первый harness искал серверный timestamp вместо более раннего local cache timestamp;
асинхронный UI dump был снят до завершения перехода, а первые reconnect попытки
показывали кеш; их точная HTTP-причина не установлена. Эти отказы не засчитаны как
успех; окончательный viewer и обновлённая серверная история проверены отдельно.
[Финальный S24/AVD18 UI результат](evidence/release-readiness/final18-device-ui-results.json).

Локальные оригиналы/черновики/параметры/несинхронизированные операции сохраняются
при доказанном обновлении на том же устройстве. После logout файлы другого
аккаунта скрыты; при входе тем же владельцем доступны. Удаление приложения/
его данных и смена телефона не переносят локальные черновики автоматически.
Серверная история доступна после авторизации по существующим IDs; нового
создания серверных записей для переноса не требуется. PDF — самостоятельная
копия, не архив для восстановления токена или локального editor state.

## Проверки и откат

Затронутые проверки: backup policy final27 Windows/27 Linux; snapshot reader final
17 Windows/17 Linux, плюс реальные full-row reads; signing 24; offsite final
40 Windows (38 passed/2 Linux-only skipped), Linux receiver и latest LF отдельно;
targeted Flutter contour/history/reference/export+TLS **35 passed**, после lint-only
поправок TLS7 повторно passed. Android app tests5/0failures и audit test build
для последнего состояния прошли. Release18 build104с, native/probe57с.
Signing-key recovery23 Windows/23 Linux, actual restored-key signing отдельно;
inventory retry14 Windows/14 Linux, actual SSH inventory отдельно.
Analyze: exit1, 0errors, 5warnings+48info, прежние53 замечания, новых нет.
Полные research70/backend216 не повторялись: их код не менялся.
Финальный контроль10:25UTC: оба публичных HTTPShealth200/statusok,
unauthenticated reports/corrections401; три image IDs и StartedAt неизменны.
[Безопасный результат](evidence/release-readiness/final-public-health.json).

Промежуточные отказы не скрываются: build16 transient artifact DNS, конкурентно
пересозданный plugin registrant; после последовательного build успешно. Общий
Gradle plugin-unit sweep упёрся в Jetifier/byte-buddy Java68; относящийся к
приложению `:app:testReleaseUnitTest` прошёл. Ошибка harness с прежним количеством
Compose исправлена под полный multiset фактических трёх контейнеров; SQL migration
или новая DB restore этим не имитируются. Первая попытка18 build потеряла Gradle
daemon при двух AVD и нехватке памяти; после штатной остановки только собственных
AVD последовательная сборка104с успешна. API33 AOSP_ATD не имеет IME: automation
BACK закрывал Activity; исправлен test helper, это не app crash/потеря данных.

Клиентский откат — forward-fix с прежним постоянным ключом/lineage и **versionCode
больше установленного**, с сохранением совместимого формата. Старый debug APK
после ротации не является доказанным откатом. Удаление/понижение с потерей данных
не применяется. Из `arborscan_app` собрать unsigned с новым кодом, затем подписать
по [ANDROID_SIGNING.md](ANDROID_SIGNING.md), проверить APK и установить:

```powershell
& "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe" devices
& "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe" -s <выбранный-serial> install -r <проверенный-signed.apk>
& "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe" -s <выбранный-serial> shell am start -n com.example.arborscan_app/.MainActivity
```

Откат backup tools выполняется только при inactive service и под `backup.lock`,
из сохранённого preimage с проверкой SHA и атомарной заменой; API/БД не откатываются.
Возвращение старого guard снова остановит процедуру при14 и не восстанавливает
удалённый автоматический набор. Его Windows-копия/старые manifests сохраняются;
действующие runtime/ручные наборы не удаляются. Reader rollback вернёт SQL timeout
limit500, поэтому предпочтителен проверенный forward-fix reader.

Для возврата только последнего metadata-fix подготовлен
[точный rollback](evidence/release-readiness/rollback-policy-20261007.sh): проверяет
inactive service, lock, отсутствие активной операции, оба SHA и atomically
возвращает `/home/arborscan/ops-policy-before-20261007T065203Z/ops_backup_policy.py`.
Reader timeout-fix и архивы сохраняются. Проверен `bash -n`, сам откат **не выполнен**.
Команда из корня checkout при необходимости:

```powershell
Get-Content -Raw deploy-vps/evidence/release-readiness/rollback-policy-20261007.sh |
  ssh -o BatchMode=yes -o StrictHostKeyChecking=yes arborscan@31.57.170.88 bash -s
```

## Текущий остаток

Доступный пакет реализован и проверен агентом. AS-14 целиком остаётся открытым:
нужен разрешённый независимый always-on SSH/Python3 receiver с приватным каталогом,
dedicated key и проверенным known_hosts; новый host/HTTPS recovery не проверен.
Следующий плановый Windows-run после исправления ещё не наблюдался.

Для AS-16 остаются Android OAuth credentials новой пары applicationId/SHA-1
и реальный Google Sign-In flow; доступ к консоли не предоставлен. API24–32 сохраняют
старый debug signer — выбранный путь не является окончательным store-release для
этих платформ. Следующий пакет: закрыть эти технические ограничения, затем
окончательная версия, отдельно разрешённый release/tag и доставка APK. В этом
пакете публичного тега нет. Научная/визуальная приёмка не подменяется проверками.

## Фиксация кода

Коммиты проверенного пакета от946a632:

| Коммит | Содержание |
|---|---|
| f77faa4 | Verify/publish/rotate14 с защитой действующих наборов и зависимостей |
| 5ec4a40 | Постоянная подпись, lineage, совместимость и data preservation |
| 09b5041 | Приватный off-site sender/receiver, сохранение исходных LF manifests |
| 7dd55fd | Ограничение timeout-pages и фактическая календарная служба |
| eed7963 | Официальный ISRG Root X1 для старого Android без TLS bypass |
| 1cf6ce3 | Recovery escrow, проверка действующего сертификата и databases; точка APK18 |
| fedb79c | Идемпотентное восстановление durable completion metadata |
| 43f7f14 | Ограниченные inventory retry и полная свежая Windows-копия |
| 0d6a959 | Фактические Android/S24/PDF18 доказательства и подготовленный rollback |

APK, key material, конфигурация с секретами, private raw evidence и дампы не
включены в Git. Проверенная подготовка independent receiver сохраняет статус
«подготовлено», даже после разрешённой интеграции в main. Отдельный публичный
release/tag/store не создаётся. Факт публикации Git фиксируется после push.
