# AS-14 / AS-16 — выбранный технический выпуск, 08.10.2026

Работа от main `3c2838b99f8445f83f98f7b3c17f1390160d3b4a`, ветка
`codex/technical-release-completion`. Канонический план1.26 сохранён; приложенная
чат-копия1.9 не заменяла более новый реестр. Независимый постоянный получатель и
новый хост/public HTTPS отложены по явному выбору пользователя. Сейчас используется
только существующая `D:\ArborScanBackups`; свежесть зависит от включённого Windows.
Визуал AS-15 ожидает отдельного полного промпта. Научные критерии не закрываются.

## Статус действий

APK23 готов и заново проверен. Backup tools установлены и проверены.
После выполненного пользователем запуска штатной systemd-службы новая копия
`20261008T141824Z` опубликована и полностью проверена; service exit0,
14:34:00–14:51:27UTC. Fresh replacement доставлен в существующую Windows-папку,
native/runtime/SHA проверены. Ротация действительно выполнена на обоих концах.
Технический выпуск **состоялся**: main и рабочая ветка реально fast-forward/
push на3e0e1e615c5181a12b8743c15fd24a8a750624c8; annotatedv1.3.0
object10b6b41418fea8479deffd20a0aed830e5803420 и peeledcommit3e0e1e6
подтверждены в origin. Этот postrelease отчёт сохраняет тег неизменным.
Большая разрешённая уборка выполнена; **цель одного checkout ещё не завершена**:
4worktree защищены закреплением текущего чата/workspace в Codex, штатный
archive_worktree отказал. Запрошен только конкретный шаг снятия закрепления;
Git/remove-force и другие обходы не выполнялись.

## Измеренное место до уборки

Реальный Windows NTFS, только именованные материалы ArborScan. AllocationSize
из FILE_STANDARD_INFO, без повторного суммирования hardlink/shared junction
targets; sparse logical length не назван занятым местом. Ошибок0, неизвестной
allocation0. Метаданные NTFS/не относящиеся к проекту файлы не учитываются.
[Полный исходный замер](evidence/technical-release-completion/disk-before.json).
Здесь GB — десятичные, GiB отдельно; размеры каталогов не суммируются с их частями.

| Корень / назначение | Логические GB | Занятые GB | Хранение |
|---|---:|---:|---|
| D:\ArborScanBackups |32.438|32.450|Единственная папка backup,14 verified policy + защищённые/служебные материалы |
| D:\arborscan_backend |66.928|60.406|Финальный main, история, исходники, модели, APK, компактные доказательства |
| Четыре C worktree |1.820|1.843|После подтверждённого release удалить штатно с сохранением unique ignored evidence |
| Четыре исторических D source-fragment папки |0.000216|0.000250|11 недостижимых Git blobs сохранены отдельно; удалить после release |
| D:\ArborScanAvds |22.418|9.448|Собственные завершённые тестовые AVD; разрешённая уборка после release |
| D:\ArborScanSdk |12.401|12.402|Общий SDK, сохранить |
| D:\ArborScanSigning |0.116|0.116|Ключи/lineage/escrow/config, сохранить полностью |
| **Все именованные корни без повторов** |**136.121**|**116.665**|**108.66GiB занятых данных** |

Главные части финального проекта: output/final-release21.880GB allocation
(включая AVD/build/cache), старый output/as12-avd11.192GB, Git12.177GB,
прежняя arborscan_app_backup_working2.742GB, output/as16-integration2.827GB,
output/release-readiness2.330GB. Модели проекта96,974,111logicalB/96,980,992allocatedB.
Git, общий SDK/ключи и current models сохраняются; ~100GB не являются размером backup.
Windows Docker VHDX в проверенных штатных путях отсутствует; объём чужих дисков
не приписывался ArborScan. VPS Docker layers отдельно shared, не прибавляются к Windows.
Перед работой свободно C112,328,704B/D8,599,552,000B; дальнейшая фактическая
экономия измеряется после действий, а не считается по dry-run.

## Backup: конкретные исправления и проверки

Новые snapshots сохраняют аддитивный format1 `services` для v3/v4/worker.
Старые format1 без поля читаются в исторической области v4/worker; их нельзя
выдать за копию нового auth image. Runtime index дополнен exact v3archive
`82f59d586d20b1e232175a53c0260ce9a68b5fc88da5f9f7d931d2096b1594a9`,
787,500,544B из уже существующего приватного Google-job: второй большой tar не создавался.
Installed policy preimage сохранён privately, source SHA в
[install proof](evidence/technical-release-completion/backup-tools-install.json).
Live API/Auth/БД/models/worker/HTTPS не изменялись.

Windows pull дополнен передачей unique runtime assets в общий SHA-каталог той же
папки и штатным RELOCATED_RUNTIME map. Набор плюс shared assets — пакет;
timestamp directory отдельно не самостоятельный full DR. SHA ошибки/прерывание
не публикуют mapping и не разрешают ротацию; partial retry сохраняется.
Старые `./file` манифесты проверяются без изменения bytes, дубли канонических
путей/traversal отклоняются. Windows junction/reparse root/parents запрещены.
Ротация14 проверяет current native/all3 replacement, references и pins,
по каждому удалению повторяет проверку; shared archives не удаляются.
Исторические automatic sets принимаются только с наблюдённым run evidence и
SHA фактического manifest; manual/unadopted/live dependencies защищены.
Установленные Windows scripts/Task находятся в D:\ArborScanBackups\tools,
не зависят от удаляемого C-worktree.

Новые реальные программные проверки: Windows82tests/80passed/2Linux-only skipped;
после последних совместимости/dry-run правок21targetedpassed (включая2NTFSjunction).
Linux финальный suite85tests/83passed/2Windows-only skipped,62.708s.
Fixtures: missing/wrong SHA, runtime inventory/all3, native marker, partial retry,
low disk/no predelete, pins/manual/adoption/changed replacement/dependencies,
bounded14rotation/idempotency, locks, canonical/traversal and dry-run without deletion.
Это tests scripts; они не заменяют actual systemd/transfer/restore receipts.

[Фактический backup receipt](evidence/technical-release-completion/backup-final.json):
VPS service Result=success/ExecMainStatus=0, операция
`release-after-google-20261008T141530Z` возобновила тот же комплект после
обрыва SSH на первом manual attempt. COMPLETE опубликован14:45:06UTC;
integrity/runtime/retention завершены14:51:27UTC, active marker отсутствует.
Manifest SHA `ff7c4aee721a501cfb20997c4c100038b09bdd401ef9c88dbf79944d30c5abfa`.
Журнал доступных владельцу service records содержит PostgreSQL17.6/TLS,
Snapshot stages succeeded и completed_verified. System-wide записи ограничены
правами journal; итог самой службы проверен через systemctl show.

Windows14:20–14:26UTC exit0 сначала доставил прежний20261008T032219Z;
свежий all3 replacement20261008T141824Z доставлен14:54UTC штатным installed
pull machinery с read-only inventory одного опубликованного комплекта и
`--retention-dry-run`; после инспекции proposal выполнен штатный `--apply`.
VPS реально удалил20261001T032150Z; Windows первый apply реально удалил
20260929T031924Z,20260930T031949Z,20261001T032150Z,20261002T031752Z.
Дополнительный actual installed Scheduled Task15:04:23–15:11:44UTC/exit0
проверил inventory14/full transfer/runtime, доставил ещё доступный старый
server set20261003T031731Z и штатно ротировал его; итог14. Таким образом,
Windows удалены5 разных старых automatic sets.
[Actual Scheduler receipt](evidence/technical-release-completion/windows-task.json).
На каждой стороне осталось14 пригодных комплектов, fresh release PINNED;
manual/непринятые historical/live dependencies и shared runtime сохранены.
Новая Windows-копия проверяет все3сервиса и4unique runtime archives;
SHA/manifest/native/RELOCATED_RUNTIME references проверены фактически.

Целевое восстановление fresh копии в **существующих** network=none labs
завершено14:49:04UTC/154.55s: v3archive загружен, exactimage60c6cf1f и
оба auth source SHA совпали; native PostgreSQL восстановлен с ownership/ACL,
synthetic workflow/row checks и users_google_sub_unique(unique/valid/ready).
Production credentials не монтировались, production database не изменялась.
Оба лабораторных контейнера остановлены; новый VM/full DR stack не создавался.
[Целевое restore доказательство](evidence/technical-release-completion/targeted-restore.json).

Прежний calendar08.10VPS03:22–03:33UTC exit0 остаётся отдельным свидетельством.
После изменения состава actual systemd service выше проверен; future all3
calendar run09.10 пока не наблюдался. Timeractive/waiting, фактический next run
в receipt. Windowscalendar08.10/11:27UTCexit1(SSH) сохранён; текущий installed
Scheduled Task реально прошёл отдельным запуском exit0, не выдаётся за будущий calendar.

## Клиент, production и границы доказательств

[Артефакт / установка / откат](TECHNICAL_RELEASE_ARTIFACT.md): actual APK
1.3.0/code23/sourcebbcd5b5/SHA71f38189…; nondebuggable, package/min24/target36,
подпись API24–32 legacy /33+permanent, lineage/capabilities/alignment16KiB.
Все270trackedclientfiles одинаковы source23/currentmain, key entries доступны;
подготовленный private signing helper использует финальный D-путь.
Recovery24: 9 Android unit tests действительно повторно выполнены (не up-to-date),0fail/0error,39s; UI/install24 отдельно не заявляются.

Отдельный recovery24 реально построен/подписан, SHA35daaa3a…; 688 payload
entries совпали с23, изменён только manifest. Подписи24–36/lineage/alignment
проверены. Recovery24 на S24 не устанавливался,23→24 перенос не проверен;
готовый APK23 не пересобран/переустановлен, данные/аккаунт S24 не трогались.

Обоснованно переиспользованы181Flutter/9Android/analyze0errors/53old/0new,
реальные Android24/33/36 update/cold, S24update62/62SHA и UI/SAF/sharecancel,
Google24/36 после security rollout, DEMO revision/history/offline/reconnect,
15просмотренныхстраниц реальныхPDF. Ссылки на exact version proofs в artifact doc.
Они не являются новым полным прогоном, полевой точностью или одобрением визуала.

Новая read-only проверка14:08UTC: обаHTTPShealth200, me/corrections/reports
безbearer401, missing/malformed Google422/401, обычный TLS безbypass.
v3policy6284f93/image60c6cf1f, v4code0437e98/image034a9d67,
workereaa39f4/image3fe28a5b; actual полный IDs/SHA в
[runtime proof](evidence/technical-release-completion/runtime-before-release.json).
Googleindex unique/valid/ready, существующиеSQLверсии1/1/1. Model_v4
SHA626f62b97b05fd65275dc3f909c3aec090f6c9957c4ef9f12c4e743a1f75a2bb,
active model4/retrainfalse. VPSGit5d8176a clean. Это разные identities;
серверы/APK не объявляются собранными из будущего release-doc commit.
Backup timeractive/waiting/09Oct; next run в backup receipt. Live containers остались прежними.

Безопасный Google rollback — corrected image/claims=false/index сохранён;
private overrides/точныеargv в [GOOGLE_SECURITY_ROLLOUT.md](GOOGLE_SECURITY_ROLLOUT.md).
Старый handler возвращает уязвимость. Rollback prepared/config-validated,
production не откатывался. Новая база восстанавливается только изолированно,
не поверх живой Supabase. Полный новый хост/public HTTPS не проверены.

## Сохранность и уборка

323unique evidence +11historicalsourcefragments +5изменённыхPDF реально
скопированы в остающийся приватный D-каталог;339файлов/25,235,582B,
каждый SHA совпал с оригиналом. Originals/user PDF не изменялись.
ManifestSHA2078f9d0b6fa4d3e13ce79c74058a7c12c75be1e1a7765863d84e30d8ef115b4.
Одна основная D-копия уже main3c2838b без reset;5userPDFdiff сохранены отдельно
от release-source. Нужные исходники/научные материалы/Git/signing/backup остаются.
[Предварительный аудит](evidence/technical-release-completion/cleanup-audit.json)
не является удалением. [Фактический receipt](evidence/technical-release-completion/cleanup-result.json)
отделён от этого предварительного аудита.

Pre-release удаление D:\arborscan_backend\arborscan_app\build\app\intermediates
и build\test_cache отклонено автоматической проверкой; эти два пути не удалены,
не повторяются и не обходятся удалением родительского каталога.
После подтверждённого тега фактически удалены20именованных каталогов:
4старых D source-copy,3завершённых AVD, as12-avd,2старых integration build,
readiness/flutter-build,8final-release build/cache/temp и obsolete
arborscan_app_backup_working. Его47trackedsourcefiles удалены через git rm;
историческое дерево5da6f973… сохранено в опубликованном Git/tag, актуальный
arborscan_app не изменён. 12старых/duplicate APK удалены по точным путям;
единственный пользовательский APK23 и private signed recovery24 сохранены.
Оригинальный S24code12/private phone backup сохраняется как пользовательская
страховочная копия, не рабочий проект. Каждый удалённый путь/размер/SHA APK
в cleanup-result; dry-run не выдан за действие.

После SHA-проверки preservation339files отсоединены46reparse entries C-worktree
**только как ссылки**, их targets не обходились. Все4archive_worktree получили
точный отказ «This worktree is protected by a pinned task or workspace».
Остались C:/Users/danik/.codex/worktrees/{release-integration,release-readiness,
final-release,google-security-rollout}/arborscan_backend (1.843GBallocation).
Их source не удалён и не объявлен архивированным; текущий чат закреплён,
запрошено снятие закрепления. Основная D-папка — main, пользовательские5PDF
побайтно сохранены/не staged. Собственные release changes отдельно committed.

На VPS реально удалены5мелких законченных test-source dirs после сохранения
184972Bprivate archive/проверкиSHA/исключенияcontainer mounts/Compose/unit/runtime
refs. 4проверенных candidate directories пока связаны mount/config существующих
изоляционных recovery/test контейнеров и сохранены как служебные зависимости.
Live v3/v4/worker/HTTPS/Storage/БД/models/ops-tools/runtime/rollback paths сохранены;
массового Docker prune/production restart нет. Список и конкретные resource refs
в исходном dependency audit; productionIDs остались прежними.

Фактический после-замер Windows:0ошибок/0unknown allocation,
[disk-after](evidence/technical-release-completion/disk-after.json).

| Показатель | До, байт | После, байт | Изменение |
|---|---:|---:|---:|
| Именованные данные NTFS allocation |116664936864|65969588656|−50695348208 |
| Свободно D |8599552000|59317977088|+50718425088 |
| Свободно C |112328704|78688256|-33640448 |

Это net measurements с учётом новых backup/recovery/proofs и текущих внешних
записей, не сумма logical sizes удалённых sparse AVD. Реально D +50.718GB
(47.235GiB); оставшиеся2blocked caches и4pinnedcheckout отдельно.
Signing --verify-release повторно прошёл после удаления cache: APK23 unchanged,
keys/lineage/SDK/privateD-helper доступны. Backup/WindowsTask uses permanentDtools.
[Final runtime check](evidence/technical-release-completion/runtime-after-cleanup.json):
обаHTTPS200, unauthorized401, обычныйTLS, currentimages/IDs/index/SQL/model неизменны.
S24данные не затронуты, новыйphoneUI не заявлен; прежнийexact23proof применим.

## Публикация

08.10.2026 main/working branch реально отправлены на releasecommit
`3e0e1e615c5181a12b8743c15fd24a8a750624c8`. Annotated `v1.3.0` object
`10b6b41418fea8479deffd20a0aed830e5803420`, peeled commit совпадает;
[origin receipt](evidence/technical-release-completion/release-publication.json).
Protection/rules API200/protectedfalse/rules0/rulesets0 проверены до FF/push;
reset/force/передвижениятега нет. Последующий cleanup/report commit отправляется
в main/ту же originрабочуюветку, оставляя этот тег на releasecommit. Магазин и
публичный GitHub Release не создавались. Это технический выпуск существующих
функций, не пользовательская визуальная приёмка/экспериментальная валидация.

Остаток текущего поручения — штатно архивировать4pinned C-worktree после снятия
закрепления и обновить замер/receipt. Следующий продуктовый этап — отдельный
полный промпт AS-15, автоматически не начинать.
