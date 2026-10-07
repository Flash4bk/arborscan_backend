# AS-14: независимая внешняя копия, 06.10.2026

Это дополнение текущего пакета `codex/release-readiness` от main `946a632`.
Прежние backup/restore результаты не заменяются новым отчётом. Общий план остаётся
`ARBORSCAN_MASTER_PLAN.md`; его обновляет основной агент после всех результатов.

## Фактический аудит до исправления ротации

Агент выполнил read-only SSH и проверку существующего Windows Scheduler:

| Проверка | Наблюдавшийся результат |
|---|---|
| VPS timer | enabled/active; LAST 06.10 03:20:22 UTC; NEXT 07.10 03:19:03 UTC |
| Последний daily-service | failed/exit-code, ExecMainStatus=1; guard «14 copies retained»; начало/конец 06.10 03:20:22 UTC |
| Исторические ежедневные запуски | Журнал 30.09–05.10 содержит успешные native/PostgreSQL и общие COMPLETE; это не успех следующего запуска |
| Наборы VPS | 14 COMPLETE; свежий ручной `20261006T131325Z` существует; таймер его не создавал |
| Диск | На момент аудита свободно 67 755 819 008 байт; /home и /opt на одном разделе |
| libpq | `.pg_service.conf` и `.pgpass` имеют mode 600; значения не выводились |
| sudo | `sudo -n true` возвращает 1; интерактивный шаг остаётся отдельным |
| Windows task | `ArborScan Offsite Backup`, Ready; LastResult=0, 06.10 10:21:57 Europe/Minsk; NEXT 07.10 08:00 |
| Windows журнал | 06.10 07:26:36 UTC: success после SHA-проверки 14 наборов; последний `20261005T032052Z` |
| Windows политика | Каждый день и при входе, StartWhenAvailable; IgnoreNew; три повтора через 15 минут. Требуется включённый ПК и вошедший пользователь |

Эти значения — снимок до текущего исправления; окончательные данные service и новой
копии находятся в отчёте основного пакета. Сам timer не доказывает создание копии.

## Отдельный реальный блокер

Из проверенных существующих конфигураций ArborScan не найден разрешённый независимый
круглосуточный receiver: отсутствуют rclone/AWS/offsite private-конфиги, а действующие
Supabase реквизиты относятся к тому же проекту. Использовать его как независимый
резервный хост нельзя. Windows-копия действительно независима от VPS, но зависит от ПК.
Просмотрены только имена переменных и наличие файлов; ключи, пароли, DSN, содержимое
env и пользовательские данные в доказательства не включались.

Минимальный недостающий шаг: указать уже разрешённый для ArborScan независимый SSH-хост
с Python 3 и закрытым каталогом достаточного объёма, настроить отдельный SSH-ключ и
проверенный known_hosts приватно на VPS. Пароль/ключ/полный приватный конфиг в чат
передавать не нужно. Новый аккаунт, платный сервис или публичный bucket не создавались.

Также sender намеренно требует SHA-проверенные runtime зависимости текущих образов.
На момент исходного аудита VPS `recovery-releases` содержит прежние image/source
архивы, но не опубликованные RELEASE_INDEX/release.json/COMPLETE. Windows имеет
проверенные прежние manifests. До фактического обновления index для текущего образа
новую копию нельзя объявлять полной для пустого хоста.
Текущий backup-policy добавляет покрытый outer SHA `RUNTIME_DEPENDENCIES.json`:
`format:1`, точные image ID и массивы `{path:absolute, sha256}`. Sender поддерживает
этот новый контракт напрямую; исторический RELEASE_INDEX также поддерживается.
При внешнем восстановлении исходный документ с absolute paths сохраняется побайтно;
отдельный `RELOCATED_RUNTIME.json` сопоставляет их проверенным полученным архивам.
Переписывать hash-covered документ адресами нового хоста нельзя.

## Подготовленный транспорт

`ops_offsite_push.py` и `ops_offsite_receiver.py` реализуют приватный SSH RPC:

- Explicit independent/always-on receiver, отказ same VPS/localhost и одинаковому
  machine-id; строгая проверка SSH host key, отдельный key, config/key/known_hosts mode 600.
- Исходный полный native/Storage/config snapshot и runtime/source/base/overlay проверяются
  по реальным SHA до передачи; наборы без native dump или зависимостей текущего образа отклоняются.
- `.staging`, отдельная блокировка каждого RPC, неизменяемый manifest, ограниченные
  чанки, проверка фактического префикса после прерывания. Повтор не создаёт вторую копию.
- Контентные объекты дедуплицируют большие immutable runtime архивы; повторы чтений
  ограничены. Потерянный ответ записи разрешается новым probe, без слепого повтора записи.
- COMPLETE на receiver появляется после полной проверки всех объектов. Локальный
  receipt появляется только после **чтения всех внешних байтов назад** и SHA на отправителе.
  Размер/ETag/успешное сообщение receiver недостаточны.
- Отказ/неполная передача не удаляет источник. Completed backups и shared image blobs
  этот транспорт вообще не удаляет. Retention receiver задаётся отдельной разрешённой
  политикой, иначе данные внешнего хоста будут накапливаться.
- `recover` получает полный пакет в новый закрытый каталог, проверяет native и runtime
  manifests/relocation map и извлекает application archive с SHA/связями report payload. Это файловое
  восстановление; не новый выполненный SQL restore и не запуск на новой публичной VM.

Receiver доступен только отдельной SSH-учётной записи. Не выдавайте этому аккаунту
доступ к production DB/API или публичное чтение каталога. Пример
`offsite-config.example.json` не содержит реквизитов и не является рабочей конфигурацией.

После назначения настоящего receiver установить его файл и закрытый root там, положить
проверенный private JSON на VPS `/home/arborscan/.config/arborscan/offsite.private.json`
с mode 600, без секретов в аргументах. Команды из проверенного ops-tools:

```sh
python3 /home/arborscan/ops-tools/ops_offsite_push.py push \
  --config /home/arborscan/.config/arborscan/offsite.private.json \
  --backup /home/arborscan/ops-backups/YYYYMMDDTHHMMSSZ \
  --release-root /home/arborscan/recovery-releases \
  --receipt-root /home/arborscan/ops-backups/offsite-receipts

# Перед pre-delete политика должна вызвать свежую проверку, а не доверять старому receipt.
python3 /home/arborscan/ops-tools/ops_offsite_push.py verify \
  --config /home/arborscan/.config/arborscan/offsite.private.json \
  --name YYYYMMDDTHHMMSSZ

# Новый private каталог; production база не используется.
python3 /home/arborscan/ops-tools/ops_offsite_push.py recover \
  --config /home/arborscan/.config/arborscan/offsite.private.json \
  --name YYYYMMDDTHHMMSSZ \
  --destination /home/arborscan/offsite-recovery-NEW
```

Эти команды подготовлены, **на настоящем независимом receiver не выполнены** из-за
отсутствия назначенного хоста/доступа. Автоматическое подключение к уже существующему
backup остаётся выключенным до приватной настройки. Успех isolated тестов не закрывает
этот инфраструктурный блокер и весь AS-14.

## Windows совместимость и проверки

Новый outer manifest включает `postgres/SHA256SUMS`. Исправлен подтверждённый дефект
pull: файл больше не пересоздаётся после проверки, если уже покрыт outer SHA; для старых
наборов восстанавливается с LF. Иначе Windows newline translation могла немедленно
сломать только что проверенный новый набор. Valid COMPLETE сохраняется во время
повторной проверки; marker отзывается только после доказанной ошибки содержимого.
Существующие каталоги, расписание, аккаунт, режим SSH и контрольные копии сохраняются.

Автоматические проверки текущего кода:

- 21 новых isolated filesystem/RPC тест: полная передача/readback, повтор, потерянный
  ответ/возобновление, плохой префикс, отказ receiver с сохранением источника,
  corrupt COMPLETE, изменение manifest, пути/симлинки, обязательные native/runtime
  части, восстановление одного report link, receipt и отказ тому же VPS; старый `./`
  manifest, новый runtime контракт/побайтное сохранение absolute paths, real receiver
  CLI с fcntl и две конкурирующие записи одного chunk.
- SSH command policy/redaction проверяются отдельным mock: StrictHostKeyChecking,
  BatchMode, отдельный key/known_hosts и отсутствие слепого повтора записи. На Linux
  приватный ControlMaster переиспользует соединение для больших архивов; это не
  проверка настоящего внешнего SSH-хоста.
- 6 Windows pull тестов, включая новые exact-byte SHA/LF и сохранение valid marker;
  4 прежних offsite verifier и 8 runtime bundle regression тестов прошли.
- Финальный объединённый Windows прогон всех четырёх модулей: **36 passed,
  2 Linux-only skipped**, 5,579 с;
  [сохранённый вывод](evidence/release-readiness/offsite-tests-windows.txt).
- Тот же код фактически проверен в отдельном Linux-контейнере точного image034a9d,
  `--network none`, read-only source/rootfs, без production env, только tmpfs тестовые
  данные: **38/38**, 28,452 с. [Вывод](evidence/release-readiness/offsite-tests-linux.txt).
  Это настоящий receiver CLI/lock, но не настоящий внешний SSH-хост.
- После последних guards: целевой Windows **19 passed, 2 Linux-only skipped**,
  6,979 с ([вывод](evidence/release-readiness/offsite-tests-windows-last.txt)); Linux
  receiver suite **20/20**, 29,118 с ([вывод](evidence/release-readiness/offsite-tests-linux-last.txt));
  последняя SSH policy **1/1**, 0,035 с ([вывод](evidence/release-readiness/offsite-ssh-policy-linux-last.txt)).
- Python AST трёх модулей и `git diff --check` без ошибок.
- C: был ниже настоящего 2 GiB guard: начальный запуск старых pull тестов корректно
  отказал; suite выполнен с временными тестовыми каталогами на D:, без ослабления guard.
  Один промежуточный rename на D: дал WinError 5; затронутый Windows suite повторён
  успешно (6/6), после чего оба окончательных прогона успешны.

06.10 в 15:26:46 UTC агент установил только исправленный публичный Windows pull module
в существующие private tools (SHA `60552bf33263dc0576cb5e571f4caf3cf83166d97fe4b141e0628cda07885571`),
предыдущий module сохранён `.pre-readiness-20261006`; запущена существующая Scheduler
задача. Её завершение/полученный набор фиксируются отдельно по настоящему результату;
сам Running не означает успешной копии. Данные/расписание/аккаунт не удалялись и не менялись.

Тестовый receiver — локальный каталог/RPC fixture, не независимая площадка. Настоящую
передачу на круглосуточный независимый receiver и копию при выключенном ПК пока не
заявляем. Действующая Windows-копия проверена отдельно ниже.

## Фактически завершённая Windows-копия

Существующая задача `ArborScan Offsite Backup` завершила запущенный агентом перенос
06.10 **15:26:46–15:40:01 UTC**, exit/result **0**, 795,010 с. Она повторно проверила
13 существующих наборов и получила `20261006T131325Z`; общий результат — 14
проверенных наборов текущего VPS inventory. SFTP один раз вернул `connection closed`
(exit255); ограниченный второй attempt продолжил staging и завершился полной
SHA-проверкой. Источник, историческая дополнительная D-копия `20260927T032112Z`,
расписание и пользовательский аккаунт сохранены.

Агент независимо прочитал реальные D-файлы: **30 outer SHA / 979475430 байт**,
**4 native PostgreSQL SHA / 52162724 байт**. Затем из полученного application archive
в новый закрытый каталог восстановлено **2261 файл / 828152403 байта**, проверено
**16 связей report payload**, 3,857 с. Это файловое восстановление Windows-копии;
новый SQL restore не выполнялся, production-реквизиты при нём не загружались.
Состав dump/application не менялся, поэтому повтор прежнего полного SQL/lab restore
для этой передачи не требуется.

После этой проверки найден ещё один конкретный Windows-дефект: `write_text` менял
LF outer manifest на CRLF. Архивы были правильными, но hash самого manifest
различался. Новый regression сначала упал на Windows, затем прошёл с явной записью
UTF-8/LF. Свежий D-manifest восстановлен из доверенного SSH-источника после проверки
всех данных: теперь его **побайтный SHA совпадает с VPS** —
`65fb5865cdc25971f8852a82e11132b3baac93c5bf8b150b241ae6985f99abdc`.
Исторические manifest и архивы не перезаписывались. Installed pull module теперь
имеет SHA `4cb9ec25e3dcf8186b77abd7dd605137f182ab27e4dd42da0716ae1f4e760c4e`;
обе прежние версии сохранены приватно.

Именно установленная Scheduler-задача повторно запущена после последнего исправления:
**19:54:16–19:56:56 UTC**, 160,104 с, **LastTaskResult=0**, все 14 существующих наборов
проверены без новых загрузок и retry. State=Ready, NEXT=07.10 08:00 Europe/Minsk;
StartWhenAvailable/IgnoreNew сохранены. Это два ручных запуска существующего
автоматического механизма, а не доказательство следующего календарного запуска.
Последний Windows suite — **40 tests: 38 passed, 2 Linux-only skipped**, 6,287 с;
новый outer-LF regression затрагивает Windows, предыдущие Linux receiver/lock
результаты относятся к неизменённым sender/receiver.

Доказательства: [Scheduler, SHA и file restore](evidence/release-readiness/windows-offsite-actual.json),
[outer-LF исправление](evidence/release-readiness/offsite-manifest-lf-repair.json),
[последний suite](evidence/release-readiness/offsite-tests-windows-final.txt).
Private raw logs, backup data и конфигурация в Git не включаются.

## Реальные текущие runtime архивы на Windows

Актуальные image archives дополнительно сохранены в private D-каталоге, сохранив
существующий base archive. Первоначальная медленная SFTP-попытка остановлена только
для собственного процесса; её неполный файл оставлен без COMPLETE. Вместо повторной
передачи общих OCI-слоёв агент использовал уже SHA-проверенные base/delta с D, получил
через trusted SSH точные tar headers/малые изменённые части и восстановил **побайтно
те же исходные tar**. Оба итоговых full-file SHA совпадают с VPS, не только отдельные
слои; повторное чтение всех трёх архивов также успешно. Основная проверенная процедура
заняла **25,742 с**, large missing blobs не потребовались:

| Архив | Размер, байт | Проверенный SHA-256 |
|---|---:|---|
| API v4 `0437e98` | 608791552 | `a232aeff128924ce45ae0fd4a5ac312459f504b62702f9ad181f72815074b43f` |
| Worker `eaa39f4` | 608723456 | `7a3e499fd51e7c5ede7a69a08b556c042b8c6d107450545db0b9a0b5ff8efe7a` |
| Прежний base `images.tar.zst` | 2563869234 | `3f8d154c102db8ee06a0d3fd7e72df67d4b423b9c7a190d29ff54c3c02ad5140` |

Image ID текущего API — `sha256:034a9d67b49adb5565a1af9454706a104721e21160d725db32a31f796ff20193`,
worker — `sha256:3fe28a5b7eb5716a71e984908520c31fff9ac325a9c5885c5e102e93996ac14a`.
[Метаданные проверенных файлов](evidence/release-readiness/windows-runtime-actual.json).
Эти актуальные runtime assets проверены отдельно: application set `20261006T131325Z`
создан до обновления API v4 и не выдаётся за новый snapshot работающего image034a9d.
Новый systemd backup фиксируется основным отчётом по фактическому завершению.

Наборы на D действительно независимы от VPS, но новая передача требует включённого
Windows-ПК. Это не закрывает требование внешней копии при выключенном ПК, не является
запуском на новой публичной VM и не изменяет отдельный блокер receiver выше.

## Календарный Windows-run 07.10 и повторы inventory

Настоящий календарный запуск `ArborScan Offsite Backup` 07.10
**05:00:01,965–05:00:07,091 UTC** завершился **exit1** с безопасным
`ssh_inventory_failed`. В доступных TaskScheduler events до следующего ручного
запуска не найдено автоматических повторов, хотя RestartCount=3/PT15M настроен.
Точная историческая причина SSH не установлена: старый stderr не сохранялся;
проверка доступного ssh.service journal 05:00–05:01 UTC не дала записей. Это не
подтверждённый отказ авторизации, не доказательство серверной перегрузки и не
успешная календарная внешняя копия.

Новый `read_inventory` добавляет не более **трёх read-only попыток по180с** с паузами
**5/10с**. Каждый раз используется прежняя команда SSH с BatchMode и строгой
проверкой host key. Timeout/nonzero exit записывает только фиксированные категории;
сырой stdout/stderr, реквизиты и исключения не попадают в журнал. Отсутствующий
клиент отказывает сразу. Некорректный, не список или пустой успешный ответ не
повторяется и не выдаётся за успешное резервное копирование. SFTP, архивы,
retention и механизм process lock не менялись.

Regression сначала воспроизвёл отказ прежнего single-attempt поведения; последний
целевой suite — **14/14 Windows,0,226с** и **14/14 Linux,0,070с**: семь новых случаев
и семь прежних transfer cases. Проверены transient failure→success, предел повторов,
timeout, неизменность auth/host-key arguments, отсутствие приватных диагностик,
invalid/empty response и local-client error. Linux — изолированный image034a9d,
network none, read-only source/rootfs, tmpfs4GiB; Windows — Python3.14.6/private D
fixtures. [Windows вывод](evidence/release-readiness/inventory-retry-windows.txt),
[Linux вывод](evidence/release-readiness/inventory-retry-linux.txt).

Source module SHA:
`0beba3e66cee6d13552e813d279beb24b1160557150e28812647b19537080122`.
Независимый read-only обзор нового helper не выявил критических дефектов; это
отдельный обзор, не подмена выполнения тестов. Первоначальный перенос работал с
прежним SHA4cb9ec25. Новый SHA0beba3e6 установлен только после завершения этого
процесса; прежний модуль сохранён приватно. Фактические результаты последующих
переносов и восстановления приведены ниже.

07.10 **06:58:58 UTC** новый helper дополнительно выполнил настоящее read-only
SSH-чтение за **122,813с**: получены14 COMPLETE sets, последний `20261006T195518Z`,
manifest SHA `d850deae43556e2dcb69980cd3619b7013b428a66a7d5860a7310657fae17eb2`.
Первая попытка успешна, retry не понадобился. Это проверка реального inventory
пути; успешный Scheduler/download с этой версией зафиксирован ниже после
завершения предыдущего работающего переноса.
[Безопасный actual proof](evidence/release-readiness/inventory-retry-live.json).

## Свежий полный набор на Windows 07.10

После календарного отказа два ручных запуска существующего Scheduler закончились
сохранёнными staging-данными и **exit1**: **06:33:06–07:06:16 UTC**,1990,482с,
и **07:11:26–07:32:37 UTC**,1270,822с. В первом были три SFTP `connection closed`,
во втором два записанных `connection closed/reset`, после чего последняя попытка
read-only delta также отказала. Частичный archive вырос до **807141376байт**;
COMPLETE при этих отказах не создавался. Историческую причину SSH/SFTP закрытия
не удалось установить: доступ к server journal ограничен, отсутствие доступных
строк не доказывает отсутствие серверной ошибки. Состояние production не менялось.

Для завершения агент в **отдельном private D-каталоге** восстановил application.tar
побайтно: доверенный SSH предоставил raw tar headers/footer и SHA текущих members,
старый локальный archive предварительно проверен целиком, переиспользованы только
совпадающие SHA payload. Уникальные reused payload — **502248890байт**, полученные
missing payload — **329686байт**, private plan через read-only SSH — **5127023байта**.
Повторяющиеся payload в tar записаны на свои исходные позиции; никакая запись
пользователя не объединялась или не менялась. Получены **830341120байт /2266members**,
точный full SHA совпал с VPS:
`fca5be26a348003e185df446ace2184ce8394a7b5c8c7d79f012bb4391aa07ad`.
Операция завершилась **09:55:04 UTC**,91,960с. Это reuse проверенных байтов, а не
полная повторная сетевая передача830MB. Неполный Scheduler staging не затрагивался.
[Безопасный actual proof](evidence/release-readiness/application-member-reuse-actual.json).

Именно существующая Scheduler-задача с установленным SHA0beba3e6 затем использовала
этот byte-exact локальный archive через прежний block-reconstruction механизм:
**09:53:32–09:57:14 UTC**, **222,485с**, **LastTaskResult=0/State=Ready**,
13verified_existing + свежий **20261006T195518Z downloaded_verified**,14наборов
текущего VPS inventory. Этот последний run прошёл без retry. На D сохранены16
исторических COMPLETE-наборов, включая уже не выбранные сервером старые копии;
ничего из них агент не удалял. Это ручной запуск существующей автоматической
задачи; календарный запуск 07.10 остаётся неуспешным. Следующее расписание —
08.10 **05:00UTC /08:00Europe/Minsk**, прежние Interactive/StartWhenAvailable/IgnoreNew
и3restart/PT15M не менялись.

Набор назван06.10 по времени его начала, но серверный daily-service завершил его
**07.10 около03:34UTC**. После окончания Windows-передачи агент повторно прочитал
все реальные файлы: **29 outer SHA /979545362байта**, **4 native PostgreSQL SHA
/52163691байт**. Outer manifest сохранён в исходных LF-байтах, SHA совпал с trusted
source: `d850deae43556e2dcb69980cd3619b7013b428a66a7d5860a7310657fae17eb2`.
Inventory aggregate979547977байт включает также служебные файлы, поэтому его нельзя
подменять суммой файлов outer manifest. PostgreSQL-архив проверен по SHA;
новый SQL restore для этой Windows-передачи не заявляется.

`RUNTIME_DEPENDENCIES.json` действительно входит в outer SHA, исходные байты и
absolute source paths сохранены. Его две image ID совпали с текущими API034a9d и
worker3fe28a. Все **3 уникальных runtime archive /3781384242байта** с D повторно
проверены целиком по SHA и скопированы в **отдельный private relocation package**.
Схема receiver package validation и `verify_relocated_runtime` прошли на настоящих
файлах: **4 исходные ссылки**, включая общий base для двух images, разрешены;
original index не переписывался. Полный локальный package — **34файла
/4760932219байт**, каждый файл скопирован и прочитан обратно по SHA. Это закрытая
Windows-копия вместе с актуальными runtime assets, не внешняя always-on площадка.

Из application.tar этого relocated package восстановлены и проверены **2265файлов
/828196863байта**, **18 report-version связей**,6,354с. Это настоящее file-only
восстановление в новый закрытый каталог; production credentials не загружались,
PostgreSQL не перезаписывался, API/worker не запускались. Проверка full package,
runtime relocation и файлового восстановления заняла38,789с, завершение
**09:59:12UTC**. Предыдущие SQL/lab restore результаты относятся к своим версиям;
эти действия не подменяются одной SHA-проверкой и не повторялись без причины.
[Полное Windows доказательство](evidence/release-readiness/windows-offsite-fresh-20261007.json).

Остаток AS-14 сохранён: новый backup при выключенном Windows-ПК не обеспечен,
authorized independent always-on receiver отсутствует. Отдельная новая VM/полный
HTTPS recovery не запускались. Успешный календарный run исправленной Windows-задачи
ещё предстоит; ручной success не выдаётся за него. Production main/API/DB/модели
в ходе этих off-site действий не менялись.
