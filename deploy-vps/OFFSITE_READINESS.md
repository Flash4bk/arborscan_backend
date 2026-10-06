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
