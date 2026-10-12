# AS-14 / AS-16: резервные копии локального стека

Этот документ описывает замену облачного snapshot и отделяет её выполненные
шаги от оставшихся проверок. Backup tools уже установлены; первый новый
systemd-запуск завершился по timeout, повтор после исправления - exit0.
Оба новых Windows-комплекта имеют проверенный COMPLETE. Отказ CRLF-привязки
исправлен; финальный reader установлен. Настоящий Windows Scheduler завершён
exit0, original14 защищены, delete=[]. Будущий календарный запуск не заявлен.
Старые облачные копии и их доказательства сохраняются.

## Исходное состояние службы, прочитанное 12.10.2026 до установки

- `arborscan-backup.timer`: active / waiting; следующий запуск 13.10.2026
  03:21:33 UTC, последний запуск 12.10.2026 03:20:04 UTC.
- `arborscan-backup.service`: failed, `Result=exit-code`, `ExecMainStatus=1`;
  запуск закончился в 03:31:12 UTC. `User=arborscan`, UMask 0077,
  TimeoutStartSec 40min.
- Установленный entry point: `/bin/sh /home/arborscan/ops-tools/ops_backup.sh`.
  Его SHA256 `511e3ffb450f6817b2a24f76c7fabe513cd75ea68940332df6c4376b025a7af0`.
  Установленный `ops_backup_policy.py` имеет SHA256
  `452bbae97bcc5a5f04317b9dc4f57074372cde69c2ca3ba61f6c01f76ba12097`.
- Эти SHA относятся к прежней облачной процедуре до установки. Историческая
  ошибка 03:31 и следующий timer deadline сохраняются как исходные сведения;
  последующий install не объявляется успешным запуском systemd.

## Фактически выполненная установка 12.10.2026

- В `/home/arborscan/ops-tools` установлен проверенный local-only bundle из
  точного Git-коммита `983706f9917bf9d9f4334ffbc1ac5b668f151984`.
  Источник: `/home/arborscan/selfhost-20261012T140855Z/install-tools-983706f`.
  Установщик завершился с `selfhost_backup_tools_installed`.
- Прежние файлы, config и modes сохранены в
  `/home/arborscan/selfhost-20261012T140855Z/backup-tools-rollback-20261012T192531Z`.
  Journal format2 фиксирует завершение установки; приватный config включает
  новый стек и exact seven-image runtime index.
- Новый entry `ops_backup.sh` имеет SHA256
  `92ca7bddaf5d0eec0631eb746f5dbc57538819bea3784ac2d1a3f68de4258852`;
  `ops_backup_policy.py`:
  `49dc6f5831672f4df54aa6311f1a657b792812f7b6ce37cb5db6d104befe2b3f`.
- Registry `SELFHOST_TRANSITION_PROTECTION.private.json` действительно защищает
  точные исходные 14 комплектов по их manifest SHA. Незавершённая облачная
  попытка сохранена отдельно приватной операцией `20261012T191204Z` под
  `backup.lock`; её staging/partial archives и старые COMPLETE не удалены.
- Receipt установки подтверждает `systemd_changed=false`, `service_started=false`,
  `backup_sets_changed=false`. Units/timer и API при установке не менялись.
  Последующий systemd-запуск учитывается отдельно ниже.
- На Windows установлены девять reader/tool модулей из Git
  `682ec13bd6e896d656153669c1f85ff1e996b242`; прежние пять scripts сохранены в
  `D:\ArborScanBackups\tools\pre-selfhost-20261012T191315Z`.
  Task Scheduler не менялся. Первый transfer остановился на 180-секундном
  timeout при повторном хешировании historical remote chunks.

## Последующие проверки и оптимизация

- Первая проверка установленной systemd-службы действительно началась
  12.10.2026 в 19:26:49 UTC. Она опубликовала проверенный новый комплект
  `20261012T193726Z`, но закончилась 20:06:49 UTC по `TimeoutStartSec=40min`.
  Наличие valid COMPLETE не превращает timeout в scheduled success;
  та попытка не завершилась успешно.
- Установлена причина длительной работы: многократное повторное хеширование
  около 4,57 ГБ исходных 14 комплектов на промежуточных этапах inventory/plan.
  Исправление зафиксировано в `ddb7cc21f`: повторные проверки внутри операции
  используют проверенный cache. Окончательная независимая проверка байтов
  перед удалением остаётся uncached. SHA, transition protection и правила
  допустимости удаления не ослаблены. Оптимизированный bundle из точного
  `ddb7cc21f46182585a37f9459a7de13051aeee2c` установлен в 20:39:49 UTC;
  policy SHA `51b6a4646971cf2562ba3e5c036b2de298cbf8232f1d6739c46032609880ff34`.
  Источник `ROOT/install-tools-ddb7cc2`, откат tools сохранён в
  `ROOT/backup-tools-rollback-20261012T203949Z`. Units/API/копии не менялись.
  Под `backup.lock` полностью проверены native/runtime/manifest опубликованного
  набора: 1747867196 байт, manifest SHA
  `7f2a73ab64ca22813ccd9aa3d1f69ca1c48d17c59f24b7c7a2a43a132ba55a6e`.
  Связанный указатель сохранён как
  `active-backup.local-retained-20261012T203914Z.private.json`.
  Повторный systemd-запуск прошёл 20:45:27-21:05:21 UTC: Result=success,
  ExecMainStatus=0, service inactive. В доступном прикладном journal действительно
  прочитан `verified_existing`, deleted=[]; старые сообщения systemd manager
  ограничены правами, их чтение не заявлено. Last-success связан с тем же
  manifest, completed/verified time сохраняет 19:53:32 UTC, а не время retry.
  Replay проверяет ранее созданный snapshot и не обновляет содержащиеся в нём
  данные. Более поздний integrity check не обозначает новую точку копирования БД.
  Сохранены 16 комплектов: 14 transition originals и новый pool из двух.
  Новых копий при повторе и удалений нет. Timer enabled/active, следующий
  срок по чтению после завершения - 13.10 03:18:48 UTC. Будущий календарный
  запуск и продолжительность полного нового pool14 ещё не проверены.
- Windows reader с verified-manifest hints действительно установлен из
  `31967af`; SHA256 установленного reader начинается с `a90d8cae`.
  Возобновляемый manual transfer в 20:12 UTC прервался из-за SSH-соединения.
  Повтор доставил `20261012T183850Z` в 20:56:15 UTC: COMPLETE, тот же manifest
  SHA `06a0d01b…`, 17 covered files и полный seven-service runtime closure.
  Второй комплект `20261012T193726Z` получил COMPLETE в 21:09:35 UTC,
  manifest SHA `7f2a73ab…`. Safe receipt
  [windows-copy-proof.json](evidence/selfhost-recovery/windows-copy-proof.json)
  фиксирует оба комплекта, совпадение SHA четырёх native PostgreSQL файлов и
  настоящий заголовок PGDMP. Это проверка перенесённых архивов; изолированный
  SQL/raw restore выполнен для `183850Z`, для `193726Z` он не заявлен.
  В 21:17:42 UTC свободно 6 478 790 656 байт на D:; это разовый остаток,
  не доказательство ёмкости будущего rolling14.
- Общий Windows pull затем остановился с
  `transition_original_not_verified_locally`: у пяти исходных manifest
  исторически CRLF, девять совпадают с source SHA побайтно; все 14 совпадают
  после приведения переводов строк к LF. Это установлено отдельным read-only
  [windows-transition-diagnosis.json](evidence/selfhost-recovery/windows-transition-diagnosis.json).
  Это первоначальный отказ; исходные архивы и внешние SHA256SUMS сохранены.
  Исправление e65d836 сохраняет source SHA и добавляет local raw SHA только
  при точном CRLF→LF совпадении. Все covered files проверяются при adoption
  и повторе. Созданный registry защищает все14, включая пять CRLF записей.
- Программный пакет проверен на Linux: 111 тестов прошли, 3 пропущены;
  отдельная проверка installer после исправления shell test harness -
  30 тестов прошли. На Windows: 84 прошли, 9 пропущены. Эти результаты
  подтверждают программные проверки и не заменяют фактическое завершение
  systemd-службы или Windows transfer.
- Дополнительный source review выявил прежние redirect/hardlink дефекты
  изменяемых путей reader. Проверены настоящие Windows symlink, junction и
  hardlink в изолированных fixtures; чужие исходные файлы не изменились после
  исправления. Read-only общие архивы допускаются. Последний reader
  `6456170b17dd3e4b2431bdc3f9032c5b8467b138`: 86 tests +5 subtests.
  Финальный e65d836 вместе с CRLF-регрессиями прошёл 108 tests +5 subtests
  и установлен в 21:25:46 UTC. Reader SHA9232cfd8…, contract SHA82f8ad38…;
  точные hashes/rollback в windows-reader-final-install.json. Настоящий
  Scheduler21:26:14-21:32:55UTC прошёл exit0/Ready/success с прежним timeout180
  и неизменным определением задачи. Проверены16 server sets, на Windows19
  полных наборов: original14 плюс pool5 (два новых/три прежних локальных).
  Plan delete=[] и actual retention deleted=[]; лишней копии не создавалось.

## Формат и совместимость

Имя каталога по-прежнему `YYYYMMDDTHHMMSSZ`. Остаются `application.tar`,
`local-files.tar`, приватные environment/containers/Compose, `postgres/`,
SHA256SUMS и COMPLETE. `SELFHOST_BACKUP.json` (format 1,
kind `local_selfhost_backup`) включает ровно семь ролей:
db, rest, storage, gateway, v3, api, worker.

Добавляются `storage-files.tar` и `runtime-files.tar`. Первый содержит реальные
байты выбранных поколений локального Storage, второй - содержимое фактического
model-quality mount. Mount может быть named volume или приватным bind внутри
каталога кандидата. Созданный, остановленный worker сохраняется без запуска
заданий и обучения. Core db/rest/storage/gateway должны работать.

В обоих raw-архивах SHA-covered `filesystem-metadata.json` хранит uid/gid,
mode без SUID/SGID, mtime_ns, xattrs в base64, права родительских каталогов.
Текстовые xattrs также присутствуют в PAX. Обычный `tarfile.extractall()`
не заменяет восстановление этих метаданных. `restore_volume_archive()`
проверяет архив до записи и допускает только новый пустой целевой каталог;
на Linux восстанавливает владельца, права и xattrs. Ошибка оставляет отдельный
частичный target для диагностики, существующие данные не стираются.

Подтверждённый layout Storage:
`GLOBAL_S3_BUCKET/TENANT_ID/bucket/key[/version]`. При null version последний
компонент отсутствует. Для текущего кандидата оба префикса - `arborscan`.
Logical `application.tar` сохраняет старые `objects/bucket/key` и tables/schema;
автоматически полученные точки редактора не создаются.

`ops_runtime_contract.py` добавляет seven-service branch только при
`selfhost:true`, покрытом SHA маркере и согласованных Compose project/roles,
именах и exact image IDs. Старые format-1 копии с двумя и тремя application
services читаются по прежним правилам. Новые копии требуют обновлённых модулей
Linux policy и Windows reader; старым reader их передавать для ротации нельзя.

## Нативная БД и консистентность

Открывается READ ONLY REPEATABLE READ транзакция, экспортируется snapshot.
JSON public tables и Storage inventory берутся из неё; `pg_dump --snapshot`
использует тот же ID до закрытия транзакции. Выполняется настоящий
`pg_restore --list` по созданному файлу, roles выгружаются без паролей.
Это проверка списка архива, а не заявление об успешном SQL restore.

Результат JSON передаётся из живого psql через однострочный base64-frame.
Тип PostgreSQL json сохраняет форматирующие переводы строк вложенных значений;
построчное чтение необрамлённого JSON недопустимо. В первой фактической попытке
12.10.2026 такой ответ был отклонён как `invalid_snapshot_response` до создания
native archive/COMPLETE. Исправление проверено fixture с multiline JSON и кириллицей;
незавершённые приватные stages сохранены, прежние копии не менялись.

Storage файлы читаются helper-контейнером с `--network none`, read-only root
и read-only mount. Проверяются SHA и inode/size/mtime/ctime/xattrs до/после
чтения. После чтения свежий Storage metadata inventory должен совпасть со
snapshot. Запись/удаление во время копии приводит к incomplete и повтору,
а не остановке приложения. Изменение heartbeat model-quality во время чтения
тоже может потребовать повтора. Ежедневный успех проверяется настоящей службой.

После проверки extraction `application.tar` удаляется только созданный данным
запуском staging/restored. Это производная проверочная копия, не исходный архив
и не прежний комплект. Хранить ещё одну полную extracted Storage ежедневно
не требуется. Проверенные архивы и receipt остаются.

## Image closure без новых загрузок

```sh
python3 /path/to/verified-tools/ops_selfhost_backup.py \
  --prepare-runtime-index \
  --root /home/arborscan/selfhost-20261012T140855Z \
  --runtime-index /home/arborscan/ops-runtime/RUNTIME_INDEX.json
```

Создаётся новый индекс в
`/home/arborscan/ops-runtime/selfhost-20261012T140855Z/RUNTIME_INDEX.json`.
Исходный индекс не переписывается. Проверенные существующие SHA-assets
используются повторно; только отсутствующие exact local images выгружаются
через `docker image save`. Pull, запуск worker и покупка ресурсов отсутствуют.

Containerd Docker может использовать SHA OCI index как image ID. Проверяется
этот exact index, manifest/config/layers доступной linux/amd64 платформы.
Arm64, не установленная на текущем хосте, не объявляется включённой.
Валидный completed partial export можно закончить после проверки; повреждённый
partial сохраняется и блокирует публикацию. Проверка tar не заменяет isolated
docker-load с проверкой exact image IDs.

В Windows closure переносится существующим SHA-addressed shared store и
RELOCATED_RUNTIME. Один asset не скачивается заново для каждой ежедневной копии.
Новые семь образов нельзя потерять при переносе только application.tar.

Windows reader передаёт только имена уже присутствующих COMPLETE и SHA их
SHA256SUMS. При совпадении manifest SHA удалённая инвентаризация не вычисляет
блоки для возобновления передачи этих старых файлов. Это подсказка для экономии
чтения, а не доказательство целостности: reader затем заново проверяет SHA всех
покрытых локальных файлов, а удалённые shared runtime archives по-прежнему
полностью хешируются. Неполный или отличающийся комплект получает обычные
block hashes; повреждённый локальный комплект не объявляется проверенным.
Обычный предел inventory остаётся 180 секунд; для первого восстановления можно
явно выбрать `--inventory-timeout 600`. Timeout не вызывает автоматический
повтор: удалённый процесс может пережить закрытие SSH. До нового запуска нужно
подтвердить завершение предыдущего собственного inventory, не останавливая
посторонние сеансы. Этот программный контракт проверен fixtures; фактический
Windows transfer отдельно подтверждается по его receipt и реальным файлам.

## Сохранение исходных 14 и новый rolling14

До переключения зафиксировать исходные 14 **точных** имён и SHA256SUMS SHA256
в приватном `SELFHOST_TRANSITION_PROTECTION.private.json` рядом с комплектами:

```json
{"format":1,"kind":"selfhost_backup_transition","sets":[
  {"name":"YYYYMMDDTHHMMSSZ","manifest_sha256":"64 lowercase hex"}
]}
```

Эти комплекты не меняются и исключаются из нового retention pool. Можно держать
защищённые исходные14 и новый rolling14: при успешной пятнадцатой новой копии
удаляется только старейшая допустимая новая. PINNED, runtime/Compose зависимости,
manual/unadopted и повреждённые комплекты сохраняют прежние ограничения.
Отсутствующий или изменённый защищённый manifest блокирует процедуру.
Никакой автоматической отмены transition protection нет.

Обновлённый Windows pull принимает эту же защиту только после загрузки и
проверки точных локальных оригиналов. Имеющийся иной registry не переписывается.
На Windows исходные14 привязаны после полного SHA readback всех covered
файлов. Для пяти CRLF manifest registry дополнительно содержит
local_manifest_sha256. Reader проверяет одновременно неизменённый local raw
SHA и точный source LF SHA; retention сравнивает local raw SHA. Существующий
иной registry не заменяется. Результат actual Scheduler зафиксирован отдельно.
Выбранный Windows root остаётся `D:\ArborScanBackups`. Перед созданием дополнительной
копии нужны свободное место на новый комплект, недостающие runtime assets и запас;
предварительное удаление старой копии для освобождения места запрещено.

Фактический стартовый запас 12.10.2026: VPS 42 572 529 664 байта перед первой
manual local попыткой; Windows D: 11 280 244 736 байт по проверке основного агента.
Этого нельзя заранее объявлять достаточным для protected14 + rolling14:
нужны измеренный размер новой копии, недостающие shared image assets и запас
на временный restore. При недостатке места процедура останавливается до
создания/удаления копий. Она не удаляет исходные14, прежние restore fixtures или
другие данные пользователя, чтобы искусственно пройти эту проверку.

## Безопасный порядок переключения

1. Проверить свободное место, отсутствие пересекающегося backup, установленный
   timer/service, текущие scripts SHA. Сохранить installed scripts приватно.
2. Проверить новый seven-image index и создать manual local snapshot без ротации:

   ```sh
   python3 /path/to/verified-tools/ops_selfhost_backup.py \
     --root /home/arborscan/selfhost-20261012T140855Z \
     --runtime-index /home/arborscan/ops-runtime/selfhost-20261012T140855Z/RUNTIME_INDEX.json \
     --storage-layout version-directory
   ```

3. В отдельной изолированной БД реально восстановить native archive, сверить
   public/Storage data hashes и ACL/indexes/functions. В пустом raw target
   восстановить Storage bytes/permissions/xattrs и проверить actual file bytes.
   Проверить image closure/load. Production БД и файлы не использовать target.

   `ops_selfhost_restore_check.py --backup /home/arborscan/ops-backups/UTC \
   --evidence-root ROOT/backup-restore-checks` создаёт отдельный PostgreSQL без
   сети/production credentials и новые raw volumes. Сравниваются все public
   rows, Storage inventory, migration versions и SHA/uid/gid/mode/mtime/xattrs.
   Скопированные sessions/jobs не удаляются, API/worker не запускаются. Receipt
   содержит `restored_backup`, `restore_verified` и `backup_manifest_sha256`,
   чтобы результат нельзя было применить к другому комплекту.
4. Создать transition registry по точным исходным14. Сохранить незавершённый
   cloud `active-backup.json` приватно отдельным именем, не стирать staging и
   partial archives. Новый scheduled mode отказывается молча использовать
   cloud transaction. Зафиксировать это отдельным доказательством.
5. Только после подтверждённого cutover install проверенные backup modules и
   Windows reader modules. Установка нового dispatch без config сохраняет cloud
   mode. Приватный config 600 в `/home/arborscan/ops-tools` включает локальный:

   ```json
   {"format":1,"root":"/home/arborscan/selfhost-20261012T140855Z",
    "runtime_index":"/home/arborscan/ops-runtime/selfhost-20261012T140855Z/RUNTIME_INDEX.json",
    "storage_layout":"version-directory"}
   ```

   Имя файла: `SELFHOST_BACKUP_CONFIG.private.json`. Root должен иметь PREPARED,
   actual current Compose overlays/binds должны оставаться внутри него. Архив
   сохраняет actual config files и модельные bind bytes.

   Для установки подготовлен `ops_install_selfhost_backup.py`. Он требует
   actual `PROMOTED`/journal phase, остановленную backup service, остановленный
   worker, проверенные transition originals и отсутствие active attempt.
   Source должен находиться в ROOT и содержать `INSTALL_SOURCE_SHA256.json`
   format1/files с SHA каждого из шести файлов: `ops_runtime_contract.py`,
   `ops_verify_restore.py`, `ops_selfhost_backup.py`, `ops_backup_policy.py`,
   `ops_backup_dispatch.py`, `ops_backup.sh`.

   ```sh
   python3 /path/to/verified-tools/ops_install_selfhost_backup.py \
     --root /home/arborscan/selfhost-20261012T140855Z \
     --source /home/arborscan/selfhost-20261012T140855Z/verified-install-source \
     --runtime-index /home/arborscan/ops-runtime/selfhost-20261012T140855Z/RUNTIME_INDEX.json
   ```

   Прежние bytes/modes/config сохраняются в private rollback directory.
   Receipt `INSTALL_RECEIPT.private.json` format2 фиксируется до замены файлов;
   фаза и намерение каждой следующей записи сохраняются атомарно. Первой
   заменой entry `ops_backup.sh` временно становится закрытая заглушка:
   она завершает backup с ошибкой без чтения источников и изменения комплектов.
   Затем устанавливаются модули и private config. Рабочий entry публикуется
   последним, после проверки SHA всего комплекта. Так таймер не запускает
   смесь прежних и новых модулей.

   Обычная ошибка оставляет фазу `paused`, заглушку и сохранённые исходные
   файлы; аварийный обрыв может оставить `installing` с записанным намерением.
   Автоматического возврата к historical policy при активном таймере нет.
   Для продолжения использовать тот же проверенный source и index, явно
   указав каталог сохранившегося receipt:

   ```sh
   python3 /path/to/verified-tools/ops_install_selfhost_backup.py \
     --root /home/arborscan/selfhost-20261012T140855Z \
     --source /home/arborscan/selfhost-20261012T140855Z/verified-install-source \
     --runtime-index /home/arborscan/ops-runtime/selfhost-20261012T140855Z/RUNTIME_INDEX.json \
     --resume-directory /home/arborscan/selfhost-20261012T140855Z/backup-tools-rollback-UTC
   ```

   `UTC` заменить точным именем каталога этой операции. Resume допускает
   только исходные или предназначенные для установки bytes и известную
   заглушку; постороннюю операторскую правку не затирает. До завершения
   resume запуски timer через заглушку не создают новую копию. Оставшийся
   private partial сохраняется и не мешает повтору атомарной записи.
   Утилита не запускает службу, не меняет units/timer, не архивирует active
   attempt автоматически и не удаляет комплекты. Фактическая установка этого
   варианта из `983706f` выполнена; точные source и rollback directory указаны
   выше. Пример команды с `verified-install-source` описывает подготовку новой
   операции, а не второй обязательный install.
6. Запустить **установленную** systemd-службу, проверить exit0, journal, COMPLETE,
   SHA и seven-service closure. При интерактивном sudo недоступен только шаг:
   `sudo systemctl start --no-block arborscan-backup.service`.
7. Выполнить Windows pull с `--retention-dry-run`, прочитать real files и closure,
   проверить protection и план удаления. Затем разрешённую ротацию и штатный
   Scheduler, не выдавая ручной Python вызов за проверку Scheduler.

## Откат процедуры

При сбое локальной проверки installed source не переключать. Для прерванной
установки использовать сохранённый receipt: продолжить её через
`--resume-directory` либо выполнить описанный ниже guarded rollback.
Вручную переименовывать config и смешивать версии модулей не требуется.
Прежняя облачная процедура
может оставаться неработоспособной из-за HTTP402: откат tools не устраняет тарифный
лимит. Старые архивы/registry и новые COMPLETE не удалять. Откат API/БД выполняется
отдельно по actual cutover receipt, не этой backup-процедурой.
После новых записей в локальный стек нельзя возвращать БД к прежнему облачному
snapshot: он не содержит эти записи. Откат образа приложения должен сохранять
актуальную локальную БД и Storage; обратная миграция данных требует отдельного
проверенного переноса. Guarded rollback tools не выполняет такую миграцию.

Подготовленный tool rollback проверяет SHA сохранённых файлов и допускает
точные old/new bytes, известную заглушку или отсутствие впервые добавленного
файла. Поэтому он работает и после частичной установки. Более новую
операторскую правку он не затирает. Откат также сохраняет намерения в receipt:
после обрыва его можно повторить с тем же каталогом, завершённый откат
идемпотентен. Прежний entry возвращается последним после восстановления
модулей и config с исходными mode. Перед возвратом
historical policy должны быть остановлены service и timer: прежняя версия может
не учитывать transition protection. Утилита сама systemd не меняет.
После этого точная команда:

```sh
python3 /path/to/verified-tools/ops_install_selfhost_backup.py \
  --rollback-directory /home/arborscan/selfhost-UTC/backup-tools-rollback-UTC
```

## Доказательства и границы

12.10.2026 на VPS действительно создан и проверен отдельный selfhost runtime
index: 7 services / 7 unique images, 3 новые local image exports и 1 восстановленный
проверенный partial; cloud_requests 0, старый runtime index не изменён.
Проверенные tools этой операции: кандидат `backup-tools-d54d8ddf308f`.
Эта проверка runtime index сама по себе не подтверждает новый production
backup или systemd success; последующие фактические комплекты описаны отдельно.

Локальные программные проверки охватывают настоящие tar bytes, SHA, empty-target
restore, поддержание legacy2/3 и selfhost7, идемпотентный shared transfer, отказ
при missing role/image/marker, защита original14 и отдельный new14 pool,
scheduled replay, отказ silent cloud transaction reuse, dispatch config guard.
Windows не предоставляет Linux xattrs: соответствующий fixture отдельно
пропущен. Linux ownership/xattrs и настоящий PostgreSQL restore проверены ниже;
это не заменяет проверку установленной systemd-службы или Windows Scheduler.

Hardened installer отдельно проверен на Windows: 29 pytest прошли, один тест
точных Unix mode пропущен из-за среды. Тесты используют настоящие private
файлы и моделируют обрыв после каждой атомарной замены, публикации entry,
удаления нового config и до `os.replace`; проверены resume, повтор отката,
сохранение старых bytes и отказ при посторонних изменениях. Эти fixtures
проверяют прерывания; отдельно подтверждён успешный install на VPS из
`983706f`. Аварийную установку и legacy rollback на production не провоцировали.

12.10.2026 создан настоящий local-only комплект
`/home/arborscan/ops-backups/20261012T183850Z`, размер 1 747 317 940 байт.
SHA256 файла SHA256SUMS:
`06a0d01b3778e7a4969779a0e61ee016fd2ab4d3a27a4e46cd9ebf08dbfb0924`.
Он включает 7-image dependencies; cloud requests и ротация при создании отсутствуют.
Исходные облачные комплекты не изменены.

Реальный isolated native SQL/raw restore имеет receipt
`/home/arborscan/selfhost-20261012T140855Z/backup-restore-checks/20261012T184346Z-bcad9cd5ecc3/result.json`.
Сверены все 15 public tables, включая 41 auth_sessions и 0 ml_jobs, и все
2258 Storage objects. На двух новых volumes восстановлены 2258 Storage files /
2825 directories и 2 runtime files / 1 directory с проверкой bytes, uid/gid,
mode, mtime_ns и xattrs. Migration versions [1,1,1]; native pg_restore успешно
восстановил owner/role/ACL entries, copied sessions/jobs не удалялись.
API/worker/training не запускались. Изолированная БД подтверждённо Exited(0).

Проверенная actual tool bundle: `backup-tools-52b1d83c7d92` в ROOT.
SHA backup module `f22f139c04d5199bd76f789b23f411e98547c0255f423505fe5fda57b95ed302`,
restore checker `38c521cd917403707551b738719f128edd7b45bc8fe5ccbbcc29a673e05a7fca`.
SSH транспорт этого запуска перестал доставлять окончательный stdout; результат
и Exited(0) проверены отдельным ограниченным read-only запросом. Остановлен только
этот stale локальный SSH child, не серверная служба или чужой сеанс.

14 новых комплектов такого размера занимают около 24,46 ГБ без shared images и
staging. Начальные 11,28 ГБ свободного D не подтверждают capacity этого pool.
Первую копию и будущую ротацию следует оценивать раздельно; недостаток места не
разрешает удаление защищённых originals. Backup modules на VPS и Windows reader
установлены, существующие service/timer и Windows Task Scheduler не менялись.
Оптимизированный Linux bundle установлен, manual-trigger systemd replay завершён
exit0, два новых Windows-комплекта COMPLETE и полный штатный Scheduler
проверены. Registryoriginal14/source+localSHA, retain14 и plan/apply deleted=[]
подтверждены. На 21:33 UTC D свободно45 120 217 088 байт, volume Healthy.
Предыдущий срез21:22 показывал6 478 790 656 байт; рост места не засчитывается
как уборка этой процедуры, причина роста здесь не исследовалась. Сам Scheduler
ничего не удалял. Наблюдённый размер позволяет рассчитать запас для следующих
комплектов, но рост данных/runtime и полный pool14 не испытаны.
Будущий календарный запуск и отдельный
isolated image-load не подтверждены. Проверку содержимого image archives не выдаём за загрузку
этих образов на изолированном хосте. Новые интерфейсные проверки телефона или
Google-входа этим backup-документом не заявлены.
