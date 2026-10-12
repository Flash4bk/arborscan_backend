# AS-14 / AS-16: проверка восстановления 12.10.2026

JSON содержат только контрольные хеши, версии, счётчики и результаты проверок.
Здесь нет аккаунтов, строк БД, фотографий, паролей, токенов, env или полного
Docker inspect. Приватные исходники доказательств сохранены на VPS с правами
600 в `/home/arborscan/selfhost-20261012T140855Z`.

- `audit-before-fixtures.json`: точное совпадение всех public/Storage данных
  до DEMO-записей и HTTP-чтений, затем SHA 2252 файлов, декодирование 19 фото
  отчётов и 16 пар оригинал/PNG, прежние images/policy/models.
- `source-proof.json`: один свежий READ ONLY REPEATABLE READ snapshot источника
  непосредственно перед переключением. Байты файлов 10.10 сверены с текущими
  SQL ключами/версиями/размерами/датами; свежая облачная выгрузка байтов при 402
  не заявлена.
- `audit-restart.json`, `inference-restart.json`: повторное чтение после restart
  и контрольный анализ реального фото. Это функциональные серверные проверки,
  не измерительная валидация или проверка интерфейса телефона.
- `quarantine-result.json`: четыре свои синтетические ревизии исключены из
  настоящего экспортёра; 8/8 отказов для segmentation/classification. Реальные
  решения не менялись, обучение не запускалось.
- `public-check-result.json`: реальные запросы по публичному HTTPS с обычной
  проверкой TLS, своими DEMO-аккаунтами и данными. Вход по паролю, права,
  версии, контуры, inference, сохранение/повтор/чтение фото/PNG/измерений.
  Новый Google UI и S24 не проверялись.
- `runtime-after-cutover.json`: фактически запущенные images/states и только
  счётчики ошибок последних 20 минут, без содержимого логов.
- `isolated-restore.json`: настоящий native PostgreSQL restore и восстановление
  raw Storage/runtime в пустые отдельные тома; receipt связан с exact backup и
  SHA его manifest. Это не утверждение о новом Docker-load всех семи images.
- `backup-transition-result.json`: сохранённая облачная попытка и защита
  исходных четырнадцати комплектов по manifest SHA.
- `backup-tools-install.json`, `windows-tools-install.json`,
  `windows-reader-update.json`: установка точных проверенных modules с
  сохранением предыдущих файлов. Установка сама не доказывает успешный запуск
  systemd/Scheduler или завершённый transfer.

Развёрнутый переключатель: `682ec13bd6e896d656153669c1f85ff1e996b242`.
Установщик backup: `983706f9917bf9d9f4334ffbc1ac5b668f151984`.
Последний Windows reader/contract: `e65d836717020b732f508e4b6a55eed8121a854b`,
установлен под pull.lock в 21:25:46 UTC; первоначальная установка 31967af
сохранена отдельным receipt.
API images не пересобирались; полный SHA приведён в runtime receipt.

- `systemd-first-run.json`: реальный timeout 19:26:49-20:06:49 UTC и
  существующая COMPLETE. Чтение system journal ограничено правами; ноль
  видимых строк не означает отсутствие сообщений или ошибок.
- `published-backup-verification.json`: полный SHA/native/runtime readback
  набора `20261012T193726Z` под lock и сохранение указателя операции.
- `backup-tools-install-optimized.json`: фактическая установка `ddb7cc21f`,
  отдельный rollback directory; запуск службы установщик не выполнял.
- `test-results.json`: отдельные Windows84/9, Linux111/3 и installer30/0 для
  последнего cache. Исправленные omissions тестового upload не скрыты.
- `systemd-success.json`: реальный повтор установленной службы20:45:27-21:05:21,
  exit0/inactive, прикладной journal verified_existing, тот же manifest и прежнее
  время COMPLETE. Удаления и новая копия при replay отсутствуют; original14
  защищены. Это ручной trigger systemd, не будущий календарный запуск.
- `windows-copy-proof.json`: обе полные копии на ПК с точными source manifest
  SHA, настоящими SHA четырёх native файлов и семью runtime services/images.
  Реальный isolated restore относится только к набору183850Z.
- `windows-transition-diagnosis.json`: первоначальный отказ общего pull:
  9 исходных manifest совпали побайтно, 5 имели CRLF; после только CRLF→LF
  все14 совпали с trusted source. Старые manifest не переписывались.
- `windows-reader-final-install.json`: точная установка reader/contract e65d836
  с сохранением прежних bytes, настройка Task Scheduler не изменена.
- `windows-final-https.json`: оба public health200 и anonymous corrections401
  в Windows с обычной проверкой TLS после завершения VPS-службы.
- `windows-scheduler-success.json`: настоящий запуск существующей задачи
  21:26:14-21:32:55UTC, LastTaskResult0/Ready/journal success; прежнее определение
  задачи совпало по SHA, inventory timeout180. Original14 защищены source/local
  SHA; Windows19 полных наборов, pool5, plan/apply deleted=[]. Это ручной trigger
  Scheduler, будущий календарный запуск не заявлен. Текущий D45,12ГБ свободно;
  рост относительно прежнего среза не считается выполненной агентом уборкой.

Программные проверки: базовый операционный набор 264 passed / 3 skipped /
97 subtests; после него целевые installer 29 passed / 1 skipped и reader,
retention, delta 63 passed / 5 subtests. Наборы частично пересекаются, их нельзя
складывать в число уникальных тестов. Пропуски связаны с Linux xattrs/Unix modes
и Windows symlink privileges. Linux xattrs/ownership проверены настоящим
isolated restore. Flutter-код этой веткой не изменён, новый APK не собирался.

Последний reader/CRLF/link набор e65d836:108 passed/5subtests. Проверены
смешанные14 source LF/local CRLF, полное чтение covered SHA перед registry,
повтор, изменённые данные/manifest и реальные Windows symlink/junction/hardlink.
Это пересекающийся целевой прогон, не ещё108 независимых от предыдущего набора
тестов и не проверка интерфейса S24.

Текущие факты службы/Windows и ограничения ёмкости фиксируются отдельно в
[SELFHOST_BACKUP.md](../../SELFHOST_BACKUP.md). Общий план сохраняет различие
между технической проверкой агентом и пользовательской приёмкой.
