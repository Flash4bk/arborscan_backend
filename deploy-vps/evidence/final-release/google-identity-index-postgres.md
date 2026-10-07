# AS-16: настоящий PostgreSQL для подготовленного Google identity index

Проверка агента **07.10.2026, 13:28:09–13:28:27 UTC**: **16/16 passed**,
17.391 с. [Санитизированный JSON](google-identity-index-postgres.json) содержит
точные SHA SQL/runner, digest образа, версию сервера и каждый результат.

Среда — собственный одноразовый Docker-контейнер на **существующем VPS**,
PostgreSQL **17.6**, уже имевшийся образ
`sha256:178f0976b54a39237096bfa310c1a352dbc82fb1b08dda45cdb8acb5d40c1426`.
Это настоящий PostgreSQL с независимыми сессиями, не PGlite и не mock.
Контейнер: `network=none`, read-only root, пользователь postgres без capabilities,
база/Unix socket в `/tmp` tmpfs; без host bind mounts, production env/реквизитов
и данных. Использована только синтетическая `public.users`.
Собственный контейнер удалён после проверки ID, имени, image, label и network.
API v3/v4/worker не перезапускались. Backup/restore не повторялся.

**Production-миграция не выполнялась. Новая VM/её HTTPS не проверялись.**
Успех этой серии не доказывает тип/состояние реальной users, доступ production
оператора или выполнение ограничения в production. Identity claims остаются
выключенными до отдельного разрешённого rollout и успешных production guards.

## Проверенные сценарии

- Первый запуск и повтор сохраняют один valid/ready UNIQUE index с прежним OID;
  строки пользователей не переписываются. Проверено для `text` и `varchar(255)`.
- Несколько старых NULL/empty subjects разрешены. Case-distinct subjects не
  объединяются. Невалидные и слишком длинные старые subjects отвергаются.
- Старые duplicate subjects отвергаются до создания индекса; строки сохраняются.
- Одноимённые nonunique, wrong-column, wrong-predicate и non-index relation
  отвергаются. Invalid index после неудачного CONCURRENTLY build не принимается.
- Duplicate, появившийся после успешного preflight до build, вызывает 23505;
  invalid index не объявляется ready, оба исходных пользователя сохраняются.
- Обёртка всей миграции в транзакцию отвергается с 25001. Нужен autocommit.
- Две независимые транзакции, вставляющие один subject, реально конфликтуют:
  второй запрос ждёт transactionid lock, после первого commit получает 23505;
  остаётся один пользователь.
- Две транзакции, привязывающие один subject к двум старым строкам с NULL,
  также имеют одного победителя; второй запрос получает 23505, обе строки
  сохраняются, subject остаётся только у победителя.
- Неподдерживаемый `char(255)` отвергается до создания индекса.

## Найденный дефект и прежние попытки

Первая реальная PostgreSQL-серия дала **13/14**, не успешное завершение:
[предыдущий JSON](google-identity-index-postgres-before.json). Для varchar
PostgreSQL добавляет явное `::text` в deparsed predicate. Старый exact-text guard
ошибочно отклонял уже созданный valid index и его повторный запуск. SQL исправлен:
type whitelist `text/varchar` проверяется заранее, два точных канонических
predicate expressions принимаются обоими guards. Структура ошибочных индексов
по-прежнему отвергается. Финальная серия относится к исправленной SQL версии.

После этой серии независимый review runner выявил ещё два дефекта транспортной
обвязки: host с ведущим `-` мог попасть в SSH options, а потеря stdout Docker run
оставляла собственный контейнер без cleanup. Теперь CLI отвергает такой host до
I/O, а cleanup при неизвестном ID ищет только точное собственное UUID-имя и
проверяет все прежние ownership guards перед удалением. Ни broad lookup, ни
удаление чужих контейнеров не добавлены. Новые 8 regression tests прошли на
Windows/Python 3.12 и в [реальном Python 3.11.16 серверного образа](google-auth-python311-runtime.json).
SQL не менялась и 16 PG tests повторно не запускались: PG JSON честно сохраняет
SHA ранее выполненного runner `5ecfc811…`, runtime JSON фиксирует SHA исправленного
runner `7b7cccdb…`. Эти targeted tests проверяют parser/cleanup transport, а не
заменяют реальную PostgreSQL-серию.

Две Windows-попытки через отдельный SSH на каждый запрос завершились ошибками
setUp и не считаются DDL-проверкой: single-test diagnostic установил SSH exit255
с connection closed, тогда как тот же reset SQL через рабочий сеанс вернул 0.
Причина закрытия SSH не установлена. Остаточный собственный lab был точно
идентифицирован и удалён. Финальный runner выполнен рядом с Docker одним
SSH-сеансом, поэтому внутренние запросы не зависели от множества SSH connections.
Сырые diagnostic outputs остались только в приватном D output, вне Git.

## Повторение в разрешённой Docker-лаборатории

Нужны Python standard library, Docker и этот уже проверенный pinned image.
Публичные файлы [runner](../../../tests_v4/google_identity_unique_index_postgres.py)
и [подготовленная SQL](../../google-identity-unique-index.sql) должны оставаться
в структуре репозитория. На выделенной разрешённой Linux Docker-среде:

```bash
python3 tests_v4/google_identity_unique_index_postgres.py \
  --evidence /tmp/arborscan-google-index-synthetic-result.json
```

Runner сам создаёт отдельную synthetic database и не принимает DB URI, пароль
или имя существующего контейнера. SSH transport option предназначен только для
уже разрешённого Docker test-host; в выполненной успешной серии использовался
локальный Docker внутри одного SSH-сеанса. Контейнер/состояние production не
заменяются. Не использовать результат этого runner как подтверждение отдельного
production SQL шага.
