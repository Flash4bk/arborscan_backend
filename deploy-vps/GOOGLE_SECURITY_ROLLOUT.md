# AS-16 — применение исправления безопасности Google, 08.10.2026

Основание: `AS16_GOOGLE_SECURITY_ROLLOUT_PROMPT.md`, актуальный main
`d3ae1e129e54beba58f889630a97bdfd7fd8657a`, отдельная
`codex/google-security-rollout`. Исходный `D:\arborscan_backend` и его
пользовательские изменения сохранены. Приложенная чат-копия плана 1.6 не
заменяет каноническую редакцию 1.24; более новые результаты сохранены.

**Production обновлён после отдельного разрешения владельца:**
«Разрешаю только это обновление v3 и уникальный индекс».
Первоначальный §3 запрещал v3/schema, поэтому до этого разрешения выполнены
только подготовка и изолированные проверки. Фактический путь —
`/api/v3/auth/google` в `arborscan-api`, собственные
`public.users/auth_sessions`, не Supabase Auth/GoTrue и не API v4.
Применены только security-only v3 и additive unique subject index;
создание/связывание Google-профилей включено после проверки индекса.

## Проблема и подготовленное поведение

Подготовленный policy commit:
`6284f93204f3ef2f92543604eb5c7a2cea93c3cd`.
Старый обработчик мог брать email из тела клиента при отсутствии email в
проверенных claims, не требовал `email_verified=true` и перезаписывал прежний
`google_sub` при совпадении email. Это воспроизведено **только на синтетических
аккаунтах**, не против production-пользователей.

Исправленный обработчик использует подтверждённые Google claims и стабильный
subject. Неподтверждённые данные дают 401, конфликт владельцев — 409; client
email/name/avatar/role не назначают identity или права. Уже однозначно связанный
профиль получает тот же owner/password/role. Новое создание/связывание запрещено
при `claims=false` (503 с понятным ответом); включение требует проверенного
уникального индекса. Email-профиль связывается только при Google authority для
Gmail/verified Workspace, не по произвольной внешней почте.
Контракт и прежние проверки: [FINAL_RELEASE_GOOGLE.md](FINAL_RELEASE_GOOGLE.md).

## Фактический состав кандидата

Вместо полной пересборки main с новыми зависимостями использован ровно текущий
immutable v3 image как base. Генератор
[ops_google_security_candidate.py](ops_google_security_candidate.py) проверяет
pinned Git objects/checkout SHA, известный baseline handler, весь baseline SHA,
неоднозначные маршруты и запрет перезаписи output. Он заменяет только функцию
`auth_google`, добавляет подготовленные import/claims constant и auth module.
**Все остальные байты production server.py сохранены.**

| Поле | Проверенное значение |
|---|---|
| Исходный production v3 / preimage | `sha256:5a0b1d63e6c5eb13d14af84590dafebf900f59156613b987000ae8b6938f563b` |
| Фактически запущенный production v3 | `sha256:60c6cf1fc185e41aaf6275522df851b60f142887554685ce7aec49b3f702a990` |
| Overlay server.py SHA | `a5490340b1d3bd56e999d7c2facba0ec3fc7142ad1273079cc37483e5d3d2979` |
| Auth module SHA | `451b1273a68ecba63253eb12c48bc1a35c159d41329336234e60b4b6267b3bec` |
| Live baseline server SHA | `076268a262b78df5028a77c6b99d898b56d4e111437e102ee8b509a4cd1e23df` |

Build `--pull=false --network none` добавил один filesystem layer. Проверены
base tag→ID до/после, source SHA внутри image, прежние RootFS layers,
Cmd/Entrypoint/User/Healthcheck и environment с единственным добавленным
default `ARBORSCAN_GOOGLE_IDENTITY_CLAIMS_ENABLED=false`. Pip/apt/model downloads
не выполнялись. Это композиция конкретного live base и конкретного policy
commit, **не развёртывание всего main**.
[Build/provenance](evidence/google-security-rollout/candidate-build.json).

## Выполненные проверки агента

| Среда / группа | Результат и граница |
|---|---|
| Windows Python 3.14.6, overlay safety | 12/12: pinned objects/checkout, tamper, byte preservation, minimal Dockerfile, path/symlink/overwrite guards; не production deployment |
| Windows Python 3.12.14, exact-route HTTP | Prepared 21/21, baseline 20 ожидаемых наблюдений. Реальные FastAPI/Uvicorn/REST и RSA/google-auth; AST routes, без полного ML import/startup |
| VPS isolated original image, Python 3.11.16 | Baseline 21 ожидаемых наблюдений: полный server import/lifespan/models-ready, actual HTTP и настоящий verifier. Восемь небезопасных policy/claims-disabled случаев дали 200, 11 synthetic identity writes; это red-доказательство, а не «безопасный baseline» |
| VPS isolated candidate image, Python 3.11.16 | 22/22: полный import/startup/models-ready; malformed/signature/audience/issuer/expiry/future-iat refusals, absent/false/string email_verified, email fallback, wrong/ambiguous subject, client-field injection, canonical owner, bearer/me/logout/expiry/two-owner isolation. Identity/training writes 0/0 |
| Production после переключения | Оба HTTPS health 200/status ok; без bearer me/corrections/reports 401; missing token 422, malformed token 401. Live source SHA и healthy/claims=true подтверждены. TLS проверяется штатно. Не воспроизводили опасные claims на production |
| API24, настоящий Android UI | Exact APK23/прежний signer: два Google-входа/выхода, два cold restart с server-confirmed auth/me того же разрешённого аккаунта, два открытия пустой тестовой истории без ошибки; отмена picker оставила UI неавторизованным. Это эмулятор, не S24 |
| API36, настоящий Android UI | Exact APK23/новый signer: два Google-входа/выхода, два app cold restart с server auth/me прежнего аккаунта, два открытия тестовой истории без ошибки и picker cancel. Сессия также сохранилась после cold boot AVD. Это эмулятор, не S24/полевая точность |

Полные isolated проверки запускались sequential, `network none`, readonly root
и models, tmpfs cache/home, 2 GiB memory limit, без production env/credentials.
Supabase заменён loopback REST fixture; **Google verifier не заменён**, только
certificate transport получает ephemeral RSA public key. Private key/token
остаются в памяти. Production Google login этими signed synthetic tokens не
доказывается. Все hashes models до/после совпали, training-state GET без writes.
Harness SHA `a3cf75fc20d45010d01ac854741e43ba3c42aa7fa64385d244b6467a61197b23`.
Runtime: fastapi0.141.1, uvicorn0.52.4, requests2.34.2, google-auth2.57.1,
cryptography50.0.1; Windows dependency versions отдельно в evidence.

Доказательства: [full baseline](evidence/google-security-rollout/http-baseline-image.json),
[full candidate](evidence/google-security-rollout/http-prepared-image.json),
[local prepared](evidence/google-security-rollout/http-prepared-local.json),
[local baseline](evidence/google-security-rollout/http-baseline-local.json),
[harness self-test](evidence/google-security-rollout/http-harness-self-test.json),
[production/readiness](evidence/google-security-rollout/readiness.json).
[Postdeploy](evidence/google-security-rollout/postdeploy.json),
[настоящий API24 UI](evidence/google-security-rollout/native24-ui.json),
[настоящий API36 UI](evidence/google-security-rollout/native36-ui.json).
Первоначальный сбой AST-обвязки и чтение optional legacy runtime metadata
сохранены приватно и исправлены в проверочных процедурах; это не дефекты API.

Ранее проверенные 17 policy tests и 16 isolated PostgreSQL17.6 checks сохранены
по ссылкам в FINAL_RELEASE_GOOGLE.md. SQL/policy не изменялись; SQL restore,
Flutter181/Android9/analyze/build/PDF/полный S24 UI повторно не выполнялись.

## Production, резервная копия и совместимость

08.10, 10:20–10:37 Minsk: VPS Git main `5d8176a`, дерево чистое, не переключалось.
Все production container IDs/images/start times прежние и healthy:
v3 `5a0b1d63…`, v4 `034a9d67…`/код0437e98, worker `3fe28a5b…`.
[Initial state](evidence/google-security-rollout/initial-state.json).

Проверен fresh COMPLETE `20261008T032219Z`: **29 outer SHA entries**, native
PostgreSQL coverage/PGDMP и три dependency archive SHA; 979618890 payload bytes.
Это проверка целостности, **не новое восстановление базы**.
Отдельный current runtime/config/source snapshot mode700/600:
`/home/arborscan/ops-google-security/20261008T072247Z`.
Exact original Docker archive 787466240 bytes, SHA
`e703d1ac976a8266b2f89ab6f6954d145f35cb0b9230eef7db52922b4945d167`;
пять файлов проверены повторным SHA. Содержимое env/inspect не публикуется.
[Backup/preflight](evidence/google-security-rollout/preparation.json).

Начальная DB preflight была **только SELECT**: PostgreSQL17.6, google_sub=text,
invalid rows0, duplicate subject groups0, index absent. После разрешения
**реально выполнен** exact `CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS`
для `public.users_google_sub_unique`, не только просмотр SQL. Индекс
unique/valid/ready, единственный key `public.users.google_sub`, predicate
`((google_sub IS NOT NULL) AND (google_sub <> ''::text))` подтверждены.
Строки пользователей не переписывались; версии contour/history/quality остались
1/1/1. Утренний systemd backup exit0/03:33:24UTC, timer active/waiting,
14 COMPLETE. Следующий запуск 09.10,06:18:04Minsk. Ротация не менялась.

APK остаётся **1.3.0/code23**, source `bbcd5b539427e62da58739fc2dbeff8550ee72c6`,
SHA `71f38189ee62a7825029ec7c25bf635891d3adcd0fa3eea3d9b618a223e4a31c`.
Client/body/opaque session contract совместим; старые дополнительные поля
принимаются, но игнорируются для identity. Google client ID, две Android
регистрации/Testing audience, applicationId/signing lineage неизменны.
Прежние реальные native API24/old signer и API36/new signer проверки сохранены;
offline/retry неизменённого клиента повторно не запускались. После rollout
выполнены отдельные целевые API24 и API36 UI-циклы. Read-only запросы до переключения,
после API24 и после API36 подтвердили один прежний owner/subject/role/created_at без дубля:
[до](evidence/google-security-rollout/identity-before.json),
[после API24](evidence/google-security-rollout/identity-after24.json),
[после API36](evidence/google-security-rollout/identity-after36.json).
Email/name/owner/subject/token и raw screenshots остаются приватными;
в evidence только результаты и SHA скриншотов. При выходе email может оставаться
в редактируемом поле входа — это не авторизованная карточка. Profile «Обновить»
обновляет статистику, поэтому auth/me проверен полным перезапуском.
S24 был доступен, но аккаунт/данные/приложение этим пакетом не изменялись.
Сессия opaque ArborScan; проверка восстановления не называется refresh-token.
Первый cold boot API36 остановил OS/SystemUI ANR, не доказанный дефект ArborScan:
[непройденный первоначальный срез](evidence/google-security-rollout/native36-initial-boot-blocked.json).
Обычный перезапуск собственного AVD без wipe/clear/reinstall восстановил работу;
финальные native проверки выполнены12:27UTC, результат не выдан за успех
заблокированной первой попытки.

## Выполненное переключение и подготовленный откат

Private overrides подготовлены из **фактического runtime environment**, не из
потенциально более нового env-file. Compose5.5.1 `config --format json` проверил
`!reset env_file`, `!override environment`, точный image, исходный loopback port
и совпадение environment. Остальные Compose/runtime параметры наследуются.
Overrides не содержатся в Git; paths/SHA/argv:
[commands](evidence/google-security-rollout/commands.json).

После разрешения v3 и fresh snapshot/config/image/invariants выполнено:

```sh
JOB=/home/arborscan/ops-google-security/20261008T072247Z
docker compose -p arborscan --project-directory /opt/arborscan/deploy-vps \
  -f /opt/arborscan/deploy-vps/docker-compose.yml \
  -f "$JOB/deploy-claims-false.private.yml" \
  up -d --no-deps --no-build --pull never api
```

Следующим шагом выполнен отдельно разрешённый schema change. SHA SQL
`123097353e4c29e592b0c73baf25a555f311fac8a7e770e8366fa27329a83f83`
сверен; exact [google-identity-unique-index.sql](google-identity-unique-index.sql)
применён через private service с autocommit, без `--single-transaction`:

```sh
/home/arborscan/.local/bin/psql -w 'service=arborscan_backup' \
  -v ON_ERROR_STOP=1 -f "$JOB/google-identity-unique-index.sql"
```

Pre/post guards проверили тип, значения, дубликаты и exact unique/valid/ready
partial index. Затем применён тот же Compose argv с
`deploy-claims-true.private.yml`. UTC: claims=false 07:47:32–07:49:35,
SQL 07:50:35–07:50:38, claims=true 07:51:40–07:54:18.
[False phase](evidence/google-security-rollout/applied-false.json),
[production SQL](evidence/google-security-rollout/applied-sql.json),
[true phase](evidence/google-security-rollout/applied-true.json).
Первая попытка SHA guard остановилась до любых изменений из-за checkout CRLF;
передан exact pinned Git object с LF. Guard не отключался.
Container `aa91bec2effcb10cb062370e329b1bbb1ce5fb19c07988e37ae4d2c2c5d96218`
запущен 07:51:42 UTC, healthy. Startup logs проверены приватно: fatal/model startup
failure нет. v4/worker IDs/images/start times, model SHA, Nginx/operator config,
VPS Git main5d8176a/чистое дерево и training-state неизменны. Worker не перезапускался;
обучение не инициировалось, данные отчётов/контуров/модерации не изменялись.

Неизменные start times выше относятся к переключению07:54UTC. Позднее обнаружена
**перезагрузка VPS11:56:56UTC**, не инициированная агентом: все три прежних
container IDs/images сохранились, current start times стали11:57UTC.
Причина перезагрузки не установлена. Final read-only проверка после native UI
подтвердила retained60c6/claims=true/index, полный прежний effective runtime env
за исключением claims flag, operator/base compose SHA, VPS Git, model_v4 SHA и
training-state. Проверены обе health/unauthorized/invalid token, **stdout и stderr**
запуска: startup complete, fatal/traceback нет. Timer после boot active/waiting,
следующий09.10,03:18:10UTC/06:18:10Minsk.
[Финальный runtime после boot/UI](evidence/google-security-rollout/final-runtime.json).
Первоначальное ожидание неизменных start times остановило final verification;
host boot установлен отдельно, current timestamps сохранены.

Предпочтительный безопасный возврат — исправленный image с
`deploy-claims-false.private.yml`: новые связи временно недоступны, linked login
работает; index оставить. `rollback-original.private.yml` восстанавливает
exact preimage/environment, **вместе с исходным дефектом**. Для аварийного
возврата prepared `emergency-google-disabled.private.yml` использует preimage
с пустым GOOGLE_CLIENT_ID: Google login временно недоступен, password login и
существующие сессии остаются. Это containment, не проверенный production rollback.
Никакого restore dump поверх production/переписи user rows/удаления index.
Client rollback остаётся отдельным code>23 с прежней lineage без clear/uninstall.
**Откат подготовлен и config-validated, но не выполнялся.**

Безопасный pause новых связей — приведённая выше команда с
`deploy-claims-false.private.yml`. Аварийное отключение Google с preimage,
если исправленный image непригоден (только containment, не постоянное решение):

```sh
JOB=/home/arborscan/ops-google-security/20261008T072247Z
docker compose -p arborscan --project-directory /opt/arborscan/deploy-vps \
  -f /opt/arborscan/deploy-vps/docker-compose.yml \
  -f "$JOB/emergency-google-disabled.private.yml" \
  up -d --no-deps --no-build --pull never api
```

После любого отката проверить exact image/env/health, отказ без bearer и
неизменность v4/worker/model/index. Не удалять index и не восстанавливать dump.
`rollback-original.private.yml` можно подставить в тот же argv только понимая,
что старый Google handler вновь открывает найденный дефект; это небезопасный
постоянный вариант, без заявления о его production-проверке.

Current candidate Docker archive также реально создан: 787500544 B, SHA
`82f59d586d20b1e232175a53c0260ce9a68b5fc88da5f9f7d931d2096b1594a9`,
mode600, [доказательство](evidence/google-security-rollout/candidate-image.json).
Private source/config/preimage и candidate archive хранятся в JOB, не в Git.
Текущий container после host reboot сохраняет свой image/env. При ручном
`docker compose up` нужно использовать **active claims-true override**:
исходные compose/env/VPS Git сохранены и сами не выбирают новый image.
Для AS-14 переноса на новый хост необходимы current v3 archive/effective config
и SQL index; их наличие в независимой/ежедневной копии пока не подтверждено.

## Оставшийся шаг

Разрешённый security rollout выполнен. Stable tag не создан:
ресурсный остаток AS-14 открыт, ограничение явно не принято владельцем.
Один комплект ресурсов — [AS14_RESOURCE_REQUEST.md](AS14_RESOURCE_REQUEST.md).
Научные AS-10/AS-11/AS-13, качество моделей и визуальная приёмка AS-15 сохранены.
