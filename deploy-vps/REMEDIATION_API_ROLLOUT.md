# Подготовленное обновление API: source37b

AS-01 / AS-02 / AS-04 / AS-10 / AS-11 / AS-14. Дата: 17.11.2026.
**PREPARED: рабочие API не обновлены. Требуется отдельное разрешение.**

Исходники приложения сервера: `37b5be131820f65b2180cdbfc33be48bbf832a4d`.
Builder `b1699313` реально проверил source SHA, Python3.11.16, 100/59 пакетов
и CPU Torch/NMS/cv2. Это два code-only overlay прежних runtime, без pip/apt,
новых весов или обучения. HTTP/auth/science end-to-end после активации пока нет.

| Сервис Compose | Действующий образ | Проверенный кандидат |
|---|---|---|
| `v3` | `sha256:60c6cf1fc185e41aaf6275522df851b60f142887554685ce7aec49b3f702a990` | `sha256:cbef234f2240fcb1bec3980b5e992fcdfac97967822b464422eff65ccda673d7` |
| `api` (v4) | `sha256:2abb56ed5d84e4ddbe4465da851c842861498101f030fab016f66630c4ee1762` | `sha256:66bde6a3cbf80a3eebf7caa9135cd289430e5d1bd66d256eca2a0050cb1e76fe` |

Исправления: owner-auth для legacy feedback, immutable предложение в
существующей очереди вместо назначения принятия клиентом; ограничения входов
и безопасные ошибки; подтверждение реальной сохранённой версии; согласованные
метаданные/исторические оценки; отдельный сценарий одного круглого сечения.
Сценарий возвращает момент в Н·м по явным диаметру/effective MOR. Это не β,
критический ветер или вероятность. Исходные измерения/контуры сохраняются.

## Точная граница конфигурации

Файл `/home/arborscan/selfhost-20261012T140855Z/compose.private.json`:

- текущий SHA `81d9cf5b65f4b50db1f2171157940ba078552e84880ff1f162e8cb9c64622785`;
- подготовленный four-field target SHA
  `1c8607265237b0a6a58c08857dc6a132cbdb87388eb8590c02b43ddfced0b911`;
- меняются только `/services/v3/image`, `/services/api/image` и
  `GIT_COMMIT` этих двух сервисов на полный source37b SHA.

Проверена семантическая разница только четырёх полей. Остальные ENV сохраняются;
их значения в отчёты/чат не выводятся. Metadata ENV нужно обновить явно:
унаследованное значение иначе сообщало бы прежний commit. Mounts проверены:
они не перекрывают исходники `/app`, `arborscan_v4`, `arborscan_science`.
Модели/качество/кеши используют прежние mounts. Git VPS не перезаписывается.
Worker, DB, gateway, Storage, Nginx/HTTPS и маршруты этим обновлением не меняются.

Readonly SQL-catalog проверил существующие version1/1/1, legacy action,
report-version JSONB/UUID-parent контракт и service_role EXECUTE. Новых миграций
для source37b не требуется по этим интерфейсам; live SQL-изменения не выполнялись.

## Backup и готовность до активации

Набор `20261117T033225Z` реально проверен systemd, изолированным SQL/raw restore
и Windows pull. Он опубликован в03:45:47 UTC; это не точное время SQL snapshot.
Дополнительно **выполнен новый capture**
`selfhost-remediation37b-predeploy-20261117-1939`: COMPLETE
`20261117T200418Z`, manifest SHA
`52a447b85e5849ebc22998d3cd4bfe4a2e48619952ea4f4a64dbc225167c4214`.
Публикация 20:08:32 UTC, итоговая integrity 20:11:14 UTC; exit0 за1090,420с.
PostgreSQL/JSON относятся к одному exported snapshot; runtime покрывает семь
**прежних рабочих** образов. Это новая съёмка, не daily replay и не systemd run.
Standard Windows pull за178,898с с default180 и последующий штатный latest-two
apply прошли. На D остались `20261117T033225Z` и `20261117T200418Z`, восемь общих
runtime-архивов сохранены; проверка 20:18:25 UTC подтвердила неизменность tools,
policy и task. [Доказательство](evidence/remediation-20261117/fresh-backup.json).
Новый набор отдельно не восстанавливался в SQL; actual restore относится к
утреннему `20261117T033225Z`, а fresh проверен native/full-SHA/runtime gates.
При отложенном разрешении перед активацией повторно оценить свежесть копии.

Проверены полные Docker-save архивы кандидатов:

| Сервис | Байты | SHA256 |
|---|---:|---|
| v3 | 787652608 | `d1ffcbaf5b5e171e802f3adf1463148a27875453d29e4ab1501deb35a940e631` |
| v4 | 608916480 | `55113016387fed20c98d6fa511dd27d0e33acbf9a808346e5b25cf12016be2ca` |

Новый immutable index:
`/home/arborscan/ops-runtime/repair-candidate37b-20261117T190149Z-66cc143d/RUNTIME_INDEX.json`,
SHA `983469c9582ca6640510c44f2fdc1f38a000c86e306654cc91c7bbca26a6134f`.
Он сохраняет прежние 8 mappings/9 assets и добавляет два, всего 10/11.
Все archive SHA проверены на VPS. Coverage текущих семи ролей и **планируемых**
будущих семи проверена отдельно; будущие контейнеры ещё не запущены.
Действующий указатель остаётся на старом index SHA `be74e405…6242`.
Offsite кандидатов **выполнен** отдельно от копии действующих сервисов:
на D из243855 байт переданных дельт восстановлены1396569088 байт двух полных
архивов с точными SHA/OCI/config/platform. Coverage: current8/future8,
rollback union10/full index11 assets. Receipt SHA
`92b0ad154dd1e2754ab78f9e3f92b4f45dd124c8b4ba2fd3cc2e165b271508bd`.
Private originals/targets конфигурации сохранены с ограниченным ACL; их
содержимое не включается в Git. Это не запуск/активация кандидатов.
[Доказательство](evidence/remediation-20261117/runtime-rollout.json).

Backup config меняет только `/runtime_index`: текущий SHA
`0faf23fe8d4c9a594279d996dfafd7987ee3a0d3182e643b6060f7d6e80e782f`,
подготовленный SHA `eae5d47735f8f707cae63c21ce9c554c4dd4572571121999dfd1f73a46084747`.

При экспорте сохранён floor 7444493326 байт: постоянный резерв 4 ГиБ и
оценка 3149526030 байт следующего snapshot/validation/headroom. Это оценка,
не верхняя граница роста данных; writer повторно контролирует свободное место.
Полные архивы фактически заняли 1396569088 байт, старые архивы не удалялись.

## Применение после разрешения

1. Под существующим `backup.lock` при inactive service повторно сравнить
   current Compose/config/index/source SHA и identities всех семи контейнеров.
   При дрейфе остановиться. Сохранить точные старые bytes/metadata приватно.
2. После проверки свежей pre-deployment/offsite копии атомарно установить
   подготовленный four-field target и только `runtime_index` указатель в
   существующем приватном backup config. Остальные поля/секреты не менять.
3. Пересоздать **последовательно только два сервиса**, без build/pull/dependencies:

```sh
docker --context default compose --project-name selfhost-20261012t140855z \
  --project-directory /home/arborscan/selfhost-20261012T140855Z \
  --file /home/arborscan/selfhost-20261012T140855Z/compose.private.json \
  up -d --no-deps --no-build --pull never --timeout 150 v3

# Проверить v3 health/точный image/source перед следующим сервисом.
docker --context default compose --project-name selfhost-20261012t140855z \
  --project-directory /home/arborscan/selfhost-20261012T140855Z \
  --file /home/arborscan/selfhost-20261012T140855Z/compose.private.json \
  up -d --no-deps --no-build --pull never --timeout 150 api
```

4. Проверить два public HTTPS health с обычным TLS, source37b/image IDs,
   отсутствие изменений остальных пяти контейнеров, mounts/ENV (кроме указанных
   GIT_COMMIT), активную версию/байты моделей и stopped worker.
5. На отдельных помеченных тестовых записях выполнить auth/owner/unauthorized,
   подтверждённое сохранение/повтор/конфликт, legacy proposal без принятия,
   source-bound SI section/history/reopen и недоступность модерации обычному
   пользователю. Реальные записи/решения не менять, обучение не запускать.
   Не выдавать эти проверки за S24/полевую точность.
6. Сделать post-deployment snapshot с **другим новым** `selfhost-...` operation ID,
   проверить фактически захваченные runtime IDs и offsite copy. Повтор старой
   daily/pre-deployment операции не является этим новым снимком.

## Откат

При неуспешном startup/smoke: под тем же lock и CAS восстановить точные старые
Compose и pointer bytes, затем теми же командами пересоздать только фактически
переключённые v3/api. Не трогать ещё не переключённый сервис.
Проверить старые image IDs/health и неизменность остальных сервисов/данных.
Базу/контуры/фото не откатывать поверх живых данных; схемы не менялись.
Старый API читает обычный report, новый клиент при старом science API сообщает
о недоступности 404/405. Новые JSON-поля отдельно сохраняются в версиях.
Старые runtime assets и native/Storage backup не удаляются.

## Что этим не закрывается

Обновление Nginx query logging не входит в four-field scope. Поддержка старых
клиентов с query token сохраняет отдельно описанный SEC-03 proxy-log остаток;
новый клиент использует Bearer. S24 release35 ещё не установлен, UI-проверка
нового release на AVD остановлена по памяти ПК. Независимый backup receiver,
new-host drill остаются отложенными по прежнему решению пользователя. Полный
F1/TMC/U1/D1, β-данные и полевая проверка остаются открытыми.
Main/push/merge/tag и новая пользовательская приёмка не выполняются.
