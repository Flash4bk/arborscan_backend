# AS-16: подготовленная процедура обновления только API v4

Дата: 06.10.2026. Скрипт: [ops_integration_v4.py](../../ops_integration_v4.py).
Эта процедура подготовлена и проверена локальными safety-тестами; фактические
prepare/deploy/public/API результаты записывает основной агент в RELEASE_INTEGRATION.md.
Настоящее развёртывание этим документом не объявляется выполненным.

## Исходные допущения

- Текущий проверенный v4: image ID
  `sha256:9dc0ad620110a347b7bbd5a050ad84a6d2453e16165abc15aee47550ff8e017b`,
  OCI revision `4483d48100514ec249bb7e72cff6565f0f7eb324`.
  Любое расхождение останавливает prepare, а не автоматически переопределяет baseline.
- Кандидат заранее собран основным агентом из точного проверенного commit в
  отдельном checkout на существующем совместимом runtime. OCI image revision
  должен точно совпадать с полным commit; тег разрешается в immutable image ID.
  Скрипт не собирает образ и не устанавливает зависимости.
- Исправление кандидата не требует SQL/смены форматов хранилища; старый v4 должен
  читать сохранённые кандидатом записи. Если это условие нарушается, применение
  данным скриптом недостаточно и не разрешается без отдельного разбора.
- Пользователь `arborscan` может читать действующие Compose-файлы,
  `/etc/arborscan/arborscan.env`, модели и конфигурационные файлы `/etc/nginx`,
  обращаться к Docker; доступно место не меньше размера исходного образа + 1 GiB.
  Недоступные права не обходятся. Секреты остаются приватными.
- Нужен последний COMPLETE backup в `/home/arborscan/ops-backups`, не старше
  36 часов. Проверяются полный SHA256SUMS, PostgreSQL COMPLETE/собственный manifest,
  заголовок custom archive PGDMP и сохранённое свидетельство offline-перечитывания
  application/Storage archive. Это проверка существующей резервной копии,
  **не выполненный PostgreSQL restore и не SQL-миграция**.
- Реальный `ops_backup.sh` исключает из outer manifest все файлы `SHA256SUMS`,
  включая PostgreSQL manifest. Его наличие и hashes проверяются отдельно;
  каждый внутренний payload должен также присутствовать во внешнем manifest
  с тем же hash. Оба manifest hashes фиксируются для проверки перед deploy.

## prepare, deploy, rollback

Использовать путь скрипта из того же точного checkout кандидата. Подставляемые
значения `CANDIDATE_CHECKOUT`, `EXACT_COMMIT`, `CANDIDATE_IMAGE`, `LATEST_BACKUP`
должны быть фактически проверены основным агентом, а не взяты из старого отчёта.

```bash
python3 CANDIDATE_CHECKOUT/deploy-vps/ops_integration_v4.py prepare \
  --directory /home/arborscan/as16-v4-YYYYMMDDTHHMMSSZ \
  --backup /home/arborscan/ops-backups/LATEST_BACKUP \
  --commit EXACT_COMMIT --image CANDIDATE_IMAGE
```

prepare не перезапускает production. Создаёт новый каталог mode 700, приватные
копии фактического container inspect, полного env, исходных Compose-файлов,
точный архив прежнего runtime image и manifests mode 600. Фиксирует конфигурацию,
модели, HTTPS-конфигурацию, image/container ID/StartedAt/environment v3 и worker.
Создаёт overrides **только `api-v4`** с полным неизменённым эффективным environment:
погодный ключ и остальные переменные не исчезают при откате приложения.
Marker PREPARED публикуется только после проверки всех копий и invariants.

```bash
python3 CANDIDATE_CHECKOUT/deploy-vps/ops_integration_v4.py deploy \
  --directory /home/arborscan/as16-v4-YYYYMMDDTHHMMSSZ
```

Перед мутацией повторно проверяет backup, private manifests, image/source revision,
runtime identity, неизменность config/environment и исключённых компонентов.
Команда Compose имеет единственную service-цель:

```text
up -d --no-build --pull never --no-deps --force-recreate api-v4
```

Разные планы/повторные вызовы этого оператора сериализованы глобальным
неблокирующим flock. Logs сохраняются приватно. Возвращаемый JSON не содержит
environment, ключей, пользователей или таблиц. После изменения API проверяется
health до 90 с и повторная неизменность v3/worker/models/HTTPS; public HTTPS,
auth и отдельные тестовые write/read сценарии выполняются основным агентом.
Без результата этих проверок интеграция не считается завершённой.

```bash
python3 CANDIDATE_CHECKOUT/deploy-vps/ops_integration_v4.py rollback \
  --directory /home/arborscan/as16-v4-YYYYMMDDTHHMMSSZ
```

Откат восстанавливает **точный прежний image ID и полный прежний environment,
включая действующую погоду**. Если исходный образ исчез из Docker cache, он
загружается из проверенного приватного runtime-image archive; его ID и OCI
revision снова проверяются до Compose. Исходные contours/reports/SQL не
перезаписываются и не восстанавливаются поверх production; это откат приложения.

Timeout, unhealthy запуск или отсутствие container после собственного вызова
не выдаются за успех. Наблюдаемое состояние фиксируется приватно, чтобы выполнить
подготовленный rollback. Чужой recreated container, неизвестный image/environment
или изменение v3/worker/config/models останавливают операцию. Автоматический
рестарт собственного candidate с тем же container ID/image/environment допускает
reviewed rollback, но не обход исходного identity guard при новом deploy.

## Проверенное локально

```powershell
python -m unittest tests_v4.test_ops_integration_v4 tests_v4.test_ops_weather_v4_refresh -q
```

**37 tests passed**, из них 29 guarded-rollout tests и 8 существующих
weather-safety tests. Только временные синтетические файлы и mocked Docker:
prepare без restart, copied hashes, полный env/погода, current/candidate revisions,
повреждение/пропуск/выход manifest за backup, отсутствие PG COMPLETE, неверный
PGDMP при корректных hashes, модели/HTTPS/config/v3/worker/API drift, bounded
Compose target, timeout/unhealthy/отсутствующий container, сохранность отката,
восстановление исходного image cache и глобальный nonblocking lock.

Включены семь регрессионных cases после безопасного отказа actual prepare:
реальная outer структура без `postgres/SHA256SUMS` принимается; отсутствие или
неполнота inner manifest, payload вне outer coverage, несовпадение hashes между
manifest, неверный inner hash и изменение inner после prepare блокируют работу.
Безопасный отказ до создания rollout-каталога не являлся развёртыванием.

Зависимости: Linux Python >=3.11, Docker CLI с Compose JSON config, существующие
читаемые operator-файлы. Интерактивный sudo, новый DB-доступ или пароли скрипту
не нужны при перечисленных правах. Запуск safety-тестов не является live deploy.
