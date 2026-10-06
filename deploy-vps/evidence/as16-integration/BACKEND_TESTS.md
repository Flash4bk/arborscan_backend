# AS-16: итоговые backend и Windows проверки

06.10.2026 полный существующий backend-набор проверен в отдельном контейнере
на фактическом API v4 image
`sha256:9dc0ad620110a347b7bbd5a050ad84a6d2453e16165abc15aee47550ff8e017b`.
В него подставлены исходники/тесты кандидата интеграции; их SHA сверены с текущим
worktree после нормализации CRLF. Исходники для исправления test harness не менялись.

Итог — **182 passed, 1 skipped, 3 subtests passed**, exit 0, **10.79 с**.
JUnit содержит 186 записей, включая три subtest. Три исследовательских test-модуля
проверены отдельно в чистом research venv: 70 passed.
Версии, хеши, команда и разграничение записаны в [backend-tests.json](backend-tests.json).

Linux пропускает один Windows Job Object test. Он отдельно выполнен на настоящей
Windows в свежем venv: **1 passed за 0.13 с**, без пропусков. Проверено, что завершение
родителя останавливает его отдельно созданный дочерний фоновый процесс.
Свидетельство — [windows-job-tests.json](windows-job-tests.json).

Два прежних runtime предупреждения сохранены: Starlette о `httpx` и anyio о
`BlockingPortal`. Они присутствовали при первоначальном сборе тестов и в итоговом
прогоне; не подавлялись и не называются ошибками.

Подготовительные неудачи не скрыты:

- Первоначальный архив не содержал двух нужных tracked tools; collection завершился
  двумя ошибками. В staging добавлены 14 tracked `tools/*.py`, приватные untracked
  инструменты и datasets не передавались.
- Прогон с tmpfs 1 GiB дал 181 passed и один failure: реальный guard процедуры
  backup-transfer требует более 2 GiB свободного места. Увеличен только виртуальный
  лимит tmpfs тестового контейнера до 4 GiB. Повторён весь итоговый backend-набор.

Изоляция фактически проверена: `network=none`, rootfs readonly, staging/model-quality/
models mounts readonly, `/tmp` в tmpfs; production env и доступ к базе не передавались.
API healthcheck отключён только у тестового child, в котором API-сервис не запускается.
Результат получен из реального JUnit, переданного через лог отдельного контейнера
до потери tmpfs после завершения. Сырой XML и log сохранены локально в игнорируемом
`output/as16-integration/`, в Git не включаются.

После опыта API v3, API v4 и worker оставались healthy; их image ID и StartedAt
совпали с исходным снимком. Production-контейнеры не перезапускались, реальные
записи/решения/модели не изменялись. Это fixture unit/integration tests; выполнение
SQL-миграции в настоящей БД или приёмка пользователем ими не заявляются.

## Итоговый полный прогон в фактически собранном candidate — 06.10.2026

Фактический candidate image
`sha256:034a9d67b49adb5565a1af9454706a104721e21160d725db32a31f796ff20193`
содержит API commit `0437e98e0570e0b74a1705b5d3095d32f038ddda`.
Полный набор исходников и тестов заморожен на
`9010886489d8201a0a7c8310c1010e5f2d229515`: API/config bytes между двумя
коммитами совпадают; второй добавляет отдельный оператор развёртывания и 22 tests.
Это прогон в построенном образе кандидата, а не только в прежнем runtime image.

Результат — **216 passed, 1 skipped, 3 subtests passed, 2 warnings**, exit 0,
**12.36 с**. JUnit: 220 cases, failures/errors 0, skipped 1, duration 12.355 с.
Среда: Linux Python 3.11.16. Все 100 Git objects выгружены с LF из точного commit
и независимо сверены в remote staging; runner дополнительно хешировал 95 своих
tracked файлов, пять research-файлов проверяются отдельным scientific suite.
Изоляция повторно проверена: network none, rootfs и mounts readonly, tmpfs 4 GiB,
production env отсутствует, healthcheck выключен только у test child.

Точный контейнер, хеши исходников/XML/log/metadata, версии и проверка неизменности
production image/StartedAt сохранены в
[backend-tests-candidate901.json](backend-tests-candidate901.json).
Первоначальные 182 tests выше и промежуточные 194 tests — исторические результаты.
Linux skip и два прежних предупреждения сохранились; Windows JobObject case
выполнен отдельно на Windows, как указано выше. Это не миграция/restore БД.

После этого прогона исправлен исключительно оператор backup manifest: API bytes
и candidate image не менялись. Обновлённый оператор с семью дополнительными
cases проверен отдельно: **37 passed**, из них 29 rollout и 8 weather,
4.343 с. Доказательство и граница этого targeted-прогона:
[v4-rollout-safety-tests.json](v4-rollout-safety-tests.json).
