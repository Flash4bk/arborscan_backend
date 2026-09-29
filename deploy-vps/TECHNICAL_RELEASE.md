# AS-16: технический кандидат перед редизайном

Состояние 30.09.2026: технический кандидат интегрирован и развёрнут;
дополнительная проверка независимого восстановления продолжается.
Это не окончательная release-подпись и не приёмка пользователем.
Дополнительная проверка AS-14 не переоткрывает прежние успешные проверки.

## Версии и интеграция

База `22b2a6db1cd5868e96b3b25f3204180fb2f7971f`; ветка `codex/technical-release`.
Проверено через merge-base: все локальные ветки measurement-core, server-history,
model-quality, geometry-validation, report-export и reliability-ops уже являются
предками кандидата. Повторных merge/cherry-pick не было.
Origin main при проверке: `60329e18586cb0351c18a909289de01d99843f85`.
GitHub branches/main protected=false, rules/branches/main=[]; теги отсутствовали.
Перед будущим push эти сведения необходимо проверить повторно.

Код кандидата: `eaa39f415f6de434773cc099d02f407d435c86b1`.
Продуктовые изменения: version 1.1.0+2; серверные алгоритмы/контракты не менялись.
Инструмент восстановления теперь принимает явные backup и pinned image manifest.
Новая база/миграции для релиза не требуются. β/DBH/пористость не подставлялись.

## APK и подпись

ApplicationId `com.example.arborscan_app` сохранён.
До обновления на S24: versionName=1.0.0, versionCode=1, DEBUGGABLE.
Сертификат Android Debug, SHA256:
`68ff9861f52fdb22744b30ee27092008097f9eff4ead7eb38f80e662115f5b97`.
Кандидат: **debug**, versionName=`1.1.0-tech.eaa39f4`, versionCode=2.
SHA256 APK: `47adf864bd2fec9375cccaa6106ab06b40e12470d2c75e8efc7270472841345f`.

Сборка: `flutter build apk --debug --no-pub --build-name=1.1.0-tech.eaa39f4 --build-number=2`.
Установлен `adb -s R5CY40HNVCP install --no-streaming -r ...` → Success;
фактическая новая версия проверена dumpsys. Сертификат до/после совпал.
Приложение не удалялось, данные/аккаунт не очищались. Инвентаризация файлов до/после
не обнаружила потери сохранённых пользовательских данных; изменения служебных
res_timestamp/profileInstalled ожидаемы. Секреты/сами данные в отчёт не включены.
APK также обновлён на отдельном emulator-5582.

Gradle release в текущем проекте также использует debug signingConfig.
Это **не окончательный подписанный release APK**. Новый ключ не создавался.
Для перехода на другой ключ нельзя просто установить APK поверх текущего с тем же
applicationId. До отдельного решения о подписи безопасное обновление — прежним
ключом; uninstall/clear для обхода конфликта запрещены. Неотправленные локальные
черновики не гарантируются серверной копией.

## Выполненные проверки кандидата

- Flutter: 56 passed. Android testDebugUnitTest: BUILD SUCCESSFUL.
- Analyze вывел 195 строк диагностик с дублями; уникальных 105, ровно прежний
  baseline (0 новых, 0 errors, 7 уникальных warnings). Exit code анализатора 1.
- Backend в отдельном контейнере, без production env: 107 passed, 1 skipped,
  3 subtests, 2 прежних предупреждения. Пропущенный Windows Job Object test
  отдельно прошёл на Windows. Первые попытки обнаружили только неполный тестовый
  mount/readonly runtime; фикстуры и временный runtime исправлены, suite повторён.
- На S24 открыты прежние локальная DEMO-запись и обе серверные DEMO AS-14 версии;
  значения 4.8/2.1 и 9.6/4.2 м сохранены. Из последней версии восстановлен эталон,
  изменена высота эталона 2 → 3 м, сохранена **отдельная третья тестовая версия**.
  После force-stop и запуска прочитаны 14.3971/6.2987 м; прежние версии остались.
  Это проверка арифметики/сохранения синтетики, не измерительной точности.
- Через Samsung SAF сохранены два реальных PDF, выгружены именно новые файлы
  от 29.09 (с суффиксом `(1)`, а не старые PDF). Проверены 8 страниц, значения,
  фото/разметка сравнены с прежними fixtures. Все страницы отрендерены и визуально
  проверены. Меню «Поделиться» открыто/закрыто без выбора адресатов; старый
  локальный PDF открыт через системный просмотрщик.
- Результаты агента не называются приёмкой пользователем. Сеть S24 не отключалась.

## Production и условие переключения

На старте /opt/arborscan чистый, commit `5d8176a22fddd6634c266a1ed09f1845019065c5`.
API/worker: `arborscan-api-v4:reliability-691fb77`, image
`sha256:afb6c57845667f9aff8a1e9236fbe7b81255f2c8ccc53ccc5b7b10cb77abb26e`.
V3: `sha256:5a0b1d63e6c5eb13d14af84590dafebf900f59156613b987000ae8b6938f563b`.
Кандидат собран в отдельном каталоге /home/arborscan/technical-release-eaa39f4,
`arborscan-api-v4:technical-eaa39f4`. Это исходное состояние до переключения;
фактически выполненный rollout описан ниже. API v3, HTTPS, модели и реальные
решения модерации не изменялись. До переключения проверены изолированный smoke,
свежая копия, отсутствие активного обучения и подготовлен rollback.
Откат кода обязан сохранять текущую БД; старый дамп поверх production не применять.

## Независимый набор восстановления: подготовка

Исходный проверенный data set: `D:/ArborScanBackups/20260929T031924Z`.
Native PostgreSQL 17.6/роли/ACL, Storage, модели, runtime/configuration включены
в этот набор; локальные несинхронизированные данные телефона — нет.
Auth приложения — собственный v3/public.users/auth_sessions, REST — PostgREST,
файлы — отдельный Storage API. Один PostgreSQL dump не заменяет этот стек.

Создан серверный архив всех шести runtime images (API используется и worker):
`/home/arborscan/recovery-releases/reliability-691fb77/images.tar.zst`.
SHA256 `3f8d154c102db8ee06a0d3fd7e72df67d4b423b9c7a190d29ff54c3c02ad5140`.
Размер около 2.39 GiB. Независимое место:
`D:/ArborScanBackups/releases/reliability-691fb77` с наследованием приватного ACL.
Там сохранён полный source archive eaa39f4. Перенос images пока НЕ завершён:
.partial/проверенные блоки не равны готовому внешнему набору.

Для проверки создан отдельный вложенный Docker daemon без host docker socket,
без production bind mounts; его image store/volumes изначально пусты.
Это чистое контейнерное окружение **на том же VPS**, не новая виртуальная машина.
Bootstrap Docker engine/Linux kernel — явные зависимости. Docker 27 отличается
от host Docker 29 способом идентификации OCI images. Использован совместимый
Docker 29.8.0-dind, digest 5efed980cba3fc126cf54e21a5a6ff8849d05b6e0623d6e7612f48e9cd6cd17e.
До полной проверки внешнего архива запуск восстановленного стека не засчитывается.
Потребуются загрузка именно внешнего архива, изолированная сеть, новые lab secrets,
инвалидация старых сессий/очередей только в копии, HTTP-сценарии и restart.
Внешние OAuth/PlantNet/Vault требуют настоящих реквизитов и не объявляются автономными.

Расписания AS-14 сохранены. Пока Windows выключен/пользователь вышел, новые backup
остаются только на VPS. Большой архив версии не должен копироваться ежедневно:
backup связывается с immutable release archive по image IDs и SHA256.
AS-09/10 отложены, AS-11/13 открыты, реальное обучение AS-05–07 не заявлено.
Следующий продуктовый этап после технического релиза — AS-15.


## Дополнение к проверкам перед переключением

Кандидат `sha256:3fe28a5b7eb5716a71e984908520c31fff9ac325a9c5885c5e102e93996ac14a`
успешно проверен в прежнем изолированном стеке AS-14: сохранение/чтение, права,
две новые версии, модерация, bounded inference, отказ worker до обучения, restart.
Дополнительно прошли HTTP smoke истории (EXIF, конфликт, immutable retry) и
качества (независимая порода/снимки/отказ обучения на недостаточных данных).
Исправлена жёсткая production-привязка регистрации в smoke: SMOKE_AUTH_BASE
позволяет явно направить её в изолированный v3. Сам продуктовый auth не менялся.
Эта проверка использовала прежний изолированный стек, не чистый внешний restore.

Свежая копия перед переключением: `/home/arborscan/ops-backups/20260929T205205Z`,
backup-exit=0, native dump 52094101 байт, оба COMPLETE. SHA256 повторно проверены.
Подготовленный rollback в `/home/arborscan/as16-predeploy/rollout` проверяет
прежние image IDs, хеши compose/env/моделей, неизменность v3 и отсутствие active jobs.
Команды переключения/отката (не подменяют фактический результат):

```sh
python3 /home/arborscan/ops-tools/ops_technical_rollout.py deploy --plan /home/arborscan/as16-predeploy/rollout
python3 /home/arborscan/ops-tools/ops_technical_rollout.py rollback --plan /home/arborscan/as16-predeploy/rollout
```

Откат только API/worker, без восстановления старой БД и без удаления новых записей.
Изолированный emulator-5582 с выключенными Wi-Fi/data сохранил локальный PDF через
SAF, восстановил запись после force-stop; отмена нового сохранения показала
«Сохранение отменено». Сеть личного S24 не отключалась, его аккаунт не заменялся.

Для нового образа не требуется второй ежедневный архив на гигабайты:
`ops_image_delta.py` сохраняет проверенный overlay к immutable base archive.
Фактический overlay кандидата содержит 9 изменённых членов, **52420 байт**.
Его manifest фиксирует SHA исходного base и overlay; apply отказывается от
повреждённого base и не перезаписывает существующий результат. Три теста прошли.
Это транспорт Docker-save содержимого, не слой поверх живого Docker daemon.
Фактическое применение overlay и docker load в отдельном Docker 29.8.0
выполнены: получен точный image ID 3fe28a5b7eb5716a71e984908520c31fff9ac325a9c5885c5e102e93996ac14a.
Это проверка формата архива; она пока не заменяет полного восстановления
из завершённого внешнего набора.


## Фактически выполненное переключение — 29.09 UTC / 30.09 Minsk

Проверены неизменность origin/main и отсутствие branch protection/rules;
main обновлён fast-forward `60329e1 → 96eaa99`, push успешен.
API v4 и worker переключены на точный проверенный образ
`sha256:3fe28a5b7eb5716a71e984908520c31fff9ac325a9c5885c5e102e93996ac14a`
(тег `arborscan-api-v4:technical-eaa39f4`, OCI revision eaa39f415f6de434773cc099d02f407d435c86b1).
Никаких миграций, обучения или активации модели не выполнялось.

Первая попытка Compose остановилась **до изменения контейнеров** из-за обязательной
интерполяции MODEL_QUALITY_IMAGE. Инструмент исправлен: переменной присваивается
проверенный pinned image ID, итоговый overlay отдельно фиксирует API/worker.
Повтор выполнен успешно. API/worker healthy, restarts=0, ERROR/Traceback в
проверенных startup-логах 0. V3 image и StartedAt, env и хеши моделей совпали с планом.
Оба публичных HTTPS health — ok; reports/capabilities/corrections/ML без токена — 401.
Проверены readiness/HTTP-сценарии на изолированных синтетических аккаунтах.
Post-production quality smoke прошёл. Первый параллельный history smoke завершился
ошибкой с успешным finally-cleanup, без точной диагностики причины; последовательный
повтор прошёл целиком. Не приписываем первоначальному сбою неподтверждённую причину.
В smoke добавлена безопасная диагностика типа/строки/HTTP-кода без тел и токенов.
Доказательства: evidence/as16/production-*. Фикстуры удалены; реальные решения не менялись.

На эмуляторе первый force-stop сразу после нажатия SAVE, **до подтверждения**,
оставил пустой файл document provider. Этот файл не засчитан. При повторе дождались
«PDF сохранён», выгрузили новый файл 48662 байта, проверили 4 страницы и визуальный
рендер; после следующего force-stop локальная запись открылась снова без сети.
Принудительное завершение процесса во время записи внешнего SAF-файла может оставить
неполный файл; ложного подтверждения сохранения не было. Неполный DEMO-файл оставлен
только на тестовом эмуляторе, не затрагивает личные данные S24.

Проверенные отдельные PDF этого прогона — 12 страниц (8 Samsung + 4 emulator).
Большой архив образов всё ещё передаётся во внешнее хранилище и не имеет COMPLETE;
его готовность и чистое восстановление нельзя выводить из успешного rollout.


## Маркер технического релиза

Тег `v1.1.0-tech.1` обозначает проверенный технический код и эксплуатационные
инструменты этой интеграции. Точный коммит документации определяется командой
`git rev-parse v1.1.0-tech.1^{commit}`. Код работающих API/worker и APK — eaa39f4,
последующие коммиты меняют проверочные инструменты и документацию.
Это не метка завершения дополнительного внешнего восстановления, AS-13 или AS-15.
Для повторной диагностики сверять OCI revision и image ID, а не Git checkout
/opt/arborscan: исходный checkout VPS намеренно сохранён без перезаписи.

Архив исходников eaa39f4 во внешнем защищённом каталоге:
SHA256 `3f16a921f63521d566c1187472ecd5c605976bbc4d1a8da88327da4705f8d6ef`.
SHA256 небольшого image overlay:
`06e02add4c754a8c127079bc9a7ab0afb2480988816e2a9d985d54c8212b6f0b`.
В подкаталоге tooling сохранены инструменты восстановления и отдельный SHA manifest.
Полный images.tar.zst считается доступным снаружи только после проверки целого
файла; наличие chunks или успешная загрузка образа на VPS этого не доказывает.


## Восстановление в пустом Docker engine — фактический опыт 29.09 UTC

Создан новый Docker 29.8.0 с отдельным volume, без host Docker socket,
без bind mounts production, без опубликованных наружу портов. Внешняя сеть
контейнера internal. Это privileged вложенный контейнер на существующем ядре VPS,
не новая виртуальная машина. Bootstrap: публичный Docker digest выше, Alpine
python3 3.14.7-r1 и zstd 1.5.7-r2; доступ к этим bootstrap-репозиториям — явная
зависимость. Приложения/модели не скачивались из живого Docker image store.

Использован транспортный кэш **архивов**, SHA-манифест данных взят с Windows и
сопоставлен побайтно; каждый файл кэша проверен перед копированием. Архив образов
также проверен по фиксированному SHA. Это не физическая обратная передача всего
набора с Windows. Пока полный архив образов на Windows не проверен целиком,
доказана работа процедуры с архивами, но дополнительный off-site критерий не закрыт.

- Загружены шесть образов base archive и образ кандидата, восстановленный overlay.
- Native PostgreSQL/роли/ACL восстановлены; старые сессии инвалидированы и прежние
  queued/running задания помечены failed **только в копии**. Созданы новые lab secrets.
- Восстановлены 2239 файлов, 827887708 байт; связи 4 отчётов проверены.
- Storage: 2221 объект, 759097658 байт, импорт 67.853 секунды.
- HTTP: вход, изоляция владельцев, 4 старых отчёта, 5 контуров, фото/маски по SHA,
  ограниченный inference, 2 новые тестовые версии, отказ/принятие/новый draft.
- Worker отклонил invalid snapshot до обучения; активная модель не изменилась.
- После restart семи сервисов вход/отчёты/контуры/worker прочитаны повторно;
  health API/v3/DB/worker корректен, внешний доступ API к production заблокирован.

Опыт от начала подготовки проверенного кэша/импорта образов 21:22:55 UTC
до конечного аудита 21:39:38 UTC занял **16 мин 43 сек**, включая диагностику.
Возраст backup в начале этого интервала — **18 ч 03 мин 31 сек** относительно
его запуска 03:19:24 UTC. Bootstrap engine выполнен раньше; длительная внешняя
передача также не включена. Это измеренные фазы опыта, не полный гарантированный
RTO/RPO и не время развёртывания нового сервера с нуля.

Найдено и исправлено различие BusyBox/GNU sha256sum: --quiet заменён захватом
stdout/stderr при сохранении -c и ненулевого exit при повреждении. Первый импорт
Storage остановился на начальном запуске; повтор прошёл. Добавлена отдельная
read-only проверка готовности Storage с ограничением 60 секунд, без произвольных
повторов изменяющих запросов. Проверка задержанного запуска на 10 секунд прошла:
2221 объект импортирован, elapsed 90.98 секунды (импорт 71.415).
Никакие из этих исправлений не меняют продуктовый API или APK.

Доказательства: evidence/as16/nested-app-checks.txt, engine-audit.json,
storage-readiness.txt, integrity-gate.txt. Служебные секреты и архивы в Git не входят.

### Команды воспроизведения в отдельном Docker 29.8

Команды выполняются **внутри отдельного engine**, содержащего только лабораторные
volumes. /inputs/backup — проверенные файлы внешнего backup; /inputs/release —
архив версии и tooling. Не направлять этот Docker context в production daemon.
Первоначальный image store и volumes должны быть пусты; bootstrap должен быть
заранее подготовлен и изолирован от production-сетей.

```sh
cd /inputs/release
sha256sum -c SHA256SUMS
zstd -qdc images.tar.zst | docker load
python3 tooling/ops_image_delta.py apply images.tar.zst candidate-eaa39f4.delta.tar.gz candidate.tar
docker load -i candidate.tar
python3 tooling/ops_verify_restore.py /inputs/backup/application.tar /inputs/backup/restored
python3 tooling/ops_recovery_stack.py --backup /inputs/backup --images-manifest pinned-images.json > /tmp/recovery-result.json
LAB=$(python3 -c 'import json; print(json.load(open("/tmp/recovery-result.json"))["root"])')
python3 tooling/ops_recovery_files.py "$LAB"
```

Для восстановления исходной версии продолжить ops_recovery_start.py без замены.
Для проверенного технического кандидата перед стартом заменить api/worker image
в **созданном lab compose**, сохранив остальные поля и приватные credentials:

```sh
python3 - "$LAB" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
assert root.parent == pathlib.Path('/home/arborscan') and root.name.startswith('as14-recovery-')
p = root / 'compose.private.json'
c = json.loads(p.read_text())
assert c['networks']['default']['internal']
for service in ('api', 'worker'):
    c['services'][service]['image'] = 'sha256:3fe28a5b7eb5716a71e984908520c31fff9ac325a9c5885c5e102e93996ac14a'
p.write_text(json.dumps(c))
PY
python3 tooling/ops_recovery_start.py "$LAB"
docker compose -f "$LAB/compose.private.json" cp tooling/ops_recovery_smoke.py api:/tmp/recovery-smoke.py
docker compose -f "$LAB/compose.private.json" exec -T api python /tmp/recovery-smoke.py
python3 tooling/ops_recovery_audit.py "$LAB"
docker compose -f "$LAB/compose.private.json" restart
python3 tooling/ops_recovery_start.py "$LAB"
docker compose -f "$LAB/compose.private.json" exec -T api python /tmp/recovery-smoke.py
python3 tooling/ops_recovery_audit.py "$LAB"
```

После опыта остановить именно lab compose через `stop`, не удаляя volumes.
Восстановление production из дампа этими командами не выполнялось и не требуется
для отката технического API. Откат production — ранее подготовленный rollout.py
rollback с точными прежними образами, без замены текущей базы.


### Связь ежедневных данных с архивом версии

В защищённом корне D:/ArborScanBackups хранится RELEASE_INDEX.json: фактические
Image из containers.private.json у API/worker указывают на immutable release.json.
Индекс содержит базовый образ afb6c57 и кандидат 3fe28a5 (base + overlay).
Новая неизвестная версия не считается восстановимой без своего архива.
ops_verify_release_bundle.py проверяет обе записи контейнеров, COMPLETE, пути,
все SHA артефактов и соответствие overlay базовому архиву. Восемь целевых тестов
проверили общий архив без повторного копирования, base/overlay, неизвестный image,
незавершённую передачу, повреждение, пропуск файла, неверный base и выход за корень.

```powershell
python -B deploy-vps/ops_verify_offsite.py D:/ArborScanBackups/20260929T031924Z
python -B deploy-vps/ops_verify_release_bundle.py D:/ArborScanBackups/20260929T031924Z D:/ArborScanBackups
```

Обе проверки обязательны: SHA данных не доказывает наличия runtime image.
Процедура читает уже сохранённый архив; повторной ежедневной передачи images нет.
Для нового деплоя сначала архивировать новую версию/overlay, проверить загрузку,
обновить индекс и повторить проверку. Рабочие расписания backup/off-site не менялись.

Отдельно скопирован host-config-readable.tar: 21 читаемый файл Nginx/systemd/
deploy-hook, SHA256 474c574665e37d600dbe54c4cf4e38fc4506a7dddcae5f3d9990c64ebc1d32b5.
Проверено отсутствие приватных ключей в этом архиве. Ключи TLS и учётные данные
Certbot **не включены**; новый публичный HTTPS требует перевыпуска сертификата
либо предоставления этих материалов через отдельное защищённое хранение.
Это явное ограничение: запуск новой публичной VM/HTTPS не проверен. Существующие
сертификаты, Nginx, таймеры и API v3 не менялись.
