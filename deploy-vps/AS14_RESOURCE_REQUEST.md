# AS-14 — один запрос ресурсов для внешней копии и восстановления

Запрос ниже отложен явным заданием08.10: используется только существующая Windows-папка. Новые receiver/VM/domain не запрашиваются. Текущие факты: [TECHNICAL_RELEASE_COMPLETION.md](TECHNICAL_RELEASE_COMPLETION.md).

Подготовлено 08.10.2026. Основание — существующий
[общий план](ARBORSCAN_MASTER_PLAN.md), версия 1.25, и
[FINAL_RELEASE_AS14.md](FINAL_RELEASE_AS14.md). **AS-14 остаётся открыт:**
независимый получатель, чистая VM и тестовый HTTPS ещё не назначены.
Запрос первоначально подготовлен по прежним результатам; ниже добавлены свежие
read-only размеры от 08.10. Backup/restore ради запроса повторно не запускались.

## Что предоставить одним комплектом

| Ресурс | Минимальные требования и разрешения |
|---|---|
| Независимый круглосуточный получатель | Отдельный от production VPS `31.57.170.88`, Supabase-проекта и Windows-ПК Linux/SSH-хост; Python 3.11+, POSIX filesystem с `flock`, примерно 1 GiB RAM; **не менее 32 GiB свободного приватного места, предпочтительно 64 GiB**. Отдельная SSH-учётная запись с доступом только к backup root и receiver script. Право первоначально установить публичный скрипт и dedicated public key. Docker, production API и DB-права получателю не нужны. |
| Чистая изолированная VM | Linux amd64, **2 vCPU, 8 GiB RAM, около 100 GB диска**, SSH и разрешение установить Docker/Compose/Nginx/Certbot. Отдельная VM либо право создать её на уже разрешённом hypervisor; приватный доступ к независимому архивному комплекту и bootstrap-материалам. Без production mounts, живой базы и её ключей в работающей lab. |
| Тестовый HTTPS | Отдельный hostname/IP для этой VM; право настроить **только его** DNS и сертификат; входящие 80/443 для HTTP-01 ACME и исходящий доступ к необходимым bootstrap/ACME-службам. Production DNS, `arborscan-ip`, маршруты и HTTPS не переключаются. |

Новый VPS, аккаунт или платный тариф покупать не требуется. Нужны назначенные
разрешённые ресурсы, а не повторная копия на том же production VPS.
Владелец сообщает адреса, SSH user/port, private paths и доступные права.
Пароли, private keys, provider tokens и содержимое конфигурации в чат или Git
не передаются. SSH host fingerprint проверяется доверенным каналом;
одного неподтверждённого `ssh-keyscan` недостаточно.

## Основание ёмкости

Исторический аудит **07.10.2026, 12:37–12:40 UTC** в
[FINAL_RELEASE_AS14.md](FINAL_RELEASE_AS14.md) измерил по существующим manifests:

- 14 VPS-наборов: **13 604 705 463 байта**;
- три общих immutable runtime-архива: **3 781 384 242 байта**;
- вместе: **17 386 089 705 байт / 17,386 GB**;
- один полный package, включая runtime: **4 760 932 219 байт**;
- retained sets + полный staging package + резерв 2 GiB:
  **24 294 505 572 байта / 24,295 GB / 22,626 GiB**.

Поэтому 32/64 GiB — свободное место именно под приватные данные, отдельно от ОС
и чужих файлов. Это расчёт по тогдашним данным, не гарантия ёмкости при их росте.
Native PostgreSQL уже входит в сумму; повторно прибавлять его нельзя.
Прежние SHA/readback/file restore подтверждены в
[windows-offsite-fresh-20261007.json](evidence/release-readiness/windows-offsite-fresh-20261007.json).
Размеры и эта ссылка не являются новой проверкой целостности или новой VM.

08.10, **10:37 Minsk** read-only инвентаризация актуальных 14 COMPLETE-наборов
уточнила размеры: manifest payloads **13 605 154 553 B**, shared runtime
**3 781 384 242 B**, вместе **17 386 538 795 B**; latest full staging package
**4 761 003 132 B**, с reserve2GiB — **24 295 025 575 B / 22,626 GiB**.
Минимум32/предпочтительно64GiB не изменился. Это суммирование размеров, **не
повторная SHA-проверка всех14 наборов**; fresh latest отдельно проверен в
[GOOGLE_SECURITY_ROLLOUT.md](GOOGLE_SECURITY_ROLLOUT.md).
[Текущая metadata](evidence/google-security-rollout/readiness.json).

После разрешённого rollout **08.10.2026** текущий v3 —
`sha256:60c6cf1fc185e41aaf6275522df851b60f142887554685ce7aec49b3f702a990`,
подготовленный policy6284f93 поверх live base5a0b1d63.
Отдельный приватный Docker-архив этого image действительно создан:
**787 500 544 B**, SHA-256
`82f59d586d20b1e232175a53c0260ce9a68b5fc88da5f9f7d931d2096b1594a9`.
Он **не входит** в приведённые выше прежние 14 наборов/shared runtime.
Для хранения нового архива и ещё одной его staging-копии добавить
**1 575 001 088 B**: итог **25 870 026 663 B / около 24,09 GiB**.
Минимум32/предпочтительно64GiB свободного места не изменился.
Это дополнение расчёта ёмкости и наличие локального приватного архива,
не выполненная внешняя передача или проверка восстановления нового v3.
Фактические image/claims/index после применения:
[postdeploy](evidence/google-security-rollout/postdeploy.json).

Для DR именно **нового текущего v3** необходимо включить exact archive,
фактический приватный Compose override/effective environment и подготовленный
Google unique subject index с проверкой применения в восстановленной базе.
Прежний daily bundle и его `RUNTIME_DEPENDENCIES.json` автоматически не
подтверждают включение нового v3/config/index. Архив/конфигурацию/SQL связать с
точной версией и проверить после назначения ресурсов; старый комплект сохранить.
Production backup/runtime mapping этим запросом не меняется.

## Приватные поля и подготовленные команды

Из [offsite-config.example.json](offsite-config.example.json) приватно настроить
`destination_id`, `host`, `user`, `port`, `remote_root`, `receiver_script`,
`identity_file`, `known_hosts_file`, `is_independent` и `always_on`.
Канонический путь на отправителе:
`/home/arborscan/.config/arborscan/offsite.private.json`.
Config/key/known_hosts — mode 600, private parent — mode 700.
Dedicated private key остаётся на отправителе; на получателе возможен
`restrict,command=...ops_offsite_receiver.py --root ...` без обычного shell.

Подготовлены [ops_offsite_push.py](ops_offsite_push.py),
[ops_offsite_receiver.py](ops_offsite_receiver.py),
[ops_verify_release_bundle.py](ops_verify_release_bundle.py) и
[ops_verify_restore.py](ops_verify_restore.py). Перед применением агент проверит
их установку и SHA из одного точного checkout. CLI ниже — шаблоны после назначения
ресурсов; имя выбранного проверенного COMPLETE-набора подставляется приватно.

```sh
python3 /home/arborscan/ops-tools/ops_offsite_push.py push \
  --config /home/arborscan/.config/arborscan/offsite.private.json \
  --backup /home/arborscan/ops-backups/VERIFIED_SET \
  --release-root /home/arborscan/recovery-releases \
  --receipt-root /home/arborscan/ops-backups/offsite-receipts
python3 /home/arborscan/ops-tools/ops_offsite_push.py verify \
  --config /home/arborscan/.config/arborscan/offsite.private.json --name VERIFIED_SET
python3 /home/arborscan/ops-tools/ops_offsite_push.py recover \
  --config /home/arborscan/.config/arborscan/offsite.private.json --name VERIFIED_SET \
  --destination /home/arborscan/offsite-recovery-NEW
```

На новой VM агент импортирует exact runtime archives и проверит manifest/SHA,
затем выполнит `ops_recovery_stack.py --backup ... --images-manifest ...`,
`ops_recovery_files.py`, `ops_recovery_start.py`, `ops_recovery_smoke.py`,
restart и `ops_recovery_audit.py`. Исторические defaults snapshot/image ID
использовать нельзя: текущие API/worker, изоляцию и пути сначала сверить.
Проверка отдельного HTTPS включает Nginx, штатную проверку hostname/цепочки TLS,
reload-hook и `certbot renew --dry-run --run-deploy-hooks` для тестового сертификата.
TLS private keys/Certbot account не возникают из PostgreSQL/Storage dump;
bootstrap и необходимые внешние загрузки учитываются отдельно.

**Это подготовленные механизмы, а не выполненный независимый перенос.**
Ежедневный внешний sender/выбор COMPLETE-набора/lock/retry ещё не подключены;
плановый перенос при выключенном Windows не проверен. После первого фактического
push/readback/recover агент подключит и проверит разрешённую ежедневную передачу.
**Receiver retention/GC не настроены:** CLI не удаляет completed sets или shared
blobs. Production policy на 14 наборов не означает автоматические 14 копий на
получателе; отдельная безопасная политика требует подготовки и проверки.
