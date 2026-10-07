# AS-14 в финальном AS-16: ресурсы внешней копии и новой VM

Дата аудита: **07.10.2026, 12:37–12:40 UTC**. Рабочая основа:
`codex/final-release` от main `10b54e77ae77fa7a51621bc1902889ce4d2a8ef6`.
Общий план — [ARBORSCAN_MASTER_PLAN.md](ARBORSCAN_MASTER_PLAN.md), версия 1.23
на начало пакета. Этот документ фиксирует AS-14 и не создаёт второй план.

**Статус: реализация независимого транспорта подготовлена; настоящий always-on
перенос, новая VM и её HTTPS заблокированы отсутствием назначенных ресурсов.**
Проверенная Windows-копия сохранена. AS-14 целиком не закрыт; принятие этих
ограничений пользователем не заявлено.

## Что проверено сейчас

Выполнены только read-only проверки Git, проектных конфигураций/SSH-метаданных,
размеров уже существующих архивов, systemd и Windows Scheduler. Backup, перенос,
проверка SHA всех больших архивов, восстановление ключа и прежний lab повторно
не запускались. Production, его DNS/HTTPS, БД, модели и контейнеры не менялись.
Значения секретов, connection strings и пользовательские записи не выводились.

| Ресурс / проверка | Фактический результат этого аудита |
|---|---|
| SSH production `arborscan@31.57.170.88` | Read-only доступ работает, штатная проверка host key сохранена |
| VPS backup service | `Result=success`, `ExecMainStatus=0`; последний запуск 07.10 03:21:52–03:34:25 UTC |
| VPS backup timer | `active/waiting`; следующее срабатывание 08.10 03:20:05 UTC |
| VPS COMPLETE-наборы | 14; последний `20261006T195518Z`, 29 файлов outer manifest |
| Canonical offsite private config на VPS | Не существует; проектных offsite config candidates не найдено |
| Dedicated offsite identity / known_hosts на VPS | Оба предусмотренных файла отсутствуют |
| Второй проектный SSH-хост | В проверенных локальных/VPS проектных SSH-конфигурациях не найден |
| Windows назначения | Действующая закрытая `D:\ArborScanBackups`; 16 исторических COMPLETE-наборов сохранены |
| Windows task | `Ready`, `LastTaskResult=0`; последний запуск вручную агентом 07.10 09:53:32 UTC |
| Новое календарное Windows-срабатывание | После прежнего audit ещё не наступило; NEXT 08.10 05:00 UTC / 08:00 Europe/Minsk |
| Windows требования | `Interactive`, RestartCount=3/PT15M; новый перенос требует включённого ПК и вошедшего пользователя |
| Проектный тестовый HTTPS | Две читаемые проектные Nginx-конфигурации содержат только production server_name; отдельное тестовое имя не найдено |
| Certbot метаданные | Виден проектный renewal `arborscan-ip.conf`; отдельный recovery-test certificate не обнаружен |
| Чистая отдельная Linux VM | Разрешённый адрес, SSH-доступ и право подготовки не предоставлены и в проектных конфигурациях не найдены |

Этот inventory ограничен ресурсами ArborScan. Он не утверждает, что у владельца
нет других серверов, доменов или личных аккаунтов. Незаявленные ресурсы не
использовались. Отсутствие нового календарного запуска не выдаётся за новый
успешный запуск: утренний Windows-run 07.10 отказал, последующие успешные запуски
были ручными. Его причина и история сохранены в
[OFFSITE_READINESS.md](OFFSITE_READINESS.md).

## Объём и ёмкость независимого receiver

Размеры ниже получены повторным `stat` файлов уже существующих manifest-наборов;
это новый аудит ёмкости, **не повторная проверка содержимого по SHA**. Доказательства
ранее выполненных SHA/readback/restore находятся в
[windows-offsite-fresh-20261007.json](evidence/release-readiness/windows-offsite-fresh-20261007.json).

| Состав | Байты | Десятичные GB |
|---|---:|---:|
| Все 14 VPS-наборов: файлы, перечисленные outer manifests | 13 604 705 463 | 13,605 |
| Минимальный / максимальный один набор | 927 004 164 / 979 545 362 | 0,927 / 0,980 |
| Текущие runtime: 3 уникальных immutable архива | 3 781 384 242 | 3,781 |
| 14 существующих наборов + один общий комплект runtime | 17 386 089 705 | 17,386 |
| Один свежий полный package, включая runtime и служебные файлы | 4 760 932 219 | 4,761 |

Свежий native PostgreSQL комплект занимает **52 163 691 байт** и уже входит в
outer сумму. Его не нужно второй раз прибавлять к ёмкости. Runtime включает
2 563 869 234 байта общего base, 608 791 552 байта API v4 и 608 723 456 байт worker.
Для двух текущих image ID есть четыре ссылки на эти три архива: общий base
хранится один раз. Original `RUNTIME_DEPENDENCIES.json` покрыт outer SHA и сохраняет
исходные absolute paths; отдельная relocation map не переписывает этот документ.

Для планирования без предположения о дедупликации данных: **14 текущих наборов
максимального нынешнего размера + shared runtime = 17,495 GB**. Консервативный
рабочий минимум по фактическим 14 наборам, ещё одному полному staging package и
резерву 2 GiB — **24 294 505 572 байта / 24,295 GB / 22,626 GiB**.

Запрос назначения: **не менее 32 GiB свободного места именно в private data
filesystem; предпочтительно 64 GiB свободного места**, отдельно от ОС и чужих
данных. Например, уже разрешённый Linux-хост с примерно 100 GB общим диском может
дать такой запас после проверки его фактического свободного пространства.
Это расчёт по нынешним данным, не выбор платного тарифа и не обещание постоянной
ёмкости при росте Storage или смене образов. Для пассивного SSH receiver достаточно
Python **3.11+**, обычного POSIX filesystem с `flock`, стабильного SSH и порядка
1 GiB RAM; Docker, доступ к Supabase и production API ему не нужны.

**Лимит 14 относится к проверенной VPS policy**, где новый комплект проверяется и
публикуется до dependency-safe ротации. Sender/receiver намеренно не удаляют
completed sets или shared objects. Внешняя копия с этими инструментами растёт,
пока отдельная разрешённая retention/GC policy не подготовлена и проверена.
Нельзя заявлять «receiver хранит автоматически ровно 14» или удалять blobs по
возрасту: на них могут ссылаться другие retained sets. При новом receiver сначала
не удалять внешние наборы, наблюдать место и согласовать отдельную политику;
локальные 14 и исторические Windows-копии сохраняют существующую политику.

## Готовые механизмы и безопасная настройка

Существующие [ops_offsite_push.py](ops_offsite_push.py),
[ops_offsite_receiver.py](ops_offsite_receiver.py) и
[offsite-config.example.json](offsite-config.example.json) уже проверены на
изолированных CLI/RPC fixtures. Они требуют independent/always-on назначения,
отказывают production/localhost и тому же machine-id, проверяют private paths,
strict host key и SHA всех частей native/Storage/config/runtime комплекта.
Повтор использует SHA-addressed объекты и проверенный prefix; потерянный ответ
записи не вызывает слепой повтор. Receipt создаётся после чтения всех внешних
байтов обратно. Прошлые Windows/Linux suite результаты и конкретные исправления
сохранены в [OFFSITE_READINESS.md](OFFSITE_READINESS.md), сейчас они не повторялись.

На назначенном receiver нужны отдельная SSH-учётная запись `arborscan_backup`,
private data root и право владельца один раз установить public receiver script.
Не нужны production DB/RLS/Auth-права, root-доступ для ежедневных передач или
доступ к данным через HTTP. Подготовленный пример начальной установки владельцем
**после назначения разрешённого хоста**, не выполненный в этом пакете:

```sh
# Новый выделенный аккаунт; если такой уже есть, сначала проверить его назначение.
sudo useradd --create-home --shell /bin/sh arborscan_backup
sudo install -d -o arborscan_backup -g arborscan_backup -m 700 \
  /srv/arborscan-private-backups /home/arborscan_backup/.ssh
sudo install -d -o root -g arborscan_backup -m 750 /srv/arborscan-backup-tools
sudo install -o root -g arborscan_backup -m 550 ./ops_offsite_receiver.py \
  /srv/arborscan-backup-tools/ops_offsite_receiver.py
```

Файл receiver доставить из зафиксированного проверенного checkout, без приватных
материалов проекта. Dedicated public key добавить в `authorized_keys` с mode600,
сохранив существующие разрешённые записи. Для отдельного receiver-аккаунта допустим
forced command, не обычный доступ к production shell:

```text
restrict,command="/usr/bin/python3 /srv/arborscan-backup-tools/ops_offsite_receiver.py --root /srv/arborscan-private-backups" ssh-ed25519 <DEDICATED_PUBLIC_KEY>
```

Private key остаётся только в закрытом файле на VPS; проверенный по доверенному
каналу receiver host key — в отдельном known_hosts. Один `ssh-keyscan` без сравнения
с доверенным fingerprint не считается проверкой. Config/key/known_hosts — mode600,
private parent — mode700. Значения key/password/config в чат или Git не переносить.
JSON копируется из существующего example в
`/home/arborscan/.config/arborscan/offsite.private.json`; владелец задаёт host/user,
port, private remote root, receiver script и локальные private file paths. Пример
с `REPLACE_WITH_AUTHORIZED_INDEPENDENT_HOST` не является рабочей регистрацией.

После доступа первый полный перенос, readback и обратное восстановление выполняет
агент; пользователь не должен повторять доступные агенту проверки. Точные CLI
параметры готового инструмента для нынешнего набора:

**До будущих `push/verify/recover` на VPS нужно подготовить sender и его три
публичные зависимости из одного точного проверенного checkout:**
`ops_offsite_push.py`, `ops_offsite_receiver.py`, `ops_verify_release_bundle.py`
и `ops_verify_restore.py`. Они должны находиться рядом в закрытом tooling
каталоге, например `/home/arborscan/ops-tools`, с проверенными SHA и правом чтения
у пользователя `arborscan`. При обновлении сохранить прежние версии, не удалять
другие инструменты и не перезапускать API. Наличие и версии всех четырёх файлов
на VPS в этом срезе не проверялись; перечисленные ниже пути задают подготовленное
место установки. Если выбран другой каталог, изменить все три CLI-пути вместе.
Эти модули используют Python standard library; реквизиты остаются только в
закрытом config/key/known_hosts. Эта подготовка ещё не является выполненным
внешним переносом или подключением daily sender.

```sh
python3 /home/arborscan/ops-tools/ops_offsite_push.py push \
  --config /home/arborscan/.config/arborscan/offsite.private.json \
  --backup /home/arborscan/ops-backups/20261006T195518Z \
  --release-root /home/arborscan/recovery-releases \
  --receipt-root /home/arborscan/ops-backups/offsite-receipts

python3 /home/arborscan/ops-tools/ops_offsite_push.py verify \
  --config /home/arborscan/.config/arborscan/offsite.private.json \
  --name 20261006T195518Z

python3 /home/arborscan/ops-tools/ops_offsite_push.py recover \
  --config /home/arborscan/.config/arborscan/offsite.private.json \
  --name 20261006T195518Z \
  --destination /home/arborscan/offsite-recovery-NEW
```

Destination для `recover` должен отсутствовать. Эти команды на independent host
**не выполнены**. Повторная копия на том же VPS ими не подменяется. Automatic sender
unit/hook к успешной публикации daily backup пока не установлен: доступный CLI
принимает конкретный immutable backup path. После первого настоящего readback нужно
отдельно подключить выбор завершённого набора/lock/retry к Linux-расписанию и проверить
плановый run при выключенном Windows. Статический `--backup` из примера не годится
для ежедневной автоматизации. Подключение и его доказательство относятся к остаткам,
а не к уже включённому systemd-сервису production backup.

## Новая VM и отдельный HTTPS

Предыдущая [runtime-restore-result.json](evidence/release-readiness/runtime-restore-result.json)
подтверждает пустой network-none nested Docker на **том же VPS kernel**, без host
socket/production mounts: текущие образы восстановлены и изолированно запущен Python.
Прежний полноценный lab проверял PostgreSQL/Storage/HTTP/права и restart. Они не
выдаются за новый Linux-хост, выпуск сертификата или восстановление HTTPS.

Для оставшейся проверки нужны:

1. Уже разрешённая **отдельная чистая Linux VM** либо право создать её на существующем
   разрешённом hypervisor, SSH и право установить Docker/Compose/Nginx/Certbot.
   Практический размер для семи сервисов прежнего lab: **2 vCPU, 8 GiB RAM и около
   100 GB диска**. Это запас для распаковки image layers, закрытого архивного пакета,
   файлов/БД и ОС, не измеренное требование production. Поддерживаемая исходными
   pinned images архитектура — Linux amd64.
2. Закрытый доступ к уже независимо полученному комплекту и отдельным bootstrap
   материалам. TLS private keys и Certbot account не возникают из PostgreSQL/Storage
   backup; существующий readable host-config archive их не содержит.
3. Отдельно разрешённое тестовое имя/IP, право настроить **только его** DNS и получить
   тестовый сертификат; для HTTP-01 — входящие80/443 на новой VM. Production
   `31.57.170.88`, его DNS, `arborscan-ip` и маршруты не менять.

После ресурсов восстановление начинается только из independent package, нового
private каталога и точного images manifest. Существующие `ops_recovery_stack.py`
принимают `--backup` и `--images-manifest`; исторические default backup/image ID
использовать для нового кандидата нельзя. Его prepared tooling затем импортирует
Storage, запускает локальный isolated stack и проверяет HTTP/права/файловые связи/
тестовые версии/модерацию/restart, без training и production credentials. Утилиты
содержат проверяемые guards `/home/arborscan/as14-recovery-*`, localhost ports и
internal Docker network; новые пути/сертификат адаптируются и проверяются на
назначенной VM, не подменяются чтением файлов исходного VPS.

Для HTTPS сначала зафиксировать исходную тестовую конфигурацию и её откат, затем
выпустить/перевыпустить сертификат **отдельного имени**, проверить Nginx, цепочку и
hostname штатным TLS-клиентом, reload hook и `renew --dry-run`. ACME staging, если
используется, называется staging и не подтверждает публичное доверие production.
Необходимый доступ к DNS/ACME и право менять только test VM — отдельные prerequisites;
здесь выпуск не запускался. Bootstrap загрузки/внешние сервисы следует учитывать
явно, а не читать скрыто рабочие каталоги старого VPS.

Проверка Supabase backup boundary сверена с
[официальной документацией Database Backups](https://supabase.com/docs/guides/platform/backups):
DB backup содержит метаданные Storage, но не заменяет копию самих объектов; пароли
custom roles требуют отдельно защищённой процедуры. Поэтому pg_dump не объявляется
полным independent application/HTTPS backup. Supabase project/тариф/ключи этим
аудитом не менялись.

## Конкретный недостающий шаг владельца

07.10 пользователь уточнил, что под доступным VPS имеется в виду тот же
рабочий `31.57.170.88`. Он не считается независимым получателем собственной
резервной копии или новой изолированной VM. Уточнение не является принятием
эксплуатационного ограничения; существующие пригодные копии сохранены.

Назначить существующий разрешённый **независимый always-on SSH/Linux receiver**
с 32 GiB минимального /64 GiB предпочтительного свободного private data места,
dedicated account и правом первоначальной установки public receiver script.
Отдельно предоставить **чистую test Linux VM** и **разрешённое test HTTPS имя/IP**
с DNS/ACME/80–443 правами. Пароли, private keys, provider tokens и connection
strings в чат не присылать: настроить их приватно на соответствующих хостах и
сообщить только наличие разрешённых ресурсов/доступа.

До этого агент может завершать остальные AS-16 проверки и доставку кандидата,
но не обозначает independent перенос, новый VM/HTTPS recovery или весь AS-14
завершёнными. Новых ресурсов/платных аккаунтов не создано, эксплуатационная
граница владельцем не принята, научные AS-10/AS-11/AS-13 этим документом не закрыты.
