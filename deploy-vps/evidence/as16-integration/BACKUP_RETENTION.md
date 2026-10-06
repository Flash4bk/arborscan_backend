# AS-16: резервная копия и исправление архивирования перед обновлением v4

06.10.2026 `arborscan-backup.service` завершилась в 03:20:22 UTC с exit 1:
`Backup refused: 14 copies retained; archive elsewhere before continuing`.
Найдены 14 COMPLETE; свободно около 65.8 GiB, недостаток места не был причиной.
Timer остался active/waiting. Агент реально проверил SHA всех 14 внешних manifests,
всех 11 прежних native PostgreSQL manifests (44 файла) и 2240 внутренних SHA
последнего application archive. Прежняя копия `20261005T032052Z` была старше
33.77 часа и предшествовала вечернему изменению конфигурации погоды.
Доказательства: [backup-retention-before.json](backup-retention-before.json).

## Архивирование: первоначальная операция исправлена

В 13:12:40 UTC первоначально перенесён полный `20260926T202023Z` из active base
в `ops-archive`, после проверки независимой D-копии и всех 19 SHA. Затем обычный
backup завершён успешно. Однако дальнейшая подготовка rollout выявила:
compose labels **работающих v4 и worker** ссылаются на
`/home/arborscan/ops-backups/20260926T202023Z/reliability.yml`.
Запущенные контейнеры продолжали работать, но повторное использование compose
sources было недоступно. Это реальный дефект процедуры выбора архивного набора;
первоначальная операция не оставлена в таком состоянии.

В 13:31:55 UTC полный каталог атомарно **возвращён на исходное место** под тем же
`/home/arborscan/ops-backups/backup.lock`. Проверены до и после 19 SHA / 927004164 B;
`reliability.yml` существует, SHA
`8219012d566fef3328aef3cde1a75842cd8927060dca1f89680af3284596f701`.
Нет symlink, подставного файла или удаления данных. Active count временно стал 15.

После этого просмотрены labels и mount sources **всех трёх работающих контейнеров**,
включая v3, v4 и worker; проверены 29 абсолютных путей, в том числе mount ancestors.
Другой исторический COMPLETE `20260927T032112Z` не имеет этих зависимостей.
Его независимая копия `D:/ArborScanBackups/20260927T032112Z` реально проверена:
все 22 SHA, 927004700 B. Нормированный LF hash manifest совпадает с VPS:
`a551b875dd5b2c8fbc2bef540fedf4850c48cb961a95cdcb12d3c4d2e85e4479`.

В 13:33:50 UTC под existing flock атомарно перемещён **только другой набор**:

- Был: `/home/arborscan/ops-backups/20260927T032112Z`.
- Стал: `/home/arborscan/ops-archive/20260927T032112Z`.
- Все 22 SHA повторно проверены до и после, runtime identities не изменились.
- Active COMPLETE снова 14. Самый старый набор и его runtime config на исходном месте.
- Свежая копия и все native PostgreSQL наборы сохранены.

Эта архивная историческая копия создана до подключения native PostgreSQL;
это явно не называется native dump. Ни одного файла не удалено. Guard 14,
backup base, service и timer не изменены; контейнеры не перезапускались.
История обеих операций и независимые проверки сохранены в
[backup-archive-transition.json](backup-archive-transition.json) и
[retention-runtime-dependencies.json](retention-runtime-dependencies.json).

## Реально выполненная свежая копия

Между первоначальным архивированием и его исправлением выполнен штатный путь:

```sh
/bin/sh /home/arborscan/ops-tools/ops_backup.sh
```

Без sudo и override guard/base, с обычным service environment. Exit 0, 415.292 с;
завершение 13:20:20 UTC. Набор `/home/arborscan/ops-backups/20261006T131325Z`
содержит общий и PostgreSQL COMPLETE. После исправления он не менялся, повторный
backup не запускался.

| Доказательство | Фактический результат |
|---|---|
| Outer manifest | 30 файлов / 979475430 B, все SHA успешны |
| PostgreSQL manifest | 4 файла, все SHA успешны; dump 52102608 B, PGDMP, сервер 17.6 |
| Application manifest | 2261 SHA; 2243 Storage objects / 15 таблиц; workflow/history/quality — 1 |
| Offline file restore штатной процедуры | 2261 файл / 828152403 B, 16 report links; database_restored=false |
| Текущий env config | Совпадает с копией, mode 0600; погода настроена, значения не публикуются |
| Runtime snapshot | Image/StartedAt v4 и worker совпали с live на момент проверки, до rollout |

Outer SHA256SUMS:
`65fb5865cdc25971f8852a82e11132b3baac93c5bf8b150b241ae6985f99abdc`.
Native SHA256SUMS:
`f7e1455730236bdb14c4405547384a39f8aa19f6b090f55534fd9efca7cec744`.
Текущий и сохранённый env config SHA:
`d021e2bc5c91eb992d7b40ded3ea917f9a19b40b1db519437ce5f1e707ba5d33`.

**Ограничение config rollback:** свежий набор не содержит direct `reliability.yml`
и такого member нет в `local-files.tar` — это проверено readonly в 13:34:15 UTC.
Rollout должен дополнительно сохранить текущий восстановленный compose source
в собственный приватный configuration snapshot. Свежая копия самостоятельно не
выдаётся за полный набор всех compose sources. Подготовка rollout ранее правильно
отказала при отсутствии source; её защиту не обходили.

Полное свидетельство: [fresh-deploy-backup.json](fresh-deploy-backup.json).
Native dump в этом подшаге не применялся к новой или production базе; SHA-проверка
не называется новым SQL restore. Application/Storage snapshot не объявляется
атомарным снимком БД; отдельно создан native transaction-consistent dump.
Ни env, private inspection/logs, backup archives, ни данные пользователей в Git не включены.

## Подготовленный обратный перенос текущего архивного набора

Этот rollback для **заменяющего** набора пока не выполнялся. Он сохраняет все новые
копии; active count станет 15, и прежний guard продолжит отказывать. Это возврат
расположения файлов, а не SQL restore или откат API.

```sh
python3 - <<'PY'
from pathlib import Path, PurePosixPath
import fcntl, hashlib

base = Path('/home/arborscan/ops-backups')
archived = Path('/home/arborscan/ops-archive/20260927T032112Z')
original = base / '20260927T032112Z'
expected = 'a551b875dd5b2c8fbc2bef540fedf4850c48cb961a95cdcb12d3c4d2e85e4479'

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

assert base.resolve() == base and archived.resolve() == archived
assert (archived / 'COMPLETE').is_file() and not original.exists()
assert base.stat().st_dev == archived.stat().st_dev
with (base / 'backup.lock').open('r+b') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not original.exists()
    assert sha(archived / 'SHA256SUMS') == expected
    for line in (archived / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        relative = PurePosixPath(name)
        target = (archived / name).resolve()
        assert not relative.is_absolute() and '..' not in relative.parts
        assert target.is_relative_to(archived) and sha(target) == digest
    archived.rename(original)
    assert sha(original / 'SHA256SUMS') == expected
print('Historical backup moved back; newer sets retained; guard unchanged.')
PY
```

## Оставшийся эксплуатационный пункт AS-14

Systemd-service сохраняет исторический **failed**, exit 1; успешный вызов самого
backup script не выдаётся за успешный systemd run. Timer active/waiting, следующий
запуск 07.10.2026 03:19:03 UTC (06:19:03 Europe/Minsk). При active count 14 он снова
остановится штатным guard. Следующий эксплуатационный пакет должен выполнить
независимую off-site передачу свежей копии, дальнейшее проверенное архивирование
и проверку штатного systemd запуска. При выборе набора обязательно исключать
действующие compose sources и mount ancestors **всех** работающих контейнеров,
а затем повторно сверять identities под lock. Retention guard не перерабатывался.
