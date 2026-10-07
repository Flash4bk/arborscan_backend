# AS-16: Google policy в фактическом Python deployment image

Проверка агента **07.10.2026, 13:36:09–13:36:11 UTC**, **2.573 с**:
auth module import успешен, **17/17 synthetic Google identity/route policy tests**
и **8/8 transport/cleanup regression tests** прошли. [JSON evidence](google-auth-python311-runtime.json)
содержит точные SHA каждого файла, image и Dockerfile, список проверок.

Использован уже существующий pinned v3 image
`sha256:5a0b1d63e6c5eb13d14af84590dafebf900f59156613b987000ae8b6938f563b`,
фактический **Python 3.11.16**. Отдельный одноразовый Docker на существующем VPS:
network none, read-only root, user UID/GID текущего непривилегированного аккаунта,
capabilities none, tmpfs для временных файлов, единственный read-only bind
с публичным snapshot кандидата. Scratch создан с mode700, файлы600; production
env/config/data/models не подключались. Контейнер и собственный scratch удалены.
API v3/v4/worker не перезапускались, БД/HTTPS не менялись, модель не стартовала,
API ports не публиковались. Это не отдельная clean VM.

`auth_google_identity` импортирован непосредственно в Python 3.11.16.
Тесты компилируют точную `auth_google` route из AST snapshot `server.py` и используют
синтетический Database/verifier/session transport. Они проверяют policy/owner/
role/race behavior, совместимость синтаксиса/stdlib и импорт нового модуля.
Это **не реальный Google login**, не production HTTP/API и не SQL-миграция.

Отдельный **text review** проверил включение `auth_google_identity.py` в COPY
`deploy-vps/Dockerfile`: модуль нужен рядом с server.py, иначе будущий образ не
сможет его импортировать. Полная новая Docker-сборка не выполнялась, поэтому
container import proof не выдаётся за проверку всех будущих build layers.
Фактическая PostgreSQL UNIQUE-проверка зафиксирована отдельно в
[PG evidence](google-identity-index-postgres.md).
