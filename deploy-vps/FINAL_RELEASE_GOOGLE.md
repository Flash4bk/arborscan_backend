# AS-16 — Google-вход: фактический контракт и блокер выпуска

Срез 07–08.10.2026. Это дополнение к [FINAL_RELEASE.md](FINAL_RELEASE.md),
с отдельными actual native проверками ниже. Подготовленное исправление не развёрнуто.

## Используемый путь

`google_sign_in 6.3.0` / Android `6.2.1` получает Google ID token нативно.
Клиент отправляет его по HTTPS на `/api/v3/auth/google`; API v3 проверяет
подпись, audience, issuer и срок через `google.oauth2.id_token`, затем создаёт
собственную непрозрачную сессию в `public.users/auth_sessions` Supabase.
Это **не Supabase Auth / GoTrue**: нет браузерного callback, Supabase JWT,
refresh token или требуемого изменения Google provider/redirect в Supabase.
Сессия сохраняется на устройстве и повторно проверяется `/auth/me`;
её срок по умолчанию — 30 дней, сервер может переопределить его через
`ARBORSCAN_AUTH_TOKEN_TTL_DAYS`. Фактический TTL production в этом срезе не
проверялся. Автоматический refresh-token обмен отсутствует.

Реальный read-only срез подтвердил соответствие Supabase-проекту
`mfjxhtxwaablygwdjxhx` и Web client ID ниже. Google handler запущенного v3
совпадает с baseline main `10b54e7` по SHA функции. Подготовленное исправление
в production отсутствует. В реальной схеме уникальны ID и email;
уникального индекса `google_sub` пока нет. Значения пользователей не публикуются.

## Поля существующего Google Cloud-проекта

Номер проекта: **946297507051**, фактически подтверждён в доступной консоли;
project ID `project-ec8aaaae-b7b4-4e51-b8c`. Новый проект не создавался.
[Страница Credentials этого проекта](https://console.cloud.google.com/apis/credentials?project=946297507051).
Первоначально консоль требовала owner login. При повторном открытии консоли
агент прочитал существующие ArborScan credentials и выполнил разрешённую регистрацию
нового сертификата **07.10,19:00:57UTC**. Это фактическая внешняя настройка Google,
не изменение VPS/Supabase и не доказательство успешного нового входа.

Web client / `serverClientId`:
`946297507051-33c4msb91harv7rqppf2f31qn10n1m2m.apps.googleusercontent.com`.
Android `clientId` в плагин не передаётся; нужны две Android-регистрации
в этом же проекте с `package name = com.example.arborscan_app`:

| Платформа | SHA-1 действующего сертификата |
|---|---|
| API24–32 | `2C:08:B4:D6:37:57:4C:16:BE:5C:BA:1A:59:5A:4B:AA:C8:54:1A:4C` |
| API33+ | `E4:DF:D7:6D:C3:EC:5F:96:BE:6B:74:09:49:C6:78:AE:BF:1F:11:0E` |

Перед изменением сохранить безопасный срез: имена/типы credentials, публичные
client IDs, package/SHA-1 и OAuth consent/testing audience. Секреты не копировать
в Git, APK или чат. Сначала сверить наличие каждой пары; добавлять только
отсутствующую, прежние регистрации сохранить. Если consent находится в Testing,
разрешённый тестовый Google-аккаунт должен быть включён в Test users.
Дополнительные scopes не нужны: остаются email/profile.
Откат регистрации: удалить только вновь добавленный Android credential по
зафиксированному ID; не менять Web client, старую пару или пользовательские identity.
Этот откат **подготовлен, не выполнен**.

Старая регистрация `ArborScan Android Debug`, client
`946297507051-u505jg1q01gfl27d6prji4ojhdv1lq6q.apps.googleusercontent.com`,
имеет точно указанные package/старый SHA-1; её поля сохранены.
Добавлена только отсутствующая `ArborScan Android Permanent API33`, client
`946297507051-hr95ptf0d1aonhfj0ptpcf8576dsqhp6.apps.googleusercontent.com`,
с тем же package и новым SHA-1. После создания повторно открыты сохранённые
поля Name/package/SHA-1. Web client, API-ключи/ограничения, scopes и другие
приложения проекта не изменялись; client secrets/API keys не читались.
Безопасные [срез до](evidence/final-release/google-registration-before.json),
[результат](evidence/final-release/google-registration-after.json) и
[реальный снимок создания](evidence/final-release/google-android-client-created.jpg)
не содержат аккаунт владельца или секреты.

При создании client audience фактически **Testing/External**, один test user;
агент не менял список. Позднее пользователь сообщил о добавлении отдельного
разрешённого существующего test account. Агент read-only проверил текущие
агрегаты: **2 users /2 test /0 other**, режим всё ещё Testing/External;
emails не опубликованы. [Доказательство готовности аудитории](evidence/final-release/google-test-audience-ready.json).
Владелец затем выполнил приватный вход/MFA на тестовом Android36; агент
самостоятельно завершил native/server/UI сценарии ниже. Для API24 тот же
разрешённый существующий аккаунт затем реально проверен отдельно ниже.
Консоль сообщает возможную задержку
применения от5минут до нескольких часов; регистрация не равна successful native
login/серверной сессии. Не нажимались Publish app, Upgrade или создание аккаунтов.

## Подготовленное серверное исправление

На синтетических verified claims воспроизведены три дефекта прежнего handler:
email мог браться из тела клиента, `email_verified` не проверялся, а прежний
`google_sub` профиля мог перезаписываться по совпавшему email. Реальные identity
и токены для воспроизведения не использовались. Это подтверждение политики
старого кода, не утверждение эксплуатации дефекта против пользователей.

`auth_google_identity.py` принимает только проверенные claims. Поиск начинает
со стабильного subject; конфликт владельцев/неоднозначность запрещены.
Существующие owner ID, пароль и роль сохраняются. Связь с прежним email-профилем
допускается только для подтверждённого Gmail/Workspace, при включённой будущей
возможности; внешний email сам по себе не даёт права перепривязать профиль.
Conditional PATCH защищает строку, уникальный индекс обеспечивает глобальную
уникальность subject. Старые клиенты могут продолжать присылать дополнительные
поля, но сервер их не использует для identity. Новый клиент посылает ID token.

`ARBORSCAN_GOOGLE_IDENTITY_CLAIMS_ENABLED=false` по умолчанию запрещает создание
и новые связи (503 с понятным сообщением); уже однозначно связанные Google
профили могут входить без изменения identity. Флаг нельзя включать до проверки
реального уникального индекса. COPY Dockerfile включает новый модуль.

Импорт auth-модуля и **17/17 synthetic identity/route policy tests** проверены
в фактическом **Python 3.11.16** уже существующего серверного image, в отдельном
network-none/read-only Docker. Дополнительно там прошли **8/8 safety tests**
тестовой обвязки. Точные source SHA и границы —
[Python deployment runtime evidence](evidence/final-release/google-auth-python311-runtime.md).
Это импорт модуля и AST-проверка маршрута, **не полная новая Docker-сборка,
не production API rollout и не реальный Google login**.

Подготовленная недеструктивная SQL-предпосылка:
[google-identity-unique-index.sql](google-identity-unique-index.sql).
Она сохраняет NULL/empty legacy, проверяет тип text/varchar, значения, дубликаты
и точное определение индекса, использует `CREATE UNIQUE INDEX CONCURRENTLY`.
Реальные 16 проверок PostgreSQL 17.6 прошли в отдельном network-none/tmpfs
контейнере с синтетической users; это **не выполненная production-миграция** и
не восстановление новой VM. Доказательства и найденный varchar-дефект:
[PG-проверка](evidence/final-release/google-identity-index-postgres.md).

Будущий порядок требует отдельного разрешения на v3 и БД:

1. Зафиксировать точные image/config, проверить пригодную полную копию;
   подготовить отдельный checkout и образ exact security commit.
2. Установить исправленный v3 с claims=false на всех экземплярах; v4/worker,
   модели и identity не изменять. Проверить email и прежний Google login.
3. Через приватный PostgreSQL service выполнить SQL с `ON_ERROR_STOP=1` и
   autocommit, **без** `--single-transaction`. При ошибке/invalid index остановиться;
   не удалять identity, не обходить guard и не включать флаг.
4. После реальных guards включить claims и проверить новый вход на двух signer.

Подготовленный откат: сначала claims=false на всех экземплярах; при необходимости
вернуть точный прежний образ/config. Аддитивный индекс совместим со старым API,
его сохранить. Возврат старого handler возвращает и найденный дефект — для
постоянной эксплуатации предпочтителен исправленный образ с claims=false.
Удаление индекса допустимо только отдельным решением после отключения claims;
restore старого дампа поверх действующей БД не является откатом этого изменения.

## Что не заменяется mock

Widget-тесты проверяют настоящий ProfilePage, native plugin transport заменён:
отмена, ошибки, повтор, double tap, late replies, owner isolation, expiry,
восстановление экрана/локального черновика. Они не доказывают Google OAuth.
Реальный API36 Google flow проверен в exact APK23 со signer7fb94e… и GMS
26.37.37 (260800-994713346). Владелец ввёл credentials/MFA приватно; агент
выбрал единственный разрешённый test account в native chooser. ArborScan
показал Google-login и server session. Собственная пустая история загрузилась;
предыдущие DEMO аккаунты/их локальный журнал не открылись этому owner. После
force-stop/cold restart тот же test account получил server-confirmed session.
NormalUI logout/relogin прошёл. Native chooser/Back оставил signed-out UI;
форма сохранила собственный ранее введённый email, это не авторизованная сессия.
Offline login через реальный provider показал понятную недоступность и не
создал session. Исходная сеть восстановлена, retry/revalidation прошли.
Read-only exact test identity проверен: один email match, owner/subject/role/
created_at после повторов не изменены, duplicate owner не создан. Никакие роли
или identity проверочным скриптом не менялись; используется штатный Google handler.
[Native36 proof](evidence/final-release/google23-native36.json),
[network](evidence/final-release/google23-offline.json),
[canonical identity](evidence/final-release/google23-identity-continuity.json).

Это successful legitimate login старого production handler, не безопасность
ещё не развёрнутой новой policy и не GoTrue refresh-token проверка. Старый
signer API24 проверен отдельно с GMS 20.24.14 (040800-319035315): установленный APK23
считан обычным shell read и SHA совпал, effective certificate68ff98… подтверждён
signed instrumentation. Данные не очищены/переустановки нет. Тот же test account
реально вошёл через native Google и получил ArborScan server session; история,
force-stop/revalidation, logout/relogin и chooser cancellation прошли. Offline
показал «Нет связи с Google» без session; после возврата сети retry/revalidation
прошли. Read-only identity совпадает с API36 по owner/subject/role/created_at,
единственная строка, duplicate owner не создан.
[Actual native24](evidence/final-release/google23-native24.json),
[network24](evidence/final-release/google24-offline.json),
[identity24](evidence/final-release/google24-identity-continuity.json).

На Android7 protected ADB network commands получили Permission Denial; root/
permission override не применялся. Использован обычный Settings UI airplane
switch. Первое restore оставило persisted wifi_on3; Wi-Fi Settings Activity в
image отсутствует. Через разрешённый Settings provider восстановлен исходный
wifi_on1, итог0/1/1 и настоящий online Google retry проверены. Первичный failed
restore сохранён в google24-offline-initial-restore.json. S24 сеть/аккаунт не
менялись. Эти два legitimate production flows не доказывают безопасность
prepared handler или реальный новый unique index. До отдельно разрешённого
v3/security/index rollout stable release tag не создаётся.

Источники проверены для используемых версий:
[Google backend verification](https://developers.google.com/identity/sign-in/android/backend-auth),
[google-auth verifier](https://google-auth.readthedocs.io/en/latest/reference/google.oauth2.id_token.html),
[plugin 6.3.0](https://pub.dev/packages/google_sign_in/versions/6.3.0),
[Android 6.2.1](https://pub.dev/packages/google_sign_in_android/versions/6.2.1),
[Supabase Google](https://supabase.com/docs/guides/auth/social-login/auth-google),
[Supabase native links](https://supabase.com/docs/guides/auth/native-mobile-deep-linking).
Последние два описывают иной GoTrue-путь; применять его настройки к текущему
кастомному ArborScan-входу без изменения архитектуры не требуется.
