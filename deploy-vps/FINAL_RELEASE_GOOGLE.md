# AS-16 — Google-вход: фактический контракт и блокер выпуска

Срез 07.10.2026. Это дополнение к [FINAL_RELEASE.md](FINAL_RELEASE.md),
а не подтверждение нового входа Google или развёртывания исправления.

## Используемый путь

`google_sign_in 6.3.0` / Android `6.2.1` получает Google ID token нативно.
Клиент отправляет его по HTTPS на `/api/v3/auth/google`; API v3 проверяет
подпись, audience, issuer и срок через `google.oauth2.id_token`, затем создаёт
собственную непрозрачную сессию в `public.users/auth_sessions` Supabase.
Это **не Supabase Auth / GoTrue**: нет браузерного callback, Supabase JWT,
refresh token или требуемого изменения Google provider/redirect в Supabase.
Сессия сохраняется на устройстве и повторно проверяется `/auth/me`;
её срок на сервере — 30 дней. Автоматический refresh-token обмен отсутствует.

Реальный read-only срез подтвердил соответствие Supabase-проекту
`mfjxhtxwaablygwdjxhx` и Web client ID ниже. Google handler запущенного v3
совпадает с baseline main `10b54e7` по SHA функции. Подготовленное исправление
в production отсутствует. В реальной схеме уникальны ID и email;
уникального индекса `google_sub` пока нет. Значения пользователей не публикуются.

## Поля существующего Google Cloud-проекта

Номер проекта, следующий из действующего Web client ID: **946297507051**.
Владелец должен сверить его в уже существующем проекте, не создавать замену.
[Страница Credentials этого проекта](https://console.cloud.google.com/apis/credentials?project=946297507051).
Доступный браузер перенаправился на вход Google; список credentials не прочитан.

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
Этот откат **подготовлен, не выполнен**. Регистрации сейчас не подтверждены.

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
На изолированном AVD36 нет настроенного Google-аккаунта. С S24 пользователя
выход не выполняется. Нужны вход владельца в консоль и отдельный разрешённый
тестовый Google-аккаунт (интерактивная авторизация/MFA — владельцем, без пароля
в чат). После доступа агент выполняет UI-цикл самостоятельно.
До этого и до серверного security rollout стабильный тег выпуска не создаётся.

Источники проверены для используемых версий:
[Google backend verification](https://developers.google.com/identity/sign-in/android/backend-auth),
[google-auth verifier](https://google-auth.readthedocs.io/en/latest/reference/google.oauth2.id_token.html),
[plugin 6.3.0](https://pub.dev/packages/google_sign_in/versions/6.3.0),
[Android 6.2.1](https://pub.dev/packages/google_sign_in_android/versions/6.2.1),
[Supabase Google](https://supabase.com/docs/guides/auth/social-login/auth-google),
[Supabase native links](https://supabase.com/docs/guides/auth/native-mobile-deep-linking).
Последние два описывают иной GoTrue-путь; применять его настройки к текущему
кастомному ArborScan-входу без изменения архитектуры не требуется.
