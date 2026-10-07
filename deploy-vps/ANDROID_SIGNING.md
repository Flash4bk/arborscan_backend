# AS-16: подпись кандидата и безопасное обновление

Это контракт инструмента, а не доказательство установки. Фактические версии,
SHA APK, устройства и результаты восстановления ключа записываются отдельно в
`RELEASE_READINESS.md`. Публикация в магазине или окончательный release/tag в
этом пакете не выполняются.

## Диапазон Android и действующий сертификат

ApplicationId остаётся `com.example.arborscan_app`, minSdk — 24, targetSdk — 36.
Gradle собирает **неподписанный release без debuggable**. Прежний автоматический
release fallback на debug-ключ удалён. Неподписанный APK не является готовым
артефактом для установки: его обрабатывает `ops_android_signing.py`.

Выбран `rotation-min-sdk-version=33`. Подпись проверяется по диапазонам:

| Android / API | Действующий сертификат | Схема | Значение |
|---|---|---|---|
| 7.0–8.1 / 24–27 | Прежний Android Debug | v2 | Обновление прежним сертификатом; переход на постоянный ключ не заявляется |
| 9–12L / 28–32 | Прежний Android Debug | v3 | Совместимость сохраняется; переход на постоянный ключ не заявляется |
| 13 и выше / 33+ | Новый постоянный | v3.1 и lineage | Ротация с передачей уже установленного состояния |

Схемы v1/v2 не поддерживают lineage. Google рекомендует целевой переход через
v3.1 для Android 13+, чтобы сохранить прежний signer на старых платформах.
Поддержка API 24 не сокращается. Совместимость старых Android в этом кандидате
не означает, что debug-сертификат стал постоянной release-подписью, и не делает
APK подходящим для магазина. [apksigner](https://developer.android.com/tools/apksigner),
[AOSP v3](https://source.android.com/docs/security/features/apksigning/v3),
[AOSP v3.1](https://source.android.com/docs/security/features/apksigning/v3-1).

Lineage включает `installed-data=true` для прежнего signer, необходимый для
обновления с данными; `rollback=false` не позволяет старому ключу вернуть
контроль над уже ротированной установкой. Приёмка нового сертификата и
сохранность содержимого проверяются реальными установками отдельно: вывод
`apksigner verify` сам по себе не доказывает работу приложения или tracking AR.

## Приватные материалы и сборка

Существующие ключи не перегенерировать. Хранить keystore, старый keystore,
lineage, парольные файлы и рабочую JSON-конфигурацию вне любого Git checkout,
под приватным Windows ACL либо POSIX mode 600 с закрытым родительским каталогом.
Windows ACL отдельно проверяет оператор: инструмент не выдаёт POSIX mode за ACL.
Пароль не помещается в командную строку: доступны только `file:` и `env:`.

Шаблон `android-signing.example.json` содержит только заменяемые пути и публичные
отпечатки. Скопировать его в защищённый каталог, заполнить из уже проверенных
сертификатов. Все пути должны быть абсолютными. Материал подписи требуется
сохранить независимо от одного компьютера и проверить восстановление/подпись
в другом приватном каталоге; сама эта инструкция не означает выполненную копию.

Для впервые создаваемой lineage (команда откажется заменять существующую):

```powershell
python deploy-vps/ops_android_signing.py rotate --config D:/ArborScanSigning/private/signing.json
```

Сборка unsigned release с явно заданной новой версией, без прежних dart-define:

```powershell
flutter build apk --release --no-pub --build-name=<версия-кандидата> --build-number=<код-больше-установленного>
python deploy-vps/ops_android_signing.py sign --config D:/ArborScanSigning/private/signing.json --input arborscan_app/build/app/outputs/flutter-apk/app-release.apk --output D:/ArborScanSigning/candidates/candidate.apk --metadata D:/ArborScanSigning/candidates/candidate.json --minimum-version-code=<код-кандидата>
python deploy-vps/ops_android_signing.py verify --config D:/ArborScanSigning/private/signing.json --apk D:/ArborScanSigning/candidates/candidate.apk --minimum-version-code=<код-кандидата>
```

Команду Flutter выполнять из `arborscan_app`, Python — из корня проекта; путь
`--input` в показанной Python-команде относится к корню. Не выдавать исходный
unsigned `app-release.apk` за подписанный кандидат.

Инструмент проверяет manifest (applicationId, min/target SDK, versionCode,
INTERNET, отсутствие debuggable), SHA-256 и SHA-1 каждого действующего signer,
схемы в диапазонах API 24–27/28–32/33/36 и выравнивание native библиотек на 16 KiB.
Публичный JSON содержит только эти метаданные/хеши и явное
`device_installation_verified=false`. Стадия подписи временная; выходной APK
появляется атомарно после проверок. Повтор существующие APK/lineage не заменяет.

APK нельзя менять после подписи. `zipalign` вызывается до `apksigner`, затем
проверяется готовый файл. Повторная подпись или иной versionCode требуют нового
пути выходного APK и повторной проверки.

## Карты, авторизация и откат

Google Maps Android и Google Sign-In могут требовать регистрацию нового
сертификата. В существующем Google Cloud проекте нужны Android credentials /
Android application restriction: package `com.example.arborscan_app` и **новый
публичный SHA-1**; старую разрешённую пару сохранить для API 24–32. Используемый
Web OAuth clientId и серверная Bearer-авторизация сами по себе не заменяют Android
credentials. [Подпись и API-провайдеры](https://developer.android.com/studio/publish/app-signing).
Ограничения ключа/TLS/auth не отключать. Проверку облачной настройки нельзя
подменять наличием API key в манифесте или mock-тестом.

После успешной ротации старый debug APK не является проверенным откатом:
signature policy и versionCode могут запретить его установку. Безопасный клиентский
откат — исправляющий release с **тем же постоянным ключом/lineage и большим
versionCode**, с сохранением прежнего формата хранения. Сначала проверить такую
вторую установку на тестовом устройстве, затем применять доказанный путь на S24.
Удаление приложения, очистка данных или принудительное понижение версии не
являются процедурой сохранного отката.

Если lineage сохраняет установку, новый поток пользовательского экспорта/импорта
не добавляется без причины: реальные файлы, владельцы, происхождение значений,
черновики, несинхронизированные операции и авторизация проверяются до/после.
Невозможность безопасного перехода на отдельной версии Android отмечается
отдельно, а не маскируется успехом Android 16.

Целевые программные проверки:

```powershell
python -m unittest tests_v4.test_ops_android_signing -v
```

Тесты используют синтетический SDK output для отказов, изоляции приватных
материалов, публикации и совместимости. Реальные APK, Android/Keystore,
обновление данных, offline/reconnect и системные диалоги — отдельные доказательства.

## Восстановление ключей после потери Windows-ПК

Публичный `ops_signing_key_recovery.py` не зависит от прежнего private helper на D.
Для восстановления нужны приватные **оба** файла с независимой копии:
`key-backup.aesgcm` и `recovery-secret.private`. Их значения не печатать и не
передавать в аргументах. Доступ к VPS через существующий SSH/консоль хостинга
необходимо сохранить отдельно; наличие копии не восстанавливает утраченные
реквизиты доступа к самому хостингу.

На другом Windows-ПК создать новый закрытый каталог, не Git checkout:

```powershell
$signingRecoveryRoot = 'D:\ArborScanSigningRecovery'
if (Test-Path -LiteralPath $signingRecoveryRoot) { throw 'Выберите новый каталог восстановления' }
New-Item -ItemType Directory -Path $signingRecoveryRoot | Out-Null
$signingRecoverySid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value
icacls $signingRecoveryRoot /inheritance:r /grant:r "*$($signingRecoverySid):(OI)(CI)F" '*S-1-5-18:(OI)(CI)F' '*S-1-5-32-544:(OI)(CI)F'
if ($LASTEXITCODE -ne 0) { throw 'Приватный ACL не установлен' }
scp -o StrictHostKeyChecking=yes arborscan@31.57.170.88:/home/arborscan/signing-private/key-backup.aesgcm "$signingRecoveryRoot\key-backup.aesgcm"
scp -o StrictHostKeyChecking=yes arborscan@31.57.170.88:/home/arborscan/signing-private/recovery-secret.private "$signingRecoveryRoot\recovery-secret.private"
py -3 -m venv "$signingRecoveryRoot\venv"
& "$signingRecoveryRoot\venv\Scripts\python.exe" -m pip install 'cryptography==50.0.1'
& "$signingRecoveryRoot\venv\Scripts\python.exe" deploy-vps/ops_signing_key_recovery.py --escrow "$signingRecoveryRoot\key-backup.aesgcm" --recovery-secret-file "$signingRecoveryRoot\recovery-secret.private" --destination "$signingRecoveryRoot\restored"
```

Последнюю команду выполнять из проверенного checkout с этим публичным инструментом;
само содержимое ключей хранится вне checkout. SSH host key проверить обычным доверенным
способом, не отключать `StrictHostKeyChecking`. Каждый неуспешный этап останавливать;
не продолжать после ошибки SCP/установки зависимости. Для Windows требуется рабочий
PowerShell ACL reader: используется установленный `pwsh`, иначе Windows PowerShell.
Отсутствующий native ACL module даёт отказ, а не отключает проверку.

На Linux аналогичная команда использует абсолютные приватные пути, файлы mode600
и существующий закрытый parent mode700. Поддерживается атомарный `renameat2`
NOREPLACE. Нужен Python3.11+ и isolated `cryptography`; новая системная установка
или сервер PostgreSQL для этой процедуры не требуются.

Контракт escrow: prefix `ARBORSCAN-KEYS-1\n`, nonce12, AES-256-GCM с AAD
`ArborScan signing material v1`. Восстановление требует ровно четыре regular файла:
`release.p12`, `legacy-debug.keystore`, `signing.lineage`, `credentials.private.json`.
JSON содержит только непустой строковый `release_password`. Лимиты: uncompressed
tar4MiB, отдельный member1MiB, credentials64KiB, secret file1KiB. Аутентификация,
полнота/структура/JSON и фактические SHA staged-файлов проверяются до публикации.
Links/junctions/traversal, duplicate/unexpected/missing members, скрытые trailing
данные, публичные ACL и расположение в Git отклоняются. На Windows ACL допускает
только текущего владельца, SYSTEM/Administrators и OWNER RIGHTS при проверенном
владельце; [значение OWNER RIGHTS определено Microsoft](https://learn.microsoft.com/en-us/windows-server/identity/ad-ds/manage/understand-special-identities-groups).

Публикация атомарная и не заменяет даже пустой существующий destination. При обычной
ошибке удаляется только staging текущей операции; после жёсткого выключения оставшийся
staging приватен, не считается завершённым и автоматически не удаляется чужой операцией.
Повтор после успешного восстановления требует нового destination. Исходные encrypted
копии и прежние ключи не изменяются. Пароль из recovered JSON перенести приватно в
password-file/env для signing config с новыми путями; не печатать его через `type`,
`cat`, `echo` или в аргументах `keytool/apksigner`.

Успешная расшифровка не является проверкой сертификата. Затем сверить публичные
отпечатки с release evidence и выполнить проверку подписи validation APK через
`ops_android_signing.py` с восстановленным ключом/lineage. Фактическое восстановление
реальных материалов и подпись фиксируются отдельно основным отчётом, а не выводятся
из тестовых fixtures.

Программная проверка использует новые AES keys, PKCS12 test-certificate и архивы;
реальные ArborScan ключи/пароли не читаются:

```powershell
python -m unittest tests_v4.test_ops_signing_key_recovery -v
```

Финальный module SHA `45a0a996b9a4a14e2682bd0a57876a1b6c65853dc763b4eb58a90ddf9343bf8f`:
Generated recovery: **23/23 на Windows за 222,369 с** (Python3.12.14/cryptography50.0.1,
настоящие private ACL, symlinks, конкуренция и no-replace) и **23/23 на Linux за 1,808 с**
(VPS host/cryptography41.0.7, отдельные private fixture каталоги, POSIX и renameat2).
Последнее изменение обрабатывает слишком глубокий JSON безопасным отказом;
содержимое действующего escrow не меняется. [Windows вывод](evidence/release-readiness/key-recovery-windows.txt),
[Linux вывод](evidence/release-readiness/key-recovery-linux.txt).

Отдельный outer-LF regression прошёл **1/1 за 0,019 с** в изолированном
image034a9d, `--network none`, read-only source/rootfs, tmpfs4GiB;
[вывод](evidence/release-readiness/outer-lf-linux-final.txt). Первый маленький tmpfs32MiB
корректно отказал по настоящему2GiB free-space guard; guard не ослаблялся.
Generated fixtures не являются восстановлением реальных ключей. Агент отдельно
получил real escrow с VPS через проверенный SSH, восстановил четыре материала в новый
private каталог, сравнил содержимое и выполнил подпись/проверку validation APK.
Публичный новый certificate SHA256 —
`7fb94ede4099199db9166609c34d6278ebe51c7d80bd6ce1cd0208bf863863b8`;
результат этого отдельного действия находится в `key-recovery-tool-actual.json`
основного release evidence. Сам recovery CLI сохраняет `certificate_verified=false`:
положительный результат относится к последующей реальной проверке apksigner.
Эта копия ключей независима от build-PC, но не является внешним backup production
сервера, отдельной новой VM или проверкой TLS на новом хосте.
