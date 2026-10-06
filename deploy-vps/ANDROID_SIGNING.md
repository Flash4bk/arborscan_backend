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
