# AS-16 — артефакт 1.3.0, установка и доступный откат

08.10.2026. Этот документ относится к **неизменённому подписанному APK23**.
Публикация main/тега и завершающая уборка фиксируются отдельно в
[FINAL_RELEASE.md](FINAL_RELEASE.md) и каноническом
[плане](ARBORSCAN_MASTER_PLAN.md). Доставка — APK; публикация в магазине и
публичный GitHub Release не входят в этот пакет.

## Выдаваемый файл

| Поле | Значение |
|---|---|
| Файл | `D:\arborscan_backend\output\final-release\ArborScan-1.3.0-23.apk` |
| Версия / code | `1.3.0` / `23` |
| Размер | `115594211` байт |
| SHA-256 | `71f38189ee62a7825029ec7c25bf635891d3adcd0fa3eea3d9b618a223e4a31c` |
| Исходник APK | `bbcd5b539427e62da58739fc2dbeff8550ee72c6` |
| ApplicationId | `com.example.arborscan_app` |
| Android min / target / compile | `24 / 36 / 36` |
| Режим | release, `debuggable=false`, INTERNET присутствует |
| Подпись API24–32 | прежний signer, SHA-256 `68ff9861f52fdb22744b30ee27092008097f9eff4ead7eb38f80e662115f5b97` |
| Подпись API33+ | постоянный signer, SHA-256 `7fb94ede4099199db9166609c34d6278ebe51c7d80bd6ce1cd0208bf863863b8` |
| Lineage SHA-256 | `93d10d0e6b813bf9053debf1cc879378a4e90e2d0e07ac755eba50eb57a87dfe` |

Новая read-only проверка 08.10: фактические bytes/SHA/manifest, `apksigner verify`
на 24–27/28–32/**33–36**, embedded lineage и 16KiB ZIP alignment прошли.
У обоих сертификатов `installed-data=true`, `rollback=false`.
Прежний сертификат API24–32 является debug certificate совместимости;
nondebuggable относится к режиму приложения. Это ограничение подписи сохранено,
готовность магазина из него не следует.

В audited HEAD `3c2838b99f8445f83f98f7b3c17f1390160d3b4a` **все 270 tracked
клиентских файлов совпадают** с source APK: tree
`02dcff79d7121e25f7264cf849752f364be4a181`; working client diff и nonignored
untracked client files — 0. Сборка/подпись/переустановка и versionCode в текущем
этапе не менялись. [Новые доказательства](evidence/technical-release-completion/apk.json).

## Короткая установка и использование

На S24 уже установлен этот APK23 с сохранённым аккаунтом и 62/62 SHA файлов.
Для обновления другой существующей установки передайте именно указанный файл,
откройте его в файловом менеджере и подтвердите обычное обновление. Если Android
запросит разрешение установок для выбранного файлового менеджера, предоставьте
его этому приложению. Сначала сверить SHA на Windows:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath 'D:\arborscan_backend\output\final-release\ArborScan-1.3.0-23.apk'
# Альтернативная установка через уже разрешённый USB debugging:
adb -s DEVICE_SERIAL install -r 'D:\arborscan_backend\output\final-release\ArborScan-1.3.0-23.apk'
```

При несовместимой подписи или более высокой установленной версии остановиться;
удаление приложения, очистка данных и принудительное снижение code не входят
в сохранное обновление. Приватные каталоги, архивы и ключи получателю APK не передаются.

В «Анализе» добавьте фото и используйте AR либо известный вертикальный эталон.
Эталон и дерево должны находиться примерно на одной глубине. Сохраняйте отчёт;
«История» открывает выбранную серверную версию. Правка контура создаёт новую
ревизию, её отправка на проверку выполняется отдельно; решения доступны по правам
аккаунта. PDF экспортирует выбранную версию с её источниками и единицами;
«Сохранить» открывает системный диалог, «Поделиться» — системное меню.

Offline доступны уже сохранённые локальные/скачанные данные. После восстановления
сети можно обновить историю и повторить незавершённую отправку. Серверные записи
доступны после входа на другом устройстве; локальные черновики и кеш не означают
межустройственную синхронизацию. Перенос телефона средствами Android здесь не
проверялся. Сессия может истечь, тогда требуется вход.

## Release notes и матрица проверки

1.3.0 сохраняет версии отчётов/контуров, источники измерений, карту и исторические
weather/soil snapshots, offline/cache/PDF. Исправлены profile/owner async races,
AR renderer/GLES fallback и отсутствие Android gallery provider. Серверная Google
policy уже развёрнута отдельно08.10 и использует verified claims/стабильный subject.
Технический тег описывает этот состав; APK и серверы не объявляются собранными из
последующего документационного коммита.

| Проверка | Обоснованно сохранённое свидетельство |
|---|---|
| Flutter / native / analyze | exact source23:181 passed /9 passed /0errors,53прежних,0новых; [checks](evidence/final-release/candidate23-checks.json) |
| Android runtime | genuine clean/home/cold24/33/36; overlay/repeat24:73SHA,33:2SHA,36:3SHA; direct18→23/repeat36:8SHA+report/draft restore; [matrix](evidence/final-release/android-matrix23.json) |
| Физический S24 |22→23:62/62SHA до UI, прежний аккаунт/cold history/AR-return/gallery-cancel; [update](evidence/final-release/s24-update23.json), [UI](evidence/final-release/s24-ui23.json) |
| Авторский DEMO / PDF | moderation/conflict/retry, две server versions;2PDF/10pages32+32+8checks, S24PDF5pages12checks; все15pages просмотрены; [workflow](evidence/final-release/moderation23-workflow.json), [PDF](evidence/final-release/pdf23-qa.json), [S24 PDF](evidence/final-release/s24-pdf23-qa.json) |
| Offline / providers | AVD36 offline/cold/PDF/reconnect; настоящие Maps/OpenWeather/partial SoilGrids; GPS S24 cancelled без save; [offline](evidence/final-release/offline23.json), [providers](evidence/final-release/provider23.json) |
| Google / сохранность owner | настоящий API24/old иAPI36/new login/logout/relogin/cold/history/cancel после rollout; [24](evidence/google-security-rollout/native24-ui.json), [36](evidence/google-security-rollout/native36-ui.json), [identity continuity](evidence/google-security-rollout/identity-after36.json). Прежние offline/retry относятся к неизменённому клиенту |

Переиспользование результатов основано на exact APK SHA и совпавшем клиентском
дереве; новый полный Flutter/native/UI/PDF прогон не заявлен. Runtime каждой
версии Android, OTA32→33, S24 Google logout/login и физическая точность этим
пакетом не проверялись.

## Состав runtime

Матрица основана на ранее выполненном read-only срезе08.10,12:32UTC;
новые финальные probes и backup/cleanup — в FINAL_RELEASE.md.

| Компонент | Source / версия / неизменяемая identity |
|---|---|
| APK | source `bbcd5b539427e62da58739fc2dbeff8550ee72c6`;1.3.0/code23; SHA выше |
| API v3 Google | policy `6284f93204f3ef2f92543604eb5c7a2cea93c3cd` поверх live base `5a0b1d63…`; image `sha256:60c6cf1fc185e41aaf6275522df851b60f142887554685ce7aec49b3f702a990`; claims=true |
| API v4 | source `0437e98`; image `sha256:034a9d67b49adb5565a1af9454706a104721e21160d725db32a31f796ff20193`; API/schema `4.0.0-alpha.1` |
| Quality worker | OCI/source `eaa39f415f6de434773cc099d02f407d435c86b1`; image `sha256:3fe28a5b7eb5716a71e984908520c31fff9ac325a9c5885c5e102e93996ac14a` |
| PostgreSQL / workflows | PostgreSQL17.6; contours/history/quality `1/1/1`; `users_google_sub_unique` unique/valid/ready |
| Активная сегментация | version4, segment, confidence0.25/imgsz1024; weights26626466bytes, SHA `626f62b97b05fd65275dc3f909c3aec090f6c9957c4ef9f12c4e743a1f75a2bb` |
| Геометрия / эталон | report method2 / `known_object_segment_v2` |
| AR | `ar_measurement_v4`; height `fixed_vertical_tree_plane_ray_intersection_v1`, DBH `cylindrical_tangent_rays_median_v2` при пригодных входах |
| β / устойчивость | research solver отдельно, не подключён к production; β по фото/AR и проверенная механическая устойчивость недоступны |

[Runtime proof](evidence/google-security-rollout/final-runtime.json),
[method/model manifest](evidence/final-release/release-runtime-manifest.json),
[worker provenance](TECHNICAL_RELEASE.md). Один Git-тег сохраняет разные source
компонентов; он не заменяет их actual image IDs.

## Доступный клиентский откат

**Готового подписанного APK с code выше23 на момент аудита нет.** Наличие старых
APK19–22 не является совместимым сохранным откатом. Доступны source23 в Git,
оба приватных `PrivateKeyEntry` с release fingerprints, валидный private config,
DPAPI credentials, exact lineage и encrypted key escrow. Новый audit проверил
чтение/открытие ключевых записей и совпадение сертификатов; новую подпись не
выполнял. Прежнее фактическое recovery+validation signing сохранено в
[RELEASE_READINESS.md](RELEASE_READINESS.md).

Фактически построен и подписан отдельный recovery APK:
`D:\arborscan_backend\output\final-release\technical-completion-private\ArborScan-1.3.0-recovery.1-24.apk`.
Версия1.3.0-recovery.1/code24, SHA
`35daaa3a324e92abbb4ad789f224f1f58e0c5a963a18c4b61b1081768e87358e`,
115594211B. Сборка155.9s из unchangedclienttree02dcff79/source d3ae1e1.
Recovery24: 9 Android unit tests действительно повторно выполнены (не up-to-date),0fail/0error,39s; UI/install24 отдельно не заявляются.

688 ZIP payload entries побайтно совпали с APK23; отличается только
AndroidManifest.xml (versionCode/name), кроме внешнего signing block.
Подписи24–32/33–36, тот же lineage/installed-data=true и16KiB alignment
реально проверены. [Recovery доказательство](evidence/technical-release-completion/recovery24.json).
Это доступная сборка для сохранного forward recovery при установленном code23,
**не предыдущий функциональный релиз**. На S24 её не устанавливали, перенос данных
23→24 отдельно не проверен; прежние сохранные обновления23 не присвоены24.
После code24 потребуется code25+; принудительный downgrade не применяется.

Сохраняемый helper
`D:\arborscan_backend\output\final-release\technical-completion-private\sign_forward_fix.private.py`
использует остающийся D-репозиторий и D:\ArborScanSigning, DPAPI только в памяти.
Его read-only --verify-release выполнен на23; sign-forward-fix реально использован
для24. Для применения подготовленного recovery после отдельной тестовой проверки:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath 'D:\arborscan_backend\output\final-release\technical-completion-private\ArborScan-1.3.0-recovery.1-24.apk'
adb -s DEVICE_SERIAL install -r 'D:\arborscan_backend\output\final-release\technical-completion-private\ArborScan-1.3.0-recovery.1-24.apk'
```

Для будущего подтверждённого дефекта подготовить compatible source в единственном
D-репозитории, выполнить нужные tests/analyze, собрать code строго выше
установленного, затем подписать теми же keys/lineage. Публичная форма подготовки24:

```powershell
# Из D:\arborscan_backend\arborscan_app. Подтвердить свободное место для сборки.
# Создать только reproducible рабочие каталоги на D; общие SDK сохраняются:
$forwardFixWork='D:\arborscan_backend\output\final-release\forward-fix-work'
New-Item -ItemType Directory -Force -Path "$forwardFixWork\temp", "$forwardFixWork\gradle-home", "$forwardFixWork\pub-cache" | Out-Null
$env:TEMP="$forwardFixWork\temp"
$env:TMP=$env:TEMP
$env:GRADLE_USER_HOME="$forwardFixWork\gradle-home"
$env:PUB_CACHE="$forwardFixWork\pub-cache"
$env:JAVA_HOME='C:\Program Files\Android\openjdk\jdk-21.0.8'
$env:JAVA_TOOL_OPTIONS="-Djava.io.tmpdir=$env:TEMP"
# Штатная release сборка с pub, locked dependencies и сохранёнными SDK:
& 'C:\src\flutter\bin\flutter.bat' build apk --release --build-name=1.3.0-fix.1 --build-number=24
# Android unit tests после release tooling regeneration:
& '.\android\gradlew.bat' -p '.\android' :app:testReleaseUnitTest --no-daemon --console=plain
# Отдельная явная подпись; output/metadata должны отсутствовать:
python 'D:\arborscan_backend\output\final-release\technical-completion-private\sign_forward_fix.private.py' --sign-forward-fix --input 'D:\arborscan_backend\arborscan_app\build\app\outputs\flutter-apk\app-release.apk' --output 'D:\arborscan_backend\output\final-release\ArborScan-1.3.0-fix.1-24.apk' --metadata 'D:\arborscan_backend\output\final-release\signed24.json' --version-code 24
```

Выше — команды будущей исправленной сборки; recovery24 уже построен и подписан,
но не установлен. Если на
момент реального применения установлен code24 или выше, выбрать больший code.
Сначала проверить новый APK и сохранное overlay/repeat/semantic restore на
изолированном DEMO Android, затем `adb install -r` с before/after SHA до UI.
Ни production dump restore, ни удаление приложения не являются частью этого пути.

Безопасная серверная приостановка новых Google-связей использует исправленный
image с `claims=false`, индекс сохраняется. Точные private overrides/argv в
[GOOGLE_SECURITY_ROLLOUT.md](GOOGLE_SECURITY_ROLLOUT.md); возврат старого handler
возвращает уязвимость. Этот откат подготовлен/config-validated, не выполнен.

## Ограничения выбранного выпуска

Копии AS-14 поступают в существующую `D:\ArborScanBackups`, когда Windows включён.
Независимый always-on receiver и восстановление новой VM/public HTTPS отложены
по текущему выбору пользователя. AS-15 ожидает отдельного полного промпта;
визуальная приёмка не назначается. Независимое реальное качество/обучение моделей,
экспериментальные β/ветровая устойчивость/полевая точность сохраняются в
AS-05…AS-07/AS-10/AS-11/AS-13. Street View ограничен доступным покрытием.
Почва/погода могут отсутствовать или быть частичными; их source/time не являются
измерением корней/ветра у дерева. DBH/пористость требуют пригодных входов.
