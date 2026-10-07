# AS-16 — Android-кандидат: подготовка и фактическая матрица

Срез 07.10.2026. Подготовка и фактические проверки APK19–23 разделены по версии.
**Текущий подписанный кандидат — APK23; доступная матрица выполнена.**
APK22 имеет четыре успешных сравнения обновления/repeat на API24/36 и три
чистые установки/home/cold, но выявил native gallery failure в ATD33 без picker.
Старые invalid baseline19/22 AVD сохранены без repair/uninstall/clear.
Матрица минимум24/граница33/актуальный36 проверена; runtime каждой версии Android,
OTA32→33, Google login и стабильный выпуск этим результатом не заявлены.
Основание: `AS16_FINAL_RELEASE_PROMPT.md`, канонический план 1.23 (актуальные
результаты root уже внесены в 1.24),
[ANDROID_SIGNING.md](ANDROID_SIGNING.md) и реальные доказательства
[sdk-matrix18.json](evidence/release-readiness/sdk-matrix18.json).
Android matrix runner не менял Git, Flutter/Gradle-сборку, ключи, S24, VPS и БД.
Подписанную сборку и проверку S24 выполнил root отдельно; ниже указано, какие
результаты получены на реальных AVD, а какие ещё ожидают исполнения.

## Текущий кандидат: APK23 — фактические результаты

`1.3.0+23`, source `bbcd5b539427e62da58739fc2dbeff8550ee72c6`,
SHA-256 `71f38189ee62a7825029ec7c25bf635891d3adcd0fa3eea3d9b618a223e4a31c`,
115594211 байт. [Static23](evidence/final-release/artifact23-independent-static.json)
реально проверила manifest, nondebuggable, min24/target36, старый signer24–32,
постоянный signer33+, embedded lineage и 16K alignment подписанного output.
Эта проверка не читала ключ или private signing config и не заменяет runtime.

Старый existing36 после host cold boot не смог прочитать прежний base.apk22:
[PackageManager guard](evidence/final-release/api36-before23-package-scan-failure.json)
сохранил settings22, но scanned package/versionName/path отсутствовали. Android
сам удалил invalid package; известный прежний путь после этого дал ENOENT.
Исходный hash/subtype parse error недоступны, причина не доказана. До установки23
или действия с данными guard остановился. Этот AVD5582 и старый invalid33/5584
не ремонтировались и не очищались. Прежние74/74 SHA перехода19→22 остаются
доказательством именно того исторического момента, а не успешного22→23.

На отдельном **исправном task-created lab36/5594** реальные
[22→23](evidence/final-release/api36-overlay23-summary.json) и
[repeat23](evidence/final-release/api36-repeat23-summary.json) сохранили3/3 файла
пяти папок до запуска UI: changed/missing/new=0. Это ограниченная prefs/cache
fixture, не прежние74 representative files. Затем выполнена настоящая
[перезагрузка Android OS23](evidence/final-release/api36-lab23-guest-reboot.json):
boot изменился, нормальный ADB прочитал установленный base.apk с exact SHA до/после,
PackageManager снова распознал23, home открылась. Host AVD cold boot после23
в этом доказательстве не заявлен.

Для прямого18→23 без переименования цепочки task-created lab36 проверен strict
unauthenticated fixture guard и удалено только собственное приложение/test.
[Genuine18 install](evidence/final-release/api36-direct18-genuine-install.json)
подтвердил absent-before-install, exact readiness18 SHA, actualSDK36, release18,
тот же permanent signer,0 файлов перед firstUI и доступный штатный image picker.
Это тот же lab Android OS, не новая VM. [Normal UI baseline18](evidence/final-release/release18-lab36-baseline.json)
реально создана без custom seeding/run-as: synthetic JPEG256×384, четырёхточечный
контур/три отрезка, сохранённый локально1m эталонный отчёт5.00m/2.93m и отдельный
несохранённый3m черновик с неполной геометрией. Server report/correction writes=0.
[Прямой18→23](evidence/final-release/api36-direct18-overlay23-summary.json) и
[repeat23](evidence/final-release/api36-direct18-repeat23-summary.json) сохранили8/8
файлов пяти папок **до UI**, changed/missing/new=0, actualSDK36/code23/cert совпали.
После этого настоящая [Android OS reboot23](evidence/final-release/api36-lab36-direct18-23-guest-reboot.json)
также прошла: base.apk SHA до/после exact, PackageManager/home исправны.
Эти8 файлов — новый ограниченный normal UI dataset, не прежние74 historical files.
Semantic restore того же отчёта и черновика прошёл после Android OS reboot:
сохранённый отчёт5.00м/2.93м и отдельный ввод3м открыты обычным UI.
[Семантический proof](evidence/final-release/release23-lab36-direct18-semantics.json).
Фото отдельно по пикселям не сравнивалось; его файл вошёл в8/8 SHA.

[Матрица23](evidence/final-release/android-matrix23.json) включает следующие
самостоятельные проверки именно этого APK:

- API24 existing22→23 и repeat сохранили73/73 файлов до UI; home/cold прошли.
  [Overlay](evidence/final-release/api24-overlay23-summary.json),
  [repeat](evidence/final-release/api24-repeat23-summary.json).
- Исправный отдельный API33 lab22→23/repeat сохранил2/2 небольших prefs/cache
  файлов. Это не прежний invalid5584 и не представительский набор73 файлов.
  [Overlay](evidence/final-release/api33-overlay23-summary.json),
  [repeat](evidence/final-release/api33-repeat23-summary.json).
- Genuine clean23 с отсутствующим applicationId перед установкой и0 файлов
  прошёл на24/33/36. Реальные home и force-stop/cold restart проверены отдельно
  от static package inspection: [24](evidence/final-release/api24-clean23-ui.json),
  [33](evidence/final-release/api33-clean23-ui.json),
  [36](evidence/final-release/api36-clean23-ui.json).
- Actual Gallery24 открыл DocumentsUI и Back вернул responsive analysis.
  На stripped33 exact GET_CONTENT provider отсутствует: normal tap и повтор
  показывают friendly inline error после обычной прокрутки, Profile→Analysis
  работает, тот же PID, нового Reply/fatal/ANR нет.
  [Red→green23](evidence/final-release/api33-gallery23-result.json).
- API36 без ARCore реально показывает системный Google Play запрос входа.
  Back/cancel/retry возвращают то же фото и process; ARCore остаётся отсутствующим.
  Это не завершённая установка или tracking.
  [Actual23](evidence/final-release/api36-ar23-installer-cancel.json).
- После force-stop/sync и штатного завершения AVD выполнен ещё **host cold boot**
  API36. Normal ADB читает base.apk с exact SHA, PackageManager/code23/signer
  исправны, home открылась, без repair/clear/reinstall на этом запуске.
  [Host cold boot](evidence/final-release/api36-final23-visible-host-cold.json).

Старые invalid33/36 состояния остались нетронутыми; их причина не установлена.
Чистое приложение проверялось только в собственных task-created labs, после
ownership/unauthenticated guards; пользовательский S24 не очищался. Полный
авторский UI/Google и S24 фиксируются root в отдельных документах и не следуют
из матрицы подписи/установки.

## Исторический фактический кандидат: APK22

`1.3.0+22`, source `e78c29d09ef265c7114694288d432d5d835293f6`,
SHA-256 `65835130dfd35e36460ebcdde8f1c5d85761c3b256845e21611f3c07007ea87f`,
115594211 байт. [Static22](evidence/final-release/artifact22-independent-static.json)
проверила именно signed output: прежний applicationId, min24/target36,
`debuggable=false`, INTERNET, реальные подписи по диапазонам, embedded lineage
и 16K native alignment. Никакие private signing config/key не читались runner.
Root отдельно выполнил [автоматические проверки22](evidence/final-release/candidate22-checks.json):
163 Flutter, 9 Android после release regeneration, analyze0errors/53прежних.
Эти проверки не заменяют результаты реальных Android ниже.

[Matrix22](evidence/final-release/android-matrix22.json) содержит четыре свежих
before/after сравнения **до UI**: existing19→22 и repeat того же APK22 `-r`
на actual API24/36. Все 73/73 и74/74 файла пяти папок совпали по SHA; changed,
missing и new=0. API24 показывает прежний сертификат, API36 — постоянный новый.
74 — фактический свежий DEMO inventory после отдельного UI19, не старое число71.
Repeat не называется новым versionCode или выполненным откатом.

В [existing API33](evidence/final-release/api33-existing19-overlay22-blocked.json)
Android при cold boot не прочитал старый base.apk19 и сам пометил его invalid.
Settings ещё содержали code19, но scanned package/versionName/pm path отсутствовали.
Guard остановился **до установки22 и любых действий с данными**. Причина
повреждения старого APK не доказана. AVD сохранён без repair/uninstall/clear;
19→22/repeat22 API33 не отмечаются пройденными.

Три отдельных clean lab подтвердили genuine absence applicationId перед
установкой22, actual SDK/cert/nondebuggable и пустые пять папок до первого запуска:
[24](evidence/final-release/api24-clean22-summary.json),
[33](evidence/final-release/api33-clean22-summary.json),
[36](evidence/final-release/api36-clean22-summary.json).
Реальная home и force-stop/cold restart прошли на всех трёх:
[UI24](evidence/final-release/api24-clean22-ui.json),
[UI33](evidence/final-release/api33-clean22-ui.json),
[UI36](evidence/final-release/api36-clean22-ui.json).
Первый XML холодного API36 снимка попал в transition и не был сохранён;
последующий настоящий home/PID подтвердил результат, старый XML не использовался.
API24 — чистая установка приложения в том же созданном lab OS после strict
[fixture21/no-account guard](evidence/final-release/api24-lab21-removal-before-clean22.json),
а не заявление об очередной переустановке Android. Existing AVD и S24 не очищались.

В новом API33 дополнительно выполнена настоящая перезагрузка **Android OS**:
[PackageManager снова прочитал22 и home открылась](evidence/final-release/api33-clean22-android-reboot.json),
[actual screenshot](evidence/final-release/api33-clean22-after-android-reboot.png).
Это отделяет исправный fresh22 package от неизвестной причины invalid старого19.
Но normal Gallery tap в stripped AOSP ATD без DocumentsUI/GET_CONTENT handler
выявил [native `Reply already submitted`](evidence/final-release/api33-clean22-gallery-result.json),
[target-only stack](evidence/final-release/api33-clean22-gallery-failure.txt).
Exact22 R8 mapping связывает стек с ImagePickerDelegate/DartMessenger Reply.
Этот сценарий **failed**; узкий provider preflight включён в23 и проверяется
отдельно, а исходный22 результат не записан
как просто успешный unsupported-provider возврат. AR/photo flow33 не достигнут.

В [clean24 AR22](evidence/final-release/api24-ar22-result.json) два responsive
AR→same-tree→return сохранили PID и те же пиксели synthetic DEMO фото,
показали [GLES2 unavailable](evidence/final-release/api24-clean22-graphics-fallback.png)
и прошли cold restart без нового fatal/ANR. Это повтор native guard на exact22,
а не переименование21 proof. [Фото после return](evidence/final-release/api24-clean22-photo-retained.png).
ARCore installer/tracking при GLES2 здесь не достигается.

В [clean36 AR22](evidence/final-release/api36-ar22-installer-cancel.json)
normal Android PhotoPicker выбрал synthetic DEMO JPEG. При actual GLES3 и
отсутствующем ARCore после lab-only temporary camera permission появился
настоящий installer: текст «Installing Google Play Services for AR…» подтверждён
actual UI XML/assertion в JSON. [Сохранённый screenshot22](evidence/final-release/api36-clean22-ar-installer.png)
попал в белый переходный кадр и **не показывает этот текст визуально**;
это исторический transition screenshot, не доказательство видимого installer.
Системный BACK отменил операцию: [MainActivity/то же фото](evidence/final-release/api36-clean22-ar-cancel-photo-retained.png),
PID тот же, photo crop пиксельно одинаков, ARCore всё ещё отсутствует, нового
app fatal/ANR нет. Завершённая установка сервиса и tracking не заявлены.

Все testing AVD после этой матрицы штатно остановлены; existing36/full UI22
не передавался следующему агенту до решения root о gallery fix. S24 проверяет
root отдельным evidence, этот runner его не трогал. API28–32 имеют static
проверку диапазона; runtime на каждой Android версии и OTA32→33 не заявлены.

## Исторический AR результат: APK21 red→green

`1.3.0+21`, source `495ea37900c18ecbe2dcf0dbda81714654691ab8`,
SHA-256 `ea03f87679f68056429b914b2a49b5b24f52ce9b033dc9ee3750e22ab94b313f`,
115577827 байт. [Независимая static21](evidence/final-release/artifact21-independent-static.json)
сверила реальные manifest/signatures/lineage/16K alignment. GLES<3/unknown
проверяется до inflation; cached `EngineInstance.getEngine()` также вызван до
создания Fragment. Headless mode, geometry и модели не менялись.
Основание GLES requirement — [официальный sample Google Sceneform](https://github.com/google-ar/sceneform-android-sdk/blob/master/samples/gltf/app/src/main/java/com/google/ar/sceneform/samples/gltf/GltfActivity.java);
public `EngineInstance.getEngine(): IEngine` и раннее создание cached Filament
проверены javap в реально используемом locked core1.23.0, без смены dependency.

В том же новом API24 lab OS/AVD удалены только fixture20 и собственный probe
после [strict identity/package/certificate/unauthenticated UI guard](evidence/final-release/api24-lab20-removal-before-clean21.json).
Затем [настоящая clean install21](evidence/final-release/api24-clean21-summary.json),
видимая home/PID и force-stop/cold restart прошли. Первый слишком ранний semantics
snapshot ещё не содержал home; последующий реальный снимок и cold launch её
подтвердили, что явно отмечено в [UI21](evidence/final-release/api24-clean21-ui.json).
Это не переустановка ОС или новый Android image.

[Actual AR21 proof](evidence/final-release/api24-ar21-result.json): выбран известный
синтетический DEMO JPEG через системный DocumentsUI, два последовательных
AR→same-tree confirmation→responsive return в MainActivity. PID тот же,
photo region побайтно одинаков в трёх настоящих screenshots до/после обоих
возвратов; [фото осталось](evidence/final-release/api24-clean21-photo-retained.png).
Прокрутка реагирует и показывает [понятную графическую недоступность](evidence/final-release/api24-clean21-graphics-fallback.png).
Cold restart после повторного AR также вернул home; текущие AndroidRuntime/
ActivityManager logs не содержат нового fatal/ANR. **AR return21 passed.**
Само выбранное в main фото после process death здесь не заявлено сохранённым;
проверка сохранённого измерительного черновика проводится отдельным UI-пакетом.

GLES2 lab не достигает ARCore installer и не доказывает install/cancel/tracking
или физическую точность. Lab штатно остановлен. Existing36 overlay21 и clean33/36
21 не выполнялись: перед финальной матрицей независимый аудит нашёл отдельную
legacy-owner hydration race в Profile UI, исправленную для22. Доказательства21
сохраняются отдельно; фактическая матрица22 описана выше.

## Фактический промежуточный APK19

Root собрал и подписал `1.3.0+19`, source
`91f17b5b936fba48ad297a2848eea1761757b785`, 115577827 байт. SHA-256:
`c0fd15a4c2e9280984df3b4b21d58440ce26530388710858fb2efee0f66df30b`.
Независимая [статическая проверка](evidence/final-release/artifact19-independent-static.json)
подтвердила min24/target36, INTERNET, `debuggable=false`, v2/v3/v3.1 по
диапазонам, прежнюю lineage с data capability без rollback и native 16K alignment.
Это проверка настоящего signed output; ключи/config не читались.

[Android matrix19](evidence/final-release/android-matrix19.json) содержит шесть
реальных five-folder SHA сравнений **до любого запуска UI**: 18→19 и повтор
того же APK19 `-r` на API24/33/36. Совпали все 73/73/71 файлов соответственно,
changed/missing/new=0. Actual SDK и текущий signer прочитаны через PackageManager.
Старые AVD, аккаунты и snapshots не очищались; raw inventories остаются приватными.

В новом пустом API24 доказано отсутствие applicationId, clean install APK19,
первый видимый запуск и cold restart в MainActivity. Выбор синтетического DEMO
JPEG через системный DocumentsUI и подтверждение AR выявили **реальное падение**:
`ArSceneView` layout inflation → Filament `Engine.create` →
`IllegalStateException: Couldn't create Engine`. Оно возникает до обработки
ARCore unavailable/install. [Безопасный stack excerpt](evidence/final-release/api24-ar19-failure.txt),
[Android crash dialog](evidence/final-release/api24-clean19-ar-crash.png),
[cold restart home](evidence/final-release/api24-clean19-cold-home.png).
Это конкретный renderer failure в эмуляторе, но приложение должно возвращать
понятную недоступность без process crash; критерий AR return APK19 **не пройден**.
Root подготовил узкий guard, не меняя геометрию. Clean API33/36 APK19 не запускались:
проверка переносится на исправленный higher-code APK, а не объявляется успешной.

После фиксации ошибки clean24 штатно остановлен. Существующий AVD36 с +19
возвращён для отдельного UI-агента: boot=1, actualSDK36/code19/nondebuggable,
MainActivity запущен. Работа матрицы не делала logout/clear/wipe либо Google login.

### Фактический повтор на APK20: crash устранён, AR return ещё не пройден

Root собрал и подписал `1.3.0+20`, source
`750e9b61d2c8a49982e776093f0a8508e88dfa45`, SHA-256
`fed4c9f1cfc4c467ba8e0c83a651afc4aefc20a7ef7fca689252136edc6b1833`,
115577827 байт. Независимая [статическая проверка20](evidence/final-release/artifact20-independent-static.json)
снова подтвердила release manifest/signers/lineage/native alignment.

Для clean20 в **том же новом лабораторном API24 OS/AVD** удалены только ранее
установленные fixture APK19 и его `.test`: перед удалением проверены exact
AVD/SDK24/code19/nondebuggable/old certificate и реальный профиль с предложением
входа/регистрации, без аккаунта. Существующие AVD/S24 не удалялись и не очищались;
OS/AVD не сбрасывался. Затем настоящее отсутствие applicationId, clean install20,
первый видимый запуск, PID и force-stop/cold restart прошли; это чистая установка
приложения, не заявление о создании второй VM/переустановке Android.

Повтор выбора того же DEMO JPEG и подтверждения AR на20 поймал InflateException.
Процесс сохранился, MainActivity вернулся в foreground и фото визуально осталось,
но интерфейс перестал отвечать: Android показал **«ArborScan isn't responding»**.
Реальный `feature:reqGlEsVersion=0x20000` и EGL_BAD_CONFIG подтверждают недостаточную
графическую поддержку API24 lab. Разрешённое read-only чтение ANR trace, без
`adb root` или обхода доступа, локализовало main-thread teardown:
DefaultSpecialEffectsController → FragmentStateManager → FragmentManagerImpl →
FragmentActivity/AppCompatActivity/ArMeasureActivity.onDestroy.
[Actual20 результат](evidence/final-release/api24-ar20-result.json),
[target-only stack](evidence/final-release/api24-ar20-anr.txt),
[системный ANR dialog](evidence/final-release/api24-clean20-ar-anr.png).
Наличие процесса и возвращённого фото **не заменяет responsive return**.
ARCore installer не достигнут; установка/отмена ARCore здесь не проверены.

Lab24 штатно force-stop/emu-kill после сохранения доказательств. Матрица20
existing24/33/36 и clean33/36 была остановлена до исправления graphics preflight/
частичной inflation; эти строки20 не считаются выполненными. Actual21 AR guard
проверен отдельно выше.

## Проверенный исходный срез

- На момент чтения новый worktree имел чистый detached HEAD
  `10b54e77ae77fa7a51621bc1902889ce4d2a8ef6`. Ветка/интеграция — отдельное действие.
- В исходниках `version: 1.3.0+18`; установленный кандидат имеет имя
  `1.3.0-readiness.1cf6ce3`, code 18. SHA APK:
  `de4b262e6983b1d6941383f97e1c30bd4e1388c9d3e23c7714e784095f398033`.
- `applicationId=com.example.arborscan_app`, minSdk 24, target/compileSdk 36,
  AGP 8.9.1, Gradle 8.11.1, Kotlin 2.1.0. Release unsigned, `debuggable=false`.
  Инструмент подписи применяется после сборки; debug fallback отсутствует.
- Google-зависимости lock-файла: `google_sign_in` 6.3.0,
  `google_sign_in_android` 6.2.1. Проверки OAuth принадлежат отдельному пакету;
  сохранённая email/Google-сессия не доказывает новый Google-вход.
- `git ls-remote --tags origin` показал только `v1.1.0-tech.1` и
  `v1.1.0-tech.2`. Стабильного `v1.3.0` в этом срезе нет. Перед созданием тега
  проверить origin повторно; технические теги не перемещать.

Предложение для пользовательской версии: **1.3.0**, ближайший code **19**.
Перед выпуском подтвердить, что не выдан кандидат с code 19 или выше; тогда
использовать следующий свободный code. Новый артефакт получает собственные SHA,
metadata и имя файла. Если обязательный Google OAuth остаётся заблокирован,
это проверенный кандидат, без стабильного release/tag и заявления завершения.

## Сохраняемый контракт подписи

| Реальный Android | Действующий сертификат SHA-256 | SHA-1 для разрешённых регистраций |
|---|---|---|
| API24–32 | `68ff9861f52fdb22744b30ee27092008097f9eff4ead7eb38f80e662115f5b97` | `2c08b4d637574c16be5cba1a595a4baac8541a4c` |
| API33+ | `7fb94ede4099199db9166609c34d6278ebe51c7d80bd6ce1cd0208bf863863b8` | `e4dfd76dc3ec5f96be6b740949c678aebf1f110e` |

`rotation-min-sdk=33`; lineage SHA-256
`93d10d0e6b813bf9053debf1cc879378a4e90e2d0e07ac755eba50eb57a87dfe`.
API24–32 продолжают использовать прежний debug-сертификат. Это совместимость
прежней установки, не переход всех платформ на постоянный signer и не готовность
к публикации в магазине. SHA APK не является отпечатком сертификата.
Не создавать новую lineage и не повторять восстановление ключа.

## Сборка при ограниченном C

Фактический desktop-срез: C свободно 309784576 байт, D — 31085649920 байт.
Путь SDK остаётся `C:\Users\danik\AppData\Local\Android\Sdk`; Flutter —
`C:\src\flutter`. Уже установленный SDK читать, не скачивать/обновлять на C.
В этой задаче не удалять пользовательские файлы, старые APK, AVD, backup или ключи.

Существующий readiness build является junction на
`D:\arborscan_backend\output\release-readiness\flutter-build` (1.30 GB).
Он и доказательства +18 остаются сохранёнными. Для нового worktree использовать
**другой** output, чтобы старый артефакт не оказался результатом новой сборки.
Полный `C:\Users\danik\.gradle` занимает 12.13 GB. Достаточно копирования
проверенных относящихся к текущему Gradle каталогов, без изменения оригиналов:

| Кэш | Размер в срезе |
|---|---:|
| `caches/modules-2` | 1484535078 байт |
| `caches/8.11.1` | 4444396379 байт |
| `wrapper/dists/gradle-8.11.1-all` | 707501724 байт |
| `caches/jars-9` + `build-cache-1` | 42359598 байт |
| Pub cache | 663348357 байт |

Последовательность подготовки; выполненные шаги и оставшиеся проверки разделены ниже:

1. Дождаться завершения любых Gradle/Flutter jobs. Снова проверить свободное
   место и зарезервировать не менее 8 GB сверх копируемых кэшей. Не запускать
   Gradle и два AVD одновременно: в этом срезе доступно около 3.3 GiB RAM.
2. Создать новые task-owned каталоги под
   `D:\arborscan_backend\output\final-release`: `gradle-home`, `pub-cache`,
   `temp`, `flutter-build`, `dart-tool`. При существующем каталоге сначала
   выяснить владельца/содержимое; не выполнять recursive delete или overwrite.
3. Скопировать пять перечисленных Gradle-подкаталогов в соответствующие
   относительные пути нового `gradle-home`, Pub cache — в новый `pub-cache`.
   Не копировать произвольные `.gradle/gradle.properties`, credential-файлы,
   daemon/worker state или signing material. Источники не перемещать.
4. В свежем worktree создать junction `arborscan_app/build` → новый
   `flutter-build`, `.dart_tool` → новый `dart-tool`. Пути должны отсутствовать;
   если они уже появились, проверить их до действий. `android/.gradle` также
   направить в отдельный task-owned D-каталог до первого Gradle запуска.
5. Только в окружении текущего процесса задать `GRADLE_USER_HOME`, `PUB_CACHE`,
   `TEMP`, `TMP` на новые D-каталоги; Java — существующий JDK21. В новом
   `android/local.properties` задать только прежние `sdk.dir` и `flutter.sdk`.
   Не копировать stale versionCode/versionName или Google/серверные секреты.
6. `flutter pub get --offline` создаёт **новый** package_config для текущего
   checkout/Pub cache. Старый `.dart_tool/package_config.json` не переносить:
   он содержит ссылки на другой checkout. При недостающей зависимости разрешено
   получить её обычным HTTPS в новый D-cache; не обновлять lock без основания.
7. Полный Flutter suite и analyze текущего исходника; затем release build без
   прежних dart-define со штатным pub/platform-tooling regeneration; **после
   release build** Android release unit tests. Не запускать их параллельно,
   если операции меняют registrants/metadata/выходные файлы.

### Фактически выполненная подготовка D

07.10 завершено копирование выбранных кэшей (6.679 GB Gradle, 0.663 GB Pub)
и reusable старого build (1.298 GB). Все исходные каталоги сохранены. Прерванное
копирование продолжено без очистки уже законченных частей; robocopy exit 0/1,
итоговые количество и объём сверены. Повторный root offline pub get завершился
успешно после готовности D; это разрешение зависимостей, не прогон тестов.

Новый worktree имеет отдельные junctions `build`, `.dart_tool`, `android/.gradle`
на task-owned каталоги `D:\arborscan_backend\output\final-release`; старый
physical readiness build не изменён. Закрытый DACL проверен на корне и образце
скопированного файла: только текущий пользователь, SYSTEM, Administrators.
`Set-Acl` потребовал SACL-привилегию; применён существующий подход проекта
`icacls` только для DACL, без смены владельца или обхода прав. После копирования
D свободно 22137708544 байт, C — 283992064 байт (временной срез, перепроверять).

Приватный `Invoke-FinalRelease.private.ps1` выставляет процессные D-пути для
GRADLE/Pub/TEMP/TMP/Java temp и Java21. Команда `-Action tests -TestFiles
'test/profile_google_session_test.dart'` ограничивает запуск выбранным файлом;
без TestFiles выполняется полный suite. До Gradle требуется 4 GiB свободной RAM.
Подготовка не запускала Flutter tests/analyze/build либо Gradle.

На **существующем DEMO AVD36**, без действий с S24, реально переустановлен
неизменённый signed audit probeV3 (SHA `c8592b96cfff67f8de4c4d9efa15eeff8b9622b4d15d2fbe041bbd822567d1c3`).
Приложение остановлено, получен приватный до-19 SHA-инвентарь пяти папок:
71 файл (61 files, 6 shared_prefs, 4 app_flutter; no_backup/databases пусты).
PackageManager подтвердил SDK36, code18, новый signer `7fb94…`. UI, logout,
uninstall и data clear не выполнялись. Это baseline, не доказательство 18→19.

Приватный `android19_matrix.private.py` подготовлен и синтаксически проверен;
runtime-операции до появления exact signed19 не выполнялись. Отдельно подготовлены
только новые config/ini для clean API24/33/36 под приватным DACL: без userdata,
snapshots, запуска или установки. API36.1 использует уже установленный
`google_apis_playstore` x86_64 revision 4; исходные images не изменены/не загружались.
Runner проверяет реальные
manifest/signatures/embedded lineage/16K alignment без чтения signing config,
принимает только заданный SHA/code/name, работает только с whitelisted emulator
serials и имеет отдельные режимы overlay, repeat, clean-prepare/start/install.
Repeat — повтор **того же** APK с `-r`, с before/after SHA до UI; это проверка
повторной установки, не новый versionCode и не выполненный сохранный откат.
Clean-start запрещён при другом активном эмуляторе; старые AVD не очищаются.

Историческая публичная форма выполненной сборки/подписи22 после фиксации code и sourcee78c29d
(исторические APK19 имеют отдельные metadata, их команды не являются текущими):

```powershell
# Из arborscan_app, после процессного окружения и нового pub get:
C:\src\flutter\bin\flutter.bat build apk --release --build-name=1.3.0 --build-number=22
# Из корня текущего worktree; private config не читать и не печатать:
python deploy-vps/ops_android_signing.py sign --config D:/ArborScanSigning/signing.private.json --input arborscan_app/build/app/outputs/flutter-apk/app-release.apk --output D:/arborscan_backend/output/final-release/ArborScan-1.3.0-22.apk --metadata D:/arborscan_backend/output/final-release/signed22.json --minimum-version-code=22
python deploy-vps/ops_android_signing.py verify --config D:/ArborScanSigning/signing.private.json --apk D:/arborscan_backend/output/final-release/ArborScan-1.3.0-22.apk --minimum-version-code=22
```

Не использовать команду `clean` как освобождение места. Выход подписанного APK
не должен существовать: signing tool запрещает замену. После подписи APK не менять.
Для свежего worktree Flutter 3.41 release build выполняется **со штатным pub**:
`--no-pub` пропускает release platform-tooling regeneration и может оставить
Java registrant от dev-плагина `flutter_native_splash`, которого release Gradle
не включает. Первый native-test/build отказал на missing
`FlutterNativeSplashPlugin`; эти попытки не считаются успешными проверками.
Private build helper исправлен root: default pub, без обновления SDK/зависимостей.
Тесты Gradle выполняются после release regeneration; актуальные итоги будут
зафиксированы отдельно после завершения, а не по прежней сборке +18.
Реальную подпись запускает root через существующий приватный DPAPI helper,
который готовит необходимые секреты только в памяти/окружении дочернего процесса.
Показанные CLI — публичная форма команды; чтение приватного config или запуск
без необходимого приватного окружения не является подготовкой ключа.
Сверить versionName/code, non-debuggable, min/target SDK, сертификаты каждого
диапазона, native alignment и hash именно подписанного выходного файла.

## Реальные установки и ещё не выполненная часть

Результаты APK19 ниже фактически исполнены. Непройденное/ожидающее явно отделено;
APK20 не заменяет их: его clean24 AR return failed, exact21 guard passed.
Final22 signed/overlay/clean проверки реально выполнены выше; API33 existing
overlay и его gallery provider failure сохраняют открытые ограничения.

| Среда | Обновление с сохранением | Чистая установка | Граница проверки |
|---|---|---|---|
| API24 Google APIs, `Codex_AS16_API24_20261006`, D, port 5586 | 18→19/repeat19 и19→22/repeat22: 73/73 SHA, прежний signer | отдельный API24 port5590: clean22/home/cold/два responsive ARfallback прошли | GLES2 unavailable, не ARCore installer/tracking/Google login |
| API33 AOSP ATD, `Codex_AS16_API33_20261006`, D, port 5584 | Исторический18→19/repeat19:73/73; новый19→22 guard остановлен invalid baseline19, не passed | новый API33 port5592: clean22/home/cold/AndroidOS reboot passed; Gallery tap FAILED native Reply already submitted | Нет DocumentsUI/Google providers; old19 не ремонтировался, Google/Maps/AR flow не подтверждены |
| API36.1 `Codex_AS12_20260925`, D, port 5582 | 18→19/repeat19:71/71; свежий19→22/repeat22:74/74 SHA, новый signer | новый API36 port5594:clean22/home/cold passed, отсутствующий ARCore installer/BACK/photo return прошёл | ActualSDK36; никаких tracking/field accuracy claims |
| S24 Ultra (выполнял root) | 18→19: 62/62 SHA до UI, аккаунт/cold history/AR return прошли | не выполнять на телефоне пользователя | [Update19](evidence/final-release/s24-update19.json); AR tracking/field accuracy не заявлены |

SDK images для API24/33 уже сверены с официальными каталогами и SHA; не скачивать
их повторно. Чистые AVD создавать **только с новым именем/каталогом** на D,
используя имеющийся system image, без переноса userdata, snapshots и аккаунтов.
Не применять `-wipe-data` к существующим AVD и не удалять в них приложение.

Подготовленные новые имена: `Codex_AS16_FINAL_CLEAN_API24`,
`Codex_AS16_FINAL_CLEAN_API33`, `Codex_AS16_FINAL_CLEAN_API36`;
каталоги `D:\arborscan_backend\output\final-release\clean-avds\<name>.avd`
под уже проверенным приватным DACL, отдельные свободные порты
5590/5592/5594. Перед запуском проверить, что имена/порты не заняты. Скопировать
только подходящий `config.ini`, сменить avd.id/name, задать абсолютный путь
существующего system image; не копировать `.avd` целиком. Новые AVD использовать
без snapshot save/load (`-no-snapshot`), чтобы не создавать дополнительные
дампы RAM; userdata нового AVD при штатном завершении сохраняется отдельно.

Запускать по одному AVD, `Start-Process -WindowStyle Hidden`, логи/временные
файлы на D. Свободное место проверять перед созданием каждого: старые AVD имеют
разные размеры backing files; длина sparse-файла не равна занятому месту.

В прежнем пакете API24 baseline AVD начинался с debug15, затем обновлён через18
до19. Он не является чистой установкой финального кандидата. API28–32 проверялись
`apksigner verify` по диапазону, не отдельным runtime API32. Сейчас отдельного
API32 image нет. Матрица минимум24/граница33/актуальный36 не должна называться
проверкой на каждой поддерживаемой версии или доказательством Android OTA32→33.

### Порядок проверки одного сохранного обновления

1. Проверить exact signed APK SHA; actual SDK из `getprop`, фактические
   фактический baseline versionCode и signer через PackageManager: первоначально18,
   в свежем переходе22 уже19; damaged package не ремонтировать молча для доказательства.
2. `am force-stop` приложения. Существующий неизменённый signed audit probeV3
   можно использовать повторно после проверки его target/applicationId и
   совместимой подписи; при изменении instrumentation пересобрать и подписать.
3. Получить приватный **до**-инвентарь `files`, `shared_prefs`, `app_flutter`,
   `no_backup`, `databases`. Хранить пути/записи только в приватном D output.
   Для тестовых AVD отметить представительские черновик, локальный эталон,
   две серверные версии, PNG/редактор и несинхронизированные данные.
4. `adb -s <serial> install -r <exact-candidate.apk>`; никакого uninstall,
   `pm clear`, downgrade, подписи сторонним ключом или обхода прав.
5. **До запуска UI** повторить five-folder SHA inventory и PackageManager signer;
   сравнить каждое содержимое, не только количество. При любом изменении/пропаже
   остановиться и выяснить причину. Сравнить фактический code/name/hash.
6. Только после сравнения запускать UI. Проверить семантику записей/владельцев,
   восстановление локального эталона/черновика, выбранную серверную версию,
   PNG и редактор; легитимные последующие cache/session writes отделить от
   неизменности момента overlay-install.
7. Force-stop/cold restart → восстановление; на DEMO AVD offline/cache/PDF →
   возврат сети → реально серверный список. Restore network даже при сбое;
   сообщение об успехе и cached список не заменяют серверную загрузку.
8. Без raw tokens/email/GPS в публичных доказательствах: environment/API,
   exact version/SHA, signer, before/after/identical/changed/missing, конкретные
   безопасные UI assertions и маскированные DEMO screenshots.

Старые 15→16→17→18, API24 TLS baseline и S24 12→18/61 SHA остаются прежними
доказательствами; key recovery/backup/старые переходы без причины не повторять.
18→кандидат — свежая проверка тем же действующим ключом. Если после неё нужен
ещё исправленный APK, повторить candidate→higher-code на DEMO AVD как второе
обновление и сначала сравнить содержимое; не выдавать будущий шаг за выполненный.

### Чистая установка и затронутый функциональный цикл

В отдельном пустом AVD подтвердить отсутствие applicationId, установить final
release, получить real code/signer, запустить и cold restart. Нельзя использовать
существующий аккаунт S24 для Google logout/login или подменять чистую установку
очисткой его данных. Нет AR/Play services — понятная недоступность/возврат без
падения, email-вход остаётся рабочим; это не успешный provider flow.

Затронутый release-цикл проходит агент на отдельных DEMO-записях: вход; новый
отчёт/эталон и восстановление; выбранные версии и новая серверная версия;
ревизия/конфликт/администраторская модерация; карта/доступные GPS/погода/почва;
offline/reconnect; реальные PDF каждой нужной версии, SAF-save/open/cancel и
share-sheet/cancel без отправки. PDF выгрузить, сверить числа/единицы/фото/разметку
и визуально проверить каждую страницу. Логи сервиса не заменяют эти UI-проверки.
Реальный новый Google-вход на старом и новом сертификате — отдельные результаты,
включая отмену/ошибку/возврат/сессию/restart/logout/relogin тестового аккаунта.

## Выпуск и сохранный откат

Тег `v1.3.0` допустим только после закрытия обязательных технических сценариев
и повторной проверки его уникальности; наличие подготовленного stable name
не завершает Google OAuth и открытый AS-14. Root фиксирует actual source commit,
release manifest, push/main/tag и серверную совместимость отдельно.
Откат клиента — исправляющий release теми же ключами/lineage с **большим** code,
совместимым форматом данных и предварительным DEMO overlay-test. Установка старого
readiness18, принудительное снижение code, uninstall или data clear не являются
сохранным откатом. Production API/БД/модели ради имени релиза не меняются.
