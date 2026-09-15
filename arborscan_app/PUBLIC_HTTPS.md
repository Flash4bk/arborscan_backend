# Публичный HTTPS ArborScan

Ветка: `codex/public-https`, исходный `origin/main`: `dde5e5c`.
Рабочее дерево до начала было чистым. Изменения подготовлены для публикации
в этой ветке; слияние в main не выполнялось.

## Адреса

- V3: `https://31.57.170.88/api/v3`
- V4: `https://31.57.170.88/api/v4`
- Анализ: `https://31.57.170.88/api/v4/v4/analyze-tree`
- Контуры: `https://31.57.170.88/api/v4/v4/corrections`

Повторение `v4` намеренное. Nginx удаляет только внешний префикс.
`ApiConfig.endpoint` сохраняет его и нормализует разделяющий слеш.
Переопределения `ARBORSCAN_V3_BASE_URL` и `ARBORSCAN_V4_BASE_URL` сохранены.
Подсказки про SSH и adb reverse отображаются только для loopback-адресов.

Проверены запросы профиля, истории, карты, администрирования, feedback,
анализа и контуров. Действующих Railway-адресов не обнаружено.
AR передаёт измерения в текущий запрос анализа v4; собственных HTTP-запросов
в AR-канале и Android AR-коде нет. StickPage вычисляет эталон локально.
Единственная загрузка сетевого изображения в текущем Flutter-коде — аватар:
относительный путь теперь учитывает API-префикс, абсолютный URL сохраняется.
Изображения отчёта и контуров используются в формате bytes/base64.
Алгоритмы измерений, редактор, авторизация и формат контуров не изменены.
Проверка TLS не отключалась.

## Проверки, выполненные агентом (2026-09-15)

- Реальные HTTPS GET обоих `/health` с Windows: `status: ok`, стандартная проверка TLS.
- `flutter test`: 18 тестов прошли, включая сохранение, историю и новый набор URL-тестов.
- `flutter test --dart-define=ARBORSCAN_V3_BASE_URL=http://127.0.0.1:8000/ --dart-define=ARBORSCAN_V4_BASE_URL=http://127.0.0.1:8001/`: 18 тестов прошли.
- `flutter analyze`: 108 замечаний, код завершения 1; 0 ошибок, 7 warnings,
  101 info. Сравнение с анализом исходного main, без номеров строк: новых и
  исчезнувших диагностик нет.
- `flutter build apk --debug`: успешно, без dart-define.
- APK: `build/app/outputs/flutter-apk/app-debug.apk`.
- `git diff --check`: ошибок whitespace нет.
- Android: после подтверждения USB-отладки обнаружен Samsung Galaxy S24 Ultra
  (SM-S928B). APK без dart-define установлен поверх приложения через `adb install -r`:
  `Success`; данные приложения не очищались. Запуск через `am start -W`:
  `Status: ok`, `LaunchState: COLD`.
- Reverse для `tcp:8000` и `tcp:8001` проверены: оба правила уже отсутствовали;
  адресные команды удаления вернули `listener not found`. Другие правила и
  SSH-сеансы не затрагивались.

## Проверки, подтверждённые пользователем

На Samsung Galaxy S24 Ultra с отключёнными USB и Wi-Fi, через мобильный интернет,
пользователь подтвердил успешные вход в аккаунт, анализ фото, редактирование и
сохранение контура. После принудительной остановки приложения через настройки
Android и повторного запуска оригинальное фото и PNG-маска загрузились из
«История → Сохранённые контуры». Результат ручных действий подтверждён пользователем,
а не автоматизированной проверкой агента.

Пользователь также сообщил об успешном пробном продлении сертификата на VPS:
`certbot renew --cert-name arborscan-ip --dry-run --run-deploy-hooks` завершился
с результатом `all simulated renewals succeeded`. Проверка Nginx в deploy-hook
также успешна. Агент этот успешный запуск самостоятельно не выполнял.

## Команды для повторной проверки на телефоне

Подключите телефон с USB-отладкой и подтвердите доступ к нему. В PowerShell:

```powershell
cd D:\arborscan_backend\arborscan_app
$adb = 'C:\Users\danik\AppData\Local\Android\sdk\platform-tools\adb.exe'
& $adb devices -l
# Замените DEVICE_ID на серийный номер выбранного телефона.
& $adb -s DEVICE_ID reverse --remove tcp:8000
& $adb -s DEVICE_ID reverse --remove tcp:8001
& $adb -s DEVICE_ID reverse --list
& $adb -s DEVICE_ID install -r build/app/outputs/flutter-apk/app-debug.apk
& $adb -s DEVICE_ID shell am start -n com.example.arborscan_app/.MainActivity
```

Если конкретного reverse-правила нет, команда удаления сообщит об этом;
другие правила не удаляйте. SSH-сеансы закрывать не требуется.
Альтернатива установке APK: `flutter run -d DEVICE_ID` без старых dart-define.

Для повторения уже подтверждённого сценария отключите USB и Wi-Fi,
оставьте мобильный интернет и откройте приложение с иконки:

1. Войдите в профиль через мобильный интернет.
2. Проанализируйте фото, исправьте контур и нажмите «Сохранить контур».
3. Дождитесь именно «Сохранено».
4. Полностью остановите приложение (при необходимости командой ниже), затем
   откройте его снова и перейдите в «История → Сохранённые контуры».
5. Откройте запись: должны загрузиться оригинальное фото и PNG-маска.

```powershell
& $adb -s DEVICE_ID shell am force-stop com.example.arborscan_app
& $adb -s DEVICE_ID shell am start -n com.example.arborscan_app/.MainActivity
```

## Проверка конфигурации продления агентом

SSH к `arborscan@31.57.170.88` работает. Certbot: `/snap/bin/certbot`, версия 5.8.0.
Таймер `snap.certbot.renew.timer`: enabled и active; на момент проверки следующий
запуск был назначен на 2026-09-15 23:28 UTC.
Исполняемый deploy-hook `/etc/letsencrypt/renewal-hooks/deploy/arborscan-nginx-reload`
содержит `set -eu`, `/usr/sbin/nginx -t` и `/usr/bin/systemctl reload nginx`.

Первоначальная попытка агента выполнить dry-run с `sudo -n` остановилась:
`sudo: a password is required`. Затем пользователь выполнил интерактивную проверку
успешно, как указано выше. Команда для будущей повторной проверки:

```powershell
ssh -t arborscan@31.57.170.88 "sudo /snap/bin/certbot renew --cert-name arborscan-ip --dry-run --run-deploy-hooks"
```

Пароль вводится только в терминале sudo, не в чате. Конфигурация VPS не менялась,
контейнеры не перезапускались, SSH-сеансы пользователя не закрывались.
