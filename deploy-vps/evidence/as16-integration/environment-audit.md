# AS-16 / AS-09 — фактическая сверка среды и карты

Дата: 06.10.2026. Проверен интеграционный checkout `codex/release-integration`,
исходный HEAD `0fe95eb9a6d5c115b8c094bff68addaa86d7d14a`. Сверен полный относящийся
к AS-09 diff относительно main, включая API, snapshot-validation, историю,
Flutter-карту, источники координат, экспорт и существующие тесты.
Новых изменений API, провайдеров, схемы БД или алгоритмов не потребовалось.
Итоговые проверки кандидата и фактическое состояние VPS/APK фиксируются отдельно
в [RELEASE_INTEGRATION.md](../../RELEASE_INTEGRATION.md).

## Компактная матрица

| Функция | Актуальный код / API / хранение | Имеющееся доказательство | Конкретный остаток / граница |
|---|---|---|---|
| Свойства почвы | `environment_api.fetch_soil`: фиксированный SoilGrids WCS, `*_0-5cm_mean`; clay/sand/silt %, SOC г/кг, pH воды. Исходные целые значения делятся на 10; 3×3 TIFF, центральный пиксель, NEAREST | Пять существующих unit-тестов сырой шкалы/единиц; новый реальный прямой WCS-запрос ниже; прежние Windows/S24 +8 и PDF в GEO_ENVIRONMENT | Только пять свойств слоя 0–5 см. Это средние модельной сетки 250 м, не тип почвы или проба у дерева; физические параметры корней/AS-11 не получены |
| Происхождение почвы | envelope: ISRIC/SoilGrids 2.0, 250 м, WCS_nearest_cell, CC BY 4.0/DOI, request_point, retrieved_at, cached, limitations; сохраняется в версии | `test_environment` проверяет unit/depth/source/point, deep-copy и индекс; UI/PDF используют тот же снимок | `data_at`, q05/q95 остаются null. Версия 2.0 не фиксирует отдельный rolling raster release; воспроизводимость отчёта обеспечивают сохранённые значения, не новая выборка |
| Отказ почвы | no-data −32768/TIFF nodata → null; HTTP/TLS/timeout/malformed/quota → явная причина; partial сохраняет доступные свойства | Backend-тесты фактического нуля, отсутствия покрытия, ошибок, partial и позднего результата; Flutter-тесты пропуска | Нет восстановления отсутствующих значений усреднением соседей. При отказе провайдера анализ/сохранение точки остаются доступны |
| Погода | `/v4/environment` через серверный OpenWeather Current Weather; °C, м/с, °, hPa, %; dt → UTC data_at, retrieval отдельно | Предыдущий живой HTTPS/API/S24/AVD цикл 05.10 +11/+12; тесты единиц/дат/ноля/null/TLS/secrets | Это текущая погода провайдера на запрос, а не архив старого фото и не ветер у дерева. Ключ остаётся на сервере; отдельный новый ключ/аккаунт не требуется |
| Неизменяемая история | snapshot.environment v1; `environment_edit` создаёт новую версию с parent, полный отчёт/оригинал не заменяется; summary новых версий несёт тот же envelope | Backend deep-copy/old-version/index tests; две реальные серверные версии и 10 страниц PDF прежнего S24; child weather +11 и HTTP readback | Старые версии не получают текущую погоду/почву при открытии. Валидный клиентский snapshot не объявляется независимо проверенным измерением |
| Место обследования | SurveyPoint: WGS84, source exif/device/manual, timestamp, доступная accuracy/approximate/last_known, camera_or_device либо tree | EXIF original-byte/orientation/hemisphere/timezone tests, backend CRS/range tests; живой GPS S24 05.10 | GPS камеры/телефона не превращается автоматически в точку дерева. Нет EXIF GPS HEIC/PNG; доступны явный GPS или ручная точка |
| Отмена и отказ GPS | Запрос устройства только явно; fresh/last-known показываются с временем и точностью, применение требует действия; отмена сохраняет прежнее место | Прежняя S24-отмена предложения GPS; AVD denied с сохранённой ручной точкой; controller-generation tests | Полевая точность координат не подтверждена. Последняя известная точка не применяется молча; не назначается (0,0) при отсутствии GPS |
| Карта → версия | SurveyMapRepository объединяет owner-bound journal/cache, v4 summary, ограниченное открытие старых details и v3 history; запись/карточка ведут к конкретному version_id | Repository tests нескольких версий/совпадающих координат/legacy discovery; прежний AVD chooser и S24 карта→точная версия | Автоматический центр GoogleMap не сохраняется как место. Старый summary не индексируется обратно на сервере; разные версии остаются разными записями |
| Смена аккаунта | API owner из current_user; private/no-store и Vary Authorization; cache ключ owner/point. Flutter guard token/owner/epoch, controller очищается; journal/cache owner-scoped | Backend cache isolation + route-auth tests; Flutter поздние ответы/диалоги/карточки/403/cache-denied | Проверки других аккаунтов только изолированные. Реальный пользовательский аккаунт не переключается ради аудита |
| Offline | owner journal, previously opened report-cache и saved map-index, оригинал/снимок доступны из локального файла; refresh сохраняет старое при ошибке | Existing service-recreation/network/cache tests; прежний AVD UI offline→force-stop→точный отчёт | Cached reports не означают offline Google Maps tiles или автоматически загруженную всю серверную историю. Свежий S24-кандидат проверяется отдельно; прежний AVD не объявляется S24 |
| Street View | Существующие Google Maps native/web действия; error launching external app имеет штатную ошибку | Наличие действия/отсутствие фиктивных панорам в коде | Пользователь 06.10 принял отсутствие покрытия нужных мест как ограничение. Не дефект, не требует нового провайдера или обхода |
| PDF | ReportExportData.environment + report_pdf печатают frozen GPS/provenance, погоду, почву/depth/units/date/limits, legacy отдельно | report_environment_export_test: разные версии, null/0, прочитанные PDF; прежние реальные S24/AVD файлы | Сохранение/экспорт не делает внешнего lookup. Предыдущие PDF проверены в прежней сборке; финальная версия кандидата требует собственных UI/file evidence |

## Реальные данные SoilGrids, повторная независимая проверка

На Windows выполнены ровно пять прямых read-only HTTPS WCS-запросов для отдельной
тестовой точки `50.2, 10.2` (не координаты пользователя). TLS штатный,
redirects выключены, ответ ограничен 64 КБ; каждый ответ 200, TIFF 3×3,
считан центральный пиксель, слой 0–5 см mean. Получено:

| Свойство | Исходное целое | Сохранённая физическая шкала |
|---|---:|---:|
| clay | 295 | 29.5 % |
| sand | 195 | 19.5 % |
| silt | 510 | 51.0 % |
| soc | 549 | 54.9 г/кг |
| phh2o | 57 | 5.7 pH |

[Полный безопасный протокол](environment-provider-check.json) содержит
реальные SHA-256 ответов, размер/режим TIFF, UTC retrieval и SHA семи сверенных
файлов. Это **проверка прямого провайдера с Windows**, не тест `/v4/environment`
на production и не проверка нового интерфейса S24. Прежнее совпадение этих
значений в S24 +8, сохранённых версиях и PDF остаётся отдельным свидетельством.

## Проверенные первичные документы

ISRIC публикует integer→conventional factors для этих слоёв; использование
0–5 см mean и null вместо отсутствующей неопределённости соответствует
[таблице SoilGrids layers](https://docs.isric.org/globaldata/soilgrids/SoilGrids_faqs_01.html).
WCS того же поставщика и EPSG:4326 доступны как альтернативы приостановленному
REST, согласно [ISRIC access](https://docs.isric.org/globaldata/soilgrids/SoilGrids_faqs_02.html)
и [GetCoverage](https://docs.isric.org/globaldata/soilgrids/wcs.html).
Параметры погоды/единицы сверены с
[OpenWeather Current Weather](https://openweathermap.org/api/current).
Документы просмотрены 06.10.2026. Тесты raw-unit conversion и полевых ограничений
проверяют контракт ArborScan; они не доказывают точность самих моделей провайдера.

## Итог аудита

Программный пробел «почва отсутствует» не подтвердился: реализация, живые данные,
сохранение снимка, история и PDF существовали до текущего пакета. Новая прямая
проверка провайдера прошла. Подтверждённых дефектов единиц, глубины, привязки к
версии или обновления старой среды при открытии не найдено; повторная разработка
не выполнялась. Старые services/soil_service.py и weather_service.py относятся
к историческому сервисному пути; новый v4/Flutter snapshot не использует его
REST-представление и не исправляет задним числом старые значения.

Технический остаток AS-09 этого кандидата — фактический S24/AVD offline/reconnect
и затронутый сквозной UI/PDF-контроль в итоговой сборке, фиксируемый главным
агентом. Отсутствие покрытия Street View принято; профиль корней, архивная погода,
точный raster-release и экспериментальная полевая точность не заявляются.


### Итог интерфейсного остатка AS-16

06.10 на AVD14созданы две DEMOверсии с actual OpenWeather/SoilGrids. Финальный15 вновь открыл их, сохранил2отдельныхPDF/10страниц, fullHTTPSsnapshot unchanged. Offlinecachedreport послеforce-stop, map5records/reconnect refresh проверены15; GPSdenied/manualempty validation проверены14. S24 отсутствует. Файлы/метки точныхсред: [UI README](ui/README.md), [RELEASE_INTEGRATION.md](../../RELEASE_INTEGRATION.md). Подтверждённых новых программных пробелов среды нет; provider/grid/полевая точность/coverage ограничения сохранены.
