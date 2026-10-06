# AS-16: чистая воспроизводимость программного AS-10

06.10.2026 агент проверил исследовательский пакет интеграционного checkout
`0fe95eb9a6d5c115b8c094bff68addaa86d7d14a` в новом изолированном venv.
Физический метод и его исходники не менялись. Это синтетические программные
проверки, а не экспериментальный β дерева или подтверждение полевой точности.

Установка выполнена только из `research/requirements-beta.txt`: CPython 3.14.6,
NumPy 2.4.6, SciPy 1.17.1, Matplotlib 3.11.2, pytest 9.1.1. `pip check` завершился
без ошибок. `include-system-site-packages=false`, `PYTHONPATH` очищен,
пользовательский site отключён; все четыре пакета импортированы из нового venv.
Прежний `output/as10/test-deps` не использовался. Полный список реально установленных
зависимостей сохранён в [clean_reproducibility.json](clean_reproducibility.json).

| Фактическая проверка | Результат |
|---|---|
| Единый текущий набор physics + identification + workflow | **70 passed**; 0 failures/errors/skips; pytest сообщил 230.42 с, JUnit — 230.405 с |
| CLI по сохранённому `research/examples/beta_chain_synthetic_v2.json` | β = **2.400001788258296 кг/с**, ошибка от синтетических 2.4 — 1.788258296×10⁻⁶ кг/с; subprocess 17.703 с |
| Повторный CLI из созданного `input.json` в новый каталог | Та же исходная структура/канонический hash, тот же β и побайтно одинаковый CSV; subprocess 16.644 с |
| Реальные файлы CLI и replay | Проверены SHA всех 7 файлов каждого COMPLETE, PNG декодированы; все строки CSV численно совпадают с координатами/остатками JSON |
| Существующий расширенный `verify_beta_dynamics` | exit 0, **126.512 с**; независимые аналитические случаи, энергия/работа, уточнение шага/разбиения, шум/ошибочные входы и единицы проверены |
| Короткое синтетическое окно n=20 | Кандидат 80 кг/с сохранён отдельно; **β = null**, `not_established`: сигнал 5.386×10⁻⁷ м меньше явно гипотетического разрешения 10⁻⁵ м |
| Визуальный просмотр настоящих PNG | Все **8** изображений открыты агентом: читаемые оси, единицы, легенды и метка SYNTHETIC |

Все результаты и контрольные SHA сохранены в JSON. Сырой XML, stdout/stderr,
полные CLI-выходы и venv остаются локально под игнорируемым
`output/as16-integration/` и в Git не включаются. При Windows checkout исходники
используют CRLF; в свидетельстве сохранены SHA точных файлов и SHA содержимого
Git blob. Их текст совпадает после CRLF→LF, прежние AS-10 SHA Git blob сохранены.
Входной пример оставляет первоначальное происхождение своих синтетических
наблюдений; новые результаты записывают текущие версии и SHA собственного запуска.

Чистый запуск из корня проекта:

```powershell
$env:PYTHONPATH = ''
$env:PYTHONNOUSERSITE = '1'
& 'C:\Python314\python.exe' -m venv output/as16-integration/research-env
& '.\output\as16-integration\research-env\Scripts\python.exe' -m pip install -r research/requirements-beta.txt
& '.\output\as16-integration\research-env\Scripts\python.exe' -m pytest tests_v4/test_planar_chain.py tests_v4/test_beta_identification.py tests_v4/test_beta_dynamics.py -q
& '.\output\as16-integration\research-env\Scripts\python.exe' -m research.beta_dynamics research/examples/beta_chain_synthetic_v2.json output/as16-integration/repeat-example-001
& '.\output\as16-integration\research-env\Scripts\python.exe' -m research.beta_dynamics output/as16-integration/repeat-example-001/input.json output/as16-integration/repeat-replay-001
& '.\output\as16-integration\research-env\Scripts\python.exe' -m research.verify_beta_dynamics output/as16-integration/repeat-protocol-001
```

Повторные расчёты требуют новых имён каталогов: прежние результаты не
перезаписываются. `--publish-research-evidence` не использовался.

Сохранённый пример: [траектории](saved_example_comparison.png),
[остатки](saved_example_residuals.png), [S(β)](saved_example_objective.png),
[энергия](saved_example_energy.png). Слабый сигнал n=20:
[траектории](weak_n20_comparison.png), [остатки](weak_n20_residuals.png),
[S(β)](weak_n20_objective.png), [энергия](weak_n20_energy.png).

Исследовательский модуль не подключён к API, worker или Flutter. β обычного
фото/AR не назначается. Отсутствие реальных динамических данных и научная
валидация остаются границей AS-10; дополнительные полевые входы в этом
интеграционном пакете не запрашивались. Ограничения физики, источников и
сходимости сохраняются в `research/BETA_DYNAMICS.md` и `research/BETA_SPEC.md`.
