# AS-14: доступ к PostgreSQL и старым таблицам — 28.09.2026

## Поиск существующего подключения

Проверены конфиги локального проекта (включая ignored env/config), окружение
Windows с типовыми PG/DB-переменными, `/opt/arborscan`, `/etc/arborscan` и Env
запущенных контейнеров API v3, API v4, worker. Пароль/DSN PostgreSQL не найден.
Содержимое env, ключи и строки подключения не выводились. Инструмент проверки:
`tools/as14_find_pg_config.py` — читает конфиги и выводит только имена кандидатов.
Существующие `/home/arborscan/.pg_service.conf` и `.pgpass` имеют права 600.
Пользователь сообщил, что реального пароля в .pgpass нет; файлы не перезаписаны.
SELECT 1 через libpq не выполнен. SQL через management MCP работает, но не заменяет pg_dump.

Три контейнера используют HTTPS Data API/Storage и service_role. Сброс пароля
роли postgres потребует обновить прямые и pooler-подключения этой роли: .pgpass,
SQL-клиенты, CI/скрипты с DSN (если есть вне проверенной среды). В текущем снимке
pg_stat_activity видны только управляемые соединения Supabase; выключенные внешние
интеграции этим не исключены. У найденной конфигурации ArborScan API-ключи и URL
не требуют замены из-за смены DB-пароля. Это вывод из конфигурации, не обещание
нулевого простоя. Пароль агентом не сбрасывался.

Официальные источники: [способы подключения](https://supabase.com/docs/guides/database/connecting-to-postgres),
[сброс DB-пароля](https://supabase.com/docs/guides/troubleshooting/how-do-i-reset-my-supabase-database-password-oTs5sB).

## Исправленные права

В каталоге обнаружены пять старых таблиц с RLS=false и ALL privileges для anon
и authenticated: dataset_builds, model_versions, predictions, training_queue,
training_state. Владельцем был postgres; service_role также имел ALL privileges.
Политик этих таблиц не было, PUBLIC-грантов не было. Исходная ACL каждой:
`{postgres=arwdDxtm/postgres,anon=arwdDxtm/postgres,authenticated=arwdDxtm/postgres,service_role=arwdDxtm/postgres}`.
Это снимок метаданных до изменения; данные не выгружались через SQL-аудит.

Фактически применена через Supabase migration API миграция
`20260928124032_as14_restrict_legacy_internal_tables`. Точный SQL:

```sql
SET LOCAL lock_timeout = '5s';
REVOKE ALL PRIVILEGES ON TABLE public.dataset_builds, public.model_versions, public.predictions, public.training_queue, public.training_state FROM anon, authenticated;
ALTER TABLE public.dataset_builds ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.model_versions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.training_queue ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.training_state ENABLE ROW LEVEL SECURITY;
DO $check$
DECLARE t regclass; role_name text; permission_name text;
BEGIN
  IF NOT (SELECT rolbypassrls FROM pg_roles WHERE rolname = 'service_role') THEN
    RAISE EXCEPTION 'service_role must retain RLS bypass';
  END IF;
  FOREACH t IN ARRAY ARRAY['public.dataset_builds'::regclass, 'public.model_versions'::regclass, 'public.predictions'::regclass, 'public.training_queue'::regclass, 'public.training_state'::regclass] LOOP
    FOREACH permission_name IN ARRAY ARRAY['SELECT','INSERT','UPDATE','DELETE'] LOOP
      IF NOT has_table_privilege('service_role', t, permission_name) THEN RAISE EXCEPTION 'service privilege missing'; END IF;
      FOREACH role_name IN ARRAY ARRAY['anon','authenticated'] LOOP
        IF has_table_privilege(role_name, t, permission_name) THEN RAISE EXCEPTION 'public privilege remains'; END IF;
      END LOOP;
    END LOOP;
  END LOOP;
END $check$;
```

Проверки после применения:

- RLS=true, CRUD anon/authenticated=false, все четыре CRUD-права service_role сохранены.
- Реальные SELECT LIMIT 0 под каждой из двух ролей для каждой таблицы: 10 отказов
  insufficient_privilege, проверенных обработчиком исключений; строк не читали.
- Из работающего API v3: SELECT с limit=0 через PostgREST для всех пяти таблиц —
  HTTP 200, пустой массив. `tools/as14_legacy_access_smoke.py` воспроизводит проверку.
- Публичные HTTPS health v3 и v4 — ok. Контейнеры, модели, строки таблиц не менялись.
- Supabase security advisor: прежних RLS-disabled предупреждений больше нет;
  INFO «RLS without policy» ожидаем для таблиц только с серверным доступом.
  Прежний WARN mutable search_path у public.set_updated_at сохранён, не скрывается.
  [Описание предупреждения](https://supabase.com/docs/guides/database/database-linter?lint=0011_function_search_path_mutable).

Откат приложения не требует отката прав: прежний API использует service_role.
Возврат **точного прежнего, небезопасного** доступа возможен отдельной операцией
только если действительно необходимо восстановить исходную ACL:

```sql
BEGIN;
ALTER TABLE public.dataset_builds DISABLE ROW LEVEL SECURITY;
ALTER TABLE public.model_versions DISABLE ROW LEVEL SECURITY;
ALTER TABLE public.predictions DISABLE ROW LEVEL SECURITY;
ALTER TABLE public.training_queue DISABLE ROW LEVEL SECURITY;
ALTER TABLE public.training_state DISABLE ROW LEVEL SECURITY;
GRANT ALL PRIVILEGES ON TABLE public.dataset_builds, public.model_versions,
  public.predictions, public.training_queue, public.training_state TO anon, authenticated;
COMMIT;
```

Этот откат не выполнялся. Полный restore PostgreSQL всё ещё не проверен;
изменение ACL и SQL-тест прав не являются восстановлением базы.
