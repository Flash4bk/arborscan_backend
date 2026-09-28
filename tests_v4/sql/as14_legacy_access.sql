-- Read-only permission regression test; no user rows returned or changed.
DO $test$
DECLARE role_name text; table_name text; permission_name text;
BEGIN
  FOREACH table_name IN ARRAY ARRAY['dataset_builds','model_versions','predictions','training_queue','training_state'] LOOP
    IF NOT (SELECT relrowsecurity FROM pg_class WHERE oid = format('public.%I', table_name)::regclass) THEN
      RAISE EXCEPTION 'RLS disabled';
    END IF;
    FOREACH permission_name IN ARRAY ARRAY['SELECT','INSERT','UPDATE','DELETE'] LOOP
      IF NOT has_table_privilege('service_role', format('public.%I', table_name), permission_name) THEN
        RAISE EXCEPTION 'Missing server privilege';
      END IF;
      FOREACH role_name IN ARRAY ARRAY['anon','authenticated'] LOOP
        IF has_table_privilege(role_name, format('public.%I', table_name), permission_name) THEN
          RAISE EXCEPTION 'Unexpected client privilege';
        END IF;
      END LOOP;
    END LOOP;
  END LOOP;
  FOREACH role_name IN ARRAY ARRAY['anon','authenticated'] LOOP
    EXECUTE format('SET LOCAL ROLE %I', role_name);
    FOREACH table_name IN ARRAY ARRAY['dataset_builds','model_versions','predictions','training_queue','training_state'] LOOP
      BEGIN
        EXECUTE format('SELECT 1 FROM public.%I LIMIT 0', table_name);
        RAISE EXCEPTION 'Expected access denial';
      EXCEPTION WHEN insufficient_privilege THEN NULL;
      END;
    END LOOP;
    RESET ROLE;
  END LOOP;
END $test$;
SELECT 'PASS: RLS, CRUD rights and 10 actual SELECT denials' AS result;
