
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY;
SET LOCAL TIME ZONE 'UTC';
SET LOCAL DateStyle = 'ISO, YMD';
SET LOCAL IntervalStyle = 'postgres';
SET LOCAL extra_float_digits = 1;
SELECT format($q$SELECT json_build_object('table', %L, 'rows', count(*),
 'sha256', encode(sha256(convert_to(coalesce(jsonb_agg(to_jsonb(t)
 ORDER BY to_jsonb(t)::text COLLATE "C")::text, '[]'), 'UTF8')), 'hex')) FROM public.%I t$q$,
 tablename, tablename) FROM pg_tables WHERE schemaname='public' ORDER BY tablename
\gexec
SELECT json_build_object('versions',json_build_object(
 'contours',public.contour_workflow_version(),
 'history',public.server_history_version(),
 'quality',public.model_quality_version()));
SELECT json_build_object('relationships',json_build_object(
 'orphan_sessions',(SELECT count(*) FROM public.auth_sessions a LEFT JOIN public.users u ON u.id=a.user_id WHERE u.id IS NULL),
 'orphan_reports',(SELECT count(*) FROM public.report_versions a LEFT JOIN public.users u ON u.id=a.owner_id WHERE u.id IS NULL),
 'orphan_contours',(SELECT count(*) FROM public.contour_revisions a LEFT JOIN public.users u ON u.id=a.owner_id WHERE u.id IS NULL)));
COMMIT;
