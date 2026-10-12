
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY;
SET LOCAL TIME ZONE 'UTC';
SET LOCAL DateStyle = 'ISO, YMD';
SET LOCAL IntervalStyle = 'postgres';
SET LOCAL extra_float_digits = 1;
SELECT json_build_object('storage_table','objects','rows',count(*),
 'sha256',encode(sha256(convert_to(coalesce(jsonb_agg(to_jsonb(t)
 ORDER BY to_jsonb(t)::text COLLATE "C")::text,'[]'),'UTF8')),'hex')) FROM storage.objects t;
SELECT json_build_object('storage_table','buckets','rows',count(*),
 'sha256',encode(sha256(convert_to(coalesce(jsonb_agg(to_jsonb(t)
 ORDER BY to_jsonb(t)::text COLLATE "C")::text,'[]'),'UTF8')),'hex')) FROM storage.buckets t;
COMMIT;
