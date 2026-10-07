-- PREPARED ONLY: not executed against production by this release package.
-- ArborScan uses public.users/auth_sessions, not Supabase Auth identities.
-- Run only after separate approval and a verified backup, via psql with
-- ON_ERROR_STOP=1 and autocommit. Do not use --single-transaction: PostgreSQL
-- CREATE INDEX CONCURRENTLY cannot run inside a transaction block.
-- Do not enable ARBORSCAN_GOOGLE_IDENTITY_CLAIMS_ENABLED until both guards pass.
-- The index does not rewrite any user, role, password, report or Google subject.
-- Only text and varchar subject columns are supported. PostgreSQL deparses the
-- varchar predicate with an explicit text cast; both exact forms are verified.

DO $preflight$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_attribute a
    WHERE a.attrelid='public.users'::regclass AND a.attname='google_sub'
      AND a.attnum > 0 AND NOT a.attisdropped
      AND a.atttypid IN ('text'::regtype, 'varchar'::regtype)
  ) THEN
    RAISE EXCEPTION 'Google identity prerequisite failed: unsupported subject column type';
  END IF;
  IF EXISTS (
    SELECT 1 FROM public.users
    WHERE google_sub IS NOT NULL AND google_sub <> ''
      AND google_sub !~ '^[A-Za-z0-9_-]{1,255}$'
  ) THEN
    RAISE EXCEPTION 'Google identity prerequisite failed: invalid subject values';
  END IF;
  IF EXISTS (
    SELECT google_sub FROM public.users
    WHERE google_sub IS NOT NULL AND google_sub <> ''
    GROUP BY google_sub HAVING count(*) > 1
  ) THEN
    RAISE EXCEPTION 'Google identity prerequisite failed: duplicate subjects';
  END IF;
  IF to_regclass('public.users_google_sub_unique') IS NOT NULL AND NOT EXISTS (
    SELECT 1 FROM pg_index i
    JOIN pg_attribute a ON a.attrelid=i.indrelid AND a.attname='google_sub'
    WHERE i.indexrelid=to_regclass('public.users_google_sub_unique')
      AND i.indrelid='public.users'::regclass
      AND i.indisunique AND i.indisvalid AND i.indisready
      AND i.indnkeyatts=1 AND i.indkey[0]=a.attnum
      AND a.atttypid IN ('text'::regtype, 'varchar'::regtype)
      AND pg_get_expr(i.indpred,i.indrelid) IN (
        '((google_sub IS NOT NULL) AND (google_sub <> ''''::text))',
        '((google_sub IS NOT NULL) AND ((google_sub)::text <> ''''::text))'
      )
  ) THEN
    RAISE EXCEPTION 'Google identity prerequisite failed: existing index has another definition or is invalid';
  END IF;
END
$preflight$;

CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS users_google_sub_unique
  ON public.users (google_sub)
  WHERE google_sub IS NOT NULL AND google_sub <> '';

DO $verified$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_index i
    JOIN pg_attribute a ON a.attrelid=i.indrelid AND a.attname='google_sub'
    WHERE i.indexrelid=to_regclass('public.users_google_sub_unique')
      AND i.indrelid='public.users'::regclass
      AND i.indisunique AND i.indisvalid AND i.indisready
      AND i.indnkeyatts=1 AND i.indkey[0]=a.attnum
      AND a.atttypid IN ('text'::regtype, 'varchar'::regtype)
      AND pg_get_expr(i.indpred,i.indrelid) IN (
        '((google_sub IS NOT NULL) AND (google_sub <> ''''::text))',
        '((google_sub IS NOT NULL) AND ((google_sub)::text <> ''''::text))'
      )
  ) THEN
    RAISE EXCEPTION 'Google identity prerequisite failed: unique index not ready';
  END IF;
END
$verified$;

SELECT 'google_subject_unique_v1' AS prerequisite,
       i.indisunique AS unique_subject, i.indisvalid AS valid,
       i.indisready AS ready, pg_get_expr(i.indpred,i.indrelid) AS predicate
FROM pg_index i
WHERE i.indexrelid=to_regclass('public.users_google_sub_unique');

-- Rollback is deliberately separate: first disable identity claims on every
-- API instance, then revert the API image if necessary. Keeping this additive
-- index is compatible with the old code. If separately approved for removal:
-- DROP INDEX CONCURRENTLY public.users_google_sub_unique;
-- No SQL rollback may rewrite an original user identity or restore a stale dump.
