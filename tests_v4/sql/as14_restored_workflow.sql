-- Only invoked by ops_pg_restore_test.py in a network-disabled test container.
BEGIN;
DO $$ BEGIN
  IF current_setting('as14.restore_test',true) IS DISTINCT FROM 'enabled' THEN
    RAISE EXCEPTION 'Isolated restore marker required';
  END IF;
END $$;
DO $$
DECLARE a uuid:=gen_random_uuid(); b uuid:=gen_random_uuid(); admin_id uuid:=gen_random_uuid();
  tree uuid:=gen_random_uuid(); v uuid:=gen_random_uuid(); child uuid:=gen_random_uuid();
  mask text:=gen_random_uuid()::text; mask_child text:=gen_random_uuid()::text;
  r jsonb; first_r jsonb; denied boolean;
BEGIN
  INSERT INTO public.users(id,name,email,password_hash,salt,role,created_at,updated_at)
    VALUES(a,'Restore fixture',a::text||'@invalid.test','not-a-login','synthetic','user',now(),now()),
          (b,'Restore fixture',b::text||'@invalid.test','not-a-login','synthetic','user',now(),now()),
          (admin_id,'Restore fixture',admin_id::text||'@invalid.test','not-a-login','synthetic','admin',now(),now());
  SET LOCAL ROLE service_role;
  first_r:=public.contour_transition('register',a,mask,tree,repeat('b',64));
  IF public.contour_transition('register',a,mask,tree,repeat('b',64))<>first_r THEN RAISE EXCEPTION 'Contour retry mismatch'; END IF;
  PERFORM public.contour_transition('submit',a,mask);
  denied:=false;
  BEGIN PERFORM public.contour_transition('decide',a,mask,p_actor=>a,p_decision=>'accepted');
  EXCEPTION WHEN insufficient_privilege THEN denied:=true; END;
  IF NOT denied THEN RAISE EXCEPTION 'Ordinary user moderated'; END IF;
  r:=public.contour_transition('decide',a,mask,p_actor=>admin_id,p_decision=>'accepted');
  IF r->>'status'<>'accepted' THEN RAISE EXCEPTION 'Acceptance failed'; END IF;
  r:=public.contour_transition('register',a,mask_child,tree,repeat('b',64),mask);
  IF r->>'status'<>'draft' THEN RAISE EXCEPTION 'Child approval inherited'; END IF;
  PERFORM public.contour_transition('submit',a,mask_child);
  r:=public.contour_transition('decide',a,mask_child,p_actor=>admin_id,p_decision=>'rejected',p_reason=>'Synthetic restore check');
  IF r->'decisions'->-1->>'reason'<>'Synthetic restore check' THEN RAISE EXCEPTION 'Decision history missing'; END IF;
  first_r:=public.save_report_version(a,tree,v,null,repeat('a',64),repeat('b',64),mask,'{}');
  IF public.save_report_version(a,tree,v,null,repeat('a',64),repeat('b',64),mask,'{}')<>first_r THEN RAISE EXCEPTION 'Report retry mismatch'; END IF;
  denied:=false;
  BEGIN PERFORM public.save_report_version(b,tree,child,v,repeat('c',64),repeat('b',64),null,'{}');
  EXCEPTION WHEN SQLSTATE 'P0002' THEN denied:=true; END;
  IF NOT denied THEN RAISE EXCEPTION 'Cross-owner parent accepted'; END IF;
  PERFORM public.save_report_version(a,tree,child,v,repeat('c',64),repeat('b',64),mask_child,'{}');
  denied:=false;
  BEGIN PERFORM public.save_report_version(a,tree,gen_random_uuid(),v,repeat('d',64),repeat('b',64),mask_child,'{}');
  EXCEPTION WHEN unique_violation THEN denied:=true; END;
  IF NOT denied THEN RAISE EXCEPTION 'Conflicting revision accepted'; END IF;
  RESET ROLE;
END $$;
ROLLBACK;
SELECT 'PASS: restored RPC retries, ownership, conflict, moderation, revision status' AS result;
