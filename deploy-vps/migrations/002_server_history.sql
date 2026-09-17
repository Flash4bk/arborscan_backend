-- Additive: no changes to analyses, contour decisions or training datasets.
begin;
create table if not exists public.report_versions (
  owner_id uuid not null references public.users(id),
  analysis_id uuid not null,
  version_id uuid not null,
  parent_id uuid,
  payload_sha256 text not null check (payload_sha256 ~ '^[0-9a-f]{64}$'),
  image_sha256 text not null check (image_sha256 ~ '^[0-9a-f]{64}$'),
  correction_id text,
  summary jsonb not null check (jsonb_typeof(summary)='object'),
  created_at timestamptz not null default now(),
  primary key(owner_id, version_id),
  unique(owner_id, analysis_id, version_id),
  foreign key(owner_id, analysis_id, parent_id)
    references public.report_versions(owner_id, analysis_id, version_id),
  foreign key(owner_id, correction_id)
    references public.contour_revisions(owner_id, correction_id)
);
create unique index if not exists report_one_root on public.report_versions(owner_id,analysis_id) where parent_id is null;
create unique index if not exists report_one_child on public.report_versions(owner_id,parent_id) where parent_id is not null;
create index if not exists report_history_order on public.report_versions(owner_id,created_at desc,version_id);
alter table public.report_versions enable row level security;
revoke all on public.report_versions from public,anon,authenticated;
grant select on public.report_versions to service_role;

create or replace function public.save_report_version(
 p_owner uuid, p_analysis uuid, p_version uuid, p_parent uuid,
 p_payload text, p_image text, p_correction text, p_summary jsonb
) returns jsonb language plpgsql security definer set search_path=public,pg_temp as $$
declare r public.report_versions; par public.report_versions;
begin
 perform pg_advisory_xact_lock(hashtextextended(p_owner::text,0));
 select * into r from public.report_versions where owner_id=p_owner and version_id=p_version;
 if found then
   if r.analysis_id<>p_analysis or r.payload_sha256<>p_payload or r.parent_id is distinct from p_parent then
     raise exception 'Idempotency conflict' using errcode='P0001';
   end if;
   return to_jsonb(r);
 end if;
 if exists(select 1 from public.analyses where id=p_analysis and user_id is distinct from p_owner) then
   raise exception 'Analysis unavailable' using errcode='P0002';
 end if;
 if p_parent is not null then
   select * into par from public.report_versions where owner_id=p_owner and version_id=p_parent and analysis_id=p_analysis;
   if not found then raise exception 'Parent unavailable' using errcode='P0002'; end if;
   if par.image_sha256<>p_image then raise exception 'Original changed' using errcode='P0001'; end if;
 end if;
 if p_correction is not null and not exists(select 1 from public.contour_revisions
   where owner_id=p_owner and correction_id=p_correction and analysis_id=p_analysis and image_sha256=p_image) then
   raise exception 'Contour unavailable' using errcode='P0002';
 end if;
 insert into public.report_versions(owner_id,analysis_id,version_id,parent_id,payload_sha256,image_sha256,correction_id,summary)
 values(p_owner,p_analysis,p_version,p_parent,p_payload,p_image,p_correction,p_summary) returning * into r;
 return to_jsonb(r);
end $$;
revoke all on function public.save_report_version(uuid,uuid,uuid,uuid,text,text,text,jsonb) from public,anon,authenticated;
grant execute on function public.save_report_version(uuid,uuid,uuid,uuid,text,text,text,jsonb) to service_role;
create or replace function public.server_history_version() returns integer
language sql stable set search_path=public,pg_temp as $$ select 1 $$;
revoke all on function public.server_history_version() from public,anon,authenticated;
grant execute on function public.server_history_version() to service_role;
notify pgrst,'reload schema';
commit;
