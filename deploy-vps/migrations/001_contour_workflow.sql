-- Additive migration. Apply with the Supabase database owner, never anon.
begin;
create table if not exists public.contour_revisions (
  owner_id uuid not null,
  correction_id text not null,
  analysis_id uuid not null,
  image_sha256 text not null,
  parent_id text,
  legacy boolean not null default false,
  status text not null default 'draft' check (status in ('draft','submitted','accepted','rejected')),
  created_at timestamptz not null default now(),
  decisions jsonb not null default '[]'::jsonb,
  primary key (owner_id, correction_id),
  foreign key (owner_id, parent_id) references public.contour_revisions(owner_id, correction_id)
);
create unique index if not exists contour_one_child on public.contour_revisions(owner_id, parent_id)
  where parent_id is not null;
create unique index if not exists contour_one_root on public.contour_revisions(owner_id, analysis_id, image_sha256)
  where parent_id is null and not legacy;
create index if not exists contour_queue on public.contour_revisions(status, created_at, correction_id);
alter table public.contour_revisions enable row level security;
revoke all on public.contour_revisions from public, anon, authenticated;
grant select on public.contour_revisions to service_role;

create or replace function public.contour_transition(
  p_action text, p_owner uuid, p_id text, p_analysis uuid default null,
  p_image text default null, p_parent text default null,
  p_actor uuid default null, p_decision text default null, p_reason text default null
) returns jsonb language plpgsql security definer set search_path = public, pg_temp as $$
declare r public.contour_revisions; parent public.contour_revisions;
begin
  -- Serializes even initial inserts and retries across workers/processes.
  perform pg_advisory_xact_lock(hashtextextended(p_owner::text, 0));
  select * into r from public.contour_revisions where owner_id=p_owner and correction_id=p_id for update;
  if p_action in ('register','legacy') then
    if found then return to_jsonb(r); end if;
    if p_parent is not null then
      select * into parent from public.contour_revisions where owner_id=p_owner and correction_id=p_parent;
      if not found or parent.analysis_id<>p_analysis or parent.image_sha256<>p_image then
        raise exception 'Invalid parent' using errcode='P0001';
      end if;
      if exists(select 1 from public.contour_revisions where owner_id=p_owner and parent_id=p_parent) then
        raise exception 'Revision conflict' using errcode='P0001';
      end if;
    end if;
    insert into public.contour_revisions(owner_id,correction_id,analysis_id,image_sha256,parent_id,status,legacy)
    values(p_owner,p_id,p_analysis,p_image,p_parent,case when p_action='legacy' then 'submitted' else 'draft' end,p_action='legacy')
    returning * into r;
    return to_jsonb(r);
  end if;
  if not found then raise exception 'Revision not found' using errcode='P0002'; end if;
  if p_action='submit' then
    if r.status<>'draft' then return to_jsonb(r); end if;
    update public.contour_revisions set status='submitted', decisions=decisions ||
      jsonb_build_array(jsonb_build_object('action','submitted','actor_id',p_owner,'at',now()))
      where owner_id=p_owner and correction_id=p_id returning * into r;
  elsif p_action='decide' then
    -- Recheck the existing application role in the same transaction.
    if not exists(select 1 from public.users where id=p_actor and lower(trim(role))='admin') then
      raise exception 'Admin required' using errcode='42501';
    end if;
    if p_decision not in ('accepted','rejected') or p_decision is null or
       (p_decision='rejected' and length(trim(coalesce(p_reason,'')))=0) or
       length(coalesce(p_reason,''))>2000 then
      raise exception 'Invalid decision' using errcode='22023';
    end if;
    if r.status<>'submitted' then
      if r.status=p_decision and r.decisions->-1->>'actor_id'=p_actor::text and
         coalesce(r.decisions->-1->>'reason','')=coalesce(p_reason,'') then return to_jsonb(r); end if;
      raise exception 'Decision conflict' using errcode='P0001';
    end if;
    update public.contour_revisions set status=p_decision, decisions=decisions ||
      jsonb_build_array(jsonb_build_object('action',p_decision,'actor_id',p_actor,'at',now(),'reason',p_reason))
      where owner_id=p_owner and correction_id=p_id returning * into r;
  else raise exception 'Invalid action' using errcode='22023';
  end if;
  return to_jsonb(r);
end $$;
revoke all on function public.contour_transition(text,uuid,text,uuid,text,text,uuid,text,text) from public,anon,authenticated;
grant execute on function public.contour_transition(text,uuid,text,uuid,text,text,uuid,text,text) to service_role;
-- Published atomically with the table, indexes and transition RPC. A table-only
-- probe would incorrectly advertise support after an incomplete manual setup.
create or replace function public.contour_workflow_version()
returns integer language sql stable set search_path = public, pg_temp as $$ select 1 $$;
revoke all on function public.contour_workflow_version() from public,anon,authenticated;
grant execute on function public.contour_workflow_version() to service_role;
notify pgrst, 'reload schema';
commit;
