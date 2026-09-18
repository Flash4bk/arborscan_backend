begin;
create table if not exists public.ml_taxon_labels (
 id uuid primary key, owner_id uuid not null, analysis_id uuid not null,
 image_sha256 text not null, correction_id text not null, parent_id uuid references public.ml_taxon_labels(id),
 actor_id uuid not null references public.users(id), label jsonb not null,
 original_prediction jsonb, created_at timestamptz not null default now(),
 foreign key(owner_id,correction_id) references public.contour_revisions(owner_id,correction_id)
);
create unique index if not exists ml_label_child on public.ml_taxon_labels(parent_id) where parent_id is not null;
create unique index if not exists ml_label_root on public.ml_taxon_labels(owner_id,image_sha256) where parent_id is null;
create table if not exists public.ml_snapshots (
 id uuid primary key, actor_id uuid not null references public.users(id),
 model_type text not null check(model_type in ('segmentation','classification')),
 manifest jsonb not null, created_at timestamptz not null default now()
);
create table if not exists public.ml_jobs (
 id uuid primary key, actor_id uuid not null references public.users(id),
 snapshot_id uuid not null references public.ml_snapshots(id), params jsonb not null,
 state text not null default 'queued' check(state in ('queued','running','cancel_requested','cancelled','failed','completed')),
 progress jsonb not null default '{}'::jsonb, lease_id uuid, lease_until timestamptz,
 created_at timestamptz not null default now(), updated_at timestamptz not null default now()
);
create unique index if not exists ml_one_active_job on public.ml_jobs((true)) where state in ('queued','running','cancel_requested');
create table if not exists public.ml_models (
 id uuid primary key references public.ml_jobs(id), model_type text not null,
 metadata jsonb not null, actor_id uuid not null references public.users(id), created_at timestamptz not null default now()
);
create table if not exists public.ml_active_models (
 model_type text primary key, model_id uuid references public.ml_models(id), generation bigint not null default 0,
 history jsonb not null default '[]'::jsonb
);
insert into public.ml_active_models(model_type) values('segmentation'),('classification') on conflict do nothing;
alter table public.ml_taxon_labels enable row level security;
alter table public.ml_snapshots enable row level security;
alter table public.ml_jobs enable row level security;
alter table public.ml_models enable row level security;
alter table public.ml_active_models enable row level security;
revoke all on public.ml_taxon_labels,public.ml_snapshots,public.ml_jobs,public.ml_models,public.ml_active_models from public,anon,authenticated;
grant select on public.ml_taxon_labels,public.ml_snapshots,public.ml_jobs,public.ml_models,public.ml_active_models to service_role;

create or replace function public.ml_transition(p_action text,p_id uuid,p_actor uuid default null,p_data jsonb default '{}'::jsonb)
returns jsonb language plpgsql security definer set search_path=public,pg_temp as $$
declare j public.ml_jobs; l public.ml_taxon_labels; a public.ml_active_models; s public.ml_snapshots; m public.ml_models;
begin
 perform pg_advisory_xact_lock(908172635);
 if p_action not in ('claim','heartbeat','finish') and not exists(select 1 from public.users where id=p_actor and lower(trim(role))='admin') then
   raise exception 'Admin required' using errcode='42501';
 end if;
 if p_action='label' then
   select * into l from public.ml_taxon_labels where id=p_id;
   if found then
     if l.label is distinct from p_data->'label' or l.actor_id is distinct from p_actor
       or l.owner_id is distinct from (p_data->>'owner_id')::uuid
       or l.analysis_id is distinct from (p_data->>'analysis_id')::uuid
       or l.correction_id is distinct from p_data->>'correction_id'
       or l.image_sha256 is distinct from p_data->>'image_sha256'
       or l.parent_id is distinct from (p_data->>'parent_id')::uuid
     then raise exception 'Label retry conflict' using errcode='P0001';end if;
     return to_jsonb(l);
   end if;
   if not exists(select 1 from public.contour_revisions where owner_id=(p_data->>'owner_id')::uuid
     and correction_id=p_data->>'correction_id' and image_sha256=p_data->>'image_sha256' and analysis_id=(p_data->>'analysis_id')::uuid) then
      raise exception 'Source unavailable' using errcode='P0002';end if;
   if p_data->>'parent_id' is not null and not exists(select 1 from public.ml_taxon_labels where id=(p_data->>'parent_id')::uuid
     and owner_id=(p_data->>'owner_id')::uuid and image_sha256=p_data->>'image_sha256') then
      raise exception 'Label parent mismatch' using errcode='P0001';end if;
   insert into public.ml_taxon_labels(id,owner_id,analysis_id,image_sha256,correction_id,parent_id,actor_id,label,original_prediction)
   values(p_id,(p_data->>'owner_id')::uuid,(p_data->>'analysis_id')::uuid,p_data->>'image_sha256',p_data->>'correction_id',
     (p_data->>'parent_id')::uuid,p_actor,p_data->'label',p_data->'original_prediction') returning * into l;
   return to_jsonb(l);
 elsif p_action='snapshot' then
   select * into s from public.ml_snapshots where id=p_id;
   if found then
     if s.manifest is distinct from p_data->'manifest' or s.model_type is distinct from p_data->>'model_type'
       or s.actor_id is distinct from p_actor then raise exception 'Snapshot retry conflict' using errcode='P0001';end if;
     return to_jsonb(s);
   end if;
   insert into public.ml_snapshots(id,actor_id,model_type,manifest) values(p_id,p_actor,p_data->>'model_type',p_data->'manifest') returning * into s;
   return to_jsonb(s);
 elsif p_action='enqueue' then
   select * into j from public.ml_jobs where id=p_id;
   if found then
     if j.snapshot_id is distinct from (p_data->>'snapshot_id')::uuid or j.params is distinct from p_data->'params'
       or j.actor_id is distinct from p_actor then raise exception 'Job retry conflict' using errcode='P0001';end if;
     return to_jsonb(j);
   end if;
   insert into public.ml_jobs(id,actor_id,snapshot_id,params) values(p_id,p_actor,(p_data->>'snapshot_id')::uuid,p_data->'params') returning * into j;
   return to_jsonb(j);
 elsif p_action='cancel' then
   update public.ml_jobs set state=case when state='queued' then 'cancelled' else 'cancel_requested' end,updated_at=now()
   where id=p_id and state in ('queued','running','cancel_requested') returning * into j;
   if not found then select * into j from public.ml_jobs where id=p_id;end if;
   return to_jsonb(j);
 elsif p_action='claim' then
   update public.ml_jobs set state='failed',progress=jsonb_build_object('error','worker_lease_expired'),updated_at=now()
     where state in ('running','cancel_requested') and lease_until<now();
   select * into j from public.ml_jobs where state='queued' order by created_at limit 1 for update;
   if not found then return null;end if;
   update public.ml_jobs set state='running',lease_id=p_id,lease_until=now()+interval '90 seconds',updated_at=now()
     where id=j.id returning * into j;
   return to_jsonb(j);
 elsif p_action in ('heartbeat','finish') then
   if p_action='finish' then
     select * into j from public.ml_jobs where id=p_id and lease_id=(p_data->>'lease_id')::uuid
       and state in ('completed','cancelled','failed');
     if found then
       if j.state is distinct from p_data->>'state' or j.progress is distinct from p_data->'progress' then
         raise exception 'Finish retry conflict' using errcode='P0001';end if;
       if j.state='completed' and not exists(select 1 from public.ml_models where id=p_id and metadata=p_data->'model') then
         raise exception 'Model retry conflict' using errcode='P0001';end if;
       return to_jsonb(j);
     end if;
   end if;
   select * into j from public.ml_jobs where id=p_id and lease_id=(p_data->>'lease_id')::uuid and lease_until>now() and state in ('running','cancel_requested');
   if not found then raise exception 'Worker lease lost' using errcode='P0001';end if;
   if p_action='heartbeat' then
     update public.ml_jobs set progress=coalesce(p_data->'progress',progress),lease_until=now()+interval '90 seconds',updated_at=now() where id=p_id returning * into j;
   else
     if p_data->>'state' not in ('completed','cancelled','failed') then raise exception 'Invalid terminal state';end if;
     if j.state='cancel_requested' and p_data->>'state'='completed' then raise exception 'Cancellation pending' using errcode='P0001';end if;
     if p_data->>'state'='completed' then
       insert into public.ml_models(id,model_type,metadata,actor_id) select j.id,snap.model_type,p_data->'model',j.actor_id from public.ml_snapshots snap where snap.id=j.snapshot_id;
     end if;
     update public.ml_jobs set state=p_data->>'state',progress=p_data->'progress',updated_at=now() where id=p_id returning * into j;
   end if;
   return to_jsonb(j);
 elsif p_action='activate' then
   select * into a from public.ml_active_models where model_type='segmentation' for update;
   if a.generation is distinct from (p_data->>'expected_generation')::bigint then raise exception 'Activation conflict' using errcode='P0001';end if;
   if p_id is not null then
     select * into m from public.ml_models where id=p_id and model_type='segmentation';
     if not found or (m.metadata->>'compatible')::boolean is not true or (m.metadata->>'eligible_for_activation')::boolean is not true then
       raise exception 'Candidate experimental or incompatible' using errcode='P0001';end if;
   end if;
   update public.ml_active_models set model_id=p_id,generation=generation+1,history=history||jsonb_build_array(
     jsonb_build_object('from',a.model_id,'to',p_id,'actor',p_actor,'at',now())) where model_type='segmentation' returning * into a;
   return to_jsonb(a);
 end if;
 raise exception 'Unknown operation';
end $$;
revoke all on function public.ml_transition(text,uuid,uuid,jsonb) from public,anon,authenticated;
grant execute on function public.ml_transition(text,uuid,uuid,jsonb) to service_role;
create or replace function public.model_quality_version() returns integer language sql stable set search_path=public,pg_temp as $$select 1$$;
revoke all on function public.model_quality_version() from public,anon,authenticated;
grant execute on function public.model_quality_version() to service_role;
notify pgrst,'reload schema';
commit;
