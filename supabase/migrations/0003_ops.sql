-- 0003_ops.sql
create table public.alert_rules (
   id             uuid primary key default gen_random_uuid(),
   name           text not null,
   metric         text not null,
   operator       text not null check (operator in ('>','>=','<','<=','=')),
   threshold      numeric not null,
   window_minutes int not null default 15,
   severity       text not null check (severity in ('info','warning','critical')),
   channel        text not null default 'email',     -- channel name only; secrets stay on device
   enabled        boolean not null default true,
   updated_at     timestamptz not null default now()
);

create table public.alerts (
   id              uuid primary key default gen_random_uuid(),
   rule_id         uuid references public.alert_rules (id),
   device_id       text references public.devices (id),
   severity        text not null,
   title           text not null,
   status          text not null default 'active'
                     check (status in ('active','acknowledged','resolved','muted')),
   first_at        timestamptz not null default now(),
   last_at         timestamptz not null default now(),
   acknowledged_by uuid references public.profiles (id),
   muted_until     timestamptz
);

create table public.config_current (
   device_id   text primary key references public.devices (id),
   config      jsonb not null,          -- non-secret configuration only
   config_hash text not null,
   applied_at  timestamptz not null
);

create table public.config_changes (
  id                uuid primary key default gen_random_uuid(),
  device_id         text not null references public.devices (id),
  key               text not null,
  new_value         jsonb not null,
  reason            text not null check (length(reason) between 3 and 500),
  status            text not null default 'pending'
                      check (status in ('pending','applied','rejected','rolled_back')),
  requested_by      uuid not null default auth.uid() references public.profiles (id),
  requested_at      timestamptz not null default now(),

  applied_at             timestamptz,
  rejection_reason       text
);

create table public.api_key_status (
   env           text primary key check (env in ('dev','staging','prod')),
   label         text not null,
   last4         text not null check (length(last4) = 4),
   created_at    date,
   rotated_at    date,
   rotation_due  date,
   last_check_at timestamptz,
   last_check_ok boolean
);

create table public.eval_runs (
   id             text primary key,
   run_at         timestamptz not null,
   model          text not null,
   prompt_version text not null,
   cases          int not null,
   passed         int not null,
   accuracy       numeric(5,4) not null
);

create table public.eval_failures (
   run_id     text not null references public.eval_runs (id) on delete cascade,
   case_id    text not null,
   utterance text not null,           -- from the test file, never user data
   expected   jsonb not null,
   actual     jsonb not null,
   primary key (run_id, case_id)
);

create table public.audit_log (
   id      bigint generated always as identity primary key,
   at      timestamptz not null default now(),
   actor   uuid,
   action  text not null,
   target  text not null,
   details jsonb not null default '{}'::jsonb
);

create or replace function public.audit_trigger()
returns trigger language plpgsql security definer set search_path = public as $$
begin
  insert into public.audit_log (actor, action, target, details)
  values (auth.uid(), tg_op, tg_table_name,
          jsonb_build_object('new', to_jsonb(new), 'old', to_jsonb(old)));
  return coalesce(new, old);
end $$;

create trigger audit_config_changes after insert or update on public.config_changes
  for each row execute function public.audit_trigger();
create trigger audit_alert_rules after insert or update or delete on public.alert_rules
  for each row execute function public.audit_trigger();
create trigger audit_profiles after insert or update or delete on public.profiles
  for each row execute function public.audit_trigger();
