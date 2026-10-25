-- 0001_profiles.sql
create type public.app_role as enum ('owner', 'developer', 'reviewer', 'viewer');

create table public.profiles (
   id          uuid primary key references auth.users (id) on delete cascade,
   email       text not null unique,
   role        public.app_role not null default 'viewer',
   active      boolean not null default true,
   created_at timestamptz not null default now(),
   last_sign_in_at timestamptz
);

alter table public.profiles enable row level security;

-- Returns the caller's role, or null. SECURITY DEFINER so policies can call it
-- without recursive RLS evaluation on profiles.
create or replace function public.current_role_name()
returns public.app_role
language sql
stable
security definer
set search_path = public
as $$
  select role from public.profiles where id = auth.uid() and active
$$;

create or replace function public.has_role(min_role public.app_role)
returns boolean
language sql
stable
security definer
set search_path = public
as $$
  select coalesce(
    (select case p.role
              when 'owner' then 4 when 'developer' then 3
              when 'reviewer' then 2 when 'viewer' then 1 end
       from public.profiles p where p.id = auth.uid() and p.active)
    >=
    (case min_role when 'owner' then 4 when 'developer' then 3
                   when 'reviewer' then 2 when 'viewer' then 1 end),
    false)
$$;

revoke all on function public.current_role_name() from public;
revoke all on function public.has_role(public.app_role) from public;
grant execute on function public.current_role_name() to authenticated;
grant execute on function public.has_role(public.app_role) to authenticated;

create policy profiles_self_read on public.profiles
  for select to authenticated
  using (id = auth.uid() or public.has_role('owner'));

create policy profiles_owner_write on public.profiles
  for all to authenticated
  using (public.has_role('owner'))
  with check (public.has_role('owner'));
