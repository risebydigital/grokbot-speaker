-- 0004_rls.sql
alter table public.devices                   enable row level security;
alter table public.turns                     enable row level security;
alter table public.llm_calls                 enable row level security;
alter table public.tool_calls                enable row level security;
alter table public.errors                    enable row level security;
alter table public.heartbeats                enable row level security;
alter table public.rollup_hourly             enable row level security;
alter table public.rollup_daily              enable row level security;
alter table public.alert_rules               enable row level security;
alter table public.alerts                    enable row level security;
alter table public.config_current            enable row level security;
alter table public.config_changes            enable row level security;

alter table public.api_key_status            enable row level security;
alter table public.eval_runs                 enable row level security;
alter table public.eval_failures             enable row level security;
alter table public.audit_log                 enable row level security;

-- Read access by minimum role. The backend writes with the service key,
-- which bypasses RLS, so no insert policies are needed for telemetry tables.
create policy read_devices   on public.devices        for select to authenticated using (public.has_role('viewer'));
create policy read_rollup_h on public.rollup_hourly for select to authenticated using (public.has_role('viewer'));
create policy read_rollup_d on public.rollup_daily    for select to authenticated using (public.has_role('viewer'));
create policy read_heartbeat on public.heartbeats     for select to authenticated using (public.has_role('viewer'));
create policy read_alerts    on public.alerts         for select to authenticated using (public.has_role('viewer'));
create policy read_rules     on public.alert_rules    for select to authenticated using (public.has_role('viewer'));

-- Turns metadata: every role. Redacted text: reviewers and above only.
alter table public.turn_text enable row level security;
create policy read_turns     on public.turns      for select to authenticated using (public.has_role('viewer'));
create policy read_turn_text on public.turn_text for select to authenticated using (public.has_role('reviewer'));
-- Reviewers may flag a turn; column privileges limit updates to that one column.
revoke update on public.turns from authenticated;
grant update (flagged_for_review) on public.turns to authenticated;
create policy flag_turns     on public.turns      for update to authenticated
  using (public.has_role('reviewer')) with check (public.has_role('reviewer'));

create policy read_llm       on public.llm_calls      for select to authenticated using (public.has_role('developer'));
create policy read_tools     on public.tool_calls     for select to authenticated using (public.has_role('developer'));
create policy read_errors    on public.errors         for select to authenticated using (public.has_role('developer'));
create policy read_evals     on public.eval_runs      for select to authenticated using (public.has_role('developer'));
create policy read_evalf     on public.eval_failures for select to authenticated using (public.has_role('developer'));
create policy read_config    on public.config_current for select to authenticated using (public.has_role('developer'));
create policy read_changes   on public.config_changes for select to authenticated using (public.has_role('developer'));
create policy read_keys      on public.api_key_status for select to authenticated using (public.has_role('owner'));
create policy read_audit     on public.audit_log      for select to authenticated using (public.has_role('owner'));

-- Writes from the dashboard.
create policy propose_change on public.config_changes for insert to authenticated
  with check (public.has_role('developer') and status = 'pending' and requested_by = auth.uid());
create policy ack_alerts      on public.alerts for update to authenticated
  using (public.has_role('viewer')) with check (status in ('acknowledged','resolved','muted'));
create policy edit_rules      on public.alert_rules for all to authenticated
  using (public.has_role('owner')) with check (public.has_role('owner'));
