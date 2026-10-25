-- 0002_telemetry.sql
create table public.devices (
   id                 text primary key,
   name               text not null,
   app_version        text,
   last_heartbeat_at timestamptz,
   status             text not null default 'unknown'
                        check (status in ('healthy','degraded','down','unknown')),
   updated_at         timestamptz not null default now()
);

create table public.turns (
   id                 text primary key,               -- turn_id (ULID)
   device_id          text not null references public.devices (id),
   session_id         text not null,
   started_at         timestamptz not null,
   status             text not null check (status in ('ok','degraded','error','cancelled')),
   intent             text,
   prompt_hash        text not null,
   response_hash      text,
   sensitive          boolean not null default false,
   sensitive_category text,
   hash_epoch         int not null default 1,
   tools_called       text[] not null default '{}',
   rounds             int not null default 0,
   e2e_first_audio_ms int,
   total_ms           int,
   tokens_total       int not null default 0,
   cost_usd           numeric(12,6),
   model              text,
   stage_timings      jsonb not null default '{}'::jsonb,
   flagged_for_review boolean not null default false,
   ingested_at        timestamptz not null default now()
);
create index turns_device_time on public.turns (device_id, started_at desc);
create index turns_hash on public.turns (prompt_hash);
create index turns_status_time on public.turns (status, started_at desc);

create table public.llm_calls (
  id                  text primary key,               -- request_id
  turn_id             text references public.turns (id) on delete cascade,
  device_id           text not null references public.devices (id),
  model               text not null,
  round               int not null,
  status              text not null,
  http_status         int,
  ttft_ms             int,

  duration_ms               int not null,
  prompt_tokens             int not null default 0,
  completion_tokens         int not null default 0,
  reasoning_tokens          int not null default 0,
  cached_tokens             int not null default 0,
  cost_usd                  numeric(12,6),
  cost_estimated            boolean not null default false,
  xai_response_id           text,
  system_fingerprint        text,
  created_at                timestamptz not null
);
create index llm_calls_time on public.llm_calls (created_at desc);

create table public.tool_calls (
   id            text primary key,
   turn_id       text references public.turns (id) on delete cascade,
   device_id     text not null references public.devices (id),
   tool_name     text not null,
   status        text not null check (status in ('ok','error','timeout','rejected')),
   duration_ms   int not null,
   error_code    text,
   args_redacted jsonb not null default '{}'::jsonb,
   created_at    timestamptz not null
);
create index tool_calls_name_time on public.tool_calls (tool_name, created_at desc);

create table public.errors (
   id              text primary key,                  -- event_id
   turn_id         text references public.turns (id) on delete set null,
   device_id       text not null references public.devices (id),
   category        text not null,
   error_code      text not null,
   message         text not null,
   stage           text,
   fallback_spoken boolean not null default false,
   created_at      timestamptz not null
);
create index errors_code_time on public.errors (error_code, created_at desc);

create table public.heartbeats (
   device_id      text not null references public.devices (id),
   at             timestamptz not null,
   cpu_pct        numeric(5,2),
   mem_pct        numeric(5,2),
   disk_pct       numeric(5,2),
   temp_c         numeric(5,2),
   throttled      boolean,
   sync_lag_s     int,
   api_reachable boolean,
   last_api_ok_at timestamptz,
   primary key (device_id, at)
);

create table public.rollup_hourly (
   bucket             timestamptz not null,
   device_id          text not null references public.devices (id),
   turns              int not null default 0,
   turns_ok           int not null default 0,
   errors             int not null default 0,
   tokens_prompt      bigint not null default 0,
   tokens_completion bigint not null default 0,
   tokens_reasoning   bigint not null default 0,
   tokens_cached      bigint not null default 0,
   cost_usd           numeric(12,6) not null default 0,
   p50_e2e_ms int, p95_e2e_ms int, p50_ttft_ms int, p95_ttft_ms int,
   p50_stt_ms int, p95_stt_ms int,
   wake_events int not null default 0, false_wakes int not null default 0,
   heartbeat_minutes int not null default 0,
   primary key (bucket, device_id)
);

create table public.rollup_daily (
  bucket             date not null,
  device_id          text not null references public.devices (id),
  turns              int not null default 0,

  turns_ok           int not null default 0,
  errors             int not null default 0,
  tokens_prompt      bigint not null default 0,
  tokens_completion bigint not null default 0,
  tokens_reasoning   bigint not null default 0,
  tokens_cached      bigint not null default 0,
  cost_usd           numeric(12,6) not null default 0,
  p50_e2e_ms int, p95_e2e_ms int, p50_ttft_ms int, p95_ttft_ms int,
  p50_stt_ms int, p95_stt_ms int,
  uptime_pct         numeric(5,2),
  primary key (bucket, device_id)
);

-- Redacted text lives in its own table so that access can be limited by RLS alone
-- (RLS filters rows, not columns). Null for sensitive turns and after retention.
create table public.turn_text (
   turn_id           text primary key references public.turns (id) on delete cascade,
   prompt_redacted   text,
   response_redacted text,
   expires_at        timestamptz not null default now() + interval '30 days'
);

create view public.heartbeats_latest with (security_invoker = true) as
  select distinct on (device_id) *
  from public.heartbeats
  order by device_id, at desc;
