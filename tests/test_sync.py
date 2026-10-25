import json
from pathlib import Path

from gva.backend.schema_columns import migration_columns
from gva.backend.sync_supabase import SupabaseRest, SyncError, sync_once
from gva.ids import new_ulid
from gva.telemetry.events import EventLog
from gva.telemetry.redact import prompt_hash
from gva.telemetry.store import Store
from tests.fakes import MemorySupabase, test_config

TABLES = {
    "profiles",
    "devices",
    "turns",
    "llm_calls",
    "tool_calls",
    "errors",
    "heartbeats",
    "rollup_hourly",
    "rollup_daily",
    "turn_text",
    "alert_rules",
    "alerts",
    "config_current",
    "config_changes",
    "api_key_status",
    "eval_runs",
    "eval_failures",
    "audit_log",
}


def test_migrations_cover_the_eighteen_tables():
    columns = migration_columns()
    assert set(columns) == TABLES
    assert "queue_depths" not in columns["heartbeats"]
    assert "prompt_hash" in columns["turns"]
    assert "uptime_pct" in columns["rollup_daily"]
    assert "uptime_pct" not in columns["rollup_hourly"]
    assert "requested_by" in columns["config_changes"]


def _pepper() -> str:
    return "pepper-" + ("p" * 32)


def _emit_turn(events: EventLog, text: str = "hello") -> str:
    turn_id = new_ulid()
    events.ctx.turn_id = turn_id
    events.ctx.session_id = new_ulid()
    events.ctx.request_id = new_ulid()
    digest = prompt_hash(text, _pepper())
    events.emit(
        "llm.completed",
        model="grok-4.7",
        round=1,
        status="ok",
        duration_ms=12,
        ttft_ms=4,
        cost_usd=0.01,
        cost_estimated=False,
        usage={
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
            "cached_tokens": 0,
            "reasoning_tokens": 0,
        },
    )
    events.emit(
        "tool.completed",
        tool_name="get_time",
        tool_call_id="call-1",
        args_redacted={"what": "time"},
        status="ok",
        duration_ms=3,
        result_bytes=20,
    )
    events.emit(
        "turn.completed",
        status="ok",
        intent="time",
        prompt_redacted=text,
        prompt_hash=digest,
        prompt_chars=len(text),
        response_redacted="It is noon.",
        response_hash=prompt_hash("it is noon.", _pepper()),
        sensitive=False,
        hash_epoch=1,
        tools_called=["get_time"],
        rounds=1,
        stage_timings={"stt_ms": 8, "llm_total_ms": 12},
        total_ms=40,
        tokens_total=5,
        cost_usd=0.01,
        fallback_spoken=False,
        started_at=events.store.recent("llm.completed", 1)[0]["ts"],
        model="grok-4.7",
    )
    events.emit(
        "heartbeat",
        cpu_pct=1.0,
        mem_pct=2.0,
        disk_pct=3.0,
        temp_c=40.0,
        throttled=False,
        queue_depths={"capture": 0},
        api_reachable=True,
    )
    events.ctx.turn_id = None
    events.ctx.request_id = None
    return turn_id


def _bodies(http: MemorySupabase) -> str:
    return "\n".join(body for _url, body in http.posts)


def test_sync_matches_sql_columns_and_hides_secrets(tmp_path, monkeypatch):
    cfg = test_config(tmp_path, monkeypatch)
    store = Store(Path(cfg.telemetry.db_path))
    events = EventLog(cfg.device.id, cfg.device.env, Path(cfg.telemetry.log_dir), store, validate=True)
    _emit_turn(events)
    service_key = "service-" + ("s" * 32)
    http = MemorySupabase()
    client = SupabaseRest("https://example.supabase.co", service_key, http=http)
    report = sync_once(cfg, store, client, events=events)
    assert report.uploaded >= 4
    blob = _bodies(http)
    assert service_key not in blob
    assert "queue_depths" not in blob
    columns = migration_columns()
    for url, body in http.posts:
        table = url.split("/rest/v1/")[1].split("?")[0]
        rows = json.loads(body)
        for row in rows:
            assert set(row) <= columns[table]
            if table == "turns":
                assert "ingested_at" not in row
                assert row["prompt_hash"].startswith("hmac-sha256:")
            if table == "turn_text":
                assert row["prompt_redacted"] == "hello"
            if table == "heartbeats":
                assert "queue_depths" not in row
            if table == "api_key_status":
                assert len(row["last4"]) == 4
                assert row["last4"] != service_key[-4:]
    assert any("/devices" in url for url, _body in http.posts)
    device_at = next(i for i, (url, _body) in enumerate(http.posts) if url.endswith("/devices?on_conflict=id") or "/devices?" in url)
    turn_at = next(i for i, (url, _body) in enumerate(http.posts) if "/turns?" in url)
    assert device_at < turn_at


def test_canary_is_not_uploaded(tmp_path, monkeypatch):
    cfg = test_config(tmp_path, monkeypatch)
    store = Store(Path(cfg.telemetry.db_path))
    events = EventLog(cfg.device.id, cfg.device.env, Path(cfg.telemetry.log_dir), store, validate=True)
    _emit_turn(events, text="4111111111111111")
    # The event schema stores the redacted form. Force a leaked canary into the row the sync reads.
    leaked = store.recent("turn.completed", 1)[0]
    leaked["prompt_redacted"] = "4111111111111111"
    store.insert_event(leaked)
    http = MemorySupabase()
    client = SupabaseRest("https://example.supabase.co", "service-" + ("s" * 24), http=http)
    sync_once(cfg, store, client)
    assert "4111111111111111" not in _bodies(http)


def test_config_change_apply_reject_and_rollback(tmp_path, monkeypatch):
    cfg = test_config(tmp_path, monkeypatch)
    target = tmp_path / "config.yaml"
    target.write_text(Path(cfg.source_path).read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setenv("GVA_CONFIG", str(target))
    cfg = test_config(tmp_path, monkeypatch)
    cfg.source_path = target
    cfg.source_data = __import__("yaml").safe_load(target.read_text(encoding="utf-8"))
    store = Store(Path(cfg.telemetry.db_path))
    events = EventLog(cfg.device.id, cfg.device.env, Path(cfg.telemetry.log_dir), store, validate=True)
    change_id = "11111111-1111-4111-8111-111111111111"
    http = MemorySupabase(
        gets=[
            [
                {
                    "id": change_id,
                    "device_id": cfg.device.id,
                    "key": "wake.threshold",
                    "new_value": 0.7,
                    "reason": "fewer false wakes",
                    "status": "pending",
                }
            ]
        ]
    )
    client = SupabaseRest("https://example.supabase.co", "service-" + ("s" * 24), http=http)
    report = sync_once(cfg, store, client, events=events)
    assert report.changes_applied == 1
    assert "threshold: 0.7" in target.read_text(encoding="utf-8") or "threshold: 0.70" in target.read_text(encoding="utf-8")

    http.gets.append(
        [
            {
                "id": "22222222-2222-4222-8222-222222222222",
                "device_id": cfg.device.id,
                "key": "llm.api_key",
                "new_value": "nope",
                "status": "pending",
            }
        ]
    )
    rejected = sync_once(cfg, store, client, events=events)
    assert rejected.changes_rejected == 1
    assert any("rejected" in body for _url, body in http.patches)

    events.emit(
        "error",
        level="critical",
        category="internal",
        error_code="internal.health",
        message="The assistant failed its health check.",
        stage="config",
        retryable=False,
        retry_count=0,
        fallback_spoken=False,
    )
    http.gets.append([])
    rolled = sync_once(cfg, store, client, events=events)
    assert rolled.rolled_back == 1
    text = target.read_text(encoding="utf-8")
    assert "threshold: 0.6" in text or "threshold: 0.60" in text


def test_sync_error_hides_the_key():
    class Boom:
        status_code = 401
        text = "service-" + ("s" * 24)

        def json(self):
            return {}

    try:
        from gva.backend.sync_supabase import _raise_for_status

        _raise_for_status(Boom(), "turns")
    except SyncError as exc:
        assert "service-" not in str(exc)
        assert "401" in str(exc)
    else:
        raise AssertionError("expected SyncError")
