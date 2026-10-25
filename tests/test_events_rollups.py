import json

from gva.ids import new_ulid
from gva.telemetry.events import EventLog
from gva.telemetry.redact import prompt_hash
from gva.telemetry.rollups import rebuild
from gva.telemetry.store import Store
from tests.fakes import test_config


def test_ulid_and_event_round_trip(tmp_path, monkeypatch):
    cfg = test_config(tmp_path, monkeypatch)
    store = Store(tmp_path / "metrics.db")
    events = EventLog(cfg.device.id, cfg.device.env, tmp_path / "log", store, validate=True)
    event_id = new_ulid()
    assert len(event_id) == 26
    pepper = "pepper-" + ("p" * 32)
    digest = prompt_hash("hello", pepper)
    events.ctx.turn_id = new_ulid()
    events.ctx.session_id = new_ulid()
    events.emit(
        "turn.completed",
        status="ok",
        prompt_redacted="hello",
        prompt_hash=digest,
        prompt_chars=5,
        response_redacted="Hi.",
        response_hash=prompt_hash("hi.", pepper),
        stage_timings={"stt_ms": 10},
        total_ms=20,
        fallback_spoken=False,
        cost_usd=0.01,
        tokens_total=3,
    )
    events.emit(
        "llm.completed",
        model="grok-4.7",
        round=1,
        status="ok",
        duration_ms=15,
        cost_usd=0.25,
        usage={"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14, "cached_tokens": 0, "reasoning_tokens": 0},
    )
    path = next((tmp_path / "log").glob("events-*.jsonl"))
    line = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert line["schema_version"] == "1.0.0"
    assert line["ts"].endswith("Z")
    hourly, daily = rebuild(store)
    assert hourly[0]["cost_usd"] == 0.25
    assert hourly[0]["turns"] == 1
    assert daily[0]["turns_ok"] == 1
