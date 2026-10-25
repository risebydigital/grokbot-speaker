"""Push redacted telemetry to Supabase and apply pending config changes.

The device uses SUPABASE_SERVICE_KEY, which bypasses row-level security.
A new ``sb_secret_`` key is sent only as the ``apikey`` header. A legacy
service-role JWT is also sent as ``Authorization: Bearer``. The key is read
from the environment at call time and is never written to a payload, a log
line, or an exception message. The browser dashboard must use the anon key only.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from gva import __version__
from gva.backend.schema_columns import ON_CONFLICT, fit
from gva.backend.supabase_headers import supabase_rest_headers
from gva.config import (
    Config,
    ConfigChangeError,
    config_fingerprint,
    env_value,
    save_config,
    set_dotted,
    validate_change,
)
from gva.ids import utc_now_ms
from gva.runtime.grok_client import GrokClient
from gva.telemetry.redact import contains_canary, redact
from gva.telemetry.rollups import rebuild
from gva.telemetry.store import Store

# Uploaded as rows. Everything else is acknowledged locally so it cannot clog the queue.
# bridge.* stays in the local log. Those events have no voice-ops table.
UPLOAD_TYPES = {
    "turn.completed",
    "llm.completed",
    "tool.completed",
    "tool.rejected",
    "error",
    "heartbeat",
}

CHILD_TYPES = {"llm.completed", "tool.completed", "tool.rejected", "error"}


class SyncError(RuntimeError):
    """HTTP failure. The message is safe to log."""


@dataclass
class SyncReport:
    uploaded: int = 0
    deferred: int = 0
    dropped: int = 0
    changes_applied: int = 0
    changes_rejected: int = 0
    rolled_back: int = 0
    tables: list[str] = field(default_factory=list)


class SupabaseRest:
    def __init__(self, url: str, service_key: str, http: Any | None = None):
        self.url = url.rstrip("/")
        self._key = service_key
        self.http = http or httpx.Client(timeout=20)

    def upsert(self, table: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        conflict = ON_CONFLICT[table]
        response = self.http.post(
            f"{self.url}/rest/v1/{table}?on_conflict={conflict}",
            headers=self._headers("resolution=merge-duplicates,return=minimal"),
            content=json.dumps(rows, default=str),
        )
        _raise_for_status(response, table)

    def select(self, table: str, query: str) -> list[dict[str, Any]]:
        response = self.http.get(
            f"{self.url}/rest/v1/{table}?{query}",
            headers=self._headers(),
        )
        _raise_for_status(response, table)
        body = response.json() if hasattr(response, "json") else json.loads(response.text or "[]")
        if not isinstance(body, list):
            raise SyncError(f"supabase {table} returned an unexpected payload")
        return body

    def patch(self, table: str, query: str, body: dict[str, Any]) -> None:
        response = self.http.patch(
            f"{self.url}/rest/v1/{table}?{query}",
            headers=self._headers("return=minimal"),
            content=json.dumps(body, default=str),
        )
        _raise_for_status(response, table)

    def _headers(self, prefer: str | None = None) -> dict[str, str]:
        return supabase_rest_headers(self._key, prefer=prefer)


def client_from_env(http: Any | None = None) -> SupabaseRest | None:
    url = env_value("SUPABASE_URL")
    key = env_value("SUPABASE_SERVICE_KEY")
    if not url or not key:
        return None
    return SupabaseRest(url, key, http=http)


def sync_once(
    cfg: Config,
    store: Store,
    client: SupabaseRest,
    *,
    grok: GrokClient | None = None,
    events: Any | None = None,
) -> SyncReport:
    report = SyncReport()
    secrets = _secret_values()
    pending = store.unsynced(cfg.sync.batch_size)
    local_ids: list[str] = []
    upload: list[dict[str, Any]] = []
    for event in pending:
        if event.get("level") == "debug" or event.get("type") not in UPLOAD_TYPES:
            local_ids.append(event["event_id"])
            continue
        upload.append(event)
    upload, deferred = _with_parents(store, upload)
    report.deferred = deferred

    grouped = _project(cfg, store, upload, secrets)
    dropped_ids = grouped.pop("_dropped", [])
    report.dropped = len(dropped_ids)
    if grouped.pop("_refused", False):
        if events is not None:
            events.emit(
                "error",
                level="error",
                category="internal",
                error_code="internal.sync_refused",
                message="A telemetry row still contained a canary and was not uploaded.",
                stage="sync",
                retryable=False,
                retry_count=0,
                fallback_spoken=False,
            )
    _upsert_all(client, cfg, store, grouped, report)
    uploaded_ids = [event["event_id"] for event in upload if event["event_id"] not in set(dropped_ids)]
    # Dropped rows are acknowledged so a canary cannot block the queue or be retried.
    store.mark_synced(local_ids + uploaded_ids + dropped_ids, utc_now_ms())
    report.uploaded = len(uploaded_ids)

    applied, rejected = apply_config_changes(cfg, client, store, grok=grok, events=events)
    report.changes_applied = applied
    report.changes_rejected = rejected
    report.rolled_back = rollback_if_unhealthy(cfg, client, store)
    if events is not None:
        events.emit(
            "sync.completed",
            uploaded=report.uploaded,
            deferred=report.deferred,
            dropped=report.dropped,
            changes_applied=report.changes_applied,
            rolled_back=report.rolled_back,
        )
    return report


def run_forever(cfg: Config, store: Store, client: SupabaseRest, grok: GrokClient | None = None) -> None:
    delay = cfg.sync.interval_seconds
    while True:
        try:
            sync_once(cfg, store, client, grok=grok)
            delay = cfg.sync.interval_seconds
        except SyncError:
            delay = min(300, max(delay * 2, cfg.sync.interval_seconds))
        time.sleep(delay)


def apply_config_changes(
    cfg: Config,
    client: SupabaseRest,
    store: Store,
    *,
    grok: GrokClient | None = None,
    events: Any | None = None,
) -> tuple[int, int]:
    query = f"device_id=eq.{cfg.device.id}&status=eq.pending&select=*"
    try:
        rows = client.select("config_changes", query)
    except SyncError:
        raise
    applied = 0
    rejected = 0
    for change in rows:
        change_id = str(change.get("id") or "")
        key = str(change.get("key") or "")
        value = _unwrap(change.get("new_value"))
        try:
            coerced = validate_change(key, value)
        except ConfigChangeError as exc:
            client.patch(
                "config_changes",
                f"id=eq.{change_id}",
                {"status": "rejected", "rejection_reason": str(exc)[:300], "applied_at": utc_now_ms()},
            )
            rejected += 1
            continue
        if key == "action.health_check":
            _health_check(cfg, client, grok)
        else:
            snapshot = _dump(cfg)
            _write_setting(cfg, key, coerced)
            watch = datetime.now(timezone.utc) + timedelta(seconds=cfg.sync.rollback_seconds)
            store.record_change(change_id, key, snapshot, utc_now_ms(), watch.isoformat())
        client.patch(
            "config_changes",
            f"id=eq.{change_id}",
            {"status": "applied", "applied_at": utc_now_ms()},
        )
        applied += 1
        if events is not None and key != "action.health_check":
            events.emit(
                "config.changed",
                source="backend",
                key=key,
                new_value=coerced if not isinstance(coerced, str) or len(coerced) < 80 else "updated",
            )
    _push_config_current(cfg, client)
    _push_key_status(cfg, client, ok=None)
    return applied, rejected


def rollback_if_unhealthy(cfg: Config, client: SupabaseRest, store: Store) -> int:
    rolled = 0
    now = datetime.now(timezone.utc)
    for change in store.watching_changes():
        try:
            deadline = datetime.fromisoformat(change["watch_until"])
        except ValueError:
            store.finish_change(change["change_id"], "applied")
            continue
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=timezone.utc)
        unhealthy = _unhealthy_since(store, change["applied_at"])
        if unhealthy and now <= deadline:
            restored = _load_snapshot(change["old_snapshot"])
            if restored is not None:
                cfg.source_data = restored
                _reload(cfg)
                save_config(cfg)
            client.patch(
                "config_changes",
                f"id=eq.{change['change_id']}",
                {"status": "rolled_back", "rejection_reason": "Health worsened within the rollback window."},
            )
            store.finish_change(change["change_id"], "rolled_back")
            rolled += 1
        elif now > deadline:
            store.finish_change(change["change_id"], "applied")
    return rolled


def push_eval(client: SupabaseRest, run: dict[str, Any], failures: list[dict[str, Any]]) -> None:
    client.upsert("eval_runs", [fit("eval_runs", run)])
    if failures:
        client.upsert("eval_failures", [fit("eval_failures", row) for row in failures])


def _with_parents(store: Store, events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    by_id = {event["event_id"]: event for event in events}
    deferred = 0
    kept: list[dict[str, Any]] = []
    for event in events:
        if event.get("type") not in CHILD_TYPES or not event.get("turn_id"):
            kept.append(event)
            continue
        turn_id = event["turn_id"]
        parent = _find_turn(events, turn_id) or store.event_by_turn(turn_id, "turn.completed")
        if parent is None:
            if store.turn_synced(turn_id):
                kept.append(event)
            else:
                deferred += 1
            continue
        if parent["event_id"] not in by_id and not store.turn_synced(turn_id):
            by_id[parent["event_id"]] = parent
            kept.append(parent)
        kept.append(event)
    # Parents first.
    kept.sort(key=lambda event: (0 if event.get("type") == "turn.completed" else 1, event.get("ts") or ""))
    return kept, deferred


def _find_turn(events: list[dict[str, Any]], turn_id: str) -> dict[str, Any] | None:
    for event in events:
        if event.get("type") == "turn.completed" and event.get("turn_id") == turn_id:
            return event
    return None


def _project(cfg: Config, store: Store, events: list[dict[str, Any]], secrets: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "turns": [],
        "llm_calls": [],
        "tool_calls": [],
        "errors": [],
        "heartbeats": [],
        "turn_text": [],
        "_dropped": [],
        "_refused": False,
    }
    for event in events:
        etype = event.get("type")
        row: dict[str, Any] | None
        text_row = None
        table = ""
        if etype == "turn.completed":
            row, text_row = _turn_rows(cfg, store, event)
            table = "turns"
        elif etype == "llm.completed":
            row = _llm_row(event)
            table = "llm_calls"
        elif etype in {"tool.completed", "tool.rejected"}:
            row = _tool_row(event)
            table = "tool_calls"
        elif etype == "error":
            row = _error_row(event)
            table = "errors"
        elif etype == "heartbeat":
            row = _heartbeat_row(cfg, event)
            table = "heartbeats"
        else:
            continue
        blob = json.dumps({"row": row, "text": text_row}, default=str)
        blob = _scrub(blob, secrets)
        if contains_canary(blob) or any(secret and secret in blob for secret in secrets):
            out["_dropped"].append(event["event_id"])
            out["_refused"] = True
            continue
        cleaned = json.loads(blob)
        if table:
            out[table].append(fit(table, cleaned["row"]))
        if text_row is not None:
            out["turn_text"].append(fit("turn_text", cleaned["text"]))
    return out


def _upsert_all(client: SupabaseRest, cfg: Config, store: Store, grouped: dict[str, Any], report: SyncReport) -> None:
    heartbeats = grouped.get("heartbeats") or []
    latest = heartbeats[-1] if heartbeats else None
    device = fit(
        "devices",
        {
            "id": cfg.device.id,
            "name": cfg.device.name,
            "app_version": __version__,
            "last_heartbeat_at": latest.get("at") if latest else None,
            "status": _device_status(latest),
        },
    )
    client.upsert("devices", [device])
    report.tables.append("devices")
    for table in ("turns", "llm_calls", "tool_calls", "errors", "heartbeats", "turn_text"):
        rows = grouped.get(table) or []
        if rows:
            client.upsert(table, rows)
            report.tables.append(table)
    hourly, daily = rebuild(store)
    if hourly:
        client.upsert("rollup_hourly", [fit("rollup_hourly", row) for row in hourly])
        report.tables.append("rollup_hourly")
    if daily:
        client.upsert("rollup_daily", [fit("rollup_daily", row) for row in daily])
        report.tables.append("rollup_daily")
    _push_config_current(cfg, client)
    _push_key_status(cfg, client, ok=latest.get("api_reachable") if latest else None)


def _turn_rows(cfg: Config, store: Store, event: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    turn_id = event.get("turn_id") or event["event_id"]
    sensitive = bool(event.get("sensitive"))
    prompt = _sync_text(event.get("prompt_redacted"), cfg, sensitive)
    response = _sync_text(event.get("response_redacted"), cfg, sensitive)
    row = {
        "id": turn_id,
        "device_id": event["device_id"],
        "session_id": event.get("session_id") or "local",
        "started_at": event.get("started_at") or event["ts"],
        "status": event.get("status") or "ok",
        "intent": event.get("intent"),
        "prompt_hash": event.get("prompt_hash"),
        "response_hash": event.get("response_hash"),
        "sensitive": sensitive,
        "sensitive_category": event.get("sensitive_category"),
        "hash_epoch": event.get("hash_epoch") or cfg.privacy.hash_epoch,
        "tools_called": event.get("tools_called") or [],
        "rounds": event.get("rounds") or 0,
        "e2e_first_audio_ms": event.get("e2e_first_audio_ms"),
        "total_ms": event.get("total_ms"),
        "tokens_total": event.get("tokens_total") or 0,
        "cost_usd": event.get("cost_usd"),
        "model": event.get("model"),
        "stage_timings": event.get("stage_timings") or {},
        "flagged_for_review": store.is_flagged(turn_id),
    }
    text = {
        "turn_id": turn_id,
        "prompt_redacted": None if sensitive else prompt,
        "response_redacted": None if sensitive else response,
        "expires_at": (datetime.now(timezone.utc) + timedelta(days=30)).isoformat(),
    }
    return row, text


def _llm_row(event: dict[str, Any]) -> dict[str, Any]:
    usage = event.get("usage") or {}
    status = event.get("status") or "error"
    if status not in {"ok", "error", "cancelled"}:
        status = "error"
    return {
        "id": event.get("request_id") or event["event_id"],
        "turn_id": event.get("turn_id"),
        "device_id": event["device_id"],
        "model": event.get("model") or "unknown",
        "round": event.get("round") or 1,
        "status": status,
        "http_status": event.get("http_status"),
        "ttft_ms": event.get("ttft_ms"),
        "duration_ms": event.get("duration_ms") or 0,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "reasoning_tokens": int(usage.get("reasoning_tokens") or 0),
        "cached_tokens": int(usage.get("cached_tokens") or 0),
        "cost_usd": event.get("cost_usd"),
        "cost_estimated": bool(event.get("cost_estimated")),
        "xai_response_id": event.get("xai_response_id"),
        "system_fingerprint": event.get("system_fingerprint"),
        "created_at": event["ts"],
    }


def _tool_row(event: dict[str, Any]) -> dict[str, Any]:
    status = event.get("status") or ("rejected" if event.get("type") == "tool.rejected" else "error")
    if status not in {"ok", "error", "timeout", "rejected"}:
        status = "error"
    return {
        "id": event["event_id"],
        "turn_id": event.get("turn_id"),
        "device_id": event["device_id"],
        "tool_name": event.get("tool_name") or "unknown",
        "status": status,
        "duration_ms": event.get("duration_ms") or 0,
        "error_code": event.get("error_code"),
        "args_redacted": event.get("args_redacted") or {},
        "created_at": event["ts"],
    }


def _error_row(event: dict[str, Any]) -> dict[str, Any]:
    message = event.get("message") or "error"
    cleaned = redact(str(message), short_numbers=True).text[:300]
    return {
        "id": event["event_id"],
        "turn_id": event.get("turn_id"),
        "device_id": event["device_id"],
        "category": event.get("category") or "internal",
        "error_code": event.get("error_code") or "internal.unhandled",
        "message": cleaned,
        "stage": event.get("stage"),
        "fallback_spoken": bool(event.get("fallback_spoken")),
        "created_at": event["ts"],
    }


def _heartbeat_row(cfg: Config, event: dict[str, Any]) -> dict[str, Any]:
    return {
        "device_id": event["device_id"],
        "at": event["ts"],
        "cpu_pct": event.get("cpu_pct"),
        "mem_pct": event.get("mem_pct"),
        "disk_pct": event.get("disk_pct"),
        "temp_c": event.get("temp_c"),
        "throttled": event.get("throttled"),
        "sync_lag_s": 0,
        "api_reachable": event.get("api_reachable"),
        "last_api_ok_at": event.get("last_api_ok_at"),
    }


def _sync_text(value: Any, cfg: Config, sensitive: bool) -> str | None:
    if sensitive or not isinstance(value, str) or not value:
        return None
    if cfg.privacy.sync_redact_short_numbers:
        value = redact(
            value,
            short_numbers=True,
            household_names=cfg.privacy.household_names,
            categories=cfg.privacy.sensitive_categories,
        ).text
    return value[:2000]


def _device_status(heartbeat: dict[str, Any] | None) -> str:
    if not heartbeat:
        return "unknown"
    if heartbeat.get("api_reachable") is False:
        return "degraded"
    return "healthy"


def _push_config_current(cfg: Config, client: SupabaseRest) -> None:
    data = cfg.model_dump(mode="json")
    client.upsert(
        "config_current",
        [
            fit(
                "config_current",
                {
                    "device_id": cfg.device.id,
                    "config": data,
                    "config_hash": config_fingerprint(data),
                    "applied_at": utc_now_ms(),
                },
            )
        ],
    )


def _push_key_status(cfg: Config, client: SupabaseRest, ok: bool | None) -> None:
    key = env_value("XAI_API_KEY")
    if not key or len(key) < 4:
        return
    row = {
        "env": cfg.device.env,
        "label": "xai",
        "last4": key[-4:],
        "last_check_at": utc_now_ms(),
        "last_check_ok": ok,
    }
    client.upsert("api_key_status", [fit("api_key_status", row)])


def _health_check(cfg: Config, client: SupabaseRest, grok: GrokClient | None) -> None:
    checker = grok or GrokClient(cfg)
    ok, _detail = checker.list_models()
    _push_key_status(cfg, client, ok=ok)


def _unhealthy_since(store: Store, since: str) -> bool:
    for event in store.events_between(since, "9999-01-01T00:00:00Z"):
        if event.get("type") == "error" and event.get("level") == "critical":
            return True
        if event.get("type") == "heartbeat" and event.get("api_reachable") is False:
            return True
    return False


def _write_setting(cfg: Config, key: str, value: Any) -> None:
    if cfg.source_path is None:
        raise ConfigChangeError("configuration file is not writable")
    set_dotted(cfg.source_data, key, value)
    _reload(cfg)
    save_config(cfg)


def _reload(cfg: Config) -> None:
    updated = Config.model_validate(cfg.source_data)
    for name in updated.model_fields:
        if name in {"source_path", "source_data"}:
            continue
        setattr(cfg, name, getattr(updated, name))


def _dump(cfg: Config) -> str:
    return json.dumps(cfg.source_data, default=str)


def _load_snapshot(raw: str) -> dict[str, Any] | None:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _unwrap(value: Any) -> Any:
    if isinstance(value, dict) and set(value) == {"value"}:
        return value["value"]
    return value


def _secret_values() -> list[str]:
    values = []
    for name in ("XAI_API_KEY", "SUPABASE_SERVICE_KEY", "GVA_HASH_PEPPER", "GVA_API_TOKEN_SECRET"):
        value = env_value(name)
        if value:
            values.append(value)
    return values


def _scrub(blob: str, secrets: list[str]) -> str:
    for secret in secrets:
        if secret and secret in blob:
            blob = blob.replace(secret, "[SECRET]")
    return blob


def _raise_for_status(response: Any, table: str) -> None:
    status = getattr(response, "status_code", 200)
    if status >= 400:
        raise SyncError(f"supabase {table} returned HTTP {status}")
