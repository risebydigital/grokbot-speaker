"""Local SQLite metrics database. The JSONL log is the durable event stream; this mirrors it."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    ts TEXT NOT NULL,
    mono_ms INTEGER,
    device_id TEXT NOT NULL,
    session_id TEXT,
    turn_id TEXT,
    request_id TEXT,
    level TEXT,
    source TEXT,
    body TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_type_ts ON events(type, ts);
CREATE INDEX IF NOT EXISTS events_turn ON events(turn_id);

CREATE TABLE IF NOT EXISTS ingest_state (
    file TEXT PRIMARY KEY,
    offset INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS sync_state (
    event_id TEXT PRIMARY KEY,
    synced_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rollup_hourly (
    bucket TEXT NOT NULL,
    device_id TEXT NOT NULL,
    turns INTEGER NOT NULL DEFAULT 0,
    turns_ok INTEGER NOT NULL DEFAULT 0,
    errors INTEGER NOT NULL DEFAULT 0,
    tokens_prompt INTEGER NOT NULL DEFAULT 0,
    tokens_completion INTEGER NOT NULL DEFAULT 0,
    tokens_reasoning INTEGER NOT NULL DEFAULT 0,
    tokens_cached INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    p50_e2e_ms INTEGER,
    p95_e2e_ms INTEGER,
    p50_ttft_ms INTEGER,
    p95_ttft_ms INTEGER,
    p50_stt_ms INTEGER,
    p95_stt_ms INTEGER,
    wake_events INTEGER NOT NULL DEFAULT 0,
    false_wakes INTEGER NOT NULL DEFAULT 0,
    heartbeat_minutes INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (bucket, device_id)
);

CREATE TABLE IF NOT EXISTS rollup_daily (
    bucket TEXT NOT NULL,
    device_id TEXT NOT NULL,
    turns INTEGER NOT NULL DEFAULT 0,
    turns_ok INTEGER NOT NULL DEFAULT 0,
    errors INTEGER NOT NULL DEFAULT 0,
    tokens_prompt INTEGER NOT NULL DEFAULT 0,
    tokens_completion INTEGER NOT NULL DEFAULT 0,
    tokens_reasoning INTEGER NOT NULL DEFAULT 0,
    tokens_cached INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    p50_e2e_ms INTEGER,
    p95_e2e_ms INTEGER,
    p50_ttft_ms INTEGER,
    p95_ttft_ms INTEGER,
    p50_stt_ms INTEGER,
    p95_stt_ms INTEGER,
    uptime_pct REAL,
    PRIMARY KEY (bucket, device_id)
);

CREATE TABLE IF NOT EXISTS timers (
    id TEXT PRIMARY KEY,
    label TEXT,
    duration_seconds INTEGER NOT NULL,
    started_at TEXT NOT NULL,
    ends_at TEXT NOT NULL,
    status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS alarms (
    id TEXT PRIMARY KEY,
    label TEXT,
    time_hhmm TEXT NOT NULL,
    day TEXT,
    fires_at TEXT NOT NULL,
    status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS device_state (
    id TEXT PRIMARY KEY,
    state TEXT,
    level INTEGER
);

CREATE TABLE IF NOT EXISTS replay_cache (
    turn_id TEXT PRIMARY KEY,
    raw_text TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS turn_flags (
    turn_id TEXT PRIMARY KEY,
    flagged INTEGER NOT NULL,
    flagged_at TEXT
);

CREATE TABLE IF NOT EXISTS applied_changes (
    change_id TEXT PRIMARY KEY,
    key TEXT NOT NULL,
    old_snapshot TEXT NOT NULL,
    applied_at TEXT NOT NULL,
    watch_until TEXT NOT NULL,
    status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS alert_rules (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    metric TEXT NOT NULL,
    operator TEXT NOT NULL,
    threshold REAL NOT NULL,
    window_minutes INTEGER NOT NULL,
    severity TEXT NOT NULL,
    channel TEXT NOT NULL,
    enabled INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS alerts (
    id TEXT PRIMARY KEY,
    rule_id TEXT,
    device_id TEXT,
    severity TEXT NOT NULL,
    title TEXT NOT NULL,
    status TEXT NOT NULL,
    first_at TEXT NOT NULL,
    last_at TEXT NOT NULL,
    acknowledged_by TEXT,
    muted_until TEXT,
    synced INTEGER NOT NULL DEFAULT 0
);
"""


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=5)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def insert_event(self, event: dict[str, Any]) -> bool:
        cur = self._conn.execute(
            """INSERT OR IGNORE INTO events
               (event_id, type, ts, mono_ms, device_id, session_id, turn_id, request_id, level, source, body)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event["event_id"],
                event["type"],
                event["ts"],
                event.get("mono_ms"),
                event["device_id"],
                event.get("session_id"),
                event.get("turn_id"),
                event.get("request_id"),
                event.get("level"),
                event.get("source"),
                json.dumps(event, separators=(",", ":"), ensure_ascii=False),
            ),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def recent(self, type_: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        if type_:
            rows = self._conn.execute(
                "SELECT body FROM events WHERE type = ? ORDER BY ts DESC LIMIT ?",
                (type_, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT body FROM events ORDER BY ts DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [json.loads(row["body"]) for row in rows]

    def events_between(self, start: str, end: str, type_: str | None = None) -> list[dict[str, Any]]:
        if type_:
            rows = self._conn.execute(
                "SELECT body FROM events WHERE ts >= ? AND ts < ? AND type = ? ORDER BY ts",
                (start, end, type_),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT body FROM events WHERE ts >= ? AND ts < ? ORDER BY ts",
                (start, end),
            ).fetchall()
        return [json.loads(row["body"]) for row in rows]

    def all_events(self) -> list[dict[str, Any]]:
        rows = self._conn.execute("SELECT body FROM events ORDER BY ts, event_id").fetchall()
        return [json.loads(row["body"]) for row in rows]

    def unsynced(self, limit: int) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """SELECT e.body FROM events e
               LEFT JOIN sync_state s ON s.event_id = e.event_id
               WHERE s.event_id IS NULL
               ORDER BY e.ts, e.event_id
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [json.loads(row["body"]) for row in rows]

    def mark_synced(self, event_ids: list[str], at: str) -> None:
        self._conn.executemany(
            "INSERT OR REPLACE INTO sync_state (event_id, synced_at) VALUES (?, ?)",
            [(event_id, at) for event_id in event_ids],
        )
        self._conn.commit()

    def ingest_offset(self, name: str) -> int:
        row = self._conn.execute("SELECT offset FROM ingest_state WHERE file = ?", (name,)).fetchone()
        return int(row["offset"]) if row else 0

    def set_ingest_offset(self, name: str, offset: int) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO ingest_state (file, offset) VALUES (?, ?)",
            (name, offset),
        )
        self._conn.commit()

    def replace_rollups(self, hourly: list[dict[str, Any]], daily: list[dict[str, Any]]) -> None:
        self._conn.execute("DELETE FROM rollup_hourly")
        self._conn.execute("DELETE FROM rollup_daily")
        if hourly:
            cols = list(hourly[0])
            placeholders = ",".join("?" for _ in cols)
            self._conn.executemany(
                f"INSERT INTO rollup_hourly ({','.join(cols)}) VALUES ({placeholders})",
                [tuple(row[c] for c in cols) for row in hourly],
            )
        if daily:
            cols = list(daily[0])
            placeholders = ",".join("?" for _ in cols)
            self._conn.executemany(
                f"INSERT INTO rollup_daily ({','.join(cols)}) VALUES ({placeholders})",
                [tuple(row[c] for c in cols) for row in daily],
            )
        self._conn.commit()

    def rollup_hourly(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._conn.execute("SELECT * FROM rollup_hourly ORDER BY bucket").fetchall()]

    def rollup_daily(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._conn.execute("SELECT * FROM rollup_daily ORDER BY bucket").fetchall()]

    def remember_replay(self, turn_id: str, text: str, at: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO replay_cache (turn_id, raw_text, created_at) VALUES (?, ?, ?)",
            (turn_id, text, at),
        )
        self._conn.commit()

    def replay_text(self, turn_id: str) -> str | None:
        row = self._conn.execute("SELECT raw_text FROM replay_cache WHERE turn_id = ?", (turn_id,)).fetchone()
        return row["raw_text"] if row else None

    def purge_replay(self, older_than: str) -> None:
        self._conn.execute("DELETE FROM replay_cache WHERE created_at < ?", (older_than,))
        self._conn.commit()

    def flag_turn(self, turn_id: str, at: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO turn_flags (turn_id, flagged, flagged_at) VALUES (?, 1, ?)",
            (turn_id, at),
        )
        self._conn.commit()

    def is_flagged(self, turn_id: str) -> bool:
        row = self._conn.execute("SELECT flagged FROM turn_flags WHERE turn_id = ?", (turn_id,)).fetchone()
        return bool(row and row["flagged"])

    def upsert_timer(self, row: dict[str, Any]) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO timers (id, label, duration_seconds, started_at, ends_at, status)
               VALUES (:id, :label, :duration_seconds, :started_at, :ends_at, :status)""",
            row,
        )
        self._conn.commit()

    def list_timers(self, status: str = "running") -> list[dict[str, Any]]:
        rows = self._conn.execute("SELECT * FROM timers WHERE status = ? ORDER BY ends_at", (status,)).fetchall()
        return [dict(row) for row in rows]

    def due_timers(self, now_iso: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM timers WHERE status = 'running' AND ends_at <= ? ORDER BY ends_at",
            (now_iso,),
        ).fetchall()
        return [dict(row) for row in rows]

    def upsert_alarm(self, row: dict[str, Any]) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO alarms (id, label, time_hhmm, day, fires_at, status)
               VALUES (:id, :label, :time_hhmm, :day, :fires_at, :status)""",
            row,
        )
        self._conn.commit()

    def due_alarms(self, now_iso: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM alarms WHERE status = 'set' AND fires_at <= ?",
            (now_iso,),
        ).fetchall()
        return [dict(row) for row in rows]

    def set_device(self, device_id: str, state: str | None, level: int | None) -> None:
        current = self._conn.execute("SELECT state, level FROM device_state WHERE id = ?", (device_id,)).fetchone()
        if current:
            state = current["state"] if state is None else state
            level = current["level"] if level is None else level
        self._conn.execute(
            "INSERT OR REPLACE INTO device_state (id, state, level) VALUES (?, ?, ?)",
            (device_id, state, level),
        )
        self._conn.commit()

    def get_device(self, device_id: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM device_state WHERE id = ?", (device_id,)).fetchone()
        return dict(row) if row else None

    def record_change(self, change_id: str, key: str, snapshot: str, applied_at: str, watch_until: str) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO applied_changes
               (change_id, key, old_snapshot, applied_at, watch_until, status)
               VALUES (?, ?, ?, ?, ?, 'watching')""",
            (change_id, key, snapshot, applied_at, watch_until),
        )
        self._conn.commit()

    def watching_changes(self) -> list[dict[str, Any]]:
        rows = self._conn.execute("SELECT * FROM applied_changes WHERE status = 'watching'").fetchall()
        return [dict(row) for row in rows]

    def finish_change(self, change_id: str, status: str) -> None:
        self._conn.execute("UPDATE applied_changes SET status = ? WHERE change_id = ?", (status, change_id))
        self._conn.commit()

    def delete_events_before(self, ts: str) -> int:
        cur = self._conn.execute("DELETE FROM events WHERE ts < ?", (ts,))
        self._conn.commit()
        return cur.rowcount

    def vacuum(self) -> None:
        self._conn.execute("VACUUM")

    def event_by_turn(self, turn_id: str, type_: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT body FROM events WHERE type = ? AND turn_id = ? ORDER BY ts DESC LIMIT 1",
            (type_, turn_id),
        ).fetchone()
        return json.loads(row["body"]) if row else None

    def turn_synced(self, turn_id: str) -> bool:
        row = self._conn.execute(
            """SELECT 1 FROM events e
               JOIN sync_state s ON s.event_id = e.event_id
               WHERE e.type = 'turn.completed' AND e.turn_id = ?""",
            (turn_id,),
        ).fetchone()
        return row is not None

    def alert_rule_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS n FROM alert_rules").fetchone()
        return int(row["n"])

    def insert_alert_rule(self, row: dict[str, Any]) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO alert_rules
               (id, name, metric, operator, threshold, window_minutes, severity, channel, enabled)
               VALUES (:id, :name, :metric, :operator, :threshold, :window_minutes, :severity, :channel, :enabled)""",
            row,
        )
        self._conn.commit()

    def alert_rules(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._conn.execute("SELECT * FROM alert_rules").fetchall()]

    def insert_alert(self, row: dict[str, Any]) -> None:
        self._conn.execute(
            """INSERT INTO alerts
               (id, rule_id, device_id, severity, title, status, first_at, last_at, acknowledged_by, muted_until, synced)
               VALUES (:id, :rule_id, :device_id, :severity, :title, :status, :first_at, :last_at,
                       :acknowledged_by, :muted_until, 0)""",
            row,
        )
        self._conn.commit()

    def spend_since(self, start: str) -> float:
        row = self._conn.execute(
            """SELECT COALESCE(SUM(json_extract(body, '$.cost_usd')), 0) AS cost
               FROM events WHERE type = 'llm.completed' AND ts >= ?""",
            (start,),
        ).fetchone()
        return float(row["cost"] or 0)
