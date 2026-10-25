"""Event envelope, JSON Schema validation, and the JSONL writer."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jsonschema
from jsonschema import Draft202012Validator

from gva import __version__
from gva.ids import new_ulid
from gva.paths import schema_path
from gva.telemetry.store import Store

_MONO_BASE = __import__("time").monotonic()


def mono_ms() -> int:
    return int((__import__("time").monotonic() - _MONO_BASE) * 1000)


def utc_ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass
class LogContext:
    source: str = "runtime"
    session_id: str | None = None
    turn_id: str | None = None
    request_id: str | None = None


@dataclass
class EventLog:
    device_id: str
    env: str
    log_dir: Path
    store: Store
    schema_version: str = "1.0.0"
    validate: bool = True
    ctx: LogContext = field(default_factory=LogContext)
    _handle: Any = None
    _day: str | None = None
    _since_fsync: int = 0
    _validator: Draft202012Validator | None = None

    def __post_init__(self) -> None:
        self.log_dir = Path(self.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if self.validate:
            schema = json.loads(schema_path().read_text(encoding="utf-8"))
            self._validator = Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())

    def emit(self, type_: str, level: str = "info", **fields: Any) -> dict[str, Any]:
        event: dict[str, Any] = {
            "schema_version": self.schema_version,
            "event_id": fields.pop("event_id", None) or new_ulid(),
            "type": type_,
            "ts": fields.pop("ts", None) or utc_ts(),
            "mono_ms": int(fields.pop("mono_ms")) if "mono_ms" in fields else mono_ms(),
            "device_id": self.device_id,
            "app_version": __version__,
            "env": self.env,
            "source": fields.pop("source", None) or self.ctx.source,
            "level": level,
        }
        for key in ("session_id", "turn_id", "request_id"):
            if key in fields:
                event[key] = fields.pop(key)
            else:
                event[key] = getattr(self.ctx, key)
        event.update(fields)
        event["mono_ms"] = int(event["mono_ms"])
        if self._validator is not None:
            errors = sorted(self._validator.iter_errors(event), key=lambda err: list(err.path))
            if errors:
                detail = "; ".join(err.message for err in errors[:5])
                raise ValueError(f"event {type_} failed schema validation: {detail}")
        self._write(event)
        self.store.insert_event(event)
        return event

    def _write(self, event: dict[str, Any]) -> None:
        day = event["ts"][:10]
        if self._handle is None or day != self._day:
            if self._handle is not None:
                self._handle.flush()
                os_fsync(self._handle)
                self._handle.close()
            path = self.log_dir / f"events-{day}.jsonl"
            self._handle = path.open("a", encoding="utf-8")
            self._day = day
            self._since_fsync = 0
        self._handle.write(json.dumps(event, separators=(",", ":"), ensure_ascii=False) + "\n")
        self._handle.flush()
        self._since_fsync += 1
        important = event["type"] in {"turn.completed", "error", "app.stopped", "config.changed"}
        if important or self._since_fsync >= 10:
            os_fsync(self._handle)
            self._since_fsync = 0

    def close(self) -> None:
        if self._handle is not None:
            self._handle.flush()
            os_fsync(self._handle)
            self._handle.close()
            self._handle = None


def os_fsync(handle) -> None:
    try:
        import os

        os.fsync(handle.fileno())
    except OSError:
        return


def tail(log_dir: Path, type_: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
    files = sorted(Path(log_dir).glob("events-*.jsonl"))
    lines: list[str] = []
    for path in reversed(files):
        text = path.read_text(encoding="utf-8", errors="replace").splitlines()
        lines = text + lines
        if len(lines) > limit * 5:
            break
    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if type_ and event.get("type") != type_:
            continue
        out.append(event)
    return out[-limit:]
