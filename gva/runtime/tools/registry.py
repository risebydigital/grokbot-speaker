"""Load tool definitions and execute them with validation, timeouts and redaction."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import jsonschema
import yaml

from gva.config import Config
from gva.paths import tools_dir
from gva.runtime.bridge import bridge_configured
from gva.runtime.tools.builtin import HANDLERS
from gva.telemetry.redact import redact_value
from gva.telemetry.store import Store

MAX_RESULT_CHARS = 4096


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    timeout_s: float
    risk: str
    personal_data: bool
    handler: Callable


@dataclass
class ToolContext:
    cfg: Config
    store: Store
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)
    http: Any | None = None
    bridge: Any | None = None


@dataclass
class ToolOutcome:
    name: str
    call_id: str
    status: str
    args: dict[str, Any]
    args_redacted: dict[str, Any]
    result: dict[str, Any]
    duration_ms: int
    error_code: str | None = None
    risk: str = "low"
    rejected: bool = False


@dataclass
class ToolRegistry:
    specs: dict[str, ToolSpec]
    ctx: ToolContext
    _done: dict[str, ToolOutcome] = field(default_factory=dict)

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description.strip(),
                    "parameters": spec.parameters,
                },
            }
            for spec in self.specs.values()
        ]

    def run(self, call_id: str, name: str, arguments: str) -> ToolOutcome:
        if call_id in self._done:
            return self._done[call_id]
        spec = self.specs.get(name)
        if spec is None:
            outcome = ToolOutcome(
                name=name,
                call_id=call_id,
                status="rejected",
                args={},
                args_redacted={},
                result={"error": "unknown_tool"},
                duration_ms=0,
                error_code="tool.unknown",
                rejected=True,
            )
            self._done[call_id] = outcome
            return outcome
        try:
            parsed = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            outcome = self._finish(spec, call_id, {}, {"error": "invalid_json"}, "error", "tool.invalid_args", 0)
            self._done[call_id] = outcome
            return outcome
        if not isinstance(parsed, dict):
            parsed = {}
        try:
            jsonschema.validate(parsed, spec.parameters)
        except jsonschema.ValidationError as exc:
            outcome = self._finish(
                spec, call_id, parsed, {"error": "invalid_args", "detail": exc.message[:200]}, "error", "tool.invalid_args", 0
            )
            self._done[call_id] = outcome
            return outcome
        started = __import__("time").monotonic()
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(spec.handler, parsed, self.ctx)
                result = future.result(timeout=spec.timeout_s)
            status = "ok" if result.get("ok", True) else "error"
            code = None if status == "ok" else "tool.failed"
        except FuturesTimeout:
            result = {"error": "timeout"}
            status = "timeout"
            code = "tool.timeout"
        except Exception as exc:
            result = {"error": "failed", "detail": exc.__class__.__name__}
            status = "error"
            code = "tool.failed"
        duration = int((__import__("time").monotonic() - started) * 1000)
        outcome = self._finish(spec, call_id, parsed, result, status, code, duration)
        self._done[call_id] = outcome
        return outcome

    def _finish(self, spec, call_id, args, result, status, code, duration) -> ToolOutcome:
        redacted = redact_value(
            args,
            household_names=self.ctx.cfg.privacy.household_names,
            short_numbers=False,
            categories=self.ctx.cfg.privacy.sensitive_categories,
        )
        if not isinstance(result, dict):
            result = {"ok": True, "value": result}
        encoded = json.dumps(result)[:MAX_RESULT_CHARS]
        trimmed = json.loads(encoded) if encoded.endswith("}") else {"ok": False, "error": "truncated"}
        return ToolOutcome(
            name=spec.name,
            call_id=call_id,
            status=status,
            args=args,
            args_redacted=redacted if isinstance(redacted, dict) else {},
            result=trimmed,
            duration_ms=duration,
            error_code=code,
            risk=spec.risk,
        )


def load_registry(cfg: Config, store: Store, http: Any | None = None, now: Callable | None = None) -> ToolRegistry:
    specs: dict[str, ToolSpec] = {}
    directory = tools_dir()
    if directory.is_dir():
        for path in sorted(directory.glob("*.yaml")):
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            name = data["name"]
            if name == "ask_rbd_team" and not bridge_configured(cfg):
                continue
            if name == "get_calendar_events" and bridge_configured(cfg):
                data["description"] = (
                    "No separate calendar is connected here. Rise By Digital work, "
                    "Asana, reports, client work, and email go to ask_rbd_team instead."
                )
            _apply_device_enum(cfg, name, data)
            handler = HANDLERS.get(name)
            if handler is None:
                continue
            specs[name] = ToolSpec(
                name=name,
                description=str(data.get("description") or name),
                parameters=data["parameters"],
                timeout_s=float(data.get("timeout_s") or 5),
                risk=str(data.get("risk") or "low"),
                personal_data=bool(data.get("personal_data")),
                handler=handler,
            )
    ctx = ToolContext(cfg=cfg, store=store, http=http)
    if now is not None:
        ctx.now = now
    return ToolRegistry(specs=specs, ctx=ctx)


def _apply_device_enum(cfg: Config, name: str, data: dict[str, Any]) -> None:
    if name not in {"switch_device", "set_device_level"}:
        return
    devices = cfg.home.devices
    if name == "set_device_level":
        devices = [device for device in devices if "level" in device.kinds]
    ids = [device.id for device in devices] or ["none"]
    data["parameters"].setdefault("properties", {}).setdefault("device", {})["enum"] = ids
