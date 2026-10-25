"""Session history, the Grok tool loop, and redaction before anything is stored."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable
from zoneinfo import ZoneInfo

from gva.config import Config, env_value
from gva.ids import new_ulid
from gva.paths import prompt_path
from gva.runtime.grok_client import GrokClient, LlmResult
from gva.runtime.textutil import flush_sentences, is_no, is_yes, take_sentences
from gva.runtime.tools.registry import ToolRegistry
from gva.telemetry.events import EventLog
from gva.telemetry.redact import prompt_hash, redact
from gva.telemetry.store import Store


@dataclass
class TurnResult:
    status: str
    spoken: str
    tools_called: list[str]
    rounds: int
    tokens_total: int
    cost_usd: float | None
    model: str | None
    fallback_spoken: bool
    error_category: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    retryable: bool = False
    llm_ttft_ms: int | None = None
    llm_total_ms: int = 0
    tool_ms: int = 0
    prompt_ms: int = 0
    sensitive: bool = False
    sensitive_category: str | None = None
    prompt_redacted: str | None = None
    prompt_hash: str | None = None
    prompt_chars: int = 0
    response_redacted: str | None = None
    response_hash: str | None = None
    interrupted: bool = False


@dataclass
class Conversation:
    cfg: Config
    grok: GrokClient
    tools: ToolRegistry
    events: EventLog
    store: Store
    history: list[dict[str, str]] = field(default_factory=list)
    pending: dict[str, Any] | None = None
    _degraded_day: str | None = None

    def reset(self) -> None:
        self.history.clear()
        self.pending = None

    def execute(self, user_text: str, speak: Callable[[str], bool]) -> TurnResult:
        """Run one turn. `speak` returns True if playback was interrupted."""
        pepper = env_value("GVA_HASH_PEPPER") or ""
        prepared = _prepare(user_text, self.cfg, pepper)
        if self.cfg.telemetry.replay_cache_hours > 0 and self.events.ctx.turn_id:
            self.store.remember_replay(self.events.ctx.turn_id, user_text, _now())
        if self.pending is not None:
            return self._resolve_pending(user_text, speak, prepared)
        messages = self._assemble(user_text)
        started = time.monotonic()
        prompt_ms = int((time.monotonic() - started) * 1000)
        model = self._choose_model()
        spoken_parts: list[str] = []
        tools_called: list[str] = []
        rounds = 0
        tokens = 0
        cost = 0.0
        saw_cost = False
        ttft = None
        llm_total = 0
        tool_ms = 0
        interrupted = False
        deadline = time.monotonic() + self.cfg.conversation.turn_budget_seconds

        while rounds < self.cfg.llm.max_tool_rounds:
            rounds += 1
            buffer: list[str] = []

            def on_delta(piece: str, buf=buffer) -> None:
                buf.append(piece)

            self.events.ctx.request_id = new_ulid()
            self.events.emit(
                "llm.started",
                model=model,
                round=rounds,
                message_count=len(messages),
                tools_offered=[item["function"]["name"] for item in self.tools.schemas()],
                stream=True,
            )
            result = self.grok.complete(
                messages,
                self.tools.schemas(),
                model=model,
                on_delta=on_delta,
                deadline=deadline,
            )
            self._emit_llm(result, rounds)
            llm_total += result.duration_ms
            if ttft is None:
                ttft = result.ttft_ms
            if result.usage:
                tokens += int(result.usage.get("total_tokens") or 0)
            if result.cost_usd is not None:
                cost += result.cost_usd
                saw_cost = True
            if result.status == "cancelled":
                return TurnResult(
                    status="cancelled",
                    spoken=" ".join(spoken_parts).strip(),
                    tools_called=tools_called,
                    rounds=rounds,
                    tokens_total=tokens,
                    cost_usd=round(cost, 6) if saw_cost else None,
                    model=model,
                    fallback_spoken=False,
                    llm_ttft_ms=ttft,
                    llm_total_ms=llm_total,
                    tool_ms=tool_ms,
                    prompt_ms=prompt_ms,
                    interrupted=True,
                    **_prompt_fields(prepared),
                )
            if result.status != "ok":
                phrase = _fallback_phrase(result.error_category)
                interrupted = speak(phrase)
                self._emit_error(result, fallback_spoken=True)
                return TurnResult(
                    status="error",
                    spoken=phrase,
                    tools_called=tools_called,
                    rounds=rounds,
                    tokens_total=tokens,
                    cost_usd=round(cost, 6) if saw_cost else None,
                    model=model,
                    fallback_spoken=True,
                    error_category=result.error_category,
                    error_code=result.error_code,
                    error_message=result.error_message,
                    retryable=result.retryable,
                    llm_ttft_ms=ttft,
                    llm_total_ms=llm_total,
                    tool_ms=tool_ms,
                    prompt_ms=prompt_ms,
                    interrupted=interrupted,
                    **_prompt_fields(prepared),
                )
            if result.tool_calls:
                messages.append(_assistant_tool_message(result))
                for call in result.tool_calls:
                    spec = self.tools.specs.get(call.name)
                    if spec is not None and spec.risk == "high":
                        self.pending = {"name": call.name, "arguments": call.arguments, "id": call.id}
                        question = "Just to confirm, shall I go ahead? Say yes to continue."
                        speak(question)
                        self._remember(user_text, question)
                        response = _prepare(question, self.cfg, pepper)
                        return TurnResult(
                            status="ok",
                            spoken=question,
                            tools_called=tools_called,
                            rounds=rounds,
                            tokens_total=tokens,
                            cost_usd=round(cost, 6) if saw_cost else None,
                            model=model,
                            fallback_spoken=False,
                            llm_ttft_ms=ttft,
                            llm_total_ms=llm_total,
                            tool_ms=tool_ms,
                            prompt_ms=prompt_ms,
                            response_redacted=response["shown"],
                            response_hash=response["hash"],
                            **_prompt_fields(prepared),
                        )
                    self.events.emit("tool.started", tool_name=call.name, tool_call_id=call.id)
                    outcome = self.tools.run(call.id, call.name, call.arguments)
                    tool_ms += outcome.duration_ms
                    tools_called.append(call.name)
                    self._emit_tool(outcome)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": json.dumps(outcome.result)[:4096],
                        }
                    )
                continue
            text = result.text or "".join(buffer)
            interrupted = self._speak_stream(text, speak, spoken_parts)
            break
        else:
            text = "Sorry, I couldn't finish that."
            speak(text)
            spoken_parts.append(text)

        spoken = " ".join(part.strip() for part in spoken_parts).strip()
        self._remember(user_text, spoken)
        response = _prepare(spoken, self.cfg, pepper)
        status = "cancelled" if interrupted else "ok"
        return TurnResult(
            status=status,
            spoken=spoken,
            tools_called=tools_called,
            rounds=rounds,
            tokens_total=tokens,
            cost_usd=round(cost, 6) if saw_cost else None,
            model=model,
            fallback_spoken=False,
            llm_ttft_ms=ttft,
            llm_total_ms=llm_total,
            tool_ms=tool_ms,
            prompt_ms=prompt_ms,
            interrupted=interrupted,
            response_redacted=response["shown"],
            response_hash=response["hash"],
            **_prompt_fields(prepared),
        )

    def _resolve_pending(self, user_text: str, speak, prepared) -> TurnResult:
        pending = self.pending
        self.pending = None
        pepper = env_value("GVA_HASH_PEPPER") or ""
        if pending and is_yes(user_text):
            self.events.emit("tool.started", tool_name=pending["name"], tool_call_id=pending["id"])
            outcome = self.tools.run(pending["id"], pending["name"], pending["arguments"])
            self._emit_tool(outcome)
            spoken = "Done." if outcome.status == "ok" else "That didn't work."
            speak(spoken)
            response = _prepare(spoken, self.cfg, pepper)
            self._remember(user_text, spoken)
            return TurnResult(
                status="ok" if outcome.status == "ok" else "degraded",
                spoken=spoken,
                tools_called=[pending["name"]],
                rounds=0,
                tokens_total=0,
                cost_usd=0.0,
                model=self.cfg.llm.model,
                fallback_spoken=False,
                tool_ms=outcome.duration_ms,
                response_redacted=response["shown"],
                response_hash=response["hash"],
                **_prompt_fields(prepared),
            )
        spoken = "OK, I won't do that." if is_no(user_text) or True else ""
        if not is_yes(user_text):
            speak(spoken)
        response = _prepare(spoken, self.cfg, pepper)
        self._remember(user_text, spoken)
        return TurnResult(
            status="cancelled",
            spoken=spoken,
            tools_called=[],
            rounds=0,
            tokens_total=0,
            cost_usd=0.0,
            model=self.cfg.llm.model,
            fallback_spoken=False,
            response_redacted=response["shown"],
            response_hash=response["hash"],
            **_prompt_fields(prepared),
        )

    def _assemble(self, user_text: str) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": render_system(self.cfg)}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_text})
        system, *rest = messages
        while len(rest) > self.cfg.llm.history_messages + 1:
            rest.pop(0)
        while _estimate(rest) > self.cfg.llm.history_token_budget and len(rest) > 1:
            rest.pop(0)
        return [system, *rest]

    def _remember(self, user_text: str, spoken: str) -> None:
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": spoken or "OK."})
        while len(self.history) > self.cfg.llm.history_messages:
            self.history.pop(0)

    def _choose_model(self) -> str:
        cap = self.cfg.budget.daily_soft_cap_usd
        if not cap:
            return self.cfg.llm.model
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        spent = self.store.spend_since(day + "T00:00:00.000Z")
        if spent < cap:
            return self.cfg.llm.model
        if self._degraded_day != day:
            self._degraded_day = day
            self.events.emit(
                "budget.degraded",
                level="warning",
                period="day",
                spent_usd=round(spent, 4),
                model=self.cfg.llm.fallback_model,
            )
        return self.cfg.llm.fallback_model

    def _speak_stream(self, text: str, speak: Callable[[str], bool], spoken_parts: list[str]) -> bool:
        ready, rest = take_sentences(text)
        ready.extend(flush_sentences(rest))
        for sentence in ready:
            spoken_parts.append(sentence)
            if speak(sentence):
                return True
        return False

    def _emit_llm(self, result: LlmResult, round_number: int) -> None:
        fields: dict[str, Any] = {
            "model": result.model,
            "round": round_number,
            "stream": True,
            "status": result.status if result.status != "error" else "error",
            "http_status": result.http_status,
            "xai_response_id": result.xai_response_id,
            "system_fingerprint": result.system_fingerprint,
            "finish_reason": result.finish_reason,
            "ttft_ms": result.ttft_ms,
            "duration_ms": result.duration_ms,
            "retry_count": result.retry_count,
            "tool_calls_requested": [call.name for call in result.tool_calls],
            "usage_missing": result.usage_missing,
            "cost_usd": result.cost_usd,
            "cost_estimated": result.cost_estimated,
        }
        if result.usage is not None:
            fields["usage"] = result.usage
        self.events.emit("llm.completed", level="info" if result.status == "ok" else "error", **fields)

    def _emit_tool(self, outcome) -> None:
        if outcome.rejected:
            self.events.emit(
                "tool.rejected",
                level="warning",
                tool_name=outcome.name,
                tool_call_id=outcome.call_id,
                status="rejected",
                duration_ms=outcome.duration_ms,
                error_code=outcome.error_code or "tool.unknown",
            )
            return
        level = "info" if outcome.status == "ok" else "warning"
        self.events.emit(
            "tool.completed",
            level=level,
            tool_name=outcome.name,
            tool_call_id=outcome.call_id,
            args_redacted=outcome.args_redacted,
            status=outcome.status,
            error_code=outcome.error_code,
            duration_ms=outcome.duration_ms,
            result_bytes=len(json.dumps(outcome.result)),
        )

    def _emit_error(self, result: LlmResult, fallback_spoken: bool) -> None:
        self.events.emit(
            "error",
            level="critical" if result.error_code in {"llm.unauthorised", "llm.forbidden", "llm.model_not_found"} else "error",
            category=result.error_category or "internal",
            error_code=result.error_code or "internal.unhandled",
            message=(result.error_message or "language model failed")[:300],
            stage="llm",
            retryable=result.retryable,
            retry_count=result.retry_count,
            fallback_spoken=fallback_spoken,
        )


def render_system(cfg: Config) -> str:
    path = prompt_path(cfg.llm.system_prompt_version)
    text = path.read_text(encoding="utf-8") if path.is_file() else "You are Milo, a voice assistant."
    try:
        zone = ZoneInfo(cfg.device.timezone)
    except Exception:
        zone = ZoneInfo("UTC")
    now = datetime.now(zone)
    stamp = now.strftime("%A ") + f"{now.day} " + now.strftime("%B %Y, %H:%M")
    rendered = text.replace("{{now_local}}", stamp).replace("{{default_location}}", cfg.device.default_location)
    from gva.runtime.bridge import CALENDAR_BULLET, WORK_BULLET, bridge_configured

    if bridge_configured(cfg) and CALENDAR_BULLET in rendered:
        rendered = rendered.replace(CALENDAR_BULLET, WORK_BULLET)
    return rendered


def _prepare(text: str, cfg: Config, pepper: str) -> dict[str, Any]:
    kwargs = {
        "household_names": cfg.privacy.household_names,
        "categories": cfg.privacy.sensitive_categories,
    }
    local = redact(text, short_numbers=cfg.privacy.redact_short_numbers, **kwargs)
    strict = redact(text, short_numbers=True, **kwargs)
    digest = prompt_hash(strict.text, pepper) if len(pepper) >= 32 else prompt_hash(strict.text, "0" * 32)
    shown = f"[SENSITIVE: {local.category}]" if local.sensitive else local.text
    return {
        "shown": shown[:2000],
        "hash": digest,
        "sensitive": local.sensitive,
        "category": local.category,
        "chars": len(text),
    }


def _prompt_fields(prepared: dict[str, Any]) -> dict[str, Any]:
    return {
        "sensitive": prepared["sensitive"],
        "sensitive_category": prepared["category"],
        "prompt_redacted": prepared["shown"],
        "prompt_hash": prepared["hash"],
        "prompt_chars": prepared["chars"],
    }


def _fallback_phrase(category: str | None) -> str:
    return {
        "llm_rate_limit": "I'm a bit busy, try again in a moment.",
        "llm_server": "I can't reach my brain right now.",
        "llm_client": "Something's wrong with my settings.",
        "network": "I'm offline at the moment.",
        "audio": "I'm having trouble hearing.",
        "stt": "Sorry, I didn't catch that.",
        "tts": "Sorry, something went wrong.",
    }.get(category or "", "Sorry, something went wrong.")


def _assistant_tool_message(result: LlmResult) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": result.text or None,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments or "{}"},
            }
            for call in result.tool_calls
        ],
    }


def _estimate(messages: list[dict[str, Any]]) -> int:
    return sum(len(str(message.get("content") or "")) for message in messages) // 4


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
