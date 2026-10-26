"""Speech-to-speech session on xAI's realtime voice API.

This is the conversation that already worked on the Pi: openWakeWord hears
"hey jarvis", then a WebSocket to grok-voice-latest speaks and listens.
The local whisper/Piper pipeline remains available as conversation.engine local.

Wake behaviour that stopped the false re-trigger, kept as it was:
a fresh openWakeWord model after every session, two detections in a row,
and a few seconds of cooldown. The microphone queue is also drained.

Playback follows the same idea as Naka (an asyncio-side queue so socket reads
are not tied to the sound device) and snowman (reset the wake decoder for each
listening period, and send response.cancel when the wake phrase barges in).
The echo path is the small single-delay canceller in aec.py, which stays light
on a Pi 4. Those projects were used for the shape of the loop, not their source.
"""

from __future__ import annotations

import asyncio
import base64
import json
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from gva.config import Config, env_value
from gva.ids import new_ulid, utc_now_ms
from gva.runtime.audio import FALLBACK_PHRASE, play_fallback_phrase, resample_pcm
from gva.runtime.bridge import (
    DEFER_LINE,
    NOT_YET_LINE,
    UNREACHABLE_LINE,
    WAITING_LINE,
    is_defer_phrase,
    is_suggestion_decline,
)
from gva.runtime.conversation import render_system
from gva.runtime.textutil import is_stop_phrase
from gva.telemetry.events import EventLog
from gva.telemetry.redact import prompt_hash, redact
from gva.telemetry.store import Store


def audio_cost(seconds: float, usd_per_minute: float) -> float:
    return round(max(0.0, float(seconds)) / 60.0 * float(usd_per_minute), 6)


def mentions_stop(text: str, phrases: list[str]) -> bool:
    """Whole-word match, including the phrases the working script already used."""
    if is_stop_phrase(text, phrases):
        return True
    folded = " ".join(text.casefold().replace(".", " ").replace("!", " ").replace("?", " ").split())
    words = set(folded.split())
    for phrase in phrases:
        cleaned = phrase.casefold().strip()
        if not cleaned:
            continue
        parts = cleaned.split()
        if len(parts) == 1 and parts[0] in words:
            return True
        if len(parts) > 1 and cleaned in folded:
            return True
    return False


@dataclass
class RealtimeTurn:
    user_text: str = ""
    assistant_text: str = ""
    tools: list[str] = field(default_factory=list)
    response_id: str | None = None


class RealtimeProtocol:
    """Turns socket events into playback, tool calls, and telemetry. No network of its own."""

    def __init__(self, cfg: Config, events: EventLog, store: Store, tools, sink):
        self.cfg = cfg
        self.events = events
        self.store = store
        self.tools = tools
        self.sink = sink
        self.close = False
        self.speaking = False
        self.billed_seconds = 0.0
        self.reported_cost = False
        self.text_messages = 0
        self.last_activity = time.monotonic()
        self.last_socket = time.monotonic()
        # Decouples socket reads from the output device. The player task drains it.
        self.playback: queue.Queue[bytes] = queue.Queue(maxsize=256)
        self._user_text = ""
        self._assistant: list[str] = []
        self._tools: list[str] = []
        self._pending_tools = 0
        self._turn_open = False
        self._response_id: str | None = None
        self._response_pcm_bytes = 0
        self._stt_bytes = 0
        self._stt_started: float | None = None
        # A tool response often ends with no audio. The spoken reply is the next one.
        self._defer_silence_fallback = False
        self._need_response_create = False
        self._bridge_ack_pending = False
        self._bridge_ack_phrase = ""
        self._defer_close_armed = False
        self._close_after_reply = False
        self._response_instructions: str | None = None
        self._response_active = False
        self._bridge_items: list[dict[str, Any]] = []
        self._bridge_inflight: dict[str, Any] | None = None
        self._bridge_awaiting_audio: dict[str, Any] | None = None
        self._bridge_lock = __import__("threading").Lock()
        self.on_bridge_spoken = None
        self.fallback_player = None
        self.session_id = new_ulid()

    def session_update(self) -> dict[str, Any]:
        voice = self.cfg.conversation.realtime
        tool_defs = []
        for spec in self.tools.specs.values():
            tool_defs.append(
                {
                    "type": "function",
                    "name": spec.name,
                    "description": spec.description.strip(),
                    "parameters": spec.parameters,
                }
            )
        return {
            "type": "session.update",
            "session": {
                "voice": voice.voice,
                "instructions": render_system(self.cfg),
                "turn_detection": {"type": "server_vad", "silence_duration_ms": voice.silence_duration_ms},
                "audio": {
                    "input": {"format": {"type": "audio/pcm", "rate": voice.sample_rate}},
                    "output": {"format": {"type": "audio/pcm", "rate": voice.sample_rate}},
                },
                "tools": tool_defs,
            },
        }

    def audio_message(self, pcm_capture_rate: bytes) -> dict[str, str] | None:
        voice_rate = self.cfg.conversation.realtime.sample_rate
        pcm = resample_pcm(pcm_capture_rate, self.cfg.audio.capture_rate, voice_rate)
        if len(pcm) < 2:
            return None
        if self._stt_started is None:
            self._stt_started = time.monotonic()
        self._stt_bytes += len(pcm)
        return {"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode("ascii")}

    def on_event(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        etype = event.get("type") or ""
        if etype in {
            "input_audio_buffer.speech_started",
            "input_audio_buffer.speech_stopped",
            "input_audio_buffer.committed",
        }:
            self.note_activity()
            return []
        if etype == "response.output_audio.delta" or etype == "response.audio.delta":
            payload = event.get("delta") or ""
            if payload:
                self.enqueue_pcm(base64.b64decode(payload))
            self._response_id = event.get("response_id") or self._response_id
            self.note_activity()
            return []
        if etype in {"response.output_audio_transcript.delta", "response.audio_transcript.delta"}:
            self._assistant.append(event.get("delta") or "")
            self.note_activity()
            return []
        if etype == "conversation.item.input_audio_transcription.completed":
            self.note_activity()
            self._user_text = event.get("transcript") or ""
            self._emit_stt(self._user_text)
            bridge = self._live_bridge()
            pending = (
                bridge is not None
                and callable(getattr(bridge, "answer_pending", None))
                and bridge.answer_pending()
            )
            if pending and is_defer_phrase(self._user_text):
                bridge.defer()
                return self._defer_reply_messages()
            if pending and is_suggestion_decline(self._user_text):
                bridge.decline_suggestions()
                if self._response_active:
                    return [self.cancel_playback()]
                self.speaking = False
                self.drain_playback()
                resume = getattr(self.sink, "resume", None)
                if resume:
                    resume()
                return []
            if mentions_stop(self._user_text, self.cfg.conversation.stop_phrases):
                self.speaking = False
                self.drain_playback()
                self.sink.stop()
                self._finish_turn(event, status="cancelled", intent="stop")
                self.close = True
            return []
        if etype == "response.created":
            self.note_activity()
            self._turn_open = True
            self._response_active = True
            self._response_pcm_bytes = 0
            self._response_id = event.get("response", {}).get("id") or event.get("response_id")
            return []
        if etype == "response.done":
            self.note_activity()
            self.speaking = False
            # response.create while a response is still active is rejected, and the
            # model then says nothing. Send it only after this response has finished.
            heard_ms = _pcm_ms(self._response_pcm_bytes, self.cfg.conversation.realtime.sample_rate)
            owe_reply = self._need_response_create
            self._need_response_create = False
            if owe_reply:
                self._defer_silence_fallback = True
            self._finish_turn(event, status="ok", intent=None)
            self._settle_bridge_audio(heard_ms)
            if owe_reply:
                message: dict[str, Any] = {"type": "response.create"}
                instructions = self._response_instructions
                self._response_instructions = None
                if instructions:
                    message["response"] = {"instructions": instructions}
                    self._response_active = True
                else:
                    self._response_active = False
                return [message]
            self._response_active = False
            if self._close_after_reply:
                self._close_after_reply = False
                self.close = True
            return []
        if etype == "response.function_call_arguments.done":
            self.note_activity()
            return self._tool(event)
        if etype == "error":
            err = event.get("error") or {}
            message = str(err.get("message") or event.get("message") or "voice session error")[:300]
            self.events.emit(
                "error",
                level="error",
                category="llm_server",
                error_code="llm.server_error",
                message=_safe(message),
                stage="realtime",
                retryable=True,
                retry_count=0,
                fallback_spoken=False,
            )
        return []

    def note_text_message(self) -> None:
        """A client conversation.item.create that is a real text turn, not a tool result."""
        self.text_messages += 1

    def note_activity(self) -> None:
        """Conversation moved. Microphone frames must not call this."""
        self.last_activity = time.monotonic()

    def note_socket(self) -> None:
        self.last_socket = time.monotonic()

    def idle(self, now: float, timeout_s: float) -> bool:
        return (now - self.last_activity) > timeout_s

    def stalled(self, now: float, watchdog_s: float) -> bool:
        return (now - self.last_socket) > watchdog_s

    def enqueue_pcm(self, pcm: bytes) -> None:
        if len(pcm) < 2:
            return
        self._response_pcm_bytes += len(pcm) - (len(pcm) % 2)
        self.speaking = True
        try:
            self.playback.put_nowait(pcm)
        except queue.Full:
            try:
                self.playback.get_nowait()
            except queue.Empty:
                pass
            try:
                self.playback.put_nowait(pcm)
            except queue.Full:
                pass

    def drain_playback(self) -> None:
        while True:
            try:
                self.playback.get_nowait()
            except queue.Empty:
                return

    def cancel_playback(self) -> dict[str, str]:
        """Drop queued speech and ask the server to stop the current response."""
        self.speaking = False
        self.drain_playback()
        self.sink.stop()
        resume = getattr(self.sink, "resume", None)
        if resume:
            resume()
        self.note_activity()
        return {"type": "response.cancel"}

    def finish_if_unbilled(self, elapsed_s: float) -> None:
        """server_vad bills the whole session. Estimate only when the API sent no usage."""
        if self.reported_cost or elapsed_s <= 0:
            return
        price = self.cfg.conversation.realtime.audio_usd_per_minute
        cost = audio_cost(elapsed_s, price)
        if self.text_messages:
            cost = round(cost + self.text_messages * self.cfg.conversation.realtime.text_usd_per_message, 6)
        self._emit_llm(cost, estimated=True, duration_ms=int(elapsed_s * 1000), http_status=None)
        self.reported_cost = True

    def _tool(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        name = str(event.get("name") or "")
        call_id = str(event.get("call_id") or new_ulid())
        raw = event.get("arguments") or "{}"
        if isinstance(raw, dict):
            arguments = json.dumps(raw)
        else:
            arguments = str(raw)
        self.events.ctx.session_id = self.session_id
        self.events.emit("tool.started", tool_name=name, tool_call_id=call_id)
        outcome = self.tools.run(call_id, name, arguments)
        self._tools.append(name)
        if name in {"ask_rbd_team", "check_team_answer", "defer_until_answer"}:
            # Questions and answers are personal. Keep them out of the event log.
            outcome.args_redacted = {}
        suppress_reply = False
        if name == "ask_rbd_team":
            if outcome.result.get("status") == "sent":
                phrase = WAITING_LINE
                items = self._waiting_items()
                if items:
                    outcome.result = dict(outcome.result)
                    outcome.result["pending_items"] = items
                    from gva.runtime.bridge import context_waiting_instructions

                    self._response_instructions = context_waiting_instructions()
                else:
                    self._response_instructions = f"Say only: {phrase}"
            else:
                phrase = UNREACHABLE_LINE
                self._response_instructions = f"Say only: {phrase}"
            self._bridge_ack_pending = True
            self._bridge_ack_phrase = phrase
        elif name == "check_team_answer":
            if outcome.result.get("status") == "ready":
                phrase = str(outcome.result.get("answer") or "").strip() or NOT_YET_LINE
                self._response_instructions = (
                    "Read the team's answer out naturally and briefly, in British English. "
                    "Do not add anything that is not in the answer."
                )
            else:
                phrase = NOT_YET_LINE
                self._response_instructions = f"Say only: {phrase}"
            self._bridge_ack_pending = True
            self._bridge_ack_phrase = phrase
        elif name == "defer_until_answer":
            if outcome.result.get("status") == "deferred":
                if self._defer_close_armed:
                    # The confirm line is already queued. Do not say it twice.
                    suppress_reply = True
                else:
                    self._arm_defer_close()
            else:
                phrase = "Nothing is waiting from the team."
                self._response_instructions = f"Say only: {phrase}"
                self._bridge_ack_pending = True
                self._bridge_ack_phrase = phrase
        level = "info" if outcome.status == "ok" else "warning"
        if outcome.rejected:
            self.events.emit(
                "tool.rejected",
                level="warning",
                tool_name=name,
                tool_call_id=call_id,
                status="rejected",
                duration_ms=outcome.duration_ms,
                error_code=outcome.error_code or "tool.unknown",
            )
        else:
            self.events.emit(
                "tool.completed",
                level=level,
                tool_name=name,
                tool_call_id=call_id,
                args_redacted=outcome.args_redacted,
                status=outcome.status,
                error_code=outcome.error_code,
                duration_ms=outcome.duration_ms,
                result_bytes=len(json.dumps(outcome.result)),
            )
        # Hand the result back now. response.create waits for response.done so the
        # server is not asked to speak while this response is still open.
        if not suppress_reply:
            self._need_response_create = True
        return [
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(outcome.result)[:4096],
                },
            },
        ]

    def _finish_turn(self, event: dict[str, Any], status: str, intent: str | None) -> None:
        rate = self.cfg.conversation.realtime.sample_rate
        audio_ms = _pcm_ms(self._response_pcm_bytes, rate)
        defer_fallback = self._defer_silence_fallback
        self._defer_silence_fallback = False
        ack_pending = self._bridge_ack_pending
        ack_phrase = self._bridge_ack_phrase
        if (
            self._turn_open is False
            and not self._user_text
            and not self._assistant
            and status == "ok"
            and audio_ms == 0
        ):
            usage = _usage(event)
            self._account(usage, event)
            return
        fallback = False
        if status == "ok" and audio_ms == 0 and not defer_fallback:
            if ack_pending and ack_phrase:
                self._bridge_ack_pending = False
                extra = self._speak_phrase(ack_phrase)
            else:
                extra = self._speak_fallback()
            if extra:
                fallback = True
                audio_ms += extra
        elif audio_ms > 0:
            self._bridge_ack_pending = False
        usage = _usage(event)
        cost = self._account(usage, event)
        user = self._user_text
        assistant = "".join(self._assistant).strip()
        shown_user, user_hash, sensitive, category = _redacted(user, self.cfg)
        shown_reply, reply_hash, _, _ = _redacted(assistant, self.cfg)
        turn_id = new_ulid()
        self.events.ctx.session_id = self.session_id
        self.events.ctx.turn_id = turn_id
        self.events.ctx.request_id = self._response_id or new_ulid()
        fields = {
            "status": status,
            "intent": intent or ("stop" if mentions_stop(user, self.cfg.conversation.stop_phrases) else _intent(self._tools)),
            "prompt_redacted": shown_user,
            "prompt_hash": user_hash,
            "prompt_chars": len(user),
            "response_redacted": shown_reply,
            "sensitive": sensitive,
            "sensitive_category": category,
            "hash_epoch": self.cfg.privacy.hash_epoch,
            "tools_called": list(self._tools),
            "rounds": 1 if self._tools or assistant or user else 0,
            "stage_timings": {"llm_total_ms": 0},
            "e2e_first_audio_ms": None,
            "total_ms": 0,
            "tokens_total": 0,
            "cost_usd": cost,
            "fallback_spoken": fallback,
            "started_at": utc_now_ms(),
            "model": self.cfg.conversation.realtime.model,
        }
        if reply_hash:
            fields["response_hash"] = reply_hash
        self.events.emit("turn.completed", level="info" if status == "ok" else "warning", **fields)
        self.events.emit(
            "playback.completed",
            audio_ms=audio_ms,
            underruns=self._take_underruns(),
            interrupted=status == "cancelled",
        )
        self._response_pcm_bytes = 0
        self._user_text = ""
        self._assistant = []
        self._tools = []
        self._turn_open = False
        self.events.ctx.turn_id = None

    def bridge_blocked(self) -> bool:
        """True while a response is open. A late answer waits, then speaks."""
        return bool(self._response_active or self._need_response_create)

    def playback_pending(self) -> bool:
        """True until the current reply has finished coming out of the speaker."""
        if self.bridge_blocked() or self.speaking:
            return True
        try:
            if not self.playback.empty():
                return True
        except Exception:
            return True
        queued = getattr(self.sink, "queued_frames", None)
        if callable(queued):
            try:
                if int(queued()) > 0:
                    return True
            except Exception:
                return False
        return False

    def _live_bridge(self):
        tools = self.tools
        if tools is None:
            return None
        return getattr(getattr(tools, "ctx", None), "bridge", None)

    def _arm_defer_close(self) -> None:
        self._defer_close_armed = True
        self._close_after_reply = True
        self._response_instructions = f"Say only: {DEFER_LINE}"
        self._bridge_ack_pending = True
        self._bridge_ack_phrase = DEFER_LINE

    def _defer_reply_messages(self) -> list[dict[str, Any]]:
        """One confirm line, then the session ends. A second request does not speak again."""
        if self._defer_close_armed:
            return []
        self._arm_defer_close()
        if self._response_active:
            self._need_response_create = True
            return [self.cancel_playback()]
        if self._need_response_create:
            return []
        self._response_instructions = None
        return [{"type": "response.create", "response": {"instructions": f"Say only: {DEFER_LINE}"}}]

    def _waiting_items(self) -> list[dict[str, Any]]:
        if not self.cfg.bot_bridge.chat_while_waiting or not self.cfg.bot_bridge.suggest_pending_items:
            return []
        bridge = self._live_bridge()
        if bridge is None or not hasattr(bridge, "waiting_context"):
            return []
        if hasattr(bridge, "suggestions_allowed") and not bridge.suggestions_allowed():
            return []
        try:
            items = list(bridge.waiting_context() or [])
        except Exception:
            return []
        items = items[:2]
        if items and hasattr(bridge, "note_suggestions_offered"):
            bridge.note_suggestions_offered()
        return items

    def enqueue_bridge(
        self,
        messages: list[dict[str, Any]],
        text: str,
        request_id: str = "",
        kind: str = "answer",
        chime: bool = True,
    ) -> None:
        with self._bridge_lock:
            self._bridge_items.append(
                {
                    "messages": messages,
                    "text": text,
                    "request_id": request_id,
                    "kind": kind,
                    "chime": chime,
                    "sent": False,
                    "confirmed": False,
                }
            )

    def pop_bridge_ready(self) -> dict[str, Any] | None:
        with self._bridge_lock:
            if self._bridge_inflight is not None or self.playback_pending() or not self._bridge_items:
                return None
            self._bridge_inflight = self._bridge_items.pop(0)
            return self._bridge_inflight

    def finish_bridge_send(self, ok: bool) -> None:
        with self._bridge_lock:
            item = self._bridge_inflight
            self._bridge_inflight = None
            if item is None:
                return
            if ok:
                item["sent"] = True
                self._response_active = True
                if item.get("request_id"):
                    self._bridge_awaiting_audio = item
            else:
                self._bridge_items.insert(0, item)

    def drain_unsent(self) -> list[dict[str, str]]:
        with self._bridge_lock:
            pending: list[dict[str, Any]] = []
            if self._bridge_awaiting_audio and not self._bridge_awaiting_audio.get("confirmed"):
                pending.append(self._bridge_awaiting_audio)
                self._bridge_awaiting_audio = None
            if self._bridge_inflight and not self._bridge_inflight.get("confirmed"):
                pending.append(self._bridge_inflight)
                self._bridge_inflight = None
            for item in self._bridge_items:
                if not item.get("confirmed"):
                    pending.append(item)
            self._bridge_items.clear()
            return [
                {
                    "request_id": str(item.get("request_id") or ""),
                    "text": str(item.get("text") or ""),
                    "kind": str(item.get("kind") or "answer"),
                    "chime": bool(item.get("chime", True)),
                }
                for item in pending
                if item.get("text")
            ]

    def _settle_bridge_audio(self, audio_ms: int) -> None:
        """The readout either produced audio, or it has to be said another way."""
        with self._bridge_lock:
            item = self._bridge_awaiting_audio
            if item is None:
                return
            self._bridge_awaiting_audio = None
            if audio_ms > 0:
                item["confirmed"] = True
            else:
                item["sent"] = False
        hook = self.on_bridge_spoken
        if hook is None:
            if audio_ms <= 0:
                with self._bridge_lock:
                    self._bridge_items.insert(0, item)
            return
        try:
            asyncio.get_running_loop()
            in_loop = True
        except RuntimeError:
            in_loop = False
        if in_loop:
            threading.Thread(
                target=hook, args=(item, audio_ms > 0), name="gva-bridge-speak", daemon=True
            ).start()
            return
        hook(item, audio_ms > 0)

    def _speak_phrase(self, phrase: str) -> int:
        """Speak one short line. A test player wins. Otherwise try Piper."""
        player = self.fallback_player
        if player is not None:
            try:
                result = player(self.sink, self.cfg, phrase)
            except Exception:
                return 0
            if isinstance(result, bool):
                return 1 if result else 0
            try:
                return max(0, int(result))
            except (TypeError, ValueError):
                return 0
        try:
            from gva.runtime.tts import PiperTts

            pcm, rate = PiperTts(self.cfg).synthesise(phrase)
        except Exception:
            return 0
        if not pcm:
            return 0
        target = int(getattr(self.sink, "sample_rate", 0) or self.cfg.conversation.realtime.sample_rate or rate)
        if rate != target:
            pcm = resample_pcm(pcm, rate, target)
            rate = target
        try:
            self.sink.write(pcm)
        except Exception:
            return 0
        return int(len(pcm) / 2 / max(rate, 1) * 1000)

    def _speak_fallback(self) -> int:
        player = self.fallback_player or play_fallback_phrase
        try:
            result = player(self.sink, self.cfg, FALLBACK_PHRASE)
        except Exception:
            return 0
        if isinstance(result, bool):
            return 1 if result else 0
        try:
            return max(0, int(result))
        except (TypeError, ValueError):
            return 0

    def _take_underruns(self) -> int:
        consume = getattr(self.sink, "consume_underruns", None)
        if callable(consume):
            try:
                return max(0, int(consume()))
            except Exception:
                return 0
        try:
            return max(0, int(getattr(self.sink, "underruns", 0) or 0))
        except (TypeError, ValueError):
            return 0

    def _account(self, usage: dict[str, Any] | None, event: dict[str, Any]) -> float | None:
        price = self.cfg.conversation.realtime.audio_usd_per_minute
        if usage and usage.get("billable_audio_seconds") is not None:
            total = float(usage["billable_audio_seconds"])
            delta = max(0.0, total - self.billed_seconds)
            self.billed_seconds = total
            cost = audio_cost(delta, price)
            self._emit_llm(cost, estimated=False, duration_ms=int(delta * 1000), http_status=200)
            self.reported_cost = True
            return cost
        return None

    def _emit_llm(self, cost: float | None, estimated: bool, duration_ms: int, http_status: int | None) -> None:
        self.events.ctx.session_id = self.session_id
        if not self.events.ctx.request_id:
            self.events.ctx.request_id = new_ulid()
        self.events.emit(
            "llm.completed",
            model=self.cfg.conversation.realtime.model,
            round=1,
            stream=True,
            status="ok",
            http_status=http_status,
            duration_ms=max(0, duration_ms),
            retry_count=0,
            tool_calls_requested=list(self._tools),
            usage_missing=True,
            cost_usd=cost,
            cost_estimated=estimated,
        )

    def _emit_stt(self, text: str) -> None:
        shown, digest, _, _ = _redacted(text, self.cfg)
        rate = self.cfg.conversation.realtime.sample_rate
        heard = _pcm_ms(self._stt_bytes, rate)
        if self._stt_started is None:
            elapsed = 0
        else:
            elapsed = max(0, int((time.monotonic() - self._stt_started) * 1000))
        self._stt_bytes = 0
        self._stt_started = None
        self.events.ctx.session_id = self.session_id
        self.events.emit(
            "stt.completed",
            engine="xai",
            model=self.cfg.conversation.realtime.model,
            audio_ms=heard,
            duration_ms=elapsed,
            confidence=None,
            language="en",
            prompt_redacted=shown,
            prompt_hash=digest,
            rejected_reason=None,
        )


def _pcm_ms(nbytes: int, rate: int) -> int:
    if rate <= 0 or nbytes < 2:
        return 0
    return int(nbytes / 2 / rate * 1000)


def _usage(event: dict[str, Any]) -> dict[str, Any] | None:
    response = event.get("response") or {}
    usage = event.get("usage") or response.get("usage")
    return usage if isinstance(usage, dict) else None


def _redacted(text: str, cfg: Config) -> tuple[str | None, str, bool, str | None]:
    pepper = env_value("GVA_HASH_PEPPER") or ("0" * 32)
    if len(pepper) < 32:
        pepper = "0" * 32
    kwargs = {
        "household_names": cfg.privacy.household_names,
        "categories": cfg.privacy.sensitive_categories,
    }
    local = redact(text or "", short_numbers=cfg.privacy.redact_short_numbers, **kwargs)
    strict = redact(text or "", short_numbers=True, **kwargs)
    shown = f"[SENSITIVE: {local.category}]" if local.sensitive else local.text
    shown = shown[:2000] if shown else None
    return shown, prompt_hash(strict.text, pepper), local.sensitive, local.category


def _intent(tools: list[str]) -> str:
    if not tools:
        return "chat"
    for key in ("timer", "alarm", "weather", "device", "calendar", "time"):
        if key in tools[0]:
            return key
    return tools[0]


def _safe(text: str) -> str:
    if "Bearer" in text or "xai-" in text:
        return "voice session error"
    return text[:300]


class RealtimeAssistant:
    def __init__(self, cfg: Config, store: Store, events: EventLog, capture, sink, wake, tools, connector=None, bridge=None):
        self.cfg = cfg
        self.store = store
        self.events = events
        self.capture = capture
        self.sink = sink
        self.wake = wake
        self.tools = tools
        self.connector = connector
        self.bridge = bridge
        self._bridge_leftover: list[Any] = []
        self._follow_up = threading.Event()
        self._silence_override_s: float | None = None
        self._session_silence_s: float | None = None
        if bridge is not None:
            bridge.on_follow_up = self._follow_up.set
        self.shutdown = __import__("threading").Event()
        self._last_heartbeat = time.monotonic()
        self.sessions = 0

    def run(self, max_sessions: int | None = None) -> None:
        self.capture.start()
        self.events.emit(
            "app.started",
            model=self.cfg.conversation.realtime.model,
            stt_engine="xai",
            tts_engine="xai",
            wake_model=getattr(self.wake, "model_name", self.cfg.wake.model),
            wake_phrase=self.cfg.wake.phrase,
            conversation_engine="realtime",
        )
        try:
            while not self.shutdown.is_set():
                self._heartbeat()
                if not self._wait_for_wake():
                    break
                started = time.monotonic()
                try:
                    asyncio.run(self._session())
                except Exception as exc:
                    self.events.emit(
                        "error",
                        level="error",
                        category="network",
                        error_code="net.offline",
                        message=_safe(exc.__class__.__name__),
                        stage="realtime",
                        retryable=True,
                        retry_count=0,
                        fallback_spoken=False,
                    )
                self._speak_bridge_leftover()
                self.sessions += 1
                self._after_session(time.monotonic() - started)
                if max_sessions is not None and self.sessions >= max_sessions:
                    break
        finally:
            self.events.emit("app.stopped")
            self.capture.stop()
            close = getattr(self.sink, "close", None)
            if close:
                close()

    def _speak_bridge_leftover(self) -> None:
        leftover = self._bridge_leftover
        self._bridge_leftover = []
        if self.bridge is None:
            return
        for item in leftover:
            if isinstance(item, str):
                self.bridge.speak_unconfirmed("", item)
            else:
                self.bridge.speak_unconfirmed(
                    item.get("request_id") or "",
                    item.get("text") or "",
                    chime=bool(item.get("chime", True)),
                    kind=str(item.get("kind") or "answer"),
                )

    def _wait_for_wake(self) -> bool:
        while not self.shutdown.is_set():
            if self._follow_up.is_set():
                self._follow_up.clear()
                seconds = getattr(self.cfg.bot_bridge, "nudge_follow_up_s", None)
                self._silence_override_s = float(seconds if seconds else 8)
                return True
            frame = self.capture.read(0.2)
            if frame is None:
                return False
            if frame == b"":
                continue
            pcm = resample_pcm(frame, self.cfg.audio.capture_rate, self.cfg.audio.sample_rate)
            score = self.wake.score(pcm)
            if self.wake.observe(score, time.monotonic()):
                self.events.emit(
                    "wake.detected",
                    wake_model=getattr(self.wake, "model_name", self.cfg.wake.model),
                    score=round(float(score), 4),
                    false_wake=False,
                )
                return True
        return False

    async def _session(self) -> None:
        key = env_value("XAI_API_KEY")
        if not key:
            raise RuntimeError("XAI_API_KEY is not set")
        # Fresh decoder for this listening period, so the wake that opened the
        # session does not immediately cancel the reply.
        reset = getattr(self.wake, "reset", None)
        if reset:
            reset()
        flush = getattr(self.capture, "flush", None)
        if flush:
            flush()
        voice = self.cfg.conversation.realtime
        url = voice.url.rstrip("/") + f"?model={voice.model}"
        self._session_silence_s = self._silence_override_s
        self._silence_override_s = None
        protocol = RealtimeProtocol(self.cfg, self.events, self.store, self.tools, self.sink)
        self.events.ctx.session_id = protocol.session_id
        if self.bridge is not None:
            self.bridge.attach(protocol)
            self.bridge.recall(deliver=True)
        try:
            aec = None
            if self.cfg.audio.aec_enabled:
                from gva.runtime.aec import AdaptiveEchoCanceller

                aec = AdaptiveEchoCanceller(self.cfg.audio.aec_delay_ms, self.cfg.audio.sample_rate)
            connect = self.connector or _default_connect
            started = time.monotonic()
            protocol.last_activity = started
            protocol.last_socket = started
            socket = connect(url, additional_headers={"Authorization": f"Bearer {key}"})
            async with socket as ws:
                await self._run_socket(ws, protocol, aec, started)
        finally:
            self._session_silence_s = None
            protocol.close = True
            if self.bridge is not None:
                # Speak outside this event loop so a closed session can open a short one.
                self._bridge_leftover = self.bridge.take_unsent(protocol)

    async def _run_socket(self, ws, protocol: RealtimeProtocol, aec, started: float) -> None:
        await ws.send(json.dumps(protocol.session_update()))
        player = asyncio.create_task(self._play_queued(protocol, aec))
        sender = asyncio.create_task(self._send_audio(ws, protocol, aec))
        watchdog = asyncio.create_task(self._watchdog(ws, protocol))
        bridge_tx = asyncio.create_task(self._send_bridge(ws, protocol))
        try:
            async for raw in ws:
                protocol.note_socket()
                if protocol.close:
                    break
                if isinstance(raw, bytes):
                    continue
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                for message in protocol.on_event(event):
                    if message.get("item", {}).get("type") == "message":
                        protocol.note_text_message()
                    await ws.send(json.dumps(message))
                if protocol.close:
                    break
        finally:
            protocol.close = True
            for task in (sender, player, watchdog, bridge_tx):
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            protocol.finish_if_unbilled(time.monotonic() - started)
            await _close_socket(ws)

    async def _send_bridge(self, ws, protocol: RealtimeProtocol) -> None:
        """Send a team's answer once the current response has finished."""
        while not protocol.close:
            item = protocol.pop_bridge_ready()
            if item is None:
                await asyncio.sleep(0.05)
                continue
            try:
                if protocol.close:
                    protocol.finish_bridge_send(False)
                    return
                for message in item["messages"]:
                    await ws.send(json.dumps(message))
                protocol.finish_bridge_send(True)
            except asyncio.CancelledError:
                protocol.finish_bridge_send(False)
                raise
            except Exception:
                protocol.finish_bridge_send(False)
                await asyncio.sleep(0.2)

    async def _play_queued(self, protocol: RealtimeProtocol, aec) -> None:
        """Drain the playback queue on its own task so a slow device cannot stall the socket."""
        voice_rate = self.cfg.conversation.realtime.sample_rate
        while not protocol.close:
            try:
                pcm = protocol.playback.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.01)
                continue
            protocol.sink.write(pcm)
            if aec is not None:
                aec.push_reference(resample_pcm(pcm, voice_rate, self.cfg.audio.sample_rate))

    async def _watchdog(self, ws, protocol: RealtimeProtocol) -> None:
        limit = self.cfg.conversation.realtime.watchdog_s
        while not protocol.close:
            await asyncio.sleep(0.5)
            if protocol.stalled(time.monotonic(), limit):
                protocol.close = True
                self.events.emit(
                    "error",
                    level="error",
                    category="network",
                    error_code="net.offline",
                    message="realtime socket stalled",
                    stage="realtime",
                    retryable=True,
                    retry_count=0,
                    fallback_spoken=False,
                )
                await _close_socket(ws)
                return

    async def _send_audio(self, ws, protocol: RealtimeProtocol, aec) -> None:
        voice = self.cfg.conversation.realtime
        while not protocol.close and not self.shutdown.is_set():
            # Idle follows server conversation events. A live microphone must not refresh it.
            silence_s = self._session_silence_s if self._session_silence_s is not None else voice.silence_timeout_s
            if idle_should_close(protocol, self.bridge, time.monotonic(), silence_s):
                protocol.close = True
                await _close_socket(ws)
                return
            frame = await asyncio.to_thread(self.capture.read, 0.05)
            if frame is None:
                protocol.close = True
                await _close_socket(ws)
                return
            if not frame:
                continue
            if protocol.speaking and self.cfg.conversation.barge_in != "off":
                heard = resample_pcm(frame, self.cfg.audio.capture_rate, self.cfg.audio.sample_rate)
                if aec is not None:
                    heard = aec.process(heard)
                score = self.wake.score(heard)
                if self.wake.observe(score, time.monotonic()):
                    self.events.emit(
                        "wake.detected",
                        wake_model=getattr(self.wake, "model_name", self.cfg.wake.model),
                        score=round(float(score), 4),
                        false_wake=False,
                    )
                    await ws.send(json.dumps(protocol.cancel_playback()))
                    reset = getattr(self.wake, "reset", None)
                    if reset:
                        reset()
            message = protocol.audio_message(frame)
            if message:
                await ws.send(json.dumps(message))

    def _after_session(self, _elapsed: float) -> None:
        reset = getattr(self.wake, "reset", None)
        if reset:
            reset()
        flush = getattr(self.capture, "flush", None)
        if flush:
            flush()
        deadline = time.monotonic() + self.cfg.wake.cooldown_seconds
        while time.monotonic() < deadline and not self.shutdown.is_set():
            frame = self.capture.read(0.2)
            if frame is None:
                break
        if flush:
            flush()

    def _heartbeat(self) -> None:
        now = time.monotonic()
        if now - self._last_heartbeat < self.cfg.telemetry.heartbeat_seconds:
            return
        self._last_heartbeat = now
        from pathlib import Path

        from gva.telemetry.system import snapshot

        metrics = snapshot(Path(self.cfg.telemetry.log_dir))
        self.events.emit(
            "heartbeat",
            cpu_pct=float(metrics["cpu_pct"] or 0),
            mem_pct=float(metrics["mem_pct"] or 0),
            disk_pct=float(metrics["disk_pct"] or 0),
            temp_c=metrics["temp_c"],
            throttled=bool(metrics["throttled"]) if metrics["throttled"] is not None else False,
            queue_depths={"capture": 0},
            api_reachable=True,
        )


def idle_should_close(protocol, bridge, now: float, silence_s: float) -> bool:
    """Silence ends a call, except while a team answer is still on its way."""
    if bridge is not None and bridge.holds_session():
        return False
    return protocol.idle(now, silence_s)


async def _close_socket(ws) -> None:
    closer = getattr(ws, "close", None)
    if closer is None:
        return
    result = closer()
    if asyncio.iscoroutine(result):
        await result


def _default_connect(url: str, additional_headers: dict[str, str] | None = None):
    import websockets

    try:
        return websockets.connect(url, additional_headers=additional_headers)
    except TypeError:
        return websockets.connect(url, extra_headers=additional_headers)
