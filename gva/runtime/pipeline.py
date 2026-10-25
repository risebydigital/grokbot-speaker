"""Always-on voice loop. Capture never calls Grok; playback can be interrupted."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from gva.config import Config, env_value
from gva.ids import new_ulid, utc_now_ms
from gva.runtime.aec import AdaptiveEchoCanceller
from gva.runtime.audio import Ring, chunk_bytes, tone
from gva.runtime.conversation import Conversation, TurnResult
from gva.runtime.textutil import is_stop_phrase
from gva.runtime.vad import Endpointer
from gva.telemetry.events import EventLog
from gva.telemetry.redact import prompt_hash, redact
from gva.telemetry.store import Store
from gva.telemetry.system import snapshot


@dataclass
class PlayResult:
    interrupted: bool
    reason: str | None
    barge_text: str | None
    audio_ms: int


class VoiceAssistant:
    def __init__(self, cfg: Config, store: Store, events: EventLog, capture, sink, wake, vad, stt, tts, grok, tools):
        self.cfg = cfg
        self.store = store
        self.events = events
        self.capture = capture
        self.sink = sink
        self.wake = wake
        self.vad = vad
        self.stt = stt
        self.tts = tts
        self.grok = grok
        self.tools = tools
        self.conversation = Conversation(cfg, grok, tools, events, store)
        self.aec = AdaptiveEchoCanceller(cfg.audio.aec_delay_ms, cfg.audio.sample_rate) if cfg.audio.aec_enabled else None
        preroll_frames = max(1, int(cfg.wake.pre_roll_ms / cfg.audio.chunk_ms))
        self.recent = Ring(preroll_frames + 2)
        self.state = "idle"
        self.shutdown = __import__("threading").Event()
        self.session_id: str | None = None
        self.session_touched = 0.0
        self.follow_up_until = 0.0
        self.api_reachable: bool | None = None
        self.last_api_ok_at: str | None = None
        self._last_heartbeat = 0.0
        self._finite = False
        self._pending_preroll = b""
        self._queued_text: str | None = None
        self._last_wake_score = 0.0

    def run(self) -> None:
        self.capture.start()
        self.events.emit(
            "app.started",
            config_hash=self.cfg.device.id,
            model=self.cfg.llm.model,
            stt_engine=self.cfg.stt.engine,
            tts_engine=self.cfg.tts.engine,
            wake_model=getattr(self.wake, "model_name", self.cfg.wake.model),
            wake_phrase=self.cfg.wake.phrase,
        )
        try:
            while not self.shutdown.is_set():
                self._service()
                frame = self.capture.read(0.2)
                if frame is None:
                    break
                if frame == b"":
                    if self._finite:
                        break
                    continue
                self._on_idle_frame(frame)
        finally:
            self.events.emit("app.stopped")
            self.capture.stop()

    def run_frames(self, frames: list[bytes]) -> None:
        from gva.runtime.audio import ListCapture

        self.capture = ListCapture(frames)
        self._finite = True
        self.run()

    def ask(self, text: str) -> TurnResult:
        """Typed prompt. Same tool loop as a spoken turn, without the microphone."""
        self._ensure_session()
        return self._run_text(text, timings={"endpoint_ms": 0, "stt_ms": 0}, end_of_speech=time.monotonic())

    def say(self, text: str) -> None:
        self._speak_text(text)

    def mute(self, muted: bool) -> None:
        path = self._mute_path()
        if muted:
            path.write_text("muted\n", encoding="utf-8")
            self._set_state("muted")
        else:
            path.unlink(missing_ok=True)
            self._set_state("idle")

    def _on_idle_frame(self, frame: bytes) -> None:
        if self._muted():
            self._set_state("muted")
            return
        self.recent.push(frame)
        now = time.monotonic()
        if self.follow_up_until and now <= self.follow_up_until:
            if self.vad.is_speech(frame):
                self._capture(frame, woke=False, wake_score=0.0, detected_at=utc_now_ms())
                return
            if now > self.follow_up_until:
                self.follow_up_until = 0.0
            return
        if self.follow_up_until and now > self.follow_up_until:
            self.follow_up_until = 0.0
            self._set_state("idle")
        if self.session_id and now - self.session_touched > self.cfg.conversation.session_idle_minutes * 60:
            self.session_id = None
            self.conversation.reset()
        score = self.wake.score(frame)
        self._last_wake_score = score
        if self.cfg.wake.vad_gate and not self.vad.is_speech(frame) and score < self.cfg.wake.threshold:
            return
        if self.wake.observe(score, now):
            detected = utc_now_ms()
            self._capture(frame, woke=True, wake_score=score, detected_at=detected)

    def _capture(self, first: bytes, *, woke: bool, wake_score: float, detected_at: str) -> None:
        self._set_state("listening")
        if woke:
            self._play_earcon()
        preroll_count = max(1, int(self.cfg.wake.pre_roll_ms / self.cfg.audio.chunk_ms))
        preroll = self._pending_preroll or self.recent.tail(preroll_count)
        self._pending_preroll = b""
        endpoint = Endpointer(
            self.vad,
            self.cfg.vad.end_silence_ms,
            self.cfg.vad.max_utterance_seconds,
            self.cfg.audio.chunk_ms,
        )
        audio = bytearray()
        if preroll and not preroll.endswith(first):
            audio.extend(preroll)
        audio.extend(first)
        endpoint.add(first)
        started = time.monotonic()
        while not endpoint.closed and not self.shutdown.is_set():
            frame = self.capture.read(0.5)
            if frame is None or frame == b"":
                break
            audio.extend(frame)
            self.recent.push(frame)
            endpoint.add(frame)
        endpoint_ms = int((time.monotonic() - started) * 1000)
        if woke:
            self.events.emit(
                "wake.detected",
                ts=detected_at,
                wake_model=getattr(self.wake, "model_name", self.cfg.wake.model),
                score=round(float(wake_score), 4),
                false_wake=not endpoint.heard_speech,
            )
        if not endpoint.heard_speech:
            self._set_state("idle")
            return
        self._set_state("transcribing")
        stt_started = time.monotonic()
        result = self.stt.transcribe(bytes(audio))
        stt_ms = result.duration_ms or int((time.monotonic() - stt_started) * 1000)
        self._emit_stt(result)
        if result.rejected_reason:
            self._rejected(result, endpoint_ms, stt_ms)
            return
        if is_stop_phrase(result.text, self.cfg.conversation.stop_phrases):
            self._stop_turn(endpoint_ms, stt_ms, acknowledge=not woke or True)
            return
        self._run_text(
            result.text,
            timings={"endpoint_ms": endpoint_ms, "stt_ms": stt_ms},
            end_of_speech=time.monotonic(),
        )

    def _run_text(self, text: str, timings: dict[str, int], end_of_speech: float) -> TurnResult:
        self._ensure_session()
        turn_id = new_ulid()
        started_at = utc_now_ms()
        started_mono = time.monotonic()
        self.events.ctx.turn_id = turn_id
        self.events.ctx.session_id = self.session_id
        first_audio_at: list[float] = []

        def speak(sentence: str) -> bool:
            played = self._speak_text(sentence)
            if played.first_audio_at and not first_audio_at:
                first_audio_at.append(played.first_audio_at)
            return played.interrupted

        self._set_state("thinking")
        outcome = self.conversation.execute(text, speak)
        if outcome.error_code and outcome.error_code.startswith("llm."):
            self.api_reachable = outcome.error_category == "llm_rate_limit"
        elif outcome.status == "ok":
            self.api_reachable = True
            self.last_api_ok_at = utc_now_ms()
        total_ms = int((time.monotonic() - started_mono) * 1000)
        e2e = None
        if first_audio_at:
            e2e = max(0, int((first_audio_at[0] - end_of_speech) * 1000))
        stage = {
            "endpoint_ms": int(timings.get("endpoint_ms", 0)),
            "stt_ms": int(timings.get("stt_ms", 0)),
            "prompt_ms": int(outcome.prompt_ms),
            "llm_ttft_ms": int(outcome.llm_ttft_ms or 0),
            "llm_total_ms": int(outcome.llm_total_ms),
            "tool_ms": int(outcome.tool_ms),
            "tts_first_ms": int(timings.get("tts_first_ms", 0)),
        }
        fields = {
            "status": outcome.status,
            "intent": _intent(outcome.tools_called),
            "prompt_redacted": outcome.prompt_redacted,
            "prompt_hash": outcome.prompt_hash,
            "prompt_chars": outcome.prompt_chars,
            "response_redacted": outcome.response_redacted,
            "sensitive": outcome.sensitive,
            "sensitive_category": outcome.sensitive_category,
            "hash_epoch": self.cfg.privacy.hash_epoch,
            "tools_called": outcome.tools_called,
            "rounds": outcome.rounds,
            "stage_timings": stage,
            "e2e_first_audio_ms": e2e,
            "total_ms": total_ms,
            "tokens_total": outcome.tokens_total,
            "cost_usd": outcome.cost_usd,
            "fallback_spoken": outcome.fallback_spoken,
            "started_at": started_at,
            "model": outcome.model,
        }
        if outcome.response_hash:
            fields["response_hash"] = outcome.response_hash
        self.events.emit("turn.completed", level="info" if outcome.status == "ok" else "error", **fields)
        self.events.emit(
            "playback.completed",
            audio_ms=0,
            interrupted=outcome.interrupted,
        )
        self.session_touched = time.monotonic()
        if outcome.status != "cancelled":
            self.follow_up_until = time.monotonic() + self.cfg.conversation.follow_up_seconds
            self._set_state("listening")
        else:
            self.follow_up_until = 0.0
            self._set_state("idle")
        self.events.ctx.turn_id = None
        self._drain_interrupt()
        return outcome

    def _speak_text(self, text: str):
        self._set_state("speaking")
        logic = getattr(self.wake, "logic", None)
        if logic is not None:
            logic.reset_refractory()
        tts_started = time.monotonic()
        try:
            pcm, rate = self.tts.synthesise(text)
        except Exception:
            self.events.emit(
                "error",
                level="error",
                category="tts",
                error_code="tts.failed",
                message="Speech synthesis failed.",
                stage="tts",
                retryable=False,
                retry_count=0,
                fallback_spoken=True,
            )
            self._set_state("error")
            return _Speech(True, False, None)
        tts_ms = int((time.monotonic() - tts_started) * 1000)
        self.events.emit(
            "tts.completed",
            engine=self.cfg.tts.engine,
            voice=self.cfg.tts.voice,
            chars=len(text),
            duration_ms=tts_ms,
        )
        return self._play(pcm, rate)

    def _play(self, pcm: bytes, sample_rate: int):
        step = max(2, int(sample_rate * self.cfg.audio.chunk_ms / 1000) * 2)
        offset = 0
        first_audio_at = None
        speech_ms = 0
        heard = bytearray()
        while offset < len(pcm) and not self.shutdown.is_set():
            for _ in range(3):
                mic = self.capture.read(0.0, nowait=True)
                if not mic:
                    break
                residual = self.aec.process(mic) if self.aec else mic
                heard.extend(residual)
                decision, speech_ms = self._barge(residual, speech_ms)
                if decision:
                    self.sink.stop()
                    if decision == "wake":
                        self._pending_preroll = bytes(heard)
                        return _Speech(True, True, first_audio_at, "wake")
                    extra = self._collect(700)
                    clip = bytes(heard) + extra
                    try:
                        heard_text = self.stt.transcribe(clip)
                    except Exception:
                        heard_text = None
                    phrase = heard_text.text if heard_text else ""
                    if heard_text and is_stop_phrase(phrase, self.cfg.conversation.stop_phrases):
                        return _Speech(True, True, first_audio_at, "stop")
                    if heard_text and not heard_text.rejected_reason and phrase:
                        self._queued_text = phrase
                        return _Speech(True, True, first_audio_at, "barge_text", phrase)
                    resume = getattr(self.sink, "resume", None)
                    if resume:
                        resume()
                    speech_ms = 0
                    break
            piece = pcm[offset : offset + step]
            offset += step
            if self.aec:
                self.aec.push_reference(piece)
            if first_audio_at is None:
                first_audio_at = time.monotonic()
            self.sink.write(piece)
        return _Speech(False, False, first_audio_at)

    def _barge(self, residual: bytes, speech_ms: int) -> tuple[str | None, int]:
        score = self.wake.score(residual)
        if self.wake.observe(score, time.monotonic()):
            return "wake", speech_ms
        if not self.cfg.conversation.energy_barge_in:
            return None, speech_ms
        if self.vad.is_speech(residual):
            speech_ms += self.cfg.audio.chunk_ms
        else:
            speech_ms = 0
        if speech_ms >= self.cfg.conversation.barge_in_min_ms:
            return "energy", speech_ms
        return None, speech_ms

    def _collect(self, ms: int) -> bytes:
        needed = int(self.cfg.audio.sample_rate * ms / 1000) * 2
        buf = bytearray()
        while len(buf) < needed:
            frame = self.capture.read(0.05, nowait=True)
            if not frame:
                break
            residual = self.aec.process(frame) if self.aec else frame
            buf.extend(residual)
        return bytes(buf)

    def _rejected(self, result, endpoint_ms: int, stt_ms: int) -> None:
        phrase = "Sorry, I didn't catch that."
        self.events.emit(
            "error",
            level="warning",
            category="stt",
            error_code="stt.empty" if result.rejected_reason in {"empty", "hallucination"} else "stt.low_confidence",
            message="I didn't catch that.",
            stage="stt",
            retryable=False,
            retry_count=0,
            fallback_spoken=True,
        )
        self._run_fallback_turn(phrase, result.text or "", endpoint_ms, stt_ms)

    def _stop_turn(self, endpoint_ms: int, stt_ms: int, acknowledge: bool) -> None:
        if acknowledge:
            self._speak_text("OK.")
        shown, digest, sensitive, category = self._redacted("stop")
        ack, ack_hash, _, _ = self._redacted("OK.")
        self._ensure_session()
        self.events.ctx.session_id = self.session_id
        self.events.ctx.turn_id = new_ulid()
        self.events.emit(
            "turn.completed",
            status="cancelled",
            intent="stop",
            prompt_redacted=shown,
            prompt_hash=digest,
            prompt_chars=4,
            response_redacted=ack,
            response_hash=ack_hash,
            sensitive=sensitive,
            sensitive_category=category,
            hash_epoch=self.cfg.privacy.hash_epoch,
            tools_called=[],
            rounds=0,
            stage_timings={"endpoint_ms": endpoint_ms, "stt_ms": stt_ms, "prompt_ms": 0, "llm_total_ms": 0, "tool_ms": 0},
            e2e_first_audio_ms=None,
            total_ms=endpoint_ms + stt_ms,
            tokens_total=0,
            cost_usd=0.0,
            fallback_spoken=False,
            started_at=utc_now_ms(),
            model=None,
        )
        self.follow_up_until = 0.0
        self._set_state("idle")
        self.events.ctx.turn_id = None
        self._drain_interrupt()

    def _run_fallback_turn(self, phrase: str, heard: str, endpoint_ms: int, stt_ms: int) -> None:
        self._speak_text(phrase)
        shown, digest, sensitive, category = self._redacted(heard or "")
        reply, reply_hash, _, _ = self._redacted(phrase)
        self._ensure_session()
        self.events.ctx.session_id = self.session_id
        self.events.ctx.turn_id = new_ulid()
        self.events.emit(
            "turn.completed",
            level="warning",
            status="error",
            intent="stt",
            prompt_redacted=shown,
            prompt_hash=digest,
            prompt_chars=len(heard),
            response_redacted=reply,
            response_hash=reply_hash,
            sensitive=sensitive,
            sensitive_category=category,
            hash_epoch=self.cfg.privacy.hash_epoch,
            tools_called=[],
            rounds=0,
            stage_timings={"endpoint_ms": endpoint_ms, "stt_ms": stt_ms, "prompt_ms": 0, "llm_total_ms": 0, "tool_ms": 0},
            e2e_first_audio_ms=None,
            total_ms=endpoint_ms + stt_ms,
            tokens_total=0,
            cost_usd=0.0,
            fallback_spoken=True,
            started_at=utc_now_ms(),
            model=None,
        )
        self.follow_up_until = time.monotonic() + self.cfg.conversation.follow_up_seconds
        self._set_state("listening")
        self.events.ctx.turn_id = None
        self._drain_interrupt()

    def _redacted(self, text: str) -> tuple[str | None, str, bool, str | None]:
        pepper = env_value("GVA_HASH_PEPPER") or ("0" * 32)
        if len(pepper) < 32:
            pepper = "0" * 32
        kwargs = {
            "household_names": self.cfg.privacy.household_names,
            "categories": self.cfg.privacy.sensitive_categories,
        }
        local = redact(text, short_numbers=self.cfg.privacy.redact_short_numbers, **kwargs)
        strict = redact(text, short_numbers=True, **kwargs)
        shown = f"[SENSITIVE: {local.category}]" if local.sensitive else local.text
        shown = shown[:2000] if shown else None
        return shown, prompt_hash(strict.text, pepper), local.sensitive, local.category

    def _drain_interrupt(self) -> None:
        """Start the next turn after barge-in. Stop leaves the room quiet."""
        queued = self._queued_text
        self._queued_text = None
        pending = self._pending_preroll
        if queued:
            self._run_text(queued, {"endpoint_ms": 0, "stt_ms": 0}, time.monotonic())
            return
        if not pending:
            return
        width = chunk_bytes(self.cfg.audio.sample_rate, self.cfg.audio.chunk_ms)
        first = pending[-width:] if len(pending) >= width else pending
        self._capture(
            first,
            woke=True,
            wake_score=max(self._last_wake_score, self.cfg.wake.threshold),
            detected_at=utc_now_ms(),
        )

    def _emit_stt(self, result) -> None:
        shown, digest, _, _ = self._redacted(result.text or "")
        fields = {
            "engine": result.engine,
            "model": result.model,
            "audio_ms": int(result.audio_ms),
            "duration_ms": int(result.duration_ms),
            "confidence": result.confidence,
            "language": result.language,
            "prompt_redacted": shown[:2000] if shown else None,
            "prompt_hash": digest,
            "rejected_reason": result.rejected_reason,
        }
        self.events.emit("stt.completed", level="info" if not result.rejected_reason else "warning", **fields)

    def _service(self) -> None:
        now = time.monotonic()
        if now - self._last_heartbeat >= self.cfg.telemetry.heartbeat_seconds:
            self._last_heartbeat = now
            self._heartbeat()
        self._announce_due()

    def _heartbeat(self) -> None:
        metrics = snapshot(Path(self.cfg.telemetry.log_dir))
        self.events.emit(
            "heartbeat",
            cpu_pct=float(metrics["cpu_pct"] or 0),
            mem_pct=float(metrics["mem_pct"] or 0),
            disk_pct=float(metrics["disk_pct"] or 0),
            temp_c=metrics["temp_c"],
            throttled=bool(metrics["throttled"]) if metrics["throttled"] is not None else False,
            queue_depths={"capture": 0},
            api_reachable=self.api_reachable,
            last_api_ok_at=self.last_api_ok_at,
        )

    def _announce_due(self) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        for timer in self.store.due_timers(now_iso):
            label = timer.get("label") or "timer"
            speech = self._speak_text(f"Your {label} timer is done.")
            timer["status"] = "done"
            self.store.upsert_timer(timer)
            if speech.interrupted:
                self._drain_interrupt()
                return
        for alarm in self.store.due_alarms(now_iso):
            label = alarm.get("label") or "Alarm"
            speech = self._speak_text(f"{label}.")
            alarm["status"] = "done"
            self.store.upsert_alarm(alarm)
            if speech.interrupted:
                self._drain_interrupt()
                return

    def _ensure_session(self) -> None:
        now = time.monotonic()
        if self.session_id is None or now - self.session_touched > self.cfg.conversation.session_idle_minutes * 60:
            self.session_id = new_ulid()
            self.conversation.reset()
        self.session_touched = now
        self.events.ctx.session_id = self.session_id

    def _set_state(self, new: str) -> None:
        if new == self.state:
            return
        previous = self.state
        self.state = new
        self.events.emit("state.changed", **{"from": previous, "to": new})

    def _play_earcon(self) -> None:
        pcm = tone(120, freq=880, amp=3000, sample_rate=self.cfg.audio.sample_rate)
        if self.aec:
            self.aec.push_reference(pcm)
        self.sink.write(pcm)

    def _muted(self) -> bool:
        return self._mute_path().exists()

    def _mute_path(self) -> Path:
        return Path(self.cfg.telemetry.db_path).parent / "muted"


@dataclass
class _Speech:
    ok: bool
    interrupted: bool
    first_audio_at: float | None
    reason: str | None = None
    barge_text: str | None = None


def _intent(tools: list[str]) -> str:
    if not tools:
        return "chat"
    name = tools[0]
    for key in ("timer", "alarm", "weather", "device", "calendar", "time"):
        if key in name:
            return key
    return name
