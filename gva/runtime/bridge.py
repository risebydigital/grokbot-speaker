"""Hand Rise By Digital work questions to Adam's Grok Bot bots.

Quick chat stays on the xAI realtime session. This module only runs when the
webhook, its key, and Supabase are all set. The key and any query string on
the webhook URL are never written to logs or events.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from gva.config import ENV_FILE_STATUS, Config, env_value, load_service_env
from gva.ids import new_ulid, utc_now_ms
from gva.telemetry.events import EventLog

WAITING_LINE = "Give me a moment, I'm checking with the team."
TIMEOUT_LINE = "Sorry, the team didn't get back to me in time. Try again in a bit."
UNREACHABLE_LINE = "Sorry, I couldn't reach the team just now."

CALENDAR_BULLET = (
    "- Calendar and reports are not connected. If asked, call get_calendar_events "
    "and then say so in one sentence. Do not invent events, meetings, or deadlines."
)
WORK_BULLET = (
    "- Rise By Digital work questions (Asana tasks, reports due, client work, "
    "Adam's calendar, and emails) use ask_rbd_team. Tell the caller you are checking "
    "with the team. Do not invent tasks, meetings, or emails. Answer everything else yourself."
)

WEBHOOK_TIMEOUT_S = 10.0
QUESTION_LIMIT = 2000
_SAFE_ERROR = re.compile(r"^(HTTP [1-5][0-9]{2}|invalid response|not_configured|[A-Za-z_][A-Za-z0-9_]{0,60})$")


class BridgeError(RuntimeError):
    """Safe to show. The message never includes the webhook URL or the key."""


class BridgeTimeout(BridgeError):
    pass


def bridge_configured(cfg: Config) -> bool:
    """On when enabled and the webhook, its key, and Supabase are all set."""
    if not cfg.bot_bridge.enabled:
        return False
    return bool(
        env_value("GVA_GROKBOT_WEBHOOK_URL")
        and env_value("GVA_GROKBOT_WEBHOOK_KEY")
        and env_value("SUPABASE_URL")
        and env_value("SUPABASE_SERVICE_KEY")
    )


def bridge_status(cfg: Config, probe: Callable[[Config], tuple[bool, str]] | None = None) -> tuple[str, str | None]:
    """Doctor line, plus a warning when the answers table cannot be read.

    The warning does not include the webhook, the key, or a response body.
    """
    load_service_env()
    if not bridge_configured(cfg):
        if ENV_FILE_STATUS == "unreadable":
            return "bridge: not checked (cannot read the env file)", None
        return "bridge: not configured", None
    check = probe or probe_answers
    try:
        ok, detail = check(cfg)
    except Exception as exc:
        ok, detail = False, exc.__class__.__name__
    if ok:
        return "bridge: configured, milo_bot_answers reachable", None
    safe = _public_error(str(detail))
    return (
        "bridge: configured, milo_bot_answers not reachable",
        f"milo_bot_answers is not reachable ({safe})",
    )


def probe_answers(cfg: Config, http: Any | None = None) -> tuple[bool, str]:
    """GET one row id from public.milo_bot_answers. The body is discarded."""
    url = env_value("SUPABASE_URL")
    key = env_value("SUPABASE_SERVICE_KEY")
    if not url or not key:
        return False, "not_configured"
    endpoint = url.rstrip("/") + "/rest/v1/milo_bot_answers"
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
    }
    params = {"select": "request_id", "limit": "1"}
    owns = http is None
    client = http if http is not None else _http_client(5.0)
    try:
        response = client.get(endpoint, params=params, headers=headers, timeout=5.0)
    except Exception as exc:
        return False, exc.__class__.__name__
    finally:
        if owns:
            closer = getattr(client, "close", None)
            if closer:
                closer()
    code = int(getattr(response, "status_code", 0) or 0)
    if code == 200:
        return True, "ok"
    if code:
        return False, f"HTTP {code}"
    return False, "bridge error"


def _public_error(message: str) -> str:
    text = (message or "bridge error").strip()
    if _SAFE_ERROR.fullmatch(text):
        return text[:300]
    return "bridge error"


def _iso_in(seconds: float) -> str:
    moment = datetime.now(timezone.utc) + timedelta(seconds=float(seconds))
    return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _http_client(timeout_s: float):
    import httpx

    return httpx.Client(timeout=httpx.Timeout(timeout_s, connect=timeout_s))


class BotBridge:
    """Post a work question, poll for the answer, and speak it when it arrives."""

    def __init__(
        self,
        cfg: Config,
        events: EventLog,
        sink=None,
        pcm_rate: int | None = None,
        http: Any | None = None,
        sleeper: Callable[[float], None] | None = None,
        monotonic: Callable[[], float] | None = None,
        speaker: Callable[[str], None] | None = None,
        connector: Callable[..., Any] | None = None,
    ):
        self.cfg = cfg
        self.events = events
        self.sink = sink
        self.pcm_rate = int(pcm_rate or cfg.conversation.realtime.sample_rate)
        self._http = http
        self._owned = None
        self._sleep = sleeper or time.sleep
        self._mono = monotonic or time.monotonic
        self._closed_speaker = speaker
        self._connector = connector
        self._protocol = None
        self._lock = threading.Lock()
        self.deliver = True

    def attach(self, protocol) -> None:
        with self._lock:
            self._protocol = protocol

    def take_unsent(self, protocol) -> list[str]:
        """Unhook the session and return answers that were not sent on the socket."""
        with self._lock:
            if self._protocol is protocol:
                self._protocol = None
        return protocol.drain_unsent()

    def detach(self, protocol) -> None:
        """Drop the session. Anything not yet spoken is said with the closed path."""
        for text in self.take_unsent(protocol):
            self.speak_closed(text, chime=True)

    def submit(self, question: str, *, background: bool = True) -> str:
        """POST the webhook. The poller keeps running after the voice session closes."""
        request_id, started = self._send(question)
        if background:
            threading.Thread(
                target=self._watch,
                args=(request_id, started),
                name="gva-bridge",
                daemon=True,
            ).start()
        else:
            self._watch(request_id, started)
        return request_id

    def send_and_wait(self, question: str) -> str:
        """CLI path. Prints nothing and does not play audio. Raises on failure."""
        previous = self.deliver
        self.deliver = False
        try:
            request_id, started = self._send(question)
            answer = self._watch(request_id, started)
        finally:
            self.deliver = previous
        if not answer:
            raise BridgeTimeout(TIMEOUT_LINE)
        return answer

    def deliver_text(self, text: str, *, naturally: bool, chime: bool) -> None:
        spoken = text.strip()
        if not spoken:
            return
        with self._lock:
            protocol = self._protocol
            if protocol is not None and not protocol.close:
                protocol.enqueue_bridge(self._inject_messages(spoken, naturally=naturally), spoken)
                return
        self.speak_closed(spoken, chime=chime)

    def speak_closed(self, text: str, *, chime: bool) -> None:
        """Chime, then a short verbatim realtime session, then Piper."""
        if chime:
            self._chime()
        if self._closed_speaker is not None:
            try:
                self._closed_speaker(text)
            except Exception:
                return
            return
        try:
            if self._speak_realtime_verbatim(text):
                return
        except Exception:
            pass
        self._speak_piper(text)

    def _send(self, question: str) -> tuple[str, float]:
        url = env_value("GVA_GROKBOT_WEBHOOK_URL")
        key = env_value("GVA_GROKBOT_WEBHOOK_KEY")
        started = self._mono()
        cleaned = " ".join((question or "").split())
        if not cleaned:
            raise BridgeError("missing question")
        if not url or not key:
            self._emit_error("not_configured")
            raise BridgeError("could not reach the team")
        request_id = new_ulid()
        body = {
            "request_id": request_id,
            "device_id": self.cfg.device.id,
            "question": cleaned[:QUESTION_LIMIT],
            "asked_at": utc_now_ms(),
            "reply_by": _iso_in(self.cfg.bot_bridge.timeout_s),
        }
        header_name = (self.cfg.bot_bridge.key_header or "Authorization").strip() or "Authorization"
        prefix = self.cfg.bot_bridge.key_prefix
        if prefix is None:
            prefix = "Bearer "
        headers = {
            header_name: f"{prefix}{key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        last_error = "bridge error"
        for attempt in (1, 2):
            try:
                response = self._client().post(url, json=body, headers=headers, timeout=WEBHOOK_TIMEOUT_S)
            except Exception as exc:
                last_error = exc.__class__.__name__
                if attempt == 1:
                    continue
                break
            code = int(getattr(response, "status_code", 0) or 0)
            if 200 <= code < 300:
                self.events.emit("bridge.sent", request_id=request_id)
                return request_id, started
            last_error = f"HTTP {code}" if code else "bridge error"
            if code and code < 500 and code != 429:
                break
            if attempt == 1:
                continue
            break
        self._emit_error(last_error, request_id)
        raise BridgeError("could not reach the team")

    def _watch(self, request_id: str, started: float) -> str | None:
        try:
            row = self._poll(request_id, started)
        except Exception as exc:
            self._emit_error(exc.__class__.__name__, request_id)
            return None
        if not row:
            try:
                self.events.emit("bridge.timeout", level="warning", request_id=request_id)
            except Exception:
                pass
            if self.deliver:
                self.deliver_text(TIMEOUT_LINE, naturally=False, chime=False)
            return None
        latency = max(0, int((self._mono() - started) * 1000))
        try:
            self.events.emit("bridge.answered", request_id=request_id, latency_ms=latency)
        except Exception:
            pass
        self._mark_read(request_id)
        answer = str(row.get("answer") or "").strip()
        if self.deliver and answer:
            self.deliver_text(answer, naturally=True, chime=True)
        return answer or None

    def _poll(self, request_id: str, started: float) -> dict[str, Any] | None:
        deadline = started + float(self.cfg.bot_bridge.timeout_s)
        told = False
        while True:
            row = None
            error = None
            try:
                row, error = self._fetch(request_id)
            except Exception as exc:
                error = exc.__class__.__name__
            if error and not told:
                self._emit_error(error, request_id)
                told = True
            if row and str(row.get("answer") or "").strip():
                return row
            if self._mono() >= deadline:
                return None
            self._sleep(float(self.cfg.bot_bridge.poll_s))

    def _fetch(self, request_id: str) -> tuple[dict[str, Any] | None, str | None]:
        url = self._table_url()
        headers = self._supabase_headers()
        if not url or headers is None:
            return None, "not_configured"
        response = self._client().get(
            url,
            params={
                "request_id": f"eq.{request_id}",
                "select": "request_id,answer,answered_by,read_at",
            },
            headers=headers,
            timeout=WEBHOOK_TIMEOUT_S,
        )
        code = int(getattr(response, "status_code", 0) or 0)
        if code != 200:
            return None, f"HTTP {code}" if code else "bridge error"
        try:
            data = response.json()
        except Exception:
            return None, "invalid response"
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return data[0], None
        return None, None

    def _mark_read(self, request_id: str) -> None:
        url = self._table_url()
        headers = self._supabase_headers()
        if not url or headers is None:
            self._emit_error("not_configured", request_id)
            return
        headers = dict(headers)
        headers["Content-Type"] = "application/json"
        headers["Prefer"] = "return=minimal"
        try:
            response = self._client().patch(
                url,
                params={"request_id": f"eq.{request_id}"},
                headers=headers,
                json={"read_at": utc_now_ms()},
                timeout=WEBHOOK_TIMEOUT_S,
            )
        except Exception as exc:
            self._emit_error(exc.__class__.__name__, request_id)
            return
        code = int(getattr(response, "status_code", 0) or 0)
        if code >= 400:
            self._emit_error(f"HTTP {code}", request_id)

    def _table_url(self) -> str | None:
        url = env_value("SUPABASE_URL")
        if not url:
            return None
        return url.rstrip("/") + "/rest/v1/milo_bot_answers"

    def _supabase_headers(self) -> dict[str, str] | None:
        key = env_value("SUPABASE_SERVICE_KEY")
        if not key or not env_value("SUPABASE_URL"):
            return None
        return {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
        }

    def _client(self):
        if self._http is not None:
            return self._http
        if self._owned is None:
            self._owned = _http_client(WEBHOOK_TIMEOUT_S)
        return self._owned

    def _emit_error(self, message: str, request_id: str | None = None) -> None:
        fields: dict[str, Any] = {"message": _public_error(message)}
        if request_id:
            fields["request_id"] = request_id
        try:
            self.events.emit("bridge.error", level="error", **fields)
        except Exception:
            return

    def _inject_messages(self, text: str, *, naturally: bool) -> list[dict[str, Any]]:
        if naturally:
            instructions = (
                "Read the team's answer out naturally and briefly, in British English. "
                "Do not add anything that is not in the answer."
            )
        else:
            instructions = "Say only this, with no extra words: " + text
        return [
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            },
            {"type": "response.create", "response": {"instructions": instructions}},
        ]

    def _chime(self) -> None:
        sink = self.sink
        if sink is None:
            return
        from gva.runtime.audio import tone

        rate = int(getattr(sink, "sample_rate", 0) or self.pcm_rate or 24000)
        pcm = tone(90, 880, amp=6000, sample_rate=rate) + tone(140, 1175, amp=6000, sample_rate=rate)
        self._write_pcm(pcm, rate)

    def _write_pcm(self, pcm: bytes, rate: int) -> None:
        sink = self.sink
        if sink is None or not pcm:
            return
        target = int(getattr(sink, "sample_rate", 0) or self.pcm_rate or rate)
        if rate != target and pcm:
            from gva.runtime.audio import resample_pcm

            pcm = resample_pcm(pcm, rate, target)
        resume = getattr(sink, "resume", None)
        if resume:
            try:
                resume()
            except Exception:
                pass
        try:
            sink.write(pcm)
        except Exception:
            return

    def _speak_piper(self, text: str) -> bool:
        try:
            from gva.runtime.tts import PiperTts

            pcm, rate = PiperTts(self.cfg).synthesise(text)
        except Exception:
            return False
        if not pcm:
            return False
        self._write_pcm(pcm, rate)
        return True

    def _speak_realtime_verbatim(self, text: str) -> bool:
        if not env_value("XAI_API_KEY") and self._connector is None:
            return False
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            return False
        try:
            return bool(asyncio.run(self._verbatim_session(text)))
        except Exception:
            return False

    async def _verbatim_session(self, text: str) -> bool:
        """One short realtime session whose only job is to read `text` aloud."""
        key = env_value("XAI_API_KEY")
        voice = self.cfg.conversation.realtime
        if not key or not voice.url:
            return False
        url = voice.url.rstrip("/") + f"?model={voice.model}"
        connect = self._connector or _connect_realtime
        socket = connect(url, {"Authorization": f"Bearer {key}"})
        got_audio = False
        async with socket as ws:
            await ws.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {
                            "voice": voice.voice,
                            "instructions": (
                                "Read the user's text aloud verbatim, in British English, then stop. "
                                "Do not add anything."
                            ),
                            "turn_detection": None,
                            "audio": {
                                "input": {"format": {"type": "audio/pcm", "rate": voice.sample_rate}},
                                "output": {"format": {"type": "audio/pcm", "rate": voice.sample_rate}},
                            },
                            "tools": [],
                        },
                    }
                )
            )
            await ws.send(
                json.dumps(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": text}],
                        },
                    }
                )
            )
            await ws.send(
                json.dumps(
                    {
                        "type": "response.create",
                        "response": {
                            "instructions": "Read the text verbatim. Do not add a greeting or a comment.",
                        },
                    }
                )
            )
            deadline = time.monotonic() + 25
            while time.monotonic() < deadline:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=1)
                except TimeoutError:
                    continue
                except Exception:
                    break
                if isinstance(raw, bytes):
                    continue
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                etype = event.get("type") or ""
                if etype in {"response.output_audio.delta", "response.audio.delta"}:
                    payload = event.get("delta") or ""
                    if payload:
                        try:
                            pcm = base64.b64decode(payload)
                        except Exception:
                            continue
                        if pcm:
                            self._write_pcm(pcm, voice.sample_rate)
                            got_audio = True
                elif etype in {"response.done", "error"}:
                    break
        return got_audio


def _connect_realtime(url: str, headers: dict[str, str]):
    import websockets

    try:
        return websockets.connect(url, additional_headers=headers)
    except TypeError:
        return websockets.connect(url, extra_headers=headers)
