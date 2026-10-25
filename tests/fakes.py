"""Scripted audio, speech, and HTTP doubles. No hardware and no network."""

from __future__ import annotations

import json
import time
from pathlib import Path

from gva.config import load_config
from gva.runtime.audio import CollectingSink, ListCapture, silence, tone
from gva.runtime.grok_client import ToolCall, scripted_result
from gva.runtime.pipeline import VoiceAssistant
from gva.runtime.stt import SttResult, audio_ms
from gva.runtime.tools.registry import load_registry
from gva.runtime.vad import EnergyVad
from gva.runtime.wake import EnergyWake, ThresholdWake
from gva.telemetry.events import EventLog
from gva.telemetry.store import Store


class ScriptedStt:
    def __init__(self, texts: list[str]):
        self.texts = list(texts)
        self.calls = 0

    def transcribe(self, pcm: bytes) -> SttResult:
        text = self.texts[min(self.calls, len(self.texts) - 1)]
        self.calls += 1
        return SttResult(
            text=text,
            confidence=0.9,
            language="en",
            duration_ms=10,
            audio_ms=audio_ms(pcm),
            rejected_reason=None,
            engine="faster_whisper",
            model="base.en",
        )


class ScriptedTts:
    def synthesise(self, text: str) -> tuple[bytes, int]:
        return tone(80, amp=1000), 16000


class ScriptedGrok:
    def __init__(self, results):
        self.results = list(results)
        self.calls = 0
        self.seen: list[list] = []

    def complete(self, messages, tools=None, **kwargs):
        self.calls += 1
        self.seen.append(messages)
        result = self.results.pop(0) if self.results else scripted_result("OK.")
        on_delta = kwargs.get("on_delta")
        if on_delta and result.text:
            on_delta(result.text)
        return result


class FakeResponse:
    def __init__(self, status: int, body: str):
        self.status_code = status
        self.text = body
        self._lines = []

    def json(self):
        return json.loads(self.text or "null")

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_lines(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeHttp:
    def __init__(self, responses: list[tuple[int, list[str]]]):
        self.responses = list(responses)
        self.calls = 0

    def stream(self, method, url, headers=None, json=None, timeout=None):
        self.calls += 1
        status, lines = self.responses.pop(0)
        response = FakeResponse(status, "")
        response._lines = lines
        return response


class MemorySupabase:
    def __init__(self, gets: list[list[dict]] | None = None):
        self.posts: list[tuple[str, str]] = []
        self.patches: list[tuple[str, str]] = []
        self.gets = list(gets or [])

    def post(self, url, headers=None, content=None):
        self.posts.append((url, content.decode() if isinstance(content, bytes) else content))
        return FakeResponse(201, "")

    def patch(self, url, headers=None, content=None):
        self.patches.append((url, content.decode() if isinstance(content, bytes) else content))
        return FakeResponse(200, "")

    def get(self, url, headers=None, params=None):
        payload = self.gets.pop(0) if self.gets else []
        return FakeResponse(200, json.dumps(payload))


def test_config(tmp_path: Path, monkeypatch, **env):
    monkeypatch.setenv("GVA_LOG_DIR", str(tmp_path / "log"))
    monkeypatch.setenv("GVA_DB_PATH", str(tmp_path / "metrics.db"))
    monkeypatch.delenv("GVA_CONFIG", raising=False)
    cfg = load_config()
    cfg.wake.patience_frames = 1
    cfg.wake.refractory_seconds = 0
    cfg.audio.aec_enabled = False
    cfg.conversation.energy_barge_in = False
    cfg.conversation.follow_up_seconds = 30
    cfg.telemetry.heartbeat_seconds = 3600
    cfg.telemetry.validate_events = True
    return cfg


def utterance(speech: int = 3, quiet: int = 10) -> list[bytes]:
    loud = tone(80, amp=8000)
    still = silence(80)
    return [loud] * speech + [still] * quiet


def build_assistant(tmp_path: Path, monkeypatch, frames, stt_texts, grok_results, wake=None, **cfg_overrides):
    cfg = test_config(tmp_path, monkeypatch)
    for name, value in cfg_overrides.items():
        setattr(cfg, name, value)
    store = Store(Path(cfg.telemetry.db_path))
    events = EventLog(cfg.device.id, cfg.device.env, Path(cfg.telemetry.log_dir), store, validate=True)
    logic = ThresholdWake(cfg.wake.threshold, 1, 0)
    wake = wake or EnergyWake(logic)
    capture = ListCapture(frames)
    sink = CollectingSink()
    stt = ScriptedStt(stt_texts)
    grok = ScriptedGrok(grok_results)
    tools = load_registry(cfg, store)
    assistant = VoiceAssistant(
        cfg,
        store,
        events,
        capture,
        sink,
        wake,
        EnergyVad(cfg.vad.energy_threshold),
        stt,
        ScriptedTts(),
        grok,
        tools,
    )
    assistant._last_heartbeat = time.monotonic()
    return assistant, grok, stt, sink


def say(text: str, **kwargs):
    return scripted_result(text, **kwargs)


def call(name: str, arguments: str, call_id: str = "call-1") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments)
