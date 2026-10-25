"""Wire the on-device voice pipeline. Tests build their own doubles instead."""

from __future__ import annotations

from pathlib import Path

import httpx

from gva.config import Config
from gva.runtime.audio import SoundDeviceCapture, SoundDeviceSink, list_devices, prefer_output, resolve_device
from gva.runtime.grok_client import GrokClient
from gva.runtime.pipeline import VoiceAssistant
from gva.runtime.realtime import RealtimeAssistant
from gva.runtime.stt import FasterWhisperStt, SttRouter, XaiStt
from gva.runtime.tools.registry import load_registry
from gva.runtime.tts import PiperTts, TtsRouter, XaiTts
from gva.runtime.vad import EnergyVad, SileroVad
from gva.runtime.wake import OpenWakeWordEngine
from gva.telemetry.events import EventLog
from gva.telemetry.store import Store


def build_assistant(cfg: Config):
    """Realtime speech-to-speech by default. `conversation.engine: local` keeps the cascade."""
    if cfg.conversation.engine == "realtime":
        return build_realtime(cfg)
    return build_local(cfg)


def build_realtime(cfg: Config) -> RealtimeAssistant:
    store, events = _telemetry(cfg)
    capture = _capture(cfg, events, cfg.audio.capture_rate)
    sink = SoundDeviceSink(
        _output_index(cfg),
        cfg.conversation.realtime.sample_rate,
        prebuffer_ms=cfg.audio.playback_prebuffer_ms,
        latency=cfg.audio.playback_latency,
    )
    wake = _wake(cfg)
    tools = load_registry(cfg, store)
    return RealtimeAssistant(cfg, store, events, capture, sink, wake, tools)


def build_local(cfg: Config) -> VoiceAssistant:
    """Whisper, Grok chat, and Piper. Capture stays at the 16 kHz the cascade expects."""
    store, events = _telemetry(cfg)
    capture = _capture(cfg, events, cfg.audio.sample_rate)
    sink = SoundDeviceSink(
        _output_index(cfg),
        cfg.audio.sample_rate,
        prebuffer_ms=cfg.audio.playback_prebuffer_ms,
        latency=cfg.audio.playback_latency,
    )
    wake = _wake(cfg)
    energy = EnergyVad(cfg.vad.energy_threshold)
    vad = (
        SileroVad(energy, str(Path(cfg.wake.model_dir) / "silero_vad.onnx"))
        if cfg.vad.engine == "silero"
        else energy
    )
    http = httpx.Client(timeout=httpx.Timeout(cfg.llm.timeouts.read_s, connect=cfg.llm.timeouts.connect_s))
    stt = SttRouter(cfg, FasterWhisperStt(cfg), XaiStt(cfg, http))
    tts = TtsRouter(cfg, PiperTts(cfg), XaiTts(cfg, http))
    grok = GrokClient(cfg, http=http)
    tools = load_registry(cfg, store, http=http)
    return VoiceAssistant(cfg, store, events, capture, sink, wake, vad, stt, tts, grok, tools)


def _telemetry(cfg: Config) -> tuple[Store, EventLog]:
    store = Store(Path(cfg.telemetry.db_path))
    events = EventLog(
        device_id=cfg.device.id,
        env=cfg.device.env,
        log_dir=Path(cfg.telemetry.log_dir),
        store=store,
        schema_version=cfg.telemetry.schema_version,
        validate=cfg.telemetry.validate_events,
    )
    return store, events


def _capture(cfg: Config, events: EventLog, sample_rate: int) -> SoundDeviceCapture:
    def on_lost() -> None:
        events.emit(
            "audio.device_lost",
            level="error",
            message="The microphone stream stopped. Reopening.",
        )

    return SoundDeviceCapture(
        resolve_device(cfg.audio.input_device, "input"),
        sample_rate,
        cfg.audio.chunk_ms,
        on_lost=on_lost,
    )


def _wake(cfg: Config) -> OpenWakeWordEngine:
    return OpenWakeWordEngine(
        cfg.wake.model,
        cfg.wake.model_dir,
        cfg.wake.threshold,
        cfg.wake.patience_frames,
        cfg.wake.refractory_seconds,
        cfg.wake.phrase,
    )


def _output_index(cfg: Config) -> int | None:
    """USB playback first, then the headphone jack, unless config names a device."""
    devices = list_devices()
    if devices and devices[0].get("error"):
        if cfg.audio.output_device:
            return resolve_device(cfg.audio.output_device, "output")
        return None
    return prefer_output(devices, cfg.audio.output_device)
