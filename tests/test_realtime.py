"""Realtime session behaviour that does not open a network socket."""

import asyncio
import base64
import json
import time

from gva.config import Config
from gva.runtime.audio import CollectingSink
from gva.runtime.realtime import RealtimeAssistant, RealtimeProtocol, audio_cost
from gva.telemetry.events import EventLog
from gva.telemetry.store import Store


class _Socket:
    def __init__(self):
        self.sent = []
        self.closed = False

    async def send(self, data):
        self.sent.append(json.loads(data))

    async def close(self):
        self.closed = True


class _Frames:
    def __init__(self, pauses=True):
        self.pauses = pauses
        self.reads = 0

    def read(self, timeout=0.2, nowait=False):
        self.reads += 1
        if self.pauses:
            time.sleep(0.05)
        if self.reads > 8:
            return None
        return b"\x00\x10" * 160


class _HitWake:
    model_name = "scripted"

    def __init__(self):
        self.reset_calls = 0

    def score(self, frame):
        return 0.91

    def observe(self, score, now=None):
        return True

    def reset(self):
        self.reset_calls += 1


def _protocol(sink=None):
    return RealtimeProtocol(Config(), None, None, None, sink or CollectingSink())


def test_audio_cost_is_per_minute():
    assert audio_cost(30, 0.08) == 0.04
    assert audio_cost(0, 0.08) == 0.0


def test_audio_deltas_queue_instead_of_hitting_the_device():
    sink = CollectingSink()
    protocol = _protocol(sink)
    pcm = b"\x01\x00" * 40
    protocol.on_event({"type": "response.output_audio.delta", "delta": base64.b64encode(pcm).decode()})
    assert sink.chunks == []
    assert protocol.speaking is True
    assert protocol.playback.get_nowait() == pcm


def test_cancel_playback_is_response_cancel():
    sink = CollectingSink()
    protocol = _protocol(sink)
    protocol.enqueue_pcm(b"\x02\x00" * 8)
    sink.stop()
    message = protocol.cancel_playback()
    assert message == {"type": "response.cancel"}
    assert protocol.speaking is False
    assert protocol.playback.empty()
    assert sink.stopped is False


def test_microphone_frames_do_not_refresh_the_silence_timer():
    cfg = Config()
    cfg.conversation.realtime.silence_timeout_s = 0.15
    protocol = RealtimeProtocol(cfg, None, None, None, CollectingSink())
    protocol.last_activity = time.monotonic()
    assistant = RealtimeAssistant(cfg, None, None, _Frames(), CollectingSink(), _HitWake(), None)
    socket = _Socket()

    async def go():
        await asyncio.wait_for(assistant._send_audio(socket, protocol, None), timeout=2)

    asyncio.run(go())
    assert protocol.close is True
    assert socket.closed is True
    assert protocol.idle(time.monotonic(), 0.15) is True


def test_wake_during_playback_sends_response_cancel():
    cfg = Config()
    cfg.conversation.realtime.silence_timeout_s = 30
    cfg.audio.aec_enabled = False
    sink = CollectingSink()
    protocol = RealtimeProtocol(cfg, None, None, None, sink)
    protocol.speaking = True
    protocol.last_activity = time.monotonic()
    wake = _HitWake()
    events_store = None
    assistant = RealtimeAssistant(cfg, events_store, _QuietLog(), _Frames(pauses=False), sink, wake, None)
    socket = _Socket()

    async def go():
        await asyncio.wait_for(assistant._send_audio(socket, protocol, None), timeout=2)

    asyncio.run(go())
    kinds = [item["type"] for item in socket.sent]
    assert "response.cancel" in kinds
    assert "input_audio_buffer.append" in kinds
    assert wake.reset_calls >= 1


def test_player_task_writes_the_sink(tmp_path):
    cfg = Config()
    sink = CollectingSink()
    protocol = RealtimeProtocol(cfg, None, None, None, sink)
    protocol.enqueue_pcm(b"\x03\x00" * 12)
    assistant = RealtimeAssistant(cfg, None, None, _Frames(pauses=False), sink, _HitWake(), None)

    async def go():
        task = asyncio.create_task(assistant._play_queued(protocol, None))
        for _ in range(40):
            if sink.chunks:
                break
            await asyncio.sleep(0.02)
        protocol.close = True
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(go())
    assert sink.chunks
    assert sink.chunks[0][:2] == b"\x03\x00"


def test_watchdog_closes_a_stalled_socket(tmp_path):
    cfg = Config()
    cfg.telemetry.log_dir = str(tmp_path / "log")
    cfg.telemetry.db_path = str(tmp_path / "metrics.db")
    cfg.conversation.realtime.watchdog_s = 0.05
    store = Store(tmp_path / "metrics.db")
    events = EventLog("milo-01", "dev", tmp_path / "log", store, validate=True)
    protocol = RealtimeProtocol(cfg, events, store, None, CollectingSink())
    protocol.last_socket = time.monotonic() - 5
    assistant = RealtimeAssistant(cfg, store, events, _Frames(pauses=False), CollectingSink(), _HitWake(), None)
    socket = _Socket()

    async def go():
        await asyncio.wait_for(assistant._watchdog(socket, protocol), timeout=2)

    asyncio.run(go())
    assert protocol.close is True
    assert socket.closed is True
    assert store.recent("error", 1)[0]["error_code"] == "net.offline"


class _QuietLog:
    def emit(self, *args, **kwargs):
        return {}
