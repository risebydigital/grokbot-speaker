"""Capture supervision, stall watchdog, follow-up window, and playback underruns."""

import base64
import threading
import time

import numpy as np

from gva.config import Config
from gva.runtime.audio import JitterBuffer, SoundDeviceCapture, play_pcm
from gva.runtime.realtime import RealtimeAssistant, RealtimeProtocol, idle_should_close
from gva.telemetry.events import EventLog
from gva.telemetry.store import Store


class _HoldSink:
    """Records PCM and can pretend the device has not finished playing it."""

    def __init__(self):
        self.chunks = []
        self.held = 0
        self.stopped = False
        self.underruns = 0

    def write(self, pcm: bytes) -> None:
        if self.stopped:
            return
        self.chunks.append(pcm)
        self.held += max(0, len(pcm) // 2)

    def stop(self) -> None:
        self.stopped = True
        self.held = 0

    def resume(self) -> None:
        self.stopped = False

    def queued_frames(self) -> int:
        return self.held

    def drain(self) -> None:
        self.held = 0

    def consume_underruns(self) -> int:
        count = self.underruns
        self.underruns = 0
        return count


class _Capture:
    running = False
    last_audio_at = 0.0

    def queued(self) -> int:
        return 0


def _assistant(tmp_path, capture=None):
    cfg = Config()
    cfg.telemetry.log_dir = str(tmp_path / "log")
    cfg.telemetry.db_path = str(tmp_path / "metrics.db")
    cfg.telemetry.heartbeat_seconds = 30
    store = Store(tmp_path / "metrics.db")
    events = EventLog("milo-01", "dev", tmp_path / "log", store, validate=True)
    sink = _HoldSink()
    assistant = RealtimeAssistant(cfg, store, events, capture or _Capture(), sink, None, None)
    assistant._halt = lambda code: None
    return cfg, store, events, assistant, sink


def test_read_waits_and_flush_does_not_spin():
    capture = SoundDeviceCapture(None, 16000, 80)
    started = time.monotonic()
    assert capture.read(0.2) == b""
    assert time.monotonic() - started >= 0.15

    stop = threading.Event()

    def produce() -> None:
        while not stop.is_set():
            capture._put(b"\x01\x00" * 8)

    thread = threading.Thread(target=produce)
    thread.start()
    try:
        began = time.monotonic()
        capture.flush()
        assert time.monotonic() - began < 0.5
    finally:
        stop.set()
        thread.join(timeout=1)


def test_spinning_callback_is_stopped_and_eof_reopens(tmp_path):
    reported: list[BaseException] = []
    capture = SoundDeviceCapture(None, 16000, 80, on_error=reported.append)
    capture._rate_window_start = time.monotonic() - 2
    capture._calls = 10000
    try:
        capture._on_audio(np.zeros(16, dtype=np.int16), 16, None, None)
        raised = False
    except RuntimeError as exc:
        raised = "spun" in str(exc)
    assert raised
    assert capture._failed.is_set()
    assert "spun" in str(capture._error)

    capture._failed.clear()
    capture._error = None
    capture.reopen_delay = 0
    capture._running = True
    capture.last_audio_at = time.monotonic()

    class _Dead:
        active = False

        def abort(self) -> None:
            return None

        def close(self) -> None:
            return None

    capture._stream = _Dead()
    capture._fail(EOFError("capture stream ended"))
    opened: list[str] = []

    def _open() -> None:
        opened.append("open")
        capture._stream = type("Live", (), {"active": True})()

    capture._open_stream = _open
    capture._supervise_once()
    assert opened == ["open"]
    assert reported and isinstance(reported[-1], EOFError)
    assert "capture stream ended" in str(reported[-1])

    cfg = Config()
    cfg.telemetry.log_dir = str(tmp_path / "log")
    cfg.telemetry.db_path = str(tmp_path / "metrics.db")
    store = Store(tmp_path / "metrics.db")
    events = EventLog("milo-01", "dev", tmp_path / "log", store, validate=True)
    exc = reported[-1]
    events.emit(
        "error",
        level="error",
        category="audio",
        error_code="audio.capture",
        message=f"{exc.__class__.__name__}: {exc}"[:300],
        stage="capture",
        retryable=True,
        retry_count=0,
        fallback_spoken=False,
    )
    logged = store.recent("error", 1)[0]
    assert logged["level"] == "error"
    assert logged["message"].startswith("EOFError:")
    assert "capture stream ended" in logged["message"]


def test_silent_stream_is_reopened():
    capture = SoundDeviceCapture(None, 16000, 80, on_error=lambda exc: None)
    capture.reopen_delay = 0
    capture._running = True
    capture.last_audio_at = time.monotonic() - 6
    capture._opened_at = time.monotonic() - 6
    capture._stream = type("Live", (), {"active": True, "abort": lambda self: None, "close": lambda self: None})()
    opened: list[str] = []

    def _open() -> None:
        opened.append("open")
        capture._stream = type("Live", (), {"active": True})()

    capture._open_stream = _open
    capture._supervise_once()
    assert opened == ["open"]
    assert isinstance(capture._reported, TimeoutError)
    assert capture.last_audio_at < time.monotonic() - 5


def test_a_stream_that_just_opened_is_not_reopened_for_silence():
    capture = SoundDeviceCapture(None, 16000, 80, on_error=lambda exc: None)
    capture._running = True
    stalled_at = time.monotonic() - 30
    capture.last_audio_at = stalled_at
    capture._opened_at = time.monotonic()
    capture._stream = type("Live", (), {"active": True})()
    capture._supervise_once()
    assert not capture._failed.is_set()
    assert capture.last_audio_at == stalled_at


def test_watchdog_exits_when_audio_or_wake_stalls(tmp_path):
    _cfg, store, _events, assistant, _sink = _assistant(tmp_path)
    codes: list[int] = []
    assistant._halt = codes.append
    assistant._loops_started = True
    assistant._stop.clear()
    assistant.shutdown.clear()
    assistant._stall_s = 30
    assistant._in_session = False
    assistant._wake_at = time.monotonic()
    assistant.capture.running = True
    assistant.capture.last_audio_at = time.monotonic() - 31
    assistant._watchdog_once()
    assert codes == [1]
    audio = store.recent("error", 1)[0]
    assert audio["level"] == "error"
    assert audio["error_code"] == "audio.stalled"
    assert "RuntimeError" in audio["message"]
    assert "audio loop stalled" in audio["message"]

    codes.clear()
    assistant.shutdown.clear()
    assistant._stop.clear()
    assistant.capture.running = False
    assistant._wake_at = time.monotonic() - 31
    assistant._watchdog_once()
    assert codes == [1]
    wake = store.recent("error", 1)[0]
    assert wake["error_code"] == "internal.stalled"
    assert "wake loop stalled" in wake["message"]


def test_heartbeat_does_not_wait_for_the_wake_loop(tmp_path):
    cfg, store, _events, assistant, _sink = _assistant(tmp_path)
    cfg.telemetry.heartbeat_seconds = 30
    assistant._wake_at = time.monotonic() - 100
    assistant._stop.clear()
    thread = threading.Thread(target=assistant._heartbeat_loop)
    thread.start()
    try:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and not store.recent("heartbeat", 1):
            time.sleep(0.05)
    finally:
        assistant._stop.set()
        thread.join(timeout=2)
    beat = store.recent("heartbeat", 1)[0]
    assert beat["type"] == "heartbeat"
    assert beat["cpu_pct"] <= 100


def test_server_clear_does_not_cut_queued_audio():
    protocol = RealtimeProtocol(Config(), None, None, None, _HoldSink())
    protocol.enqueue_pcm(b"\x01\x00" * 20)
    protocol._response_id = "current"
    protocol.on_event({"type": "output_audio_buffer.cleared"})
    assert protocol.playback.qsize() == 1
    assert protocol.sink.stopped is False
    protocol._cancel_pending = True
    protocol.on_event({"type": "output_audio_buffer.cleared"})
    assert protocol.playback.empty()


def test_follow_up_starts_when_playback_finishes_not_when_queued(tmp_path):
    cfg, _store, events, _assistant_obj, sink = _assistant(tmp_path)
    cfg.conversation.realtime.listen_after_playback_s = 8
    protocol = RealtimeProtocol(cfg, events, _assistant_obj.store, None, sink)
    protocol.last_activity = 0
    pcm = b"\x02\x00" * 2400
    protocol.on_event({"type": "response.created", "response": {"id": "bridge-1"}})
    protocol.on_event(
        {"type": "response.output_audio.delta", "delta": base64.b64encode(pcm).decode("ascii"), "response_id": "bridge-1"}
    )
    protocol.on_event({"type": "response.done", "response": {"id": "bridge-1", "status": "completed"}})
    # Queued, and still sitting in the speaker: the follow-up window is not open yet.
    assert protocol.listen_until == 0
    assert idle_should_close(protocol, None, time.monotonic(), 0.01) is False
    queued = protocol.playback.get_nowait()
    protocol.sink.write(queued)
    assert protocol.sink.queued_frames() > 0
    assert idle_should_close(protocol, None, time.monotonic(), 0.01) is False
    protocol.mic_enabled = False
    protocol.sink.drain()
    protocol.note_output_idle()
    remaining = protocol.listen_until - time.monotonic()
    assert 7.5 <= remaining <= 8.5
    assert protocol.mic_enabled is True
    assert protocol.sink.stopped is False
    assert idle_should_close(protocol, None, protocol.listen_until - 0.2, 0.01) is False
    assert idle_should_close(protocol, None, protocol.listen_until + 0.05, 0.01) is True


def test_underrun_is_logged_with_a_count(tmp_path):
    _cfg, store, _events, assistant, sink = _assistant(tmp_path)
    protocol = RealtimeProtocol(assistant.cfg, assistant.events, store, None, sink)
    sink.underruns = 4
    assistant._emit_underruns(protocol)
    event = store.recent("playback.underrun", 1)[0]
    assert event["level"] == "warning"
    assert event["count"] == 4
    assert event["underruns"] == 4
    assistant._emit_underruns(protocol)
    assert len(store.recent("playback.underrun", 5)) == 1


def test_local_clip_joins_the_player_only_while_it_is_live():
    """A chime or bridge clip must not write the device beside the player, or vanish into a queue nobody drains."""
    sink = _HoldSink()
    protocol = RealtimeProtocol(Config(), None, None, None, sink)
    pcm = b"\x03\x00" * 40
    rate = protocol.cfg.conversation.realtime.sample_rate
    play_pcm(sink, pcm, rate, protocol)
    assert sink.chunks == [pcm]
    assert protocol.playback.empty()
    sink.chunks.clear()
    protocol.player_live = True
    play_pcm(sink, pcm, rate, protocol)
    assert sink.chunks == []
    assert protocol.playback.get_nowait() == pcm
    protocol.close = True
    play_pcm(sink, pcm, rate, protocol)
    assert sink.chunks == [pcm]


def test_jitter_refill_holds_a_short_chunk():
    buffer = JitterBuffer(prebuffer_frames=80, channels=1)
    buffer.push(np.ones(80, dtype=np.int16).tobytes())
    buffer.pull(80)
    assert buffer.underruns == 0
    buffer.pull(8)
    assert buffer.underruns == 1
    buffer.push(np.ones(20, dtype=np.int16).tobytes())
    assert int(buffer.pull(8).sum()) == 0
    buffer.push(np.ones(60, dtype=np.int16).tobytes())
    assert int(buffer.pull(8).sum()) > 0
