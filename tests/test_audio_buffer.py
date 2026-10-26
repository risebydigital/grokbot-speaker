"""Device buffer size. A 60 ms ALSA buffer underruns when the callback is a little late."""

import random

import numpy as np
from typer.testing import CliRunner

from gva.cli import app
from gva.config import load_config, validate_change
from gva.paths import project_root
from gva.runtime.audio import (
    JitterBuffer,
    SoundDeviceCapture,
    SoundDeviceSink,
    capture_period,
    render_audio_stats,
    scale_blocksize,
)
from gva.telemetry.events import EventLog
from gva.telemetry.store import Store


def test_output_period_is_50ms_at_48k_and_capture_keeps_a_wake_chunk():
    assert scale_blocksize(2400, 48000) == 2400
    assert scale_blocksize(2400, 16000) == 800
    # 80 ms at 48 kHz is 3840 frames, which is already larger than the 50 ms period.
    assert capture_period(80, 2400, 48000) == 3840
    sink = SoundDeviceSink(None, 48000)
    assert sink.latency == 0.2
    assert sink.blocksize == 2400
    capture = SoundDeviceCapture(None, 48000, 80)
    assert capture.latency == 0.2
    # One 80 ms wake chunk is 3840 frames, so capture does not shrink to 50 ms.
    assert capture.blocksize == 3840


def test_latency_and_blocksize_are_settings():
    assert validate_change("audio.output_latency", "0.2") == 0.2
    assert validate_change("audio.output_latency", "high") == "high"
    assert validate_change("audio.output_blocksize", "2400") == 2400
    assert validate_change("audio.capture_latency", "0.25") == 0.25
    try:
        validate_change("audio.output_blocksize", "32")
    except Exception as exc:
        assert "between" in str(exc)
    else:
        raise AssertionError("a tiny blocksize was accepted")


def test_callback_copies_out_of_the_preallocated_ring():
    ring = JitterBuffer(prebuffer_frames=8, channels=1)
    storage = ring._storage
    ring.push(np.full(8, 7, dtype=np.int16).tobytes())
    out = np.zeros((4, 1), dtype=np.int16)
    ring.pull_into(out)
    assert ring._storage is storage
    assert int(out.sum()) == 28
    assert ring.buffered() == 4


def _soak(
    *,
    rate: int,
    seconds: float,
    chunk_ms: int,
    prebuffer_ms: int,
    blocksize: int,
    device_buffer_frames: int,
    arrival_jitter_s: float,
    stall_s: float,
    stall_every: int,
    seed: int,
) -> tuple[int, int]:
    """Play `seconds` of chunked speech. Return (hardware underruns, software underruns).

    Chunks are ready about `prebuffer_ms` ahead of playback, then jittered later.
    The callback is usually one period, and every `stall_every` periods it is
    `stall_s` late. A stall longer than the device buffer is a hardware underrun
    even when the software ring still has samples.
    """
    rng = random.Random(seed)
    prebuffer_frames = int(rate * prebuffer_ms / 1000)
    ring = JitterBuffer(prebuffer_frames, channels=1)
    total = int(rate * seconds)
    chunk_frames = max(1, int(rate * chunk_ms / 1000))
    lead = prebuffer_frames / rate
    arrivals: list[tuple[float, int]] = []
    cursor = 0
    while cursor < total:
        count = min(chunk_frames, total - cursor)
        when = max(0.0, cursor / rate - lead + rng.uniform(0.0, arrival_jitter_s))
        arrivals.append((when, count))
        cursor += count
    arrivals.sort()

    callbacks: list[float] = []
    moment = 0.0
    index = 0
    while moment < seconds:
        if stall_every and index and index % stall_every == 0:
            moment += stall_s
        if moment < seconds:
            callbacks.append(moment)
        index += 1
        moment += blocksize / rate

    events: list[tuple[float, int, int]] = []
    events.extend((when, 0, count) for when, count in arrivals)
    events.extend((when, 1, 0) for when in callbacks)
    events.sort()

    hardware = 0
    last = 0.0
    started = False
    pushed = 0
    block = np.zeros((blocksize, 1), dtype=np.int16)
    tone = np.full(chunk_frames, 1000, dtype=np.int16)
    for when, kind, count in events:
        if kind == 0:
            ring.push(tone[:count].tobytes())
            pushed += count
            continue
        # Idle callbacks after the utterance are not gaps in the sentence.
        if pushed >= total and ring.buffered() == 0:
            continue
        gap_frames = int(round((when - last) * rate))
        if started and gap_frames > device_buffer_frames:
            hardware += 1
        need = max(blocksize, gap_frames)
        while need > 0 and not (pushed >= total and ring.buffered() == 0):
            ring.pull_into(block)
            need -= blocksize
        started = True
        last = when
    return hardware, ring.underruns


def test_long_speech_with_jitter_does_not_underrun():
    """60 s of 100 ms chunks. A 100 ms callback stall fits in 200 ms and not in 60 ms."""
    common = dict(
        rate=48000,
        seconds=60.0,
        chunk_ms=100,
        prebuffer_ms=250,
        arrival_jitter_s=0.04,
        stall_s=0.10,
        stall_every=40,
        seed=12,
    )
    hardware, software = _soak(blocksize=2400, device_buffer_frames=int(0.2 * 48000), **common)
    assert hardware == 0
    assert software == 0
    thin_hardware, _thin_software = _soak(blocksize=960, device_buffer_frames=2880, **common)
    assert thin_hardware > 0


def test_audio_stats_reports_the_opened_buffer_and_underruns(monkeypatch, tmp_path):
    monkeypatch.setenv("GVA_ENV_FILE", str(tmp_path / "missing.env"))
    monkeypatch.setenv("GVA_CONFIG", str(project_root() / "config" / "config.example.yaml"))
    monkeypatch.setenv("GVA_LOG_DIR", str(tmp_path / "log"))
    monkeypatch.setenv("GVA_DB_PATH", str(tmp_path / "metrics.db"))
    store = Store(tmp_path / "metrics.db")
    events = EventLog("milo-01", "dev", tmp_path / "log", store, validate=True)
    events.emit("playback.underrun", level="warning", count=2, underruns=2)
    events.emit("playback.underrun", level="warning", count=1, underruns=1)
    assert store.underrun_totals() == (2, 3)

    def fake_probe(_cfg):
        return (
            {
                "opened": True,
                "rate": 48000,
                "channels": 2,
                "blocksize": 2400,
                "latency_s": 0.2,
                "buffer_frames": 9600,
                "buffer_ms": 200,
                "prebuffer_ms": 250,
            },
            "",
            {
                "opened": True,
                "rate": 48000,
                "channels": 1,
                "blocksize": 3840,
                "latency_s": 0.2,
                "buffer_frames": 9600,
                "buffer_ms": 200,
            },
            "",
        )

    monkeypatch.setattr("gva.runtime.audio.probe_opened_streams", fake_probe)
    result = CliRunner().invoke(app, ["audio", "stats"])
    assert result.exit_code == 0, result.output
    text = result.output
    assert "48000" in text
    assert "2400" in text
    assert "9600" in text
    assert "0.200 s" in text
    assert "playback.underrun events 2" in text
    assert "underruns 3" in text
    cfg = load_config()
    rendered = render_audio_stats(cfg, {}, "", {}, "sounddevice is not available", 0, 0)
    assert "not opened" in rendered
    assert "xai-" not in rendered
