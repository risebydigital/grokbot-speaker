"""The approved chime: length, rate, peak, and a click-free resample."""

import hashlib

import numpy as np
from typer.testing import CliRunner

from gva.cli import app
from gva.config import Config, load_config, validate_change
from gva.paths import project_root
from gva.runtime.audio import CollectingSink, SoundDeviceSink
from gva.runtime.bridge import BotBridge
from gva.runtime.chime import B4, CANONICAL_RATE, D5, DURATION_S, G4, PEAK, chime_pcm, play_chime
from gva.runtime.heads_up import MeetingHeadsUp
from gva.telemetry.events import EventLog
from gva.telemetry.store import Store


def _samples(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype=np.int16)


def _band(samples: np.ndarray, rate: int, freq: float, width: float = 6.0) -> float:
    window = np.hanning(samples.size)
    spec = np.abs(np.fft.rfft(samples.astype(np.float64) * window))
    freqs = np.fft.rfftfreq(samples.size, 1.0 / rate)
    mask = np.abs(freqs - freq) <= width
    if not np.any(mask):
        return 0.0
    return float(spec[mask].max())


# Bytes from the approved generator: astype(int16) of the scaled float buffer.
_APPROVED_SHA256 = "e6b9df26a0f1861af7712b580b86ef14ab33011d9a7f8b9c26d45289031f6ad4"


def test_canonical_chime_length_rate_peak_and_no_clip():
    pcm = chime_pcm(CANONICAL_RATE, 1.0)
    audio = _samples(pcm)
    assert audio.dtype == np.int16
    assert len(audio) == int(DURATION_S * CANONICAL_RATE)
    assert len(audio) == 52920
    peak = int(np.max(np.abs(audio)))
    assert peak == PEAK
    assert peak < 32767
    assert int(np.max(audio)) <= PEAK
    assert int(np.min(audio)) >= -PEAK
    assert audio[0] == 0
    assert audio[-1] == 0
    assert hashlib.sha256(pcm).hexdigest() == _APPROVED_SHA256


def test_chime_is_g4_then_d5_landing_on_b4_and_g4():
    audio = _samples(chime_pcm())
    quiet = _band(audio, CANONICAL_RATE, 2000.0)
    g4 = _band(audio, CANONICAL_RATE, G4)
    d5 = _band(audio, CANONICAL_RATE, D5)
    b4 = _band(audio, CANONICAL_RATE, B4)
    assert g4 > quiet * 15
    assert d5 > quiet * 8
    assert b4 > quiet * 8
    # The second harmonic is present and quieter than the fundamental.
    assert 0 < _band(audio, CANONICAL_RATE, G4 * 2) < g4 * 0.25


def test_volume_lowers_the_peak_without_clipping():
    full = _samples(chime_pcm(CANONICAL_RATE, 1))
    half = _samples(chime_pcm(CANONICAL_RATE, 0.5))
    mute = _samples(chime_pcm(CANONICAL_RATE, 0))
    assert int(np.max(np.abs(full))) == PEAK
    assert int(np.max(np.abs(half))) == PEAK // 2
    assert int(np.max(np.abs(mute))) == 0
    assert len(half) == len(full)


def test_resample_to_device_rates_stays_full_and_starts_from_silence():
    for rate in (16000, 24000, 48000):
        audio = _samples(chime_pcm(rate, 1.0))
        assert int(np.max(np.abs(audio))) == PEAK
        assert int(np.min(audio)) >= -PEAK
        assert int(np.max(audio)) <= PEAK
        assert audio[0] == 0
        assert audio[-1] == 0
        assert abs(int(audio[1])) < 400
        assert abs(int(audio[-2])) < 400
        seconds = len(audio) / rate
        assert abs(seconds - DURATION_S) < 0.03


def test_device_sink_gets_a_raw_device_rate_buffer():
    sink = SoundDeviceSink(None, 24000)
    sink.device_rate = 48000
    sink.channels = 1
    sink._format_ready = True
    sink._ensure = lambda: None
    written = {}

    def write_raw(pcm: bytes) -> None:
        written["pcm"] = pcm
        SoundDeviceSink.write_raw(sink, pcm)

    sink.write_raw = write_raw
    pcm = play_chime(sink, Config())
    assert pcm == written["pcm"]
    assert len(_samples(pcm)) == len(_samples(chime_pcm(48000)))
    assert sink.buffer.buffered() == len(pcm) // 2
    assert sink._source_bytes == 0


def test_a_live_player_queues_the_approved_chime():
    from gva.runtime.realtime import RealtimeProtocol

    sink = CollectingSink()
    sink.sample_rate = 24000
    protocol = RealtimeProtocol(Config(), None, None, None, sink)
    protocol.player_live = True
    pcm = play_chime(sink, Config(), protocol=protocol)
    assert sink.chunks == []
    assert protocol.playback.get_nowait() == pcm == chime_pcm(24000, 1.0)
    protocol.close = True
    direct = play_chime(sink, Config(), protocol=protocol)
    assert b"".join(sink.chunks) == direct == chime_pcm(24000, 1.0)


def test_heads_up_and_bridge_replace_the_old_beeps(tmp_path):
    cfg = Config()
    cfg.telemetry.log_dir = str(tmp_path / "log")
    cfg.telemetry.db_path = str(tmp_path / "metrics.db")
    store = Store(tmp_path / "metrics.db")
    events = EventLog("milo-01", "dev", tmp_path / "log", store, validate=True)
    sink = CollectingSink()
    heads = MeetingHeadsUp(cfg, events, sink=sink)
    heads._chime()
    heard = b"".join(sink.chunks)
    assert heads.last_chime == "soft"
    assert heard == chime_pcm(CANONICAL_RATE, 1.0)

    bridge_sink = CollectingSink()
    bridge = BotBridge(cfg, events, sink=bridge_sink)
    bridge._chime(soft=True)
    soft = b"".join(bridge_sink.chunks)
    bridge_sink.chunks.clear()
    bridge._chime(soft=False)
    loud = b"".join(bridge_sink.chunks)
    assert bridge.last_chime == "loud"
    assert soft == loud == heard

    cfg.audio.chime_volume = 0.5
    bridge_sink.chunks.clear()
    bridge._chime(soft=True)
    assert int(np.max(np.abs(_samples(b"".join(bridge_sink.chunks))))) == PEAK // 2


def test_chime_volume_defaults_to_full_and_is_settable():
    cfg = load_config(project_root() / "config" / "config.example.yaml")
    assert cfg.audio.chime_volume == 1.0
    assert validate_change("audio.chime_volume", "1") == 1.0
    assert validate_change("audio.chime_volume", "0.4") == 0.4
    try:
        validate_change("audio.chime_volume", "1.5")
    except Exception as exc:
        assert "between" in str(exc)
    else:
        raise AssertionError("volume above 1 was accepted")


def test_chime_test_command_plays_once(monkeypatch):
    played = {}

    class FakeSink:
        device_rate = 48000
        channels = 2
        sample_rate = 24000

        def prepare(self):
            played["prepared"] = True

        def resume(self):
            played["resumed"] = True

        def write_raw(self, pcm: bytes):
            played["pcm"] = pcm

        def close(self):
            played["closed"] = True

    monkeypatch.setattr(
        "gva.runtime.audio.list_devices",
        lambda: [{"index": 3, "name": "Sanyun", "inputs": 0, "outputs": 2}],
    )
    monkeypatch.setattr("gva.runtime.audio.prefer_output", lambda rows, name: 3)
    monkeypatch.setattr("gva.runtime.audio.SoundDeviceSink", lambda *args, **kwargs: FakeSink())
    monkeypatch.setattr("gva.cli.time.sleep", lambda seconds: played.setdefault("sleep", seconds))

    result = CliRunner().invoke(app, ["chime", "test"])
    assert result.exit_code == 0, result.output
    assert "Sanyun" in result.output
    assert "48000" in result.output
    assert played["prepared"] is True
    assert played["closed"] is True
    assert played["sleep"] >= DURATION_S
    audio = _samples(played["pcm"])
    assert int(np.max(np.abs(audio))) == PEAK
    assert abs(len(audio) / 48000 - DURATION_S) < 0.03


def test_chime_test_command_when_there_is_no_speaker(monkeypatch):
    monkeypatch.setattr("gva.runtime.audio.list_devices", lambda: [{"index": 0, "name": "mic", "inputs": 1, "outputs": 0}])
    monkeypatch.setattr("gva.runtime.audio.prefer_output", lambda rows, name: None)
    result = CliRunner().invoke(app, ["chime", "test"])
    assert result.exit_code == 1
    assert "no playback device" in result.output
