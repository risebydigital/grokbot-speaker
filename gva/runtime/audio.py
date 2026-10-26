"""Audio capture, playback and device selection. 16-bit mono, several sample rates."""

from __future__ import annotations

import math
import queue
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

# Spoken when a realtime turn produces no audio. The recording is the same sentence.
FALLBACK_PHRASE = "Sorry, I couldn't get that. My calendar and reports aren't connected yet."


def chunk_samples(sample_rate: int, chunk_ms: int) -> int:
    return int(sample_rate * chunk_ms / 1000)


def chunk_bytes(sample_rate: int, chunk_ms: int) -> int:
    return chunk_samples(sample_rate, chunk_ms) * 2


def silence(ms: int, sample_rate: int = 16000) -> bytes:
    count = int(sample_rate * ms / 1000)
    return np.zeros(count, dtype=np.int16).tobytes()


def tone(ms: int, freq: float = 440, amp: int = 8000, sample_rate: int = 16000) -> bytes:
    count = max(1, int(sample_rate * ms / 1000))
    t = np.arange(count, dtype=np.float64) / sample_rate
    samples = (amp * np.sin(2 * np.pi * freq * t)).astype(np.int16)
    return samples.tobytes()


def rms(pcm: bytes) -> float:
    if len(pcm) < 2:
        return 0.0
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(np.square(audio))))


def resample_pcm(pcm: bytes, from_rate: int, to_rate: int) -> bytes:
    """Change sample rate. scipy when it is installed, otherwise linear interpolation."""
    if not pcm or from_rate == to_rate:
        return pcm
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
    count = max(1, int(round(len(audio) * to_rate / from_rate)))
    try:
        import scipy.signal

        converted = scipy.signal.resample(audio, count)
    except Exception:
        if len(audio) == 1:
            converted = np.repeat(audio, count)
        else:
            old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
            new = np.linspace(0.0, 1.0, num=count, endpoint=False)
            converted = np.interp(new, old, audio)
    return np.clip(np.round(converted), -32768, 32767).astype(np.int16).tobytes()


def _interpolation_taps(up: int, down: int) -> np.ndarray:
    """Windowed-sinc lowpass. Gain is `up` so a zero-stuffed signal keeps its level."""
    ratio = max(int(up), int(down))
    half = 10 * ratio
    index = np.arange(-half, half + 1, dtype=np.float64)
    taps = np.sinc(index / ratio) * np.hanning(index.size)
    total = float(np.sum(taps))
    if total:
        taps *= up / total
    return taps


def _float_to_pcm(samples: np.ndarray) -> bytes:
    if samples.size == 0:
        return b""
    return np.clip(np.round(samples), -32768, 32767).astype(np.int16).tobytes()


class StreamingResampler:
    """Rational resampler that keeps its FIR delay line across chunks.

    Realtime deltas are short 24 kHz mono buffers. Resampling each one on its
    own rings at the join, which is the choppy playback on a 48 kHz speaker.
    24000 to 48000 is an exact 2x upsample (up=2, down=1). The same filter,
    fed the whole utterance in one go, matches chunked output sample for sample.
    numpy only, so the Pi does not need an extra native library.
    """

    def __init__(self, from_rate: int, to_rate: int):
        self.from_rate = int(from_rate)
        self.to_rate = int(to_rate)
        if self.from_rate <= 0 or self.to_rate <= 0:
            raise ValueError("sample rates must be positive")
        divisor = math.gcd(self.from_rate, self.to_rate) or 1
        self.up = self.to_rate // divisor
        self.down = self.from_rate // divisor
        self._same = self.up == 1 and self.down == 1
        self._taps = np.zeros(0, dtype=np.float64) if self._same else _interpolation_taps(self.up, self.down)
        self.reset()

    def reset(self) -> None:
        width = 0 if self._taps.size == 0 else int(self._taps.size) - 1
        self._state = np.zeros(width, dtype=np.float64)
        self._phase = 0

    def process(self, pcm: bytes) -> bytes:
        if not pcm or self._same:
            return pcm
        usable = len(pcm) - (len(pcm) % 2)
        if usable < 2:
            return b""
        samples = np.frombuffer(pcm[:usable], dtype=np.int16).astype(np.float64)
        stuffed = np.zeros(samples.size * self.up, dtype=np.float64)
        stuffed[:: self.up] = samples
        extended = np.concatenate([self._state, stuffed])
        filtered = np.convolve(extended, self._taps, mode="valid")
        keep = int(self._taps.size) - 1
        self._state = extended[-keep:].copy() if keep else np.zeros(0, dtype=np.float64)
        produced = filtered[self._phase :: self.down]
        self._phase = (self._phase + int(filtered.size)) % self.down
        return _float_to_pcm(produced)


def upmix(mono: bytes, channels: int) -> bytes:
    """Duplicate mono PCM into interleaved device channels."""
    count = max(1, int(channels))
    if count == 1 or len(mono) < 2:
        return mono
    samples = np.frombuffer(mono, dtype=np.int16)
    return np.repeat(samples, count).astype(np.int16, copy=False).tobytes()


def choose_stream_format(
    source_rate: int,
    default_rate: float | int,
    max_channels: int,
    supported,
) -> tuple[int, int]:
    """Pick a rate and channel count the device will open.

    Try the source rate first, then the device default (often 48000 on the
    Sanyun SW208). At each rate try mono, then stereo. That speaker only
    accepts S16_LE, 2 channels, 48000 Hz.
    """
    rates: list[int] = []
    for rate in (int(source_rate), int(round(float(default_rate or 0)))):
        if rate > 0 and rate not in rates:
            rates.append(rate)
    if not rates:
        rates = [int(source_rate) or 16000]
    channels = [count for count in (1, 2) if count <= max(1, int(max_channels or 1))]
    if not channels:
        channels = [1]
    for rate in rates:
        for count in channels:
            if supported(rate, count):
                return rate, count
    fallback = 2 if int(max_channels or 0) >= 2 else 1
    return rates[-1], fallback


def fit_playback(pcm: bytes, from_rate: int, to_rate: int, channels: int) -> bytes:
    """Resample mono PCM and, when the device needs it, duplicate it to stereo."""
    mono = resample_pcm(pcm, from_rate, to_rate)
    count = max(1, int(channels))
    if count == 1 or len(mono) < 2:
        return mono
    samples = np.frombuffer(mono, dtype=np.int16)
    interleaved = np.repeat(samples, count)
    return interleaved.astype(np.int16, copy=False).tobytes()


def fit_capture(indata: np.ndarray, from_rate: int, to_rate: int) -> bytes:
    """Downmix device audio to mono and resample to the rate the pipeline expects."""
    audio = np.asarray(indata)
    if audio.ndim == 2 and audio.shape[1] > 1:
        mono = audio.astype(np.float64).mean(axis=1)
    elif audio.ndim == 2:
        mono = audio[:, 0]
    else:
        mono = audio.reshape(-1)
    pcm = np.clip(np.round(mono), -32768, 32767).astype(np.int16).tobytes()
    return resample_pcm(pcm, from_rate, to_rate)


def _query_device(sd, device: int | None, kind: str) -> dict[str, Any]:
    if device is None:
        info = sd.query_devices(kind=kind)
    else:
        info = sd.query_devices(device)
    if not isinstance(info, dict):
        raise RuntimeError(f"no {kind} device")
    return info


def _supported(check, device: int | None, rate: int, channels: int) -> bool:
    try:
        check(device=device, channels=channels, dtype="int16", samplerate=rate)
        return True
    except Exception:
        return False


def probe_output(device: int | None, source_rate: int) -> tuple[int, int]:
    import sounddevice as sd

    info = _query_device(sd, device, "output")
    return choose_stream_format(
        source_rate,
        info.get("default_samplerate") or source_rate,
        int(info.get("max_output_channels") or 1),
        lambda rate, channels: _supported(sd.check_output_settings, device, rate, channels),
    )


def probe_input(device: int | None, source_rate: int) -> tuple[int, int]:
    import sounddevice as sd

    info = _query_device(sd, device, "input")
    return choose_stream_format(
        source_rate,
        info.get("default_samplerate") or source_rate,
        int(info.get("max_input_channels") or 1),
        lambda rate, channels: _supported(sd.check_input_settings, device, rate, channels),
    )


class JitterBuffer:
    """Holds PCM until a short pre-buffer is full, then plays it as one stream.

    Calling sounddevice.play for every realtime audio delta restarts the device
    and sounds chopped. This buffer is what the output callback reads instead.
    """

    def __init__(self, prebuffer_frames: int = 0, channels: int = 1):
        self.prebuffer_frames = max(0, int(prebuffer_frames))
        self.channels = max(1, int(channels))
        self._samples = np.zeros((0, self.channels), dtype=np.int16)
        self._primed = self.prebuffer_frames == 0
        self._fed = False
        self.underruns = 0
        self._lock = threading.Lock()

    def push(self, pcm: bytes) -> None:
        """Push interleaved device-format PCM. `channels` matches the open stream."""
        if len(pcm) < 2:
            return
        flat = np.frombuffer(pcm, dtype=np.int16).copy()
        usable = (flat.size // self.channels) * self.channels
        if usable < self.channels:
            return
        frames = flat[:usable].reshape(-1, self.channels)
        with self._lock:
            self._samples = np.concatenate([self._samples, frames], axis=0)
            self._fed = True

    def pull(self, frames: int) -> np.ndarray:
        """Return shape (frames, channels). Shortfalls are silence, not a stream restart.

        Silence before the pre-buffer fills is not an underrun. Once the buffer
        has drained, further idle callbacks stay silent and are not counted again.
        """
        out = np.zeros((frames, self.channels), dtype=np.int16)
        with self._lock:
            if not self._primed:
                if len(self._samples) < self.prebuffer_frames:
                    return out
                self._primed = True
            take = min(frames, len(self._samples))
            if take:
                out[:take] = self._samples[:take]
                self._samples = self._samples[take:]
            if take < frames and self._fed:
                # A short final block is the tail of the reply. Counting that and
                # the idle callback after it reported two underruns on every turn.
                # Only a callback that receives nothing, while audio was still
                # expected, is a gap.
                if take == 0:
                    self.underruns += 1
                self._fed = False
        return out

    def clear(self) -> None:
        with self._lock:
            self._samples = np.zeros((0, self.channels), dtype=np.int16)
            self._primed = self.prebuffer_frames == 0
            self._fed = False

    def buffered(self) -> int:
        with self._lock:
            return int(len(self._samples))


def list_devices() -> list[dict[str, Any]]:
    """List ALSA devices via sounddevice. Empty if the library or a device is missing."""
    try:
        import sounddevice as sd
    except Exception as exc:  # ImportError or PortAudio not installed
        return [{"error": f"sounddevice is not available ({exc})"}]
    try:
        host = sd.query_devices()
    except Exception as exc:
        return [{"error": str(exc)}]
    default_in, default_out = sd.default.device
    rows = []
    for index, dev in enumerate(host):
        rows.append(
            {
                "index": index,
                "name": dev["name"],
                "inputs": int(dev["max_input_channels"]),
                "outputs": int(dev["max_output_channels"]),
                "default_input": index == default_in,
                "default_output": index == default_out,
                "sample_rate": dev.get("default_samplerate"),
            }
        )
    return rows


def _can_play(dev: dict[str, Any]) -> bool:
    name = str(dev.get("name", "")).lower()
    if dev.get("error") or int(dev.get("outputs") or 0) <= 0:
        return False
    # PipeWire and Pulse need a user session. The service opens ALSA hw devices.
    if any(token in name for token in ("airhug", "bluetooth", "bluez", "pipewire", "pulse")):
        return False
    return True


def prefer_output(devices: list[dict[str, Any]], configured: str = "") -> int | None:
    """Pick a speaker. A configured substring wins. Otherwise USB playback, then the headphone jack.

    The AIRHUG microphone is ignored. Bluetooth is not on the default path.
    """
    playable = [dev for dev in devices if _can_play(dev)]
    needle = (configured or "").strip().lower()
    if needle and needle != "auto":
        for dev in playable:
            if needle in str(dev.get("name", "")).lower():
                return int(dev["index"])
        return None
    for dev in playable:
        if "usb" in str(dev.get("name", "")).lower():
            return int(dev["index"])
    for dev in playable:
        name = str(dev.get("name", "")).lower()
        if "headphone" in name or "bcm2835" in name or "analog" in name:
            return int(dev["index"])
    if playable:
        return int(playable[0]["index"])
    return None


def resolve_device(substring: str, kind: str) -> int | None:
    if not substring:
        return None
    needle = substring.lower()
    for dev in list_devices():
        name = str(dev.get("name", "")).lower()
        if needle not in name:
            continue
        if kind == "input" and dev.get("inputs", 0) > 0:
            return int(dev["index"])
        if kind == "output" and dev.get("outputs", 0) > 0:
            return int(dev["index"])
    return None


class SoundDeviceCapture:
    """Dedicated capture thread. A full queue drops the oldest frame rather than blocking audio."""

    def __init__(self, device: int | None, sample_rate: int, chunk_ms: int, on_lost=None):
        self.device = device
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms
        self.on_lost = on_lost
        self.device_rate = sample_rate
        self.device_channels = 1
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=32)
        self._stream = None
        self._lost = threading.Event()

    def start(self) -> None:
        import sounddevice as sd

        self.device_rate, self.device_channels = probe_input(self.device, self.sample_rate)
        block = chunk_samples(self.device_rate, self.chunk_ms)
        self._stream = sd.InputStream(
            device=self.device,
            channels=self.device_channels,
            samplerate=self.device_rate,
            dtype="int16",
            blocksize=block,
            callback=self._on_audio,
        )
        self._stream.start()

    def _on_audio(self, indata, frames, time_info, status) -> None:
        if status:
            self._lost.set()
        pcm = fit_capture(indata, self.device_rate, self.sample_rate)
        try:
            self._queue.put_nowait(pcm)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(pcm)
            except queue.Full:
                pass

    def read(self, timeout: float = 0.2, nowait: bool = False) -> bytes | None:
        if self._lost.is_set():
            self._lost.clear()
            if self.on_lost:
                self.on_lost()
            self._reopen()
        try:
            if nowait:
                return self._queue.get_nowait()
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return b""

    def _reopen(self) -> None:
        time.sleep(5)
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
            self.start()
        except Exception:
            self._lost.set()

    def flush(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class SoundDeviceSink:
    """One callback OutputStream for the process. Chunks share a resampler and a jitter buffer."""

    def __init__(
        self,
        device: int | None,
        sample_rate: int,
        prebuffer_ms: int = 200,
        latency: str = "high",
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.device_rate = sample_rate
        self.channels = 1
        self.prebuffer_ms = int(prebuffer_ms)
        self.latency = latency or "high"
        self.buffer = JitterBuffer(int(sample_rate * self.prebuffer_ms / 1000), channels=1)
        self._resampler = StreamingResampler(sample_rate, sample_rate)
        self._source_bytes = 0
        self._stopped = False
        self._stream = None
        self._format_ready = False

    def prepare(self) -> None:
        """Choose the device rate and channel count before the first sample."""
        if self._format_ready:
            return
        self.device_rate, self.channels = probe_output(self.device, self.sample_rate)
        self.buffer = JitterBuffer(int(self.device_rate * self.prebuffer_ms / 1000), self.channels)
        self._resampler = StreamingResampler(self.sample_rate, self.device_rate)
        self._format_ready = True

    def write(self, pcm: bytes) -> None:
        if self._stopped or not pcm:
            return
        self.prepare()
        self._ensure()
        if self._resampler.from_rate != self.sample_rate or self._resampler.to_rate != self.device_rate:
            self._resampler = StreamingResampler(self.sample_rate, self.device_rate)
        self._source_bytes += len(pcm) - (len(pcm) % 2)
        fitted = upmix(self._resampler.process(pcm), self.channels)
        if fitted:
            self.buffer.push(fitted)

    def write_raw(self, pcm: bytes) -> None:
        """Queue mono PCM that is already at ``device_rate``.

        ``write`` resamples from the source rate. A chime is resampled on its
        own, including the filter tail, and must not pass through that again.
        """
        if self._stopped or not pcm:
            return
        self.prepare()
        self._ensure()
        usable = len(pcm) - (len(pcm) % 2)
        if usable < 2:
            return
        fitted = upmix(pcm[:usable], self.channels)
        if fitted:
            self.buffer.push(fitted)

    def audio_ms(self) -> int:
        rate = self.sample_rate or 1
        return int(self._source_bytes / 2 / rate * 1000)

    def consume_audio_ms(self) -> int:
        heard = self.audio_ms()
        self._source_bytes = 0
        return heard

    def consume_underruns(self) -> int:
        count = int(self.buffer.underruns)
        self.buffer.underruns = 0
        return count

    def queued_frames(self) -> int:
        """Frames still waiting in the jitter buffer. Playback is not finished while this is above zero."""
        return int(self.buffer.buffered())

    def stop(self) -> None:
        """Drop queued speech. The stream stays open; the next write does not reopen it."""
        self._stopped = True
        self.buffer.clear()
        self._resampler.reset()

    def resume(self) -> None:
        self._stopped = False

    def close(self) -> None:
        self.buffer.clear()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _ensure(self) -> None:
        if self._stream is not None:
            return
        import sounddevice as sd

        self.prepare()
        # About 20 ms. latency='high' lets PortAudio pick a USB-friendly buffer.
        block = max(256, int(self.device_rate * 0.02))

        def callback(outdata, frames, time_info, status) -> None:
            outdata[:] = self.buffer.pull(frames)

        self._stream = sd.OutputStream(
            device=self.device,
            channels=self.channels,
            samplerate=self.device_rate,
            dtype="int16",
            blocksize=block,
            latency=self.latency,
            callback=callback,
        )
        self._stream.start()


def fallback_wav_path() -> Path:
    return Path(__file__).resolve().parents[1] / "assets" / "fallback-not-connected.wav"


def load_fallback_pcm() -> tuple[bytes, int]:
    """Bundled spoken fallback. Mono PCM16. No network and no Piper voice required."""
    import wave

    path = fallback_wav_path()
    with wave.open(str(path), "rb") as handle:
        rate = int(handle.getframerate())
        channels = int(handle.getnchannels())
        pcm = handle.readframes(handle.getnframes())
    if channels > 1 and pcm:
        audio = np.frombuffer(pcm, dtype=np.int16).reshape(-1, channels).astype(np.float64).mean(axis=1)
        pcm = np.clip(np.round(audio), -32768, 32767).astype(np.int16).tobytes()
    return pcm, rate


def play_fallback_phrase(sink, cfg, phrase: str = FALLBACK_PHRASE) -> int:
    """Play the bundled sentence. Returns its duration in milliseconds, or 0."""
    if not phrase:
        return 0
    try:
        pcm, rate = load_fallback_pcm()
    except OSError:
        return 0
    target = int(getattr(sink, "sample_rate", 0) or cfg.conversation.realtime.sample_rate or rate)
    if rate != target and pcm:
        pcm = resample_pcm(pcm, rate, target)
        rate = target
    resume = getattr(sink, "resume", None)
    if resume:
        resume()
    if not pcm:
        return 0
    sink.write(pcm)
    return int(len(pcm) / 2 / max(rate, 1) * 1000)


class ListCapture:
    """Finite frame source for tests. `None` means the scripted list is exhausted.

    Frames in `barge` are only visible to non-blocking reads during playback, so a
    follow-up utterance left on `frames` is not swallowed while the reply plays.
    """

    def __init__(self, frames: list[bytes] | None = None):
        self.frames = list(frames or [])
        self.barge: list[bytes] = []
        self.index = 0

    def start(self) -> None:
        return

    def stop(self) -> None:
        return

    def flush(self) -> None:
        self.frames.clear()
        self.barge.clear()
        self.index = 0

    def read(self, timeout: float = 0.2, nowait: bool = False) -> bytes | None:
        if nowait:
            if self.barge:
                return self.barge.pop(0)
            return b""
        if self.index >= len(self.frames):
            return None
        frame = self.frames[self.index]
        self.index += 1
        return frame


class CollectingSink:
    def __init__(self):
        self.chunks: list[bytes] = []
        self.stopped = False

    def write(self, pcm: bytes) -> None:
        if self.stopped:
            return
        self.chunks.append(pcm)

    def stop(self) -> None:
        self.stopped = True

    def resume(self) -> None:
        self.stopped = False


class Ring:
    """Short memory of recent frames for wake-word pre-roll."""

    def __init__(self, max_frames: int):
        self._frames: deque[bytes] = deque(maxlen=max_frames)

    def push(self, frame: bytes) -> None:
        self._frames.append(frame)

    def tail(self, count: int) -> bytes:
        frames = list(self._frames)[-count:]
        return b"".join(frames)
