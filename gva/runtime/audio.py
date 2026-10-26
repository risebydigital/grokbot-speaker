"""Audio capture, playback and device selection. 16-bit mono, several sample rates."""

from __future__ import annotations

import math
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


def scale_blocksize(frames_at_48k: int, device_rate: int) -> int:
    """Period size. ``frames_at_48k`` of 2400 is 50 ms, scaled to the opened rate."""
    rate = max(1, int(device_rate))
    frames = int(round(int(frames_at_48k) * rate / 48000))
    return max(256, frames)


def capture_period(chunk_ms: int, frames_at_48k: int, device_rate: int) -> int:
    """Capture period. At least one wake chunk, and at least the output period."""
    rate = max(1, int(device_rate))
    chunk = max(1, int(rate * max(1, int(chunk_ms)) / 1000))
    return max(chunk, scale_blocksize(frames_at_48k, rate))


def coerce_latency(value) -> float | str:
    """PortAudio latency: seconds, or the names 'high' and 'low'."""
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"high", "low"}:
            return text
        return float(text)
    return float(value)


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
    """Preallocated ring the output callback copies from.

    Playback resampling happens before `push`, on the writer thread. `pull_into`
    only copies into the array PortAudio already owns. Silence before the
    pre-buffer fills is not an underrun. Once the buffer has drained, further
    idle callbacks stay silent and are not counted again. A true gap refills
    the pre-buffer before sound resumes.
    """

    def __init__(self, prebuffer_frames: int = 0, channels: int = 1):
        self.prebuffer_frames = max(0, int(prebuffer_frames))
        self.channels = max(1, int(channels))
        # A few seconds, so a catch-up after a late callback does not allocate.
        self._capacity = max(self.prebuffer_frames * 4, 48000 * 3, 8192)
        self._storage = np.zeros((self._capacity, self.channels), dtype=np.int16)
        self._read = 0
        self._size = 0
        self._primed = self.prebuffer_frames == 0
        self._fed = False
        self.underruns = 0
        self._lock = threading.Lock()

    def push(self, pcm: bytes) -> None:
        """Push interleaved device-format PCM. `channels` matches the open stream."""
        if len(pcm) < 2:
            return
        flat = np.frombuffer(pcm, dtype=np.int16)
        usable = (flat.size // self.channels) * self.channels
        if usable < self.channels:
            return
        frames = flat[:usable].reshape(-1, self.channels)
        with self._lock:
            self._write_frames(frames)
            self._fed = True

    def pull(self, frames: int) -> np.ndarray:
        """Return shape (frames, channels). The callback uses `pull_into` instead."""
        out = np.zeros((int(frames), self.channels), dtype=np.int16)
        self.pull_into(out)
        return out

    def pull_into(self, out: np.ndarray) -> int:
        """Copy the next frames into `out`. No new sample storage.

        Returns how many frames came from the ring. Zero is silence: either the
        pre-buffer has not filled, or this callback found the ring empty.
        """
        frames = int(out.shape[0])
        with self._lock:
            if not self._primed:
                if self._size < self.prebuffer_frames:
                    out.fill(0)
                    return 0
                self._primed = True
            take = min(frames, self._size)
            if take:
                self._read_into(out[:take])
            if take < frames:
                out[take:].fill(0)
            if take < frames and self._fed:
                # A short final block is the tail of the reply. Counting that and
                # the idle callback after it reported two underruns on every turn.
                # Only a callback that receives nothing, while audio was still
                # expected, is a gap. The pre-buffer then fills again so the next
                # chunk is not played as a trickle of tiny pieces.
                if take == 0:
                    self.underruns += 1
                    self._primed = self.prebuffer_frames == 0
                self._fed = False
            return int(take)

    def clear(self) -> None:
        with self._lock:
            self._read = 0
            self._size = 0
            self._primed = self.prebuffer_frames == 0
            self._fed = False

    def buffered(self) -> int:
        with self._lock:
            return int(self._size)

    def _write_frames(self, frames: np.ndarray) -> None:
        count = int(frames.shape[0])
        if count <= 0:
            return
        if count >= self._capacity:
            frames = frames[-self._capacity :]
            count = self._capacity
            self._read = 0
            self._size = 0
        overflow = self._size + count - self._capacity
        if overflow > 0:
            self._read = (self._read + overflow) % self._capacity
            self._size -= overflow
        write_at = (self._read + self._size) % self._capacity
        first = min(count, self._capacity - write_at)
        self._storage[write_at : write_at + first] = frames[:first]
        if count > first:
            self._storage[: count - first] = frames[first:]
        self._size += count

    def _read_into(self, out: np.ndarray) -> None:
        count = int(out.shape[0])
        first = min(count, self._capacity - self._read)
        out[:first] = self._storage[self._read : self._read + first]
        if count > first:
            out[first:] = self._storage[: count - first]
        self._read = (self._read + count) % self._capacity
        self._size -= count


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


class CaptureRing:
    """Fixed mono storage. The capture callback copies in. `read` resamples later."""

    def __init__(self, capacity: int):
        self.capacity = max(1024, int(capacity))
        self.storage = np.zeros(self.capacity, dtype=np.int16)
        self._acc = np.zeros(4096, dtype=np.int32)
        self._mono = np.zeros(4096, dtype=np.int16)
        self.read_at = 0
        self.size = 0

    def prepare_scratch(self, frames: int) -> None:
        frames = max(1, int(frames))
        if self._acc.size < frames:
            self._acc = np.zeros(frames, dtype=np.int32)
            self._mono = np.zeros(frames, dtype=np.int16)

    def clear(self) -> None:
        self.read_at = 0
        self.size = 0

    def write_bytes(self, pcm: bytes) -> None:
        if len(pcm) < 2:
            return
        self._write_flat(np.frombuffer(pcm, dtype=np.int16))

    def write_input(self, indata, frames: int) -> None:
        audio = np.asarray(indata)
        count = int(frames or 0)
        if count <= 0 or audio.size == 0:
            return
        if audio.ndim == 2 and audio.shape[1] > 1:
            if count > self._acc.size:
                self._acc = np.zeros(count, dtype=np.int32)
            acc = self._acc[:count]
            acc[:] = audio[:count, 0]
            channels = int(audio.shape[1])
            for channel in range(1, channels):
                acc += audio[:count, channel]
            acc //= channels
            mono = self._mono[:count]
            mono[:] = acc
            self._write_flat(mono)
            return
        flat = np.asarray(audio, dtype=np.int16).reshape(-1)
        self._write_flat(flat[:count])

    def take(self, max_frames: int) -> bytes:
        if self.size <= 0:
            return b""
        count = self.size if max_frames <= 0 else min(self.size, int(max_frames))
        out = np.empty(count, dtype=np.int16)
        first = min(count, self.capacity - self.read_at)
        out[:first] = self.storage[self.read_at : self.read_at + first]
        if count > first:
            out[first:] = self.storage[: count - first]
        self.read_at = (self.read_at + count) % self.capacity
        self.size -= count
        return out.tobytes()

    def _write_flat(self, samples: np.ndarray) -> None:
        count = int(samples.size)
        if count <= 0:
            return
        if count >= self.capacity:
            samples = samples[-self.capacity :]
            count = self.capacity
            self.read_at = 0
            self.size = 0
        overflow = self.size + count - self.capacity
        if overflow > 0:
            self.read_at = (self.read_at + overflow) % self.capacity
            self.size -= overflow
        write_at = (self.read_at + self.size) % self.capacity
        first = min(count, self.capacity - write_at)
        self.storage[write_at : write_at + first] = samples[:first]
        if count > first:
            self.storage[: count - first] = samples[first:]
        self.size += count


class SoundDeviceCapture:
    """Capture callback plus a supervisor that reopens the device after an error or EOF.

    The callback only copies samples. Resampling happens in `read`, so a slow
    resampler cannot stall PortAudio. `read` waits on the queue; it never
    reopens the device and never spins. The supervisor thread does that, with a
    sleep between attempts. `stop` aborts the stream instead of waiting for the
    callback, which is the PortAudio path that busy-waits forever if the
    callback is stuck.
    """

    def __init__(
        self,
        device: int | None,
        sample_rate: int,
        chunk_ms: int,
        on_lost=None,
        on_error=None,
        latency: float | str = 0.2,
        blocksize: int = 2400,
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms
        self.on_lost = on_lost
        self.on_error = on_error
        self.latency = coerce_latency(latency)
        self.blocksize_at_48k = int(blocksize)
        self.blocksize = capture_period(chunk_ms, self.blocksize_at_48k, sample_rate)
        self.device_rate = sample_rate
        self.device_channels = 1
        now = time.monotonic()
        self.last_audio_at = now
        self._opened_at = now
        self.reopen_delay = 0.5
        self._cv = threading.Condition()
        # Two seconds at 48 kHz. Resized once the device rate is known.
        self._ring = CaptureRing(48000 * 2)
        self._period = 0
        self._hw_overflows = 0
        self._stream = None
        self._running = False
        self._stopped = threading.Event()
        self._failed = threading.Event()
        self._error: BaseException | None = None
        self._reported: BaseException | None = None
        self._supervisor: threading.Thread | None = None
        self._resampler: StreamingResampler | None = None
        self._calls = 0
        self._rate_window_start = time.monotonic()

    @property
    def running(self) -> bool:
        return self._running

    def queued(self) -> int:
        with self._cv:
            return int(self._ring.size)

    def start(self) -> None:
        self._stopped.clear()
        self._running = True
        self.last_audio_at = time.monotonic()
        self._open_stream()
        self._ensure_supervisor()

    def _open_stream(self) -> None:
        if not self._running or self._stopped.is_set():
            return
        import sounddevice as sd

        self.device_rate, self.device_channels = probe_input(self.device, self.sample_rate)
        self._resampler = StreamingResampler(self.device_rate, self.sample_rate)
        block = capture_period(self.chunk_ms, self.blocksize_at_48k, self.device_rate)
        self.blocksize = block
        self._period = block
        with self._cv:
            self._ring = CaptureRing(max(block * 32, self.device_rate * 2))
            self._ring.prepare_scratch(block)
        self._stream = sd.InputStream(
            device=self.device,
            channels=self.device_channels,
            samplerate=self.device_rate,
            dtype="int16",
            blocksize=block,
            latency=self.latency,
            callback=self._on_audio,
        )
        self._stream.start()
        self._failed.clear()
        self._error = None
        self._reported = None
        self.reopen_delay = 0.5
        # A new stream gets a few seconds to deliver its first callback.
        # last_audio_at stays on the last real callback so a device that opens
        # and then stays silent can still trip the stall watchdog.
        self._opened_at = time.monotonic()

    def _on_audio(self, indata, frames, time_info, status) -> None:
        try:
            count = int(frames or 0)
            if self._callback_spinning():
                self._halt(RuntimeError("capture callback spun after the device closed"))
            if count <= 0 or indata is None or np.size(indata) == 0:
                self._halt(EOFError("capture stream ended"))
            if status:
                text = str(status)
                # One input overflow is normal if the consumer hiccups. Anything
                # else means the device is gone.
                if "overflow" not in text.casefold():
                    self._halt(RuntimeError(text or "capture device error"))
            with self._cv:
                self._ring.write_input(indata, count)
                self._cv.notify_all()
            self.last_audio_at = time.monotonic()
        except Exception as exc:
            if exc.__class__.__name__ == "CallbackStop" or getattr(exc, "_capture_halt", False):
                raise
            self._halt(exc)

    def _callback_spinning(self) -> bool:
        """True when PortAudio is invoking the callback far faster than realtime."""
        self._calls += 1
        now = time.monotonic()
        if now - self._rate_window_start < 1.0:
            return False
        calls = self._calls
        self._calls = 0
        self._rate_window_start = now
        expected = max(1, int(round(1000 / max(1, self.chunk_ms))))
        return calls > max(80, expected * 8)

    def _halt(self, exc: BaseException) -> None:
        self._fail(exc)
        try:
            import sounddevice as sd
        except Exception:
            marker = RuntimeError(str(exc))
            setattr(marker, "_capture_halt", True)
            raise marker from exc
        raise sd.CallbackStop from exc

    def _fail(self, exc: BaseException) -> None:
        self._error = exc
        self._failed.set()

    def _put(self, pcm: bytes) -> None:
        with self._cv:
            self._ring.write_bytes(pcm)
            self._cv.notify_all()

    def read(self, timeout: float = 0.2, nowait: bool = False) -> bytes | None:
        with self._cv:
            if not nowait and self._ring.size <= 0:
                self._cv.wait(timeout=timeout)
            raw = self._ring.take(self._period)
        if not raw:
            return b""
        return self._to_capture_rate(raw)

    def _to_capture_rate(self, pcm: bytes) -> bytes:
        resampler = self._resampler
        if resampler is None or resampler.from_rate == resampler.to_rate:
            return pcm
        return resampler.process(pcm)

    def _ensure_supervisor(self) -> None:
        thread = self._supervisor
        if thread is not None and thread.is_alive():
            return
        self._supervisor = threading.Thread(target=self._supervise, name="gva-capture", daemon=True)
        self._supervisor.start()

    def _supervise(self) -> None:
        while not self._stopped.wait(0.5):
            try:
                self._supervise_once()
            except Exception as exc:
                self._fail(exc)
                self._report(exc)

    def _supervise_once(self) -> None:
        if not self._running:
            return
        stream = self._stream
        inactive = stream is None or not bool(getattr(stream, "active", False))
        # Callbacks carry silence too. No callback means the stream died without an error.
        # A stream that just opened is given the same 5s before it counts as dead.
        quiet_since = max(float(self.last_audio_at), float(self._opened_at))
        silent = (time.monotonic() - quiet_since) > 5.0
        if silent and not self._failed.is_set():
            self._fail(TimeoutError("capture produced no audio"))
        if not self._failed.is_set() and not inactive:
            return
        self._reopen()

    def _reopen(self) -> None:
        if not self._running or self._stopped.is_set():
            return
        exc = self._error if self._error is not None else EOFError("capture stream ended")
        self._failed.clear()
        delay = float(self.reopen_delay or 0)
        if delay > 0:
            time.sleep(delay)
        if not self._running or self._stopped.is_set():
            return
        self._close_stream()
        try:
            self._open_stream()
        except Exception as open_exc:
            self._fail(open_exc)
            self.reopen_delay = min(5.0, max(0.5, delay * 2 or 0.5))
            exc = open_exc
        if self._running and not self._stopped.is_set():
            self._report(exc)

    def _report(self, exc: BaseException) -> None:
        if self._reported is exc:
            return
        self._reported = exc
        try:
            if self.on_error is not None:
                self.on_error(exc)
            elif self.on_lost is not None:
                self.on_lost()
        except Exception:
            return

    def flush(self) -> None:
        """Drop queued frames without spinning if the callback is still producing them."""
        with self._cv:
            self._ring.clear()

    def opened_stats(self) -> dict[str, Any]:
        """Latency and period PortAudio actually opened. Empty if the stream is closed."""
        stream = self._stream
        if stream is None:
            return {"opened": False}
        rate = float(getattr(stream, "samplerate", 0) or self.device_rate or 0)
        latency = float(getattr(stream, "latency", 0) or 0)
        block = int(getattr(stream, "blocksize", 0) or self.blocksize or 0)
        channels = int(getattr(stream, "channels", 0) or self.device_channels or 0)
        return {
            "opened": True,
            "rate": int(round(rate)) if rate else 0,
            "channels": channels,
            "blocksize": block,
            "latency_s": latency,
            "buffer_frames": int(round(latency * rate)) if latency and rate else 0,
            "buffer_ms": int(round(latency * 1000)),
        }

    def stop(self) -> None:
        self._running = False
        self._stopped.set()
        self._close_stream()
        thread = self._supervisor
        if thread is not None and thread is not threading.current_thread() and thread.is_alive():
            thread.join(timeout=2.0)
        self._supervisor = None

    def _close_stream(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is None:
            return
        abort = getattr(stream, "abort", None)
        if callable(abort):
            try:
                abort()
            except Exception as exc:
                self._fail(exc)
        closer = getattr(stream, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception as exc:
                self._fail(exc)


def _downmix_int16(indata) -> bytes:
    """Copy one callback's samples to mono PCM16. No resampling."""
    audio = np.asarray(indata)
    if audio.size == 0:
        return b""
    if audio.dtype == np.int16 and (audio.ndim == 1 or (audio.ndim == 2 and audio.shape[1] == 1)):
        return np.ascontiguousarray(audio.reshape(-1)).tobytes()
    if audio.ndim == 2 and audio.shape[1] > 1:
        mono = audio.astype(np.float64).mean(axis=1)
    elif audio.ndim == 2:
        mono = audio[:, 0]
    else:
        mono = audio.reshape(-1)
    return np.clip(np.round(mono), -32768, 32767).astype(np.int16).tobytes()


def play_pcm(sink, pcm: bytes, rate: int, protocol=None) -> None:
    """Play mono PCM through the one speaker path.

    While the realtime player task is running it is the only writer: chimes,
    Piper and bridge clips join that queue and meet the same jitter pre-buffer.
    With no player running, the sink's own pre-buffer is the playback path.
    """
    if not pcm:
        return
    if (
        protocol is not None
        and getattr(protocol, "player_live", False)
        and not getattr(protocol, "close", True)
    ):
        enqueue = getattr(protocol, "enqueue_pcm", None)
        if callable(enqueue):
            voice = int(getattr(getattr(protocol, "sink", None), "sample_rate", 0) or rate or 0)
            if voice and rate and rate != voice:
                pcm = resample_pcm(pcm, rate, voice)
            enqueue(pcm)
            return
    if sink is None:
        return
    target = int(getattr(sink, "sample_rate", 0) or rate or 0)
    if target and rate and rate != target:
        pcm = resample_pcm(pcm, rate, target)
    resume = getattr(sink, "resume", None)
    if callable(resume):
        try:
            resume()
        except Exception:
            pass
    sink.write(pcm)


class SoundDeviceSink:
    """One callback OutputStream for the process. Chunks share a resampler and a jitter buffer."""

    def __init__(
        self,
        device: int | None,
        sample_rate: int,
        prebuffer_ms: int = 250,
        latency: float | str = 0.2,
        blocksize: int = 2400,
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.device_rate = sample_rate
        self.channels = 1
        self.prebuffer_ms = int(prebuffer_ms)
        # 0.2 s asks for about 200 ms of device buffer. The name "high" is not
        # enough on the USB speaker: it still opened a 60 ms buffer.
        self.latency = coerce_latency(0.2 if latency in ("", None) else latency)
        self.blocksize_at_48k = int(blocksize)
        self.blocksize = scale_blocksize(self.blocksize_at_48k, sample_rate)
        self.buffer = JitterBuffer(int(sample_rate * self.prebuffer_ms / 1000), channels=1)
        self._resampler = StreamingResampler(sample_rate, sample_rate)
        self._source_bytes = 0
        self._stopped = False
        self._hw_underruns = 0
        self._stream = None
        self._format_ready = False
        # Callback clock. The stall watchdog treats a moving play head as progress
        # so a long reply is not mistaken for a dead audio loop.
        self.last_output_at = 0.0
        self._audible_until = 0.0
        self._stream_latency = self._configured_latency()
        self.writing = False
        self.write_started_at = 0.0

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
        self._begin_write()
        try:
            self.prepare()
            self._ensure()
            if self._resampler.from_rate != self.sample_rate or self._resampler.to_rate != self.device_rate:
                self._resampler = StreamingResampler(self.sample_rate, self.device_rate)
            self._source_bytes += len(pcm) - (len(pcm) % 2)
            fitted = upmix(self._resampler.process(pcm), self.channels)
            if fitted:
                self.buffer.push(fitted)
            self.last_output_at = time.monotonic()
        finally:
            self.writing = False

    def write_raw(self, pcm: bytes) -> None:
        """Queue mono PCM that is already at ``device_rate``.

        ``write`` resamples from the source rate. A chime is resampled on its
        own, including the filter tail, and must not pass through that again.
        """
        if self._stopped or not pcm:
            return
        self._begin_write()
        try:
            self.prepare()
            self._ensure()
            usable = len(pcm) - (len(pcm) % 2)
            if usable < 2:
                return
            fitted = upmix(pcm[:usable], self.channels)
            if fitted:
                self.buffer.push(fitted)
            self.last_output_at = time.monotonic()
        finally:
            self.writing = False

    def audio_ms(self) -> int:
        rate = self.sample_rate or 1
        return int(self._source_bytes / 2 / rate * 1000)

    def consume_audio_ms(self) -> int:
        heard = self.audio_ms()
        self._source_bytes = 0
        return heard

    def consume_underruns(self) -> int:
        hardware = int(self._hw_underruns)
        self._hw_underruns = 0
        count = int(self.buffer.underruns) + hardware
        self.buffer.underruns = 0
        return count

    def opened_stats(self) -> dict[str, Any]:
        """Latency and period PortAudio actually opened, plus underruns not yet logged."""
        stream = self._stream
        rate = float(getattr(stream, "samplerate", 0) or self.device_rate or 0) if stream is not None else float(self.device_rate or 0)
        latency = float(getattr(stream, "latency", 0) or 0) if stream is not None else 0.0
        block = int(getattr(stream, "blocksize", 0) or self.blocksize or 0) if stream is not None else int(self.blocksize or 0)
        channels = int(getattr(stream, "channels", 0) or self.channels or 0) if stream is not None else int(self.channels or 0)
        return {
            "opened": stream is not None,
            "rate": int(round(rate)) if rate else 0,
            "channels": channels,
            "blocksize": block,
            "latency_s": latency,
            "buffer_frames": int(round(latency * rate)) if latency and rate else 0,
            "buffer_ms": int(round(latency * 1000)),
            "prebuffer_ms": int(self.prebuffer_ms),
            "underruns": int(self.buffer.underruns) + int(self._hw_underruns),
        }

    def queued_frames(self) -> int:
        """Frames still waiting in the jitter buffer. Playback is not finished while this is above zero."""
        return int(self.buffer.buffered())

    def audible_until(self) -> float:
        """Monotonic time when the device has played what it has already been given.

        Queued software frames are still ahead of the DAC. After the ring is empty
        the stream latency is the hardware buffer that has not drained yet.
        """
        now = time.monotonic()
        rate = float(self.device_rate or self.sample_rate or 1)
        queued_s = self.queued_frames() / rate if rate else 0.0
        hardware = float(self._audible_until)
        if queued_s > 0:
            return max(hardware, now + queued_s + self._latency_seconds())
        return hardware

    def _configured_latency(self) -> float:
        raw = self.latency
        if isinstance(raw, (int, float)) and float(raw) > 0:
            return float(raw)
        return 0.2

    def _latency_seconds(self) -> float:
        raw = self._stream_latency
        if isinstance(raw, (int, float)) and float(raw) > 0:
            return float(raw)
        return self._configured_latency()

    def _begin_write(self) -> None:
        self.writing = True
        self.write_started_at = time.monotonic()

    def _mark_audible(self, copied: int) -> None:
        """Remember when the frames just copied will leave the hardware buffer."""
        if copied <= 0:
            return
        now = time.monotonic()
        self.last_output_at = now
        self._audible_until = now + self._latency_seconds()

    def stop(self) -> None:
        """Drop queued speech. The stream stays open; the next write does not reopen it."""
        self._stopped = True
        self.buffer.clear()
        self._resampler.reset()
        self._audible_until = 0.0

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
        block = scale_blocksize(self.blocksize_at_48k, self.device_rate)
        self.blocksize = block

        def callback(outdata, frames, time_info, status) -> None:
            # Copy only. Resampling already happened in write(), off this thread.
            if status and getattr(status, "output_underflow", False):
                self._hw_underruns += 1
            copied = self.buffer.pull_into(outdata)
            if copied:
                self._mark_audible(copied)

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
        opened = getattr(self._stream, "latency", 0) or 0
        try:
            opened_s = float(opened)
        except (TypeError, ValueError):
            opened_s = 0.0
        if opened_s > 0:
            self._stream_latency = opened_s


def _public_error(exc: BaseException) -> str:
    text = f"{exc.__class__.__name__}: {exc}"
    if "Bearer" in text or "xai-" in text:
        return exc.__class__.__name__
    return text[:300]


def probe_opened_streams(cfg) -> tuple[dict[str, Any], str, dict[str, Any], str]:
    """Open the speaker and microphone long enough to read the buffer PortAudio chose."""
    output: dict[str, Any] = {}
    capture: dict[str, Any] = {}
    output_error = ""
    capture_error = ""
    sink = SoundDeviceSink(
        None,
        cfg.conversation.realtime.sample_rate if cfg.conversation.engine == "realtime" else cfg.audio.sample_rate,
        prebuffer_ms=cfg.audio.playback_prebuffer_ms,
        latency=cfg.audio.output_latency,
        blocksize=cfg.audio.output_blocksize,
    )
    try:
        sink.device = prefer_output(list_devices(), cfg.audio.output_device)
        sink.prepare()
        sink._ensure()
        output = sink.opened_stats()
    except Exception as exc:
        output_error = _public_error(exc)
    finally:
        sink.close()
    mic = SoundDeviceCapture(
        None,
        cfg.audio.capture_rate if cfg.conversation.engine == "realtime" else cfg.audio.sample_rate,
        cfg.audio.chunk_ms,
        latency=cfg.audio.capture_latency,
        blocksize=cfg.audio.capture_blocksize,
    )
    try:
        mic.device = resolve_device(cfg.audio.input_device, "input")
        mic.start()
        capture = mic.opened_stats()
    except Exception as exc:
        capture_error = _public_error(exc)
    finally:
        mic.stop()
    return output, output_error, capture, capture_error


def render_audio_stats(
    cfg,
    output: dict[str, Any],
    output_error: str,
    capture: dict[str, Any],
    capture_error: str,
    underrun_events: int,
    underrun_samples: int,
) -> str:
    """Plain-text report. Device names and buffer sizes only."""
    lines = [
        (
            f"configured output latency {cfg.audio.output_latency} "
            f"blocksize {cfg.audio.output_blocksize} at 48000 Hz "
            f"prebuffer {cfg.audio.playback_prebuffer_ms} ms"
        ),
        (
            f"configured capture latency {cfg.audio.capture_latency} "
            f"blocksize {cfg.audio.capture_blocksize} at 48000 Hz "
            f"chunk {cfg.audio.chunk_ms} ms"
        ),
    ]
    lines.append(_stream_line("output", output, output_error))
    lines.append(_stream_line("capture", capture, capture_error))
    lines.append(f"playback.underrun events {int(underrun_events)}  underruns {int(underrun_samples)}")
    return "\n".join(lines)


def _stream_line(kind: str, stats: dict[str, Any], error: str) -> str:
    if error:
        return f"{kind}: not opened ({error})"
    if not stats or not stats.get("opened"):
        return f"{kind}: not opened"
    period_ms = 0
    rate = int(stats.get("rate") or 0)
    block = int(stats.get("blocksize") or 0)
    if rate and block:
        period_ms = int(round(block / rate * 1000))
    extra = ""
    if kind == "output":
        extra = f"  prebuffer {int(stats.get('prebuffer_ms') or 0)} ms"
    return (
        f"{kind}: {rate} Hz  {int(stats.get('channels') or 0)} ch  "
        f"blocksize {block} ({period_ms} ms)  "
        f"latency {float(stats.get('latency_s') or 0):.3f} s  "
        f"buffer {int(stats.get('buffer_frames') or 0)} frames "
        f"({int(stats.get('buffer_ms') or 0)} ms)"
        + extra
    )


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
