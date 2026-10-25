"""Audio capture, playback and device selection. 16-bit mono, several sample rates."""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from typing import Any

import numpy as np


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


class JitterBuffer:
    """Holds PCM until a short pre-buffer is full, then plays it as one stream.

    Calling sounddevice.play for every realtime audio delta restarts the device
    and sounds chopped. This buffer is what the output callback reads instead.
    """

    def __init__(self, prebuffer_frames: int = 0):
        self.prebuffer_frames = max(0, int(prebuffer_frames))
        self._samples = np.zeros(0, dtype=np.int16)
        self._primed = self.prebuffer_frames == 0
        self.underruns = 0
        self._lock = threading.Lock()

    def push(self, pcm: bytes) -> None:
        if len(pcm) < 2:
            return
        chunk = np.frombuffer(pcm, dtype=np.int16).copy()
        with self._lock:
            self._samples = np.concatenate([self._samples, chunk])

    def pull(self, frames: int) -> np.ndarray:
        out = np.zeros(frames, dtype=np.int16)
        with self._lock:
            if not self._primed:
                if len(self._samples) < self.prebuffer_frames:
                    return out
                self._primed = True
            take = min(frames, len(self._samples))
            if take:
                out[:take] = self._samples[:take]
                self._samples = self._samples[take:]
            if take < frames:
                self.underruns += 1
        return out

    def clear(self) -> None:
        with self._lock:
            self._samples = np.zeros(0, dtype=np.int16)
            self._primed = self.prebuffer_frames == 0

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
    if "airhug" in name or "bluetooth" in name or "bluez" in name:
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
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=32)
        self._stream = None
        self._lost = threading.Event()

    def start(self) -> None:
        import sounddevice as sd

        block = chunk_samples(self.sample_rate, self.chunk_ms)
        self._stream = sd.InputStream(
            device=self.device,
            channels=1,
            samplerate=self.sample_rate,
            dtype="int16",
            blocksize=block,
            callback=self._on_audio,
        )
        self._stream.start()

    def _on_audio(self, indata, frames, time_info, status) -> None:
        if status:
            self._lost.set()
        pcm = np.ascontiguousarray(indata[:, 0]).tobytes()
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
    """One OutputStream for the life of the process. Chunks join a jitter buffer."""

    def __init__(self, device: int | None, sample_rate: int, prebuffer_ms: int = 100):
        self.device = device
        self.sample_rate = sample_rate
        self.buffer = JitterBuffer(int(sample_rate * prebuffer_ms / 1000))
        self._stopped = False
        self._stream = None

    def write(self, pcm: bytes) -> None:
        if self._stopped or not pcm:
            return
        self._ensure()
        self.buffer.push(pcm)

    def stop(self) -> None:
        """Drop queued speech immediately. The stream stays open for the next utterance."""
        self._stopped = True
        self.buffer.clear()

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

        block = max(1, int(self.sample_rate * 0.02))

        def callback(outdata, frames, time_info, status) -> None:
            outdata[:, 0] = self.buffer.pull(frames)

        self._stream = sd.OutputStream(
            device=self.device,
            channels=1,
            samplerate=self.sample_rate,
            dtype="int16",
            blocksize=block,
            callback=callback,
        )
        self._stream.start()


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
