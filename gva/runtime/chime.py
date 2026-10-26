"""Milo's chime.

Pure sines, a raised-cosine attack, a quiet second harmonic, and two short
room reflections. The notes are G4, then D5, landing together on B4 and G4.
The canonical buffer is 1.2 seconds, 44.1 kHz, mono 16-bit, peak 18000.

Playback resamples that whole buffer to the output device rate, with silence
padded on either side so the filter does not click, then fades the edges.
"""

from __future__ import annotations

import math

import numpy as np

CANONICAL_RATE = 44100
DURATION_S = 1.2
PEAK = 18000

# Equal temperament, A4 = 440 Hz.
G4 = 440.0 * 2.0 ** (-2.0 / 12.0)
B4 = 440.0 * 2.0 ** (2.0 / 12.0)
D5 = 440.0 * 2.0 ** (5.0 / 12.0)

_ATTACK_S = 0.015
_RELEASE_S = 0.045
_HARMONIC = 0.035
# Delay in seconds, linear gain. Short enough to read as a small room.
_REFLECTIONS = ((0.0125, 0.14), (0.027, 0.06))
_EDGE_FADE_S = 0.040
# start, end, frequency, amplitude, decay time constant
_NOTES = (
    (0.00, 0.78, G4, 1.00, 0.26),
    (0.28, 0.86, D5, 0.92, 0.20),
    (0.58, DURATION_S, B4, 0.78, 0.34),
    (0.58, DURATION_S, G4, 0.70, 0.36),
)

_canonical: np.ndarray | None = None


def canonical_samples() -> np.ndarray:
    """The 44.1 kHz int16 buffer. Cached, because every chime is the same."""
    global _canonical
    if _canonical is None:
        _canonical = _synthesize()
    return _canonical


def canonical_pcm() -> bytes:
    return canonical_samples().tobytes()


def chime_pcm(sample_rate: int = CANONICAL_RATE, volume: float = 1.0) -> bytes:
    """Mono int16 at ``sample_rate``. ``volume`` 1 peaks at 18000. 0 is silence."""
    level = _volume(volume)
    source = canonical_samples()
    rate = int(sample_rate)
    if rate == CANONICAL_RATE:
        if level == 1.0:
            return source.tobytes()
        return _scale(source.astype(np.float64), int(round(PEAK * level)))
    if rate <= 0:
        raise ValueError("sample rate must be positive")
    audio = _resample(source.astype(np.float64), CANONICAL_RATE, rate)
    audio = _fade_edges(audio, rate, _EDGE_FADE_S)
    return _scale(audio, int(round(PEAK * level)))


def play_chime(sink, cfg=None, volume: float | None = None, *, raise_errors: bool = False) -> bytes:
    """Write one chime. Device-rate sinks get a raw buffer so it is not resampled twice."""
    if sink is None:
        return b""
    try:
        resume = getattr(sink, "resume", None)
        if resume:
            resume()
        chosen = volume
        if chosen is None and cfg is not None:
            chosen = cfg.audio.chime_volume
        level = _volume(1.0 if chosen is None else chosen)
        write_raw = getattr(sink, "write_raw", None)
        prepare = getattr(sink, "prepare", None)
        if callable(write_raw) and callable(prepare):
            prepare()
            rate = int(getattr(sink, "device_rate", 0) or CANONICAL_RATE)
            pcm = chime_pcm(rate, level)
            write_raw(pcm)
            return pcm
        rate = int(getattr(sink, "sample_rate", 0) or CANONICAL_RATE)
        pcm = chime_pcm(rate, level)
        sink.write(pcm)
        return pcm
    except Exception:
        if raise_errors:
            raise
        return b""


def _volume(value: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(number):
        return 1.0
    return min(1.0, max(0.0, number))


def _synthesize() -> np.ndarray:
    count = int(round(DURATION_S * CANONICAL_RATE))
    t = np.arange(count, dtype=np.float64) / CANONICAL_RATE
    dry = np.zeros(count, dtype=np.float64)
    for start, end, freq, amp, decay in _NOTES:
        dry += _strike(t, freq, start, end, amp, decay)
    wet = dry.copy()
    for delay_s, gain in _REFLECTIONS:
        shift = int(round(delay_s * CANONICAL_RATE))
        if 0 < shift < count:
            wet[shift:] += gain * dry[:-shift]
    wet = _fade_edges(wet, CANONICAL_RATE, _EDGE_FADE_S)
    pcm = _scale(wet, PEAK)
    return np.frombuffer(pcm, dtype=np.int16).copy()


def _strike(t: np.ndarray, freq: float, start: float, end: float, amp: float, decay: float) -> np.ndarray:
    local = t - start
    span = end - start
    active = (local >= 0.0) & (local <= span)
    out = np.zeros_like(t)
    if span <= 0.0 or not np.any(active):
        return out
    x = local[active]
    attack = 0.5 * (1.0 - np.cos(np.pi * np.clip(x / _ATTACK_S, 0.0, 1.0)))
    tail = np.exp(-x / decay)
    release = np.ones_like(x)
    release_from = span - _RELEASE_S
    fading = x > release_from
    if np.any(fading):
        phase = np.clip((x[fading] - release_from) / _RELEASE_S, 0.0, 1.0)
        release[fading] = 0.5 * (1.0 + np.cos(np.pi * phase))
    envelope = amp * attack * tail * release
    tone = np.sin(2.0 * np.pi * freq * t[active]) + _HARMONIC * np.sin(4.0 * np.pi * freq * t[active])
    out[active] = tone * envelope
    return out


def _fade_edges(audio: np.ndarray, rate: int, seconds: float) -> np.ndarray:
    faded = np.array(audio, dtype=np.float64, copy=True)
    if faded.size < 4:
        return faded
    fade = max(2, int(round(seconds * rate)))
    fade = min(fade, faded.size // 4)
    ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(fade, dtype=np.float64) / (fade - 1)))
    faded[:fade] *= ramp
    faded[-fade:] *= ramp[::-1]
    faded[0] = 0.0
    faded[-1] = 0.0
    return faded


def _scale(audio: np.ndarray, peak: int) -> bytes:
    """Int16 bytes whose largest sample is ``peak`` (or silence when peak is 0)."""
    count = int(audio.size)
    if count == 0 or peak <= 0:
        return np.zeros(count, dtype=np.int16).tobytes()
    level = float(np.max(np.abs(audio)))
    if level <= 0.0:
        return np.zeros(count, dtype=np.int16).tobytes()
    scaled = audio.astype(np.float64) * (peak / level)
    out = np.clip(np.rint(scaled), -peak, peak).astype(np.int32)
    out[0] = 0
    out[-1] = 0
    index = int(np.argmax(np.abs(out)))
    current = abs(int(out[index]))
    if 0 < current < peak:
        out[index] = peak if out[index] > 0 else -peak
    return out.astype(np.int16).tobytes()


def _resample(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Resample one complete buffer.

    44100 to 48000 is 160/147. The streaming sinc used for realtime 24 kHz
    deltas would build a huge filter for that ratio, and cutting its tail
    clicks. Padding the whole chime, resampling once, and dropping the pad
    keeps the edges at zero. scipy when it is installed, otherwise linear
    interpolation. These notes sit far below either Nyquist.
    """
    pad = max(8, int(round(from_rate * 0.02)))
    padded = np.concatenate([np.zeros(pad, dtype=np.float64), np.asarray(audio, dtype=np.float64), np.zeros(pad)])
    count = max(1, int(round(padded.size * to_rate / from_rate)))
    try:
        import scipy.signal

        converted = np.asarray(scipy.signal.resample(padded, count), dtype=np.float64)
    except Exception:
        old = np.linspace(0.0, 1.0, num=padded.size, endpoint=False)
        new = np.linspace(0.0, 1.0, num=count, endpoint=False)
        converted = np.interp(new, old, padded)
    drop = int(round(pad * to_rate / from_rate))
    if drop > 0 and converted.size > drop * 2 + 8:
        converted = converted[drop:-drop]
    return converted
