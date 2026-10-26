"""Milo's chime.

This is the generator Adam approved on 26 Oct 2026. Written pitches D4, A4
and F#4 are transposed up a fourth, so what plays is G4, then D5, landing on
B4 and G4. Each note is a sine plus a quiet second harmonic, with a
raised-cosine attack. Two short room reflections follow, then a linear fade
over the last quarter second. 44.1 kHz mono 16-bit, 1.2 seconds, peak 18000.

Playback resamples that whole buffer to the output device rate. Silence is
padded on either side so the filter does not click, and a few milliseconds
at each edge are faded. The note itself starts 20 ms in, so that fade stays
inside the silence.
"""

from __future__ import annotations

import math

import numpy as np

CANONICAL_RATE = 44100
DURATION_S = 1.2
PEAK = 18000

# Written pitches, then up a fourth (2 ** (5/12)).
_TRANSPOSE = 2.0 ** (5.0 / 12.0)
_D4 = 293.7
_FS4 = 370.0
_A4 = 440.0
G4 = _D4 * _TRANSPOSE
D5 = _A4 * _TRANSPOSE
B4 = _FS4 * _TRANSPOSE

_ATTACK_S = 0.03
_HARMONIC = 0.06
_REFLECTIONS = ((0.03, 0.12), (0.06, 0.07))
_TAIL_FADE_S = 0.25
# Inside the 20 ms of silence before the first note, and inside the tail fade.
_EDGE_FADE_S = 0.008

_approved: np.ndarray | None = None


def _approved_float() -> np.ndarray:
    """The approved buffer before the int16 cast. Peak is 18000."""
    global _approved
    if _approved is None:
        _approved = _render()
    return _approved


def canonical_samples() -> np.ndarray:
    """44.1 kHz int16. Same bytes as the approved generator's wave writer."""
    return _approved_float().astype(np.int16)


def canonical_pcm() -> bytes:
    return canonical_samples().tobytes()


def chime_pcm(sample_rate: int = CANONICAL_RATE, volume: float = 1.0) -> bytes:
    """Mono int16 at ``sample_rate``. ``volume`` 1 peaks at 18000. 0 is silence."""
    level = _volume(volume)
    rate = int(sample_rate)
    if rate == CANONICAL_RATE:
        return (_approved_float() * level).astype(np.int16).tobytes()
    if rate <= 0:
        raise ValueError("sample rate must be positive")
    audio = _resample(_approved_float(), CANONICAL_RATE, rate)
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


def _render() -> np.ndarray:
    """Direct port of the approved generator, including the int16-ready scale."""
    count = int(DURATION_S * CANONICAL_RATE)
    t = np.arange(count, dtype=np.float64) / CANONICAL_RATE
    dry = _note(t, _D4, 0.02, 0.55, decay=0.16) + _note(t, _A4, 0.14, 0.6, decay=0.16)
    dry = dry + _note(t, _FS4, 0.28, 0.7, decay=0.38) + _note(t, _D4, 0.28, 0.3, decay=0.38)
    wet = dry.copy()
    for delay_s, gain in _REFLECTIONS:
        shift = int(delay_s * CANONICAL_RATE)
        wet[shift:] += gain * dry[:-shift]
    fade = int(_TAIL_FADE_S * CANONICAL_RATE)
    wet[-fade:] *= np.linspace(1.0, 0.0, fade)
    peak = float(np.max(np.abs(wet)))
    return wet / peak * PEAK


def _note(t: np.ndarray, freq: float, start: float, amp: float, decay: float) -> np.ndarray:
    sounding = freq * _TRANSPOSE
    out = np.zeros_like(t)
    active = t >= start
    local = t[active] - start
    attack = 0.5 - 0.5 * np.cos(np.pi * np.minimum(local / _ATTACK_S, 1.0))
    envelope = attack * np.exp(-np.maximum(local - _ATTACK_S, 0.0) / decay)
    partials = np.sin(2.0 * np.pi * sounding * local) + _HARMONIC * np.sin(4.0 * np.pi * sounding * local)
    out[active] = amp * envelope * partials
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
