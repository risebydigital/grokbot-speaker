"""Single-delay echo attenuation.

The AIRHUG microphone has no hardware echo cancellation. This estimates one delay
and a gain, then subtracts the playback reference from the microphone. It is not a
full acoustic echo canceller. Wake-word barge-in does not depend on it; the stop
command uses the residual so the speaker is less likely to trip the detector.
"""

from __future__ import annotations

import numpy as np


class AdaptiveEchoCanceller:
    def __init__(self, delay_ms: int = 80, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.delay = max(0, int(sample_rate * delay_ms / 1000))
        self._length = sample_rate * 3
        self._ref = np.zeros(self._length, dtype=np.float64)
        self._pos = 0
        self.gain = 0.0

    def push_reference(self, pcm: bytes) -> None:
        if not pcm:
            return
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
        count = len(samples)
        index = (self._pos + np.arange(count)) % self._length
        self._ref[index] = samples
        self._pos = (self._pos + count) % self._length

    def process(self, mic: bytes) -> bytes:
        if not mic:
            return mic
        samples = np.frombuffer(mic, dtype=np.int16).astype(np.float64)
        count = len(samples)
        start = (self._pos - self.delay - count) % self._length
        index = (start + np.arange(count)) % self._length
        reference = self._ref[index]
        energy = float(np.dot(reference, reference)) + 1e-6
        instant = float(np.dot(reference, samples)) / energy
        self.gain = 0.85 * self.gain + 0.15 * instant
        residual = samples - self.gain * reference
        return np.clip(residual, -32768, 32767).astype(np.int16).tobytes()
