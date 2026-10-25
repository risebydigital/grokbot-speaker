"""Voice activity detection. Energy is always available; Silero is used when installed."""

from __future__ import annotations

from gva.runtime.audio import rms


class EnergyVad:
    def __init__(self, threshold: float = 500):
        self.threshold = threshold

    def is_speech(self, frame: bytes) -> bool:
        return rms(frame) >= self.threshold


class SileroVad:
    """Silero via openWakeWord's ONNX file. Fall back to energy if it is not installed.

    The VAD class has no TFLite path. `model_path` points at silero_vad.onnx.
    """

    def __init__(self, fallback: EnergyVad, model_path: str | None = None):
        self.fallback = fallback
        self.model_path = model_path
        self._vad = None
        self._failed = False

    def is_speech(self, frame: bytes) -> bool:
        if self._failed:
            return self.fallback.is_speech(frame)
        try:
            if self._vad is None:
                from openwakeword.vad import VAD

                kwargs = {}
                if self.model_path:
                    kwargs["model_path"] = self.model_path
                self._vad = VAD(**kwargs)
            import numpy as np

            audio = np.frombuffer(frame, dtype=np.int16)
            score = float(self._vad(audio))
            return score >= 0.5
        except Exception:
            self._failed = True
            return self.fallback.is_speech(frame)


class Endpointer:
    def __init__(self, vad, end_silence_ms: int, max_seconds: float, chunk_ms: int, min_speech_ms: int = 200):
        self.vad = vad
        self.end_silence_ms = end_silence_ms
        self.max_ms = int(max_seconds * 1000)
        self.chunk_ms = chunk_ms
        self.min_speech_ms = min_speech_ms
        self.speech_ms = 0
        self.silence_ms = 0
        self.elapsed_ms = 0
        self.closed = False

    def add(self, frame: bytes) -> bool:
        self.elapsed_ms += self.chunk_ms
        if self.vad.is_speech(frame):
            self.speech_ms += self.chunk_ms
            self.silence_ms = 0
        else:
            self.silence_ms += self.chunk_ms
        if self.elapsed_ms >= self.max_ms:
            self.closed = True
        elif self.speech_ms >= self.min_speech_ms and self.silence_ms >= self.end_silence_ms:
            self.closed = True
        return self.closed

    @property
    def heard_speech(self) -> bool:
        return self.speech_ms >= self.min_speech_ms
