"""Speech to text. faster-whisper is the default. xAI STT runs only when opted in."""

from __future__ import annotations

import io
import math
import wave
from dataclasses import dataclass

from gva.config import Config, env_value


@dataclass
class SttResult:
    text: str
    confidence: float | None
    language: str | None
    duration_ms: int
    audio_ms: int
    rejected_reason: str | None
    engine: str
    model: str


def audio_ms(pcm: bytes, sample_rate: int = 16000) -> int:
    return int(len(pcm) / 2 / sample_rate * 1000)


def judge(text: str, confidence: float | None, cfg: Config) -> str | None:
    cleaned = " ".join(text.strip().split())
    if len(cleaned) < 2:
        return "empty"
    folded = cleaned.casefold()
    hallucinations = {item.casefold() for item in cfg.stt.hallucinations}
    if folded in hallucinations:
        return "hallucination"
    if confidence is not None and confidence < cfg.stt.min_confidence:
        return "low_confidence"
    return None


def _result(cfg: Config, engine: str, model: str, text: str, confidence: float | None, language: str | None, pcm: bytes, duration_ms: int) -> SttResult:
    reason = judge(text, confidence, cfg)
    return SttResult(
        text=" ".join(text.strip().split()),
        confidence=confidence,
        language=language,
        duration_ms=duration_ms,
        audio_ms=audio_ms(pcm, cfg.audio.sample_rate),
        rejected_reason=reason,
        engine=engine,
        model=model,
    )


class FasterWhisperStt:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._model = None

    def transcribe(self, pcm: bytes) -> SttResult:
        import time

        import numpy as np
        from faster_whisper import WhisperModel

        if self._model is None:
            self._model = WhisperModel(
                self.cfg.stt.model,
                device="cpu",
                compute_type=self.cfg.stt.compute_type,
                cpu_threads=self.cfg.stt.cpu_threads,
            )
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        started = time.monotonic()
        segments, info = self._model.transcribe(
            audio,
            language=self.cfg.stt.language,
            beam_size=self.cfg.stt.beam_size,
            vad_filter=False,
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        texts = []
        logprobs = []
        no_speech = []
        for segment in segments:
            texts.append(segment.text)
            logprobs.append(segment.avg_logprob)
            no_speech.append(segment.no_speech_prob)
        text = " ".join(part.strip() for part in texts).strip()
        if logprobs:
            avg_log = sum(logprobs) / len(logprobs)
            avg_no = sum(no_speech) / len(no_speech)
            confidence = max(0.0, min(1.0, (1.0 - avg_no) * math.exp(avg_log)))
        else:
            confidence = 0.0
        elapsed = int((time.monotonic() - started) * 1000)
        return _result(self.cfg, "faster_whisper", self.cfg.stt.model, text, confidence, getattr(info, "language", None), pcm, elapsed)


class XaiStt:
    def __init__(self, cfg: Config, http):
        self.cfg = cfg
        self.http = http

    def transcribe(self, pcm: bytes) -> SttResult:
        import time

        key = env_value("XAI_API_KEY")
        if not key:
            raise RuntimeError("XAI_API_KEY is not set")
        buf = io.BytesIO()
        with wave.open(buf, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(self.cfg.audio.sample_rate)
            handle.writeframes(pcm)
        data = [
            ("model", self.cfg.stt.xai.model),
            ("language", self.cfg.stt.language or "en"),
            ("format", "true"),
        ]
        for term in self.cfg.stt.xai.keyterms:
            data.append(("keyterm", term))
        started = time.monotonic()
        response = self.http.post(
            self.cfg.llm.base_url.rstrip("/") + "/stt",
            headers={"Authorization": f"Bearer {key}"},
            data=data,
            files={"file": ("utterance.wav", buf.getvalue(), "audio/wav")},
            timeout=15,
        )
        response.raise_for_status()
        body = response.json()
        text = body.get("text") or ""
        elapsed = int((time.monotonic() - started) * 1000)
        return _result(self.cfg, "xai", self.cfg.stt.xai.model, text, None, self.cfg.stt.language, pcm, elapsed)


class SttRouter:
    def __init__(self, cfg: Config, local, cloud):
        self.cfg = cfg
        self.local = local
        self.cloud = cloud

    def transcribe(self, pcm: bytes) -> SttResult:
        if self.cfg.stt.engine == "xai":
            return self.cloud.transcribe(pcm)
        if self.cfg.stt.engine == "vosk":
            raise RuntimeError("The vosk engine is not installed. Use faster_whisper or xai.")
        try:
            result = self.local.transcribe(pcm)
        except Exception:
            if self.cfg.stt.xai.allow_cloud_fallback:
                return self.cloud.transcribe(pcm)
            raise
        if result.rejected_reason == "low_confidence" and self.cfg.stt.xai.allow_cloud_fallback:
            return self.cloud.transcribe(pcm)
        return result
