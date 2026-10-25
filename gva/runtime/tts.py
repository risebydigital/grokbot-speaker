"""Piper TTS, kept loaded. xAI TTS is optional and off unless configured."""

from __future__ import annotations

from pathlib import Path

from gva.config import Config, env_value
from gva.runtime.textutil import normalise_for_speech


class TtsError(RuntimeError):
    pass


class PiperTts:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._voice = None

    def voice_path(self) -> Path:
        return Path(self.cfg.tts.model_dir) / f"{self.cfg.tts.voice}.onnx"

    def synthesise(self, text: str) -> tuple[bytes, int]:
        spoken = normalise_for_speech(text, self.cfg.tts.pronunciation)
        if not spoken:
            return b"", self.cfg.audio.sample_rate
        voice = self._load()
        rate = self.cfg.audio.sample_rate
        pieces: list[bytes] = []
        length_scale = 1.0 / self.cfg.tts.speaking_rate if self.cfg.tts.speaking_rate else 1.0
        try:
            from piper import SynthesisConfig

            config = SynthesisConfig(length_scale=length_scale)
            iterator = voice.synthesize(spoken, syn_config=config)
        except TypeError:
            iterator = voice.synthesize(spoken)
        for chunk in iterator:
            audio = getattr(chunk, "audio_int16_bytes", None)
            if audio is None:
                audio = getattr(chunk, "audio_int16_array", b"")
                if not isinstance(audio, (bytes, bytearray)):
                    audio = audio.tobytes()
            pieces.append(bytes(audio))
            rate = int(getattr(chunk, "sample_rate", rate) or rate)
        return b"".join(pieces), rate

    def _load(self):
        if self._voice is not None:
            return self._voice
        path = self.voice_path()
        if not path.is_file():
            raise TtsError(
                f"Piper voice not found at {path}. Run `gva models download` on the Pi."
            )
        from piper import PiperVoice

        self._voice = PiperVoice.load(str(path))
        return self._voice


class XaiTts:
    def __init__(self, cfg: Config, http):
        self.cfg = cfg
        self.http = http

    def synthesise(self, text: str) -> tuple[bytes, int]:
        key = env_value("XAI_API_KEY")
        if not key:
            raise TtsError("XAI_API_KEY is not set")
        spoken = normalise_for_speech(text, self.cfg.tts.pronunciation)
        response = self.http.post(
            self.cfg.llm.base_url.rstrip("/") + "/tts",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "text": spoken,
                "voice_id": self.cfg.tts.xai.voice_id,
                "language": "en",
                "speed": self.cfg.tts.speaking_rate,
            },
            timeout=15,
        )
        response.raise_for_status()
        # MP3 bytes. Playback of MP3 needs a decoder; the device default is Piper PCM.
        return response.content, self.cfg.audio.sample_rate


class TtsRouter:
    def __init__(self, cfg: Config, piper, cloud=None):
        self.cfg = cfg
        self.piper = piper
        self.cloud = cloud

    def synthesise(self, text: str) -> tuple[bytes, int]:
        if self.cfg.tts.engine == "xai" or self.cfg.tts.xai.enabled:
            if self.cloud is None:
                raise TtsError("xAI TTS is enabled but no client is configured")
            return self.cloud.synthesise(text)
        return self.piper.synthesise(text)
