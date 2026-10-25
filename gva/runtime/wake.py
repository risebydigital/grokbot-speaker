"""Wake-word scoring. openWakeWord has no Milo model; the default phrase is hey jarvis."""

from __future__ import annotations

import time
from pathlib import Path

from gva.runtime.audio import rms


class ThresholdWake:
    def __init__(self, threshold: float = 0.5, patience: int = 2, refractory_s: float = 2.0):
        self.threshold = threshold
        self.patience = max(1, patience)
        self.refractory_s = refractory_s
        self._above = 0
        self._last_fire = 0.0

    def observe(self, score: float, now: float | None = None) -> bool:
        moment = time.monotonic() if now is None else now
        if moment - self._last_fire < self.refractory_s:
            self._above = 0
            return False
        if score >= self.threshold:
            self._above += 1
        else:
            self._above = 0
        if self._above >= self.patience:
            self._above = 0
            self._last_fire = moment
            return True
        return False

    def reset_refractory(self) -> None:
        self._last_fire = 0.0
        self._above = 0


class EnergyWake:
    """Test double: a loud frame scores as a wake. Not used on the device."""

    def __init__(self, logic: ThresholdWake, threshold_rms: float = 2000, name: str = "scripted"):
        self.logic = logic
        self.threshold_rms = threshold_rms
        self.model_name = name

    def score(self, frame: bytes) -> float:
        return 0.95 if rms(frame) >= self.threshold_rms else 0.0

    def observe(self, score: float, now: float | None = None) -> bool:
        return self.logic.observe(score, now)

    def reset(self) -> None:
        self.logic.reset_refractory()


class OpenWakeWordEngine:
    """Scores 80 ms frames. `model` is a pretrained name (hey_jarvis) or a path to an ONNX file."""

    def __init__(
        self,
        model: str,
        model_dir: str,
        threshold: float,
        patience: int,
        refractory_s: float,
        phrase: str,
    ):
        self.model_spec = model
        self.model_dir = Path(model_dir)
        self.phrase = phrase
        self.model_name = model
        self.logic = ThresholdWake(threshold, patience, refractory_s)
        self._model = None

    def score(self, frame: bytes) -> float:
        import numpy as np

        model = self._load()
        audio = np.frombuffer(frame, dtype=np.int16)
        prediction = model.predict(audio)
        if not prediction:
            return 0.0
        return float(max(prediction.values()))

    def observe(self, score: float, now: float | None = None) -> bool:
        return self.logic.observe(score, now)

    def reset(self) -> None:
        """Drop the model. The next score loads a fresh one, which clears a stuck high score."""
        self._model = None
        self.logic.reset_refractory()

    def _load(self):
        if self._model is not None:
            return self._model
        import openwakeword
        from openwakeword.model import Model

        _force_cpu_provider()

        path = Path(self.model_spec)
        if path.is_file():
            models = [str(path)]
            self.model_name = path.name
        else:
            self.model_dir.mkdir(parents=True, exist_ok=True)
            openwakeword.utils.download_models(
                model_names=[self.model_spec],
                target_directory=str(self.model_dir),
            )
            matches = sorted(self.model_dir.glob(f"*{self.model_spec}*.onnx"))
            if not matches:
                matches = sorted(self.model_dir.glob("*.onnx"))
            if not matches:
                raise FileNotFoundError(
                    f"No ONNX wake model for {self.model_spec!r} in {self.model_dir}. "
                    "Run `gva models download` on the device."
                )
            # Prefer the classifier, not the melspectrogram or embedding backbone.
            chosen = [item for item in matches if "embed" not in item.name and "mel" not in item.name]
            models = [str((chosen or matches)[0])]
            self.model_name = Path(models[0]).name
        self._model = Model(wakeword_models=models, inference_framework="onnx")
        return self._model


def install_cpu_provider(ort) -> None:
    """Make every new ONNX session ask for CPU only. openWakeWord does not take a provider list."""
    if getattr(ort, "_gva_cpu_only", False):
        return
    original = ort.InferenceSession

    def session(*args, **kwargs):
        kwargs["providers"] = ["CPUExecutionProvider"]
        return original(*args, **kwargs)

    ort.InferenceSession = session
    ort._gva_cpu_only = True


def _force_cpu_provider() -> None:
    try:
        import onnxruntime as ort
    except Exception:
        return
    install_cpu_provider(ort)


# What the owner can say before a Milo model exists.
DEFAULT_WAKE_PHRASE = "hey jarvis"
DEFAULT_WAKE_MODEL = "hey_jarvis"
