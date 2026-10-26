"""Wake-word scoring.

The stock model is openWakeWord's hey_jarvis ("hey jarvis"). `wake.model: milo`
loads the shipped Milo ONNX file, which is trained to accept "milo" and
"hey milo". If that file is missing, scoring falls back to hey_jarvis.

Models are ONNX. tflite-runtime has no wheel for Python 3.13, so this process
never asks openWakeWord for the TFLite interpreter.
"""

from __future__ import annotations

import shutil
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

from gva.paths import project_root
from gva.runtime.audio import rms

# openWakeWord's own default is tflite. Milo always passes this instead.
INFERENCE_FRAMEWORK = "onnx"


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


def onnx_model_url(url: str) -> str:
    """Rewrite a published model URL so the download is the ONNX file."""
    if url.endswith(".tflite"):
        return url[: -len(".tflite")] + ".onnx"
    return url


def onnx_download_urls(
    model_names: list[str] | None,
    feature_models: dict[str, Any],
    vad_models: dict[str, Any],
    models: dict[str, Any],
) -> list[str]:
    """Feature, VAD, and requested wake models. Every URL ends in .onnx.

    `None` downloads every classifier. An empty list downloads only the
    melspectrogram, embedding, and Silero files.
    """
    urls: list[str] = []

    def add(url: str) -> None:
        rewritten = onnx_model_url(str(url))
        if not rewritten.endswith(".onnx"):
            raise ValueError(f"refusing a non-ONNX model url: {url}")
        if rewritten not in urls:
            urls.append(rewritten)

    for table in (feature_models, vad_models):
        for item in table.values():
            add(item["download_url"])
    if model_names is None:
        chosen = list(models.values())
    else:
        chosen = []
        for name in model_names:
            stem = Path(name).name
            if not stem or stem.endswith(".onnx") or "/" in name or name.startswith("."):
                continue
            for key, item in models.items():
                filename = str(item["download_url"]).rsplit("/", 1)[-1]
                onnx_name = filename.replace(".tflite", ".onnx")
                if stem == key or stem in filename or stem in onnx_name:
                    chosen.append(item)
                    break
    for item in chosen:
        add(item["download_url"])
    return urls


def onnx_runtime_kwargs(model_paths: list[str], model_dir: Path) -> dict[str, Any]:
    """Arguments for openwakeword.model.Model. Always ONNX, including the feature nets."""
    directory = Path(model_dir)
    return {
        "wakeword_models": model_paths,
        "inference_framework": INFERENCE_FRAMEWORK,
        "melspec_model_path": str(directory / "melspectrogram.onnx"),
        "embedding_model_path": str(directory / "embedding_model.onnx"),
    }


def download_onnx_models(
    model_names: list[str] | None,
    target_directory: str,
    fetch: Callable[[str, Path], None] | None = None,
    catalog: Any | None = None,
) -> list[Path]:
    """Download ONNX wake models into `target_directory`. Existing files are left in place."""
    directory = Path(target_directory)
    directory.mkdir(parents=True, exist_ok=True)
    if catalog is None:
        import openwakeword

        catalog = openwakeword
    urls = onnx_download_urls(model_names, catalog.FEATURE_MODELS, catalog.VAD_MODELS, catalog.MODELS)
    saved: list[Path] = []
    for url in urls:
        dest = directory / url.rsplit("/", 1)[-1]
        saved.append(dest)
        if dest.is_file() and dest.stat().st_size > 0:
            continue
        if fetch is None:
            _fetch_url(url, dest)
        else:
            fetch(url, dest)
    return saved


def _fetch_url(url: str, dest: Path) -> None:
    temporary = dest.with_suffix(dest.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(dest)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


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
        resolved, fell_back = resolve_wake_model(model, model_dir)
        self.requested_model = model
        self.fell_back = fell_back
        self.model_spec = resolved
        self.model_dir = Path(model_dir)
        self.phrase = phrase
        self.model_name = Path(resolved).name if resolved.endswith(".onnx") else resolved
        self.logic = ThresholdWake(threshold, patience, refractory_s)
        self._model = None
        if fell_back:
            print(
                "milo.onnx was not found in the wake directory or in assets/wakewords. "
                'Using hey_jarvis. Say "hey jarvis".',
                file=sys.stderr,
            )

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
        from openwakeword.model import Model

        _force_cpu_provider()

        path = Path(self.model_spec)
        download_onnx_models([] if path.is_file() else [self.model_spec], str(self.model_dir))
        if path.is_file():
            models = [str(path)]
            self.model_name = path.name
        else:
            matches = sorted(self.model_dir.glob(f"*{self.model_spec}*.onnx"))
            if not matches:
                matches = sorted(
                    item
                    for item in self.model_dir.glob("*.onnx")
                    if "embed" not in item.name and "mel" not in item.name and "silero" not in item.name
                )
            if not matches:
                raise FileNotFoundError(
                    f"No ONNX wake model for {self.model_spec!r} in {self.model_dir}. "
                    "Run `gva models download` on the device."
                )
            # Prefer the classifier, not the melspectrogram or embedding backbone.
            chosen = [item for item in matches if "embed" not in item.name and "mel" not in item.name]
            models = [str((chosen or matches)[0])]
            self.model_name = Path(models[0]).name
        self._model = Model(**onnx_runtime_kwargs(models, self.model_dir))
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


# Stock openWakeWord model. Used when no Milo file is configured, and when
# wake.model is milo but milo.onnx is not on disk.
DEFAULT_WAKE_PHRASE = "hey jarvis"
DEFAULT_WAKE_MODEL = "hey_jarvis"
MILO_WAKE_ID = "milo"
MILO_FILENAME = "milo.onnx"


def shipped_wake_model() -> Path:
    """ONNX file shipped in the git tree. Deploy copies it to the wake directory."""
    return project_root() / "assets" / "wakewords" / MILO_FILENAME


def is_milo_model(model: str) -> bool:
    return Path(model).name.lower() in {MILO_WAKE_ID, MILO_FILENAME}


def resolve_wake_model(model: str, model_dir: str) -> tuple[str, bool]:
    """Return `(spec, fell_back)`.

    An existing ONNX path is used as-is. `milo` uses milo.onnx from the wake
    directory, then the copy shipped in the repo. If neither file is there,
    the spec is hey_jarvis and `fell_back` is true.
    """
    candidate = Path(model)
    if candidate.suffix == ".onnx":
        try:
            if candidate.is_file() and candidate.stat().st_size > 0:
                return str(candidate), False
        except OSError:
            pass
    if not is_milo_model(model):
        return model, False
    search = [Path(model_dir) / MILO_FILENAME, shipped_wake_model()]
    for path in search:
        try:
            if path.is_file() and path.stat().st_size > 0:
                return str(path), False
        except OSError:
            continue
    return DEFAULT_WAKE_MODEL, True


def copy_shipped_milo(model_dir: str | Path) -> Path | None:
    """Copy the shipped Milo model into the wake directory. Missing source is not an error."""
    source = shipped_wake_model()
    try:
        if not source.is_file() or source.stat().st_size <= 0:
            return None
        dest = Path(model_dir) / MILO_FILENAME
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.resolve() != source.resolve():
            same = dest.is_file() and dest.stat().st_size == source.stat().st_size
            if not same:
                shutil.copyfile(source, dest)
        return dest
    except OSError:
        return None


def format_wake_line(elapsed_s: float, score: float, fired: bool) -> str:
    mark = "WAKE" if fired else "."
    return f"{elapsed_s:6.2f}s  score={score:.3f}  {mark}"


def score_report(engine: Any, frames: list[bytes], chunk_s: float = 0.08) -> tuple[list[str], float]:
    """Score already-captured frames. Returns lines and the maximum score."""
    lines: list[str] = []
    peak = 0.0
    elapsed = 0.0
    for frame in frames:
        score = float(engine.score(frame))
        fired = bool(engine.observe(score))
        peak = max(peak, score)
        lines.append(format_wake_line(elapsed, score, fired))
        elapsed += chunk_s
    return lines, peak
