"""Pi install, ONNX wake models, speaker format, and config permissions."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gva.config import load_config
from gva.paths import project_root
from gva.runtime.audio import (
    JitterBuffer,
    SoundDeviceSink,
    choose_stream_format,
    fit_capture,
    fit_playback,
    tone,
)
from gva.runtime.bootstrap import voice_import_problems
from gva.runtime.wake import (
    INFERENCE_FRAMEWORK,
    OpenWakeWordEngine,
    download_onnx_models,
    onnx_download_urls,
    onnx_runtime_kwargs,
)


def _catalog():
    return SimpleNamespace(
        FEATURE_MODELS={
            "melspectrogram": {"download_url": "https://example.invalid/melspectrogram.tflite"},
            "embedding": {"download_url": "https://example.invalid/embedding_model.tflite"},
        },
        VAD_MODELS={"silero_vad": {"download_url": "https://example.invalid/silero_vad.onnx"}},
        MODELS={
            "hey_jarvis": {"download_url": "https://example.invalid/hey_jarvis_v0.1.tflite"},
            "alexa": {"download_url": "https://example.invalid/alexa_v0.1.tflite"},
        },
    )


def test_sanyun_format_is_48k_stereo():
    def supported(rate, channels):
        return rate == 48000 and channels == 2

    assert choose_stream_format(24000, 48000, 2, supported) == (48000, 2)
    assert choose_stream_format(16000, 48000, 2, supported) == (48000, 2)


def test_source_rate_mono_wins_when_the_device_allows_it():
    def supported(rate, channels):
        return rate == 16000 and channels == 1

    assert choose_stream_format(16000, 48000, 2, supported) == (16000, 1)


def test_fit_playback_resamples_and_upmixes():
    pcm = tone(100, sample_rate=16000)
    fitted = fit_playback(pcm, 16000, 48000, 2)
    samples = np.frombuffer(fitted, dtype=np.int16)
    assert samples.size == 1600 * 3 * 2
    assert samples[0] == samples[1]


def test_fit_capture_downmixes_and_resamples():
    stereo = np.column_stack([np.full(4800, 1000), np.full(4800, 3000)]).astype(np.int16)
    audio = np.frombuffer(fit_capture(stereo, 48000, 16000), dtype=np.int16)
    assert audio.size == 1600
    assert int(audio.mean()) == 2000


def test_sink_buffers_device_format(monkeypatch):
    sink = SoundDeviceSink(2, 24000)
    monkeypatch.setattr(sink, "prepare", lambda: None)
    monkeypatch.setattr(sink, "_ensure", lambda: None)
    sink.device_rate = 48000
    sink.channels = 2
    sink.buffer = JitterBuffer(0, 2)
    sink._format_ready = True
    sink.write(tone(10, sample_rate=24000))
    frame = sink.buffer.pull(1)
    assert frame.shape == (1, 2)


def test_onnx_urls_do_not_fetch_tflite():
    catalog = _catalog()
    urls = onnx_download_urls(
        ["hey_jarvis"],
        catalog.FEATURE_MODELS,
        catalog.VAD_MODELS,
        catalog.MODELS,
    )
    assert urls == [
        "https://example.invalid/melspectrogram.onnx",
        "https://example.invalid/embedding_model.onnx",
        "https://example.invalid/silero_vad.onnx",
        "https://example.invalid/hey_jarvis_v0.1.onnx",
    ]
    assert all(url.endswith(".onnx") for url in urls)
    support = onnx_download_urls([], catalog.FEATURE_MODELS, catalog.VAD_MODELS, catalog.MODELS)
    assert len(support) == 3


def test_download_writes_onnx_files_only(tmp_path):
    seen = []

    def fetch(url, dest: Path):
        seen.append(url)
        dest.write_bytes(b"onnx")

    saved = download_onnx_models(["hey_jarvis"], tmp_path, fetch=fetch, catalog=_catalog())
    assert seen
    assert all(url.endswith(".onnx") for url in seen)
    assert all(path.suffix == ".onnx" for path in saved)
    assert (tmp_path / "hey_jarvis_v0.1.onnx").read_bytes() == b"onnx"


def test_model_kwargs_are_onnx():
    kwargs = onnx_runtime_kwargs(["/usr/share/gva/wake/hey_jarvis_v0.1.onnx"], Path("/usr/share/gva/wake"))
    assert kwargs["inference_framework"] == INFERENCE_FRAMEWORK == "onnx"
    assert kwargs["melspec_model_path"].endswith("melspectrogram.onnx")
    assert kwargs["embedding_model_path"].endswith("embedding_model.onnx")


def test_engine_constructs_openwakeword_with_onnx(monkeypatch, tmp_path):
    captured = {}

    class FakeModel:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def predict(self, audio):
            return {"hey_jarvis": 0.1}

    def fake_download(names, directory, fetch=None, catalog=None):
        Path(directory, "hey_jarvis_v0.1.onnx").write_bytes(b"w")
        return []

    monkeypatch.setattr("gva.runtime.wake.download_onnx_models", fake_download)
    module = type(sys)("openwakeword")
    model_module = type(sys)("openwakeword.model")
    model_module.Model = FakeModel
    monkeypatch.setitem(sys.modules, "openwakeword", module)
    monkeypatch.setitem(sys.modules, "openwakeword.model", model_module)
    engine = OpenWakeWordEngine("hey_jarvis", str(tmp_path), 0.6, 2, 0, "hey jarvis")
    assert engine.score(b"\x00\x00" * 160) == 0.1
    assert captured["inference_framework"] == "onnx"
    assert captured["wakeword_models"][0].endswith("hey_jarvis_v0.1.onnx")


def test_doctor_names_missing_voice_packages():
    problems = voice_import_problems(find_spec=lambda name: None)
    text = " ".join(problems)
    assert "sounddevice" in text
    assert "openwakeword" in text
    assert "--no-deps" in text


def test_unreadable_config_falls_back(monkeypatch, capsys, tmp_path):
    blocked = tmp_path / "config.yaml"
    blocked.write_text("device:\n  id: hidden-id\n", encoding="utf-8")
    monkeypatch.setenv("GVA_CONFIG", str(blocked))
    monkeypatch.delenv("GVA_DEVICE_ID", raising=False)
    real = Path.is_file

    def is_file(self):
        if self == blocked:
            raise PermissionError("denied")
        return real(self)

    monkeypatch.setattr(Path, "is_file", is_file)
    cfg = load_config()
    err = capsys.readouterr().err
    assert cfg.device.id == "milo-01"
    assert "PermissionError" in err
    assert "0600" in err


def test_install_scripts_fail_closed_and_use_onnx_wake():
    root = project_root()
    voice = (root / "deploy" / "voice-packages.sh").read_text(encoding="utf-8")
    install = (root / "deploy" / "install.sh").read_text(encoding="utf-8")
    update = (root / "deploy" / "update.sh").read_text(encoding="utf-8")
    permissions = (root / "deploy" / "permissions.sh").read_text(encoding="utf-8")
    project = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert "--no-deps 'openwakeword>=0.6.0'" in voice
    for package in (
        "faster-whisper",
        "piper-tts",
        "sounddevice",
        "onnxruntime",
        "scipy",
        "scikit-learn",
        "tqdm",
        "requests",
    ):
        assert package in voice
    assert "install_voice_python" in install
    assert "Voice extras did not install" not in install
    assert "install_voice_python" in update
    assert '[voice]' not in update
    assert "chmod 644" in permissions
    assert "chmod 600" in permissions
    assert "chmod 755 /etc/gva" in permissions
    assert "openwakeword>=" not in project
    for script in ("deploy/install.sh", "deploy/update.sh", "deploy/voice-packages.sh", "deploy/permissions.sh"):
        subprocess.run(["bash", "-n", str(root / script)], check=True)
