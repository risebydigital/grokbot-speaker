"""Directories and the production readiness checks shared by the CLI."""

from __future__ import annotations

import os
from pathlib import Path

from gva.config import Config, env_value
from gva.paths import project_root


def prepare_dirs(cfg: Config) -> Config:
    """Use /var paths on the Pi. On a laptop, fall back to ./.gva when those are not writable."""
    systemd = bool(os.environ.get("INVOCATION_ID"))
    if _writable(Path(cfg.telemetry.log_dir)) and _writable(Path(cfg.telemetry.db_path).parent):
        return cfg
    if systemd:
        raise OSError(f"cannot write {cfg.telemetry.log_dir} or {cfg.telemetry.db_path}")
    root = project_root() / ".gva"
    cfg.telemetry.log_dir = str(root / "log")
    cfg.telemetry.db_path = str(root / "metrics.db")
    Path(cfg.telemetry.log_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.telemetry.db_path).parent.mkdir(parents=True, exist_ok=True)
    return cfg


def _writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def doctor_checks(cfg: Config) -> tuple[list[str], list[str]]:
    """Return (problems, warnings). Problems should stop the voice service from starting."""
    problems: list[str] = []
    warnings: list[str] = []
    pepper = env_value("GVA_HASH_PEPPER")
    if not pepper or len(pepper) < 32:
        problems.append("GVA_HASH_PEPPER is missing or shorter than 32 characters")
    if not env_value("XAI_API_KEY"):
        problems.append("XAI_API_KEY is not set")
    url = env_value("SUPABASE_URL")
    key = env_value("SUPABASE_SERVICE_KEY")
    if bool(url) != bool(key):
        problems.append("SUPABASE_URL and SUPABASE_SERVICE_KEY must both be set, or both left unset")
    if not url:
        warnings.append("Supabase is unset, so telemetry will stay on the Pi")
    try:
        prepare_dirs(cfg)
    except OSError as exc:
        problems.append(str(exc))
    if cfg.audio.input_device == "":
        warnings.append("audio input substring is empty; the default microphone will be used")
    if cfg.audio.output_device == "":
        warnings.append("speaker output is automatic: USB playback, then the 3.5 mm jack")
    model = cfg.wake.model
    if model.endswith(".onnx") and not Path(model).is_file():
        warnings.append(f"wake model file not found: {model}")
    voice = Path(cfg.tts.model_dir) / f"{cfg.tts.voice}.onnx"
    if not voice.is_file():
        warnings.append(f"Piper voice not downloaded yet ({voice.name}). Run: gva models download")
    return problems, warnings
