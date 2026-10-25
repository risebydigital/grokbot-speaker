"""Live Pi fixes: streaming playback, empty tools, state dir, deploy, systemd."""

import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from gva.cli import app
from gva.config import env_value, load_config, load_service_env
from gva.paths import project_root
from gva.runtime.audio import (
    FALLBACK_PHRASE,
    JitterBuffer,
    SoundDeviceSink,
    StreamingResampler,
    load_fallback_pcm,
    play_fallback_phrase,
    tone,
)
from gva.runtime.bootstrap import doctor_checks, prepare_dirs, state_dir
from gva.runtime.conversation import render_system
from gva.runtime.realtime import RealtimeProtocol
from gva.runtime.tools.registry import load_registry
from gva.telemetry.events import EventLog
from gva.telemetry.store import Store
from gva.config import Config


def test_streaming_resampler_matches_one_shot_at_2x():
    pcm = tone(200, freq=440, amp=8000, sample_rate=24000)
    once = np.frombuffer(StreamingResampler(24000, 48000).process(pcm), dtype=np.int16)
    chunked = StreamingResampler(24000, 48000)
    samples = np.frombuffer(pcm, dtype=np.int16)
    parts = [chunked.process(samples[i : i + 240].tobytes()) for i in range(0, samples.size, 240)]
    merged = np.frombuffer(b"".join(parts), dtype=np.int16)
    assert merged.size == once.size == samples.size * 2
    assert int(np.max(np.abs(merged.astype(np.int32) - once.astype(np.int32)))) == 0


def test_jitter_prebuffer_and_underrun_silence():
    buffer = JitterBuffer(prebuffer_frames=100, channels=1)
    buffer.push(np.zeros(50, dtype=np.int16).tobytes())
    assert buffer.pull(10).sum() == 0
    assert buffer.underruns == 0
    buffer.push(np.ones(100, dtype=np.int16).tobytes())
    heard = buffer.pull(10)
    assert heard.shape == (10, 1)
    assert buffer.underruns == 0
    buffer.pull(1000)
    assert buffer.underruns == 1
    buffer.pull(10)
    drained = buffer.underruns
    buffer.pull(10)
    assert buffer.underruns == drained


def test_sink_keeps_one_stream_and_prebuffers():
    sink = SoundDeviceSink(0, 24000, prebuffer_ms=200, latency="high")
    assert sink.prebuffer_ms == 200
    assert sink.latency == "high"
    assert sink.buffer.prebuffer_frames == 4800
    marker = object()
    sink._stream = marker
    sink.stop()
    sink.resume()
    assert sink._stream is marker


def test_fallback_phrase_is_bundled_audio():
    pcm, rate = load_fallback_pcm()
    assert rate == 24000
    assert len(pcm) > 24000
    from gva.runtime.audio import CollectingSink

    sink = CollectingSink()
    heard = play_fallback_phrase(sink, Config(), FALLBACK_PHRASE)
    assert sink.chunks
    assert heard > 500


def test_playback_config_defaults():
    cfg = load_config(project_root() / "config" / "config.example.yaml")
    assert cfg.audio.playback_prebuffer_ms == 200
    assert cfg.audio.playback_latency == "high"


def _stack(tmp_path):
    cfg = Config()
    cfg.telemetry.log_dir = str(tmp_path / "log")
    cfg.telemetry.db_path = str(tmp_path / "metrics.db")
    store = Store(tmp_path / "metrics.db")
    events = EventLog("milo-01", "dev", tmp_path / "log", store, validate=True)
    tools = load_registry(cfg, store)
    return cfg, store, events, tools


def test_empty_calendar_tool_still_asks_the_model_to_speak(tmp_path):
    cfg, store, events, tools = _stack(tmp_path)
    spoken = []
    protocol = RealtimeProtocol(cfg, events, store, tools, _Sink())
    protocol.fallback_player = lambda sink, config, phrase: spoken.append(phrase) or 900
    update = protocol.session_update()
    described = {item["name"]: item["description"] for item in update["session"]["tools"]}
    assert "not connected" in described["get_calendar_events"].lower()
    assert "not connected" in render_system(cfg).lower()

    protocol.on_event({"type": "response.created", "response": {"id": "r1"}})
    messages = protocol.on_event(
        {
            "type": "response.function_call_arguments.done",
            "name": "get_calendar_events",
            "call_id": "call-1",
            "arguments": "{}",
        }
    )
    assert messages[0]["item"]["type"] == "function_call_output"
    body = json.loads(messages[0]["item"]["output"])
    assert body["error"] == "calendar not connected"
    assert body["ok"] is False
    assert spoken == []

    follow = protocol.on_event({"type": "response.done", "response": {"id": "r1"}})
    assert follow == [{"type": "response.create"}]
    assert spoken == []
    assert store.recent("playback.completed", 1)[0]["audio_ms"] == 0

    protocol.on_event({"type": "response.created", "response": {"id": "r2"}})
    protocol.on_event({"type": "response.done", "response": {"id": "r2"}})
    assert spoken == [FALLBACK_PHRASE]
    latest = store.recent("playback.completed", 1)[0]
    assert latest["audio_ms"] == 900
    assert latest["underruns"] == 0
    assert store.recent("turn.completed", 1)[0]["fallback_spoken"] is True
    completed = store.recent("tool.completed", 1)[0]
    assert completed["status"] == "error"
    assert completed["tool_name"] == "get_calendar_events"


def test_playback_and_stt_report_real_durations(tmp_path):
    cfg, store, events, tools = _stack(tmp_path)
    protocol = RealtimeProtocol(cfg, events, store, tools, _Sink())
    protocol.audio_message(b"\x01\x00" * 4800)
    time.sleep(0.02)
    protocol._emit_stt("what reports are due next week")
    stt = store.recent("stt.completed", 1)[0]
    assert stt["audio_ms"] == 100
    assert stt["duration_ms"] >= 10

    pcm = b"\x02\x00" * 2400
    protocol.on_event({"type": "response.created", "response": {"id": "r"}})
    protocol.on_event({"type": "response.output_audio.delta", "delta": _b64(pcm)})
    protocol.on_event({"type": "response.done", "response": {"id": "r"}})
    playback = store.recent("playback.completed", 1)[0]
    assert playback["audio_ms"] == 100
    assert playback["underruns"] == 0
    assert playback["interrupted"] is False


def test_state_dir_is_never_the_app_checkout(monkeypatch, tmp_path):
    monkeypatch.setattr("gva.runtime.bootstrap._writable", lambda path: False)
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    cfg = Config()
    cfg.telemetry.log_dir = "/var/log/gva"
    cfg.telemetry.db_path = "/var/lib/gva/metrics.db"
    prepared = prepare_dirs(cfg)
    assert prepared.telemetry.db_path == str(home / ".local" / "state" / "gva" / "metrics.db")
    assert prepared.telemetry.log_dir == str(home / ".local" / "state" / "gva" / "log")
    assert project_root().as_posix() not in prepared.telemetry.db_path
    assert not (project_root() / ".gva").exists()
    monkeypatch.setenv("INVOCATION_ID", "service")
    cfg2 = Config()
    with pytest.raises(OSError):
        prepare_dirs(cfg2)
    assert not (project_root() / ".gva").exists()
    assert state_dir() == home / ".local" / "state" / "gva"


def test_env_file_loads_without_printing_secrets(monkeypatch, tmp_path, capsys):
    secret = "local-test-secret"
    pepper = "p" * 40
    env_file = tmp_path / "gva.env"
    env_file.write_text(f"XAI_API_KEY={secret}\nGVA_HASH_PEPPER={pepper}\n# comment\n", encoding="utf-8")
    monkeypatch.setenv("GVA_ENV_FILE", str(env_file))
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("GVA_HASH_PEPPER", raising=False)
    assert load_service_env() == "loaded"
    assert env_value("XAI_API_KEY") == secret
    cfg = Config()
    cfg.telemetry.log_dir = str(tmp_path / "log")
    cfg.telemetry.db_path = str(tmp_path / "metrics.db")
    problems, warnings = doctor_checks(cfg)
    text = " ".join(problems + warnings)
    assert secret not in text
    assert pepper not in text
    assert "XAI_API_KEY is not set" not in text
    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err


def test_unreadable_env_is_not_reported_as_missing(monkeypatch, tmp_path):
    env_file = tmp_path / "gva.env"
    env_file.write_text("XAI_API_KEY=hidden\n", encoding="utf-8")
    monkeypatch.setenv("GVA_ENV_FILE", str(env_file))
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("GVA_HASH_PEPPER", raising=False)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)

    def denied(path):
        raise PermissionError("denied")

    monkeypatch.setattr("gva.config._read_env", denied)
    cfg = Config()
    cfg.telemetry.log_dir = str(tmp_path / "log")
    cfg.telemetry.db_path = str(tmp_path / "metrics.db")
    problems, warnings = doctor_checks(cfg)
    text = " ".join(problems + warnings)
    assert "Cannot read /etc/gva/gva.env" in text
    assert "XAI_API_KEY is not set" not in text
    assert "GVA_HASH_PEPPER is missing" not in text
    assert "hidden" not in text


def test_sync_run_idles_when_supabase_is_unset(monkeypatch, tmp_path):
    monkeypatch.setenv("GVA_ENV_FILE", str(tmp_path / "missing.env"))
    monkeypatch.setenv("GVA_LOG_DIR", str(tmp_path / "log"))
    monkeypatch.setenv("GVA_DB_PATH", str(tmp_path / "metrics.db"))
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)
    result = CliRunner().invoke(app, ["sync", "run"])
    assert result.exit_code == 0
    assert "idle" in result.output.lower()
    assert "xai-" not in result.output


def test_doctor_does_not_create_a_checkout_dot_gva(monkeypatch, tmp_path):
    monkeypatch.setenv("GVA_ENV_FILE", str(tmp_path / "missing.env"))
    monkeypatch.setenv("GVA_LOG_DIR", "/var/log/gva")
    monkeypatch.setenv("GVA_DB_PATH", "/var/lib/gva/metrics.db")
    monkeypatch.setattr("gva.runtime.bootstrap._writable", lambda path: False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "home"))
    result = CliRunner().invoke(app, ["doctor"])
    assert "PermissionError" not in (result.output or "")
    assert result.exception is None or not isinstance(result.exception, PermissionError)
    assert not (project_root() / ".gva").exists()


def test_start_limits_live_in_the_unit_section():
    root = project_root()
    for name in ("gva-voice.service", "gva-sync.service", "gva-backend.service"):
        text = (root / "deploy" / "systemd" / name).read_text(encoding="utf-8")
        unit, service = text.split("[Service]", 1)
        assert "StartLimitIntervalSec" in unit
        assert "StartLimitBurst" in unit
        assert "StartLimit" not in service


def test_update_pulls_from_the_operator_checkout(tmp_path):
    root = project_root()
    install = (root / "deploy" / "install.sh").read_text(encoding="utf-8")
    update = (root / "deploy" / "update.sh").read_text(encoding="utf-8")
    sync = (root / "deploy" / "git-sync.sh").read_text(encoding="utf-8")
    assert "sync_app_checkout" in install
    assert "sync_app_checkout" in update
    assert "safe.directory" in sync
    assert "SUDO_USER" in sync
    assert "/milo" in sync or "milo" in sync
    assert "is-enabled" in sync
    assert "daemon-reload" in update
    assert "gva-voice.service" in update
    assert "supabase_configured" in install
    for line in install.splitlines():
        if line.strip().startswith("systemctl enable --now") and "gva-voice.service" in line:
            assert "gva-sync" not in line
    subprocess.run(["bash", "-n", str(root / "deploy" / "git-sync.sh")], check=True)
    placeholder = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; SUPABASE_URL="https://<project-ref>.supabase.co"; '
            'SUPABASE_SERVICE_KEY="<SUPABASE_SERVICE_ROLE_KEY>"; '
            "if supabase_configured; then exit 1; fi; "
            'SUPABASE_URL=""; SUPABASE_SERVICE_KEY=""; '
            "if supabase_configured; then exit 1; fi; "
            'SUPABASE_URL="https://example.supabase.co"; SUPABASE_SERVICE_KEY="service-role"; '
            "supabase_configured",
            "bash",
            str(root / "deploy" / "git-sync.sh"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert placeholder.returncode == 0, placeholder.stderr

    env = os.environ.copy()
    env.update(
        {
            "GIT_AUTHOR_NAME": "gva",
            "GIT_AUTHOR_EMAIL": "gva@localhost",
            "GIT_COMMITTER_NAME": "gva",
            "GIT_COMMITTER_EMAIL": "gva@localhost",
            "GIT_CONFIG_GLOBAL": str(tmp_path / "gitconfig"),
            "GIT_CONFIG_SYSTEM": str(tmp_path / "gitconfig"),
            "SUDO_USER": "operator",
            "GVA_OPERATOR_HOME": str(tmp_path / "home"),
        }
    )
    (tmp_path / "gitconfig").write_text("", encoding="utf-8")
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=app_dir, env=env, check=True, capture_output=True)
    (app_dir / "README.md").write_text("one\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=app_dir, env=env, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "one"], cwd=app_dir, env=env, check=True, capture_output=True)
    milo = tmp_path / "home" / "operator" / "milo"
    milo.parent.mkdir(parents=True)
    subprocess.run(["git", "clone", str(app_dir), str(milo)], env=env, check=True, capture_output=True)
    (milo / "README.md").write_text("two\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=milo, env=env, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "two"], cwd=milo, env=env, check=True, capture_output=True)
    chown = subprocess.run(["chown", "-R", "nobody:nogroup", str(app_dir)], capture_output=True)
    if chown.returncode == 0:
        probe = subprocess.run(["git", "-C", str(app_dir), "status"], env=env, capture_output=True, text=True)
        assert probe.returncode != 0
        assert "dubious" in probe.stderr.lower()
    script = root / "deploy" / "git-sync.sh"
    subprocess.run(
        ["bash", "-c", f'source "{script}" && sync_app_checkout "{app_dir}"'],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert (app_dir / "README.md").read_text(encoding="utf-8") == "two\n"


class _Sink:
    underruns = 0

    def write(self, pcm):
        return None

    def stop(self):
        return None

    def resume(self):
        return None

    def consume_underruns(self):
        return 0


def _b64(pcm: bytes) -> str:
    import base64

    return base64.b64encode(pcm).decode("ascii")
