"""`gva` command line. Secrets are read from the environment and never printed in full."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import jwt
import typer

from gva import __version__
from gva.backend.alerts import evaluate, seed_defaults
from gva.backend.ingest import ingest_logs
from gva.backend.retention import enforce
from gva.backend.sync_supabase import client_from_env, push_eval, sync_once
from gva.config import (
    ConfigChangeError,
    env_value,
    get_dotted,
    load_config,
    masked_secrets,
    save_config,
    set_dotted,
    validate_change,
)
from gva.eval.runner import EvalReport, run_fixtures
from gva.ids import new_ulid, utc_now_ms
from gva.runtime.audio import list_devices
from gva.runtime.bootstrap import doctor_checks, prepare_dirs
from gva.telemetry.events import EventLog, tail
from gva.telemetry.store import Store

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Grok Voice Assistant on the Raspberry Pi.")
audio_app = typer.Typer(help="Microphones and speakers.")
config_app = typer.Typer(help="Non-secret configuration.")
logs_app = typer.Typer(help="JSONL event log.")
tools_app = typer.Typer(help="Function-calling tools.")
models_app = typer.Typer(help="Wake-word and voice downloads.")
auth_app = typer.Typer(help="Local API tokens.")
maintenance_app = typer.Typer(help="Retention and backups.")
sync_app = typer.Typer(help="Supabase sync.")
app.add_typer(audio_app, name="audio")
app.add_typer(config_app, name="config")
app.add_typer(logs_app, name="logs")
app.add_typer(tools_app, name="tools")
app.add_typer(models_app, name="models")
app.add_typer(auth_app, name="auth")
app.add_typer(maintenance_app, name="maintenance")
app.add_typer(sync_app, name="sync")


def _cfg():
    cfg = prepare_dirs(load_config())
    return cfg


def _store(cfg):
    return Store(Path(cfg.telemetry.db_path))


def _events(cfg, store, source: str = "cli"):
    log = EventLog(
        device_id=cfg.device.id,
        env=cfg.device.env,
        log_dir=Path(cfg.telemetry.log_dir),
        store=store,
        validate=cfg.telemetry.validate_events,
    )
    log.ctx.source = source
    return log


@app.command()
def run() -> None:
    """Start the always-on voice loop."""
    cfg = _cfg()
    problems, _warnings = doctor_checks(cfg)
    if problems:
        for problem in problems:
            typer.echo(f"not ready: {problem}", err=True)
        raise typer.Exit(1)
    from gva.runtime.build import build_assistant

    assistant = build_assistant(cfg)
    assistant.run()


@app.command()
def ask(text: str) -> None:
    """Send one typed utterance through the tool loop."""
    cfg = _cfg()
    from gva.runtime.build import build_local

    assistant = build_local(cfg)
    result = assistant.ask(text)
    typer.echo(result.spoken or "")


@app.command()
def say(text: str) -> None:
    """Speak a sentence through the configured voice."""
    cfg = _cfg()
    from gva.runtime.build import build_local

    build_local(cfg).say(text)


@app.command()
def listen() -> None:
    """Capture one utterance and print the transcript. Nothing is sent to xAI."""
    cfg = _cfg()
    devices = list_devices()
    if devices and devices[0].get("error"):
        typer.echo(devices[0]["error"], err=True)
        raise typer.Exit(1)
    from gva.runtime.audio import SoundDeviceCapture, resolve_device
    from gva.runtime.stt import FasterWhisperStt
    from gva.runtime.vad import Endpointer, EnergyVad

    capture = SoundDeviceCapture(
        resolve_device(cfg.audio.input_device, "input"),
        cfg.audio.sample_rate,
        cfg.audio.chunk_ms,
    )
    capture.start()
    endpoint = Endpointer(EnergyVad(cfg.vad.energy_threshold), cfg.vad.end_silence_ms, cfg.vad.max_utterance_seconds, cfg.audio.chunk_ms)
    audio = bytearray()
    try:
        while not endpoint.closed:
            frame = capture.read(0.5)
            if not frame:
                break
            audio.extend(frame)
            endpoint.add(frame)
    finally:
        capture.stop()
    result = FasterWhisperStt(cfg).transcribe(bytes(audio))
    typer.echo(result.text)


@app.command()
def status() -> None:
    """Print version, device id, and the latest heartbeat."""
    cfg = _cfg()
    store = _store(cfg)
    beats = store.recent("heartbeat", 1)
    typer.echo(f"gva {__version__}  device {cfg.device.id}  env {cfg.device.env}")
    if beats:
        beat = beats[0]
        typer.echo(
            f"heartbeat {beat.get('ts')}  cpu {beat.get('cpu_pct')}%  mem {beat.get('mem_pct')}%  "
            f"disk {beat.get('disk_pct')}%  api {beat.get('api_reachable')}"
        )
    else:
        typer.echo("no heartbeat recorded yet")


@app.command()
def doctor() -> None:
    """Check configuration, secrets, and directories before enabling systemd."""
    cfg = _cfg()
    problems, warnings = doctor_checks(cfg)
    for warning in warnings:
        typer.echo(f"warning: {warning}")
    for problem in problems:
        typer.echo(f"problem: {problem}")
    if problems:
        raise typer.Exit(1)
    typer.echo("doctor: ok")


@app.command()
def health() -> None:
    """Exit 0 when config and the hash pepper are usable."""
    cfg = _cfg()
    problems, _warnings = doctor_checks(cfg)
    blocking = [item for item in problems if "PEPPER" in item or "XAI_API_KEY" in item]
    if blocking:
        for item in blocking:
            typer.echo(item, err=True)
        raise typer.Exit(1)
    typer.echo("ok")


@app.command("eval")
def eval_cmd(live: bool = typer.Option(False, "--live", help="Call xAI. Off in CI.")) -> None:
    """Score the golden suite. Without --live this uses fixtures and does not call xAI."""
    if live:
        typer.echo("Live eval is not run from CI. It would call the xAI API and spend credit.", err=True)
        raise typer.Exit(2)
    report = run_fixtures()
    _print_eval(report)


@app.command()
def replay(turn_id: str) -> None:
    """Show raw text only if the local replay cache still holds it."""
    cfg = _cfg()
    text = _store(cfg).replay_text(turn_id)
    if text is None:
        typer.echo("No local replay for that turn. Raw text is not kept by default and is never synced.")
        raise typer.Exit(1)
    typer.echo(text)


@app.command()
def usage() -> None:
    """Today's estimated spend from local llm.completed events."""
    cfg = _cfg()
    from datetime import datetime, timezone

    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    spent = _store(cfg).spend_since(day + "T00:00:00.000Z")
    typer.echo(f"{day}  ${spent:.4f}")


@app.command()
def mute() -> None:
    cfg = _cfg()
    from gva.runtime.build import build_local

    build_local(cfg).mute(True)
    typer.echo("muted")


@app.command()
def unmute() -> None:
    cfg = _cfg()
    from gva.runtime.build import build_local

    build_local(cfg).mute(False)
    typer.echo("listening")


@app.command("test-audio")
def test_audio(seconds: float = typer.Option(2.0, "--seconds", help="How long the tone plays.")) -> None:
    """Play a tone on the speaker Milo will use, and print that device."""
    cfg = _cfg()
    from gva.runtime.audio import SoundDeviceSink, list_devices, prefer_output, tone

    rows = list_devices()
    if rows and rows[0].get("error"):
        typer.echo(rows[0]["error"], err=True)
        raise typer.Exit(1)
    index = prefer_output(rows, cfg.audio.output_device)
    if index is None:
        typer.echo("no playback device found. Run: gva audio devices", err=True)
        raise typer.Exit(1)
    name = next((row["name"] for row in rows if row.get("index") == index), str(index))
    rate = cfg.conversation.realtime.sample_rate if cfg.conversation.engine == "realtime" else cfg.audio.sample_rate
    sink = SoundDeviceSink(index, rate)
    try:
        sink.prepare()
        typer.echo(
            f"playing a {seconds:.1f}s tone on [{index}] {name} "
            f"(source {rate} Hz, device {sink.device_rate} Hz, {sink.channels} ch)"
        )
        sink.write(tone(int(seconds * 1000), freq=440, amp=8000, sample_rate=rate))
        time.sleep(seconds + 0.4)
    except Exception as exc:
        typer.echo(f"playback failed: {exc.__class__.__name__}: {exc}", err=True)
        raise typer.Exit(1) from exc
    finally:
        sink.close()
    typer.echo("done")


@audio_app.command("devices")
def audio_devices() -> None:
    """List ALSA devices. This command does not read /etc/gva/config.yaml."""
    rows = list_devices()
    for row in rows:
        if row.get("error"):
            typer.echo(row["error"], err=True)
            raise typer.Exit(1)
        flags = []
        if row.get("default_input"):
            flags.append("default-in")
        if row.get("default_output"):
            flags.append("default-out")
        typer.echo(
            f"{row['index']:3}  in={row['inputs']} out={row['outputs']}  {row['name']}"
            + (f"  ({', '.join(flags)})" if flags else "")
        )


@config_app.command("show")
def config_show() -> None:
    cfg = _cfg()
    typer.echo(json.dumps(cfg.model_dump(mode="json"), indent=2))
    typer.echo("secrets:")
    for name, presence in masked_secrets().items():
        typer.echo(f"  {name}: {presence}")


@config_app.command("set")
def config_set(key: str, value: str) -> None:
    """Change one allow-listed setting and record config.changed."""
    cfg = _cfg()
    try:
        coerced = validate_change(key, _coerce_cli(value))
    except ConfigChangeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    if key == "action.health_check":
        typer.echo("health check is a dashboard action, not a saved setting")
        raise typer.Exit(1)
    previous = get_dotted(cfg.source_data, key)
    set_dotted(cfg.source_data, key, coerced)
    from gva.backend.sync_supabase import _reload

    _reload(cfg)
    save_config(cfg)
    store = _store(cfg)
    events = _events(cfg, store)
    events.emit(
        "config.changed",
        key=key,
        old_value_hash=hashlib.sha256(json.dumps(previous, default=str).encode()).hexdigest(),
        new_value=coerced,
    )
    typer.echo(f"updated {key}")


@logs_app.command("tail")
def logs_tail(type_: str = typer.Option(None, "--type"), limit: int = 20) -> None:
    cfg = _cfg()
    for event in tail(Path(cfg.telemetry.log_dir), type_, limit):
        typer.echo(json.dumps(event, ensure_ascii=False))


@tools_app.command("list")
def tools_list() -> None:
    cfg = _cfg()
    from gva.runtime.tools.registry import load_registry

    registry = load_registry(cfg, _store(cfg))
    for spec in registry.specs.values():
        typer.echo(f"{spec.name:24} {spec.risk:8} {spec.description.splitlines()[0]}")


@tools_app.command("test")
def tools_test(name: str, arguments: str = "{}") -> None:
    cfg = _cfg()
    from gva.runtime.tools.registry import load_registry

    registry = load_registry(cfg, _store(cfg))
    outcome = registry.run(new_ulid(), name, arguments)
    typer.echo(json.dumps({"status": outcome.status, "result": outcome.result, "error_code": outcome.error_code}))


@models_app.command("download")
def models_download() -> None:
    """Download hey_jarvis and the en_GB Piper voice. Needs a network."""
    cfg = _cfg()
    wake_dir = Path(cfg.wake.model_dir)
    try:
        wake_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        typer.echo(f"cannot create {wake_dir}: {exc.__class__.__name__}", err=True)
        raise typer.Exit(1) from exc
    try:
        from gva.runtime.wake import download_onnx_models

        spec = cfg.wake.model
        names = [] if spec.endswith(".onnx") else [spec]
        download_onnx_models(names, str(wake_dir))
        typer.echo(f"ONNX wake models in {wake_dir}")
    except Exception as exc:
        typer.echo(f"wake download failed: {exc.__class__.__name__}: {exc}", err=True)
        raise typer.Exit(1) from exc
    voice_dir = Path(cfg.tts.model_dir)
    voice_dir.mkdir(parents=True, exist_ok=True)
    voice = cfg.tts.voice
    # rhasspy layout: en/en_GB/alan/low/en_GB-alan-low.onnx
    parts = voice.split("-")
    quality = parts[-1] if len(parts) > 1 else "low"
    speaker = parts[1] if len(parts) > 2 else "alan"
    base = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/{speaker}/{quality}/{voice}"
    import httpx

    for suffix in (".onnx", ".onnx.json"):
        target = voice_dir / f"{voice}{suffix}"
        if target.is_file():
            continue
        try:
            response = httpx.get(base + suffix, follow_redirects=True, timeout=120)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            typer.echo(f"voice download failed: {exc.__class__.__name__}", err=True)
            raise typer.Exit(1) from exc
        target.write_bytes(response.content)
        typer.echo(f"saved {target}")


@models_app.command("fallbacks")
def models_fallbacks() -> None:
    """Write the spoken fallback phrases. Piper renders them when the voice is installed."""
    cfg = _cfg()
    directory = Path("/usr/share/gva/fallback")
    if not directory.exists():
        directory = Path(cfg.telemetry.db_path).parent / "fallback"
    directory.mkdir(parents=True, exist_ok=True)
    phrases = [
        "Sorry, I didn't catch that.",
        "I'm a bit busy, try again in a moment.",
        "I can't reach my brain right now.",
        "Something's wrong with my settings.",
        "I'm offline at the moment.",
        "Sorry, something went wrong.",
        "I'm having trouble hearing.",
        "OK.",
    ]
    (directory / "phrases.txt").write_text("\n".join(phrases) + "\n", encoding="utf-8")
    typer.echo(f"wrote {directory / 'phrases.txt'}")


@auth_app.command("token")
def auth_token(role: str = "owner") -> None:
    """Mint a local JWT. This does not grant Supabase access."""
    if role not in {"viewer", "reviewer", "developer", "owner"}:
        typer.echo("role must be viewer, reviewer, developer, or owner", err=True)
        raise typer.Exit(1)
    secret = env_value("GVA_API_TOKEN_SECRET")
    if not secret:
        typer.echo("GVA_API_TOKEN_SECRET is not set", err=True)
        raise typer.Exit(1)
    token = jwt.encode({"role": role, "sub": "gva-cli"}, secret, algorithm="HS256")
    typer.echo(token)


@maintenance_app.command("retention")
def maintenance_retention() -> None:
    cfg = _cfg()
    store = _store(cfg)
    removed = enforce(cfg, store)
    typer.echo(f"removed {removed} events")


@sync_app.command("once")
def sync_once_cmd() -> None:
    cfg = _cfg()
    client = client_from_env()
    if client is None:
        typer.echo("Supabase is not configured", err=True)
        raise typer.Exit(1)
    store = _store(cfg)
    ingest_logs(Path(cfg.telemetry.log_dir), store)
    report = sync_once(cfg, store, client, events=_events(cfg, store, source="backend"))
    typer.echo(
        f"uploaded {report.uploaded} deferred {report.deferred} dropped {report.dropped} "
        f"applied {report.changes_applied} rolled_back {report.rolled_back}"
    )


@sync_app.command("run")
def sync_run() -> None:
    """Loop forever. systemd uses this unit."""
    cfg = _cfg()
    client = client_from_env()
    if client is None:
        typer.echo("Supabase is not configured", err=True)
        raise typer.Exit(1)
    from gva.backend.sync_supabase import run_forever

    store = _store(cfg)
    seed_defaults(store)
    run_forever(cfg, store, client)


@app.command("alerts")
def alerts_cmd() -> None:
    cfg = _cfg()
    store = _store(cfg)
    seed_defaults(store)
    opened = evaluate(store, cfg.device.id)
    typer.echo(f"opened {len(opened)}")


def _print_eval(report: EvalReport) -> None:
    typer.echo(
        f"cases {report.cases} passed {report.passed} accuracy {report.accuracy:.4f} model {report.model}"
    )
    for failure in report.failures:
        typer.echo(f"fail {failure['case_id']}: {failure['utterance']}")
    if not report.ok:
        raise typer.Exit(1)


def _coerce_cli(value: str):
    folded = value.strip()
    if folded.lower() in {"true", "false"}:
        return folded.lower() == "true"
    try:
        if "." in folded:
            return float(folded)
        return int(folded)
    except ValueError:
        return folded


def publish_eval(report: EvalReport) -> None:
    """Used by operators who want the run in Supabase. Not called from CI."""
    client = client_from_env()
    if client is None:
        return
    run_id = new_ulid()
    push_eval(
        client,
        {
            "id": run_id,
            "run_at": utc_now_ms(),
            "model": report.model,
            "prompt_version": report.prompt_version,
            "cases": report.cases,
            "passed": report.passed,
            "accuracy": report.accuracy,
        },
        [
            {
                "run_id": run_id,
                "case_id": item["case_id"],
                "utterance": item["utterance"],
                "expected": item["expected"],
                "actual": item["actual"],
            }
            for item in report.failures
        ],
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
