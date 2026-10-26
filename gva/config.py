"""Configuration loaded from YAML. Secrets are never read from this file."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from gva.paths import project_root


class DeviceConfig(BaseModel):
    id: str = "milo-01"
    name: str = "Milo"
    timezone: str = "Europe/London"
    env: Literal["dev", "staging", "prod"] = "prod"
    default_location: str = "Nottingham, UK"


class AudioConfig(BaseModel):
    input_device: str = "AIRHUG"
    # Empty means auto: a USB playback device, then the 3.5 mm jack. Never Bluetooth.
    output_device: str = ""
    # AIRHUG's native rate. Wake and the local pipeline resample down from this.
    capture_rate: int = 48000
    sample_rate: int = 16000
    chunk_ms: int = 80
    aec_enabled: bool = True
    aec_delay_ms: int = 80
    # How much device-rate audio to queue before the callback leaves silence.
    # 200 ms sits in the 150-250 ms range that keeps a USB speaker from stuttering.
    playback_prebuffer_ms: int = 200
    # PortAudio hint. "high" suits the Sanyun USB card. "low" is the other allowed value.
    playback_latency: Literal["high", "low"] = "high"


class WakeConfig(BaseModel):
    engine: str = "openwakeword"
    # hey_jarvis is the stock model. "milo" loads assets/wakewords/milo.onnx,
    # trained for "hey milo" with bare "milo" as a secondary phrase. If that
    # file is missing, scoring falls back to hey_jarvis.
    model: str = "hey_jarvis"
    phrase: str = "hey jarvis"
    model_dir: str = "/usr/share/gva/wake"
    # 0.6 and two frames in a row is the wake setup that stopped false triggers.
    threshold: float = 0.6
    patience_frames: int = 2
    refractory_seconds: float = 2.0
    cooldown_seconds: float = 3.0
    vad_gate: bool = True
    pre_roll_ms: int = 300
    stop_model: str = ""


class VadConfig(BaseModel):
    engine: str = "silero"
    end_silence_ms: int = 800
    max_utterance_seconds: int = 12
    energy_threshold: float = 500


class XaiSttConfig(BaseModel):
    model: str = "grok-voice-transcribe-2.0"
    keyterms: list[str] = Field(default_factory=list)
    allow_cloud_fallback: bool = False


class SttConfig(BaseModel):
    engine: Literal["faster_whisper", "vosk", "xai"] = "faster_whisper"
    model: str = "base.en"
    compute_type: str = "int8"
    beam_size: int = 1
    cpu_threads: int = 2
    language: str = "en"
    min_confidence: float = 0.45
    hallucinations: list[str] = Field(default_factory=list)
    xai: XaiSttConfig = Field(default_factory=XaiSttConfig)


class TimeoutConfig(BaseModel):
    connect_s: float = 3
    read_s: float = 10
    stream_idle_s: float = 5


class RetryConfig(BaseModel):
    max_attempts: int = 3
    base_s: float = 0.5
    cap_s: float = 4


class BreakerConfig(BaseModel):
    failures: int = 5
    window_s: float = 60
    open_s: float = 30


class LlmConfig(BaseModel):
    base_url: str = "https://api.x.ai/v1"
    model: str = "grok-4.7"
    fallback_model: str = "grok-4.3"
    reasoning_effort: str | None = None
    temperature: float = 0.4
    max_completion_tokens: int = 300
    parallel_tool_calls: bool = False
    max_tool_rounds: int = 3
    history_messages: int = 6
    history_token_budget: int = 1500
    timeouts: TimeoutConfig = Field(default_factory=TimeoutConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    circuit_breaker: BreakerConfig = Field(default_factory=BreakerConfig)
    system_prompt_version: str = "v3"
    user_agent: str = "gva/0.1.0"


class XaiTtsConfig(BaseModel):
    voice_id: str = "eve"
    enabled: bool = False


class TtsConfig(BaseModel):
    engine: Literal["piper", "xai"] = "piper"
    voice: str = "en_GB-alan-low"
    model_dir: str = "/usr/share/gva/voices"
    speaking_rate: float = 1.0
    pronunciation: dict[str, str] = Field(default_factory=dict)
    xai: XaiTtsConfig = Field(default_factory=XaiTtsConfig)


class RealtimeConfig(BaseModel):
    """xAI speech-to-speech. Prices are config, not constants buried in the cost maths."""

    model: str = "grok-voice-latest"
    voice: str = "eve"
    url: str = "wss://api.x.ai/v1/realtime"
    sample_rate: int = 24000
    silence_duration_ms: int = 800
    silence_timeout_s: float = 15
    # Close the socket if the server sends nothing at all. Separate from silence_timeout_s,
    # which follows conversation events rather than microphone frames.
    watchdog_s: float = 30
    audio_usd_per_minute: float = 0.08
    text_usd_per_message: float = 0.004


class ConversationConfig(BaseModel):
    # realtime is the speech-to-speech session that already works on this Pi.
    # local is faster-whisper, then Grok chat, then Piper, for offline or cheaper runs.
    engine: Literal["realtime", "local"] = "realtime"
    realtime: RealtimeConfig = Field(default_factory=RealtimeConfig)
    follow_up_seconds: float = 6
    session_idle_minutes: float = 10
    turn_budget_seconds: float = 8
    barge_in: str = "wake_and_stop"
    energy_barge_in: bool = True
    barge_in_min_ms: int = 280
    stop_phrases: list[str] = Field(
        default_factory=lambda: [
            "stop",
            "cancel",
            "never mind",
            "nevermind",
            "be quiet",
            "shut up",
            "goodbye",
            "bye",
            "end",
            "quit",
        ]
    )


class LedConfig(BaseModel):
    enabled: bool = False
    pin: int = 18


class HomeDevice(BaseModel):
    id: str
    name: str
    kinds: list[str] = Field(default_factory=lambda: ["switch"])


class HomeConfig(BaseModel):
    devices: list[HomeDevice] = Field(default_factory=list)


class PrivacyConfig(BaseModel):
    household_names: list[str] = Field(default_factory=list)
    redact_short_numbers: bool = False
    sync_redact_short_numbers: bool = True
    hash_epoch: int = 1
    sensitive_categories: dict[str, list[str]] = Field(default_factory=dict)


class TelemetryConfig(BaseModel):
    log_dir: str = "/var/log/gva"
    db_path: str = "/var/lib/gva/metrics.db"
    schema_version: str = "1.0.0"
    heartbeat_seconds: int = 60
    validate_events: bool = True
    replay_cache_hours: int = 0
    debug_audio: bool = False
    debug_audio_dir: str = "/var/lib/gva/debug-audio"


class Prices(BaseModel):
    input: float | None = None
    cached_input: float | None = None
    output: float | None = None


class BudgetConfig(BaseModel):
    monthly_usd: float | None = None
    daily_soft_cap_usd: float | None = None
    fx_usd_gbp: float | None = None
    prices_per_mtok_usd: Prices = Field(default_factory=Prices)


class SyncConfig(BaseModel):
    enabled: bool = True
    interval_seconds: int = 15
    batch_size: int = 500
    rollback_seconds: int = 300


class BackendConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8080
    dashboard_origin: str = "https://voice-ops.example.com"
    docs_enabled: bool = False


class RetentionConfig(BaseModel):
    events_days: int = 30
    rollup_months: int = 13
    debug_audio_hours: int = 24
    heartbeat_days: int = 90


class BotBridgeConfig(BaseModel):
    """Work questions go to the Grok Bot webhook. Quick chat stays on realtime.

    `enabled` turns the bridge on by itself once the webhook URL, the webhook
    key, and Supabase are all set. Set it false to force the bridge off.
    """

    enabled: bool = True
    # Five minutes. A Grok Bot routine can start a few minutes after Milo asks.
    timeout_s: int = Field(default=300, ge=5, le=900)
    poll_s: float = Field(default=3, ge=0.2, le=60)
    # Say "Still checking with the team" once, this long after the question. 0 disables it.
    nudge_s: int = Field(default=45, ge=0, le=300)
    key_header: str = "Authorization"
    # "Bearer " includes the trailing space. An empty string sends the key with no prefix.
    key_prefix: str = "Bearer "


class Config(BaseModel):
    model_config = ConfigDict(extra="ignore")

    device: DeviceConfig = Field(default_factory=DeviceConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    wake: WakeConfig = Field(default_factory=WakeConfig)
    vad: VadConfig = Field(default_factory=VadConfig)
    stt: SttConfig = Field(default_factory=SttConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    tts: TtsConfig = Field(default_factory=TtsConfig)
    conversation: ConversationConfig = Field(default_factory=ConversationConfig)
    led: LedConfig = Field(default_factory=LedConfig)
    home: HomeConfig = Field(default_factory=HomeConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    sync: SyncConfig = Field(default_factory=SyncConfig)
    backend: BackendConfig = Field(default_factory=BackendConfig)
    retention: RetentionConfig = Field(default_factory=RetentionConfig)
    bot_bridge: BotBridgeConfig = Field(default_factory=BotBridgeConfig)
    source_path: Path | None = Field(default=None, exclude=True)
    source_data: dict[str, Any] = Field(default_factory=dict, exclude=True)


ENV_FILE_STATUS = "unread"


def _read_env(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_service_env(path: Path | None = None) -> str:
    """Load /etc/gva/gva.env into the process when a variable is not already set.

    Returns "loaded", "missing", or "unreadable". Values are never logged.
    Blank lines and comments are ignored. A value already in the environment wins,
    so a service EnvironmentFile is not overwritten by a second read.
    """
    global ENV_FILE_STATUS
    if path is None:
        override = os.environ.get("GVA_ENV_FILE")
        path = Path(override) if override else Path("/etc/gva/gva.env")
    try:
        exists = path.is_file()
    except OSError:
        ENV_FILE_STATUS = "unreadable"
        return ENV_FILE_STATUS
    if not exists:
        ENV_FILE_STATUS = "missing"
        return ENV_FILE_STATUS
    try:
        text = _read_env(path)
    except OSError:
        ENV_FILE_STATUS = "unreadable"
        return ENV_FILE_STATUS
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or not key.replace("_", "").isalnum() or key[0].isdigit():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key not in os.environ:
            os.environ[key] = value
    ENV_FILE_STATUS = "loaded"
    return ENV_FILE_STATUS


def env_value(name: str) -> str | None:
    """Return an environment variable, treating blank and placeholder values as unset."""
    raw = os.environ.get(name)
    if raw is None:
        return None
    value = raw.strip()
    if not value or value.startswith("<") or value.endswith(">"):
        return None
    return value


def _read_config_text(path: Path) -> tuple[str | None, str | None]:
    """Return (text, error note). A missing file is (None, None), not an error."""
    try:
        if not path.is_file():
            return None, None
        return path.read_text(encoding="utf-8"), None
    except OSError as exc:
        return None, (
            f"Cannot read {path} ({exc.__class__.__name__}). "
            "Using the bundled example configuration. "
            "/etc/gva should be mode 0755 and config.yaml mode 0644. "
            "Secrets stay in /etc/gva/gva.env mode 0600."
        )


def load_config(path: Path | None = None) -> Config:
    load_service_env()
    notes: list[str] = []
    if path is None:
        override = os.environ.get("GVA_CONFIG")
        etc = Path("/etc/gva/config.yaml")
        if override:
            path = Path(override)
        else:
            text, note = _read_config_text(etc)
            if note:
                notes.append(note)
                path = project_root() / "config" / "config.example.yaml"
            elif text is not None:
                path = etc
            else:
                path = project_root() / "config" / "config.example.yaml"
    path = Path(path)
    data: dict[str, Any] = {}
    text, note = _read_config_text(path)
    if note:
        notes.append(note)
        fallback = project_root() / "config" / "config.example.yaml"
        if fallback != path:
            text, fallback_note = _read_config_text(fallback)
            if fallback_note:
                notes.append(fallback_note)
                text = None
            else:
                path = fallback
    if text:
        loaded = yaml.safe_load(text)
        if isinstance(loaded, dict):
            data = loaded
    for note in notes:
        print(note, file=sys.stderr)
    if env_value("GVA_ENV"):
        data.setdefault("device", {})["env"] = os.environ["GVA_ENV"].strip()
    if os.environ.get("GVA_LOG_DIR"):
        data.setdefault("telemetry", {})["log_dir"] = os.environ["GVA_LOG_DIR"]
    if os.environ.get("GVA_DB_PATH"):
        data.setdefault("telemetry", {})["db_path"] = os.environ["GVA_DB_PATH"]
    if os.environ.get("GVA_DEVICE_ID"):
        data.setdefault("device", {})["id"] = os.environ["GVA_DEVICE_ID"]
    cfg = Config.model_validate(data)
    cfg.source_path = path
    cfg.source_data = data
    return cfg


def config_fingerprint(data: dict[str, Any]) -> str:
    raw = json.dumps(data, sort_keys=True, default=str, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def get_dotted(data: dict[str, Any], key: str) -> Any:
    current: Any = data
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def set_dotted(data: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    current = data
    for part in parts[:-1]:
        nxt = current.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            current[part] = nxt
        current = nxt
    current[parts[-1]] = value


_SECRET_BITS = ("secret", "token", "pepper", "password", "webhook", "api_key", "service_key")


class ConfigChangeError(ValueError):
    pass


def validate_change(key: str, value: Any) -> Any:
    """Validate a non-secret dashboard or CLI change. Returns the coerced value."""
    folded = key.lower()
    if any(bit in folded for bit in _SECRET_BITS):
        raise ConfigChangeError(f"{key} is a secret and cannot be changed from the dashboard or CLI")
    rule = CHANGE_RULES.get(key)
    if rule is None:
        raise ConfigChangeError(f"{key} is not a setting the device will apply")
    return rule(value)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        folded = value.strip().lower()
        if folded in {"true", "yes", "1"}:
            return True
        if folded in {"false", "no", "0"}:
            return False
    raise ConfigChangeError("expected true or false")


def _choice(options: set[str]):
    def check(value: Any) -> str:
        text = str(value)
        if text not in options:
            raise ConfigChangeError(f"expected one of {sorted(options)}")
        return text

    return check


def _number(kind: type, low: float, high: float):
    def check(value: Any):
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigChangeError("expected a number") from exc
        if not low <= number <= high:
            raise ConfigChangeError(f"expected a value between {low} and {high}")
        return kind(number) if kind is int else number

    return check


def _text(value: Any) -> str:
    text = str(value).strip()
    if not text or len(text) > 200:
        raise ConfigChangeError("expected a short string")
    return text


def _optional_text(value: Any) -> str | None:
    if value is None or value == "" or value == "null":
        return None
    return _text(value)


CHANGE_RULES = {
    "llm.model": _text,
    "llm.reasoning_effort": _optional_text,
    "llm.max_completion_tokens": _number(int, 16, 2000),
    "llm.temperature": _number(float, 0, 1),
    "llm.max_tool_rounds": _number(int, 1, 5),
    "llm.system_prompt_version": _choice({"v1", "v2", "v3"}),
    "wake.threshold": _number(float, 0.05, 0.95),
    "wake.patience_frames": _number(int, 1, 8),
    "wake.model": _text,
    "wake.phrase": _text,
    "stt.engine": _choice({"faster_whisper", "vosk", "xai"}),
    "stt.model": _text,
    "stt.compute_type": _choice({"int8", "int8_float16", "float16", "float32"}),
    "stt.beam_size": _number(int, 1, 5),
    "stt.xai.allow_cloud_fallback": _as_bool,
    "tts.engine": _choice({"piper", "xai"}),
    "tts.voice": _text,
    "tts.speaking_rate": _number(float, 0.5, 2.0),
    "conversation.follow_up_seconds": _number(float, 0, 30),
    "conversation.engine": _choice({"realtime", "local"}),
    "conversation.realtime.voice": _text,
    "conversation.realtime.model": _text,
    "wake.cooldown_seconds": _number(float, 0, 30),
    "privacy.redact_short_numbers": _as_bool,
    "privacy.sync_redact_short_numbers": _as_bool,
    "action.health_check": _as_bool,
}


def save_config(cfg: Config) -> None:
    if cfg.source_path is None:
        raise ConfigChangeError("this configuration was not loaded from a file")
    cfg.source_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cfg.source_path.with_suffix(cfg.source_path.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(cfg.source_data, sort_keys=False), encoding="utf-8")
    tmp.replace(cfg.source_path)


def masked_secrets() -> dict[str, str]:
    """Presence only. Values are never returned."""
    names = [
        "XAI_API_KEY",
        "GVA_HASH_PEPPER",
        "SUPABASE_URL",
        "SUPABASE_SERVICE_KEY",
        "GVA_API_TOKEN_SECRET",
        "GVA_ALERT_SLACK_WEBHOOK",
        "GVA_HEARTBEAT_URL",
        "GVA_GROKBOT_WEBHOOK_URL",
        "GVA_GROKBOT_WEBHOOK_KEY",
    ]
    out = {}
    for name in names:
        value = env_value(name)
        if not value:
            out[name] = "unset"
        elif name == "XAI_API_KEY" and len(value) >= 4:
            out[name] = f"set (last4 {value[-4:]})"
        elif name == "SUPABASE_URL":
            out[name] = value
        else:
            out[name] = "set"
    return out
