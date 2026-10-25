"""Bot bridge: webhook, PostgREST poll, and the realtime tool. No live network."""

import json
from pathlib import Path

from typer.testing import CliRunner

from gva.backend.sync_supabase import UPLOAD_TYPES
from gva.cli import app
from gva.config import Config, load_config, masked_secrets
from gva.paths import project_root
from gva.runtime.audio import FALLBACK_PHRASE, CollectingSink
from gva.runtime.bridge import (
    STILL_CHECKING_LINE,
    TIMEOUT_LINE,
    WAITING_LINE,
    BridgeError,
    BridgeTimeout,
    BotBridge,
    bridge_configured,
    bridge_status,
    probe_answers,
)
from gva.runtime.conversation import render_system
from gva.runtime.realtime import RealtimeProtocol
from gva.runtime.tools.registry import load_registry
from gva.telemetry.events import EventLog
from gva.telemetry.store import Store

QUESTION = "Which reports are due next week?"
ANSWER = "The board report is due on Monday."
WEBHOOK = "https://example.invalid/hook?token=bridge-query"
WEBHOOK_KEY = "bridge-test-key"
SERVICE_KEY = "service-test-key"


class _Resp:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class FakeHttp:
    def __init__(self):
        self.posts = []
        self.gets = []
        self.patches = []
        self.post_results = []
        self.ready_after = 1
        self.rows = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.posts.append({"url": url, "json": json, "headers": dict(headers or {}), "timeout": timeout})
        if self.post_results:
            result = self.post_results.pop(0)
            if isinstance(result, Exception):
                raise result
            return result
        return _Resp(200, {})

    def get(self, url, params=None, headers=None, timeout=None):
        self.gets.append({"url": url, "params": dict(params or {}), "headers": dict(headers or {})})
        if len(self.gets) < self.ready_after:
            return _Resp(200, [])
        return _Resp(200, list(self.rows))

    def patch(self, url, params=None, headers=None, json=None, timeout=None):
        self.patches.append(
            {"url": url, "params": dict(params or {}), "headers": dict(headers or {}), "json": json}
        )
        return _Resp(204, None)


class _Clock:
    def __init__(self):
        self.t = 0.0

    def mono(self):
        return self.t

    def sleep(self, seconds):
        self.t += float(seconds)


class _StubBridge:
    def __init__(self):
        self.questions = []

    def submit(self, question, background=True):
        self.questions.append(question)
        return "01ARZ3NDEKTSV4RRFFQ69G5FAV"


def _enable(monkeypatch):
    monkeypatch.setenv("GVA_ENV_FILE", "/no/such/gva.env")
    monkeypatch.setenv("GVA_GROKBOT_WEBHOOK_URL", WEBHOOK)
    monkeypatch.setenv("GVA_GROKBOT_WEBHOOK_KEY", WEBHOOK_KEY)
    monkeypatch.setenv("SUPABASE_URL", "https://example.invalid")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", SERVICE_KEY)


def _stack(tmp_path):
    cfg = Config()
    cfg.telemetry.log_dir = str(tmp_path / "log")
    cfg.telemetry.db_path = str(tmp_path / "metrics.db")
    cfg.bot_bridge.timeout_s = 9
    cfg.bot_bridge.poll_s = 3
    store = Store(tmp_path / "metrics.db")
    events = EventLog("milo-01", "dev", tmp_path / "log", store, validate=True)
    return cfg, store, events


def _bridge(tmp_path, http, speaker=None, sink=None):
    cfg, store, events = _stack(tmp_path)
    clock = _Clock()
    bridge = BotBridge(
        cfg,
        events,
        sink=sink if sink is not None else CollectingSink(),
        http=http,
        sleeper=clock.sleep,
        monotonic=clock.mono,
        speaker=speaker,
    )
    return cfg, store, events, bridge


def _logged(tmp_path, store) -> str:
    parts = [json.dumps(store.recent(limit=100))]
    for path in Path(tmp_path / "log").glob("*.jsonl"):
        parts.append(path.read_text(encoding="utf-8"))
    return "\n".join(parts)


def _assert_clean(blob: str):
    for secret in (WEBHOOK_KEY, SERVICE_KEY, "bridge-query", WEBHOOK, QUESTION, ANSWER):
        assert secret not in blob


def test_tool_is_hidden_until_the_bridge_is_configured(monkeypatch, tmp_path):
    monkeypatch.setenv("GVA_ENV_FILE", "/no/such/gva.env")
    monkeypatch.delenv("GVA_GROKBOT_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("GVA_GROKBOT_WEBHOOK_KEY", raising=False)
    cfg, store, events = _stack(tmp_path)
    assert bridge_configured(cfg) is False
    tools = load_registry(cfg, store)
    assert "ask_rbd_team" not in tools.specs
    assert "check_team_answer" not in tools.specs
    protocol = RealtimeProtocol(cfg, events, store, tools, CollectingSink())
    names = [item["name"] for item in protocol.session_update()["session"]["tools"]]
    assert "ask_rbd_team" not in names
    assert "check_team_answer" not in names
    assert "not connected" in render_system(cfg).lower()

    _enable(monkeypatch)
    cfg.bot_bridge.enabled = False
    assert bridge_configured(cfg) is False
    assert "ask_rbd_team" not in load_registry(cfg, store).specs
    assert "check_team_answer" not in load_registry(cfg, store).specs


def test_tool_is_advertised_when_configured(monkeypatch, tmp_path):
    _enable(monkeypatch)
    cfg, store, events = _stack(tmp_path)
    tools = load_registry(cfg, store)
    spec = tools.specs["ask_rbd_team"]
    assert "Asana" in spec.description
    assert "Adam's calendar" in spec.description
    assert "emails" in spec.description
    protocol = RealtimeProtocol(cfg, events, store, tools, CollectingSink())
    described = {item["name"]: item["description"] for item in protocol.session_update()["session"]["tools"]}
    assert "ask_rbd_team" in described
    assert "any answer yet" in described["check_team_answer"].lower()
    prompt = render_system(cfg)
    assert "ask_rbd_team" in prompt
    assert "call get_calendar_events" not in prompt
    assert "not connected" not in described["get_calendar_events"].lower()


def test_webhook_posts_polls_and_marks_the_row_read(monkeypatch, tmp_path):
    _enable(monkeypatch)
    http = FakeHttp()
    http.ready_after = 2
    http.rows = [{"request_id": "x", "answer": ANSWER, "answered_by": "bot", "read_at": None}]
    cfg, store, events, bridge = _bridge(tmp_path, http)
    cfg.bot_bridge.key_header = "X-Routine-Key"
    cfg.bot_bridge.key_prefix = "Token "
    spoken = []
    bridge._closed_speaker = spoken.append
    protocol = RealtimeProtocol(cfg, events, store, load_registry(cfg, store), CollectingSink())
    protocol.close = True
    bridge.attach(protocol)
    bridge.submit(QUESTION, background=False)

    assert len(http.posts) == 1
    posted = http.posts[0]
    assert posted["timeout"] == 10
    assert posted["headers"]["X-Routine-Key"] == "Token " + WEBHOOK_KEY
    body = posted["json"]
    assert body["device_id"] == "milo-01"
    assert body["question"] == QUESTION
    assert body["asked_at"].endswith("Z")
    assert body["reply_by"].endswith("Z")
    assert body["reply_by"] > body["asked_at"]
    assert len(body["request_id"]) == 26
    assert http.gets[0]["params"]["request_id"] == "eq." + body["request_id"]
    assert http.gets[0]["params"]["select"] == "request_id,answer,answered_by,read_at"
    assert http.gets[0]["headers"]["apikey"] == SERVICE_KEY
    assert http.gets[0]["headers"]["Authorization"] == "Bearer " + SERVICE_KEY
    assert len(http.gets) == 2
    assert http.patches[0]["json"]["read_at"].endswith("Z")
    assert http.patches[0]["params"]["request_id"] == "eq." + body["request_id"]
    assert http.patches[0]["headers"]["Prefer"] == "return=minimal"
    assert spoken == [ANSWER]
    assert bridge.sink.chunks
    sent = store.recent("bridge.sent", 1)[0]
    assert sent["request_id"] == body["request_id"]
    answered = store.recent("bridge.answered", 1)[0]
    assert answered["latency_ms"] == 3000
    assert "bridge.sent" not in UPLOAD_TYPES
    assert "bridge.answered" not in UPLOAD_TYPES
    _assert_clean(_logged(tmp_path, store))


def test_webhook_retries_once_and_never_logs_the_query(monkeypatch, tmp_path):
    _enable(monkeypatch)
    http = FakeHttp()
    http.post_results = [
        RuntimeError("nope " + WEBHOOK),
        _Resp(200, {"echo": WEBHOOK_KEY}),
    ]
    http.rows = [{"answer": ANSWER}]
    _cfg, store, _events, bridge = _bridge(tmp_path, http, speaker=lambda _text: None)
    bridge.submit(QUESTION, background=False)
    assert len(http.posts) == 2
    assert store.recent("bridge.sent", 1)
    blob = _logged(tmp_path, store)
    assert "bridge-query" not in blob
    assert WEBHOOK_KEY not in blob
    assert "RuntimeError" not in blob

    http2 = FakeHttp()
    http2.post_results = [RuntimeError("nope " + WEBHOOK), RuntimeError("nope " + WEBHOOK)]
    _cfg, store2, _events, bridge2 = _bridge(tmp_path / "again", http2, speaker=lambda _text: None)
    try:
        bridge2.submit(QUESTION, background=False)
        raised = False
    except BridgeError:
        raised = True
    assert raised is True
    assert len(http2.posts) == 2
    assert http2.gets == []
    assert store2.recent("bridge.error", 1)[0]["message"] == "RuntimeError"
    _assert_clean(_logged(tmp_path / "again", store2))


def test_client_error_is_not_retried(monkeypatch, tmp_path):
    _enable(monkeypatch)
    http = FakeHttp()
    http.post_results = [_Resp(400, {"detail": WEBHOOK_KEY})]
    _cfg, store, _events, bridge = _bridge(tmp_path, http)
    try:
        bridge.submit(QUESTION, background=False)
        raised = False
    except BridgeError:
        raised = True
    assert raised is True
    assert len(http.posts) == 1
    assert store.recent("bridge.error", 1)[0]["message"] == "HTTP 400"
    assert store.recent("bridge.sent", 1) == []
    _assert_clean(_logged(tmp_path, store))


def test_timeout_says_the_team_did_not_reply(monkeypatch, tmp_path):
    _enable(monkeypatch)
    http = FakeHttp()
    http.rows = []
    spoken = []
    sink = CollectingSink()
    _cfg, store, _events, bridge = _bridge(tmp_path, http, speaker=spoken.append, sink=sink)
    bridge.submit(QUESTION, background=False)
    assert spoken == [TIMEOUT_LINE]
    assert sink.chunks == []
    assert store.recent("bridge.timeout", 1)[0]["level"] == "warning"
    assert store.recent("bridge.answered", 1) == []
    assert http.patches == []
    _assert_clean(_logged(tmp_path, store))


def test_nudge_is_spoken_once_around_45_seconds(monkeypatch, tmp_path):
    _enable(monkeypatch)
    http = FakeHttp()
    spoken = []
    cfg, store, _events, bridge = _bridge(tmp_path, http, speaker=spoken.append)
    cfg.bot_bridge.timeout_s = 60
    cfg.bot_bridge.poll_s = 15
    cfg.bot_bridge.nudge_s = 45
    bridge.submit(QUESTION, background=False)
    assert spoken == [STILL_CHECKING_LINE, TIMEOUT_LINE]
    assert spoken.count(STILL_CHECKING_LINE) == 1
    _assert_clean(_logged(tmp_path, store))


def test_empty_prefix_sends_the_key_alone(monkeypatch, tmp_path):
    _enable(monkeypatch)
    http = FakeHttp()
    http.rows = [{"answer": ANSWER, "answered_by": "Brenda"}]
    cfg, store, _events, bridge = _bridge(tmp_path, http, speaker=lambda _text: None)
    cfg.bot_bridge.key_header = "X-Api-Key"
    cfg.bot_bridge.key_prefix = ""
    bridge.submit(QUESTION, background=False)
    sent = http.posts[0]["headers"]["X-Api-Key"]
    assert sent == WEBHOOK_KEY
    assert not sent.startswith("Bearer")
    _assert_clean(_logged(tmp_path, store))


def test_late_answer_is_kept_for_the_next_wake_or_a_later_ask(monkeypatch, tmp_path):
    _enable(monkeypatch)
    http = FakeHttp()
    spoken = []
    cfg, store, events, bridge = _bridge(tmp_path, http, speaker=spoken.append)
    bridge.submit(QUESTION, background=False)
    assert spoken == [TIMEOUT_LINE]
    assert http.patches == []
    pending = json.loads(bridge._pending_path().read_text(encoding="utf-8"))
    assert pending["request_ids"]
    assert QUESTION not in json.dumps(pending)
    assert ANSWER not in json.dumps(pending)

    http.rows = [{"answer": ANSWER, "answered_by": "Brenda", "read_at": None}]
    protocol = RealtimeProtocol(cfg, events, store, load_registry(cfg, store), CollectingSink())
    bridge.attach(protocol)
    found = bridge.recall(deliver=True)
    assert found == [ANSWER]
    assert spoken == [TIMEOUT_LINE]
    queued = protocol._bridge_items[0]["messages"][0]["item"]["content"][0]["text"]
    assert queued == ANSWER
    assert http.patches
    assert "Brenda" not in _logged(tmp_path, store)
    assert bridge.recall(deliver=True) == []

    tools = load_registry(cfg, store)
    tools.ctx.bridge = bridge
    waiting = tools.run("later", "check_team_answer", "{}")
    assert waiting.result["status"] == "waiting"

    http.rows = [{"answer": ANSWER, "answered_by": "Brenda"}]
    bridge._remember("01ARZ3NDEKTSV4RRFFQ69G5FAV")
    ready = tools.run("ask", "check_team_answer", "{}")
    assert ready.result["status"] == "ready"
    assert ready.result["answer"] == ANSWER
    assert ready.args_redacted == {}
    assert ANSWER not in _logged(tmp_path, store)
    _assert_clean(_logged(tmp_path, store).replace(ANSWER, ""))


def test_answer_after_the_session_closed_chimes_and_speaks(monkeypatch, tmp_path):
    _enable(monkeypatch)
    http = FakeHttp()
    http.rows = [{"answer": ANSWER, "answered_by": "bot"}]
    spoken = []
    sink = CollectingSink()
    cfg, store, events, bridge = _bridge(tmp_path, http, speaker=spoken.append, sink=sink)
    protocol = RealtimeProtocol(cfg, events, store, load_registry(cfg, store), sink)
    protocol.close = True
    bridge.attach(protocol)
    bridge.submit(QUESTION, background=False)
    assert spoken == [ANSWER]
    assert sink.chunks
    assert protocol._bridge_items == []
    _assert_clean(_logged(tmp_path, store))


def test_open_session_queues_a_natural_readout_and_speaks_if_it_closes(monkeypatch, tmp_path):
    _enable(monkeypatch)
    http = FakeHttp()
    http.rows = [{"answer": ANSWER}]
    spoken = []
    sink = CollectingSink()
    cfg, store, events, bridge = _bridge(tmp_path, http, speaker=spoken.append, sink=sink)
    protocol = RealtimeProtocol(cfg, events, store, load_registry(cfg, store), sink)
    bridge.attach(protocol)
    bridge.submit(QUESTION, background=False)
    assert spoken == []
    assert sink.chunks == []
    queued = protocol._bridge_items[0]
    assert queued["messages"][0]["item"]["content"][0]["text"] == ANSWER
    instructions = queued["messages"][1]["response"]["instructions"]
    assert "naturally and briefly" in instructions
    protocol.close = True
    bridge.detach(protocol)
    assert spoken == [ANSWER]
    assert sink.chunks
    assert protocol._bridge_items == []


def test_waiting_line_does_not_play_the_calendar_fallback(monkeypatch, tmp_path):
    _enable(monkeypatch)
    cfg, store, events = _stack(tmp_path)
    tools = load_registry(cfg, store)
    stub = _StubBridge()
    tools.ctx.bridge = stub
    spoken = []
    protocol = RealtimeProtocol(cfg, events, store, tools, CollectingSink())
    protocol.fallback_player = lambda sink, config, phrase: spoken.append(phrase) or 900
    protocol.on_event({"type": "response.created", "response": {"id": "r1"}})
    messages = protocol.on_event(
        {
            "type": "response.function_call_arguments.done",
            "name": "ask_rbd_team",
            "call_id": "call-bridge",
            "arguments": json.dumps({"question": QUESTION}),
        }
    )
    body = json.loads(messages[0]["item"]["output"])
    assert body == {"status": "sent"}
    assert stub.questions == [QUESTION]
    assert spoken == []
    follow = protocol.on_event({"type": "response.done", "response": {"id": "r1"}})
    assert follow == [
        {
            "type": "response.create",
            "response": {"instructions": "Say only: " + WAITING_LINE},
        }
    ]
    assert spoken == []
    completed = store.recent("tool.completed", 1)[0]
    assert completed["args_redacted"] == {}
    assert QUESTION not in json.dumps(completed)

    protocol.on_event({"type": "response.created", "response": {"id": "r2"}})
    protocol.on_event({"type": "response.done", "response": {"id": "r2"}})
    assert spoken == [WAITING_LINE]
    assert FALLBACK_PHRASE not in spoken
    _assert_clean(_logged(tmp_path, store))


def test_doctor_reports_bridge_without_secrets(monkeypatch, tmp_path):
    monkeypatch.setenv("GVA_ENV_FILE", str(tmp_path / "missing.env"))
    cfg = Config()
    assert bridge_status(cfg) == ("bridge: not configured", None)

    _enable(monkeypatch)

    class Probe:
        def __init__(self, code):
            self.code = code
            self.calls = []

        def get(self, url, params=None, headers=None, timeout=None):
            self.calls.append({"url": url, "params": params, "headers": headers})
            return _Resp(self.code, [{"request_id": "should-not-be-logged"}])

        def close(self):
            return None

    ok_probe = Probe(200)
    reachable, detail = probe_answers(cfg, http=ok_probe)
    assert reachable is True and detail == "ok"
    assert ok_probe.calls[0]["params"] == {"select": "request_id", "limit": "1"}
    assert "should-not-be-logged" not in detail

    missing = Probe(404)
    reachable, detail = probe_answers(cfg, http=missing)
    assert reachable is False and detail == "HTTP 404"
    line, warning = bridge_status(cfg, probe=lambda _cfg: (False, "HTTP 404"))
    assert line == "bridge: configured, milo_bot_answers not reachable"
    assert warning == "milo_bot_answers is not reachable (HTTP 404)"
    assert WEBHOOK_KEY not in line + (warning or "")
    assert "example.invalid" not in line + (warning or "")

    from gva.runtime.bootstrap import doctor_checks

    problems, _warnings = doctor_checks(cfg)
    assert not any("milo_bot_answers" in item for item in problems)

    presence = masked_secrets()
    assert presence["GVA_GROKBOT_WEBHOOK_URL"] == "set"
    assert presence["GVA_GROKBOT_WEBHOOK_KEY"] == "set"
    assert WEBHOOK not in presence["GVA_GROKBOT_WEBHOOK_URL"]
    assert WEBHOOK_KEY not in "".join(presence.values())


def test_example_files_document_the_bridge(monkeypatch):
    monkeypatch.setenv("GVA_ENV_FILE", "/no/such/gva.env")
    cfg = load_config(project_root() / "config" / "config.example.yaml")
    assert cfg.bot_bridge.enabled is True
    assert cfg.bot_bridge.timeout_s == 300
    assert cfg.bot_bridge.poll_s == 3
    assert cfg.bot_bridge.nudge_s == 45
    assert cfg.bot_bridge.key_header == "Authorization"
    assert cfg.bot_bridge.key_prefix == "Bearer "
    example = (project_root() / "gva.env.example").read_text(encoding="utf-8")
    assert "GVA_GROKBOT_WEBHOOK_URL=\n" in example
    assert "GVA_GROKBOT_WEBHOOK_KEY=\n" in example
    readme = (project_root() / "README.md").read_text(encoding="utf-8")
    assert "milo_bot_answers" in readme
    assert "milo-answer" in readme
    assert "Brenda" in readme
    assert 'key_prefix: ""' in readme
    assert "no migration" in readme.lower()
    assert "—" not in readme
    assert "–" not in readme


def test_bridge_cli(monkeypatch, tmp_path):
    monkeypatch.setenv("GVA_ENV_FILE", str(tmp_path / "missing.env"))
    monkeypatch.setenv("GVA_LOG_DIR", str(tmp_path / "log"))
    monkeypatch.setenv("GVA_DB_PATH", str(tmp_path / "metrics.db"))
    monkeypatch.delenv("GVA_GROKBOT_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("GVA_GROKBOT_WEBHOOK_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)
    result = CliRunner().invoke(app, ["bridge", "test", QUESTION])
    text = (result.output or "") + (getattr(result, "stderr", None) or "")
    assert result.exit_code == 1
    assert "not configured" in text

    _enable(monkeypatch)
    monkeypatch.setattr(
        "gva.runtime.bridge.BotBridge.send_and_wait",
        lambda self, question: ANSWER,
    )
    result = CliRunner().invoke(app, ["bridge", "test", QUESTION])
    assert result.exit_code == 0
    assert ANSWER in (result.output or "")
    assert WEBHOOK_KEY not in (result.output or "")

    def _timeout(self, question):
        raise BridgeTimeout(TIMEOUT_LINE)

    monkeypatch.setattr("gva.runtime.bridge.BotBridge.send_and_wait", _timeout)
    result = CliRunner().invoke(app, ["bridge", "test", QUESTION])
    text = (result.output or "") + (getattr(result, "stderr", None) or "")
    assert result.exit_code == 1
    assert TIMEOUT_LINE in text

    monkeypatch.setattr("gva.runtime.bridge.probe_answers", lambda cfg: (True, "ok"))
    doctor = CliRunner().invoke(app, ["doctor"])
    doctor_text = (doctor.output or "") + (getattr(doctor, "stderr", None) or "")
    assert "bridge:" in doctor_text
    assert WEBHOOK_KEY not in doctor_text
    assert "bridge-query" not in doctor_text
