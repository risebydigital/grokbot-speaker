"""Bot bridge: webhook, PostgREST poll, and the realtime tool. No live network."""

import json
import time
from pathlib import Path

from typer.testing import CliRunner

from gva.backend.sync_supabase import UPLOAD_TYPES
from gva.cli import app
from gva.config import Config, load_config, masked_secrets
from gva.paths import project_root
from gva.runtime.audio import FALLBACK_PHRASE, CollectingSink
from gva.runtime.bridge import (
    DEFER_LINE,
    STILL_CHECKING_LINE,
    TIMEOUT_LINE,
    WAITING_LINE,
    BridgeError,
    BotBridge,
    bridge_configured,
    bridge_status,
    in_nudge_quiet_hours,
    late_line,
    nudge_line,
    probe_answers,
    supabase_doctor_line,
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
    assert "defer_until_answer" not in tools.specs
    protocol = RealtimeProtocol(cfg, events, store, tools, CollectingSink())
    names = [item["name"] for item in protocol.session_update()["session"]["tools"]]
    assert "ask_rbd_team" not in names
    assert "check_team_answer" not in names
    assert "defer_until_answer" not in names
    assert "not connected" in render_system(cfg).lower()

    _enable(monkeypatch)
    cfg.bot_bridge.enabled = False
    assert bridge_configured(cfg) is False
    assert "ask_rbd_team" not in load_registry(cfg, store).specs
    assert "check_team_answer" not in load_registry(cfg, store).specs
    assert "defer_until_answer" not in load_registry(cfg, store).specs


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
    assert "tell me later" in described["defer_until_answer"].lower()
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
    assert http.patches[0]["headers"]["Authorization"] == "Bearer " + SERVICE_KEY
    assert len(http.gets) == 2
    assert http.patches[0]["json"]["read_at"].endswith("Z")
    assert http.patches[0]["params"]["request_id"] == "eq." + body["request_id"]
    assert http.patches[0]["headers"]["Prefer"] == "return=minimal"
    assert spoken == ["bot's come back. " + ANSWER]
    assert store.recent("bridge.spoken", 1)[0]["request_id"] == body["request_id"]
    assert "bridge.spoken" not in UPLOAD_TYPES
    assert "bridge.delivery_failed" not in UPLOAD_TYPES
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
    assert "couldn't get an answer" in TIMEOUT_LINE
    assert sink.chunks == []
    assert store.recent("bridge.timeout", 1)[0]["level"] == "warning"
    spoken_event = store.recent("bridge.timeout_spoken", 1)[0]
    assert spoken_event["request_id"]
    assert "bridge.timeout_spoken" not in UPLOAD_TYPES
    assert store.recent("bridge.answered", 1) == []
    assert http.patches == []
    assert bridge._kept
    pending = json.loads(bridge._pending_path().read_text(encoding="utf-8"))
    assert pending["timed_out"]
    assert QUESTION not in json.dumps(pending)
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
    assert pending["timed_out"]
    assert QUESTION not in json.dumps(pending)
    assert ANSWER not in json.dumps(pending)
    assert store.recent("bridge.timeout_spoken", 1)

    http.rows = [{"answer": ANSWER, "answered_by": "Brenda", "read_at": None}]
    protocol = RealtimeProtocol(cfg, events, store, load_registry(cfg, store), CollectingSink())
    bridge.attach(protocol)
    found = bridge.recall(deliver=True)
    assert found == [ANSWER]
    assert spoken == [TIMEOUT_LINE]
    queued = protocol._bridge_items[0]["text"]
    assert queued == late_line(ANSWER, "Brenda")
    assert queued.startswith("By the way, Brenda came back")
    assert http.patches == []
    assert "Brenda" not in _logged(tmp_path, store)
    protocol.close = True
    bridge.detach(protocol)
    assert spoken[-1] == queued
    assert http.patches
    late = store.recent("bridge.late_delivered", 1)[0]
    assert late["request_id"]
    assert ANSWER not in json.dumps(late)
    assert "Brenda" not in json.dumps(late)
    assert "bridge.late_delivered" not in UPLOAD_TYPES
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


def test_bridge_test_reports_http_status_and_whether_an_answer_arrived(monkeypatch, tmp_path):
    _enable(monkeypatch)
    http = FakeHttp()
    http.rows = [{"answer": "ready", "answered_by": "Brenda"}]
    _cfg, store, _events, bridge = _bridge(tmp_path, http, speaker=lambda _text: None)
    report = bridge.run_test()
    assert report.http_status == 200
    assert report.arrived is True
    assert report.answered_by == "Brenda"
    assert report.ok() is True
    assert report.lines() == ["webhook: HTTP 200", "answer: yes", "answered_by: Brenda"]
    posted = http.posts[0]
    assert posted["headers"]["Authorization"] == "Bearer " + WEBHOOK_KEY
    assert set(posted["json"]) == {"request_id", "device_id", "question", "asked_at", "reply_by"}
    assert posted["json"]["question"].startswith("Milo bridge test.")
    assert http.patches
    blob = _logged(tmp_path, store)
    assert WEBHOOK_KEY not in blob
    assert "ready" not in blob
    assert "Brenda" not in blob
    text = "\n".join(report.lines())
    assert WEBHOOK_KEY not in text
    assert "ready" not in text

    http2 = FakeHttp()
    http2.post_results = [_Resp(401, {"detail": WEBHOOK_KEY})]
    _cfg, store2, _events, bridge2 = _bridge(tmp_path / "denied", http2)
    missed = bridge2.run_test()
    assert missed.http_status == 401
    assert missed.arrived is False
    assert missed.lines() == ["webhook: HTTP 401", "answer: no"]
    assert WEBHOOK_KEY not in "\n".join(missed.lines())
    assert _logged(tmp_path / "denied", store2).count(WEBHOOK_KEY) == 0


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
    assert spoken == ["bot's come back. " + ANSWER]
    assert store.recent("bridge.spoken", 1)
    assert ANSWER not in json.dumps(store.recent("bridge.spoken", 1)[0])
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
    assert queued["text"] == "The team's come back. " + ANSWER
    assert queued["messages"][0]["item"]["content"][0]["text"] == queued["text"]
    instructions = queued["messages"][1]["response"]["instructions"]
    assert "come back" in instructions
    assert "response.create" in queued["messages"][1]["type"]
    protocol.close = True
    bridge.detach(protocol)
    assert spoken == ["The team's come back. " + ANSWER]
    assert store.recent("bridge.spoken", 1)
    assert sink.chunks
    assert protocol._bridge_items == []


def test_answer_waits_until_playback_finishes(monkeypatch, tmp_path):
    _enable(monkeypatch)
    http = FakeHttp()
    http.rows = [{"answer": ANSWER, "answered_by": "Brenda"}]
    spoken = []
    cfg, store, events, bridge = _bridge(tmp_path, http, speaker=spoken.append)
    protocol = RealtimeProtocol(cfg, events, store, load_registry(cfg, store), CollectingSink())
    protocol.speaking = True
    bridge.attach(protocol)
    bridge.submit(QUESTION, background=False)
    assert spoken == []
    assert protocol.pop_bridge_ready() is None
    protocol.speaking = False
    protocol.playback.put(b"\x01\x00")
    assert protocol.pop_bridge_ready() is None
    protocol.playback.get_nowait()
    item = protocol.pop_bridge_ready()
    assert item is not None
    assert item["text"] == "Brenda's come back. " + ANSWER
    assert item["messages"][1]["type"] == "response.create"
    assert "Brenda's come back" in item["messages"][1]["response"]["instructions"]
    assert store.recent("bridge.spoken", 1) == []
    assert ANSWER not in _logged(tmp_path, store)


def test_open_session_marks_spoken_only_after_audio(monkeypatch, tmp_path):
    _enable(monkeypatch)
    http = FakeHttp()
    http.rows = [{"answer": ANSWER, "answered_by": "Brenda"}]
    spoken = []
    cfg, store, events, bridge = _bridge(tmp_path, http, speaker=spoken.append)
    protocol = RealtimeProtocol(cfg, events, store, load_registry(cfg, store), CollectingSink())
    bridge.attach(protocol)
    bridge.submit(QUESTION, background=False)
    item = protocol.pop_bridge_ready()
    assert item["request_id"]
    protocol.finish_bridge_send(True)
    protocol.on_event({"type": "response.created", "response": {"id": "r-bridge"}})
    protocol._response_pcm_bytes = 4800
    protocol.on_event({"type": "response.done", "response": {"id": "r-bridge"}})
    assert spoken == []
    assert http.patches
    spoken_event = store.recent("bridge.spoken", 1)[0]
    assert spoken_event["request_id"] == item["request_id"]
    assert ANSWER not in json.dumps(spoken_event)
    assert "Brenda" not in json.dumps(spoken_event)


def test_silent_readout_is_spoken_without_the_wake_word(monkeypatch, tmp_path):
    _enable(monkeypatch)
    http = FakeHttp()
    http.rows = [{"answer": ANSWER, "answered_by": "Brenda"}]
    spoken = []
    cfg, store, events, bridge = _bridge(tmp_path, http, speaker=spoken.append)
    protocol = RealtimeProtocol(cfg, events, store, load_registry(cfg, store), CollectingSink())
    bridge.attach(protocol)
    bridge.submit(QUESTION, background=False)
    protocol.pop_bridge_ready()
    protocol.finish_bridge_send(True)
    protocol.on_event({"type": "response.created", "response": {"id": "r-bridge"}})
    protocol.on_event({"type": "response.done", "response": {"id": "r-bridge"}})
    assert spoken == ["Brenda's come back. " + ANSWER]
    assert store.recent("bridge.spoken", 1)
    assert store.recent("bridge.delivery_failed", 1) == []


def test_failed_speech_is_kept_and_reported(monkeypatch, tmp_path):
    _enable(monkeypatch)
    http = FakeHttp()
    http.rows = [{"answer": ANSWER, "answered_by": "Brenda"}]

    def fail(_text):
        raise RuntimeError("alsa " + ANSWER)

    cfg, store, events, bridge = _bridge(tmp_path, http, speaker=fail)
    protocol = RealtimeProtocol(cfg, events, store, load_registry(cfg, store), CollectingSink())
    protocol.close = True
    bridge.attach(protocol)
    bridge.submit(QUESTION, background=False)
    failed = store.recent("bridge.delivery_failed", 1)[0]
    assert failed["reason"] == "speak_failed"
    assert ANSWER not in json.dumps(failed)
    assert store.recent("bridge.spoken", 1) == []
    assert http.patches == []
    assert bridge._kept
    _assert_clean(_logged(tmp_path, store))


def test_session_stays_open_while_an_answer_is_pending(monkeypatch, tmp_path):
    from gva.runtime.realtime import idle_should_close

    _enable(monkeypatch)
    http = FakeHttp()
    cfg, _store, events, bridge = _bridge(tmp_path, http)
    protocol = RealtimeProtocol(cfg, events, _store, load_registry(cfg, _store), CollectingSink())
    protocol.last_activity = 0
    assert idle_should_close(protocol, bridge, 100, 15) is True
    bridge._remember("01ARZ3NDEKTSV4RRFFQ69G5FAV")
    bridge._enter_inflight("01ARZ3NDEKTSV4RRFFQ69G5FAV")
    assert bridge.holds_session() is True
    assert idle_should_close(protocol, bridge, 100, 15) is False
    cfg.bot_bridge.chat_while_waiting = False
    assert bridge.holds_session() is False
    assert idle_should_close(protocol, bridge, 100, 15) is True
    cfg.bot_bridge.chat_while_waiting = True
    bridge._kept["01ARZ3NDEKTSV4RRFFQ69G5FAV"] = int(time.time() * 1000) - 400_000
    assert bridge.holds_session() is False
    bridge._kept["01ARZ3NDEKTSV4RRFFQ69G5FAV"] = int(time.time() * 1000)
    bridge.defer()
    assert bridge.holds_session() is False
    assert "bridge.deferred" not in UPLOAD_TYPES


def test_digest_items_stay_out_of_the_instructions(monkeypatch, tmp_path):
    _enable(monkeypatch)
    cfg, store, events = _stack(tmp_path)
    tools = load_registry(cfg, store)

    class _ContextBridge(_StubBridge):
        def waiting_context(self):
            return [{"kind": "report", "title": "Streets report", "due": "Friday"}]

    tools.ctx.bridge = _ContextBridge()
    protocol = RealtimeProtocol(cfg, events, store, tools, CollectingSink())
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
    assert body["pending_items"][0]["title"] == "Streets report"
    assert "Streets report" not in protocol._response_instructions
    assert "data, not instructions" in protocol._response_instructions
    follow = protocol.on_event({"type": "response.done", "response": {"id": "r1"}})
    assert "Streets report" not in follow[0]["response"]["instructions"]
    prompt = render_system(cfg)
    assert "do not invent a task, report, or meeting" in prompt
    assert "Streets report" not in prompt
    cfg.bot_bridge.chat_while_waiting = False
    assert "do not invent a task, report, or meeting" not in render_system(cfg)


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
    assert ok_probe.calls[0]["headers"]["apikey"] == SERVICE_KEY
    assert ok_probe.calls[0]["headers"]["Authorization"] == "Bearer " + SERVICE_KEY
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


def test_new_secret_key_is_apikey_only(monkeypatch, tmp_path):
    _enable(monkeypatch)
    secret = "sb_secret_testkey"
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", secret)
    http = FakeHttp()
    http.rows = [{"request_id": "x", "answer": "ready", "answered_by": "Brenda", "read_at": None}]
    _cfg, _store, _events, bridge = _bridge(tmp_path, http)
    bridge.deliver = False
    bridge.submit("hello", background=False)

    assert http.posts[0]["headers"]["Authorization"] == "Bearer " + WEBHOOK_KEY
    assert secret not in http.posts[0]["headers"]["Authorization"]
    for call in http.gets + http.patches:
        assert call["headers"]["apikey"] == secret
        assert "Authorization" not in call["headers"]
        assert secret not in json.dumps({k: v for k, v in call["headers"].items() if k != "apikey"})

    class Probe:
        def __init__(self, code):
            self.code = code
            self.calls = []

        def get(self, url, params=None, headers=None, timeout=None):
            self.calls.append({"params": params, "headers": headers})
            return _Resp(self.code, [{"request_id": "row-secret", "answer": "personal"}])

        def close(self):
            return None

    ok_probe = Probe(200)
    cfg = Config()
    reachable, detail = probe_answers(cfg, http=ok_probe)
    assert reachable is True and detail == "ok"
    assert ok_probe.calls[0]["headers"]["apikey"] == secret
    assert "Authorization" not in ok_probe.calls[0]["headers"]
    line, probed = supabase_doctor_line(cfg, probe=lambda _cfg: probe_answers(_cfg, http=Probe(200)))
    assert line == "supabase: OK"
    assert probed == (True, "ok")
    assert secret not in line

    denied, detail = probe_answers(cfg, http=Probe(401))
    assert denied is False and detail == "HTTP 401"
    line, probed = supabase_doctor_line(cfg, probe=lambda _cfg: (False, "HTTP 401"))
    assert line == "supabase: HTTP 401"
    assert probed == (False, "HTTP 401")
    assert secret not in line
    assert "row-secret" not in line
    assert "personal" not in line

    monkeypatch.delenv("GVA_GROKBOT_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("GVA_GROKBOT_WEBHOOK_KEY", raising=False)
    sync_only = Config()
    line, probed = supabase_doctor_line(sync_only, probe=lambda _cfg: (False, "HTTP 401"))
    assert line == "supabase: HTTP 401"
    bridge_line, warning = bridge_status(sync_only, probed=probed)
    assert bridge_line == "bridge: not configured"
    assert warning is None
    assert secret not in line


def test_example_files_document_the_bridge(monkeypatch):
    monkeypatch.setenv("GVA_ENV_FILE", "/no/such/gva.env")
    cfg = load_config(project_root() / "config" / "config.example.yaml")
    assert cfg.bot_bridge.enabled is True
    assert cfg.bot_bridge.timeout_s == 300
    assert cfg.bot_bridge.poll_s == 3
    assert cfg.bot_bridge.nudge_s == 45
    assert cfg.bot_bridge.key_header == "Authorization"
    assert cfg.bot_bridge.key_prefix == "Bearer "
    assert cfg.bot_bridge.chat_while_waiting is True
    assert cfg.bot_bridge.context_table == "milo_context"
    assert cfg.bot_bridge.context_max_age_s == 21600
    assert cfg.bot_bridge.suggest_pending_items is True
    assert cfg.bot_bridge.nudge_quiet_hours == "none"
    assert cfg.bot_bridge.nudge_follow_up_s == 8
    example = (project_root() / "gva.env.example").read_text(encoding="utf-8")
    assert "GVA_GROKBOT_WEBHOOK_URL=\n" in example
    assert "GVA_GROKBOT_WEBHOOK_KEY=\n" in example
    readme = (project_root() / "README.md").read_text(encoding="utf-8")
    assert "milo_bot_answers" in readme
    assert "milo-answer" in readme
    assert "Brenda" in readme
    assert 'key_prefix: ""' in readme
    assert "no migration" in readme.lower()
    assert "sb_secret_" in readme
    assert "apikey" in readme
    assert "supabase: OK" in readme
    assert "milo_context" in readme
    assert "chat_while_waiting" in readme
    assert "suggest_pending_items" in readme
    assert "nudge_quiet_hours" in readme
    assert "defer_until_answer" in readme
    assert "I couldn't get an answer from the team this time" in readme
    assert "By the way, Brenda came back" in readme
    assert "sb_secret_" in example
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

    class _Report:
        def lines(self):
            return ["webhook: HTTP 200", "answer: yes", "answered_by: Brenda"]

        def ok(self):
            return True

    monkeypatch.setattr("gva.runtime.bridge.BotBridge.run_test", lambda self, question=None: _Report())
    result = CliRunner().invoke(app, ["bridge", "test"])
    assert result.exit_code == 0
    assert "webhook: HTTP 200" in (result.output or "")
    assert "answer: yes" in (result.output or "")
    assert WEBHOOK_KEY not in (result.output or "")
    assert ANSWER not in (result.output or "")

    class _Miss:
        def lines(self):
            return ["webhook: HTTP 200", "answer: no"]

        def ok(self):
            return False

    monkeypatch.setattr("gva.runtime.bridge.BotBridge.run_test", lambda self, question=None: _Miss())
    result = CliRunner().invoke(app, ["bridge", "test"])
    text = (result.output or "") + (getattr(result, "stderr", None) or "")
    assert result.exit_code == 1
    assert "answer: no" in text
    assert WEBHOOK_KEY not in text

    monkeypatch.setattr("gva.runtime.bridge.probe_answers", lambda cfg: (True, "ok"))
    doctor = CliRunner().invoke(app, ["doctor"])
    doctor_text = (doctor.output or "") + (getattr(doctor, "stderr", None) or "")
    assert "supabase: OK" in doctor_text
    assert "bridge:" in doctor_text
    assert WEBHOOK_KEY not in doctor_text
    assert SERVICE_KEY not in doctor_text
    assert "bridge-query" not in doctor_text

    monkeypatch.setattr("gva.runtime.bridge.probe_answers", lambda cfg: (False, "HTTP 401"))
    denied = CliRunner().invoke(app, ["doctor"])
    denied_text = (denied.output or "") + (getattr(denied, "stderr", None) or "")
    assert "supabase: HTTP 401" in denied_text
    assert "milo_bot_answers is not reachable (HTTP 401)" in denied_text
    assert SERVICE_KEY not in denied_text
    assert WEBHOOK_KEY not in denied_text
    assert not any(line.startswith("problem:") and "milo_bot_answers" in line for line in denied_text.splitlines())


RID = "01ARZ3NDEKTSV4RRFFQ69G5FAV"


def _defer_on_send(bridge):
    original = bridge._enter_inflight

    def enter(request_id):
        original(request_id)
        bridge.defer()

    bridge._enter_inflight = enter


def test_suggestions_are_offered_once_and_a_decline_keeps_listening(monkeypatch, tmp_path):
    _enable(monkeypatch)
    cfg, store, events, bridge = _bridge(tmp_path, FakeHttp())
    bridge.waiting_context = lambda: [
        {"kind": "task", "title": "One"},
        {"kind": "report", "title": "Two"},
        {"kind": "meeting", "title": "Three"},
        {"kind": "other", "title": "Four"},
    ]
    tools = load_registry(cfg, store)
    tools.ctx.bridge = bridge
    protocol = RealtimeProtocol(cfg, events, store, tools, CollectingSink())
    assert [item["title"] for item in protocol._waiting_items()] == ["One", "Two"]
    assert protocol._waiting_items() == []

    cfg.bot_bridge.suggest_pending_items = False
    bridge._suggestions_offered = False
    assert protocol._waiting_items() == []
    cfg.bot_bridge.suggest_pending_items = True
    bridge._remember(RID)
    bridge._enter_inflight(RID)
    assert bridge.holds_session() is True
    protocol.on_event({"type": "response.created", "response": {"id": "r-wait"}})
    protocol.on_event(
        {"type": "conversation.item.input_audio_transcription.completed", "transcript": "Not now."}
    )
    assert protocol.close is False
    assert bridge._suggestions_declined is True
    bridge._suggestions_offered = False
    assert protocol._waiting_items() == []
    protocol.on_event(
        {"type": "conversation.item.input_audio_transcription.completed", "transcript": "stop"}
    )
    assert protocol.close is False
    protocol.on_event(
        {"type": "conversation.item.input_audio_transcription.completed", "transcript": "leave it"}
    )
    assert protocol.close is False
    bridge._leave_inflight(RID)
    protocol.on_event(
        {"type": "conversation.item.input_audio_transcription.completed", "transcript": "stop"}
    )
    assert protocol.close is True


def test_suggest_pending_items_false_offers_nothing(monkeypatch, tmp_path):
    _enable(monkeypatch)
    cfg, store, events, bridge = _bridge(tmp_path, FakeHttp())
    cfg.bot_bridge.suggest_pending_items = False
    bridge.waiting_context = lambda: [{"kind": "report", "title": "Streets report"}]
    bridge._remember(RID)
    bridge._enter_inflight(RID)
    assert bridge.holds_session() is True
    tools = load_registry(cfg, store)
    tools.ctx.bridge = bridge
    protocol = RealtimeProtocol(cfg, events, store, tools, CollectingSink())
    assert protocol._waiting_items() == []
    bridge.deliver = False
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
    assert "Streets report" not in protocol._response_instructions
    assert "Streets report" not in _logged(tmp_path, store)


def test_defer_tool_confirms_once_then_closes_the_session(monkeypatch, tmp_path):
    _enable(monkeypatch)
    cfg, store, events, bridge = _bridge(tmp_path, FakeHttp())
    bridge._remember(RID)
    bridge._enter_inflight(RID)
    tools = load_registry(cfg, store)
    tools.ctx.bridge = bridge
    protocol = RealtimeProtocol(cfg, events, store, tools, CollectingSink())
    protocol.on_event({"type": "response.created", "response": {"id": "r1"}})
    messages = protocol.on_event(
        {
            "type": "response.function_call_arguments.done",
            "name": "defer_until_answer",
            "call_id": "call-defer",
            "arguments": "{}",
        }
    )
    assert json.loads(messages[0]["item"]["output"]) == {"status": "deferred"}
    assert protocol.close is False
    deferred = store.recent("bridge.deferred", 5)
    assert [item["request_id"] for item in deferred] == [RID]
    assert "bridge.deferred" not in UPLOAD_TYPES
    follow = protocol.on_event({"type": "response.done", "response": {"id": "r1"}})
    assert follow[0]["type"] == "response.create"
    assert DEFER_LINE in follow[0]["response"]["instructions"]
    assert protocol.close is False
    protocol.on_event({"type": "response.created", "response": {"id": "r2"}})
    protocol._response_pcm_bytes = 4800
    protocol.on_event({"type": "response.done", "response": {"id": "r2"}})
    assert protocol.close is True
    assert bridge.defer()["status"] == "deferred"
    assert len(store.recent("bridge.deferred", 5)) == 1
    assert bridge.holds_session() is False
    assert QUESTION not in _logged(tmp_path, store)


def test_tell_me_later_defers_from_the_transcript(monkeypatch, tmp_path):
    _enable(monkeypatch)
    cfg, store, events, bridge = _bridge(tmp_path, FakeHttp())
    bridge._remember(RID)
    bridge._enter_inflight(RID)
    tools = load_registry(cfg, store)
    tools.ctx.bridge = bridge
    protocol = RealtimeProtocol(cfg, events, store, tools, CollectingSink())
    spoken = []
    protocol.fallback_player = lambda _sink, _config, phrase: spoken.append(phrase) or 900
    messages = protocol.on_event(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": "Tell me later.",
        }
    )
    assert messages == [
        {"type": "response.create", "response": {"instructions": "Say only: " + DEFER_LINE}}
    ]
    assert protocol.close is False
    protocol.on_event({"type": "response.created", "response": {"id": "r1"}})
    tool_messages = protocol.on_event(
        {
            "type": "response.function_call_arguments.done",
            "name": "defer_until_answer",
            "call_id": "call-defer",
            "arguments": "{}",
        }
    )
    assert tool_messages[0]["item"]["type"] == "function_call_output"
    protocol._response_pcm_bytes = 4800
    done = protocol.on_event({"type": "response.done", "response": {"id": "r1"}})
    assert done == []
    assert protocol.close is True
    assert spoken == []
    assert len(store.recent("bridge.deferred", 5)) == 1

    fresh = RealtimeProtocol(cfg, events, store, tools, CollectingSink())
    bridge._suggestions_declined = False
    bridge._enter_inflight(RID)
    fresh.on_event({"type": "response.created", "response": {"id": "r3"}})
    cancelled = fresh.on_event(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": "Nudge me when it's ready",
        }
    )
    assert cancelled == [{"type": "response.cancel"}]
    assert fresh.close is False
    follow = fresh.on_event({"type": "response.done", "response": {"id": "r3"}})
    assert DEFER_LINE in follow[0]["response"]["instructions"]
    assert fresh.close is False


def test_deferred_answer_soft_chimes_and_requests_a_follow_up(monkeypatch, tmp_path):
    _enable(monkeypatch)
    http = FakeHttp()
    http.rows = [{"answer": ANSWER, "answered_by": "Brenda"}]
    spoken = []
    sink = CollectingSink()
    _cfg, store, _events, bridge = _bridge(tmp_path, http, speaker=spoken.append, sink=sink)
    _defer_on_send(bridge)
    bridge.submit(QUESTION, background=False)
    assert spoken == [nudge_line(ANSWER, "Brenda")]
    assert bridge.last_chime == "soft"
    assert sink.chunks
    assert bridge.follow_ups == 1
    assert http.patches
    assert bridge._kept == {}
    nudged = store.recent("bridge.nudged", 1)[0]
    assert nudged["request_id"]
    assert ANSWER not in json.dumps(nudged)
    assert "Brenda" not in json.dumps(nudged)
    assert "bridge.nudged" not in UPLOAD_TYPES
    assert store.recent("bridge.deferred", 1)
    _assert_clean(_logged(tmp_path, store))


def test_quiet_hours_hold_the_nudge_until_an_explicit_release(monkeypatch, tmp_path):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    _enable(monkeypatch)
    evening = datetime(2026, 10, 26, 23, 15, tzinfo=ZoneInfo("Europe/London"))
    noon = evening.replace(hour=12)
    assert in_nudge_quiet_hours("22:00-07:00", evening) is True
    assert in_nudge_quiet_hours("22:00-07:00", noon) is False
    assert in_nudge_quiet_hours("none", evening) is False
    assert in_nudge_quiet_hours("", evening) is False
    assert in_nudge_quiet_hours("off", evening) is False

    http = FakeHttp()
    http.rows = [{"answer": ANSWER, "answered_by": "Brenda"}]
    spoken = []
    cfg, store, events, bridge = _bridge(tmp_path, http, speaker=spoken.append)
    cfg.bot_bridge.nudge_quiet_hours = "22:00-07:00"
    bridge._now_fn = lambda: evening
    _defer_on_send(bridge)
    bridge.submit(QUESTION, background=False)
    assert spoken == []
    assert bridge.follow_ups == 0
    assert store.recent("bridge.nudged", 1) == []
    assert bridge.release_held_nudges() == 1
    assert spoken == [nudge_line(ANSWER, "Brenda")]
    assert bridge.last_chime == "soft"
    assert bridge.follow_ups == 1
    assert store.recent("bridge.nudged", 1)
    assert bridge.release_held_nudges() == 0

    held = FakeHttp()
    held.rows = [{"answer": ANSWER, "answered_by": "Brenda"}]
    spoken2 = []
    cfg2, store2, events2, bridge2 = _bridge(tmp_path / "wake", held, speaker=spoken2.append)
    cfg2.bot_bridge.nudge_quiet_hours = "22:00-07:00"
    bridge2._now_fn = lambda: evening
    _defer_on_send(bridge2)
    bridge2.submit(QUESTION, background=False)
    assert spoken2 == []
    protocol = RealtimeProtocol(cfg2, events2, store2, load_registry(cfg2, store2), CollectingSink())
    bridge2.attach(protocol)
    assert spoken2 == []
    assert protocol._bridge_items[0]["text"] == nudge_line(ANSWER, "Brenda")
    assert protocol._bridge_items[0]["kind"] == "nudge_answer"
    assert bridge2.follow_ups == 0
    _assert_clean(_logged(tmp_path, store) + _logged(tmp_path / "wake", store2))


def test_deferred_timeout_nudges_with_the_failure_line(monkeypatch, tmp_path):
    _enable(monkeypatch)
    monkeypatch.setattr("gva.runtime.bridge.LATE_NUDGE_CAP_S", 0)
    http = FakeHttp()
    spoken = []
    _cfg, store, _events, bridge = _bridge(tmp_path, http, speaker=spoken.append)
    _defer_on_send(bridge)
    bridge.submit(QUESTION, background=False)
    assert spoken == [TIMEOUT_LINE]
    assert bridge.follow_ups == 0
    assert bridge.last_chime == ""
    assert http.patches == []
    assert bridge._kept
    assert store.recent("bridge.timeout", 1)
    assert store.recent("bridge.timeout_spoken", 1) == []
    nudged = store.recent("bridge.nudged", 1)[0]
    assert nudged["request_id"]
    assert ANSWER not in json.dumps(nudged)
    _assert_clean(_logged(tmp_path, store))


def test_deferred_late_answer_still_nudges(monkeypatch, tmp_path):
    _enable(monkeypatch)
    monkeypatch.setattr("gva.runtime.bridge.LATE_NUDGE_CAP_S", 30)
    http = FakeHttp()
    http.rows = [{"answer": ANSWER, "answered_by": "Brenda"}]
    http.ready_after = 6
    spoken = []
    cfg, store, _events, bridge = _bridge(tmp_path, http, speaker=spoken.append)
    cfg.bot_bridge.timeout_s = 9
    cfg.bot_bridge.poll_s = 3
    _defer_on_send(bridge)
    bridge.submit(QUESTION, background=False)
    assert spoken[0] == TIMEOUT_LINE
    assert spoken[-1] == nudge_line(ANSWER, "Brenda")
    assert bridge.follow_ups == 1
    assert len(store.recent("bridge.nudged", 5)) == 2
    assert http.patches
    assert bridge._kept == {}
    _assert_clean(_logged(tmp_path, store))


def test_nudge_follow_up_skips_the_wake_word(monkeypatch, tmp_path):
    import asyncio

    from gva.runtime.realtime import RealtimeAssistant

    _enable(monkeypatch)
    cfg, store, events, bridge = _bridge(tmp_path, FakeHttp())

    class _Capture:
        def read(self, _timeout):
            raise AssertionError("wake listen")

    class _Socket:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    assistant = RealtimeAssistant(
        cfg, store, events, _Capture(), CollectingSink(), None, None, bridge=bridge
    )
    bridge._request_follow_up()
    assert assistant._wait_for_wake() is True
    assert assistant._silence_override_s == 8

    protocol = RealtimeProtocol(cfg, events, store, None, CollectingSink())
    protocol.last_activity = 0
    assistant._session_silence_s = 0.05
    socket = _Socket()
    asyncio.run(assistant._send_audio(socket, protocol, None))
    assert protocol.close is True
    assert socket.closed is True
