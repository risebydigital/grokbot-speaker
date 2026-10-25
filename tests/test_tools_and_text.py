import time

from gva.runtime.conversation import Conversation
from gva.runtime.grok_client import scripted_result
from gva.runtime.textutil import is_stop_phrase, normalise_for_speech, take_sentences
from gva.runtime.tools.registry import load_registry
from gva.telemetry.events import EventLog
from gva.telemetry.store import Store
from tests.fakes import call, test_config


def test_stop_phrases_and_speech_normaliser():
    phrases = ["stop", "cancel", "never mind", "be quiet", "shut up"]
    assert is_stop_phrase("Stop.", phrases)
    assert is_stop_phrase("please stop now", phrases)
    assert not is_stop_phrase("stopwatch", phrases)
    spoken = normalise_for_speech("**Grok** says £2 at 07:30.", {"Grok": "Grock"})
    assert "Grock" in spoken
    assert "£" not in spoken
    assert "07:30" not in spoken
    ready, rest = take_sentences("Hello there. More")
    assert ready == ["Hello there."]
    assert rest == "More"


def test_timer_unknown_tool_and_timeout(tmp_path, monkeypatch):
    cfg = test_config(tmp_path, monkeypatch)
    store = Store(tmp_path / "metrics.db")
    registry = load_registry(cfg, store)
    started = registry.run("t1", "set_timer", '{"duration_seconds": 5, "label": "tea"}')
    assert started.status == "ok"
    assert store.list_timers()[0]["label"] == "tea"
    unknown = registry.run("t2", "not_a_tool", "{}")
    assert unknown.status == "rejected"
    assert unknown.error_code == "tool.unknown"
    spec = registry.specs["get_time"]
    spec.timeout_s = 0.05

    def slow(_args, _ctx):
        time.sleep(1)
        return {"ok": True}

    spec.handler = slow
    timed = registry.run("t3", "get_time", "{}")
    assert timed.status == "timeout"


def test_high_risk_tool_waits_for_yes(tmp_path, monkeypatch):
    cfg = test_config(tmp_path, monkeypatch)
    store = Store(tmp_path / "metrics.db")
    events = EventLog(cfg.device.id, cfg.device.env, tmp_path / "log", store, validate=True)
    events.ctx.turn_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    events.ctx.session_id = "01ARZ3NDEKTSV4RRFFQ69G5FAW"
    registry = load_registry(cfg, store)
    registry.specs["switch_device"].risk = "high"

    class Grok:
        def __init__(self):
            self.calls = 0

        def complete(self, messages, tools=None, **kwargs):
            self.calls += 1
            return scripted_result(
                tools=[call("switch_device", '{"device":"kitchen_lights","state":"on"}')],
                usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            )

    grok = Grok()
    conversation = Conversation(cfg, grok, registry, events, store)
    spoken = []
    first = conversation.execute("Turn the kitchen lights on.", lambda text: spoken.append(text) or False)
    assert "confirm" in first.spoken.casefold()
    assert store.get_device("kitchen_lights") is None
    second = conversation.execute("yes", lambda text: spoken.append(text) or False)
    assert second.tools_called == ["switch_device"]
    assert store.get_device("kitchen_lights")["state"] == "on"
    assert grok.calls == 1
