import time

from gva.runtime.audio import rms, tone
from gva.runtime.wake import ThresholdWake
from tests.fakes import build_assistant, say, utterance


class OnceWake:
    """Wakes on the first loud phrase, then stays quiet so energy barge-in can be tested."""

    def __init__(self):
        self.logic = ThresholdWake(0.5, 1, 0)
        self.model_name = "scripted"
        self._fired = False

    def score(self, frame: bytes) -> float:
        return 0.95 if rms(frame) >= 2000 else 0.0

    def observe(self, score: float, now=None) -> bool:
        if self._fired:
            return False
        fired = self.logic.observe(score, now if now is not None else time.monotonic())
        if fired:
            self._fired = True
        return fired


def _types(assistant):
    return [event["type"] for event in assistant.store.all_events()]


def test_wake_and_reply(tmp_path, monkeypatch):
    assistant, grok, _stt, sink = build_assistant(
        tmp_path,
        monkeypatch,
        utterance(),
        ["what time is it"],
        [say("It is noon.")],
    )
    assistant.run()
    assert grok.calls == 1
    assert "turn.completed" in _types(assistant)
    assert any(chunk for chunk in sink.chunks)
    turns = assistant.store.recent("turn.completed", 5)
    assert turns[0]["status"] == "ok"
    assert "noon" in (turns[0].get("response_redacted") or "").casefold()


def test_stop_does_not_call_grok(tmp_path, monkeypatch):
    assistant, grok, _stt, sink = build_assistant(tmp_path, monkeypatch, utterance(), ["stop"], [say("should not play")])
    assistant.run()
    assert grok.calls == 0
    turns = assistant.store.recent("turn.completed", 5)
    assert turns[0]["intent"] == "stop"
    assert turns[0]["status"] == "cancelled"
    assert sink.chunks


def test_follow_up_without_another_wake(tmp_path, monkeypatch):
    frames = utterance() + utterance()
    assistant, grok, _stt, _sink = build_assistant(
        tmp_path,
        monkeypatch,
        frames,
        ["what time is it", "and the date"],
        [say("It is noon."), say("It is Sunday.")],
    )
    assistant.run()
    assert grok.calls == 2
    assert grok.seen[1][-1]["content"] == "and the date"


def test_energy_barge_in_stop_halts_playback(tmp_path, monkeypatch):
    assistant, grok, stt, sink = build_assistant(
        tmp_path,
        monkeypatch,
        utterance(),
        ["what time is it", "stop"],
        [say("This reply is long enough to interrupt.")],
        wake=OnceWake(),
    )
    assistant.cfg.conversation.energy_barge_in = True
    assistant.cfg.conversation.barge_in_min_ms = 80
    assistant.capture.barge = [tone(80, amp=8000)]
    assistant.run()
    assert grok.calls == 1
    assert stt.calls == 2
    assert sink.stopped


def test_wake_during_playback_starts_another_turn(tmp_path, monkeypatch):
    frames = utterance() + utterance()
    assistant, grok, _stt, sink = build_assistant(
        tmp_path,
        monkeypatch,
        frames,
        ["what time is it", "what is the weather"],
        [say("It is noon."), say("It is dry.")],
    )
    assistant.capture.barge = [tone(80, amp=8000)]
    assistant.run()
    assert sink.stopped
    assert grok.calls == 2
