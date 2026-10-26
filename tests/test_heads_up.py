"""Meeting heads-up from a milo_context digest. No live network."""

import json
from datetime import datetime, timedelta, timezone

from gva.backend.sync_supabase import UPLOAD_TYPES
from gva.config import Config
from gva.runtime.audio import CollectingSink
from gva.runtime.heads_up import MeetingHeadsUp, heads_up_line, parse_due
from gva.runtime.realtime import RealtimeProtocol
from gva.telemetry.events import EventLog
from gva.telemetry.store import Store

DUE = "2026-10-27T10:00:00+00:00"
WHEN = datetime(2026, 10, 27, 10, 0, tzinfo=timezone.utc)
NOTE = "With Sarah Park, Park Lane"
TITLE = "Call"


def _stack(tmp_path):
    cfg = Config()
    cfg.telemetry.log_dir = str(tmp_path / "log")
    cfg.telemetry.db_path = str(tmp_path / "metrics.db")
    store = Store(tmp_path / "metrics.db")
    events = EventLog("milo-01", "dev", tmp_path / "log", store, validate=True)
    return cfg, store, events


def _heads(tmp_path, spoken):
    cfg, store, events = _stack(tmp_path)
    sink = CollectingSink()
    heads = MeetingHeadsUp(cfg, events, sink=sink, speaker=spoken.append)
    return cfg, store, heads, sink


def _meeting(**extra):
    item = {"kind": "meeting", "title": TITLE, "due": DUE, "note": NOTE}
    item.update(extra)
    return item


def _logged(store) -> str:
    return json.dumps(store.recent(limit=50))


def test_parse_accepts_only_iso_with_an_offset_and_builds_the_line():
    assert parse_due(DUE) == WHEN
    assert parse_due("2026-10-27T10:00:00Z") == WHEN
    assert parse_due("2026-10-27T10:00:00") is None
    assert parse_due("2026-10-27") is None
    assert parse_due("Friday") is None
    assert parse_due("10:00") is None
    assert parse_due("") is None
    assert parse_due(None) is None
    assert (
        heads_up_line(TITLE, NOTE, 10)
        == "Heads up, your call with Sarah Park from Park Lane starts in 10 minutes."
    )
    assert heads_up_line(TITLE, "With Sarah Park", 1) == "Heads up, your call with Sarah Park starts in a minute."
    assert heads_up_line("Board call", "", 10) == "Heads up, Board call starts in 10 minutes."


def test_a_future_meeting_is_scheduled_once_and_a_past_one_is_ignored(tmp_path):
    spoken = []
    _cfg, store, heads, _sink = _heads(tmp_path, spoken)
    now = WHEN - timedelta(minutes=30)
    heads.sync(
        [
            _meeting(),
            _meeting(due="Friday"),
            _meeting(due="2026-10-27T09:00:00"),
            {"kind": "report", "title": "Streets report", "due": DUE},
            _meeting(title="Yesterday", due="2026-10-27T09:00:00+00:00"),
        ],
        now,
    )
    assert spoken == []
    assert len(store.recent("heads_up.scheduled", 10)) == 1
    assert "heads_up.scheduled" not in UPLOAD_TYPES
    heads.sync([_meeting()], now)
    assert len(store.recent("heads_up.scheduled", 10)) == 1
    assert "Sarah" not in _logged(store)
    assert "Park Lane" not in _logged(store)
    assert DUE not in _logged(store)


def test_the_same_meeting_is_not_announced_twice(tmp_path):
    spoken = []
    cfg, store, heads, _sink = _heads(tmp_path, spoken)
    fire = WHEN - timedelta(minutes=10)
    heads.sync([_meeting()], fire)
    assert spoken == ["Heads up, your call with Sarah Park from Park Lane starts in 10 minutes."]
    assert heads.follow_ups == 1
    assert len(store.recent("heads_up.spoken", 5)) == 1
    assert "Sarah" not in _logged(store)
    heads.sync([_meeting()], fire)
    assert len(spoken) == 1
    assert len(store.recent("heads_up.scheduled", 5)) == 1
    assert len(store.recent("heads_up.spoken", 5)) == 1

    again = []
    events = EventLog("milo-01", "dev", tmp_path / "log2", store, validate=True)
    later = MeetingHeadsUp(cfg, events, speaker=again.append)
    later.sync([_meeting()], fire)
    assert again == []
    scheduled = [item for item in store.recent(limit=20) if item["type"] == "heads_up.scheduled"]
    assert len(scheduled) == 1


def test_a_meeting_that_leaves_the_digest_is_cancelled(tmp_path):
    spoken = []
    _cfg, store, heads, _sink = _heads(tmp_path, spoken)
    early = WHEN - timedelta(minutes=30)
    heads.sync([_meeting()], early)
    # A failed read is not a newer digest, so the heads-up is still said on time.
    heads.sync([], WHEN - timedelta(minutes=10), status="unavailable")
    assert spoken == ["Heads up, your call with Sarah Park from Park Lane starts in 10 minutes."]
    assert store.recent("heads_up.spoken", 1)

    dropped = []
    _cfg2, store2, heads2, _sink2 = _heads(tmp_path / "drop", dropped)
    heads2.sync([_meeting()], early)
    heads2.sync([], early, status="empty")
    heads2.sync([], WHEN - timedelta(minutes=10))
    assert dropped == []
    assert store2.recent("heads_up.spoken", 5) == []
    assert len(store2.recent("heads_up.scheduled", 5)) == 1


def test_heads_up_waits_until_playback_finishes(tmp_path):
    spoken = []
    cfg, store, events = _stack(tmp_path)
    sink = CollectingSink()
    heads = MeetingHeadsUp(cfg, events, sink=sink, speaker=spoken.append)
    protocol = RealtimeProtocol(cfg, events, store, None, sink)
    protocol.speaking = True
    heads.attach(protocol)
    heads.sync([_meeting()], WHEN - timedelta(minutes=10))
    assert spoken == []
    assert sink.chunks == []
    assert store.recent("heads_up.spoken", 5) == []
    assert len(store.recent("heads_up.scheduled", 5)) == 1
    protocol.speaking = False
    heads.sync([_meeting()], WHEN - timedelta(minutes=10))
    assert spoken == ["Heads up, your call with Sarah Park from Park Lane starts in 10 minutes."]
    assert heads.last_chime == "soft"
    assert sink.chunks
    assert heads.follow_ups == 0
    assert len(store.recent("heads_up.spoken", 5)) == 1
    assert "Sarah Park" not in _logged(store)
    assert "heads_up.spoken" not in UPLOAD_TYPES
    heads.sync([_meeting()], WHEN - timedelta(minutes=10))
    assert len(spoken) == 1


def test_quiet_hours_hold_the_heads_up(tmp_path):
    spoken = []
    cfg, _store, heads, _sink = _heads(tmp_path, spoken)
    cfg.bot_bridge.nudge_quiet_hours = "23:00-23:35"
    due = datetime(2026, 10, 26, 23, 40, tzinfo=timezone.utc)
    item = {
        "kind": "meeting",
        "title": TITLE,
        "due": due.isoformat(),
        "note": NOTE,
    }
    heads.sync([item], datetime(2026, 10, 26, 23, 30, tzinfo=timezone.utc))
    assert spoken == []
    heads.sync([item], datetime(2026, 10, 26, 23, 36, tzinfo=timezone.utc))
    assert spoken == ["Heads up, your call with Sarah Park from Park Lane starts in 4 minutes."]
