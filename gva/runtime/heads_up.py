"""Ten-minute heads-up for meetings in the latest milo_context digest.

Brenda puts the start in ``due`` as ISO 8601 with an offset, and who it is
with in ``note`` (``With Sarah Park, Park Lane``). Anything that does not
parse is ignored. The same title and due are not announced twice, and a
meeting that disappears from a newer digest is dropped before it is spoken.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

from gva.config import Config
from gva.runtime.bridge import in_nudge_quiet_hours
from gva.telemetry.events import EventLog

_OFFSET = re.compile(r"(?:Z|[+-]\d{2}:\d{2})$")
_WITH_PLACE = re.compile(r"^with\s+(.+?),\s+(.+)$", re.IGNORECASE)
_POLL_S = 5.0
# Still say it if playback or quiet hours delayed us, but not once the meeting is old.
_LATE_GRACE = timedelta(minutes=2)


@dataclass
class _Plan:
    key: str
    title: str
    due_raw: str
    due: datetime
    note: str
    fire_at: datetime
    state: str  # scheduled, or due (waiting to be said)


def parse_due(value: str | None) -> datetime | None:
    """An aware instant, or None. A missing offset is ignored, never assumed."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or _OFFSET.search(text) is None:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def meeting_key(title: str, due: str) -> str:
    return " ".join(title.casefold().split()) + "\n" + " ".join(due.split())


def heads_up_line(title: str, note: str | None, minutes: int) -> str:
    """The sentence Milo says. The time is the real lead, not a guessed clock."""
    when = _when(minutes)
    who, where = _who(note)
    if who and where:
        return f"Heads up, your call with {who} from {where} starts {when}."
    if who:
        return f"Heads up, your call with {who} starts {when}."
    return f"Heads up, {title} starts {when}."


def lead_minutes(due: datetime, now: datetime) -> int:
    seconds = (due - now).total_seconds()
    return max(0, int(round(seconds / 60.0)))


def _when(minutes: int) -> str:
    if minutes <= 0:
        return "now"
    if minutes == 1:
        return "in a minute"
    return f"in {minutes} minutes"


def _who(note: str | None) -> tuple[str | None, str | None]:
    text = " ".join((note or "").split())
    if not text:
        return None, None
    match = _WITH_PLACE.fullmatch(text)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    if text.lower().startswith("with "):
        who = text[5:].strip()
        return (who or None), None
    return None, None


class MeetingHeadsUp:
    """Watch the digest and speak one heads-up per meeting."""

    def __init__(
        self,
        cfg: Config,
        events: EventLog,
        http: Any | None = None,
        sink=None,
        speaker: Callable[[str], None] | None = None,
        now: Callable[[], datetime] | None = None,
    ):
        self.cfg = cfg
        self.events = events
        self._http = http
        self._owned = None
        self.sink = sink
        self._speaker = speaker
        self._now_fn = now
        self._lock = threading.Lock()
        self._plans: dict[str, _Plan] = {}
        self._spoken: set[str] = set()
        self._protocol = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.on_follow_up: Callable[[], None] | None = None
        self.follow_ups = 0
        self.last_chime = ""
        self._load_spoken()

    def start(self) -> None:
        if not self.cfg.heads_up.enabled:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="gva-heads-up", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def attach(self, protocol) -> None:
        with self._lock:
            self._protocol = protocol

    def detach(self) -> None:
        with self._lock:
            self._protocol = None

    def poll(self, now: datetime | None = None) -> None:
        """Read the digest and speak anything that is due. Tests call this directly."""
        moment = self._clock(now)
        if not self.cfg.heads_up.enabled:
            self.sync([], moment, status="empty")
            return
        from gva.runtime.milo_context import load_digest

        status, items = load_digest(self.cfg, self._client(), now=moment)
        self.sync(items, moment, status=status)

    def sync(self, items: list[dict[str, str]] | None, now: datetime, *, status: str = "fresh") -> None:
        """Apply one digest snapshot. ``unavailable`` leaves the current schedule alone."""
        moment = self._clock(now)
        if not self.cfg.heads_up.enabled:
            status = "empty"
            items = []
        if status == "unavailable":
            self._flush(moment)
            return
        if status in {"empty", "stale"}:
            items = []
        self._reconcile(list(items or []), moment)
        self._flush(moment)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll()
            except Exception:
                pass
            self._stop.wait(_POLL_S)

    def _reconcile(self, items: list[dict[str, str]], now: datetime) -> None:
        fresh: dict[str, _Plan] = {}
        lead = timedelta(minutes=int(self.cfg.heads_up.minutes_before))
        for item in items:
            if str(item.get("kind") or "") != "meeting":
                continue
            due_raw = str(item.get("due") or "")
            due = parse_due(due_raw)
            if due is None or due <= now:
                continue
            title = " ".join(str(item.get("title") or "").split())
            if not title:
                continue
            key = meeting_key(title, due_raw)
            if key in self._spoken:
                continue
            fresh[key] = _Plan(
                key=key,
                title=title,
                due_raw=due_raw,
                due=due,
                note=str(item.get("note") or ""),
                fire_at=due - lead,
                state="scheduled",
            )
        with self._lock:
            for key in list(self._plans):
                current = self._plans.get(key)
                if key not in fresh and (current is None or current.state != "saying"):
                    self._plans.pop(key, None)
            for key, plan in fresh.items():
                current = self._plans.get(key)
                if current is None:
                    self._plans[key] = plan
                    self._emit("heads_up.scheduled")
                    continue
                current.note = plan.note
                current.due = plan.due
                current.fire_at = plan.fire_at
                current.title = plan.title

    def _flush(self, now: datetime) -> None:
        quiet = self._quiet(now)
        busy = self._busy()
        claimed: list[_Plan] = []
        gave_up = False
        with self._lock:
            for plan in list(self._plans.values()):
                if plan.state == "scheduled" and now >= plan.fire_at:
                    plan.state = "due"
                if plan.state != "due":
                    continue
                if now >= plan.due + _LATE_GRACE:
                    self._plans.pop(plan.key, None)
                    self._spoken.add(plan.key)
                    gave_up = True
                    continue
                if quiet or busy:
                    continue
                plan.state = "saying"
                claimed.append(plan)
        if gave_up:
            self._save_spoken()
        for plan in claimed:
            self._speak(plan, now)

    def _speak(self, plan: _Plan, now: datetime) -> None:
        text = heads_up_line(plan.title, plan.note, lead_minutes(plan.due, now))
        self._chime()
        if not self._say(text):
            with self._lock:
                if plan.key in self._plans:
                    plan.state = "due"
            return
        with self._lock:
            self._plans.pop(plan.key, None)
            self._spoken.add(plan.key)
        self._save_spoken()
        self._emit("heads_up.spoken")
        if not self._live():
            self._request_follow_up()

    def _busy(self) -> bool:
        """True while the current reply is still coming out of the speaker."""
        with self._lock:
            protocol = self._protocol
        if protocol is None:
            return False
        pending = getattr(protocol, "playback_pending", None)
        if not callable(pending):
            return bool(getattr(protocol, "speaking", False))
        try:
            return bool(pending())
        except Exception:
            return False

    def _live(self) -> bool:
        with self._lock:
            protocol = self._protocol
        return protocol is not None and not getattr(protocol, "close", True)

    def _quiet(self, now: datetime) -> bool:
        spec = self.cfg.bot_bridge.nudge_quiet_hours
        try:
            zone = ZoneInfo(self.cfg.device.timezone or "Europe/London")
        except Exception:
            zone = ZoneInfo("UTC")
        moment = now if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)
        return in_nudge_quiet_hours(spec, moment.astimezone(zone))

    def _say(self, text: str) -> bool:
        if self._speaker is not None:
            try:
                self._speaker(text)
            except Exception:
                return False
            return True
        try:
            from gva.runtime.tts import PiperTts

            pcm, rate = PiperTts(self.cfg).synthesise(text)
        except Exception:
            return False
        if not pcm or self.sink is None:
            return False
        from gva.runtime.audio import play_pcm

        with self._lock:
            protocol = self._protocol
        try:
            play_pcm(self.sink, pcm, rate, protocol)
        except Exception:
            return False
        return True

    def _chime(self) -> None:
        sink = self.sink
        if sink is None:
            return
        from gva.runtime.audio import tone

        self.last_chime = "soft"
        rate = int(getattr(sink, "sample_rate", 0) or 24000)
        pcm = tone(50, 523, amp=1800, sample_rate=rate) + tone(70, 784, amp=1800, sample_rate=rate)
        from gva.runtime.audio import play_pcm

        with self._lock:
            protocol = self._protocol
        try:
            play_pcm(sink, pcm, rate, protocol)
        except Exception:
            return

    def _request_follow_up(self) -> None:
        self.follow_ups += 1
        hook = self.on_follow_up
        if hook is None:
            return
        try:
            hook()
        except Exception:
            return

    def _emit(self, type_: str) -> None:
        try:
            self.events.emit(type_)
        except Exception:
            return

    def _clock(self, now: datetime | None) -> datetime:
        if now is not None:
            if now.tzinfo is None:
                return now.replace(tzinfo=timezone.utc)
            return now.astimezone(timezone.utc)
        if self._now_fn is not None:
            return self._clock(self._now_fn())
        return datetime.now(timezone.utc)

    def _client(self):
        if self._http is not None:
            return self._http
        if self._owned is None:
            import httpx

            self._owned = httpx.Client(timeout=httpx.Timeout(5.0, connect=5.0))
        return self._owned

    def _path(self) -> Path:
        return Path(self.cfg.telemetry.db_path).with_name("heads-up.json")

    def _load_spoken(self) -> None:
        try:
            data = json.loads(self._path().read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        rows = data.get("spoken") if isinstance(data, dict) else None
        if not isinstance(rows, list):
            return
        for row in rows:
            if not isinstance(row, dict):
                continue
            title = str(row.get("title") or "")
            due = str(row.get("due") or "")
            if title and due:
                self._spoken.add(meeting_key(title, due))

    def _save_spoken(self) -> None:
        path = self._path()
        rows = []
        for key in sorted(self._spoken):
            title, _, due = key.partition("\n")
            rows.append({"title": title, "due": due})
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"spoken": rows}), encoding="utf-8")
            tmp.replace(path)
        except OSError:
            return

