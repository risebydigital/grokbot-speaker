"""Delete old local events. Supabase retention is the dashboard's job."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gva.config import Config
from gva.telemetry.store import Store


def enforce(cfg: Config, store: Store) -> int:
    cutoff = datetime.now(timezone.utc) - timedelta(days=cfg.retention.events_days)
    stamp = cutoff.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    removed = store.delete_events_before(stamp)
    replay_cutoff = datetime.now(timezone.utc) - timedelta(hours=max(cfg.telemetry.replay_cache_hours, 1))
    store.purge_replay(replay_cutoff.isoformat())
    store.vacuum()
    return removed
