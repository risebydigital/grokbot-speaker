"""Tail the JSONL log into SQLite. Inserts are idempotent on event_id."""

from __future__ import annotations

import json
from pathlib import Path

from gva.telemetry.store import Store


def ingest_logs(log_dir: Path, store: Store) -> int:
    added = 0
    for path in sorted(Path(log_dir).glob("events-*.jsonl")):
        offset = store.ingest_offset(path.name)
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(offset)
            while True:
                start = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.endswith("\n"):
                    handle.seek(start)
                    break
                text = line.strip()
                if not text:
                    continue
                try:
                    event = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if "event_id" not in event or "type" not in event:
                    continue
                if store.insert_event(event):
                    added += 1
            store.set_ingest_offset(path.name, handle.tell())
    return added
