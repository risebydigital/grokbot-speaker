"""Read the Appendix B migrations so the sync payload cannot invent columns."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from gva.paths import project_root

_TYPE = r"(?:text|int|bigint|boolean|numeric|jsonb|uuid|date|timestamptz|double precision)"
_COLUMN = re.compile(rf"\b([a-z_][a-z0-9_]*)\s+{_TYPE}\b", re.I)
_TABLE = re.compile(r"create table public\.(\w+)\s*\((.*?)\);", re.I | re.S)

# Primary keys PostgREST needs for an idempotent upsert.
ON_CONFLICT = {
    "devices": "id",
    "turns": "id",
    "llm_calls": "id",
    "tool_calls": "id",
    "errors": "id",
    "heartbeats": "device_id,at",
    "rollup_hourly": "bucket,device_id",
    "rollup_daily": "bucket,device_id",
    "turn_text": "turn_id",
    "config_current": "device_id",
    "api_key_status": "env",
    "eval_runs": "id",
    "eval_failures": "run_id,case_id",
    "alerts": "id",
}

# Tables the device may insert or update. audit_log is trigger-owned. profiles is the dashboard's.
DEVICE_TABLES = set(ON_CONFLICT)

# config_changes is patched, never inserted (requested_by is a profile foreign key).
PATCH_TABLES = {"config_changes"}


def parse_create_tables(sql: str) -> dict[str, set[str]]:
    tables: dict[str, set[str]] = {}
    for match in _TABLE.finditer(sql):
        name, body = match.group(1), match.group(2)
        body = re.sub(r"--.*?$", "", body, flags=re.M)
        body = re.sub(r"\([^)]*\)", " ", body)
        tables[name] = {col.group(1) for col in _COLUMN.finditer(body)}
    return tables


@lru_cache(maxsize=1)
def migration_columns() -> dict[str, set[str]]:
    root = project_root() / "supabase" / "migrations"
    found: dict[str, set[str]] = {}
    if not root.is_dir():
        return found
    for path in sorted(root.glob("*.sql")):
        found.update(parse_create_tables(path.read_text(encoding="utf-8")))
    return found


def fit(table: str, row: dict) -> dict:
    """Drop unknown keys and JSON nulls. Zero, false and empty lists are kept."""
    allowed = migration_columns().get(table, set())
    fitted = {}
    for key, value in row.items():
        if key not in allowed or value is None:
            continue
        fitted[key] = value
    return fitted


def migrations_dir() -> Path:
    return project_root() / "supabase" / "migrations"
