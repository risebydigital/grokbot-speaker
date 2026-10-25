"""Locate the repository root so prompts, schema and example config travel with the code."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    env = os.environ.get("GVA_ROOT")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "gva").is_dir():
            return parent
    return Path("/opt/gva/app")


def schema_path() -> Path:
    root = project_root()
    candidate = root / "schemas" / "event-1.0.0.json"
    if candidate.is_file():
        return candidate
    packaged = Path(__file__).resolve().parent / "telemetry" / "event-1.0.0.json"
    return packaged


def prompt_path(version: str) -> Path:
    root = project_root() / "prompts" / "system" / f"{version}.txt"
    if root.is_file():
        return root
    return Path(__file__).resolve().parent / "prompts" / f"{version}.txt"


def tools_dir() -> Path:
    return project_root() / "config" / "tools"
