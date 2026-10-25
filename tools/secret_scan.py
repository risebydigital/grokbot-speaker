#!/usr/bin/env python3
"""Small secret scan for CI. Same idea as gitleaks, without a network download.

Placeholders in the example env files are allowed. A real key must not be committed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PATTERNS = [
    ("xai-key", re.compile(r"xai-[A-Za-z0-9_\-]{16,}")),
    ("jwt", re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}")),
    ("slack-webhook", re.compile(r"https://hooks\.slack\.com/services/T[A-Z0-9]+/B[A-Z0-9]+/[A-Za-z0-9]+")),
    ("aws-access-key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("github-token", re.compile(r"ghp_[A-Za-z0-9]{20,}")),
    ("private-key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
]

SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "models",
    ".gva",
}
SKIP_FILES = {".env.example", "gva.env.example"}
ALLOW_LINE = re.compile(r"<YOUR_|<SUPABASE_|<64_HEX_|<RANDOM_|<SLACK_|<HEALTHCHECK_|<project-ref>")


def main() -> int:
    hits = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.name in SKIP_FILES or path.suffix in {".png", ".wav", ".onnx", ".mp3", ".pyc"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if ALLOW_LINE.search(line):
                continue
            for name, pattern in PATTERNS:
                if pattern.search(line):
                    hits.append(f"{path.relative_to(ROOT)}:{lineno}: {name}")
    if hits:
        print("secret scan failed:", file=sys.stderr)
        for hit in hits:
            print(hit, file=sys.stderr)
        return 1
    print("secret scan: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
