"""Score golden utterances against fixtures, or against a live model when asked."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from gva.paths import project_root


@dataclass
class EvalReport:
    cases: int
    passed: int
    accuracy: float
    failures: list[dict[str, Any]] = field(default_factory=list)
    model: str = "fixture"
    prompt_version: str = "v3"

    @property
    def ok(self) -> bool:
        return self.passed == self.cases and self.cases > 0


def golden_path() -> Path:
    return project_root() / "tests" / "golden" / "core.yaml"


def load_cases(path: Path | None = None) -> list[dict[str, Any]]:
    target = path or golden_path()
    data = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("golden file must be a list of cases")
    return data


def score_case(case: dict[str, Any], actual: dict[str, Any]) -> bool:
    expect = case.get("expect") or {}
    if expect.get("refuses"):
        text = str(actual.get("text") or "").casefold()
        refused = bool(actual.get("refused")) or any(
            phrase in text for phrase in ("can't", "cannot", "won't", "will not", "i'm not able")
        )
        return refused and not actual.get("tool")
    if expect.get("no_tool"):
        return not actual.get("tool")
    if "tool" in expect and actual.get("tool") != expect["tool"]:
        return False
    for key, value in (expect.get("args") or {}).items():
        if (actual.get("args") or {}).get(key) != value:
            return False
    if "text_contains" in expect:
        text = str(actual.get("text") or "").casefold()
        if str(expect["text_contains"]).casefold() not in text:
            return False
    return True


def run_fixtures(cases: list[dict[str, Any]] | None = None) -> EvalReport:
    rows = cases if cases is not None else load_cases()
    failures = []
    passed = 0
    for case in rows:
        fixture = case.get("fixture")
        if not isinstance(fixture, dict):
            failures.append(
                {
                    "case_id": case.get("id"),
                    "utterance": case.get("utterance"),
                    "expected": case.get("expect"),
                    "actual": {"error": "missing fixture"},
                }
            )
            continue
        if score_case(case, fixture):
            passed += 1
        else:
            failures.append(
                {
                    "case_id": case.get("id"),
                    "utterance": case.get("utterance"),
                    "expected": case.get("expect"),
                    "actual": fixture,
                }
            )
    total = len(rows)
    accuracy = round(passed / total, 4) if total else 0.0
    return EvalReport(cases=total, passed=passed, accuracy=accuracy, failures=failures)
