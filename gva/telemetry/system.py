"""Host metrics that do not need extra packages. Reads /proc and sysfs."""

from __future__ import annotations

import shutil
import time
from pathlib import Path


_LAST_CPU: tuple[int, int] | None = None
_LAST_CPU_AT: float = 0.0


def _cpu_percent() -> float | None:
    global _LAST_CPU, _LAST_CPU_AT
    try:
        parts = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()
        numbers = [int(part) for part in parts[1:]]
    except (OSError, ValueError, IndexError):
        return None
    idle = numbers[3] + (numbers[4] if len(numbers) > 4 else 0)
    total = sum(numbers)
    now = time.monotonic()
    if _LAST_CPU is None or now - _LAST_CPU_AT < 0.2:
        _LAST_CPU = (total, idle)
        _LAST_CPU_AT = now
        return 0.0
    prev_total, prev_idle = _LAST_CPU
    _LAST_CPU = (total, idle)
    _LAST_CPU_AT = now
    delta_total = total - prev_total
    delta_idle = idle - prev_idle
    if delta_total <= 0:
        return 0.0
    return round(max(0.0, min(100.0, (1 - delta_idle / delta_total) * 100)), 2)


def _mem_percent() -> float | None:
    try:
        info = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, _, rest = line.partition(":")
            info[key] = int(rest.strip().split()[0])
        total = info.get("MemTotal") or 0
        available = info.get("MemAvailable") or 0
        if total <= 0:
            return None
        return round(max(0.0, min(100.0, (1 - available / total) * 100)), 2)
    except (OSError, ValueError, IndexError):
        return None


def _temp_c() -> float | None:
    for path in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            milli = int(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        if milli > 0:
            return round(milli / 1000.0, 2)
    return None


def _throttled() -> bool | None:
    path = Path("/usr/bin/vcgencmd")
    if not path.exists():
        return None
    import subprocess

    try:
        out = subprocess.check_output([str(path), "get_throttled"], text=True, timeout=2)
    except (OSError, subprocess.SubprocessError):
        return None
    # Output looks like throttled=0x0. Any non-zero bit means the Pi has throttled or under-volted.
    raw = out.strip().split("=", 1)[-1]
    try:
        return int(raw, 16) != 0
    except ValueError:
        return None


def snapshot(log_dir: Path) -> dict[str, float | bool | None]:
    usage = shutil.disk_usage(log_dir if log_dir.exists() else Path("/"))
    disk = round(usage.used / usage.total * 100, 2) if usage.total else None
    cpu = _cpu_percent()
    mem = _mem_percent()
    return {
        "cpu_pct": 0.0 if cpu is None else cpu,
        "mem_pct": 0.0 if mem is None else mem,
        "disk_pct": 0.0 if disk is None else disk,
        "temp_c": _temp_c(),
        "throttled": _throttled(),
    }
