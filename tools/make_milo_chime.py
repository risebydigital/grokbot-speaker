#!/usr/bin/env python3
"""Write the canonical chime to assets/sounds/milo-chime.wav.

The voice service does not read that file. It synthesises the same buffer
from gva.runtime.chime and resamples it for the speaker.
"""

from __future__ import annotations

import wave
from pathlib import Path

from gva.runtime.chime import CANONICAL_RATE, canonical_pcm

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "assets" / "sounds" / "milo-chime.wav"


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    pcm = canonical_pcm()
    with wave.open(str(DEST), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(CANONICAL_RATE)
        handle.writeframes(pcm)
    print(DEST)


if __name__ == "__main__":
    main()
