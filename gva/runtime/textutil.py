"""Speech-facing text: stop commands, sentence chunking, and a small spoken normaliser."""

from __future__ import annotations

import re

_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eighteen", "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def is_stop_phrase(text: str, phrases: list[str]) -> bool:
    folded = " ".join(text.casefold().strip().strip(".!?").split())
    if not folded:
        return False
    accepted = {phrase.casefold() for phrase in phrases}
    if folded in accepted:
        return True
    words = folded.split()
    if words and words[0] in accepted:
        return True
    if len(words) <= 3 and any(word in {"stop", "cancel"} for word in words):
        return True
    return False


def is_yes(text: str) -> bool:
    folded = " ".join(text.casefold().strip().strip(".!?").split())
    return folded in {"yes", "yeah", "yep", "please", "go ahead", "do it", "confirm"}


def is_no(text: str) -> bool:
    folded = " ".join(text.casefold().strip().strip(".!?").split())
    return folded in {"no", "nope", "don't", "do not", "cancel", "stop"}


def take_sentences(buffer: str, max_clause: int = 120) -> tuple[list[str], str]:
    """Release text up to a sentence boundary that is followed by whitespace."""
    ready: list[str] = []
    rest = buffer
    while True:
        match = re.search(r"[.!?]\s+", rest)
        if match:
            ready.append(rest[: match.end()].strip())
            rest = rest[match.end() :]
            continue
        if len(rest) > max_clause:
            comma = rest.rfind(",", 0, max_clause)
            cut = comma + 1 if comma > 20 else max_clause
            piece = rest[:cut].strip()
            if piece:
                ready.append(piece)
            rest = rest[cut:].lstrip()
            continue
        break
    return ready, rest


def flush_sentences(buffer: str) -> list[str]:
    text = buffer.strip()
    return [text] if text else []


def _small(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        ten, one = divmod(n, 10)
        return _TENS[ten] if one == 0 else f"{_TENS[ten]} {_ONES[one]}"
    return str(n)


def _int_words(n: int) -> str:
    if n < 0:
        return f"minus {_int_words(-n)}"
    if n < 100:
        return _small(n)
    if n < 1000:
        hundred, rest = divmod(n, 100)
        head = f"{_ONES[hundred]} hundred"
        return head if rest == 0 else f"{head} and {_small(rest)}"
    if n < 1000000:
        thousand, rest = divmod(n, 1000)
        head = f"{_int_words(thousand)} thousand"
        if rest == 0:
            return head
        if rest < 100:
            return f"{head} and {_small(rest)}"
        return f"{head} {_int_words(rest)}"
    return str(n)


def _clock(match: re.Match[str]) -> str:
    hour = int(match.group(1))
    minute = int(match.group(2))
    if minute == 0:
        return f"{_int_words(hour % 12 or 12)} o'clock"
    if minute == 15:
        return f"quarter past {_int_words(hour % 12 or 12)}"
    if minute == 30:
        return f"half past {_int_words(hour % 12 or 12)}"
    if minute == 45:
        return f"quarter to {_int_words((hour + 1) % 12 or 12)}"
    return f"{_int_words(hour % 12 or 12)} {_int_words(minute)}"


def _pounds(match: re.Match[str]) -> str:
    raw = match.group(1).replace(",", "")
    pounds = int(raw)
    words = _int_words(pounds)
    unit = "pound" if pounds == 1 else "pounds"
    return f"{words} {unit}"


def normalise_for_speech(text: str, pronunciation: dict[str, str] | None = None) -> str:
    """Make a reply easier for Piper to say. The text sent to Grok is not changed."""
    out = text.strip()
    out = re.sub(r"[*_`#>]", "", out)
    out = re.sub(r"https?://\S+", "a link", out)
    out = re.sub(r"£\s?([0-9][0-9,]*)", _pounds, out)
    out = re.sub(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", _clock, out)
    for source, spoken in (pronunciation or {}).items():
        out = re.sub(rf"\b{re.escape(source)}\b", spoken, out)
    out = re.sub(r"\s+", " ", out).strip()
    return out
