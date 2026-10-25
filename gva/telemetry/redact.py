"""Redaction and HMAC hashing. Raw prompt text is not what gets persisted."""

from __future__ import annotations

import hashlib
import hmac
import re
import unicodedata
from dataclasses import dataclass

_SECRET = re.compile(
    r"(?i)\b(?:xai-[A-Za-z0-9_\-]{8,}|sk-[A-Za-z0-9]{8,}|bearer\s+[A-Za-z0-9._\-]{8,})\b"
    r"|\b[0-9a-fA-F]{32,}\b"
    r"|eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"
)
_EMAIL = re.compile(r"\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b", re.I)
_PHONE = re.compile(r"(?<!\w)(?:\+44|0)(?:[\s\-()]*\d){9,12}(?!\d)")
_CARD_CANDIDATE = re.compile(r"(?<!\d)(?:\d[ \-]?){13,19}(?!\d)")
# Includes the brief's example form (QQ 12 34 56 C) as well as issued prefixes.
_NINO = re.compile(r"\b[A-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b", re.I)
_NHS_CANDIDATE = re.compile(r"(?<!\d)(?:\d[\s\-]?){10}(?!\d)")
_POSTCODE = re.compile(r"\b([A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2})\b", re.I)
_ADDRESS = re.compile(
    r"\b\d{1,4}\s+(?:[A-Za-z]+\s+){1,3}"
    r"(?:Street|Road|Avenue|Lane|Drive|Close|Way|Place|Crescent|Gardens|Terrace)\b",
    re.I,
)
_IP = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")
_URL_QUERY = re.compile(r"https?://\S+\?\S+", re.I)
_DOB = re.compile(
    r"(?i)\b(?:born on|date of birth(?: is)?|dob(?: is)?|birthday(?: is)?)\s+"
    r"(?:\d{1,2}\s+)?"
    r"(?:\d{1,2}(?:st|nd|rd|th)?\s+)?"
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)?"
    r"\s*\d{0,4}"
)
_LONG_DIGITS = re.compile(r"(?<!\d)\d{5,}(?!\d)")
_SHORT_DIGITS = re.compile(r"(?<!\d)\d{1,4}(?!\d)")
_NUMBER_WORD = re.compile(
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
    r"fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|"
    r"seventy|eighty|ninety|hundred|thousand)\b",
    re.I,
)

# Ofcom drama numbers and other fixtures used by the canary. Not real personal data.
CANARY_VALUES = (
    "4111111111111111",
    "07700900123",
    "canary@example.com",
    "QQ123456C",
    "9434765919",
)


@dataclass
class Redaction:
    text: str
    sensitive: bool
    category: str | None


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold().strip()
    return " ".join(text.split())


def prompt_hash(redacted_text: str, pepper: str) -> str:
    if not pepper or len(pepper) < 32:
        raise ValueError("GVA_HASH_PEPPER must be at least 32 characters")
    digest = hmac.new(pepper.encode(), normalise(redacted_text).encode(), hashlib.sha256).hexdigest()
    return f"hmac-sha256:{digest}"


def _luhn(number: str) -> bool:
    digits = [int(c) for c in number if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    alt = False
    for digit in reversed(digits):
        if alt:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
        alt = not alt
    return total % 10 == 0


def _nhs_ok(raw: str) -> bool:
    digits = "".join(c for c in raw if c.isdigit())
    if len(digits) != 10:
        return False
    total = sum(int(d) * (10 - i) for i, d in enumerate(digits[:9]))
    remainder = total % 11
    check = 11 - remainder
    if check == 11:
        check = 0
    if check == 10:
        return False
    return check == int(digits[9])


def _replace_cards(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        return "[CARD]" if _luhn(match.group(0)) else match.group(0)

    return _CARD_CANDIDATE.sub(repl, text)


def _replace_nhs(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        return "[NHS]" if _nhs_ok(match.group(0)) else match.group(0)

    return _NHS_CANDIDATE.sub(repl, text)


def _replace_names(text: str, names: list[str]) -> str:
    for name in names:
        cleaned = name.strip()
        if len(cleaned) < 2:
            continue
        text = re.sub(rf"\b{re.escape(cleaned)}\b", "[NAME]", text, flags=re.I)
    return text


def classify_sensitive(text: str, categories: dict[str, list[str]]) -> str | None:
    folded = normalise(text)
    for category, words in categories.items():
        for word in words:
            if word and normalise(word) in folded:
                return category
    return None


def redact(
    text: str,
    *,
    household_names: list[str] | None = None,
    short_numbers: bool = False,
    categories: dict[str, list[str]] | None = None,
) -> Redaction:
    """Apply the ordered redaction rules. Does not replace sensitive turns; the caller does."""
    category = classify_sensitive(text, categories or {})
    out = text
    out = _SECRET.sub("[SECRET]", out)
    out = _EMAIL.sub("[EMAIL]", out)
    out = _PHONE.sub("[PHONE]", out)
    out = _replace_cards(out)
    out = _NINO.sub("[NINO]", out)
    out = _replace_nhs(out)
    out = _POSTCODE.sub("[POSTCODE]", out)
    out = _ADDRESS.sub("[ADDRESS]", out)
    out = _IP.sub("[IP]", out)
    out = _URL_QUERY.sub("[URL]", out)
    out = _replace_names(out, household_names or [])
    out = _DOB.sub("[DOB]", out)
    out = _LONG_DIGITS.sub("[NUMBER]", out)
    if short_numbers:
        out = _NUMBER_WORD.sub("[NUMBER]", out)
        out = _SHORT_DIGITS.sub("[NUMBER]", out)
    out = re.sub(r"(?:\[NUMBER\]\s*){2,}", "[NUMBER] ", out).strip()
    return Redaction(text=out, sensitive=category is not None, category=category)


def redact_value(value, **kwargs):
    """Redact strings inside JSON-like tool arguments. Numbers are left as numbers."""
    if isinstance(value, str):
        return redact(value, **kwargs).text
    if isinstance(value, list):
        return [redact_value(item, **kwargs) for item in value]
    if isinstance(value, dict):
        return {key: redact_value(item, **kwargs) for key, item in value.items()}
    return value


def contains_canary(payload: str) -> list[str]:
    """Return canary fragments that survived into a persisted or synced payload."""
    folded = payload.replace(" ", "").replace("-", "")
    found = []
    for value in CANARY_VALUES:
        compact = value.replace(" ", "")
        if compact.lower() in folded.lower():
            found.append(value)
    return found
