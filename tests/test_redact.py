from gva.telemetry.redact import CANARY_VALUES, contains_canary, prompt_hash, redact


def test_canaries_are_redacted():
    pepper = "pepper-" + ("p" * 32)
    samples = {
        "4111111111111111": "[CARD]",
        "07700900123": "[PHONE]",
        "canary@example.com": "[EMAIL]",
        "QQ123456C": "[NINO]",
        "9434765919": "[NHS]",
    }
    for raw, token in samples.items():
        cleaned = redact(raw, short_numbers=True).text
        assert token in cleaned
        assert not contains_canary(cleaned)
        assert raw in CANARY_VALUES or raw.replace(" ", "") in "".join(CANARY_VALUES)
    digest = prompt_hash(redact("hello", short_numbers=True).text, pepper)
    again = prompt_hash(redact("  Hello  ", short_numbers=True).text, pepper)
    assert digest == again
    assert digest.startswith("hmac-sha256:")
    assert prompt_hash("hello", "q" * 32) != digest


def test_short_numbers_and_sensitive_category():
    text = redact("the code is 42", short_numbers=True).text
    assert "42" not in text
    assert "[NUMBER]" in text
    health = redact("I need my prescription", categories={"health": ["prescription"]})
    assert health.sensitive
    assert health.category == "health"


def test_pepper_must_be_long():
    try:
        prompt_hash("hello", "short")
    except ValueError as exc:
        assert "32" in str(exc)
    else:
        raise AssertionError("short pepper was accepted")
