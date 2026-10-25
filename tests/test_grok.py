from gva.runtime.grok_client import GrokClient, cost_from_usage, flatten_usage
from tests.fakes import FakeHttp, test_config


def _sse(content: str, ticks: int) -> list[str]:
    usage = {
        "prompt_tokens": 4,
        "completion_tokens": 2,
        "total_tokens": 6,
        "cost_in_usd_ticks": ticks,
    }
    return [
        'data: {"id":"resp_1","choices":[{"delta":{"content":"%s"}}]}' % content,
        'data: {"choices":[],"usage":%s}' % __import__("json").dumps(usage),
        "data: [DONE]",
    ]


def test_cost_ticks_win_over_the_price_table():
    usage = flatten_usage({"prompt_tokens": 1000, "completion_tokens": 0, "total_tokens": 1000, "cost_in_usd_ticks": 20_000_000_000})
    cost, estimated = cost_from_usage(usage, prices=None)
    assert cost == 2.0
    assert estimated is False


def test_retry_429_then_success(tmp_path, monkeypatch):
    monkeypatch.setattr("gva.runtime.grok_client.time.sleep", lambda _seconds: None)
    cfg = test_config(tmp_path, monkeypatch)
    http = FakeHttp([(429, []), (200, _sse("Hello there.", 10_000_000_000))])
    client = GrokClient(cfg, http=http)
    result = client.complete([{"role": "user", "content": "hi"}], None)
    assert http.calls == 2
    assert result.status == "ok"
    assert result.text == "Hello there."
    assert result.cost_usd == 1.0
    assert result.cost_estimated is False
    assert result.retry_count == 1


def test_401_is_not_retried(tmp_path, monkeypatch):
    cfg = test_config(tmp_path, monkeypatch)
    http = FakeHttp([(401, [])])
    client = GrokClient(cfg, http=http)
    result = client.complete([{"role": "user", "content": "hi"}], None)
    assert http.calls == 1
    assert result.error_code == "llm.unauthorised"
    assert "Bearer" not in (result.error_message or "")


def test_circuit_opens(tmp_path, monkeypatch):
    cfg = test_config(tmp_path, monkeypatch)
    cfg.llm.retry.max_attempts = 1
    cfg.llm.circuit_breaker.failures = 2
    cfg.llm.circuit_breaker.open_s = 60
    http = FakeHttp([(500, []), (500, [])])
    client = GrokClient(cfg, http=http)
    first = client.complete([{"role": "user", "content": "hi"}], None)
    second = client.complete([{"role": "user", "content": "hi"}], None)
    third = client.complete([{"role": "user", "content": "hi"}], None)
    assert first.error_code == "llm.server_error"
    assert second.error_code == "llm.server_error"
    assert third.error_code == "llm.circuit_open"
    assert http.calls == 2
