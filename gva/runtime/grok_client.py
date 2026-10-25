"""xAI Chat Completions client. Streaming, usage, retries, and a small circuit breaker."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from typing import Any, Callable

import httpx

from gva.config import Config, env_value

TICKS_PER_USD = 10_000_000_000


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class LlmResult:
    text: str
    tool_calls: list[ToolCall]
    usage: dict[str, Any] | None
    usage_missing: bool
    cost_usd: float | None
    cost_estimated: bool
    status: str
    http_status: int | None
    xai_response_id: str | None
    system_fingerprint: str | None
    finish_reason: str | None
    ttft_ms: int | None
    duration_ms: int
    retry_count: int
    model: str
    error_category: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    retryable: bool = False


def flatten_usage(raw: dict[str, Any] | None) -> dict[str, Any] | None:
    if not raw:
        return None
    prompt_details = raw.get("prompt_tokens_details") or {}
    completion_details = raw.get("completion_tokens_details") or {}
    return {
        "prompt_tokens": int(raw.get("prompt_tokens") or 0),
        "completion_tokens": int(raw.get("completion_tokens") or 0),
        "total_tokens": int(raw.get("total_tokens") or 0),
        "cached_tokens": int(prompt_details.get("cached_tokens") or raw.get("cached_tokens") or 0),
        "reasoning_tokens": int(completion_details.get("reasoning_tokens") or raw.get("reasoning_tokens") or 0),
        "cost_in_usd_ticks": raw.get("cost_in_usd_ticks"),
    }


def cost_from_usage(usage: dict[str, Any] | None, prices) -> tuple[float | None, bool]:
    if not usage:
        return None, False
    ticks = usage.get("cost_in_usd_ticks")
    if ticks is not None:
        return round(int(ticks) / TICKS_PER_USD, 6), False
    if prices is None or prices.input is None or prices.output is None:
        return None, False
    prompt = int(usage.get("prompt_tokens") or 0)
    cached = int(usage.get("cached_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    uncached = max(0, prompt - cached)
    cached_price = prices.cached_input if prices.cached_input is not None else prices.input
    estimate = (uncached * prices.input + cached * cached_price + completion * prices.output) / 1_000_000
    return round(estimate, 6), True


class CircuitBreaker:
    def __init__(self, failures: int, window_s: float, open_s: float):
        self.failures = failures
        self.window_s = window_s
        self.open_s = open_s
        self._times: list[float] = []
        self._open_until = 0.0

    def allow(self, now: float | None = None) -> bool:
        moment = time.monotonic() if now is None else now
        return moment >= self._open_until

    def record_success(self) -> None:
        self._times.clear()
        self._open_until = 0.0

    def record_failure(self, now: float | None = None) -> None:
        moment = time.monotonic() if now is None else now
        self._times = [stamp for stamp in self._times if moment - stamp <= self.window_s]
        self._times.append(moment)
        if len(self._times) >= self.failures:
            self._open_until = moment + self.open_s


def _status_error(status: int) -> tuple[str, str, bool]:
    if status == 429:
        return "llm_rate_limit", "llm.rate_limited", True
    if status in {500, 502, 503, 504}:
        return "llm_server", "llm.server_error", True
    if status == 401:
        return "llm_client", "llm.unauthorised", False
    if status == 403:
        return "llm_client", "llm.forbidden", False
    if status == 404:
        return "llm_client", "llm.model_not_found", False
    if status in {400, 422}:
        return "llm_client", "llm.bad_request", False
    if status >= 500:
        return "llm_server", "llm.server_error", True
    return "llm_client", "llm.bad_request", False


class GrokClient:
    def __init__(self, cfg: Config, http: Any | None = None):
        self.cfg = cfg
        self.http = http or httpx.Client(
            timeout=httpx.Timeout(cfg.llm.timeouts.read_s, connect=cfg.llm.timeouts.connect_s),
            headers={"User-Agent": cfg.llm.user_agent},
        )
        self.breaker = CircuitBreaker(
            cfg.llm.circuit_breaker.failures,
            cfg.llm.circuit_breaker.window_s,
            cfg.llm.circuit_breaker.open_s,
        )

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        *,
        model: str | None = None,
        cancel: Callable[[], bool] | None = None,
        on_delta: Callable[[str], None] | None = None,
        deadline: float | None = None,
    ) -> LlmResult:
        chosen = model or self.cfg.llm.model
        if not self.breaker.allow():
            return self._failed(chosen, "llm_server", "llm.circuit_open", "The language model is paused after repeated errors.", False, None, 0, 0)
        attempt = 0
        max_attempts = self.cfg.llm.retry.max_attempts
        while True:
            result = self._once(messages, tools, chosen, cancel, on_delta)
            result.retry_count = attempt
            if result.status == "ok" or result.status == "cancelled":
                if result.status == "ok":
                    self.breaker.record_success()
                return result
            self.breaker.record_failure()
            retryable = result.retryable and attempt + 1 < max_attempts
            # Voice turns retry a 5xx only once. 429 may use the configured attempt budget.
            if result.error_code == "llm.server_error" and attempt >= 1:
                retryable = False
            if result.error_code == "llm.timeout" and attempt >= 1:
                retryable = False
            if not retryable:
                return result
            wait = random.uniform(0, min(self.cfg.llm.retry.cap_s, self.cfg.llm.retry.base_s * (2**attempt)))
            if deadline is not None and time.monotonic() + wait > deadline:
                return result
            time.sleep(wait)
            attempt += 1

    def _once(self, messages, tools, model, cancel, on_delta) -> LlmResult:
        key = env_value("XAI_API_KEY")
        if not key:
            return self._failed(model, "llm_client", "llm.unauthorised", "XAI_API_KEY is not set.", False, None, 0, 0)
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_completion_tokens": self.cfg.llm.max_completion_tokens,
            "temperature": self.cfg.llm.temperature,
            "parallel_tool_calls": self.cfg.llm.parallel_tool_calls,
            "prompt_cache_key": f"gva-{self.cfg.device.id}",
            "safety_identifier": _safety_id(self.cfg.device.id),
        }
        if self.cfg.llm.reasoning_effort:
            body["reasoning_effort"] = self.cfg.llm.reasoning_effort
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": self.cfg.llm.user_agent,
        }
        started = time.monotonic()
        ttft = None
        try:
            response = self.http.stream(
                "POST",
                self.cfg.llm.base_url.rstrip("/") + "/chat/completions",
                headers=headers,
                json=body,
                timeout=self.cfg.llm.timeouts.read_s,
            )
        except httpx.TimeoutException:
            return self._failed(model, "llm_server", "llm.timeout", "The language model timed out.", True, None, 0, _ms(started))
        except httpx.HTTPError as exc:
            return self._failed(model, "network", "net.offline", _safe_message(exc), True, None, 0, _ms(started))
        try:
            with response as stream:
                status = getattr(stream, "status_code", 200)
                if status >= 400:
                    category, code, retryable = _status_error(status)
                    return self._failed(model, category, code, f"HTTP {status}", retryable, status, 0, _ms(started))
                text_parts: list[str] = []
                tools_acc: dict[int, dict[str, str]] = {}
                usage = None
                response_id = None
                fingerprint = None
                finish = None
                last_chunk = time.monotonic()
                for line in stream.iter_lines():
                    if cancel and cancel():
                        return LlmResult(
                            text="".join(text_parts),
                            tool_calls=_tools(tools_acc),
                            usage=flatten_usage(usage),
                            usage_missing=usage is None,
                            cost_usd=None,
                            cost_estimated=False,
                            status="cancelled",
                            http_status=status,
                            xai_response_id=response_id,
                            system_fingerprint=fingerprint,
                            finish_reason=finish,
                            ttft_ms=ttft,
                            duration_ms=_ms(started),
                            retry_count=0,
                            model=model,
                        )
                    now = time.monotonic()
                    if now - last_chunk > self.cfg.llm.timeouts.stream_idle_s and text_parts:
                        return self._failed(model, "llm_server", "llm.timeout", "The reply stalled.", True, status, 0, _ms(started))
                    if not line:
                        continue
                    last_chunk = now
                    payload = line[5:].strip() if line.startswith("data:") else line.strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    response_id = chunk.get("id") or response_id
                    fingerprint = chunk.get("system_fingerprint") or fingerprint
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    finish = choice.get("finish_reason") or finish
                    delta = choice.get("delta") or {}
                    content = delta.get("content")
                    tool_delta = delta.get("tool_calls")
                    if ttft is None and (content or tool_delta):
                        ttft = _ms(started)
                    if content:
                        text_parts.append(content)
                        if on_delta and not tool_delta:
                            on_delta(content)
                    for call in tool_delta or []:
                        index = int(call.get("index") or 0)
                        slot = tools_acc.setdefault(index, {"id": "", "name": "", "arguments": ""})
                        if call.get("id"):
                            slot["id"] = call["id"]
                        function = call.get("function") or {}
                        if function.get("name"):
                            slot["name"] += function["name"]
                        if function.get("arguments"):
                            slot["arguments"] += function["arguments"]
        except httpx.HTTPError as exc:
            return self._failed(model, "network", "net.offline", _safe_message(exc), True, None, 0, _ms(started))
        flat = flatten_usage(usage)
        cost, estimated = cost_from_usage(flat, self.cfg.budget.prices_per_mtok_usd)
        return LlmResult(
            text="".join(text_parts),
            tool_calls=_tools(tools_acc),
            usage=flat,
            usage_missing=flat is None,
            cost_usd=cost,
            cost_estimated=estimated,
            status="ok",
            http_status=200,
            xai_response_id=response_id,
            system_fingerprint=fingerprint,
            finish_reason=finish,
            ttft_ms=ttft,
            duration_ms=_ms(started),
            retry_count=0,
            model=model,
        )

    def _failed(self, model, category, code, message, retryable, http_status, retry_count, duration) -> LlmResult:
        return LlmResult(
            text="",
            tool_calls=[],
            usage=None,
            usage_missing=True,
            cost_usd=None,
            cost_estimated=False,
            status="error",
            http_status=http_status,
            xai_response_id=None,
            system_fingerprint=None,
            finish_reason=None,
            ttft_ms=None,
            duration_ms=duration,
            retry_count=retry_count,
            model=model,
            error_category=category,
            error_code=code,
            error_message=message[:300],
            retryable=retryable,
        )

    def list_models(self) -> tuple[bool, str]:
        key = env_value("XAI_API_KEY")
        if not key:
            return False, "XAI_API_KEY is not set"
        try:
            response = self.http.get(
                self.cfg.llm.base_url.rstrip("/") + "/models",
                headers={"Authorization": f"Bearer {key}", "User-Agent": self.cfg.llm.user_agent},
                timeout=self.cfg.llm.timeouts.connect_s + 2,
            )
        except httpx.HTTPError as exc:
            return False, _safe_message(exc)
        if response.status_code == 200:
            return True, "reachable"
        return False, f"HTTP {response.status_code}"


def _tools(acc: dict[int, dict[str, str]]) -> list[ToolCall]:
    calls = []
    for index in sorted(acc):
        slot = acc[index]
        if slot["name"]:
            calls.append(ToolCall(id=slot["id"] or f"call_{index}", name=slot["name"], arguments=slot["arguments"] or "{}"))
    return calls


def _ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def _safe_message(exc: BaseException) -> str:
    text = str(exc)
    if "Bearer" in text or "sk-" in text or "xai-" in text:
        return exc.__class__.__name__
    return text[:300]


def _safety_id(device_id: str) -> str:
    import hashlib
    import hmac

    pepper = env_value("GVA_HASH_PEPPER") or device_id
    return hmac.new(pepper.encode(), device_id.encode(), hashlib.sha256).hexdigest()[:32]


# Kept so tests can build a result without the network.
def scripted_result(text: str = "", tools: list[ToolCall] | None = None, **kwargs) -> LlmResult:
    usage = kwargs.get("usage")
    return LlmResult(
        text=text,
        tool_calls=tools or [],
        usage=usage,
        usage_missing=usage is None,
        cost_usd=kwargs.get("cost_usd", 0.0),
        cost_estimated=kwargs.get("cost_estimated", False),
        status=kwargs.get("status", "ok"),
        http_status=kwargs.get("http_status", 200),
        xai_response_id=kwargs.get("xai_response_id"),
        system_fingerprint=kwargs.get("system_fingerprint"),
        finish_reason=kwargs.get("finish_reason", "stop"),
        ttft_ms=kwargs.get("ttft_ms", 10),
        duration_ms=kwargs.get("duration_ms", 20),
        retry_count=kwargs.get("retry_count", 0),
        model=kwargs.get("model", "grok-4.7"),
        error_category=kwargs.get("error_category"),
        error_code=kwargs.get("error_code"),
        error_message=kwargs.get("error_message"),
        retryable=kwargs.get("retryable", False),
    )
