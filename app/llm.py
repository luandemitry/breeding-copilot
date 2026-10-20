"""The only place the app creates chat models.

* Provider and model are configurable with env vars (``LLM_PROVIDER`` / ``LLM_MODEL``) through
  LangChain's ``init_chat_model`` so switching Gemini -> Groq -> Ollama needs no code change.
* :class:`ThrottledLLM` adds a requests-per-minute throttle, exponential backoff on 429s and an
  optional on-disk response cache (key = hash of provider, model and messages) so that re-running
  the evaluation does not spend free-tier quota.
* :class:`FakeLLM` is a scripted stand-in used by unit tests: tests never touch the network.

Only free-tier models are intended (default: Gemini Flash via Google AI Studio).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.config import ROOT

DEFAULT_PROVIDER = "google_genai"
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_REQUESTS_PER_MINUTE = 10
DEFAULT_MAX_RETRIES = 5
DEFAULT_CACHE_DIR = ROOT / ".cache" / "llm"
BACKOFF_BASE_S = 2.0
BACKOFF_MAX_S = 60.0

class LLMConfigError(RuntimeError):
    """Missing or invalid LLM configuration (e.g. no API key)."""


class RateLimitExhausted(RuntimeError):
    """The provider kept answering 429 after all retries."""


@dataclass
class LLMCallRecord:
    """What one call cost. Observability subscribes to these."""

    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float  # duration of the provider request; for cache hits, of the original request
    cached: bool = False
    retries: int = 0
    waited_s: float = 0.0  # time spent in self-imposed throttle/backoff sleeps (not part of latency_s)


def is_rate_limit_error(exc: BaseException) -> bool:
    """Best-effort detection of a 429 / quota error across providers."""
    for attr in ("status_code", "code", "http_status"):
        if getattr(exc, attr, None) == 429:
            return True
    text = f"{type(exc).__name__} {exc}".lower()
    return any(s in text for s in ("429", "resourceexhausted", "ratelimit", "rate limit", "quota"))


def _as_messages(messages: Any) -> list[BaseMessage]:
    if isinstance(messages, str):
        return [HumanMessage(content=messages)]
    return list(messages)


def _cache_key(provider: str, model: str, messages: list[BaseMessage]) -> str:
    payload = json.dumps([provider, model, [(m.type, m.content) for m in messages]], default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


class BaseLLMClient:
    """Common listener plumbing: ``invoke(messages) -> AIMessage`` and per-call records."""

    provider = "unknown"
    model = "unknown"

    def __init__(self) -> None:
        self._listeners: list[Callable[[LLMCallRecord], None]] = []

    def add_listener(self, listener: Callable[[LLMCallRecord], None]) -> None:
        self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[LLMCallRecord], None]) -> None:
        self._listeners.remove(listener)

    def _emit(
        self,
        message: AIMessage,
        latency_s: float,
        *,
        cached: bool = False,
        retries: int = 0,
        waited_s: float = 0.0,
    ) -> None:
        usage = message.usage_metadata or {}
        record = LLMCallRecord(
            provider=self.provider,
            model=self.model,
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            latency_s=latency_s,
            cached=cached,
            retries=retries,
            waited_s=waited_s,
        )
        for listener in list(self._listeners):
            listener(record)

    def invoke(self, messages: Any) -> AIMessage:  # pragma: no cover - interface
        raise NotImplementedError


class ThrottledLLM(BaseLLMClient):
    """Wrap a LangChain chat model with throttle, 429 backoff and an optional disk cache."""

    def __init__(
        self,
        model: Any,
        *,
        provider: str,
        model_name: str,
        requests_per_minute: float = DEFAULT_REQUESTS_PER_MINUTE,
        max_retries: int = DEFAULT_MAX_RETRIES,
        cache_dir: Path | str | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        super().__init__()
        self._model = model
        self.provider, self.model = provider, model_name
        self._min_interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
        self._max_retries = max_retries
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._sleep, self._clock, self._jitter = sleep, clock, jitter
        self._last_request: float | None = None
        self._lock = threading.Lock()

    def invoke(self, messages: Any) -> AIMessage:
        msgs = _as_messages(messages)
        cached = self._read_cache(msgs)
        if cached is not None:
            message, original_latency = cached
            self._emit(message, original_latency, cached=True)
            return message

        retries, waited = 0, 0.0
        while True:
            waited += self._throttle()
            started = self._clock()
            try:
                response = self._model.invoke(msgs)
                latency = self._clock() - started
                break
            except Exception as exc:  # noqa: BLE001 - provider exceptions vary
                if not is_rate_limit_error(exc):
                    raise
                if retries >= self._max_retries:
                    raise RateLimitExhausted(f"still rate limited after {retries} retries: {exc}") from exc
                delay = min(BACKOFF_MAX_S, BACKOFF_BASE_S * 2**retries) + self._jitter()
                self._sleep(delay)
                waited += delay
                retries += 1
        message = response if isinstance(response, AIMessage) else AIMessage(content=str(response))
        self._write_cache(msgs, message, latency)
        self._emit(message, latency, retries=retries, waited_s=waited)
        return message

    def _throttle(self) -> float:
        """Sleep if needed to respect the requests-per-minute limit; return the seconds waited."""
        with self._lock:
            now = self._clock()
            waited = 0.0
            if self._last_request is not None:
                waited = max(0.0, self._min_interval - (now - self._last_request))
                if waited > 0:
                    self._sleep(waited)
                    now = self._clock()
            self._last_request = now
            return waited

    def _cache_path(self, messages: list[BaseMessage]) -> Path | None:
        if self._cache_dir is None:
            return None
        return self._cache_dir / f"{_cache_key(self.provider, self.model, messages)}.json"

    def _read_cache(self, messages: list[BaseMessage]) -> tuple[AIMessage, float] | None:
        path = self._cache_path(messages)
        if path is None or not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        message = AIMessage(content=data["content"], usage_metadata=data.get("usage_metadata"))
        return message, float(data.get("latency_s", 0.0))

    def _write_cache(self, messages: list[BaseMessage], message: AIMessage, latency_s: float) -> None:
        path = self._cache_path(messages)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"content": message.content, "usage_metadata": message.usage_metadata, "latency_s": latency_s}
        path.write_text(json.dumps(payload), encoding="utf-8")


class FakeLLM(BaseLLMClient):
    """Scripted LLM for tests. ``responses`` are returned in order; a callable receives the messages."""

    provider = "fake"
    model = "fake"

    def __init__(self, responses: Sequence[str | AIMessage] | Callable[[list[BaseMessage]], str]) -> None:
        super().__init__()
        self._responses = responses
        self._index = 0
        self.prompts: list[list[BaseMessage]] = []

    def invoke(self, messages: Any) -> AIMessage:
        msgs = _as_messages(messages)
        self.prompts.append(msgs)
        if callable(self._responses):
            item: str | AIMessage = self._responses(msgs)
        else:
            if self._index >= len(self._responses):
                raise AssertionError("FakeLLM ran out of scripted responses")
            item = self._responses[self._index]
            self._index += 1
        if isinstance(item, AIMessage):
            message = item
        else:
            prompt_tokens = sum(len(str(m.content).split()) for m in msgs)
            message = AIMessage(
                content=item,
                usage_metadata={
                    "input_tokens": prompt_tokens,
                    "output_tokens": len(item.split()),
                    "total_tokens": prompt_tokens + len(item.split()),
                },
            )
        self._emit(message, 0.0)
        return message


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    *,
    cache: bool | None = None,
    requests_per_minute: float | None = None,
    max_retries: int | None = None,
) -> ThrottledLLM:
    """Build the configured chat model wrapped with throttle/backoff (and cache when ``cache``).

    ``cache=None`` falls back to ``LLM_CACHE=1`` in the environment; the eval runner passes ``True``.
    """
    load_dotenv(ROOT / ".env")
    provider = provider or os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER)
    model = model or os.getenv("LLM_MODEL", DEFAULT_MODEL)
    rpm = requests_per_minute
    if rpm is None:
        rpm = float(os.getenv("LLM_REQUESTS_PER_MINUTE", DEFAULT_REQUESTS_PER_MINUTE))
    retries = max_retries if max_retries is not None else int(os.getenv("LLM_MAX_RETRIES", DEFAULT_MAX_RETRIES))
    use_cache = cache if cache is not None else os.getenv("LLM_CACHE", "0") == "1"

    kwargs: dict[str, Any] = {"temperature": 0}
    if provider in ("google_genai", "google-genai"):
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise LLMConfigError("GEMINI_API_KEY is not set (get a free key at https://aistudio.google.com/apikey)")
        kwargs.update(api_key=api_key, max_retries=1)  # retries are handled by ThrottledLLM
    elif provider == "groq":
        if not os.getenv("GROQ_API_KEY"):
            raise LLMConfigError("GROQ_API_KEY is not set")
        kwargs.update(max_retries=1)
    elif provider == "ollama" and os.getenv("OLLAMA_BASE_URL"):
        kwargs["base_url"] = os.environ["OLLAMA_BASE_URL"]

    chat_model = init_chat_model(model, model_provider=provider, **kwargs)
    cache_dir = os.getenv("LLM_CACHE_DIR", DEFAULT_CACHE_DIR) if use_cache else None
    return ThrottledLLM(
        chat_model,
        provider=provider,
        model_name=model,
        requests_per_minute=rpm,
        max_retries=retries,
        cache_dir=cache_dir,
    )
