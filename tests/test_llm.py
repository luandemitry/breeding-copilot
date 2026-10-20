import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app import llm as llm_module
from app.llm import (
    FakeLLM,
    LLMConfigError,
    RateLimitExhausted,
    ThrottledLLM,
    get_llm,
    is_rate_limit_error,
)


class RateLimitError(Exception):
    status_code = 429


class FlakyModel:
    """Fails with 429 ``failures`` times, then answers."""

    def __init__(self, failures=0, exc=None, answer="ok"):
        self.failures, self.calls, self.exc, self.answer = failures, 0, exc or RateLimitError("429 quota"), answer

    def invoke(self, messages):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.exc
        return AIMessage(content=self.answer, usage_metadata={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5})


class Clock:
    """Deterministic clock: sleeping advances time instantly."""

    def __init__(self):
        self.now, self.sleeps = 1000.0, []

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def make(model, clock, **kw):
    return ThrottledLLM(
        model, provider="p", model_name="m", sleep=clock.sleep, clock=clock.time, jitter=lambda: 0.0, **kw
    )


def test_throttle_spaces_requests():
    clock = Clock()
    llm = make(FlakyModel(), clock, requests_per_minute=6)  # one request every 10 s
    llm.invoke("a")
    llm.invoke("b")
    llm.invoke("c")
    assert clock.sleeps == [10.0, 10.0]


def test_no_wait_when_calls_are_already_spaced():
    clock = Clock()
    llm = make(FlakyModel(), clock, requests_per_minute=6)
    llm.invoke("a")
    clock.now += 30
    llm.invoke("b")
    assert clock.sleeps == []


def test_retries_429_with_exponential_backoff_then_succeeds():
    clock = Clock()
    model = FlakyModel(failures=3)
    llm = make(model, clock, requests_per_minute=0)
    assert llm.invoke("hi").content == "ok"
    assert model.calls == 4
    assert clock.sleeps == [2.0, 4.0, 8.0]


def test_raises_after_retry_exhaustion():
    clock = Clock()
    model = FlakyModel(failures=99)
    llm = make(model, clock, requests_per_minute=0, max_retries=2)
    with pytest.raises(RateLimitExhausted):
        llm.invoke("hi")
    assert model.calls == 3


def test_non_rate_limit_errors_are_not_retried():
    clock = Clock()
    model = FlakyModel(failures=1, exc=ValueError("bad request"))
    llm = make(model, clock, requests_per_minute=0)
    with pytest.raises(ValueError):
        llm.invoke("hi")
    assert model.calls == 1 and clock.sleeps == []


def test_disk_cache_avoids_second_call(tmp_path):
    clock = Clock()
    model = FlakyModel(answer="cached answer")
    llm = make(model, clock, requests_per_minute=0, cache_dir=tmp_path)
    first = llm.invoke([HumanMessage(content="same prompt")])
    second = llm.invoke([HumanMessage(content="same prompt")])
    assert model.calls == 1
    assert first.content == second.content == "cached answer"
    assert second.usage_metadata["total_tokens"] == 5
    llm.invoke("different prompt")
    assert model.calls == 2
    # A fresh wrapper (e.g. next eval run) reuses the files on disk.
    fresh = make(FlakyModel(answer="new"), clock, requests_per_minute=0, cache_dir=tmp_path)
    assert fresh.invoke("same prompt").content == "cached answer"


def test_cache_key_depends_on_model(tmp_path):
    clock = Clock()
    a = make(FlakyModel(answer="A"), clock, requests_per_minute=0, cache_dir=tmp_path)
    b = ThrottledLLM(FlakyModel(answer="B"), provider="p", model_name="other", cache_dir=tmp_path, requests_per_minute=0)
    a.invoke("q")
    assert b.invoke("q").content == "B"


def test_listeners_receive_call_records(tmp_path):
    clock = Clock()
    llm = make(FlakyModel(failures=1), clock, requests_per_minute=0, cache_dir=tmp_path)
    records = []
    llm.add_listener(records.append)
    llm.invoke("q")
    llm.invoke("q")
    assert [r.cached for r in records] == [False, True]
    assert records[0].input_tokens == 3 and records[0].output_tokens == 2 and records[0].retries == 1


def test_is_rate_limit_error():
    assert is_rate_limit_error(RateLimitError("x"))
    assert is_rate_limit_error(RuntimeError("429 Too Many Requests"))
    assert is_rate_limit_error(RuntimeError("ResourceExhausted: quota exceeded"))
    assert not is_rate_limit_error(ValueError("bad json"))


def test_fake_llm_scripted_and_callable():
    fake = FakeLLM(["one", "two"])
    assert fake.invoke("a").content == "one"
    assert fake.invoke("b").content == "two"
    assert len(fake.prompts) == 2
    with pytest.raises(AssertionError, match="ran out"):
        fake.invoke("c")
    echo = FakeLLM(lambda msgs: f"echo {msgs[-1].content}")
    assert echo.invoke("hello").content == "echo hello"


@pytest.fixture
def no_dotenv(monkeypatch):
    monkeypatch.setattr(llm_module, "load_dotenv", lambda *a, **k: None)
    for var in ("LLM_PROVIDER", "LLM_MODEL", "GEMINI_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY", "LLM_CACHE", "OLLAMA_BASE_URL"):
        monkeypatch.delenv(var, raising=False)


def test_get_llm_defaults_to_gemini_flash(no_dotenv, monkeypatch):
    seen = {}
    monkeypatch.setattr(llm_module, "init_chat_model", lambda model, **kw: seen.update(model=model, **kw) or FlakyModel())
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    client = get_llm()
    assert seen["model"] == llm_module.DEFAULT_MODEL and "flash" in seen["model"]
    assert seen["model_provider"] == "google_genai" and seen["api_key"] == "test-key"
    assert client.provider == "google_genai"


def test_provider_switch_via_env_without_code_changes(no_dotenv, monkeypatch):
    seen = {}
    monkeypatch.setattr(llm_module, "init_chat_model", lambda model, **kw: seen.update(model=model, **kw) or FlakyModel())
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("LLM_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setenv("GROQ_API_KEY", "g")
    client = get_llm()
    assert (seen["model"], seen["model_provider"]) == ("llama-3.3-70b-versatile", "groq")
    assert client.model == "llama-3.3-70b-versatile"

    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MODEL", "llama3.1")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    get_llm()
    assert seen["model_provider"] == "ollama" and seen["base_url"] == "http://localhost:11434"


def test_missing_api_key_gives_clear_error(no_dotenv):
    with pytest.raises(LLMConfigError, match="GEMINI_API_KEY"):
        get_llm()


def test_cache_flag_enables_disk_cache(no_dotenv, monkeypatch, tmp_path):
    monkeypatch.setattr(llm_module, "init_chat_model", lambda model, **kw: FlakyModel())
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("LLM_CACHE_DIR", str(tmp_path))
    client = get_llm(cache=True, requests_per_minute=0)
    client.invoke("q")
    assert list(tmp_path.glob("*.json"))


class SlowModel:
    """Takes 1.5 s of (fake) clock time per call."""

    def __init__(self, clock):
        self.clock = clock

    def invoke(self, messages):
        self.clock.now += 1.5
        return AIMessage(content="slow", usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2})


def test_latency_excludes_throttle_and_backoff_waits(tmp_path):
    clock = Clock()
    llm = make(SlowModel(clock), clock, requests_per_minute=6, cache_dir=tmp_path)
    records = []
    llm.add_listener(records.append)
    llm.invoke("a")
    llm.invoke("b")  # throttled: waits before the request
    assert [r.latency_s for r in records] == [1.5, 1.5]
    assert records[0].waited_s == 0.0 and records[1].waited_s > 0


def test_cache_hit_reports_original_request_latency(tmp_path):
    clock = Clock()
    llm = make(SlowModel(clock), clock, requests_per_minute=0, cache_dir=tmp_path)
    records = []
    llm.add_listener(records.append)
    llm.invoke("q")
    llm.invoke("q")
    assert [r.cached for r in records] == [False, True]
    assert records[1].latency_s == records[0].latency_s == 1.5


def test_backoff_time_is_reported_as_waited(tmp_path):
    clock = Clock()
    llm = make(FlakyModel(failures=2), clock, requests_per_minute=0)
    records = []
    llm.add_listener(records.append)
    llm.invoke("q")
    assert records[0].retries == 2 and records[0].waited_s == 6.0
