import json
import logging

import pytest
from langchain_core.messages import AIMessage

from app.llm import FakeLLM
from app.observability import (
    RunMetrics,
    estimate_cost,
    load_prices,
    percentile,
    summarize,
    track_run,
)


def usage(i, o):
    return AIMessage(content="x", usage_metadata={"input_tokens": i, "output_tokens": o, "total_tokens": i + o})


def test_estimate_cost_uses_price_table():
    # gemini-2.5-flash: 0.30 in / 2.50 out per 1M tokens
    assert estimate_cost("gemini-2.5-flash", 1_000_000, 1_000_000) == pytest.approx(2.80)
    assert estimate_cost("gemini-2.5-flash", 1000, 0) == pytest.approx(0.0003)
    assert estimate_cost("some-local-ollama-model", 5000, 5000) == 0.0


def test_prices_can_be_overridden_with_env(monkeypatch):
    monkeypatch.setenv("LLM_PRICES_JSON", '{"gemini-2.5-flash": [1.0, 2.0], "new-model": [3, 4]}')
    prices = load_prices()
    assert prices["gemini-2.5-flash"] == (1.0, 2.0)
    assert estimate_cost("new-model", 1_000_000, 1_000_000, prices) == pytest.approx(7.0)


def test_track_run_aggregates_tokens_and_calls():
    llm = FakeLLM([usage(100, 20), usage(50, 10)])
    with track_run(llm, "q?") as run:
        llm.invoke("a")
        llm.invoke("b")
    m = run.metrics
    assert (m.llm_calls, m.input_tokens, m.output_tokens, m.total_tokens) == (2, 150, 30, 180)
    assert m.question == "q?" and m.wall_time_s >= 0 and m.real_cost_usd == 0.0


def test_calls_outside_the_block_are_not_counted():
    llm = FakeLLM([usage(1, 1), usage(2, 2), usage(4, 4)])
    llm.invoke("before")
    with track_run(llm) as run:
        llm.invoke("inside")
    llm.invoke("after")
    assert run.metrics.input_tokens == 2 and run.metrics.llm_calls == 1


def test_cost_is_computed_from_model_prices():
    class PricedFake(FakeLLM):
        model = "gemini-2.5-flash"

    llm = PricedFake([usage(1_000_000, 100_000)])
    with track_run(llm) as run:
        llm.invoke("x")
    assert run.metrics.estimated_cost_usd == pytest.approx(0.30 + 0.25)


def test_run_is_logged_as_json(caplog, tmp_path, monkeypatch):
    log_file = tmp_path / "runs.jsonl"
    monkeypatch.setenv("RUN_LOG_PATH", str(log_file))
    llm = FakeLLM([usage(7, 3)])
    with caplog.at_level(logging.INFO, logger="breeding_copilot.runs"):
        with track_run(llm, "log me") as run:
            llm.invoke("x")
    payload = json.loads(caplog.records[-1].getMessage())
    assert payload["input_tokens"] == 7 and payload["total_tokens"] == 10 and payload["real_cost_usd"] == 0.0
    assert json.loads(log_file.read_text().splitlines()[-1])["run_id"] == run.metrics.run_id


def test_run_is_finalised_even_when_the_block_raises():
    llm = FakeLLM([usage(1, 1)])
    with pytest.raises(RuntimeError):
        with track_run(llm) as run:
            llm.invoke("x")
            raise RuntimeError("boom")
    assert run.metrics.llm_calls == 1 and run.metrics.wall_time_s > 0
    llm.add_listener(lambda r: None)  # listener list still healthy; tracker was detached
    assert len(llm._listeners) == 1


def test_percentile_and_summary():
    assert percentile([], 95) == 0.0
    assert percentile([1, 2, 3, 4, 5], 95) == pytest.approx(4.8)
    runs = [RunMetrics(run_id=str(i), wall_time_s=float(i), llm_latency_s=float(i), input_tokens=i * 10, estimated_cost_usd=i / 100) for i in range(1, 6)]
    s = summarize(runs)
    assert s["wall_time_s"]["mean"] == pytest.approx(3.0)
    assert s["wall_time_s"]["p95"] == pytest.approx(4.8)
    assert s["total_tokens"]["mean"] == pytest.approx(30.0)
    assert s["estimated_cost_usd"]["mean"] == pytest.approx(0.03)
    assert summarize([])["wall_time_s"] == {"mean": 0.0, "p95": 0.0}
