"""Per-run observability: real tokens and latency, plus an *estimated* cost.

The project runs on free-tier models, so the **real cost is zero**. ``estimated_cost_usd`` is what the
same token usage would cost at paid list prices (``DEFAULT_PRICES`` below, overridable with the
``LLM_PRICES_JSON`` env var) and exists only to show what the workload would cost in production.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

from app.llm import BaseLLMClient, LLMCallRecord

logger = logging.getLogger("breeding_copilot.runs")

# USD per 1M tokens: (input, output). Paid list prices, used for the *estimate* only; verify before relying on them.
DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "fake": (0.0, 0.0),
}


def load_prices() -> dict[str, tuple[float, float]]:
    """Default price table merged with ``LLM_PRICES_JSON`` (e.g. '{"my-model": [0.1, 0.4]}')."""
    prices = dict(DEFAULT_PRICES)
    raw = os.getenv("LLM_PRICES_JSON")
    if raw:
        prices.update({model: (float(p[0]), float(p[1])) for model, p in json.loads(raw).items()})
    return prices


def estimate_cost(model: str, input_tokens: int, output_tokens: int, prices: dict | None = None) -> float:
    """Estimated USD cost; unknown models (e.g. local Ollama) are priced at zero."""
    price_in, price_out = (prices or load_prices()).get(model, (0.0, 0.0))
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


@dataclass
class RunMetrics:
    run_id: str
    question: str = ""
    model: str = ""
    llm_calls: int = 0
    cached_calls: int = 0
    retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    llm_latency_s: float = 0.0  # sum of provider request durations (excludes throttle/backoff waits)
    waited_s: float = 0.0  # self-imposed throttle/backoff sleeping
    wall_time_s: float = 0.0  # end-to-end duration of the run
    estimated_cost_usd: float = 0.0
    real_cost_usd: float = 0.0  # free tier: always zero

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict:
        return {**asdict(self), "total_tokens": self.total_tokens}


@dataclass
class RunTracker:
    metrics: RunMetrics
    prices: dict = field(default_factory=load_prices)

    def record(self, call: LLMCallRecord) -> None:
        m = self.metrics
        m.model = call.model
        m.llm_calls += 1
        m.cached_calls += int(call.cached)
        m.retries += call.retries
        m.input_tokens += call.input_tokens
        m.output_tokens += call.output_tokens
        m.llm_latency_s += call.latency_s
        m.waited_s += call.waited_s
        m.estimated_cost_usd += estimate_cost(call.model, call.input_tokens, call.output_tokens, self.prices)


@contextmanager
def track_run(llm: BaseLLMClient, question: str = "") -> Iterator[RunTracker]:
    """Collect token/latency/cost for everything ``llm`` does inside the ``with`` block.

    On exit one JSON line is logged (and appended to ``RUN_LOG_PATH`` when set).
    """
    tracker = RunTracker(RunMetrics(run_id=uuid.uuid4().hex[:12], question=question))
    llm.add_listener(tracker.record)
    started = time.perf_counter()
    try:
        yield tracker
    finally:
        llm.remove_listener(tracker.record)
        tracker.metrics.wall_time_s = time.perf_counter() - started
        _emit_log(tracker.metrics)


def _emit_log(metrics: RunMetrics) -> None:
    line = json.dumps(metrics.to_dict())
    logger.info(line)
    log_path = os.getenv("RUN_LOG_PATH")
    if log_path:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def percentile(values: list[float], q: float) -> float:
    """``q``-th percentile (linear interpolation); 0.0 for an empty list."""
    return float(np.percentile(values, q)) if values else 0.0


def summarize(runs: list[RunMetrics]) -> dict[str, dict[str, float]]:
    """Mean and p95 of latency, tokens and estimated cost over many runs (used by the eval report)."""
    series = {
        "wall_time_s": [r.wall_time_s for r in runs],
        "llm_latency_s": [r.llm_latency_s for r in runs],
        "total_tokens": [float(r.total_tokens) for r in runs],
        "estimated_cost_usd": [r.estimated_cost_usd for r in runs],
    }
    return {
        name: {"mean": float(np.mean(vals)) if vals else 0.0, "p95": percentile(vals, 95)}
        for name, vals in series.items()
    }
