"""Pydantic models shared by tools, graph and API. Every tool returns one of these."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class TrialQueryResult(BaseModel):
    """Result of ``query_trials``: the executed SQL is the source of every number in ``rows``."""

    question: str
    sql: str | None = Field(default=None, description="Validated SQL that was executed, if any.")
    columns: list[str] = Field(default_factory=list)
    rows: list[list[Any]] = Field(default_factory=list)
    row_count: int = 0
    truncated: bool = Field(default=False, description="True when the row limit cut the result.")
    error: str | None = Field(default=None, description="Why the query was rejected or failed.")

    @property
    def ok(self) -> bool:
        return self.error is None


class Prediction(BaseModel):
    """Predicted yield for one line in one environment, with an 80% interval (p10-p90)."""

    line_id: str
    env_id: str
    p10: float
    p50: float
    p90: float
    model_version: str
    wide_interval: bool = False

    @property
    def interval_width(self) -> float:
        return self.p90 - self.p10

    @field_validator("p90")
    @classmethod
    def _quantiles_ordered(cls, p90: float, info) -> float:
        p10, p50 = info.data.get("p10"), info.data.get("p50")
        if p10 is not None and p50 is not None and not (p10 <= p50 <= p90):
            raise ValueError("quantiles must satisfy p10 <= p50 <= p90")
        return p90


class RankedLine(BaseModel):
    """A line with its selection-index score and per-trait contributions."""

    line_id: str
    rank: int = Field(ge=1)
    score: float
    contributions: dict[str, float] = Field(default_factory=dict)


class Evidence(BaseModel):
    """A traceable source for numbers in the answer."""

    kind: Literal["sql", "model", "ranking"]
    source: str = Field(description="The SQL text, the model version, or the ranking weights.")
    summary: str = ""


class Recommendation(BaseModel):
    line_id: str
    rationale: str


class FinalAnswer(BaseModel):
    """Structured final response of the agent."""

    answer: str
    recommendations: list[Recommendation] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"] = "medium"
    caveats: list[str] = Field(default_factory=list)
