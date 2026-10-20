"""Uniform entry point the graph uses to run tools.

Arguments coming from the LLM are validated with Pydantic; tool failures never raise, they come back
as a :class:`ToolRecord` with ``ok=False`` so the agent can see the error and react.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from app.config import ARTIFACT_DIR, DB_PATH
from app.llm import BaseLLMClient
from app.schemas import Candidate
from app.tools.predict import predict_performance
from app.tools.query_trials import query_trials
from app.tools.rank import rank_lines
from models.features import load_tables

RANK_TRAITS = ("yield_t_ha", "disease_score", "plant_height_cm", "predicted_yield")

TOOL_DESCRIPTIONS = """\
1. query_trials  args: {"question": "<self-contained natural-language question about historical trials>"}
   Answers questions about historical data (means, counts, top-N, filters by stress/region/year) by running
   a read-only SQL query. Returns columns and rows plus the SQL that produced them.
2. predict_performance  args: {"line_ids": ["L042", ...], "env_id": "Cerrado-2027"}
   Model prediction of yield (t/ha) for lines in an environment: p10, p50 (point estimate), p90 and the model version.
3. rank_lines  args: {"line_ids": ["L001", ...], "weights": {"yield_t_ha": 0.6, "disease_score": 0.4}, "env_id": "<optional>"}
   Selection index over the given lines. Allowed weight traits: yield_t_ha (historical mean yield), disease_score
   (historical mean; lower is better), plant_height_cm, predicted_yield (needs env_id). Returns ranked lines with scores
   and the trait values used.\
"""


class QueryArgs(BaseModel):
    question: str = Field(min_length=3)


class PredictArgs(BaseModel):
    line_ids: list[str] = Field(min_length=1)
    env_id: str


class RankArgs(BaseModel):
    line_ids: list[str] = Field(min_length=1)
    weights: dict[str, float] = Field(min_length=1)
    env_id: str | None = None


class ToolRecord(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    ok: bool = True
    result: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


@dataclass
class ToolContext:
    llm: BaseLLMClient
    db_path: Path | str = DB_PATH
    artifact_dir: Path | str = ARTIFACT_DIR


def _build_candidates(args: RankArgs, ctx: ToolContext) -> list[Candidate]:
    bad = [t for t in args.weights if t not in RANK_TRAITS]
    if bad:
        raise ValueError(f"unknown weight trait(s): {', '.join(bad)}. Allowed: {', '.join(RANK_TRAITS)}")
    lines, _, trials = load_tables(ctx.db_path)
    unknown = sorted(set(args.line_ids) - set(lines["line_id"]))
    if unknown:
        raise ValueError(f"unknown line(s): {', '.join(unknown)}")
    means = trials.groupby("line_id")[["yield_t_ha", "disease_score", "plant_height_cm"]].mean()
    predicted: dict[str, float] = {}
    if "predicted_yield" in args.weights:
        if not args.env_id:
            raise ValueError("predicted_yield requires env_id")
        preds = predict_performance(args.line_ids, args.env_id, db_path=ctx.db_path, artifact_dir=ctx.artifact_dir)
        predicted = {p.line_id: p.p50 for p in preds}
    candidates = []
    for line_id in dict.fromkeys(args.line_ids):  # de-duplicate, keep order
        traits = {}
        for trait in args.weights:
            if trait == "predicted_yield":
                traits[trait] = predicted[line_id]
            else:
                if line_id not in means.index:
                    raise ValueError(f"line {line_id} has no trials")
                traits[trait] = round(float(means.loc[line_id, trait]), 3)
        candidates.append(Candidate(line_id=line_id, traits=traits))
    return candidates


def run_tool(name: str, args: dict[str, Any], ctx: ToolContext) -> ToolRecord:
    """Validate ``args`` and run the tool; any failure is returned as ``ok=False`` with a message."""
    try:
        if name == "query_trials":
            a = QueryArgs(**args)
            r = query_trials(a.question, ctx.llm, ctx.db_path)
            return ToolRecord(tool=name, args=a.model_dump(), ok=r.ok, result=r.model_dump(), error=r.error)
        if name == "predict_performance":
            a = PredictArgs(**args)
            preds = predict_performance(a.line_ids, a.env_id, db_path=ctx.db_path, artifact_dir=ctx.artifact_dir)
            rows = [{**p.model_dump(), "interval_width": round(p.interval_width, 3)} for p in preds]
            return ToolRecord(tool=name, args=a.model_dump(), result={"predictions": rows})
        if name == "rank_lines":
            a = RankArgs(**args)
            candidates = _build_candidates(a, ctx)
            ranked = rank_lines(candidates, a.weights)
            return ToolRecord(
                tool=name,
                args=a.model_dump(),
                result={
                    "weights": a.weights,
                    "ranking": [r.model_dump() for r in ranked],
                    "candidates": [c.model_dump() for c in candidates],
                },
            )
        return ToolRecord(tool=name, args=args, ok=False, error=f"unknown tool: {name}")
    except ValidationError as exc:
        return ToolRecord(tool=name, args=args, ok=False, error=f"invalid arguments: {exc.errors()[0]['msg']}")
    except (ValueError, FileNotFoundError) as exc:  # UnknownIdError, ModelNotTrainedError, rank validation
        return ToolRecord(tool=name, args=args, ok=False, error=str(exc))
