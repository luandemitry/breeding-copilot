"""FastAPI service: ``POST /ask`` runs the agent, ``GET /health`` reports readiness."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.config import ARTIFACT_DIR, DB_PATH
from app.graph import run_agent
from app.llm import BaseLLMClient, LLMConfigError, RateLimitExhausted, get_llm
from app.schemas import FinalAnswer
from models.bundle import MODEL_FILE, ModelNotTrainedError, load_bundle

logger = logging.getLogger("breeding_copilot.api")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500, description="A question about trials or lines to advance.")


class RunInfo(BaseModel):
    """What the run cost. ``real_cost_usd`` is 0 on free-tier models; the estimate uses paid list prices."""

    model: str
    llm_calls: int
    input_tokens: int
    output_tokens: int
    wall_time_s: float
    llm_latency_s: float
    estimated_cost_usd: float
    real_cost_usd: float


class AskResponse(BaseModel):
    result: FinalAnswer
    run: RunInfo


@dataclass
class Settings:
    db_path: Path
    artifact_dir: Path


def get_settings() -> Settings:
    return Settings(db_path=DB_PATH, artifact_dir=ARTIFACT_DIR)


@lru_cache(maxsize=1)
def _shared_llm() -> BaseLLMClient:
    return get_llm()  # created lazily so the service can start (and /health work) without an API key


def get_llm_client() -> BaseLLMClient:
    return _shared_llm()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Breeding Copilot",
        version="0.1.0",
        description="Ask questions about (synthetic) plant-breeding field trials and get lines to advance.",
    )

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error: %s", exc)
        return JSONResponse(status_code=500, content={"detail": "Internal error while answering the question."})

    @app.get("/health")
    def health(settings: Settings = Depends(get_settings)) -> dict:
        model_ready = (settings.artifact_dir / MODEL_FILE).exists()
        return {
            "status": "ok" if settings.db_path.exists() and model_ready else "degraded",
            "database": settings.db_path.exists(),
            "model": load_bundle(settings.artifact_dir).version if model_ready else None,
        }

    @app.post("/ask", response_model=AskResponse)
    def ask(
        body: AskRequest,
        llm: BaseLLMClient = Depends(get_llm_client),
        settings: Settings = Depends(get_settings),
    ) -> AskResponse:
        if not settings.db_path.exists() or not (settings.artifact_dir / MODEL_FILE).exists():
            raise HTTPException(503, "Data or model missing. Run `make data && make train` first.")
        try:
            outcome = run_agent(body.question, llm=llm, db_path=settings.db_path, artifact_dir=settings.artifact_dir)
        except RateLimitExhausted as exc:
            raise HTTPException(429, f"The LLM provider is rate limiting requests (free tier). Try again shortly. {exc}") from exc
        except ModelNotTrainedError as exc:
            raise HTTPException(503, str(exc)) from exc
        m = outcome.metrics
        return AskResponse(
            result=outcome.final,
            run=RunInfo(
                model=m.model,
                llm_calls=m.llm_calls,
                input_tokens=m.input_tokens,
                output_tokens=m.output_tokens,
                wall_time_s=round(m.wall_time_s, 3),
                llm_latency_s=round(m.llm_latency_s, 3),
                estimated_cost_usd=round(m.estimated_cost_usd, 6),
                real_cost_usd=m.real_cost_usd,
            ),
        )

    @app.exception_handler(LLMConfigError)
    async def llm_config(_: Request, exc: LLMConfigError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": f"LLM is not configured: {exc}"})

    return app


app = create_app()
