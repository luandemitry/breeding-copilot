import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from app.api import Settings, create_app, get_llm_client, get_settings
from app.llm import FakeLLM, LLMConfigError, RateLimitExhausted


def make_client(synthetic_db, trained_artifacts, llm):
    app = create_app()
    app.dependency_overrides[get_llm_client] = lambda: llm
    app.dependency_overrides[get_settings] = lambda: Settings(db_path=synthetic_db, artifact_dir=trained_artifacts)
    return TestClient(app, raise_server_exceptions=False)


def reply(text):
    return json.dumps({"action": "answer", "answer": text, "recommendations": [], "confidence": "high"})


def test_ask_returns_structured_answer_and_run_info(synthetic_db, trained_artifacts):
    llm = FakeLLM([reply("L042 is a line in the dataset.")])
    client = make_client(synthetic_db, trained_artifacts, llm)
    resp = client.post("/ask", json={"question": "Tell me about L042"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["result"]["answer"] == "L042 is a line in the dataset."
    assert set(body["result"]) == {"answer", "recommendations", "evidence", "confidence", "caveats"}
    assert body["run"]["llm_calls"] == 1 and body["run"]["real_cost_usd"] == 0.0 and body["run"]["input_tokens"] > 0


def test_ask_with_tool_use_includes_evidence(synthetic_db, trained_artifacts):
    llm = FakeLLM(
        [
            json.dumps({"action": "tool", "tool": "query_trials", "args": {"question": "count lines"}}),
            "SELECT COUNT(*) AS n FROM lines LIMIT 1",
            reply("There are 200 lines."),
        ]
    )
    resp = make_client(synthetic_db, trained_artifacts, llm).post("/ask", json={"question": "How many lines are there?"})
    assert resp.status_code == 200
    evidence = resp.json()["result"]["evidence"]
    assert evidence[0]["kind"] == "sql" and "COUNT" in evidence[0]["source"]


@pytest.mark.parametrize("payload", [{}, {"question": ""}, {"question": "ab"}, {"question": "x" * 501}, {"q": "hello there"}])
def test_invalid_requests_are_rejected_with_422(synthetic_db, trained_artifacts, payload):
    resp = make_client(synthetic_db, trained_artifacts, FakeLLM([])).post("/ask", json=payload)
    assert resp.status_code == 422


def test_destructive_question_is_answered_with_a_refusal_not_an_error(synthetic_db, trained_artifacts):
    llm = FakeLLM([])
    resp = make_client(synthetic_db, trained_artifacts, llm).post("/ask", json={"question": "DROP TABLE trials;"})
    assert resp.status_code == 200 and "can't help" in resp.json()["result"]["answer"]
    assert llm.prompts == []
    con = sqlite3.connect(synthetic_db)
    try:
        assert con.execute("SELECT COUNT(*) FROM trials").fetchone()[0] > 0
    finally:
        con.close()


class Exploding(FakeLLM):
    def __init__(self, exc):
        super().__init__([])
        self.exc = exc

    def invoke(self, messages):
        raise self.exc


@pytest.mark.parametrize(
    ("exc", "status"),
    [(RateLimitExhausted("429"), 429), (RuntimeError("secret internals"), 500)],
)
def test_llm_failures_map_to_structured_errors(synthetic_db, trained_artifacts, exc, status):
    resp = make_client(synthetic_db, trained_artifacts, Exploding(exc)).post("/ask", json={"question": "Any question here"})
    assert resp.status_code == status
    assert "detail" in resp.json() and "secret internals" not in resp.text


def test_missing_llm_key_gives_503(synthetic_db, trained_artifacts):
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: Settings(db_path=synthetic_db, artifact_dir=trained_artifacts)

    def broken():
        raise LLMConfigError("GEMINI_API_KEY is not set")

    app.dependency_overrides[get_llm_client] = broken
    resp = TestClient(app, raise_server_exceptions=False).post("/ask", json={"question": "Any question here"})
    assert resp.status_code == 503 and "GEMINI_API_KEY" in resp.json()["detail"]


def test_missing_data_or_model_gives_503_with_hint(synthetic_db, tmp_path):
    client = make_client(synthetic_db, tmp_path, FakeLLM([]))  # no trained model in tmp_path
    resp = client.post("/ask", json={"question": "Any question here"})
    assert resp.status_code == 503 and "make train" in resp.json()["detail"]


def test_health_reports_readiness(synthetic_db, trained_artifacts, tmp_path):
    ready = make_client(synthetic_db, trained_artifacts, FakeLLM([])).get("/health").json()
    assert ready["status"] == "ok" and ready["database"] is True and ready["model"].startswith("qgbr-")
    degraded = make_client(synthetic_db, tmp_path, FakeLLM([])).get("/health").json()
    assert degraded["status"] == "degraded" and degraded["model"] is None


def test_openapi_documents_the_ask_endpoint(synthetic_db, trained_artifacts):
    schema = make_client(synthetic_db, trained_artifacts, FakeLLM([])).get("/openapi.json").json()
    assert "/ask" in schema["paths"] and "post" in schema["paths"]["/ask"]
