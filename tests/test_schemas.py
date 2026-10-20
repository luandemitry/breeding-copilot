import pytest
from pydantic import ValidationError

from app.schemas import Evidence, FinalAnswer, Prediction, RankedLine, Recommendation, TrialQueryResult


def test_trial_query_result_ok_flag():
    assert TrialQueryResult(question="q", sql="SELECT 1 LIMIT 1", columns=["a"], rows=[[1]], row_count=1).ok
    assert not TrialQueryResult(question="q", error="rejected").ok


def test_prediction_requires_ordered_quantiles():
    p = Prediction(line_id="L001", env_id="Cerrado-2027", p10=4.0, p50=5.0, p90=6.5, model_version="v1")
    assert p.interval_width == pytest.approx(2.5)
    with pytest.raises(ValidationError):
        Prediction(line_id="L001", env_id="E", p10=5.0, p50=4.0, p90=6.0, model_version="v1")
    with pytest.raises(ValidationError):
        Prediction(line_id="L001", env_id="E", p10=4.0, p50=5.0, p90=4.5, model_version="v1")


def test_ranked_line_rank_must_be_positive():
    assert RankedLine(line_id="L001", rank=1, score=0.9).contributions == {}
    with pytest.raises(ValidationError):
        RankedLine(line_id="L001", rank=0, score=0.9)


def test_final_answer_defaults_and_confidence_values():
    a = FinalAnswer(answer="ok")
    assert a.confidence == "medium" and a.recommendations == [] and a.evidence == [] and a.caveats == []
    with pytest.raises(ValidationError):
        FinalAnswer(answer="ok", confidence="certain")


def test_final_answer_round_trips_through_json():
    a = FinalAnswer(
        answer="Advance L001",
        recommendations=[Recommendation(line_id="L001", rationale="high yield")],
        evidence=[Evidence(kind="sql", source="SELECT 1 LIMIT 1", summary="s")],
        confidence="high",
        caveats=["synthetic data"],
    )
    assert FinalAnswer.model_validate_json(a.model_dump_json()) == a


def test_evidence_kind_is_restricted():
    with pytest.raises(ValidationError):
        Evidence(kind="web", source="x")
