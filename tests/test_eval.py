import json
import sqlite3
from collections import Counter

import pytest

from app.llm import FakeLLM
from app.schemas import FinalAnswer, Recommendation
from app.tools.registry import ToolRecord
from eval import scoring
from eval.build_cases import CASES_PATH, build_cases
from eval.run_eval import TARGETS, aggregate, evaluate_case, load_cases, render_report

# ------------------------------------------------------------------ the benchmark file


def test_committed_cases_file_is_well_formed():
    cases = load_cases(CASES_PATH)
    assert 20 <= len(cases) <= 30
    assert len({c["id"] for c in cases}) == len(cases)
    assert Counter(c["category"] for c in cases).keys() == {"history", "prediction", "recommendation", "safety"}
    for c in cases:
        assert {"id", "question", "category", "expected_tools", "allowed_tools", "kind", "expected"} <= set(c)
        assert set(c["expected_tools"]) <= set(c["allowed_tools"])
    assert sum(c["category"] == "safety" for c in cases) >= 5
    questions = " ".join(c["question"] for c in cases)
    assert "DROP" in questions and "DELETE" in questions.upper() and "ignore" in questions.lower()


@pytest.fixture(scope="module")
def cases(synthetic_db, trained_artifacts):
    return build_cases(synthetic_db, trained_artifacts)


def test_cases_are_rebuilt_deterministically(cases, synthetic_db, trained_artifacts):
    assert build_cases(synthetic_db, trained_artifacts) == cases


def test_expected_values_agree_with_an_independent_sql_oracle(cases, synthetic_db):
    con = sqlite3.connect(synthetic_db)
    try:
        sql = (
            "SELECT t.line_id FROM trials t JOIN environments e ON e.env_id = t.env_id WHERE e.stress = 'drought' "
            "GROUP BY t.line_id ORDER BY AVG(t.yield_t_ha) DESC, t.line_id LIMIT 5"
        )
        oracle = [r[0] for r in con.execute(sql)]
        n_over_5 = con.execute(
            "SELECT COUNT(*) FROM (SELECT line_id FROM trials GROUP BY line_id HAVING COUNT(DISTINCT env_id) > 5)"
        ).fetchone()[0]
    finally:
        con.close()
    by_q = {c["question"]: c for c in cases}
    assert by_q["Which 5 lines had the highest mean yield in drought environments?"]["expected"] == oracle
    assert by_q["How many lines were tested in more than 5 environments?"]["expected"] == n_over_5


# ------------------------------------------------------------------ scoring

@pytest.mark.parametrize(
    ("kind", "expected", "rows", "ok"),
    [
        ("ranked", ["A", "B"], [["A", 1.0], ["B", 0.5]], True),
        ("ranked", ["A", "B"], [["B", 1.0], ["A", 0.5]], False),
        ("scalar", 4.7946, [[4.79]], True),
        ("scalar", 4.7946, [[4.9]], False),
        ("scalar", 54, [[54]], True),
        ("table", [["drought", 3], ["none", 6]], [["none", 6], ["drought", 3]], True),
        ("table", [["drought", 3], ["none", 6]], [["none", 5], ["drought", 3]], False),
        ("row", ["L1", "E1", 8.5], [["L1", "E1", 8.5004]], True),
        ("row", ["L1", "E1", 8.5], [["L2", "E1", 8.5]], False),
    ],
)
def test_result_matches(kind, expected, rows, ok):
    assert scoring.result_matches(kind, expected, [], rows) is ok


def test_tools_ok_requires_expected_and_forbids_extras():
    case = {"expected_tools": ["query_trials", "rank_lines"], "allowed_tools": ["query_trials", "rank_lines", "predict_performance"]}
    rec = lambda name: ToolRecord(tool=name)  # noqa: E731
    assert scoring.tools_ok(case, [rec("query_trials"), rec("rank_lines")])
    assert scoring.tools_ok(case, [rec("query_trials"), rec("rank_lines"), rec("predict_performance")])
    assert not scoring.tools_ok(case, [rec("query_trials")])
    assert not scoring.tools_ok({**case, "allowed_tools": ["query_trials", "rank_lines"]}, [rec("query_trials"), rec("rank_lines"), rec("predict_performance")])


def test_db_fingerprint_detects_changes(synthetic_db, tmp_path):
    import shutil

    copy = tmp_path / "copy.db"
    shutil.copy(synthetic_db, copy)
    before = scoring.db_fingerprint(copy)
    assert before == scoring.db_fingerprint(synthetic_db)
    con = sqlite3.connect(copy)
    con.execute("UPDATE trials SET yield_t_ha = 99 WHERE rowid = 1")
    con.commit()
    con.close()
    assert scoring.db_fingerprint(copy) != before


# ------------------------------------------------------------------ end-to-end with a scripted LLM


def run_case(case, script, db, artifacts):
    return evaluate_case(case, FakeLLM(script), db, artifacts)


def _case(cases, cid):
    return next(c for c in cases if c["id"] == cid)


def test_history_case_scored_correct_when_agent_matches_pandas(cases, synthetic_db, trained_artifacts):
    case = _case(cases, "hist-01")
    sql = (
        "SELECT t.line_id, ROUND(AVG(t.yield_t_ha), 3) AS y FROM trials t JOIN environments e ON e.env_id = t.env_id "
        "WHERE e.stress = 'drought' GROUP BY t.line_id ORDER BY y DESC, t.line_id LIMIT 5"
    )
    script = [
        json.dumps({"action": "tool", "tool": "query_trials", "args": {"question": case["question"]}}),
        sql,
        json.dumps({"action": "answer", "answer": "Top lines: " + ", ".join(case["expected"]), "recommendations": [], "confidence": "high"}),
    ]
    out = run_case(case, script, synthetic_db, trained_artifacts)
    assert out.checks == {"faithful": True, "tool": True, "sql": True}


def test_history_case_fails_when_sql_is_wrong(cases, synthetic_db, trained_artifacts):
    case = _case(cases, "hist-01")
    script = [
        json.dumps({"action": "tool", "tool": "query_trials", "args": {"question": case["question"]}}),
        "SELECT line_id FROM lines ORDER BY line_id LIMIT 5",
        json.dumps({"action": "answer", "answer": "These lines.", "recommendations": [], "confidence": "high"}),
    ]
    out = run_case(case, script, synthetic_db, trained_artifacts)
    assert out.checks["tool"] and not out.checks["sql"]


def test_safety_case_blocked_without_llm_and_db_untouched(cases, synthetic_db, trained_artifacts):
    case = next(c for c in cases if c["category"] == "safety" and "DROP" in c["question"])
    llm = FakeLLM([])
    out = evaluate_case(case, llm, synthetic_db, trained_artifacts)
    assert out.checks == {"faithful": True, "safe": True} and llm.prompts == []


def test_crashed_case_counts_as_failed_not_as_crash(cases, synthetic_db, trained_artifacts):
    out = run_case(_case(cases, "hist-01"), [], synthetic_db, trained_artifacts)  # FakeLLM runs out of script
    assert out.error and not all(out.checks.values())


def test_prediction_case_checks_p50_is_reported(cases, synthetic_db, trained_artifacts):
    case = _case(cases, "pred-15")
    p50 = case["expected"]["p50"]["L042"]
    script = [
        json.dumps({"action": "tool", "tool": "predict_performance", "args": {"line_ids": ["L042"], "env_id": "Cerrado-2027"}}),
        json.dumps({"action": "answer", "answer": f"L042 should yield {p50} t/ha.", "recommendations": [], "confidence": "medium"}),
    ]
    out = run_case(case, script, synthetic_db, trained_artifacts)
    assert out.checks["tool"] and out.info["p50_reported"] is True


# ------------------------------------------------------------------ aggregation and report


def test_aggregate_and_report(cases, synthetic_db, trained_artifacts):
    ok_case = _case(cases, "hist-01")
    script = [
        json.dumps({"action": "tool", "tool": "query_trials", "args": {"question": "q"}}),
        "SELECT line_id FROM lines ORDER BY line_id LIMIT 5",
        json.dumps({"action": "answer", "answer": "Nothing useful.", "recommendations": [], "confidence": "low"}),
    ]
    llm = FakeLLM(script)
    results = [
        evaluate_case(ok_case, llm, synthetic_db, trained_artifacts),
        evaluate_case(next(c for c in cases if c["category"] == "safety"), llm, synthetic_db, trained_artifacts),
    ]
    agg = aggregate(results)
    assert agg["n_cases"] == 2 and agg["safety"] == 1.0 and agg["text_to_sql"] == 0.0 and agg["tool_choice"] == 1.0
    report = render_report(results, agg, llm, "qgbr-test")
    assert "Text-to-SQL accuracy" in report and "FAIL" in report  # wrong SQL is reported honestly
    assert "real cost was zero" in report and "qgbr-test" in report and "fake" in report
    assert "## Failures" in report and "hist-01" in report
    assert set(TARGETS) == {"tool_choice", "text_to_sql", "faithfulness", "safety"}


def test_recommendation_helpers():
    case = {"expected": {"n": 2, "good_lines": ["A", "B"]}}

    class R:
        final = FinalAnswer(answer="x", recommendations=[Recommendation(line_id="A", rationale="r"), Recommendation(line_id="Z", rationale="r")])

    assert scoring.recommendation_ok(case, R())
    assert scoring.recommendation_precision(case, R()) == 0.5
    assert scoring.recommendation_precision({"expected": {"n": 1, "good_lines": None}}, R()) is None
