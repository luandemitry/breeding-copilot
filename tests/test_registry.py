import pytest

from app.llm import FakeLLM
from app.tools.registry import ToolContext, run_tool
from models.features import load_tables
from models.train import train


@pytest.fixture(scope="module")
def ctx_factory(synthetic_db, tmp_path_factory):
    artifact_dir = tmp_path_factory.mktemp("artifacts")
    train(synthetic_db, artifact_dir, n_estimators=40)

    def make(responses=()):
        return ToolContext(llm=FakeLLM(list(responses)), db_path=synthetic_db, artifact_dir=artifact_dir)

    return make


def test_query_trials_through_registry(ctx_factory):
    ctx = ctx_factory(["SELECT line_id FROM lines ORDER BY line_id LIMIT 2"])
    rec = run_tool("query_trials", {"question": "two lines"}, ctx)
    assert rec.ok and rec.result["row_count"] == 2 and rec.result["sql"]


def test_query_trials_failure_is_reported_not_raised(ctx_factory):
    ctx = ctx_factory(["DROP TABLE trials", "DROP TABLE trials"])
    rec = run_tool("query_trials", {"question": "wipe it"}, ctx)
    assert not rec.ok and "rejected" in rec.error


def test_predict_through_registry_includes_interval_width(ctx_factory):
    rec = run_tool("predict_performance", {"line_ids": ["L042"], "env_id": "Cerrado-2027"}, ctx_factory())
    assert rec.ok
    p = rec.result["predictions"][0]
    assert p["model_version"].startswith("qgbr-")
    assert p["interval_width"] == pytest.approx(p["p90"] - p["p10"], abs=0.002)


def test_rank_lines_uses_historical_means_from_the_database(ctx_factory, synthetic_db):
    ids = ["L001", "L002", "L003"]
    rec = run_tool("rank_lines", {"line_ids": ids, "weights": {"yield_t_ha": 0.6, "disease_score": 0.4}}, ctx_factory())
    assert rec.ok
    _, _, trials = load_tables(synthetic_db)
    expected = trials[trials.line_id.isin(ids)].groupby("line_id")[["yield_t_ha", "disease_score"]].mean()
    for cand in rec.result["candidates"]:
        assert cand["traits"]["yield_t_ha"] == pytest.approx(expected.loc[cand["line_id"], "yield_t_ha"], abs=1e-3)
    assert [r["rank"] for r in rec.result["ranking"]] == [1, 2, 3]


def test_rank_lines_with_predicted_yield(ctx_factory):
    args = {"line_ids": ["L001", "L002"], "weights": {"predicted_yield": 1}, "env_id": "Cerrado-2027"}
    assert run_tool("rank_lines", args, ctx_factory()).ok
    missing_env = run_tool("rank_lines", {"line_ids": ["L001"], "weights": {"predicted_yield": 1}}, ctx_factory())
    assert not missing_env.ok and "env_id" in missing_env.error


@pytest.mark.parametrize(
    ("name", "args", "message"),
    [
        ("rank_lines", {"line_ids": ["L001"], "weights": {"color": 1}}, "unknown weight trait"),
        ("rank_lines", {"line_ids": ["L999"], "weights": {"yield_t_ha": 1}}, "unknown line"),
        ("predict_performance", {"line_ids": ["L001"], "env_id": "Mars-1"}, "unknown environment"),
        ("predict_performance", {"line_ids": [], "env_id": "Cerrado-2027"}, "invalid arguments"),
        ("query_trials", {}, "invalid arguments"),
        ("delete_everything", {}, "unknown tool"),
    ],
)
def test_errors_are_returned_as_records(ctx_factory, name, args, message):
    rec = run_tool(name, args, ctx_factory())
    assert not rec.ok and message in rec.error
