import pytest

from app.schemas import Prediction
from app.tools.predict import MAX_LINES, UnknownIdError, predict_performance
from models.bundle import ModelNotTrainedError
from models.train import train


@pytest.fixture(scope="module")
def artifact_dir(synthetic_db, tmp_path_factory):
    path = tmp_path_factory.mktemp("artifacts")
    train(synthetic_db, path, n_estimators=60)
    return path


def predict(artifact_dir, synthetic_db, line_ids, env_id, **kw):
    return predict_performance(line_ids, env_id, db_path=synthetic_db, artifact_dir=artifact_dir, **kw)


def test_returns_ordered_quantiles_and_model_version(artifact_dir, synthetic_db):
    preds = predict(artifact_dir, synthetic_db, ["L042", "L001"], "Cerrado-2027")
    assert [p.line_id for p in preds] == ["L042", "L001"]  # order of the request is preserved
    for p in preds:
        assert isinstance(p, Prediction)
        assert p.env_id == "Cerrado-2027"
        assert p.p10 <= p.p50 <= p.p90
        assert p.model_version.startswith("qgbr-")


def test_prediction_depends_on_environment(artifact_dir, synthetic_db):
    drought = predict(artifact_dir, synthetic_db, ["L042"], "Matopiba-2025")[0]
    mild = predict(artifact_dir, synthetic_db, ["L042"], "Pampa-2025")[0]
    assert drought.p50 < mild.p50  # drought penalty


def test_unknown_ids_raise_clear_errors(artifact_dir, synthetic_db):
    with pytest.raises(UnknownIdError, match="environment: Mars-2099"):
        predict(artifact_dir, synthetic_db, ["L001"], "Mars-2099")
    with pytest.raises(UnknownIdError, match="L998, L999"):
        predict(artifact_dir, synthetic_db, ["L001", "L999", "L998"], "Cerrado-2027")


def test_input_validation(artifact_dir, synthetic_db):
    with pytest.raises(ValueError, match="empty"):
        predict(artifact_dir, synthetic_db, [], "Cerrado-2027")
    with pytest.raises(ValueError, match="at most"):
        predict(artifact_dir, synthetic_db, ["L001"] * (MAX_LINES + 1), "Cerrado-2027")


def test_wide_interval_flag_follows_threshold(artifact_dir, synthetic_db):
    always = predict(artifact_dir, synthetic_db, ["L001", "L002"], "Matopiba-2025", wide_ratio=0.0)
    never = predict(artifact_dir, synthetic_db, ["L001", "L002"], "Matopiba-2025", wide_ratio=100.0)
    assert all(p.wide_interval for p in always)
    assert not any(p.wide_interval for p in never)


def test_wide_interval_threshold_can_come_from_env(artifact_dir, synthetic_db, monkeypatch):
    monkeypatch.setenv("PREDICT_WIDE_RATIO", "0")
    assert all(p.wide_interval for p in predict(artifact_dir, synthetic_db, ["L001"], "Sul-2025"))


def test_missing_model_gives_clear_error(synthetic_db, tmp_path):
    with pytest.raises(ModelNotTrainedError, match="make train"):
        predict(tmp_path, synthetic_db, ["L001"], "Cerrado-2027")
