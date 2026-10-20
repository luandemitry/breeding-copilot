import json

import pytest

from models.bundle import METRICS_FILE, MODEL_FILE, ModelNotTrainedError, load_bundle
from models.features import FEATURE_COLUMNS, feature_matrix, training_frame
from models.train import train


@pytest.fixture(scope="module")
def trained(synthetic_db, tmp_path_factory):
    artifact_dir = tmp_path_factory.mktemp("artifacts")
    metrics = train(synthetic_db, artifact_dir, n_estimators=80)
    return metrics, artifact_dir


def test_writes_artifact_and_metrics(trained):
    metrics, artifact_dir = trained
    assert (artifact_dir / MODEL_FILE).exists()
    saved = json.loads((artifact_dir / METRICS_FILE).read_text())
    assert saved["version"] == metrics.version
    assert metrics.version.startswith("qgbr-")


def test_beats_naive_baseline(trained):
    metrics, _ = trained
    assert metrics.rmse_p50 < 0.8 * metrics.rmse_baseline


def test_interval_has_reasonable_coverage(trained):
    metrics, _ = trained
    assert 0.6 <= metrics.interval_coverage_p10_p90 <= 0.95


def test_loaded_bundle_predicts_ordered_quantiles(trained, synthetic_db):
    _, artifact_dir = trained
    bundle = load_bundle(artifact_dir)
    X = feature_matrix(training_frame(synthetic_db)).head(50)
    assert list(X.columns) == FEATURE_COLUMNS
    p10, p50, p90 = bundle.predict(X)
    assert (p10 <= p50).all() and (p50 <= p90).all()


def test_version_is_deterministic(synthetic_db, tmp_path):
    a = train(synthetic_db, tmp_path / "a", n_estimators=20)
    b = train(synthetic_db, tmp_path / "b", n_estimators=20)
    assert a.version == b.version
    assert a.rmse_p50 == b.rmse_p50


def test_missing_artifact_raises_clear_error(tmp_path):
    with pytest.raises(ModelNotTrainedError, match="make train"):
        load_bundle(tmp_path)
