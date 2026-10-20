import pytest

from data.generate import generate


@pytest.fixture(scope="session")
def synthetic_db(tmp_path_factory):
    """A generated synthetic database shared across the whole test session."""
    return generate(tmp_path_factory.mktemp("session_db") / "breeding.db")


@pytest.fixture(scope="session")
def trained_artifacts(synthetic_db, tmp_path_factory):
    """A small trained model (fast) for tests that need predictions."""
    from models.train import train

    artifact_dir = tmp_path_factory.mktemp("session_artifacts")
    train(synthetic_db, artifact_dir, n_estimators=40)
    return artifact_dir
