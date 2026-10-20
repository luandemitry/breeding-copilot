import pytest

from data.generate import generate


@pytest.fixture(scope="session")
def synthetic_db(tmp_path_factory):
    """A generated synthetic database shared across the whole test session."""
    return generate(tmp_path_factory.mktemp("session_db") / "breeding.db")
