"""Central paths and settings. Everything is overridable through environment variables."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DB_PATH = Path(os.getenv("DB_PATH", ROOT / "data" / "breeding.db"))
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", ROOT / "artifacts"))
