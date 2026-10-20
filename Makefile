# Recipes are single commands so they behave the same under GNU make (CI) and Windows make.
.PHONY: setup data train test eval run

setup:
	uv sync

data:
	uv run python -m data.generate

train:
	uv run python -m models.train

test:
	uv run pytest

eval:
	uv run python -m eval.run_eval

run:
	uv run uvicorn app.api:app --host 0.0.0.0 --port 8000
