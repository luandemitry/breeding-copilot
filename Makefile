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
	@echo make eval: not implemented yet, see issue 12

run:
	@echo make run: not implemented yet, see issue 13
