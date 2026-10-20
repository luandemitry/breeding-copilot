# Recipes are single commands so they behave the same under GNU make (CI) and Windows make.
.PHONY: setup data train test eval run

setup:
	uv sync

data:
	@echo make data: not implemented yet, see issue 3

train:
	@echo make train: not implemented yet, see issue 4

test:
	uv run pytest

eval:
	@echo make eval: not implemented yet, see issue 12

run:
	@echo make run: not implemented yet, see issue 13
