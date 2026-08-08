.PHONY: install db-up db-init test lint format

install:
	pip install -e ".[dev]"

db-up:
	docker compose up -d postgres mlflow

db-init:
	python -m creditguard.db.init_db

test:
	pytest -q

lint:
	ruff check .

format:
	black .
