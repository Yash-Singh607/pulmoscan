.PHONY: help install install-dev lint format test cov setup-data eda train serve docker-build docker-run clean

help:
	@echo "Targets:"
	@echo "  install      Install runtime dependencies"
	@echo "  install-dev  Install package with dev extras + pre-commit"
	@echo "  lint         Run ruff"
	@echo "  format       Run black + ruff --fix"
	@echo "  test         Run pytest"
	@echo "  cov          Run pytest with coverage"
	@echo "  setup-data   Download the Kaggle dataset"
	@echo "  eda          Run exploratory data analysis"
	@echo "  train        Train the model"
	@echo "  serve        Start the FastAPI inference server"
	@echo "  docker-build Build the serving image"
	@echo "  docker-run   Run the serving container on :8000"

install:
	pip install -r requirements.txt

install-dev:
	pip install -e ".[dev,serve,data]"
	pre-commit install

lint:
	ruff check chestxray tests

format:
	black chestxray tests
	ruff check --fix chestxray tests

test:
	pytest

cov:
	pytest --cov=chestxray --cov-report=term-missing

setup-data:
	python -m chestxray.cli setup-data

eda:
	python -m chestxray.cli eda

train:
	python -m chestxray.cli train

serve:
	python -m chestxray.cli serve

docker-build:
	docker build -t chestxray:latest .

docker-run:
	docker run --rm -p 8000:8000 -v $(PWD)/checkpoints:/app/checkpoints chestxray:latest

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ *.egg-info build dist .coverage
