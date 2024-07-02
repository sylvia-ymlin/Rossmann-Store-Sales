.PHONY: help install train evaluate drift-check test lint typecheck run docker-build docker-run clean

help:
	@echo "Rossmann Sales Prediction - Make Commands"
	@echo ""
	@echo "  make install       Install dependencies"
	@echo "  make train         Run training pipeline"
	@echo "  make evaluate      Run holdout and backtesting evaluation"
	@echo "  make drift-check   Build an offline drift report from inference logs"
	@echo "  make test          Run tests"
	@echo "  make lint          Run linting"
	@echo "  make typecheck     Run type checking"
	@echo "  make run           Run the FastAPI app locally"
	@echo "  make docker-build  Build the inference image"
	@echo "  make docker-run    Run the inference image"
	@echo "  make clean         Clean up"

install:
	pip install -r requirements.txt

train:
	python src/training/train.py

evaluate:
	python scripts/evaluate_model.py

drift-check:
	python scripts/check_drift.py

test:
	python scripts/run_tests.py

lint:
	ruff check src/ tests/ web/ scripts/

typecheck:
	mypy src/ tests/ web/ scripts/ --ignore-missing-imports

run:
	uvicorn src.serving.api:app --reload --port 7860

docker-build:
	docker build -t rossmann-sales .

docker-run:
	docker run -p 7860:7860 rossmann-sales

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
