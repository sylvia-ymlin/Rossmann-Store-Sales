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
	python3 -m pip install -r requirements.txt

train:
	python3 -m src.training.train

evaluate:
	python3 -m scripts.evaluate_model

drift-check:
	python3 -m scripts.check_drift

test:
	python3 -m scripts.run_tests

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
