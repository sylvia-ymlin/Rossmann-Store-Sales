.PHONY: help install train test lint typecheck docker-build docker-run clean

help:
	@echo "Rossmann Sales Prediction - Make Commands"
	@echo ""
	@echo "  make install       Install dependencies"
	@echo "  make train         Run training pipeline"
	@echo "  make test          Run tests"
	@echo "  make lint          Run linting"
	@echo "  make typecheck     Run type checking"
	@echo "  make docker-build  Build Docker image"
	@echo "  make docker-run    Run Docker container"
	@echo "  make clean         Clean up"

install:
	pip install -r requirements.txt

train:
	python -c "from pipelines.training_pipeline import training_pipeline; training_pipeline()"

test:
	pytest tests/ -v

lint:
	ruff check src/ steps/ pipelines/

typecheck:
	mypy src/ steps/ pipelines/ --ignore-missing-imports

docker-build:
	docker build -t rossmann-sales .

docker-run:
	docker run -p 7860:7860 rossmann-sales

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
