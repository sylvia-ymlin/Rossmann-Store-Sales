---
title: Rossmann Store Sales
emoji: 📊
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
---

# Rossmann Store Sales Intelligence

> **Architecture Status**: Refactored to V2 Standard (FastAPI + Config-Driven + Docker)

## The Problem
Retailers struggle with manual sales forecasting, leading to stockouts or excessive inventory across 1,115 stores. Accurate prediction requires handling complex seasonality, moving holidays (Easter), and competition effects.

## The Solution
An end-to-end **MLOps Prediction System** that automates high-precision forecasting.
- **Algorithm**: XGBoost with custom Feature Engineering (Fourier Seasonality, Drift Detection).
- **Architecture**: Config-driven FastAPI backend with a custom "Hand-Drawn" HTML frontend.
- **Deployment**: containerized (Docker) for Hugging Face Spaces.

## Quick Start

### Option 1: Docker (Recommended)
```bash
# Build the image
docker build -t rossmann-sales .

# Run the container (Port 7860)
docker run -p 7860:7860 rossmann-sales
```

### Option 2: Local Python
```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn src.app:app --reload --port 7860
```
Visit `http://localhost:7860` to access the interface.

## Configuration
The project is fully driven by `config.yaml`. You can adjust model parameters and pipeline steps without changing code.

```yaml
# config.yaml
feature_engineering:
  - strategy: "fourier_seasonality"
    period: 365.25
    order: 5
model_params:
  xgboost:
    n_estimators: 1000
    learning_rate: 0.05
```

## Key Engineering Features
1.  **Strict Configuration**: All hyperparameters are centralized in `config.yaml` and validated via Pydantic (`src/config.py`).
2.  **Modular Pipeline**: Feature engineering steps (Seasonality, Easter effects) are dynamically loaded.
3.  **Production Ready**: Non-root Docker container compatible with modern cloud platforms (HF Spaces).

## Performance
- **Accuracy**: ~11.7% RMSPE
- **Latency**: <50ms per inference
