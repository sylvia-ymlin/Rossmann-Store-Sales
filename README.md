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

## The problem
Retailers struggle with manual sales forecasting, leading to stockouts or excessive inventory across 1,115 stores.

## What I built
An end-to-end MLOps framework that automates high-precision forecasting using XGBoost and real-time monitoring.

## Why it matters
Automated precision forecasting reduces operational waste and ensures product availability for millions of customers.

## Quick Start
1. `pip install -r requirements.txt`
2. `python scripts/train_production_model.py`
3. `streamlit run streamlit_portfolio/app.py`

## Results
- Model Accuracy: ~11.7% RMSPE
- System Latency: <50ms per inference

## What I learned
- Implementing Fourier seasonal terms significantly improves periodic demand capture.
- Automated drift detection is critical for maintaining performance in dynamic retail environments.
