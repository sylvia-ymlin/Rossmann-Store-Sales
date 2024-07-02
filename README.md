# Rossmann Store Sales Forecasting

This project predicts daily Rossmann store sales from tabular retail data. It is a small end-to-end forecasting and MLOps learning project built around one training pipeline, one evaluation script, local experiment tracking, and one simple FastAPI endpoint with a browser page for manual testing.

## What The Project Does

- merges historical sales with store metadata
- builds calendar, holiday, and store features
- trains an XGBoost regressor on `log1p(Sales)`
- evaluates the model with a time-based holdout split and rolling backtests
- logs experiments locally with MLflow
- serves predictions through a small API and demo page

## Problem And Approach

The task is to predict daily `Sales` for a Rossmann store on a given date. The project uses a structured tabular approach rather than a large forecasting stack.

Main ideas:

- time-aware evaluation instead of random train/validation splits
- simple feature engineering for seasonality and holidays
- one XGBoost model for both offline evaluation and online prediction
- local file-based MLflow tracking for training and evaluation runs

## Project Structure

```text
src/training/    data loading, feature engineering, split helpers, model training
src/serving/     FastAPI prediction service
src/shared/      config, MLflow helper, and request/response schemas
scripts/         evaluation, drift check, and test runner
web/             minimal HTML demo page
reports/metrics/ generated training and evaluation outputs
tests/           unit tests for training, serving, and split logic
Dockerfile       minimal container image for API inference
```

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train a model:

```bash
make train
```

Run evaluation:

```bash
make evaluate
```

This writes:

- `models/rossmann_model.json`
- `models/model_metadata.json`
- `reports/metrics/training_summary.json`
- `reports/metrics/model_evaluation.json`

If `mlflow` is installed, both commands also create local experiment runs under `mlruns/`
by default. You can override that with `MLFLOW_TRACKING_URI`.

Build an offline drift report from logged inference requests:

```bash
make drift-check
```

This writes `reports/metrics/drift_report.json` when inference logs are available.

Start the API demo:

```bash
make run
```

Then open [http://localhost:7860](http://localhost:7860).

The API reads both the model artifact and the generated metadata file. `/health`
and `/predict` include the current `model_version`.

## Docker Deployment

To build the containerized inference service, first generate a model locally:

```bash
make train
```

Then build and run the image:

```bash
make docker-build
make docker-run
```

The container only serves inference. It does not train the model during image build.

Run tests:

```bash
make test
```

The repository includes a small test runner wrapper for this environment because the bundled `readline` import in the current Conda Python build crashes during raw `pytest` startup on macOS.

## API Example

```bash
curl -X POST http://localhost:7860/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Store": 1,
    "Date": "2015-07-31",
    "Promo": 1,
    "StateHoliday": "0",
    "SchoolHoliday": 1,
    "ForecastDays": 7
  }'
```

The API looks up static store metadata from `store.csv`, so the request stays small.
Each request also appends one structured JSONL record to `logs/inference_requests.jsonl`
with timestamp, store id, forecast horizon, model version, and latency.

## Example Results

After running `make train` and `make evaluate`, you will have evaluation artifacts in `reports/metrics/`.

From the current saved backtest summary:

| Metric | Value |
| --- | ---: |
| Average tuned RMSPE | 13.2412 |
| Average baseline RMSPE | 22.9997 |
| Average improvement vs baseline | 9.7585 |

These numbers reflect the current local run. Recompute them after retraining if the code or data changes.

## CI

The repository includes one small GitHub Actions workflow that runs lint and tests
on `main` pushes and pull requests. It is intended as a validation step, not a
deployment pipeline.

## Files You Need

The code expects the Rossmann dataset files under `data/raw/`.

- `store.csv` is included
- `train.csv` is required for full training and evaluation

The repository also includes `test.csv`, `sample_submission.csv`, and `train_schema.csv`, but they are not part of the main workflow.

## Limitations

- this is a compact forecasting demo, not a leaderboard-focused Kaggle solution
- feature engineering is intentionally simple and mostly manual
- saved metrics may become stale if code or data changes
- MLflow tracking is local and file-based; there is no remote tracking server or registry
- CI validates the codebase but does not deploy artifacts or publish models
- Drift checking is offline and based on logged inference requests, not live monitoring
- the explanation output is only a model contribution view, not a causal interpretation
- the API assumes the requested store exists in `store.csv`
