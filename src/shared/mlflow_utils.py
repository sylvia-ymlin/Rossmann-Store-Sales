import contextlib
import logging
import os
from typing import Iterator

from src.shared.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

DEFAULT_MLFLOW_DIR = PROJECT_ROOT / "mlruns"


def get_mlflow():
    """Returns the mlflow module if installed, otherwise None."""
    try:
        import mlflow
    except ImportError:
        logger.info("MLflow is not installed; skipping experiment tracking.")
        return None
    return mlflow


def configure_tracking_uri(mlflow_module) -> str:
    """Configures local file-based MLflow tracking by default."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        DEFAULT_MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
        tracking_uri = DEFAULT_MLFLOW_DIR.resolve().as_uri()
    mlflow_module.set_tracking_uri(tracking_uri)
    return tracking_uri


@contextlib.contextmanager
def start_run(run_name: str, experiment_name: str) -> Iterator[object | None]:
    """Starts an MLflow run if the dependency is available."""
    mlflow_module = get_mlflow()
    if mlflow_module is None:
        yield None
        return

    tracking_uri = configure_tracking_uri(mlflow_module)
    mlflow_module.set_experiment(experiment_name)
    logger.info("Logging MLflow run '%s' to %s", run_name, tracking_uri)
    with mlflow_module.start_run(run_name=run_name) as run:
        yield run
