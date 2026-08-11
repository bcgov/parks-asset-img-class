"""Minimal MLflow smoke test (issue #10).

Fits a sklearn dummy classifier and a dummy regressor on tiny synthetic
data, logs both runs to ``./mlflow.db``, and prints the run ids so you can
confirm they landed.

Usage::

    python scripts/mlflow_smoke_test.py
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlflow  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.dummy import DummyClassifier, DummyRegressor  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error  # noqa: E402

from src.mlflow_utils import make_run_name, make_standard_tags, setup_mlflow  # noqa: E402

import dagshub
dagshub.init(repo_owner='sgauth01', repo_name='parks-asset-img-class', mlflow=True)

def main() -> int:
    """Run the script from parsed command-line arguments."""
    setup_mlflow()
    mlflow.autolog()
    
    rng = np.random.default_rng(42)

    # Classification: majority class on a 3-class imbalanced target.
    y_cls = rng.choice(["a", "b", "c"], size=120, p=[0.55, 0.30, 0.15])
    X_cls = np.zeros((len(y_cls), 1))
    cls = DummyClassifier(strategy="most_frequent").fit(X_cls[:84], y_cls[:84])
    pred_cls = cls.predict(X_cls[84:])
    with mlflow.start_run(
        run_name=make_run_name("T2_decking_material_smoke", "majority_class"),
        tags=make_standard_tags(
            task="T2_decking_material_smoke",
            model_family="baseline",
            model_name="majority_class",
            data_version="synthetic",
            split_seed=42,
            extra={"smoke_test": "true"},
        ),
    ) as run_cls:
        mlflow.log_metric("accuracy", float(accuracy_score(y_cls[84:], pred_cls)))
        mlflow.log_metric(
            "macro_f1",
            float(f1_score(y_cls[84:], pred_cls, average="macro", zero_division=0)),
        )
        print(f"classification run id: {run_cls.info.run_id}")

    # Regression: median predictor on a lognormal target.
    y_reg = rng.lognormal(mean=1.5, sigma=0.5, size=120)
    X_reg = np.zeros((len(y_reg), 1))
    reg = DummyRegressor(strategy="median").fit(X_reg[:84], y_reg[:84])
    pred_reg = reg.predict(X_reg[84:])
    with mlflow.start_run(
        run_name=make_run_name("T2_length_m_smoke", "median_regressor"),
        tags=make_standard_tags(
            task="T2_length_m_smoke",
            model_family="baseline",
            model_name="median_regressor",
            data_version="synthetic",
            split_seed=42,
            extra={"smoke_test": "true"},
        ),
    ) as run_reg:
        mlflow.log_metric("mae", float(mean_absolute_error(y_reg[84:], pred_reg)))
        print(f"regression run id: {run_reg.info.run_id}")

    print("\nView the runs locally with:")
    print("    mlflow ui --backend-store-uri sqlite:///mlflow.db")
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
