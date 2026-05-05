"""
common.py
=========
Shared utility functions used across multiple pipeline stages.

Functions
---------
- save_object      : serialise any Python object with joblib
- load_object      : deserialise a joblib artifact
- save_json        : write a dict to a JSON file
- evaluate_model   : compute R², RMSE, MAE for a single model
- evaluate_all_models : train + compare a dict of models and return
                        a ranked DataFrame
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import Any

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)

from src.utils.logger import get_logger
from src.utils.exception import LeachingException

logger = get_logger(__name__)


# ── Artifact I/O ─────────────────────────────────────────────────────

def save_object(file_path: str, obj: Any) -> None:
    """Serialise *obj* to *file_path* using joblib."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(obj, file_path)
        logger.info(f"Saved  → {file_path}")
    except Exception as e:
        raise LeachingException(e, sys) from e


def load_object(file_path: str) -> Any:
    """Deserialise a joblib artifact from *file_path*."""
    try:
        obj = joblib.load(file_path)
        logger.info(f"Loaded ← {file_path}")
        return obj
    except Exception as e:
        raise LeachingException(e, sys) from e


def save_json(file_path: str, data: dict) -> None:
    """Persist a dictionary as a pretty-printed JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as fh:
            json.dump(data, fh, indent=4, default=str)
        logger.info(f"JSON   → {file_path}")
    except Exception as e:
        raise LeachingException(e, sys) from e


# ── Evaluation helpers ────────────────────────────────────────────────

def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Compute standard regression metrics.

    Returns
    -------
    dict  →  {"r2": float, "rmse": float, "mae": float}
    """
    return {
        "r2":   round(float(r2_score(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "mae":  round(float(mean_absolute_error(y_true, y_pred)), 4),
    }


def evaluate_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    models:  dict,
    params:  dict,
) -> pd.DataFrame:
    """
    Train every model in *models*, apply the matching params from
    *params*, evaluate on the test set, and return a comparison
    DataFrame sorted by Test R² (descending).

    Parameters
    ----------
    models : {name: sklearn estimator}
    params : {name: dict}  — applied via ``set_params(**p)``

    Returns
    -------
    pd.DataFrame  with columns: Model | Train_R2 | Test_R2 | RMSE | MAE
    """
    results = []

    for name, model in models.items():
        try:
            if name in params and params[name]:
                model.set_params(**params[name])

            model.fit(X_train, y_train)

            train_m = evaluate_model(y_train, model.predict(X_train))
            test_m  = evaluate_model(y_test,  model.predict(X_test))

            results.append({
                "Model":    name,
                "Train_R2": train_m["r2"],
                "Test_R2":  test_m["r2"],
                "RMSE":     test_m["rmse"],
                "MAE":      test_m["mae"],
            })

            logger.info(
                f"  {name:<28} │ "
                f"Train R²={train_m['r2']:.4f} │ "
                f"Test  R²={test_m['r2']:.4f} │ "
                f"RMSE={test_m['rmse']:.4f}"
            )

        except Exception as exc:
            logger.warning(f"  {name} skipped — {exc}")

    df = pd.DataFrame(results).sort_values("Test_R2", ascending=False)
    df.reset_index(drop=True, inplace=True)
    return df
