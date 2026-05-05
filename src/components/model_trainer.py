"""
model_trainer.py
================
Stage 3 — Train multiple regression models, rank them by Test R²,
and save the winner to ``artifacts/models/best_model.joblib``.

Models compared
---------------
┌─────────────────────┬──────────────────────────────────────────────┐
│ Model               │ Why included                                 │
├─────────────────────┼──────────────────────────────────────────────┤
│ Random Forest       │ Robust ensemble; handles non-linearity       │
│ XGBoost             │ Best-in-class for tabular regression         │
│ LightGBM            │ Faster alternative; great for small datasets │
│ Ridge Regression    │ Linear baseline with L2 regularisation       │
│ SVR (RBF)           │ Strong generaliser on small datasets         │
└─────────────────────┴──────────────────────────────────────────────┘

Selection criterion : highest Test R² on the REAL-data test set only.
Minimum threshold   : R² ≥ 0.60  (raises an error if not met)

Outputs
-------
artifacts/models/best_model.joblib
artifacts/reports/model_comparison.json
"""

import os
import sys

import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from src.utils.logger import get_logger
from src.utils.exception import LeachingException
from src.utils.common import evaluate_all_models, save_object, save_json

logger = get_logger(__name__)


# ── Config ────────────────────────────────────────────────────────────

@dataclass
class ModelTrainerConfig:
    best_model_path:  str   = os.path.join("artifacts", "models",   "best_model.joblib")
    report_path:      str   = os.path.join("artifacts", "reports",  "model_comparison.json")
    min_r2_threshold: float = 0.60


# ── Component ─────────────────────────────────────────────────────────

class ModelTrainer:
    """Trains all candidate models and persists the best one."""

    def __init__(self, config: ModelTrainerConfig = ModelTrainerConfig()):
        self.config = config

    # ── Private ───────────────────────────────────────────────────────

    @staticmethod
    def _get_models() -> dict:
        """Instantiate all candidate estimators."""
        return {
            "Random Forest": RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
            "XGBoost": XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            ),
            "LightGBM": LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=63,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            ),
            "Ridge Regression": Ridge(alpha=1.0),
            "SVR (RBF)": SVR(kernel="rbf", C=10.0, epsilon=0.1),
        }

    @staticmethod
    def _get_hyperparams() -> dict:
        """
        Fine-tuned hyperparameters applied via set_params().
        Kept conservative to avoid overfitting on the small test set.
        """
        return {
            "Random Forest":    {"n_estimators": 300, "min_samples_leaf": 2},
            "XGBoost":          {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 6},
            "LightGBM":         {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 63},
            "Ridge Regression": {"alpha": 1.0},
            "SVR (RBF)":        {"C": 10.0, "epsilon": 0.1},
        }

    # ── Public ────────────────────────────────────────────────────────

    def initiate_model_training(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test:  np.ndarray,
        y_test:  np.ndarray,
    ) -> tuple:
        """
        Train, compare, and save the best model.

        Returns
        -------
        best_model    : fitted sklearn estimator
        comparison_df : pd.DataFrame ranking all models
        best_model_path : str
        """
        logger.info("━━━ Stage 3: Model Training ━━━━━━━━━━━━━━━━━━━━━━━")
        try:
            models = self._get_models()
            params = self._get_hyperparams()

            # Train all and compare
            comparison_df = evaluate_all_models(
                X_train, y_train, X_test, y_test, models, params
            )
            logger.info("\n" + comparison_df.to_string(index=False))

            # Pick best model
            best_row  = comparison_df.iloc[0]
            best_name = best_row["Model"]
            best_r2   = best_row["Test_R2"]

            # Quality gate
            if best_r2 < self.config.min_r2_threshold:
                raise ValueError(
                    f"Best model '{best_name}' achieved Test R²={best_r2:.4f}, "
                    f"below the minimum threshold of {self.config.min_r2_threshold}. "
                    "Review feature engineering or data quality."
                )

            # Refit winner on full training data with best params
            best_model = models[best_name]
            if best_name in params:
                best_model.set_params(**params[best_name])
            best_model.fit(X_train, y_train)

            logger.info(
                f"✔ Best model: {best_name}  |  Test R²={best_r2:.4f}"
            )

            # Persist artifacts
            save_object(self.config.best_model_path, best_model)
            save_json(
                self.config.report_path,
                {
                    "best_model":   best_name,
                    "test_r2":      float(best_r2),
                    "all_results":  comparison_df.to_dict(orient="records"),
                },
            )

            logger.info("━━━ Model Training complete ━━━━━━━━━━━━━━━━━━━━━━━")
            return best_model, comparison_df, self.config.best_model_path

        except Exception as e:
            raise LeachingException(e, sys) from e


# ── Standalone run ────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation

    train_path, test_path = DataIngestion().initiate_data_ingestion()
    X_train, y_train, X_test, y_test, _ = (
        DataTransformation().initiate_data_transformation(train_path, test_path)
    )
    best_model, report, path = ModelTrainer().initiate_model_training(
        X_train, y_train, X_test, y_test
    )
    print(f"\nBest model saved → {path}")
    print(report.to_string(index=False))
