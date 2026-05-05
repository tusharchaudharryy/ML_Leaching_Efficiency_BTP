"""
model_evaluation.py
===================
Stage 4 — Evaluate the saved best model on the real-data test set,
generate diagnostic plots, and optionally run SHAP analysis.

Outputs (all saved to artifacts/)
----------------------------------
plots/actual_vs_predicted.png  — scatter of actual vs predicted %
plots/residuals.png            — residual vs fitted + residual histogram
plots/shap_summary.png         — SHAP bar chart (if shap is installed)
reports/evaluation_report.json — final R², RMSE, MAE metrics
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")               # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
from dataclasses import dataclass

from src.utils.logger import get_logger
from src.utils.exception import LeachingException
from src.utils.common import evaluate_model, load_object, save_json

logger = get_logger(__name__)

PLOT_DIR   = os.path.join("artifacts", "plots")
REPORT_DIR = os.path.join("artifacts", "reports")


# ── Config ────────────────────────────────────────────────────────────

@dataclass
class ModelEvaluationConfig:
    eval_report_path: str = os.path.join(REPORT_DIR, "evaluation_report.json")


# ── Component ─────────────────────────────────────────────────────────

class ModelEvaluation:
    """Generates evaluation plots and saves the final metrics report."""

    def __init__(
        self,
        config: ModelEvaluationConfig = ModelEvaluationConfig(),
    ):
        self.config = config
        os.makedirs(PLOT_DIR,   exist_ok=True)
        os.makedirs(REPORT_DIR, exist_ok=True)

    # ── Plot helpers ──────────────────────────────────────────────────

    def _plot_actual_vs_predicted(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> None:
        """Scatter plot of actual vs. predicted leaching efficiency."""
        fig, ax = plt.subplots(figsize=(7, 6))

        ax.scatter(
            y_true, y_pred,
            alpha=0.65, edgecolors="k", linewidths=0.3,
            color="#2E75B6", label="Predictions", zorder=3,
        )
        lims = [
            min(y_true.min(), y_pred.min()) - 3,
            max(y_true.max(), y_pred.max()) + 3,
        ]
        ax.plot(lims, lims, "r--", lw=1.8, label="Perfect fit", zorder=2)

        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("Actual Leaching Efficiency (%)",    fontsize=12)
        ax.set_ylabel("Predicted Leaching Efficiency (%)", fontsize=12)
        ax.set_title("Actual vs. Predicted — Real Test Set", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10); ax.grid(alpha=0.25)

        path = os.path.join(PLOT_DIR, "actual_vs_predicted.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Plot saved → {path}")

    def _plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> None:
        """Residual diagnostics — scatter and histogram."""
        residuals = y_true - y_pred

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Left: residuals vs fitted values
        axes[0].scatter(y_pred, residuals, alpha=0.6, color="#00B050",
                        edgecolors="k", linewidths=0.3)
        axes[0].axhline(0, color="red", linestyle="--", lw=1.5)
        axes[0].set_xlabel("Fitted values (Predicted %)", fontsize=11)
        axes[0].set_ylabel("Residual (%)", fontsize=11)
        axes[0].set_title("Residuals vs. Fitted", fontsize=12)
        axes[0].grid(alpha=0.25)

        # Right: histogram
        axes[1].hist(residuals, bins=20, color="#2E75B6",
                     edgecolor="white", alpha=0.85)
        axes[1].axvline(0, color="red", linestyle="--", lw=1.5)
        axes[1].set_xlabel("Residual (%)", fontsize=11)
        axes[1].set_ylabel("Frequency",    fontsize=11)
        axes[1].set_title("Residual Distribution", fontsize=12)
        axes[1].grid(alpha=0.25)

        fig.suptitle("Residual Diagnostics — Real Test Set",
                     fontsize=14, fontweight="bold", y=1.02)
        path = os.path.join(PLOT_DIR, "residuals.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Plot saved → {path}")

    def _plot_shap(
        self,
        model: object,
        X_test: np.ndarray,
        feature_names: list[str],
    ) -> None:
        """SHAP feature importance bar chart (tree models only)."""
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            shap_vals  = explainer.shap_values(X_test)

            plt.figure(figsize=(10, 7))
            shap.summary_plot(
                shap_vals,
                X_test,
                feature_names=feature_names,
                plot_type="bar",
                show=False,
            )
            plt.title("SHAP Feature Importance", fontsize=13, fontweight="bold")
            path = os.path.join(PLOT_DIR, "shap_summary.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"SHAP plot saved → {path}")

        except ImportError:
            logger.warning(
                "shap is not installed. "
                "Run `pip install shap` to enable SHAP plots."
            )
        except Exception as exc:
            logger.warning(f"SHAP plot skipped — {exc}")

    # ── Public ────────────────────────────────────────────────────────

    def initiate_model_evaluation(
        self,
        model_path:    str,
        X_test:        np.ndarray,
        y_test:        np.ndarray,
        feature_names: list[str] | None = None,
    ) -> dict:
        """
        Load best model, compute metrics, and save all outputs.

        Returns
        -------
        dict  →  {"r2": float, "rmse": float, "mae": float}
        """
        logger.info("━━━ Stage 4: Model Evaluation ━━━━━━━━━━━━━━━━━━━━━")
        try:
            model  = load_object(model_path)
            y_pred = model.predict(X_test)

            metrics = evaluate_model(y_test, y_pred)
            logger.info(
                f"  Test R²  = {metrics['r2']:.4f}\n"
                f"  Test RMSE = {metrics['rmse']:.4f} %\n"
                f"  Test MAE  = {metrics['mae']:.4f} %"
            )

            # Generate plots
            self._plot_actual_vs_predicted(y_test, y_pred)
            self._plot_residuals(y_test, y_pred)
            if feature_names is not None:
                self._plot_shap(model, X_test, feature_names)

            # Save metrics report
            save_json(self.config.eval_report_path, metrics)

            logger.info("━━━ Model Evaluation complete ━━━━━━━━━━━━━━━━━━━━━")
            return metrics

        except Exception as e:
            raise LeachingException(e, sys) from e
