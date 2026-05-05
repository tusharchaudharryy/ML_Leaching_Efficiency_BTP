"""
training_pipeline.py
====================
Orchestrates all four pipeline stages in sequence:

    Stage 1 → DataIngestion
    Stage 2 → DataTransformation
    Stage 3 → ModelTrainer
    Stage 4 → ModelEvaluation

Run directly:
    python -m src.pipeline.training_pipeline

Or call from application.py:
    from src.pipeline.training_pipeline import TrainingPipeline
    metrics = TrainingPipeline().run()
"""

import sys

from src.utils.logger import get_logger
from src.utils.exception import LeachingException

from src.components.data_ingestion      import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer       import ModelTrainer
from src.components.model_evaluation    import ModelEvaluation

logger = get_logger(__name__)


class TrainingPipeline:
    """End-to-end training pipeline for the leaching ML project."""

    def run(self) -> dict:
        """
        Execute all four stages sequentially.

        Returns
        -------
        dict  →  evaluation metrics of the best model on the real test set.
                 Keys: r2, rmse, mae
        """
        logger.info("╔══════════════════════════════════════════════╗")
        logger.info("║    ORGANIC ACID LEACHING ML PIPELINE         ║")
        logger.info("║    Predicting Metal Leaching Efficiency (%)  ║")
        logger.info("╚══════════════════════════════════════════════╝")

        try:
            # ── Stage 1: Ingestion ────────────────────────────────────
            logger.info("► Stage 1/4 — Data Ingestion")
            train_path, test_path = DataIngestion().initiate_data_ingestion()

            # ── Stage 2: Transformation ───────────────────────────────
            logger.info("► Stage 2/4 — Data Transformation")
            X_train, y_train, X_test, y_test, preprocessor_path = (
                DataTransformation().initiate_data_transformation(
                    train_path, test_path
                )
            )

            # ── Stage 3: Training ─────────────────────────────────────
            logger.info("► Stage 3/4 — Model Training")
            best_model, comparison_df, best_model_path = (
                ModelTrainer().initiate_model_training(
                    X_train, y_train, X_test, y_test
                )
            )

            # ── Stage 4: Evaluation ───────────────────────────────────
            logger.info("► Stage 4/4 — Model Evaluation")
            metrics = ModelEvaluation().initiate_model_evaluation(
                model_path=best_model_path,
                X_test=X_test,
                y_test=y_test,
            )

            logger.info("╔══════════════════════════════════════════════╗")
            logger.info(f"║  R²={metrics['r2']:.4f}  RMSE={metrics['rmse']:.2f}%  MAE={metrics['mae']:.2f}%  ║")
            logger.info("║           Pipeline complete ✔                ║")
            logger.info("╚══════════════════════════════════════════════╝")

            return metrics

        except Exception as e:
            logger.error("Pipeline failed!", exc_info=True)
            raise LeachingException(e, sys) from e


def main():
    """Entry point for `train` CLI command (defined in setup.py)."""
    pipeline = TrainingPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
