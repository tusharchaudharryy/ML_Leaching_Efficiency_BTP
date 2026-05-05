"""
data_ingestion.py
=================
Stage 1 — Load the augmented leaching dataset and produce a
**source-aware train / test split**.

Design rule (critical)
----------------------
The ``Source`` column controls the split:

    train  ← Synthetic + Derived rows  (~10 500 rows)
    test   ← Real rows ONLY            (~123 rows)

We never evaluate on synthetic data. This ensures that reported
metrics reflect true generalisation to real experimental conditions.

Outputs
-------
artifacts/train.csv
artifacts/test.csv
"""

import os
import sys

import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger
from src.utils.exception import LeachingException

logger = get_logger(__name__)


# ── Config ────────────────────────────────────────────────────────────

@dataclass
class DataIngestionConfig:
    raw_data_path:   str   = os.path.join("data", "raw_dataset.csv")
    train_data_path: str   = os.path.join("artifacts", "train.csv")
    test_data_path:  str   = os.path.join("artifacts", "test.csv")
    test_size:       float = 0.15    # fallback if Source col is missing
    random_state:    int   = 42


# ── Component ─────────────────────────────────────────────────────────

class DataIngestion:
    """
    Loads the dataset from ``data/raw_dataset.csv`` and saves
    train/test CSV files to ``artifacts/``.
    """

    def __init__(self, config: DataIngestionConfig = DataIngestionConfig()):
        self.config = config

    def initiate_data_ingestion(self) -> tuple[str, str]:
        """
        Run data ingestion.

        Returns
        -------
        train_data_path : str
        test_data_path  : str
        """
        logger.info("━━━ Stage 1: Data Ingestion ━━━━━━━━━━━━━━━━━━━━━━━")
        try:
            # Load raw dataset
            df = pd.read_csv(self.config.raw_data_path)
            logger.info(
                f"Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns"
            )

            os.makedirs("artifacts", exist_ok=True)

            # Source-aware split (preferred)
            if "Source" in df.columns:
                test_df  = df[df["Source"] == "Real"].copy()
                train_df = df[df["Source"] != "Real"].copy()
                logger.info(
                    f"Source-aware split  →  "
                    f"train={len(train_df):,}  |  test={len(test_df):,} (Real only)"
                )
            else:
                logger.warning(
                    "Column 'Source' not found — using random "
                    f"{self.config.test_size:.0%} split as fallback."
                )
                train_df, test_df = train_test_split(
                    df,
                    test_size=self.config.test_size,
                    random_state=self.config.random_state,
                )

            # Persist splits
            train_df.to_csv(self.config.train_data_path, index=False)
            test_df.to_csv(self.config.test_data_path,  index=False)
            logger.info(f"Train saved → {self.config.train_data_path}")
            logger.info(f"Test  saved → {self.config.test_data_path}")
            logger.info("━━━ Data Ingestion complete ━━━━━━━━━━━━━━━━━━━━━━━")

            return self.config.train_data_path, self.config.test_data_path

        except Exception as e:
            raise LeachingException(e, sys) from e


# ── Standalone run ────────────────────────────────────────────────────

if __name__ == "__main__":
    obj = DataIngestion()
    train, test = obj.initiate_data_ingestion()
    print(f"\nTrain path : {train}")
    print(f"Test  path : {test}")
