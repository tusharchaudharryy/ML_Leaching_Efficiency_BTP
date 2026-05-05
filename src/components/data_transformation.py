"""
data_transformation.py
=======================
Stage 2 — Feature engineering, imputation, encoding, and scaling.

Pipeline steps
--------------
1. Drop metadata / non-feature columns
2. Engineer 6 domain-informed features (log transforms, Arrhenius term)
3. Impute numeric NaN values (median strategy)
4. One-hot encode categorical features
5. StandardScale all numeric columns

The fitted ``ColumnTransformer`` is saved to
``artifacts/models/preprocessor.joblib`` for reuse during inference.

Engineered features (physical motivation)
-----------------------------------------
log_Time_hrs   : leaching kinetics follow log-time diminishing returns
log_SLR_gL     : solid-liquid ratio effect is sub-linear
log_Conc       : acid dissociation contribution is ~log(concentration)
Conc_x_Temp    : synergistic interaction between acid strength and heat
Temp_x_logTime : Arrhenius temperature × log-time interaction term
inv_Temp_K     : 1000/(T + 273.15) — direct Arrhenius activation term
"""

import os
import sys

import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.utils.logger import get_logger
from src.utils.exception import LeachingException
from src.utils.common import save_object

logger = get_logger(__name__)


# ── Constants ─────────────────────────────────────────────────────────

TARGET_COL = "Efficiency_pct"

NUMERIC_FEATURES = [
    # Experimental conditions
    "Concentration_M", "Temperature_C", "Time_hrs", "SLR_gL", "Has_Reductant",
    # RDKit molecular descriptors
    "RDKIT_MW", "RDKIT_LogP", "RDKIT_TPSA",
    "RDKIT_HBD", "RDKIT_HBA", "RDKIT_RotBonds",
    "RDKIT_HeavyAtoms", "RDKIT_Has_Carboxyl", "RDKIT_Has_Hydroxyl",
    "RDKIT_Has_Halogen", "RDKIT_Has_Phosphorus",
    "RDKIT_Is_Ionic", "RDKIT_Morgan_FP_Density",
    # EHS scores
    "EHS_Environment", "EHS_Health", "EHS_Safety", "EHS_Total", "GreenScore",
]

CATEGORICAL_FEATURES = [
    "Solvent_Type",
    "Battery_Chemistry_Std",
    "Reductant_Std",
    "Target_Metal",
]

# Columns dropped before modelling (metadata, near-zero variance, redundant)
DROP_COLS = [
    "DOI", "Title", "Solvent_Name", "SMILES",
    "Source", "Augmentation_Method", "EHS_Label",
    "RDKIT_AromaticRings", "RDKIT_RingCount",
]

# Engineered feature names (added by _engineer_features)
ENGINEERED_FEATURES = [
    "log_Time_hrs", "log_SLR_gL", "log_Conc",
    "Conc_x_Temp", "Temp_x_logTime", "inv_Temp_K",
]


# ── Config ────────────────────────────────────────────────────────────

@dataclass
class DataTransformationConfig:
    preprocessor_path: str = os.path.join(
        "artifacts", "models", "preprocessor.joblib"
    )


# ── Component ─────────────────────────────────────────────────────────

class DataTransformation:
    """Builds and applies the full feature preprocessing pipeline."""

    def __init__(
        self,
        config: DataTransformationConfig = DataTransformationConfig(),
    ):
        self.config = config

    # ── Private ───────────────────────────────────────────────────────

    @staticmethod
    def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add 6 physics-motivated engineered features."""
        df = df.copy()
        eps = 1e-6  # avoid log(0)

        df["log_Time_hrs"]   = np.log1p(df["Time_hrs"].clip(lower=eps))
        df["log_SLR_gL"]     = np.log1p(df["SLR_gL"].clip(lower=eps))
        df["log_Conc"]       = np.log1p(df["Concentration_M"].clip(lower=eps))
        df["Conc_x_Temp"]    = df["Concentration_M"] * df["Temperature_C"]
        df["Temp_x_logTime"] = df["Temperature_C"]   * df["log_Time_hrs"]
        df["inv_Temp_K"]     = 1000.0 / (df["Temperature_C"].clip(lower=1.0) + 273.15)

        return df

    @staticmethod
    def _build_preprocessor(
        numeric_cols: list[str],
        cat_cols: list[str],
    ) -> ColumnTransformer:
        """
        Build a ColumnTransformer:
          - numeric  : median imputation → StandardScaler
          - categorical : most-frequent imputation → OneHotEncoder
        """
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ])
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline,       numeric_cols),
                ("cat", categorical_pipeline,   cat_cols),
            ],
            remainder="drop",
        )

    # ── Public ────────────────────────────────────────────────────────

    def initiate_data_transformation(
        self,
        train_path: str,
        test_path:  str,
    ) -> tuple:
        """
        Run the full transformation on train and test CSVs.

        Returns
        -------
        X_train, y_train, X_test, y_test, preprocessor_path
        """
        logger.info("━━━ Stage 2: Data Transformation ━━━━━━━━━━━━━━━━━━")
        try:
            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)
            logger.info(f"Train: {train_df.shape}  |  Test: {test_df.shape}")

            # Feature engineering (same for train and test)
            train_df = self._engineer_features(train_df)
            test_df  = self._engineer_features(test_df)

            # Resolve which columns actually exist in the data
            all_numeric = [
                c for c in NUMERIC_FEATURES + ENGINEERED_FEATURES
                if c in train_df.columns
            ]
            all_cat = [c for c in CATEGORICAL_FEATURES if c in train_df.columns]

            logger.info(f"Numeric features    : {len(all_numeric)}")
            logger.info(f"Categorical features: {len(all_cat)}")
            logger.info(f"Features total      : {len(all_numeric) + len(all_cat)}")

            # Separate target
            y_train = train_df[TARGET_COL].values.astype(float)
            y_test  = test_df[TARGET_COL].values.astype(float)

            # Build preprocessor — fit on TRAIN only, transform both
            preprocessor = self._build_preprocessor(all_numeric, all_cat)
            X_train = preprocessor.fit_transform(train_df)
            X_test  = preprocessor.transform(test_df)

            logger.info(f"X_train: {X_train.shape}  |  X_test: {X_test.shape}")

            # Save preprocessor artifact
            save_object(self.config.preprocessor_path, preprocessor)

            logger.info("━━━ Data Transformation complete ━━━━━━━━━━━━━━━━━━")
            return X_train, y_train, X_test, y_test, self.config.preprocessor_path

        except Exception as e:
            raise LeachingException(e, sys) from e
