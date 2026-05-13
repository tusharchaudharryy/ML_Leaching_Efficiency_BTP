"""
data_transformation.py
=======================
Stage 2 -- Feature engineering, imputation, encoding, and scaling.

Pipeline steps
--------------
1. Drop metadata / non-feature columns
2. Engineer 6 domain-informed features (log transforms, Arrhenius term)
3. Impute numeric NaN values (median strategy)
4. One-hot encode categorical features
5. StandardScale all numeric columns

The fitted ``ColumnTransformer`` is saved to
``artifacts/models/preprocessor.joblib`` for reuse during inference.

See src/utils/features.py for the engineered feature definitions.
"""

import os
import sys

import pandas as pd
from dataclasses import dataclass

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.utils.logger import get_logger
from src.utils.exception import LeachingException
from src.utils.common import save_object
from src.utils.features import engineer_features, ENGINEERED_FEATURE_NAMES

logger = get_logger(__name__)


# -- Constants ----------------------------------------------------------------

TARGET_COL = "Efficiency_pct"

NUMERIC_FEATURES = [
    # Experimental process conditions
    "Concentration_M", "Temperature_C", "Time_hrs", "SLR_gL", "Has_Reductant",
    # RDKit molecular descriptors (near-constant binary flags excluded -- see DROP_COLS)
    "RDKIT_MW", "RDKIT_LogP", "RDKIT_TPSA",
    "RDKIT_HBD", "RDKIT_HBA", "RDKIT_RotBonds",
    "RDKIT_HeavyAtoms", "RDKIT_Morgan_FP_Density",
    # NOTE: EHS scores (EHS_Environment, EHS_Health, EHS_Safety, EHS_Total, GreenScore)
    # are excluded per supervisor instruction. They have near-zero correlation with
    # Efficiency_pct (|r| < 0.08) and are sustainability metrics, not process predictors.
]

CATEGORICAL_FEATURES = [
    "Solvent_Type",
    "Battery_Chemistry_Std",
    "Reductant_Std",
    "Target_Metal",
]

# Columns dropped before modelling.
# Near-constant binary flags (>95% same value, verified on full dataset):
#   RDKIT_Has_Carboxyl  97.8% = 1
#   RDKIT_Has_Hydroxyl  99.5% = 1
#   RDKIT_Has_Halogen   99.7% = 0
#   RDKIT_Has_Phosphorus 99.4% = 0
#   RDKIT_Is_Ionic      99.8% = 0
# These contribute almost no discriminative signal and add noise.
DROP_COLS = [
    "DOI", "Title", "Solvent_Name", "SMILES",
    "Source", "Augmentation_Method",
    # EHS_Label: classification column -- supervisor confirmed target is regression only
    "EHS_Label",
    # EHS sustainability scores: excluded per supervisor instruction (|r| < 0.08 with target)
    "EHS_Environment", "EHS_Health", "EHS_Safety", "EHS_Total", "GreenScore",
    # Near-zero variance
    "RDKIT_AromaticRings", "RDKIT_RingCount",
    # Near-constant binary flags (>95% same value)
    "RDKIT_Has_Carboxyl", "RDKIT_Has_Hydroxyl",
    "RDKIT_Has_Halogen", "RDKIT_Has_Phosphorus", "RDKIT_Is_Ionic",
]


# -- Config -------------------------------------------------------------------

@dataclass
class DataTransformationConfig:
    preprocessor_path: str = os.path.join(
        "artifacts", "models", "preprocessor.joblib"
    )


# -- Component ----------------------------------------------------------------

class DataTransformation:
    """Builds and applies the full feature preprocessing pipeline."""

    def __init__(
        self,
        config: DataTransformationConfig = None,
    ):
        self.config = config or DataTransformationConfig()

    @staticmethod
    def _build_preprocessor(
        numeric_cols: list[str],
        cat_cols: list[str],
    ) -> ColumnTransformer:
        """
        Build a ColumnTransformer:
          - numeric     : median imputation then StandardScaler
          - categorical : most-frequent imputation then OneHotEncoder
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
                ("num", numeric_pipeline,     numeric_cols),
                ("cat", categorical_pipeline, cat_cols),
            ],
            remainder="drop",
        )

    def initiate_data_transformation(
        self,
        train_path: str,
        test_path:  str,
    ) -> tuple:
        """
        Run the full transformation on train and test CSVs.

        Returns
        -------
        X_train, y_train, X_test, y_test, preprocessor_path, feature_names
        """
        logger.info("--- Stage 2: Data Transformation ---")
        try:
            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)
            logger.info(f"Train: {train_df.shape}  |  Test: {test_df.shape}")

            # Reductant_Std has structural nulls for Has_Reductant=0 rows.
            # Fill with "None" before imputation so the most-frequent imputer
            # does not incorrectly substitute "H2O2" for no-reductant rows.
            for df_ in (train_df, test_df):
                if "Reductant_Std" in df_.columns:
                    df_["Reductant_Std"] = df_["Reductant_Std"].fillna("None")

            # Feature engineering using the shared function (same for train and test)
            train_df = engineer_features(train_df)
            test_df  = engineer_features(test_df)

            # Resolve which columns actually exist in the data
            all_numeric = [
                c for c in NUMERIC_FEATURES + ENGINEERED_FEATURE_NAMES
                if c in train_df.columns
            ]
            all_cat = [c for c in CATEGORICAL_FEATURES if c in train_df.columns]

            logger.info(f"Numeric features    : {len(all_numeric)}")
            logger.info(f"Categorical features: {len(all_cat)}")
            logger.info(f"Features total      : {len(all_numeric) + len(all_cat)}")

            # Separate target
            y_train = train_df[TARGET_COL].values.astype(float)
            y_test  = test_df[TARGET_COL].values.astype(float)

            # Build preprocessor -- fit on TRAIN only, transform both
            preprocessor = self._build_preprocessor(all_numeric, all_cat)
            X_train = preprocessor.fit_transform(train_df)
            X_test  = preprocessor.transform(test_df)

            logger.info(f"X_train: {X_train.shape}  |  X_test: {X_test.shape}")

            # Extract feature names from the fitted preprocessor
            feature_names = list(preprocessor.get_feature_names_out())

            # Save preprocessor artifact
            save_object(self.config.preprocessor_path, preprocessor)

            logger.info("--- Data Transformation complete ---")
            return (
                X_train, y_train,
                X_test,  y_test,
                self.config.preprocessor_path,
                feature_names,
            )

        except Exception as e:
            raise LeachingException(e, sys) from e
