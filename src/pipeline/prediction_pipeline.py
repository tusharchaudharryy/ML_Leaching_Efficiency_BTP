"""
prediction_pipeline.py
=======================
Inference pipeline — accepts a structured input describing one
leaching experiment, applies the saved preprocessor, runs the
saved best model, and returns the predicted efficiency (%).

Used by:
  - application.py  (Flask /predict POST endpoint)
  - ad-hoc scripts

Quick example
-------------
    from src.pipeline.prediction_pipeline import PredictPipeline, LeachingInput

    sample = LeachingInput(
        Concentration_M=1.25,
        Temperature_C=80.0,
        Time_hrs=1.0,
        SLR_gL=20.0,
        Has_Reductant=1,
        Solvent_Type="Organic Acid",
        Battery_Chemistry_Std="LCO",
        Reductant_Std="H2O2",
        Target_Metal="Co",
        RDKIT_MW=192.12,
        RDKIT_LogP=-1.248,
        RDKIT_TPSA=132.12,
        RDKIT_HBD=4,
        RDKIT_HBA=7,
        RDKIT_RotBonds=5,
        RDKIT_HeavyAtoms=13,
        RDKIT_Has_Carboxyl=1,
        RDKIT_Has_Hydroxyl=1,
        RDKIT_Has_Halogen=0,
        RDKIT_Has_Phosphorus=0,
        RDKIT_Is_Ionic=0,
        RDKIT_Morgan_FP_Density=0.04,
        EHS_Environment=2.75,
        EHS_Health=2.75,
        EHS_Safety=2.25,
        EHS_Total=2.70,
        GreenScore=81.1,
    )

    result = PredictPipeline().predict(sample)
    print(f"Predicted leaching efficiency: {result:.2f}%")
"""

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict

from src.utils.logger import get_logger
from src.utils.exception import LeachingException
from src.utils.common import load_object

logger = get_logger(__name__)

_PREPROCESSOR_PATH = os.path.join("artifacts", "models", "preprocessor.joblib")
_BEST_MODEL_PATH   = os.path.join("artifacts", "models", "best_model.joblib")


# ── Input schema ──────────────────────────────────────────────────────

@dataclass
class LeachingInput:
    """
    One row of input for a leaching efficiency prediction.

    All fields mirror the columns produced by the augmentation and
    labelling pipeline.
    """
    # Experimental conditions
    Concentration_M:          float
    Temperature_C:            float
    Time_hrs:                 float
    SLR_gL:                   float
    Has_Reductant:            int

    # Categorical identifiers
    Solvent_Type:             str
    Battery_Chemistry_Std:    str
    Reductant_Std:            str
    Target_Metal:             str

    # RDKit molecular descriptors
    RDKIT_MW:                 float
    RDKIT_LogP:               float
    RDKIT_TPSA:               float
    RDKIT_HBD:                int
    RDKIT_HBA:                int
    RDKIT_RotBonds:           int
    RDKIT_HeavyAtoms:         int
    RDKIT_Has_Carboxyl:       int
    RDKIT_Has_Hydroxyl:       int
    RDKIT_Has_Halogen:        int
    RDKIT_Has_Phosphorus:     int
    RDKIT_Is_Ionic:           int
    RDKIT_Morgan_FP_Density:  float

    # EHS sustainability scores
    EHS_Environment:          float
    EHS_Health:               float
    EHS_Safety:               float
    EHS_Total:                float
    GreenScore:               float

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the input dataclass to a single-row DataFrame."""
        return pd.DataFrame([asdict(self)])


# ── Prediction pipeline ───────────────────────────────────────────────

class PredictPipeline:
    """
    Loads saved preprocessor + best model from artifacts/ and
    produces a prediction for a LeachingInput instance.
    """

    def __init__(self):
        self.preprocessor = load_object(_PREPROCESSOR_PATH)
        self.model        = load_object(_BEST_MODEL_PATH)
        logger.info("PredictPipeline initialised — preprocessor + model loaded.")

    def predict(self, data: LeachingInput) -> float:
        """
        Predict leaching efficiency for one input.

        Parameters
        ----------
        data : LeachingInput dataclass instance

        Returns
        -------
        float : predicted efficiency in [0, 100] %
        """
        try:
            df = data.to_dataframe()

            # Replicate feature engineering from DataTransformation
            eps = 1e-6
            df["log_Time_hrs"]   = np.log1p(df["Time_hrs"].clip(lower=eps))
            df["log_SLR_gL"]     = np.log1p(df["SLR_gL"].clip(lower=eps))
            df["log_Conc"]       = np.log1p(df["Concentration_M"].clip(lower=eps))
            df["Conc_x_Temp"]    = df["Concentration_M"] * df["Temperature_C"]
            df["Temp_x_logTime"] = df["Temperature_C"]   * df["log_Time_hrs"]
            df["inv_Temp_K"]     = 1000.0 / (df["Temperature_C"].clip(lower=1.0) + 273.15)

            # Preprocess + predict
            X    = self.preprocessor.transform(df)
            pred = float(self.model.predict(X)[0])
            pred = float(np.clip(pred, 0.0, 100.0))

            logger.info(f"Prediction: {pred:.2f}%")
            return pred

        except Exception as e:
            raise LeachingException(e, sys) from e
