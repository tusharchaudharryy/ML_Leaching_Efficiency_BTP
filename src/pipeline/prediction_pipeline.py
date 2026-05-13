"""
prediction_pipeline.py
=======================
Inference pipeline -- accepts a structured input describing one
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
import pathlib

import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict

from src.utils.logger import get_logger
from src.utils.exception import LeachingException
from src.utils.common import load_object
from src.utils.features import engineer_features

logger = get_logger(__name__)

_PROJECT_ROOT      = pathlib.Path(__file__).parents[2]
_PREPROCESSOR_PATH = str(_PROJECT_ROOT / "artifacts" / "models" / "preprocessor.joblib")
_BEST_MODEL_PATH   = str(_PROJECT_ROOT / "artifacts" / "models" / "best_model.joblib")


# -- Input schema -------------------------------------------------------------

@dataclass
class LeachingInput:
    """
    One row of input for a leaching efficiency prediction.

    All fields mirror the columns produced by the augmentation and
    labelling pipeline.  __post_init__ validates physical bounds.
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

    def __post_init__(self) -> None:
        """Validate physical plausibility of inputs at the system boundary."""
        if self.Concentration_M <= 0:
            raise ValueError(
                f"Concentration_M must be > 0, got {self.Concentration_M}"
            )
        if self.Temperature_C < 0:
            raise ValueError(
                f"Temperature_C must be >= 0 (Celsius), got {self.Temperature_C}"
            )
        if self.Time_hrs <= 0:
            raise ValueError(f"Time_hrs must be > 0, got {self.Time_hrs}")
        if self.SLR_gL <= 0:
            raise ValueError(f"SLR_gL must be > 0, got {self.SLR_gL}")
        if self.Has_Reductant not in (0, 1):
            raise ValueError(
                f"Has_Reductant must be 0 or 1, got {self.Has_Reductant}"
            )
        for flag_name in (
            "RDKIT_Has_Carboxyl", "RDKIT_Has_Hydroxyl",
            "RDKIT_Has_Halogen", "RDKIT_Has_Phosphorus", "RDKIT_Is_Ionic",
        ):
            val = getattr(self, flag_name)
            if val not in (0, 1):
                raise ValueError(f"{flag_name} must be 0 or 1, got {val}")
        if not (0 <= self.GreenScore <= 100):
            raise ValueError(
                f"GreenScore must be in [0, 100], got {self.GreenScore}"
            )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the input dataclass to a single-row DataFrame."""
        return pd.DataFrame([asdict(self)])


# -- Prediction pipeline ------------------------------------------------------

class PredictPipeline:
    """
    Loads saved preprocessor + best model from artifacts/ and
    produces a prediction for a LeachingInput instance.

    Raises FileNotFoundError with a helpful message if the training
    pipeline has not been run yet.
    """

    def __init__(self):
        missing = [
            p for p in (_PREPROCESSOR_PATH, _BEST_MODEL_PATH)
            if not os.path.exists(p)
        ]
        if missing:
            raise FileNotFoundError(
                "Required model artifacts are missing:\n"
                + "\n".join(f"  {p}" for p in missing)
                + "\nRun the training pipeline first:\n"
                "  python -m src.pipeline.training_pipeline"
            )
        self.preprocessor = load_object(_PREPROCESSOR_PATH)
        self.model        = load_object(_BEST_MODEL_PATH)
        logger.info("PredictPipeline initialised -- preprocessor + model loaded.")

    def predict(self, data: LeachingInput) -> float:
        """
        Predict leaching efficiency for one input.

        Parameters
        ----------
        data : LeachingInput dataclass instance

        Returns
        -------
        float : predicted efficiency clipped to [0, 100] %
        """
        try:
            df = data.to_dataframe()

            # Apply the same feature engineering used during training
            df = engineer_features(df)

            X    = self.preprocessor.transform(df)
            pred = float(self.model.predict(X)[0])
            pred = float(np.clip(pred, 0.0, 100.0))

            logger.info(f"Prediction: {pred:.2f}%")
            return pred

        except Exception as e:
            raise LeachingException(e, sys) from e
