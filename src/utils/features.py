"""
features.py
===========
Single source of truth for feature engineering.

Imported by both DataTransformation (training) and PredictPipeline
(inference) so that the computed columns are always identical.
"""

import numpy as np
import pandas as pd

_EPS = 1e-6  # guard against log(0)

ENGINEERED_FEATURE_NAMES = [
    "log_Time_hrs",
    "log_SLR_gL",
    "log_Conc",
    "Conc_x_Temp",
    "Temp_x_logTime",
    "inv_Temp_K",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 6 physics-motivated engineered columns and return a copy.

    Requires columns: Time_hrs, SLR_gL, Concentration_M, Temperature_C
    """
    df = df.copy()

    df["log_Time_hrs"]   = np.log1p(df["Time_hrs"].clip(lower=_EPS))
    df["log_SLR_gL"]     = np.log1p(df["SLR_gL"].clip(lower=_EPS))
    df["log_Conc"]       = np.log1p(df["Concentration_M"].clip(lower=_EPS))
    df["Conc_x_Temp"]    = df["Concentration_M"] * df["Temperature_C"]
    df["Temp_x_logTime"] = df["Temperature_C"] * df["log_Time_hrs"]
    # 1000/(T+273.15) is the Arrhenius inverse-temperature term
    df["inv_Temp_K"]     = 1000.0 / (df["Temperature_C"].clip(lower=1.0) + 273.15)

    return df
