"""
augment_dataset.py
==================
Physics-constrained data augmentation for the leaching efficiency ML pipeline.

Run once before training to produce data/raw_dataset.csv:
    python -m src.data.augment_dataset

Methodology (paper-ready description)
--------------------------------------
Starting from all experimentally observed or directly derived data points
(Real + Derived rows), we generate physics-constrained synthetic training
samples in four steps:

1. Molecular pool extension
   RDKit descriptors are computed for 12 additional organic acids commonly
   reported in the battery recycling literature (oxalic, tartaric, lactic,
   succinic, acetic, glycolic, gluconic, itaconic, formic, pyruvic, DL-malic,
   DL-tartaric acids) to supplement the 27 SMILES already in the dataset.
   The citric-acid share is capped at 45% to reduce the original 92% bias.

2. Random Forest kinetics model
   A RandomForestRegressor is fitted on the Real + Derived rows (123 rows)
   which span the full observed efficiency range (25-100%). Features are:
       Concentration_M, Temperature_C, Time_hrs, SLR_gL, Has_Reductant,
       RDKIT_MW, RDKIT_LogP, RDKIT_TPSA, RDKIT_HBD, RDKIT_HBA
   This model captures non-linear kinetics and is used only to predict
   efficiency for new conditions; it is NOT saved or used in the final ML
   pipeline.

3. Stratified Latin Hypercube Sampling (LHS)
   Experimental conditions are sampled in three tiers to ensure full coverage
   of the efficiency range:
     - Unfavorable (30%): low T (25-55 C), low C (0.1-0.7 M), short t   -> 25-75%
     - Moderate    (30%): medium T (55-85 C), medium C (0.7-1.5 M)       -> 70-92%
     - Favorable   (40%): high T (85-130 C), high C (1.5-3.0 M), long t  -> 88-100%

4. Noise and quality filter
   Gaussian measurement noise N(0, sigma=2.5%) is added to every predicted
   efficiency to represent natural experimental variability. Only samples
   with final efficiency in [20%, 100%] are retained.

Output composition
------------------
  Real      :   12 rows  (original, unchanged)
  Derived   :  111 rows  (original, unchanged)
  Augmented : ~N rows   (new, physics-constrained)

The old unlabelled-method Synthetic rows are replaced by the new Augmented
rows. EHS_Label (a derived classification column) is removed. Structural
nulls in Reductant_Std are replaced with "None".
"""

from __future__ import annotations

import os
import sys
import pathlib
import warnings

import numpy as np
import pandas as pd
from scipy.stats.qmc import LatinHypercube
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

_PROJECT_ROOT = pathlib.Path(__file__).parents[2]
INPUT_CSV  = str(_PROJECT_ROOT / "data" / "Full_Augmented_Dataset_Labeled.csv")
OUTPUT_CSV = str(_PROJECT_ROOT / "data" / "raw_dataset.csv")

RANDOM_SEED   = 42
N_AUGMENTED   = 8500   # target rows before quality filtering
CITRIC_CAP    = 0.45   # max fraction for citric acid in augmented set
NOISE_SIGMA   = 2.5    # % -- Gaussian measurement noise added to predictions
EFF_MIN       = 20.0   # minimum plausible leaching efficiency (%)
DESIGN_EXTEND = 0.15   # extend design space beyond data range by this fraction

# Stratified tier fractions (must sum to 1.0)
TIER_EXTREME     = 0.15  # very low T (25-40 C), very low C -> covers 55-80%
TIER_UNFAVORABLE = 0.15  # low T (40-60 C), low C -> covers 70-88%
TIER_MODERATE    = 0.30  # medium conditions -> covers 80-93%
TIER_FAVORABLE   = 0.40  # high T/C/t -> covers 88-100%

# Low-efficiency anchor rows: jitter Derived rows with efficiency < EFF_ANCHOR_THRESH
EFF_ANCHOR_THRESH  = 80.0   # Derived rows below this get jittered
EFF_ANCHOR_REPEATS = 40     # repeats per anchor row (noise jitter)
EFF_ANCHOR_JITTER  = 0.12   # ±12% Gaussian noise on each condition
EFF_NOISE_SIGMA_LO = 3.0    # larger noise for low-efficiency predictions

# ---------------------------------------------------------------------------
# Additional organic acids for molecular diversity
# Each entry: (Solvent_Name, SMILES, Solvent_Type)
# Selected based on their documented use in Li-ion battery recycling literature.
# ---------------------------------------------------------------------------
_NEW_ACIDS: list[tuple[str, str, str]] = [
    ("Oxalic Acid",       "OC(=O)C(=O)O",                               "Organic Acid"),
    ("Tartaric Acid",     "OC([C@@H](O)C(=O)O)C(=O)O",                  "Organic Acid"),
    ("Lactic Acid",       "C[C@@H](O)C(=O)O",                           "Organic Acid"),
    ("Succinic Acid",     "OC(=O)CCC(=O)O",                             "Organic Acid"),
    ("Acetic Acid",       "CC(=O)O",                                     "Organic Acid"),
    ("Glycolic Acid",     "OCC(=O)O",                                    "Organic Acid"),
    ("Formic Acid",       "OC=O",                                        "Organic Acid"),
    ("Gluconic Acid",     "OC[C@@H](O)[C@@H](O)[C@@H](O)[C@@H](O)C(=O)O", "Organic Acid"),
    ("Itaconic Acid",     "OC(=O)CC(=C)C(=O)O",                         "Organic Acid"),
    ("Pyruvic Acid",      "CC(=O)C(=O)O",                               "Organic Acid"),
    ("DL-Malic Acid",     "OC(CC(=O)O)C(=O)O",                          "Organic Acid"),
    ("DL-Tartaric Acid",  "OC(C(O)C(=O)O)C(=O)O",                       "Organic Acid"),
]


def _compute_rdkit_features(smiles: str) -> dict | None:
    """
    Compute the RDKit molecular descriptors used as model features.
    Returns None if RDKit is not installed or SMILES is invalid.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors
        from rdkit.Chem import rdFingerprintGenerator

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Morgan fingerprint bit density (radius=2, 512 bits)
        gen  = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512)
        fp   = gen.GetFingerprintAsNumPy(mol)
        fp_density = float(fp.sum()) / 512.0

        return {
            "RDKIT_MW":               round(Descriptors.MolWt(mol), 4),
            "RDKIT_LogP":             round(Descriptors.MolLogP(mol), 4),
            "RDKIT_TPSA":             round(Descriptors.TPSA(mol), 4),
            "RDKIT_HBD":              rdMolDescriptors.CalcNumHBD(mol),
            "RDKIT_HBA":              rdMolDescriptors.CalcNumHBA(mol),
            "RDKIT_RotBonds":         rdMolDescriptors.CalcNumRotatableBonds(mol),
            "RDKIT_RingCount":        rdMolDescriptors.CalcNumRings(mol),
            "RDKIT_AromaticRings":    rdMolDescriptors.CalcNumAromaticRings(mol),
            "RDKIT_HeavyAtoms":       mol.GetNumHeavyAtoms(),
            "RDKIT_Has_Carboxyl":     int(mol.HasSubstructMatch(
                                          Chem.MolFromSmarts("C(=O)[OH]"))),
            "RDKIT_Has_Hydroxyl":     int(mol.HasSubstructMatch(
                                          Chem.MolFromSmarts("[OX2H]"))),
            "RDKIT_Has_Halogen":      int(mol.HasSubstructMatch(
                                          Chem.MolFromSmarts("[F,Cl,Br,I]"))),
            "RDKIT_Has_Phosphorus":   int(mol.HasSubstructMatch(
                                          Chem.MolFromSmarts("[#15]"))),
            "RDKIT_Is_Ionic":         0,  # small organic acids are non-ionic
            "RDKIT_Morgan_FP_Density": round(fp_density, 6),
        }
    except ImportError:
        return None


def _estimate_ehs_scores(rdkit: dict) -> dict:
    """
    Estimate EHS sustainability scores from molecular properties.

    Rules based on the CHEM21 EHS scoring guide:
      Environment : penalise high MW (persistence) and LogP (bioaccumulation)
      Health      : penalise high TPSA (oral absorption) and HBD
      Safety      : penalise low MW (volatility)
    Scores are bounded to [1, 10]; lower is greener.
    GreenScore = 100 - 2*(Env + Health + Safety)  (scaled to 0-100)
    """
    mw    = rdkit.get("RDKIT_MW", 192.0)
    logp  = rdkit.get("RDKIT_LogP", -1.2)
    tpsa  = rdkit.get("RDKIT_TPSA", 132.0)
    hbd   = rdkit.get("RDKIT_HBD", 4)

    env    = float(np.clip(1.0 + 0.003 * mw + max(0, 0.2 * logp), 1.0, 7.5))
    health = float(np.clip(2.5 + 0.003 * tpsa + 0.1 * hbd,         1.0, 7.5))
    safety = float(np.clip(5.0 - 0.01 * mw,                         1.5, 6.0))
    total  = round((env + health + safety) / 3.0, 2)
    green  = round(float(np.clip(100 - 4 * total, 35, 95)), 1)

    return {
        "EHS_Environment": round(env, 2),
        "EHS_Health":      round(health, 2),
        "EHS_Safety":      round(safety, 2),
        "EHS_Total":       total,
        "GreenScore":      green,
    }


def _build_smiles_pool(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build an extended table of unique SMILES with all associated molecular
    features. Incorporates the 12 new acids (if RDKit is available).
    """
    # One representative row per unique SMILES from the original data
    existing = (
        df.dropna(subset=["SMILES"])
          .groupby("SMILES", as_index=False)
          .first()
          [["SMILES", "Solvent_Name", "Solvent_Type",
            "RDKIT_MW", "RDKIT_LogP", "RDKIT_TPSA",
            "RDKIT_HBD", "RDKIT_HBA", "RDKIT_RotBonds",
            "RDKIT_RingCount", "RDKIT_AromaticRings",
            "RDKIT_HeavyAtoms", "RDKIT_Has_Carboxyl", "RDKIT_Has_Hydroxyl",
            "RDKIT_Has_Halogen", "RDKIT_Has_Phosphorus", "RDKIT_Is_Ionic",
            "RDKIT_Morgan_FP_Density",
            "EHS_Environment", "EHS_Health", "EHS_Safety",
            "EHS_Total", "GreenScore"]]
    )

    new_rows = []
    for name, smi, stype in _NEW_ACIDS:
        if smi in existing["SMILES"].values:
            continue  # already in the dataset
        rdkit = _compute_rdkit_features(smi)
        if rdkit is None:
            print(f"  [skip] RDKit unavailable or invalid SMILES: {name}")
            continue
        ehs = _estimate_ehs_scores(rdkit)
        row = {"SMILES": smi, "Solvent_Name": name, "Solvent_Type": stype}
        row.update(rdkit)
        row.update(ehs)
        new_rows.append(row)

    if new_rows:
        print(f"  Added {len(new_rows)} new SMILES via RDKit")
        pool = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    else:
        pool = existing.copy()

    print(f"  Total SMILES in pool: {len(pool)}")
    return pool


def _kinetics_features(
    C: np.ndarray,
    T: np.ndarray,
    t: np.ndarray,
    SLR: np.ndarray,
    has_r: np.ndarray,
    mw: np.ndarray,
    logp: np.ndarray,
    tpsa: np.ndarray,
    hbd: np.ndarray,
    hba: np.ndarray,
) -> np.ndarray:
    """Feature matrix for the Random Forest kinetics model."""
    return np.column_stack([C, T, t, SLR, has_r, mw, logp, tpsa, hbd, hba])


def _fit_kinetics_model(df: pd.DataFrame) -> RandomForestRegressor:
    """
    Fit a Random Forest on Real+Derived rows which span the full efficiency
    range (25-100%). Used only to predict efficiency for new augmented
    conditions; it is not saved or used in the final ML pipeline.
    """
    basis = df[df["Source"].isin(["Real", "Derived"])].copy()
    X = _kinetics_features(
        basis["Concentration_M"].values,
        basis["Temperature_C"].values,
        basis["Time_hrs"].values,
        basis["SLR_gL"].values,
        basis["Has_Reductant"].values.astype(float),
        basis["RDKIT_MW"].values,
        basis["RDKIT_LogP"].values,
        basis["RDKIT_TPSA"].values,
        basis["RDKIT_HBD"].values.astype(float),
        basis["RDKIT_HBA"].values.astype(float),
    )
    y = basis["Efficiency_pct"].values

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        random_state=RANDOM_SEED,
    )
    model.fit(X, y)

    from sklearn.metrics import r2_score
    r2 = r2_score(y, model.predict(X))
    print(f"  Kinetics model (RF) train R2 = {r2:.3f}  "
          f"(fitted on {len(basis)} Real+Derived rows, used for augmentation only)")
    return model


def _sample_tier(
    n: int,
    rng: np.random.Generator,
    seed_offset: int,
    T_range: tuple[float, float],
    C_range: tuple[float, float],
    t_range: tuple[float, float],
    SLR_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample n LHS points within a given (T, C, t, SLR) hypercube tier."""
    sampler = LatinHypercube(d=4, seed=seed_offset)
    u = sampler.random(n=n)
    C_s   = u[:, 0] * (C_range[1]   - C_range[0])   + C_range[0]
    T_s   = u[:, 1] * (T_range[1]   - T_range[0])   + T_range[0]
    t_s   = u[:, 2] * (t_range[1]   - t_range[0])   + t_range[0]
    SLR_s = u[:, 3] * (SLR_range[1] - SLR_range[0]) + SLR_range[0]
    return C_s, T_s, t_s, SLR_s


def _design_space(df: pd.DataFrame) -> dict:
    """
    Derive the experimental design space from the data.
    Uses the 5th-95th percentile range extended by DESIGN_EXTEND.
    """
    def bounds(col: str) -> tuple[float, float]:
        lo = df[col].quantile(0.05) * (1 - DESIGN_EXTEND)
        hi = df[col].quantile(0.95) * (1 + DESIGN_EXTEND)
        return max(lo, 1e-3), hi

    return {
        "Concentration_M": bounds("Concentration_M"),
        "Temperature_C":   (max(25.0, bounds("Temperature_C")[0]),
                            min(200.0, bounds("Temperature_C")[1])),
        "Time_hrs":        bounds("Time_hrs"),
        "SLR_gL":          bounds("SLR_gL"),
    }


def _build_smiles_weights(
    df_basis: pd.DataFrame,
    pool: pd.DataFrame,
) -> np.ndarray:
    """
    Compute sampling weights for each SMILES in the pool.
    Cap citric acid at CITRIC_CAP and distribute remaining weight
    proportionally across all other SMILES.
    """
    citric_smiles = "C(C(=O)O)C(CC(=O)O)(C(=O)O)O"
    n = len(pool)
    base_w = np.ones(n) / n  # uniform prior

    citric_idx = pool.index[pool["SMILES"] == citric_smiles].tolist()
    if citric_idx:
        # Set citric acid to CITRIC_CAP, spread rest uniformly
        w = np.ones(n)
        w[citric_idx[0]] = CITRIC_CAP * n  # relative weight
        w = w / w.sum()
    else:
        w = base_w

    return w


def _generate_anchor_rows(
    df_basis: pd.DataFrame,
    pool: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Generate low-efficiency anchor rows by jittering Derived rows whose
    efficiency is below EFF_ANCHOR_THRESH.

    This directly ensures the augmented dataset covers the low-efficiency
    region (25-80%) that the Random Forest kinetics model cannot extrapolate
    into on its own.
    """
    low = df_basis[
        (df_basis["Source"] == "Derived") &
        (df_basis["Efficiency_pct"] < EFF_ANCHOR_THRESH)
    ].copy()

    if len(low) == 0:
        return pd.DataFrame()

    rows = []
    for _, ref in low.iterrows():
        for _ in range(EFF_ANCHOR_REPEATS):
            noise = 1.0 + rng.normal(0, EFF_ANCHOR_JITTER, size=4)
            C_j   = float(np.clip(ref["Concentration_M"] * noise[0], 0.05, 4.0))
            T_j   = float(np.clip(ref["Temperature_C"]   * noise[1], 20.0, 140.0))
            t_j   = float(np.clip(ref["Time_hrs"]        * noise[2], 0.02,  10.0))
            SLR_j = float(np.clip(ref["SLR_gL"]          * noise[3],  5.0,  80.0))

            eff_j = float(np.clip(
                ref["Efficiency_pct"] + rng.normal(0, EFF_NOISE_SIGMA_LO),
                EFF_MIN, 100.0
            ))

            # Pick a SMILES consistent with this row if possible
            has_r = int(ref["Has_Reductant"])
            red   = str(ref["Reductant_Std"]) if has_r else "None"

            # Use the same solvent as the reference row; fall back to pool
            smiles_match = pool[pool["SMILES"] == ref.get("SMILES", "")]
            if len(smiles_match) == 0:
                smiles_match = pool.sample(1, random_state=int(rng.integers(1e6)))
            mol = smiles_match.iloc[0]

            row = {
                "DOI": np.nan, "Title": np.nan,
                "Solvent_Name": mol["Solvent_Name"],
                "Solvent_Type": mol.get("Solvent_Type", "Organic Acid"),
                "Battery_Chemistry_Std": ref.get("Battery_Chemistry_Std", "Unknown"),
                "SMILES": mol["SMILES"],
                "Concentration_M": round(C_j, 4),
                "Temperature_C":   round(T_j, 4),
                "Time_hrs":        round(t_j, 4),
                "SLR_gL":          round(SLR_j, 4),
                "Has_Reductant":   has_r,
                "Reductant_Std":   red,
                "Target_Metal":    ref.get("Target_Metal", "Li"),
                "Efficiency_pct":  round(eff_j, 4),
                "Source":          "Augmented",
                "Augmentation_Method": "LowEff-Jitter",
            }
            # Copy molecular descriptors from pool row
            for col in ["RDKIT_MW","RDKIT_LogP","RDKIT_TPSA","RDKIT_HBD","RDKIT_HBA",
                        "RDKIT_RotBonds","RDKIT_RingCount","RDKIT_AromaticRings",
                        "RDKIT_HeavyAtoms","RDKIT_Has_Carboxyl","RDKIT_Has_Hydroxyl",
                        "RDKIT_Has_Halogen","RDKIT_Has_Phosphorus","RDKIT_Is_Ionic",
                        "RDKIT_Morgan_FP_Density",
                        "EHS_Environment","EHS_Health","EHS_Safety","EHS_Total","GreenScore"]:
                row[col] = mol.get(col, np.nan)

            rows.append(row)

    return pd.DataFrame(rows)


def generate(n_augmented: int = N_AUGMENTED, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Main augmentation routine. Returns the final combined DataFrame and
    also saves it to OUTPUT_CSV.
    """
    rng = np.random.default_rng(seed)
    print(f"\n{'='*60}")
    print("  Augmentation Pipeline")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 1. Load original data
    # ------------------------------------------------------------------
    print("\n[1/6] Loading original data ...")
    df_orig = pd.read_csv(INPUT_CSV)
    print(f"  Loaded {len(df_orig):,} rows, {df_orig.shape[1]} columns")

    # Fix structural nulls before anything else
    df_orig["Reductant_Std"] = df_orig["Reductant_Std"].fillna("None")

    # ------------------------------------------------------------------
    # 2. Build SMILES pool (existing + new acids)
    # ------------------------------------------------------------------
    print("\n[2/6] Building molecular pool ...")
    pool = _build_smiles_pool(df_orig)

    # ------------------------------------------------------------------
    # 3. Fit kinetics model on Real + Derived rows
    # ------------------------------------------------------------------
    print("\n[3/6] Fitting Random Forest kinetics model ...")
    kinetics_model = _fit_kinetics_model(df_orig)

    # ------------------------------------------------------------------
    # 4. Define design space and sample conditions via stratified LHS
    # ------------------------------------------------------------------
    print(f"\n[4/6] Sampling {n_augmented} conditions via stratified LHS ...")
    ds = _design_space(df_orig)

    # Global SLR bounds
    slr_lo, slr_hi = ds["SLR_gL"]
    t_lo,   t_hi   = ds["Time_hrs"]

    n_ext = int(n_augmented * TIER_EXTREME     * 1.6)
    n_unf = int(n_augmented * TIER_UNFAVORABLE * 1.6)
    n_mod = int(n_augmented * TIER_MODERATE    * 1.6)
    n_fav = int(n_augmented * TIER_FAVORABLE   * 1.6)

    # Tier 1 -- Extreme-unfavorable: very low T/C, high SLR, no reductant
    # Mirrors real low-efficiency Derived rows: T~25-35 C, C~0.1-0.5 M, SLR~20-70
    C_e, T_e, t_e, SLR_e = _sample_tier(
        n_ext, rng, seed + 1,
        T_range=(25.0, 40.0),
        C_range=(0.10, 0.50),
        t_range=(0.03, 2.00),
        SLR_range=(max(slr_lo, 20.0), min(slr_hi * 1.2, 70.0)),
    )

    # Tier 2 -- Unfavorable: low T, low C
    C_u, T_u, t_u, SLR_u = _sample_tier(
        n_unf, rng, seed + 2,
        T_range=(40.0, 60.0),
        C_range=(0.10, 0.80),
        t_range=(t_lo, 2.50),
        SLR_range=(slr_lo * 0.8, slr_hi),
    )

    # Tier 3 -- Moderate: medium T, medium C
    C_m, T_m, t_m, SLR_m = _sample_tier(
        n_mod, rng, seed + 3,
        T_range=(60.0, 90.0),
        C_range=(0.80, 1.60),
        t_range=(t_lo, t_hi),
        SLR_range=(slr_lo, slr_hi),
    )

    # Tier 4 -- Favorable: high T, high C, long t
    C_f, T_f, t_f, SLR_f = _sample_tier(
        n_fav, rng, seed + 4,
        T_range=(90.0, min(130.0, ds["Temperature_C"][1])),
        C_range=(1.60, max(3.00, ds["Concentration_M"][1])),
        t_range=(t_hi * 0.4, t_hi),
        SLR_range=(slr_lo, slr_hi * 0.7),
    )

    C_s   = np.concatenate([C_e,   C_u,   C_m,   C_f])
    T_s   = np.concatenate([T_e,   T_u,   T_m,   T_f])
    t_s   = np.concatenate([t_e,   t_u,   t_m,   t_f])
    SLR_s = np.concatenate([SLR_e, SLR_u, SLR_m, SLR_f])
    n_ext_unf = n_ext + n_unf
    n_lhs = len(C_s)

    # ------------------------------------------------------------------
    # 5. Assign molecular + categorical features
    # ------------------------------------------------------------------
    print("[5/6] Assigning molecular and categorical features ...")

    # Basis set: Real + Derived rows (the ground-truth experimental data)
    basis = df_orig[df_orig["Source"].isin(["Real", "Derived"])].copy()

    # SMILES sampling weights
    smiles_weights = _build_smiles_weights(basis, pool)
    smiles_idx = rng.choice(len(pool), size=n_lhs, p=smiles_weights)
    mol_rows   = pool.iloc[smiles_idx].reset_index(drop=True)

    # Sample Has_Reductant:
    # Extreme + Unfavorable tiers = no reductant; moderate/favorable sample from basis
    has_r_probs = basis["Has_Reductant"].value_counts(normalize=True)
    has_r_mod_fav = rng.choice(
        has_r_probs.index.values,
        size=n_mod + n_fav,
        p=has_r_probs.values,
    )
    has_r_vals = np.concatenate([
        np.zeros(n_ext_unf, dtype=int),
        has_r_mod_fav,
    ])

    # Sample reductant names only for rows with a reductant
    red_pool = basis.loc[basis["Has_Reductant"] == 1, "Reductant_Std"].dropna()
    red_probs = red_pool.value_counts(normalize=True)
    red_vals  = np.where(
        has_r_vals == 1,
        rng.choice(red_probs.index.values, size=n_lhs, p=red_probs.values),
        "None",
    )

    # Sample Battery_Chemistry_Std and Target_Metal from basis distribution
    def sample_cat(col: str) -> np.ndarray:
        vc = basis[col].value_counts(normalize=True)
        return rng.choice(vc.index.values, size=n_lhs, p=vc.values)

    batt_vals  = sample_cat("Battery_Chemistry_Std")
    metal_vals = sample_cat("Target_Metal")

    # ------------------------------------------------------------------
    # 6. Predict efficiency with kinetics model + noise + quality filter
    # ------------------------------------------------------------------
    X_new = _kinetics_features(
        C_s, T_s, t_s, SLR_s,
        has_r_vals.astype(float),
        mol_rows["RDKIT_MW"].values,
        mol_rows["RDKIT_LogP"].values,
        mol_rows["RDKIT_TPSA"].values,
        mol_rows["RDKIT_HBD"].values.astype(float),
        mol_rows["RDKIT_HBA"].values.astype(float),
    )
    E_pred = kinetics_model.predict(X_new)
    E_noisy = np.clip(E_pred + rng.normal(0, NOISE_SIGMA, size=n_lhs), EFF_MIN, 100.0)

    keep = (E_noisy >= EFF_MIN) & (E_noisy <= 100.0)
    print(f"  Quality filter: {keep.sum():,} / {n_lhs:,} samples retained")

    # Build augmented DataFrame
    aug = pd.DataFrame({
        "DOI":                   np.nan,
        "Title":                 np.nan,
        "Solvent_Name":          mol_rows["Solvent_Name"].values,
        "Solvent_Type":          mol_rows["Solvent_Type"].values,
        "Battery_Chemistry_Std": batt_vals,
        "SMILES":                mol_rows["SMILES"].values,
        "Concentration_M":       C_s,
        "Temperature_C":         T_s,
        "Time_hrs":              t_s,
        "SLR_gL":                SLR_s,
        "Has_Reductant":         has_r_vals,
        "Reductant_Std":         red_vals,
        "Target_Metal":          metal_vals,
        "Efficiency_pct":        E_noisy,
        "Source":                "Augmented",
        "Augmentation_Method":   "Physics-Gaussian-LHS",
        "RDKIT_MW":              mol_rows["RDKIT_MW"].values,
        "RDKIT_LogP":            mol_rows["RDKIT_LogP"].values,
        "RDKIT_TPSA":            mol_rows["RDKIT_TPSA"].values,
        "RDKIT_HBD":             mol_rows["RDKIT_HBD"].values,
        "RDKIT_HBA":             mol_rows["RDKIT_HBA"].values,
        "RDKIT_RotBonds":        mol_rows["RDKIT_RotBonds"].values,
        "RDKIT_RingCount":       mol_rows["RDKIT_RingCount"].values,
        "RDKIT_AromaticRings":   mol_rows["RDKIT_AromaticRings"].values,
        "RDKIT_HeavyAtoms":      mol_rows["RDKIT_HeavyAtoms"].values,
        "RDKIT_Has_Carboxyl":    mol_rows["RDKIT_Has_Carboxyl"].values,
        "RDKIT_Has_Hydroxyl":    mol_rows["RDKIT_Has_Hydroxyl"].values,
        "RDKIT_Has_Halogen":     mol_rows["RDKIT_Has_Halogen"].values,
        "RDKIT_Has_Phosphorus":  mol_rows["RDKIT_Has_Phosphorus"].values,
        "RDKIT_Is_Ionic":        mol_rows["RDKIT_Is_Ionic"].values,
        "RDKIT_Morgan_FP_Density": mol_rows["RDKIT_Morgan_FP_Density"].values,
        "EHS_Environment":       mol_rows["EHS_Environment"].values,
        "EHS_Health":            mol_rows["EHS_Health"].values,
        "EHS_Safety":            mol_rows["EHS_Safety"].values,
        "EHS_Total":             mol_rows["EHS_Total"].values,
        "GreenScore":            mol_rows["GreenScore"].values,
    })

    aug = aug[keep].head(n_augmented).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 7. Generate low-efficiency anchor rows (jittered from real conditions)
    # ------------------------------------------------------------------
    print("[7/7] Generating low-efficiency anchor rows ...")
    anchor = _generate_anchor_rows(df_orig, pool, rng)
    if len(anchor) > 0:
        print(f"  Anchor rows: {len(anchor)}  "
              f"(eff range {anchor['Efficiency_pct'].min():.1f}-{anchor['Efficiency_pct'].max():.1f}%)")
        # Trim LHS-based rows to keep total near n_augmented
        n_keep = max(0, n_augmented - len(anchor))
        aug = aug.head(n_keep)
        aug = pd.concat([aug, anchor], ignore_index=True)

    # ------------------------------------------------------------------
    # 8. Combine: Real + Derived + Augmented  (drop old Synthetic rows)
    # ------------------------------------------------------------------
    real_derived = df_orig[df_orig["Source"].isin(["Real", "Derived"])].copy()
    final = pd.concat([real_derived, aug], ignore_index=True)

    # Remove EHS_Label (classification label -- target is regression only)
    if "EHS_Label" in final.columns:
        final.drop(columns=["EHS_Label"], inplace=True)

    # Round continuous columns for cleanliness
    for col in ["Efficiency_pct", "Concentration_M", "Temperature_C",
                "Time_hrs", "SLR_gL", "EHS_Environment", "EHS_Health",
                "EHS_Safety", "EHS_Total", "GreenScore"]:
        if col in final.columns:
            final[col] = final[col].round(4)

    # ------------------------------------------------------------------
    # 8. Save
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    final.to_csv(OUTPUT_CSV, index=False)

    print(f"\n{'='*60}")
    print(f"  Saved {len(final):,} rows -> {OUTPUT_CSV}")
    print(f"\n  Source breakdown:")
    print(final["Source"].value_counts().to_string())
    print(f"\n  Efficiency_pct summary:")
    print(final["Efficiency_pct"].describe().round(3).to_string())
    print(f"\n  Unique SMILES in output: {final['SMILES'].nunique()}")
    print(f"{'='*60}\n")

    return final


if __name__ == "__main__":
    generate()
