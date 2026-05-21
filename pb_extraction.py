
# -*-coding:utf-8-*-
"""

Bubble Point Extraction -CCE ML PROJECT

Utilities for extracting bubble point pressure from CCE data.

This module provides:
-Get observed_pb_map(): reads observed bubble points from the dataset (raws where V_rel==1.0), one per temperature.
-pb_from model (): computes bubble point pressure for a   given ML model at selected temperature by scanning a pressure grid (high to low) and linearly interpolating the first crossing at V_rel=1.0.


Author: Tareq Aljamou
Date: Feb 2026
Course: Exam Project FM1AZP110 -AML
Note: 
    -AI assistance: Microsoft Copilot was used to draft parts of this module.
    -The author reviewed, modified, tested, and validated all content.

"""

from typing import Dict, Iterable, Tuple, Optional, Union

import numpy as np
import pandas as pd

from .feature_engineering import add_features

def get_observed_pb_map(df: pd.DataFrame) -> Dict[float, float]:
    """
    Build a dictionary {Temperature: Pb_observed_psi} using rows where V_rel == 1.0.

    Notes
    -----
    - The Weatherford dataset contains exactly one V_rel==1 row per temperature.
      If more exist in other datasets, we take the first occurrence in order.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: 'Pressure', 'V_rel', 'Temperature'.

    Returns
    -------
    dict
        Mapping of temperature (float) -> observed bubble-point pressure (float).
    """
    # Ensure sort order is preserved (if present) and take first V_rel==1 per T
    sub = df[np.isclose(df["V_rel"], 1.0, atol=1e-6)]

    pb_by_T = (
        sub.groupby("Temperature", sort=True)["Pressure"]
        .first()
        .to_dict()
    )
    return pb_by_T

def pb_from_model(
    model,
    features: Iterable[str],
    temperature: float,
    df_T: pd.DataFrame,
    steps: Tuple[int, ...] = (25, 5, 1),
    vrel_target: float = 1.0,
) -> float:
    """
    Compute bubble-point pressure by scanning a descending pressure grid and
    interpolating the first crossing at V_rel == vrel_target.

    The scan uses multi-resolution steps (coarse -> fine) to reduce computations
    while keeping psi-level precision.

    Parameters
    ----------
    model :
        Any regressor with a `.predict(X)` method.
    features : iterable of str
        Columns to pass to the model (must align with training).
    temperature : float
        Temperature (°F) for which to compute Pb.
    df_T : pd.DataFrame
        Slice of the dataset for this temperature, with 'Pressure' column.
        Used to determine scan bounds.
    steps : tuple of int, optional
        Pressure step sizes in psi for coarse->fine refinement, default (25, 5, 1).
    vrel_target : float, optional
        Target V_rel (=1.0 for bubble-point).

    Returns
    -------
    float
        Estimated bubble-point pressure (psi). Returns np.nan if no crossing was found.

    Notes
    -----
    - We scan from high -> low pressure (typical CCE plotting convention).
    - We expect V_rel to be < 1.0 at high pressure and > 1.0 at low pressure.
    - The first sign-change in (V_rel - vrel_target) is linearly interpolated to Pb.
    """
    if df_T.empty:
        return float("nan")

    pmin = float(df_T["Pressure"].min())
    pmax = float(df_T["Pressure"].max())

    # Initialize search window to the full observed range at this temperature
    P_high = pmax
    P_low = pmin
    crossing_P: Optional[float] = None

    for step in steps:
       
       #break early to avoid a silent no_op iteration.
        if P_high <= P_low:
            break

        # Descending grid for conventional CCE visualization
        P = np.arange(P_high, P_low -step, -step, dtype=float)

        Xg = pd.DataFrame(
            {
                "Pressure": P,
                "Temperature": np.full_like(P, temperature, dtype=float),
            }
        )
        Xg = add_features(Xg)

        # Predict V_rel, subset columns to the features used by this model
        V = model.predict(Xg[list(features)])  # shape (len(P),)
        diff = V - float(vrel_target)
        s = np.sign(diff)

        # Find first index where sign changes or diff hits zero exactly
        idx = np.where(s[:-1] * s[1:] <= 0)[0]

        if len(idx) == 0:
            # No crossing found at this resolution -> continue to next (finer)
            # but keep the same bounds
            continue

        # First crossing governs Pb
        i = int(idx[0])
        x1, x2 = P[i], P[i + 1]
        y1, y2 = diff[i], diff[i + 1]

        # Linear interpolation for exact crossing
        if abs(y2 - y1) > 1e-12: 
            crossing_P = x1 - y1 * (x2 - x1) / (y2 - y1)
        else:
            crossing_P = float(x1)

        # Narrow the window around the detected bracket for the next pass
        P_high, P_low = max(x1, x2), min(x1, x2)

    return float(crossing_P) if crossing_P is not None else float("nan")

