# -*-coding:utf-8-*-
"""
Comparison Tables-CCE ML PROJECT

This module creates:
 -Per model bubble point tables for all temperatures.
 -A merged comparison table (Step10 vs v1 vs v2).
 -Pb MAE summary tables.

Author: Tareq Aljamou
Date: Feb 2026
Course: Exam Project FM1AZP110 -AML
Note: 
    -AI assistance: Microsoft Copilot was used to draft parts of this module.
    -The author reviewed, modified, tested, and validated all content.

"""

import numpy as np
import pandas as pd

from .pb_extraction import pb_from_model

def build_pb_table(df, model, feature_list, label, obs_pb_map):
    """
    Build a per-model Pb table for all temperatures.

    Parameters
    ----------
    df : pd.DataFrame
        Full feature-engineered dataframe.
    model : trained regressor
    feature_list : list of str
        Features used by this model (must match training order).
    label : str
        Label inserted into Pb_* and Err_* column names.
    obs_pb_map : dict
        {Temperature: Observed_Pb}

    Returns
    -------
    pd.DataFrame
        Columns: Temperature, Pb_obs, Pb_<label>, Err_<label>
    """
    rows = []
    temps = sorted(df["Temperature"].unique())

    for T in temps:
        subT = df[df["Temperature"] == T]
        Pb_obs = float(obs_pb_map.get(T, np.nan))
        Pb_pred = pb_from_model(model, feature_list, T, subT)

        if np.isnan(Pb_pred) or np.isnan(Pb_obs):
            err = np.nan
        else:
            err =Pb_pred - Pb_obs

        rows.append({
            "Temperature": T,
            "Pb_obs": Pb_obs,
            f"Pb_{label}": Pb_pred,
            f"Err_{label}": err
        })

    return pd.DataFrame(rows)

def merge_step10_v1_v2(pb_step10, pb_v1, pb_v2):
    """
    Merge the three Pb tables into one comparison table.

    Parameters
    ----------
    pb_step10, pb_v1, pb_v2 : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Merged on Temperature + Pb_obs.
    """

    merged = (
        pb_step10
        .merge(pb_v1, on=["Temperature", "Pb_obs"])
        .merge(pb_v2, on=["Temperature", "Pb_obs"])
    )

    # Add absolute errors
    for key in ["Err_Step10", "Err_v1", "Err_v2"]:
        if key in merged:
            merged[f"Abs_{key}"] = merged[key].abs()

    return merged

def pb_mae_summary(merged):
    """
    Compute Pb MAE summary across Step10, v1, v2.

    Parameters
    ----------
    merged : pd.DataFrame
        Output of merge_step10_v1_v2()

    Returns
    -------
    pd.DataFrame
        Columns: Model, Pb_MAE_abs_psi
    """
    def _safe_mean(col):
        return merged[col] .mean() if col in merged.columns else np.nan
    
    summary = pd.DataFrame({
        "Model": ["Step10", "v1", "v2"],
        "Pb_MAE_abs_psi": [
            _safe_mean("Abs_Err_Step10"),
            _safe_mean("Abs_Err_v1"),
            _safe_mean("Abs_Err_v2"),
        ]
    })

    return summary





