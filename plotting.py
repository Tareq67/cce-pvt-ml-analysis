
# -*-coding:utf-8-*-
"""
Plotting Utilities - CCE ML PROJECT

Generates figures to assess:
 -P-V_rel curves (per temperature) comparing observed vs. multiple models.
 -Pb comparison across temperature.
 -Residual plots vs Pressure.
 -Held-out temperature overlay.

Author: Tareq Aljamou
Date: Feb 2026
Course: Exam Project FM1AZP110 -AML
Note: 
    -AI assistance: Microsoft Copilot was used to draft parts of this module.
    -The author reviewed, modified, tested, and validated all content.

"""

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def _ensure_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def plot_pvrel_by_temperature(
    df: pd.DataFrame,
    preds_map: Dict[str, str],
    out_png: str,
    figsize=(12, 4),
    ncols: int = 2,
    invert_x: bool = True,
):
    """
    Plot P–V_rel curves for each temperature.
    Shows observed curve and multiple prediction columns.

    Parameters
    ----------
    df : pd.DataFrame
        Must include: 'Pressure', 'V_rel', 'Temperature' and predicted columns.
    preds_map : dict
        Mapping {legend_label: column_name} to plot per temperature.
        Example: {'LR':'Vrel_LR', 'RF':'Vrel_RF', 'XGB base':'Vrel_XGB', 'Step10':'Vrel_XGB_FE', ...}
    out_png : str
        Output path (PNG).
    figsize : tuple
        Base figure size per row.
    ncols : int
        Number of subplot columns.
    invert_x : bool
        If True, invert Pressure axis (conventional CCE decreasing-to-right).
    """
    temps = sorted(df["Temperature"].unique())
    n = len(temps)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0], figsize[1] * nrows), squeeze=False)
    axes = axes.flatten()

    for ax, T in zip(axes, temps):
        dT = df[df["Temperature"] == T]
        ax.plot(dT["Pressure"], dT["V_rel"], "k.-", label="Observed")

        for name, col in preds_map.items():
            if col in dT.columns:
                ax.plot(dT["Pressure"], dT[col], ".--", label=name)

        if invert_x:
            ax.invert_xaxis()

        ax.grid(True, alpha=0.3)
        ax.set_title(f"T = {T} °F")
        ax.set_xlabel("Pressure (psi)")
        ax.set_ylabel("V_rel (–)")
        ax.legend()

    # Hide any unused subplots
    for j in range(len(temps), len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    _ensure_dir(out_png)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

def plot_pb_comparison(
    pb_df: pd.DataFrame,
    out_png: str,
    models_to_show: Optional[List[str]] = None,
    title: str = "Bubble-Point (Pb) Comparison",
):
    """
    Plot observed Pb vs model Pb across temperatures.

    Parameters
    ----------
    pb_df : pd.DataFrame
        Must include columns: 'Temperature', 'Pb_obs', 'Pb_<model_label>'.
    out_png : str
        Output path (PNG).
    models_to_show : list of str
        Subset of model labels to overlay, e.g. ['Step10','v1','v2'].
        If None, auto-detect any Pb_* columns except 'Pb_obs'.
    title : str
        Plot title.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(pb_df["Temperature"], pb_df["Pb_obs"], "ko-", label="Observed Pb")

    # Determine which Pb_* columns to plot
    if models_to_show is None:
        models_to_show = [c.replace("Pb_", "") for c in pb_df.columns if c.startswith("Pb_") and c != "Pb_obs"]

    # Overlay model series
    for m in models_to_show:
        col = f"Pb_{m}"
        if col in pb_df.columns:
            ax.plot(pb_df["Temperature"], pb_df[col], "o--", label=col)

    ax.set_title(title)
    ax.set_xlabel("Temperature (°F)")
    ax.set_ylabel("Bubble-Point Pb (psi)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    _ensure_dir(out_png)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

def plot_residuals_by_model(
    df: pd.DataFrame,
    model_cols: Dict[str, str],
    out_png: str,
    color_by: str = "Temperature",
    invert_x: bool = True,
):
    """
    Residual scatter plots (Observed - Predicted) vs Pressure for selected models.

    Parameters
    ----------
    df : pd.DataFrame
        Must include: 'V_rel', 'Pressure', and 'Temperature'.
    model_cols : dict
        {legend_label: column_name} mapping to predicted V_rel columns.
    out_png : str
        Output path.
    color_by : str
        Column to color points by (default: 'Temperature').
    invert_x : bool
        If True, invert Pressure axis.
    """
    fig, axes = plt.subplots(1, len(model_cols), figsize=(6 * len(model_cols), 5), squeeze=False)
    axes = axes.flatten()

    sc = None 

    for ax, (name, col) in zip(axes, model_cols.items()):
        if col not in df.columns:
            continue

        resid = df["V_rel"] - df[col]
        sc = ax.scatter(df["Pressure"], resid, c=df[color_by], cmap="viridis", s=18)
        ax.axhline(0.0, color="k", lw=1)

        if invert_x:
            ax.invert_xaxis()

        ax.set_title(f"Residuals: {name}")
        ax.set_xlabel("Pressure (psi)")
        ax.set_ylabel("Observed - Predicted")
        ax.grid(True, alpha=0.3)

    # only draw colorbar if at least one scatter was plotte.
    if sc is not None:
        fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.85, label=color_by)
    
    fig.tight_layout()
    _ensure_dir(out_png)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

def plot_heldout_overlay(
    df: pd.DataFrame,
    out_png: str,
    heldout_T: float = 229.6,
    models_to_show: Optional[Dict[str, str]] = None,
    invert_x: bool = True,
):
    """
    Overlay Observed vs selected models at the held-out temperature.

    Parameters
    ----------
    df : pd.DataFrame
        Must include: 'Pressure', 'V_rel', 'Temperature', and model columns.
    out_png : str
        Output path.
    heldout_T : float
        Held-out temperature used in your physics-respecting split.
    models_to_show : dict
        {legend_label: column_name} to overlay; defaults to {'RF': 'Vrel_RF', 'XGB FE v1':'Vrel_XGB_FE_v1', 'XGB FE v2':'Vrel_XGB_FE_v2'} if present.
    invert_x : bool
        If True, invert Pressure axis.
    """
    default_models = {
        "RF":        "Vrel_RF",
        "XGB FE v1": "Vrel_XGB_v1",
        "XGB FE v2": "Vrel_XGB_v2",
    }
    plot_models = models_to_show if models_to_show is not None else default_models
        
    held =df[df["Temperature"].eq(heldout_T)].copy()
             
    fig, ax =plt.subplots(figsize=(7, 5)) 
    ax.plot(held["Pressure"], held["V_rel"], "k.-", label="Observed")      

    for name, col in plot_models.items():
        if col in held.columns:
            ax.plot(held["Pressure"], held[col], ".--", label=name)

    if invert_x:
        ax.invert_xaxis()

    ax.set_title(f"Held-out T = {heldout_T} °F")
    ax.set_xlabel("Pressure (psi)")
    ax.set_ylabel("V_rel (–)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    _ensure_dir(out_png)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

