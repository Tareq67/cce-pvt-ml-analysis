
# -*-coding:utf-8-*-
"""
Uncertainty Estimation - CCE ML PROJECT

Bootstrap-based uncertainty estimation for:

1-	V_rel predications across the full dataset.
2-	bubble-point pressure per temperature.

Design principles

 -Physics-respecting split is preserved: the held-out temperature is *never* used during any bootstrap training.
-Resampling is done within the training subset replication.
-We re-fit the provided model builder for each bootstrap replicate.
Pb is extracted using the official pb-from model () function in src .py-extraction.
Returns per-temperature summary stats (mean, std, CI) and optional bootstrap samples for further analysis.

Key functions
 -bootstrap-vrel-predications (…): residual/row bootstrap for V_rel curves.
- bootstrap-pb_by_temperature (…): bootstrap CI for pb per temperature.
-summarize_ci (…): generic CI using Normal or percentile method.


Author: Tareq Aljamou
Date: Feb 2026
Course: Exam Project FM1AZP110 -AML
Note: 
    -AI assistance: Microsoft Copilot was used to draft parts of this module.
    -The author reviewed, modified, tested, and validated all content.


"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .feature_engineering import get_feature_list
from .pb_extraction import pb_from_model


# ----------------------------- Data classes ----------------------------- #

@dataclass
class CISummary:
    mean: float
    std: float
    ci_low: float
    ci_high: float
    n: int


# ----------------------------- CI helpers ------------------------------ #

def summarize_ci(samples: np.ndarray,
                 ci: float = 0.95,
                 method: str = "percentile") -> CISummary:
    """
    Summarize a 1-D bootstrap sample array into mean, std, and (1 - alpha) CI.
    Parameters
    ----------
    samples : np.ndarray
        Bootstrap samples, shape (B,).
    ci : float
        Confidence level, e.g., 0.95 for 95% CI.
    method : {"percentile", "normal"}
        CI computation method:
        - "percentile": [alpha/2, 1 - alpha/2] quantiles
        - "normal": mean ± z * std (z from N(0,1), z≈1.96 for 95% CI)
    """
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    n = samples.size
    if n == 0:
        return CISummary(np.nan, np.nan, np.nan, np.nan, 0)

    mu = float(np.mean(samples))
    sd = float(np.std(samples, ddof=1)) if n > 1 else 0.0
    alpha = 1.0 - ci

    if method == "normal":
        from scipy.stats import norm
        z = float(norm.ppf(1.0 - alpha / 2.0))
        lo, hi = mu - z * sd, mu + z * sd
    else:
        lo = float(np.quantile(samples, alpha / 2.0))
        hi = float(np.quantile(samples, 1.0 - alpha / 2.0))

    return CISummary(mu, sd, lo, hi, n)


# ------------------------ Bootstrap core utilities --------------------- #

def _resample_indices(n: int, rng: np.random.Generator) -> np.ndarray:
    """Return bootstrap resample indices of length n (sample with replacement)."""
    return rng.integers(0, n, size=n)


def _fit_model_on_bootstrap(df: pd.DataFrame,
                            train_mask: pd.Series,
                            builder: Callable[[], object],
                            feature_key: str,
                            target_col: str,
                            rng: np.random.Generator) -> Tuple[object, List[str]]:
    """
    Bootstrap a training subset (rows where train_mask is True),
    fit a fresh model, return (model, feature_list).
    """
    feats = get_feature_list(feature_key)
    dtrain = df.loc[train_mask, feats + [target_col]].dropna().copy()

    if dtrain.empty:
        raise ValueError("Training subset is empty after filtering. Check masks.")

    # Row bootstrap within the training subset
    idx = dtrain.index.to_numpy()
    bs_idx = _resample_indices(len(idx), rng)
    bs_rows = idx[bs_idx]

    X_bs = df.loc[bs_rows, feats].values
    y_bs = df.loc[bs_rows, target_col].values

    model = builder()
    model.fit(X_bs, y_bs)
    return model, feats


# --------------------- Public API: V_rel uncertainty ------------------- #

def bootstrap_vrel_predictions(df: pd.DataFrame,
                               train_mask: pd.Series,
                               builder: Callable[[], object],
                               feature_key: str,
                               target_col: str = "V_rel",
                               B: int = 200,
                               seed: int = 42,
                               return_all: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Bootstrap V_rel predictions across the FULL dataset, using bootstrap
    re-fits on the training subset only (physics-respecting split preserved).

    Parameters
    ----------
    df : DataFrame
        Feature-engineered DataFrame including the target and all features.
    train_mask : Series[bool]
        Mask indicating training rows.
    builder : callable
        Zero-arg function that returns a fresh model (e.g., build_v1).
    feature_key : str
        Key passed to get_feature_list(...) to align features with the model.
    target_col : str
        Name of the target column in df (default "V_rel").
    B : int
        Number of bootstrap replicates.
    seed : int
        RNG seed for reproducibility.
    return_all : bool
        If True, include full bootstrap prediction matrix (can be large).

    Returns
    -------
    dict
        {
          "summary": DataFrame with columns
              ["row_index", "mean", "std", "ci_low", "ci_high", "n"],
          "samples": (optional) DataFrame shape (len(df), B) with bootstrap preds
        }
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    preds = np.zeros((n, B), dtype=float)

    for b in range(B):
        model, feats = _fit_model_on_bootstrap(
            df=df,
            train_mask=train_mask,
            builder=builder,
            feature_key=feature_key,
            target_col=target_col,
            rng=rng
        )
        preds[:, b] = model.predict(df[feats].values)

    # Summarize CI per row
    rows = []
    for i in range(n):
        s = summarize_ci(preds[i, :], ci=0.95, method="percentile")
        rows.append((df.index[i], s.mean, s.std, s.ci_low, s.ci_high, s.n))

    summary = pd.DataFrame(rows, columns=["row_index", "mean", "std", "ci_low", "ci_high", "n"])

    out = {"summary": summary}
    if return_all:
        samples_df = pd.DataFrame(preds, index=df.index,
                                  columns=[f"bs_{j+1}" for j in range(B)])
        out["samples"] = samples_df
    return out


# --------------------- Public API: Pb uncertainty ---------------------- #

def bootstrap_pb_by_temperature(df: pd.DataFrame,
                                train_mask: pd.Series,
                                builder: Callable[[], object],
                                feature_key: str,
                                temps: Optional[Iterable[float]] = None,
                                target_col: str = "V_rel",
                                B: int = 300,
                                seed: int = 42,
                                ci: float = 0.95,
                                method: str = "percentile") -> pd.DataFrame:
    """
    Bootstrap CI for Bubble-Point (Pb) per temperature.

    For each bootstrap replicate:
      - Resample training rows with replacement
      - Fit a fresh model
      - Compute Pb(T) via pb_from_model(...) using the trained model

    Parameters
    ----------
    df : DataFrame
        Feature-engineered DataFrame including V_rel and features.
    train_mask : Series[bool]
        Mask for training rows (held-out temperature is *not* used in training).
    builder : callable
        Zero-arg function returning a fresh model (e.g., build_v1).
    feature_key : str
        Key for get_feature_list(...).
    temps : iterable of float, optional
        Temperatures to evaluate; defaults to sorted unique temperatures in df.
    target_col : str
        Name of target column (default "V_rel"). Included for API symmetry.
    B : int
        Number of bootstrap replicates (e.g., 300).
    seed : int
        RNG seed.
    ci : float
        Confidence level for intervals.
    method : {"percentile", "normal"}
        CI method.

    Returns
    -------
    DataFrame
        One row per temperature with columns:
        ["Temperature", "Pb_mean", "Pb_std", "Pb_ci_low", "Pb_ci_high", "n", "Pb_samples" (list)]
    """
    rng = np.random.default_rng(seed)

    if temps is None:
        temps = sorted(df["Temperature"].unique().tolist())
    else:
        temps = list(temps)

    # Collect samples per temperature
    samples_map: Dict[float, List[float]] = {T: [] for T in temps}

    for b in range(B):
        model, feats = _fit_model_on_bootstrap(
            df=df,
            train_mask=train_mask,
            builder=builder,
            feature_key=feature_key,
            target_col=target_col,
            rng=rng
        )

        # Evaluate Pb at each temperature using the official extractor
        for T in temps:
            dT = df[df["Temperature"].eq(T)]
            if dT.empty:
                samples_map[T].append(np.nan)
                continue
            pb = pb_from_model(model, feats, float(T), dT)
            samples_map[T].append(pb)

    rows = []
    for T in temps:
        arr = np.asarray(samples_map[T], dtype=float)
        s = summarize_ci(arr, ci=ci, method=method)
        rows.append({
            "Temperature": float(T),
            "Pb_mean": s.mean,
            "Pb_std": s.std,
            "Pb_ci_low": s.ci_low,
            "Pb_ci_high": s.ci_high,
            "n": s.n,
            "Pb_samples": arr.tolist(),  # optional for further analysis
        })

    return pd.DataFrame(rows).sort_values("Temperature").reset_index(drop=True)
