# -*-coding:utf-8-*-
"""
Baseline Model -CCE ML PROJECT
Including three baseline regressors:
-Linear Regression (with standard_scale)
-Random Forest Regressor
-XGBoost baseline (no feature engineering and no monotonic constraints)

These models serve as benchmarks to assess whether more advanced physics-informed models (Step10, v1, v2) improve performance.

Author: Tareq Aljamou
Date: Feb 2026
Course: Exam Project FM1AZP110 -AML
Note: 
    -AI assistance: Microsoft Copilot was used to draft parts of this module.
    -The author reviewed, modified, tested, and validated all content.

"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def build_lr():
    """
    Linear Regression baseline with StandardScaler.
    Features: [Pressure, Temperature]

    Used to demonstrate extreme underfitting.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LinearRegression())
    ])

def build_rf():
    """
    Random Forest baseline.
    Features: [Pressure, Temperature]

    Surprisingly strong for this dataset because:
    - The dataset size is small
    - CCE curves are smooth and tree splits capture them well
    """
    return RandomForestRegressor(
        n_estimators=600,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )

def build_xgb_baseline():
    """
    XGBoost baseline (P, T only).
    No feature engineering.
    No monotonic constraints.

    This establishes a simple non-linear baseline before we add FE + constraints.
    """
    return XGBRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42
    )
