
# -*-coding:utf-8-*-
"""
Tuning Models -CCE ML PROJECT

Improved versions of the Step10 XGBoost model.
Both models use the full set of feature engineering, including the TlogP (Temperature x log Pressure interaction term).

V1:
 -Keeps logP constrained (-1)
 -Relaxes P2 constrained (0)
 -Adds TlogP (Temperature * logP)
 -Best overall performance: MAE=0.020, R2=0.987

V2:
 -same features as v1
 -Also relaxes logP constraint (0)
 -Only Pressure and invP keep monotonic constraints
 -Trends to drift at Temperature extremes (higher Pb error)

Author: Tareq Aljamou
Date: Feb 2026
Course: Exam Project FM1AZP110 -AML
Note: 
    -AI assistance: Microsoft Copilot was used to draft parts of this module.
    -The author reviewed, modified, tested, and validated all content.

"""

from xgboost import XGBRegressor

def build_v1():
    """
    Build the tuned v1 XGB model.

    Features in order:
        ['Pressure', 'Temperature', 'P2', 'invP', 'PT', 'T2', 'logP', 'TlogP']

    Monotone constraints:
        Pressure     → -1
        Temperature  →  0
        P2           →  0   (relaxed)
        invP         → +1
        PT           →  0
        T2           →  0
        logP         → -1   (kept)
        TlogP        →  0
    """
    mono = "(-1,0,0,1,0,0,-1,0)"

    return XGBRegressor(
        n_estimators=1500,
        learning_rate=0.015,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=4.0,
        monotone_constraints=mono,
        random_state=42,
    )

def build_v2():
    """
    Build the tuned v2 XGB model.

    Same features as v1, but:
        - logP constraint relaxed (0)
        - Only Pressure (-1) and invP (+1) have monotonic constraints

    Monotone constraints:
        Pressure     → -1
        Temperature  →  0
        P2           →  0
        invP         → +1
        PT           →  0
        T2           →  0
        logP         →  0   (relaxed)
        TlogP        →  0
    """
    mono = "(-1,0,0,1,0,0,0,0)"

    return XGBRegressor(
        n_estimators=1500,
        learning_rate=0.018,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=3.5,
        monotone_constraints=mono,
        random_state=42,
    )
