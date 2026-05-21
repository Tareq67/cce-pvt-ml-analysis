
# -*-coding:utf-8-*-
"""
Step10 Model -CCE ML PROJECT


Physics-informed XGBoost model using (Feature Engineering + Monotonic Constraints to enforce physical behaviour).
Features used: 
Pressure, Temperature, P2, invP, PT, T2, logP
Monotonic constraints applied to Pressure, P2, invP, and logP.


Author: Tareq Aljamou
Date: Feb 2026
Course: Exam Project FM1AZP110 -AML
Note: 
    -AI assistance: Microsoft Copilot was used to draft parts of this module.
    -The author reviewed, modified, tested, and validated all content.

"""

from xgboost import XGBRegressor

def build_step10_xgb():
    """
    Build the Step 10 XGBoost model.

    Features used (in this order):
        ['Pressure', 'Temperature', 'P2', 'invP', 'PT', 'T2', 'logP']

    Monotonic constraints:
        Pressure         → -1  (V_rel must DECREASE as Pressure increases)
        Temperature      →  0
        P2               → -1  (quadratic pressure is also decreasing)
        invP             → +1  (as 1/P increases, Pressure decreases → V_rel increases)
        PT               →  0
        T2               →  0
        logP             → -1  (logP increases with P → V_rel decreases)

    Returns:
        XGBRegressor instance
    """

    # Monotonicity vector as a string tuple (XGBoost format)
    mono_constraints = "(-1,0,-1,1,0,0,-1)"

    model = XGBRegressor(
        n_estimators=1500,
        learning_rate=0.02,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        monotone_constraints=mono_constraints,
        random_state=42,
    )

    return model
