# -*-coding:utf-8-*-
"""
Feature Engineering Model -CCE ML PROJECT
Creates engineered features from raw pressure and temperature data to improve model performance on CCE PVT analysis
Author: Tareq Aljamou
Date: Feb 2026
Course: Exam Project FM1AZP110 -AML
Note: 
    -AI assistance:Microsoft Copilot was used to draft parts of this module.
    -The author reviewed,modified,tested ,and validated all content.
"""

import numpy as np
import pandas as pd

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to dataframe.
    features added:
      P2    : pressure squared
      invP  :1/pressure (reciprocal)
      PT    : pressure x temerature (ineraction)
      T2    : tempreature squared
      logP  : log(pressure)
      TlogP :tempreature x log(pressure)

    Returns:
       Dataframe with original columns plus new features.
    """
    X = df.copy()

    # Basic quadratic expansion
    X["P2"] = X["Pressure"] ** 2

    # Reciprocal pressure: helps shape V_rel curvature near bubble point
    X["invP"] = 1.0 / X["Pressure"]

    # Pressure–Temperature interaction
    X["PT"] = X["Pressure"] * X["Temperature"]

    # Quadratic temperature: curvature in T dimension
    X["T2"] = X["Temperature"] ** 2

    # Logarithmic pressure: stabilizes learning across pressure ranges
    X["logP"] = np.log(X["Pressure"])

    # T * log(P): used in tuned models (v1 and v2)
    X["TlogP"] = X["Temperature"] * X["logP"]

    return X

def get_feature_list(version="baseline"):
    """
    Return the list of feature (version="baseline"):

    Parameters :
        version:'baseline', 'step10','v1', or 'v2'
    
    Returns:
        list of feature column names.

    """
    
    if version == "baseline":
        return ["Pressure", "Temperature"]

    if version == "step10":
        return ["Pressure", "Temperature", "P2", "invP", "PT", "T2", "logP"]

    if version in ("v1", "v2"):
        return ["Pressure", "Temperature", "P2", "invP", "PT", "T2", "logP", "TlogP"]

    raise ValueError(f"Unknown feature version: {version}")
