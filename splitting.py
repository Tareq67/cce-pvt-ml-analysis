
# -*-coding:utf-8-*-
"""
Splitting Model -CCE ML PROJECT
Performs train/test splitting by temperature groups, ensuring ML models are evaluated on a completely unseen temperature group
(A full lab CCE run held out entirely)

Author: Tareq Aljamou
Date: Feb 2026
Course: Exam Project FM1AZP110 -AML
Note:
-AI assistance: Microsoft Copilot was used to draft parts of this module.
-The author reviewed, modified, tested, and validated all content.
"""

import pandas as pd

def physics_split(df: pd.DataFrame,
                  train_temps=(176.0, 320.0, 410.0),
                  test_temp=229.6):
    """
    Create train/test masks based on temperature groups.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset including a 'Temperature' column.
    train_temps : tuple of floats
        Temperatures used for training.
    test_temp : float
        Temperature held out entirely for testing.

    Returns
    -------
    train_mask : pd.Series (bool)
        True where row belongs to training set.
    test_mask : pd.Series (bool)
        True where row belongs to test set.
    """

    # Rows of training temperatures
    train_mask = df["Temperature"].isin(train_temps)

    # Rows of the held-out temperature
    test_mask = df["Temperature"].eq(test_temp)

    return train_mask, test_mask
