# -*-coding:utf-8-*-
"""
Data Loading Model -CCE ML PROJECT
Providing data loading functionality for the CCE Ml Project,
including file reading and validation of expected column presence.
Author: Tareq Aljamou
Date: Feb 2026
Course: Exam Project FM1AZP110 -AML
Description:
    This module loads CCE data from weatherford_cce_combined.csv

Data Source:Weatherford Laboratories provided by Whitson AS 
Data preparation: Data manually extracted and combined by the author
                from Weatherford report into a CSV file.
Note: 
    -AI assistance:Microsoft Copilot was used to draft parts of this module.
    -The author reviewed,modified,tested ,and validated all content.

"""
import pandas as pd

def load_cce_dataset(path: str) -> pd.DataFrame:
    """
    CSV file created by extracting data from Weatherford report Tables 15,17,19 and 21.
 
    Parameters:
      path:file path to csv containg CCE data
    Returns:
      DataFrame with Pressure,V_rel,and Temperature columns
    Raises
     ValueError:If required columns are missing 
    """

    # Read the CSV into a DataFrame (manually created from pvt report)
    df = pd.read_csv(path)

    # Validation required columns are present 
    required = {"Pressure", "V_rel", "Temperature"}
    if not required.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {required}")

    return df  