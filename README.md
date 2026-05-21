ML Application to CCE PVT Data Analysis

Noroff Fagskole — Applied Machine Learning (FM1AZP110) 
Final Exam Project — Grade: B 
Author: Tareq Aljamou | Industry Partner: Whitson AS | Date: February 2026

---

 Overview

This project applies **machine learning to Constant Composition Expansion (CCE) laboratory PVT data** to predict relative volume behaviour of reservoir fluids under varying pressure and temperature conditions.

CCE is a fundamental PVT laboratory experiment used in petroleum engineering to characterise reservoir fluid behaviour. Traditionally, results are modelled using Equation of State (EOS) methods. This project demonstrates that ML models — with careful feature engineering and physics-informed design — can serve as credible complements to traditional EOS approaches.

Industry collaboration: Whitson AS — leading PVT software and consulting company.

---

 Key Results

| Model | MAE | Notes |

| Linear Regression (baseline) | High | Poor fit — non-linear problem |
| Random Forest (baseline) | Medium | Good generalisability |
| XGBoost (baseline) | Medium | Strong baseline |
| Step10 (physics-informed) | Low | Best generalisation |
| v1 / v2 (tuned XGBoost) | Low| Best overall performance |


---

 Project Structure

```
├── Main.ipynb                  # Main notebook — end-to-end workflow
├── run_all.py                  # Command-line runner for full pipeline
├── data_loading.py             # Data loading and initial processing
├── feature_engineering.py     # Physics-informed feature creation
├── splitting.py                # Temperature-based train/test split
├── baseline_models.py          # LR, Random Forest, XGBoost baselines
├── step10_model.py             # Physics-informed Step10 model
├── tuning_models.py            # Hyperparameter tuning (v1, v2)
├── pb_extraction.py            # Bubble point pressure extraction
├── uncertainty_estimation.py   # Model uncertainty quantification
├── comparison_tables.py        # Results comparison and tables
├── plotting.py                 # Visualisation utilities
├── src/                        # Modular source package
│   ├── __init__.py
│   └── [all modules above]
└── weatherford_cce_combined.csv # CCE PVT dataset
```

---

 Methodology

 1. Data
- Source: Weatherford CCE laboratory PVT dataset
- Features: Pressure, Temperature, relative volume (V_rel)
- Target: V_rel prediction across pressure ranges
- Data not included in this repository — proprietary CCE laboratory dataset. To reproduce results, substitute with your own CCE PVT data in the same format: columns [Pressure, Temperature, V_rel].

 2. Feature Engineering
Physics-informed features created from raw pressure and temperature:

| Feature | Formula | Physical Meaning |

| P² | Pressure² | Quadratic pressure term |
| invP | 1/Pressure | Near bubble-point curvature |
| PT | Pressure × Temperature | Interaction effect |
| T² | Temperature² | Quadratic temperature term |
| logP | log(Pressure) | Stabilises learning across pressure ranges |
| TlogP | Temperature × log(Pressure) | Tuned model feature |

3. Physics-Respecting Split
held-out temperature split** was used for evaluation — models were tested on unseen temperature conditions to ensure genuine generalisation, not just interpolation.

 4. Models Compared
- Baselines:Linear Regression, Random Forest, XGBoost
- Physics-informed:Step10 model with monotonic constraints
- Tuned: v1 and v2 — XGBoost with physics-informed features and hyperparameter optimisation

 5. Bubble Point Extraction
Bubble point pressure (Pb) was extracted from model predictions and compared against laboratory-measured values.

---

 Technologies

| Technology | Purpose |

| Python | Core implementation |
| pandas / NumPy | Data manipulation |
| scikit-learn | ML pipelines, baselines |
| XGBoost | Gradient boosting models |
| Matplotlib | Visualisation |
| Jupyter Notebook | Interactive workflow |

---

 How to Run

 Requirements
```bash
pip install pandas numpy scikit-learn xgboost matplotlib jupyter
```

 Run full pipeline
```bash
python run_all.py
```

 Or use the notebook
```bash
jupyter notebook Main.ipynb
```

---

 Domain Context

CCE (Constant Composition Expansion)** is a standard PVT laboratory test where a reservoir fluid sample is expanded at constant composition. It measures:
- Saturation pressure (bubble/dew point)
- Compressibility above saturation pressure
- Two-phase volume ratios below saturation

Accurate CCE modelling is critical for reservoir simulation, production forecasting, and well design. This project demonstrates that ML can complement traditional EOS modelling — particularly useful when EOS calibration data is limited.

---

 Note on AI Assistance

Microsoft Copilot was used to draft parts of the code modules. All content was reviewed, modified, tested, and validated by the author.

---

 Author

**Tareq Aljamou**  
Professional Degree in Applied Machine Learning — Noroff Fagskole (2025–2026)  
MSc Petroleum Engineering — NTNU  
Industry collaboration: Whitson AS  
[LinkedIn](https://www.linkedin.com/in/tareq-aljamou-1b0674145/)

