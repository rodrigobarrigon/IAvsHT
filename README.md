# IA vs Heart Team (IAvsHT)

Explainable machine learning pipeline for cardiac surgery decision support — predicting whether a cardiovascular patient should or should not undergo surgery, benchmarked against the clinical judgement of a multidisciplinary Heart Team (HT).

Developed as the Final Degree Project (*Trabajo de Fin de Grado*, TFG) in Biomedical Engineering at the **Universidad de Valladolid**, Spain.

---

## Overview

The pipeline trains and evaluates 12 supervised classifiers across three independent data partitioning strategies, selecting a final model via a calibrated Voting Ensemble. Explainability is provided through SHAP (SHapley Additive exPlanations). The three strategies are designed to assess generalisability under progressively stricter conditions:

| Strategy | Description |
|---|---|
| **Local** | Train/val on a single centre (HCUV 2023), test on three external cohorts |
| **Mixed** | Fully pooled stratified split across all four cohorts |
| **LOCO** | Leave-One-Center-Out cross-validation |

The multicentric dataset comprises **1,164 patients** from three Spanish university hospitals: Hospital Clínico Universitario de Valladolid (2023 and 2025 cohorts), Hospital Universitario Virgen de las Nieves (Granada), and Hospital Clínico Universitario de Salamanca.

> ⚠️ **Patient data are not included in this repository** due to data protection regulations. The scripts expect Excel input files (`.xlsx`) that must be obtained through the corresponding hospital research ethics committees.

---

## Repository Structure

```
IAvsHT/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── 1_preprocessing/
│   ├── IAvsHT_Creation_Split.py       # Stratified train/val split generation
│   └── IAvsHT_Validation_Split.py     # Statistical validation of the split
│
├── 2_statistical_analysis/
│   ├── IAvsHT_test_norm_homoces.py    # Normality and homoscedasticity tests
│   └── IAvsHT_Statistical_Analysis.py # Inter-cohort statistical comparisons and global classifier ranking
│
└── 3_models/
    ├── IAvsHT_local.py                # Local training and validation strategy
    ├── IAvsHT_mixed.py                # Mixed (pooled) partitioning strategy
    └── IAvsHT_LOCO.py                 # Leave-One-Center-Out strategy
```

---

## Pipeline Summary

1. **Preprocessing** — Stratified train/val split ensuring statistical balance across key clinical variables (age, critical preoperative status, two valves + coronary surgery, aorta + valve surgery).
2. **Feature selection** — Fast Correlation-Based Filter (FCBF) applied on the training set; continuous age variable is forced into the model regardless of FCBF output.
3. **Hyperparameter optimisation** — RandomizedSearchCV with stratified 5-fold cross-validation on the training set.
4. **Model evaluation and ranking** — 12 classifiers ranked on the validation set by AUC-ROC, with Balanced Accuracy as tiebreaker.
5. **Voting Ensemble** — Soft voting over the top-3 classifiers, with Platt sigmoid calibration. Final model selected by comparing calibrated AUC; if |ΔAUC| < 0.01, tiebreaker by Balanced Accuracy evaluated at each candidate's own Youden threshold.
6. **Threshold optimisation** — Youden index maximisation on the validation set.
7. **Explainability** — SHAP analysis on test cohorts: global feature importance ranking, beeswarm plots by class, and comparison between all test patients and correctly classified patients.

---

## Classifiers Evaluated

**Logistic Regression · Linear Discriminat Analysis (LDA) · K-Nearest-Neighbors (KNN) · Support Vector MAchine (SVM) · LogitBoost · MultiLayer Perceptron (MLP) · Random Forest · Extra Trees · Random Subspace · AdaBoost · LogitBoost· RUSBoost · XGBoost**

---

## Requirements

Python 3.10+

```bash
pip install -r requirements.txt
```

Key dependencies: `scikit-learn==1.3.2`, `shap==0.45.1`, `xgboost==3.2.0`, `pandas==3.0.3`, `scipy==1.17.1`, `matplotlib==3.10.9`, `openpyxl==3.1.5`.

---

## Expected Input Files

| File | Content |
|---|---|
| `IAvsHT_VALLADOLID_2023.xlsx` | HCUV 2023 Valladolid cohort (sheets: raw data, Train, Val after running `IAvsHT_Creacion_Split.py`) |
| `IAvsHT_VALLADOLID_2025.xlsx` | HCUV 2025 Valladolid cohort |
| `IAvsHT_GRANADA.xlsx` | HUVN Granada cohort |
| `IAvsHT_SALAMANCA.xlsx` | HCUS Salamanca cohort |

All files must include a binary target column `ETIQUETA` (ground truth) and a clinical decision column `OBJETIVO HT` (Heart Team decision).

---

## Execution Order

```
# Step 1 — Generate the stratified split (local strategy only)
python 1_preprocessing/IAvsHT_Creation_Split.py

# Step 2 — Validate the split
python 1_preprocessing/IAvsHT_Validation_Split.py

# Step 3 — Normality and homoscedasticity tests
python 2_statistical_analysis/IAvsHT_test_norm_homoces.py

# Step 4 — Run the desired modelling strategy
python 3_models/IAvsHT_local.py
python 3_models/IAvsHT_mixed.py
python 3_models/IAvsHT_LOCO.py

# Step 5 — Statistical analysis and global classifier ranking
python 2_statistical_analysis/IAvsHT_Statistical_Analysis.py
```

---

## Author

**Rodrigo Barrigón** - Biomedical Engineering — Universidad de Valladolid - Final Degree Project (2025)

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
