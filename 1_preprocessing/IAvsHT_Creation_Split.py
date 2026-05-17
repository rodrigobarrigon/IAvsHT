##############################################################################
# IAvsHT — Creation of Stratified Train / Validation Split
# ─────────────────────────────────────────────────────────────────────────────
# Iterates over random seeds until a statistically balanced partition is found,
# defined as no significant difference (p > 0.05) between the training and
# validation subsets in the following control variables:
#   · Age                          → Welch's t-test (continuous)
#   · Critical preoperative status → Chi-squared (binary)
#   · Bivalvular + Coronary        → Chi-squared (binary)
#   · Aorta + Valve surgery        → Chi-squared (binary)
#
# Output: Train and Val sheets written into the input Excel file.
# Input:  IAvsHT_VALLADOLID_2023.xlsx (single sheet with all patients)
##############################################################################

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_ind, chi2_contingency

# ── Data loading ──────────────────────────────────────────────────────────────

def load_dataset(filepath):
    """Load dataset and separate features (X), ground truth (y) and HT decision (y_ht)."""
    df   = pd.read_excel(filepath)
    y    = df.pop(df.columns[-1])       # Ground truth label (last column)
    y_ht = df.pop('OBJETIVO HT')        # Heart Team clinical decision (second-to-last)
    return df, y, y_ht

EXCEL_FILE = 'IAvsHT_VALLADOLID_2023.xlsx'

X_main, y_main, y_main_ht = load_dataset(EXCEL_FILE)
feature_names = X_main.columns.tolist()
X_np          = X_main.to_numpy()
y_np          = y_main.to_numpy()
indices       = np.arange(X_np.shape[0])

# ── Control variable indices ──────────────────────────────────────────────────

idx_edad       = feature_names.index('EDAD')
idx_critico    = feature_names.index('ESTADO PREOPERATORIO CRÍTICO')
idx_bivalv_cor = feature_names.index('BIVALVULAR + CORONARIO')
idx_aorta_valv = feature_names.index('CIRUGIA DE AORTA + VALVULAR')

# ── Statistical balance helpers ───────────────────────────────────────────────

def chi2_p(group_train, group_val):
    """Chi-squared p-value between two binary groups. Returns 1.0 if table is not 2x2."""
    table = pd.crosstab(
        pd.Series(np.append(group_train, group_val)),
        pd.Series(['Train'] * len(group_train) + ['Val'] * len(group_val))
    )
    if table.shape != (2, 2):
        return 1.0
    _, p, _, _ = chi2_contingency(table)
    return p

# ── Balanced split search ─────────────────────────────────────────────────────
# Criterion: p > 0.05 in all four control variables → split accepted.
# Validation set size is fixed at 151 patients; modify 'test_size' if needed.

seed        = 0
split_found = False

print("\nSearching for a balanced split (p > 0.05 in all control variables)...")

while not split_found:
    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X_np, y_np, indices,
        test_size=151,
        random_state=seed,
        stratify=y_np
    )

    p_age    = ttest_ind(X_train[:, idx_edad],       X_val[:, idx_edad],       equal_var=False)[1]
    p_crit   = chi2_p(X_train[:, idx_critico],       X_val[:, idx_critico])
    p_bivalv = chi2_p(X_train[:, idx_bivalv_cor],    X_val[:, idx_bivalv_cor])
    p_aorta  = chi2_p(X_train[:, idx_aorta_valv],    X_val[:, idx_aorta_valv])

    if p_age > 0.05 and p_crit > 0.05 and p_bivalv > 0.05 and p_aorta > 0.05:
        split_found = True
        print(f"Balanced split found at seed: {seed}")
        print(f"p-values → Age: {p_age:.3f} | Critical: {p_crit:.3f} | "
              f"Bivalvular+Cor: {p_bivalv:.3f} | Aorta+Valve: {p_aorta:.3f}")
    else:
        seed += 1
        if seed > 1000:
            print("No valid split found after 1000 attempts. Using last seed.")
            break

# ── Build output DataFrames ───────────────────────────────────────────────────
# Recover exact rows from X_main using the winning indices,
# then re-attach the HT decision and ground truth columns.

df_train = X_main.iloc[idx_train].copy()
df_val   = X_main.iloc[idx_val].copy()

for df, idx in [(df_train, idx_train), (df_val, idx_val)]:
    df['OBJETIVO HT'] = y_main_ht.iloc[idx].values
    df['ETIQUETA']    = y_main.iloc[idx].values

# ── Export to Excel ───────────────────────────────────────────────────────────
# Train and Val sheets are written into the original file without altering
# the raw data sheet.

with pd.ExcelWriter(EXCEL_FILE, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    df_train.to_excel(writer, sheet_name='Train', index=False)
    df_val.to_excel(writer, sheet_name='Val',   index=False)

print(f"Sheets 'Train' and 'Val' written to {EXCEL_FILE}.")
