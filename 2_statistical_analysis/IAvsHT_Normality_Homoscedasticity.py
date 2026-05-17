##############################################################################
# IAvsHT — Normality and Homoscedasticity Tests
# ─────────────────────────────────────────────────────────────────────────────
# For each variable and cohort, the following tests are applied:
#   · Continuous variables (nunique > 5):
#       1. Shapiro-Wilk       → normality per cohort
#                               H0: data follow a normal distribution
#       2. Brown-Forsythe     → homoscedasticity across cohorts
#                               H0: variances are equal across all cohorts
#                               Fallback: Levene if Brown-Forsythe does not converge
#   · Binary / categorical variables (nunique ≤ 5):
#       Shapiro-Wilk does not apply → frequency distribution reported instead
#
# Significance threshold: α = 0.05
# Input:  IAvsHT_VALLADOLID_2023.xlsx (sheets: Train, Val)
#         IAvsHT_VALLADOLID_2025.xlsx · IAvsHT_GRANADA.xlsx · IAvsHT_SALAMANCA.xlsx
##############################################################################

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── Data loading ──────────────────────────────────────────────────────────────
# Valladolid 2023 is reconstructed by concatenating Train and Val sheets.

df_hcuv_2023 = pd.concat([
    pd.read_excel('IAvsHT_VALLADOLID_2023.xlsx', sheet_name='Train'),
    pd.read_excel('IAvsHT_VALLADOLID_2023.xlsx', sheet_name='Val'),
], ignore_index=True)

cohorts = {
    'Valladolid (2023)': df_hcuv_2023,
    'Valladolid (2025)': pd.read_excel('IAvsHT_VALLADOLID_2025.xlsx', sheet_name=0),
    'Granada':           pd.read_excel('IAvsHT_GRANADA.xlsx',         sheet_name=0),
    'Salamanca':         pd.read_excel('IAvsHT_SALAMANCA.xlsx',       sheet_name=0),
}
cohort_names = list(cohorts.keys())

# ── Variable classification ───────────────────────────────────────────────────
# Target and HT decision columns are excluded from the analysis.

EXCLUDE = {'ETIQUETA', 'OBJETIVO HT'}

ref         = df_hcuv_2023.drop(columns=EXCLUDE, errors='ignore')
continuous  = [c for c in ref.columns if ref[c].nunique() > 5]
categorical = [c for c in ref.columns if ref[c].nunique() <= 5]

ALPHA = 0.05

# ── Statistical helpers ───────────────────────────────────────────────────────

def shapiro_wilk(series):
    """
    Shapiro-Wilk normality test. Returns (W, p).
    If N > 5000, a random subsample of 5000 observations is used.
    """
    data = series.dropna().values
    if len(data) < 3:
        return np.nan, np.nan
    if len(data) > 5000:
        data = np.random.default_rng(42).choice(data, 5000, replace=False)
    return stats.shapiro(data)


def brown_forsythe(*groups):
    """
    Brown-Forsythe test (Levene with center='median').
    More robust than standard Levene when data are non-normal.
    Returns (statistic, p).
    """
    clean = [g.dropna().values for g in groups if len(g.dropna()) >= 2]
    if len(clean) < 2:
        return np.nan, np.nan
    try:
        return stats.levene(*clean, center='median')
    except Exception:
        return np.nan, np.nan


def levene(*groups):
    """Standard Levene test (center='mean'). Used as fallback if Brown-Forsythe fails."""
    clean = [g.dropna().values for g in groups if len(g.dropna()) >= 2]
    if len(clean) < 2:
        return np.nan, np.nan
    try:
        return stats.levene(*clean, center='mean')
    except Exception:
        return np.nan, np.nan


def interpret(p, alpha=ALPHA):
    """Returns a human-readable verdict based on the p-value."""
    if np.isnan(p):
        return 'n/a'
    return f'YES (p>{alpha:.2f})' if p > alpha else f'NO (p≤{alpha:.2f})'

# ════════════════════════════════════════════════════════════════════════════
# BLOCK 1 — SHAPIRO-WILK (normality per cohort, continuous variables only)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("BLOCK 1 — SHAPIRO-WILK  (normality per cohort)")
print(f"  H0: data follow a normal distribution   α = {ALPHA}")
print("="*80)

rows_sw = []
for var in continuous:
    for name, df in cohorts.items():
        if var not in df.columns:
            continue
        series = df[var].dropna()
        W, p   = shapiro_wilk(series)
        rows_sw.append({
            'Variable': var,
            'Cohort':   name,
            'N':        len(series),
            'Mean':     round(series.mean(), 2),
            'SD':       round(series.std(),  2),
            'Median':   round(series.median(), 2),
            'W':        round(W, 4) if not np.isnan(W) else np.nan,
            'p-value':  round(p, 4) if not np.isnan(p) else np.nan,
            'Normal':   interpret(p),
        })

df_sw = pd.DataFrame(rows_sw)
print(df_sw.to_string(index=False))

# ════════════════════════════════════════════════════════════════════════════
# BLOCK 2 — BROWN-FORSYTHE / LEVENE (homoscedasticity across cohorts)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("BLOCK 2 — BROWN-FORSYTHE / LEVENE  (equality of variances across cohorts)")
print(f"  H0: variances are equal across all cohorts   α = {ALPHA}")
print("="*80)

rows_bf = []
for var in continuous:
    groups       = [df[var] for df in cohorts.values() if var in df.columns]
    stat_bf, p_bf = brown_forsythe(*groups)

    if np.isnan(p_bf):
        # Brown-Forsythe did not converge — fall back to standard Levene
        stat_final, p_final = levene(*groups)
        test_used = 'Levene (fallback)'
    else:
        stat_final, p_final = stat_bf, p_bf
        test_used = 'Brown-Forsythe'

    rows_bf.append({
        'Variable':    var,
        'Test':        test_used,
        'Statistic':   round(stat_final, 4) if not np.isnan(stat_final) else np.nan,
        'p-value':     round(p_final, 4)    if not np.isnan(p_final)    else np.nan,
        'Equal var.':  interpret(p_final),
    })

df_bf = pd.DataFrame(rows_bf)
print(df_bf.to_string(index=False))

# ════════════════════════════════════════════════════════════════════════════
# BLOCK 3 — FREQUENCY DISTRIBUTION (binary / categorical variables)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("BLOCK 3 — FREQUENCY DISTRIBUTION  (binary / categorical variables)")
print("  Shapiro-Wilk does not apply to binary data.")
print("  Percentage of positive cases (value = 1) reported per cohort.")
print("="*80)

rows_freq = []
for var in categorical:
    for name, df in cohorts.items():
        if var not in df.columns:
            continue
        series = df[var].dropna()
        for cat, n in series.value_counts().sort_index().items():
            rows_freq.append({
                'Variable': var,
                'Cohort':   name,
                'Category': cat,
                'N':        n,
                '%':        round(n / len(series) * 100, 1),
            })

df_freq = pd.DataFrame(rows_freq)

# Pivot: percentage of positives (category = 1) per variable and cohort
print("\n  — % of positive cases (value = 1) by variable and cohort —\n")
pivot_rows = []
for var in categorical:
    sub  = df_freq[(df_freq['Variable'] == var) & (df_freq['Category'] == 1)]
    row  = {'Variable': var}
    for name in cohort_names:
        match = sub[sub['Cohort'] == name]
        row[name] = f"{match['%'].values[0]:.1f}%" if len(match) > 0 else 'n/a'
    pivot_rows.append(row)

print(pd.DataFrame(pivot_rows).to_string(index=False))

# ════════════════════════════════════════════════════════════════════════════
# BLOCK 4 — EXECUTIVE SUMMARY
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("BLOCK 4 — EXECUTIVE SUMMARY")
print("="*80)

print(f"\n  Continuous variables analysed: {len(continuous)}")
for var in continuous:
    sub        = df_sw[df_sw['Variable'] == var]
    n_normal   = sub['Normal'].str.startswith('YES').sum()
    bf_row     = df_bf[df_bf['Variable'] == var].iloc[0]
    print(f"    {var}: normal in {n_normal}/{len(sub)} cohorts")
    print(f"      {bf_row['Test']}: statistic={bf_row['Statistic']}  "
          f"p={bf_row['p-value']}  → equal variances: {bf_row['Equal var.']}")

print(f"\n  Binary / categorical variables: {len(categorical)}")
print("    → Shapiro-Wilk does not apply.")
print("    → For inter-cohort comparisons: Chi-squared (or Fisher if expected N < 5).")

all_normal = df_sw['Normal'].str.startswith('YES').all() if len(df_sw) > 0 else False
print("\n  General conclusion:")
if all_normal:
    print("    All continuous variables are normally distributed in all cohorts.")
    print("    → Welch's t-test can be used for mean comparisons.")
else:
    print("    At least one continuous variable is NOT normally distributed.")
    print("    → Use Mann-Whitney U (non-parametric) for mean comparisons.")
    print("    → Use Spearman instead of Pearson for correlations.")
    print("    → Report median and IQR as descriptive statistics (instead of mean ± SD).")

print("\n" + "="*80)
print("END OF ANALYSIS")
print("="*80)
