##############################################################################
# IAvsHT — Statistical Analysis
# ─────────────────────────────────────────────────────────────────────────────
# Generates a multi-sheet Excel report and associated figures covering:
#
#   Sheet 1 · Descriptive_Table        — descriptive statistics per class and cohort
#                                         · continuous variables: median [IQR]
#                                         · binary variables: count (%)
#   Sheet 2 · Inter_Cohort_Comparison  — Kruskal-Wallis across 5 cohorts per class
#                                         + Mann-Whitney post-hoc for all C(5,2)=10 pairs
#   Sheet 3 · P_Values                 — intra-class significance per cohort
#                                         (Mann-Whitney for continuous, Chi-squared for binary)
#   Sheet 4 · Spearman_Correlation     — Spearman ρ of each variable with:
#                                         (a) ground truth label, (b) Heart Team decision
#   Sheet 5 · Global_Ranking           — Borda count across 6 individual rankings
#                                         (Local, Mixed, LOCO × 4 cohorts)
#
# Figures generated (saved as PNG, 300 dpi):
#   · Spearman_Scatter.png                 — ρ_target vs ρ_HT scatter (divergence coded by colour)
#   · Spearman_Grouped_Bars.png            — grouped horizontal bar chart ρ_target vs ρ_HT
#   · PValues_by_Variable.png              — horizontal bar chart of intra-class p-values
#   · Verification_TrainVal_Partition.png  — train/val balance check for stratification variables
#
# Significance threshold: α = 0.05
# Input:  IAvsHT_VALLADOLID_2023.xlsx  (sheets: Train, Val, Validation_Split)
#         IAvsHT_VALLADOLID_2025.xlsx · IAvsHT_GRANADA.xlsx · IAvsHT_SALAMANCA.xlsx
#         Ranking_global_TFG.xlsx      (sheets: Local, Mixed, LOCO)
# Output: Analisis_estadistico_TFG.xlsx
##############################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, chi2_contingency, spearmanr, kruskal
from itertools import combinations

# ── Output file ───────────────────────────────────────────────────────────────

OUTPUT_FILE = 'IAvsHT_Statistical_Analysis.xlsx'

# ── Input files ───────────────────────────────────────────────────────────────

FILE_VLD_2023 = 'IAvsHT_VALLADOLID_2023.xlsx'   # Valladolid 2023 cohort (N=702)
FILE_VLD_2025 = 'IAvsHT_VALLADOLID_2025.xlsx'   # Valladolid 2025 external test cohort (N=150)
FILE_SALAMANCA = 'IAvsHT_SALAMANCA.xlsx'         # Salamanca external test cohort (N=162)
FILE_GRANADA   = 'IAvsHT_GRANADA.xlsx'           # Granada external test cohort (N=150)
RANKING_EXCEL  = 'Ranking_global_TFG.xlsx'       # Pre-computed per-methodology classifier rankings

ALPHA = 0.05   # Significance level used throughout the analysis

# ════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_and_clean(filepath, sheet):
    """
    Load a single Excel sheet and normalise the target column name to 'ETIQUETA'.
    NaN values are replaced with 0 to avoid downstream errors.
    Returns an empty DataFrame if the file or sheet cannot be read.
    """
    try:
        df = pd.read_excel(filepath, sheet_name=sheet).fillna(0)
    except Exception as e:
        print(f"  Error loading {filepath} — sheet '{sheet}': {e}")
        return pd.DataFrame()

    # Locate the target column: prefer an exact or partial match on 'ETIQUETA',
    # otherwise fall back to the last column (assumed to be the ground truth).
    for col in df.columns:
        if 'ETIQUETA' in str(col).upper():
            df.rename(columns={col: 'ETIQUETA'}, inplace=True)
            return df
    if df.columns.size > 0:
        df.rename(columns={df.columns[-1]: 'ETIQUETA'}, inplace=True)
    return df


print("Loading datasets...")
df_train = load_and_clean(FILE_VLD_2023, 'Train')   # Training set   — Valladolid 2023 (551 patients)
df_val   = load_and_clean(FILE_VLD_2023, 'Val')     # Validation set — Valladolid 2023 (151 patients)
df_vall  = load_and_clean(FILE_VLD_2025, 0)         # External test  — Valladolid 2025
df_sala  = load_and_clean(FILE_SALAMANCA, 0)        # External test  — Salamanca
df_gran  = load_and_clean(FILE_GRANADA, 0)          # External test  — Granada

# Full Valladolid 2023 cohort (train + val combined, used for global statistics)
df_global = pd.concat([df_train, df_val], ignore_index=True)

print(f"  Train: {len(df_train)} | Val: {len(df_val)} | "
      f"Valladolid 2025: {len(df_vall)} | Salamanca: {len(df_sala)} | Granada: {len(df_gran)}")


def split_by_class(df):
    """Split a DataFrame into class-0 (No surgery) and class-1 (Surgery) subsets."""
    return df[df['ETIQUETA'] == 0], df[df['ETIQUETA'] == 1]

# Class-stratified subsets for each cohort
train_0, train_1 = split_by_class(df_train)
val_0,   val_1   = split_by_class(df_val)
vall_0,  vall_1  = split_by_class(df_vall)
sala_0,  sala_1  = split_by_class(df_sala)
gran_0,  gran_1  = split_by_class(df_gran)

# ════════════════════════════════════════════════════════════════════════════
# VARIABLE DEFINITIONS
# ════════════════════════════════════════════════════════════════════════════
# Each entry is a tuple: (display_label, column_name, variable_type)
#   · display_label  — human-readable label for the output tables
#   · column_name    — exact column name in the Excel files
#                      None → section header row (no data, used for table structure)
#   · variable_type  — 'cont' for continuous | 'bin' for binary | None for headers

VARIABLES = [
    ("Demographic variables",                               None,                                          None),
    ("Age",                                                 "EDAD",                                        "cont"),
    ("Age > 75",                                            "EDAD>75",                                     "bin"),
    ("Sex (Female)",                                        "SEXO ",                                       "bin"),

    ("Coronary anatomy and structural disease",             None,                                          None),
    ("Previous cardiac surgery",                            "CIRUGÍA CARDIACA PREVIA",                     "bin"),
    ("Left main involvement",                               "TRONCO",                                      "bin"),
    ("Single-vessel disease",                               "1 VASO",                                      "bin"),
    ("Two-vessel disease",                                  "2 VASOS",                                     "bin"),
    ("Three-vessel disease",                                "3 VASOS",                                     "bin"),
    ("Distal vessels quality",                              "LECHOS DISTALES (0=buenos, 1=regular, 2= malos)", "cont"),
    ("Suitable for angioplasty",                            "SUSCEPTIBLE DE ANGIOPLASTIA",                 "bin"),
    ("Number of valves affected",                           "Nº VÁLVULAS (0,1,2,3)",                       "cont"),
    ("Cardiac tumours",                                     "TUMORES CARDIACOS",                           "bin"),
    ("Congenital conditions",                               "CONGÉNITOS",                                  "bin"),

    ("Cardiac status and haemodynamic parameters",          None,                                          None),
    ("Critical preoperative status",                        "ESTADO PREOPERATORIO CRÍTICO",                "bin"),
    ("Class IV angina",                                     "ANGINA CLASE IV",                             "bin"),
    ("Recent MI",                                           "IAM RECIENTE",                                "bin"),
    ("Active endocarditis",                                 "ENDOCARDITIS ACTIVA",                         "bin"),
    ("LVEF 31–50%",                                         "FEVI 31-50%",                                 "bin"),
    ("LVEF < 30%",                                          "FEVI<30%",                                    "bin"),
    ("RVEF < 30%",                                          "FEVD<30%",                                    "bin"),
    ("TAPSE < 13 mm",                                       "TAPSE<13",                                    "bin"),
    ("Severe PAH (>55 mmHg)",                               "HTP SEVERA (>55mmHg)",                        "bin"),

    ("Extracardiac comorbidities and risk factors",         None,                                          None),
    ("Diabetes",                                            "DIABETES",                                    "bin"),
    ("Severe COPD",                                         "EPOC SEVERO",                                 "bin"),
    ("Extracardiac arteriopathy",                           "ARTERIOPATÍA EXTRACARDIACA",                  "bin"),
    ("Poor mobility",                                       "POBRE MOVILIDAD",                             "bin"),
    ("Severe stroke / Coma",                                "ICTUS SEVERO/COMA",                           "bin"),
    ("Preoperative creatinine > 2 mg/dL",                   "CREATININA PREOPERATORIOA>2",                 "bin"),
    ("Dementia",                                            "DEMENCIA",                                    "bin"),
    ("Active neoplasms",                                    "NEOPLASIAS ACTIVAS",                          "bin"),
    ("Active infections",                                   "INFECCIONES ACTIVAS",                         "bin"),
    ("Active COVID",                                        "COVID ACTIVO",                                "bin"),

    ("Classification of proposed intervention",             None,                                          None),
    ("Isolated coronary surgery",                           "CIRUGÍA CORONARIA AISLADA",                   "bin"),
    ("Isolated aortic surgery",                             "CIRUGÍA DE AORTA AISLADA",                    "bin"),
    ("Isolated aortic valve surgery",                       "CIRUGÍA DE VÁLVULA AÓRTICA AISLADA ",         "bin"),
    ("Monovalvular + coronary surgery",                     "MONOVALVULAR + CORONARIO",                    "bin"),
    ("Bivalvular + coronary surgery",                       "BIVALVULAR + CORONARIO",                      "bin"),
    ("Trivalvular + coronary surgery",                      "TRIVALVULAR + CORONARIO",                     "bin"),
    ("Aortic + valve surgery",                              "CIRUGIA DE AORTA + VALVULAR",                 "bin"),
    ("Aortic + coronary surgery",                           "CIRUGÍA DE AORTA + CORONARIO",                "bin"),
    ("Aortic + valve + coronary surgery",                   "CIRUGÍA DE AORTA + VALVULAR + CORONARIO",     "bin"),
    ("Other with CPB",                                      "OTROS CON CEC",                               "bin"),
]

# ════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def find_col(df, col_name):
    """
    Locate a column in df by exact match first, then by case-insensitive substring.
    Returns the matched column name or None if not found.
    """
    if col_name in df.columns:
        return col_name
    matches = [c for c in df.columns if col_name.strip().lower() in str(c).strip().lower()]
    return matches[0] if matches else None


def descriptive_stat(df_subset, col_name, var_type):
    """
    Compute the appropriate descriptive statistic for a given variable type:
      · continuous → median [Q1 – Q3]
      · binary     → count of positives (percentage)
    Returns 'N/A' if the column is missing or the series is empty.
    """
    real_col = find_col(df_subset, col_name)
    if real_col is None:
        return "N/A"
    s = pd.to_numeric(df_subset[real_col], errors='coerce').dropna()
    if s.empty:
        return "N/A"
    if var_type == "cont":
        return f"{s.median():.1f} [{s.quantile(0.25):.1f} – {s.quantile(0.75):.1f}]"
    return f"{int((s == 1).sum())} ({(s == 1).mean() * 100:.1f}%)"


def intraclass_pvalue(df, col_name, var_type):
    """
    Compute the intra-class p-value (class 0 vs class 1) for a single variable:
      · continuous → Mann-Whitney U (two-sided, non-parametric)
      · binary     → Pearson Chi-squared (2×2 contingency table)
    Returns a formatted label string and the raw numeric p-value.
    'N/A' / np.nan are returned when the test cannot be applied.
    """
    if df.empty:
        return "N/A", np.nan

    real_col = find_col(df, col_name)
    if real_col is None:
        return "N/A", np.nan

    s0 = pd.to_numeric(df[df['ETIQUETA'] == 0][real_col], errors='coerce').dropna()
    s1 = pd.to_numeric(df[df['ETIQUETA'] == 1][real_col], errors='coerce').dropna()

    if s0.empty or s1.empty:
        return "N/A", np.nan

    try:
        if var_type == "cont":
            # Non-parametric test — no normality assumption required
            _, p = mannwhitneyu(s0, s1, alternative='two-sided')
        else:
            # Chi-squared on the 2×2 contingency table (no Yates correction)
            table = pd.crosstab(df[real_col], df['ETIQUETA'])
            if table.shape != (2, 2):
                return "N/A", np.nan
            _, p, _, _ = chi2_contingency(table, correction=False)
    except Exception:
        return "N/A", np.nan

    if np.isnan(p):
        return "N/A", np.nan

    # Format: flag significant p-values with an asterisk
    label = "< 0.001 *" if p < 0.001 else (f"{p:.3f} *" if p < 0.05 else f"{p:.3f}")
    return label, p


def format_p(p):
    """Format a p-value for display, adding an asterisk if significant."""
    if np.isnan(p):
        return "N/A"
    return "< 0.001 *" if p < 0.001 else (f"{p:.3f} *" if p < 0.05 else f"{p:.3f}")

# ════════════════════════════════════════════════════════════════════════════
# SHEET 1 — DESCRIPTIVE TABLE
# ════════════════════════════════════════════════════════════════════════════
print("\nGenerating descriptive table...")

# One column per class per cohort
DESC_COLS = [
    "Clinical variable",
    f"TRAIN No surgery (N={len(train_0)})",  f"TRAIN Surgery (N={len(train_1)})",
    f"VAL No surgery (N={len(val_0)})",       f"VAL Surgery (N={len(val_1)})",
    f"VALLADOLID No surgery (N={len(vall_0)})", f"VALLADOLID Surgery (N={len(vall_1)})",
    f"SALAMANCA No surgery (N={len(sala_0)})", f"SALAMANCA Surgery (N={len(sala_1)})",
    f"GRANADA No surgery (N={len(gran_0)})",  f"GRANADA Surgery (N={len(gran_1)})",
]

rows_desc = []
for label, col, vtype in VARIABLES:
    if vtype is None:
        # Section header row — single label, remaining cells empty
        rows_desc.append([f"── {label} ──"] + [""] * 10)
    else:
        suffix = " (Median [IQR])" if vtype == "cont" else " — N (%)"
        rows_desc.append([
            label + suffix,
            descriptive_stat(train_0, col, vtype), descriptive_stat(train_1, col, vtype),
            descriptive_stat(val_0,   col, vtype), descriptive_stat(val_1,   col, vtype),
            descriptive_stat(vall_0,  col, vtype), descriptive_stat(vall_1,  col, vtype),
            descriptive_stat(sala_0,  col, vtype), descriptive_stat(sala_1,  col, vtype),
            descriptive_stat(gran_0,  col, vtype), descriptive_stat(gran_1,  col, vtype),
        ])

df_desc = pd.DataFrame(rows_desc, columns=DESC_COLS)

# ════════════════════════════════════════════════════════════════════════════
# SHEET 2 — INTER-COHORT COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
# Goal: detect whether the 5 cohorts are homogeneous within each clinical class.
#
# Method:
#   1. Kruskal-Wallis (H) across k=5 groups (Train, Val, Valladolid 2025,
#      Salamanca, Granada) per variable and class.
#        · continuous variables → raw numeric value
#        · binary variables     → 0/1 value (KW applicable as non-parametric
#                                 proportion test across k groups)
#   2. If p_KW < 0.05 → pairwise post-hoc Mann-Whitney U (two-sided) for all
#      C(5,2)=10 cohort pairs. P-values are reported without Bonferroni
#      correction in the table (discussed in the thesis text).
# ════════════════════════════════════════════════════════════════════════════
print("Generating inter-cohort comparison table (Kruskal-Wallis + post-hoc Mann-Whitney)...")

COHORTS = [
    ("Train",          df_train),
    ("Val",            df_val),
    ("Valladolid",     df_vall),
    ("Salamanca",      df_sala),
    ("Granada",        df_gran),
]
COHORT_NAMES = [n for n, _ in COHORTS]
COHORT_PAIRS = list(combinations(range(len(COHORT_NAMES)), 2))   # All C(5,2)=10 index pairs


def class_series(df, col_name, target_class):
    """Extract a clean numeric Series for a given class from a DataFrame."""
    real_col = find_col(df, col_name)
    if real_col is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[df['ETIQUETA'] == target_class][real_col], errors='coerce').dropna()


def kw_and_posthoc(col_name, target_class):
    """
    Run Kruskal-Wallis across all 5 cohorts for a given variable and class.
    If KW is significant (p < 0.05), run pairwise Mann-Whitney post-hoc tests.

    Returns:
        kw_label  : str  — formatted KW p-value
        kw_p      : float
        posthoc   : dict {(i, j): str} — formatted MW p-values (empty if KW not significant)
    """
    groups = [class_series(df, col_name, target_class).values for _, df in COHORTS]

    # Keep only groups with at least one observation
    valid = [(i, g) for i, g in enumerate(groups) if len(g) >= 1]
    if len(valid) < 2:
        return "N/A", np.nan, {}

    # KW requires at least two groups with more than one unique value
    with_variance = [(i, g) for i, g in valid if len(np.unique(g)) > 1]
    if len(with_variance) < 2:
        # All groups constant — no possible difference
        return "1.000", 1.0, {}

    try:
        _, kw_p = kruskal(*[g for _, g in valid])
    except Exception:
        return "N/A", np.nan, {}

    if np.isnan(kw_p):
        return "N/A", np.nan, {}

    kw_label = format_p(kw_p)
    posthoc  = {}

    if kw_p < ALPHA:
        # Post-hoc: Mann-Whitney for all cohort pairs with sufficient data
        for i, j in COHORT_PAIRS:
            gi, gj = groups[i], groups[j]
            if len(gi) < 1 or len(gj) < 1:
                posthoc[(i, j)] = "N/A"
                continue
            try:
                _, p_mw = mannwhitneyu(gi, gj, alternative='two-sided')
                posthoc[(i, j)] = format_p(p_mw)
            except Exception:
                posthoc[(i, j)] = "N/A"

    return kw_label, kw_p, posthoc


# Build column headers: Variable | KW Class0 | KW Class1 | [MW pair_ij Class0 | Class1] × 10
KW_COLS = ["Clinical variable", "KW p-value Class 0 (No surgery)", "KW p-value Class 1 (Surgery)"]
for i, j in COHORT_PAIRS:
    pair_label = f"{COHORT_NAMES[i]} vs {COHORT_NAMES[j]}"
    KW_COLS += [f"MW {pair_label} — Class 0", f"MW {pair_label} — Class 1"]

rows_kw = []
for label, col, vtype in VARIABLES:
    if vtype is None:
        rows_kw.append([f"── {label} ──"] + [""] * (len(KW_COLS) - 1))
        continue

    kw_lbl0, kw_p0, ph0 = kw_and_posthoc(col, target_class=0)
    kw_lbl1, kw_p1, ph1 = kw_and_posthoc(col, target_class=1)

    suffix = " (Median [IQR])" if vtype == "cont" else " — N (%)"
    row = [label + suffix, kw_lbl0, kw_lbl1]

    for i, j in COHORT_PAIRS:
        # Post-hoc cells are filled only when KW was significant; otherwise a dash is shown
        row.append(ph0.get((i, j), "–") if kw_p0 < ALPHA else "–")
        row.append(ph1.get((i, j), "–") if kw_p1 < ALPHA else "–")

    rows_kw.append(row)

df_kw = pd.DataFrame(rows_kw, columns=KW_COLS)

# ════════════════════════════════════════════════════════════════════════════
# SHEET 3 — INTRA-CLASS P-VALUES
# ════════════════════════════════════════════════════════════════════════════
print("Generating intra-class p-value table...")

PVAL_DATASETS = [
    ("TRAIN",           df_train),
    ("VAL",             df_val),
    ("TEST Valladolid", df_vall),
    ("TEST Salamanca",  df_sala),
    ("TEST Granada",    df_gran),
]

PVAL_COLS = ["Clinical variable"] + [f"p-value {n}" for n, _ in PVAL_DATASETS]
rows_pval = []

for label, col, vtype in VARIABLES:
    if vtype is None:
        rows_pval.append([f"── {label} ──"] + [""] * len(PVAL_DATASETS))
    else:
        row = [label]
        for _, df in PVAL_DATASETS:
            lbl, _ = intraclass_pvalue(df, col, vtype)
            row.append(lbl)
        rows_pval.append(row)

df_pval = pd.DataFrame(rows_pval, columns=PVAL_COLS)

# ════════════════════════════════════════════════════════════════════════════
# SHEET 4 — SPEARMAN CORRELATION
# ─────────────────────────────────────────────────────────────────────────────
# Computes Spearman ρ for each variable against:
#   (a) ETIQUETA  — ground truth label (consensus surgical decision)
#   (b) OBJETIVO HT — Heart Team clinical decision
# Computed on the full Valladolid 2023 cohort (train + val).
# Δ = |ρ_target| − |ρ_HT| quantifies divergence between AI and HT relevance.
# ════════════════════════════════════════════════════════════════════════════
print("Generating Spearman correlation table...")

# Locate the Heart Team decision column (flexible search)
col_ht = 'OBJETIVO HT'
if col_ht not in df_global.columns:
    matches = [c for c in df_global.columns if 'objetivo' in str(c).lower()]
    col_ht  = matches[0] if matches else None

rows_spear = []
if col_ht:
    y_label = df_global['ETIQUETA']    # Ground truth
    y_ht    = df_global[col_ht]        # Heart Team decision

    for label, col, vtype in VARIABLES:
        if vtype is None:
            continue   # Skip section headers

        real_col = find_col(df_global, col)
        if real_col is None:
            continue

        x = pd.to_numeric(df_global[real_col], errors='coerce').fillna(0)

        if len(np.unique(x)) > 1:
            rho_lbl, p_lbl = spearmanr(x, y_label)
            rho_ht,  p_ht  = spearmanr(x, y_ht)
        else:
            # Constant feature — no correlation possible
            rho_lbl, p_lbl = 0.0, 1.0
            rho_ht,  p_ht  = 0.0, 1.0

        # Replace any NaN results with neutral values
        rho_lbl = 0.0 if np.isnan(rho_lbl) else rho_lbl
        rho_ht  = 0.0 if np.isnan(rho_ht)  else rho_ht
        p_lbl   = 1.0 if np.isnan(p_lbl)   else p_lbl
        p_ht    = 1.0 if np.isnan(p_ht)    else p_ht

        rows_spear.append([
            label,
            round(rho_lbl, 3), p_lbl,
            round(rho_ht,  3), p_ht,
            round(abs(abs(rho_lbl) - abs(rho_ht)), 3)   # Divergence metric Δ
        ])

df_spear = pd.DataFrame(rows_spear, columns=[
    "Clinical variable",
    "ρ with ETIQUETA (Target)", "p-value Target",
    "ρ with HT Decision",       "p-value HT",
    "Divergence (Δ)"
]).sort_values("Divergence (Δ)", ascending=False).reset_index(drop=True)

# ── Figure A — Spearman scatter: ρ_target vs ρ_HT ───────────────────────────
# Each point is a variable. Points on the diagonal indicate perfect agreement.
# Points far from the diagonal reveal variables weighted differently by AI vs HT.
if len(df_spear) > 0:
    rho_tgt  = df_spear["ρ with ETIQUETA (Target)"].values
    rho_ht_v = df_spear["ρ with HT Decision"].values
    names_s  = df_spear["Clinical variable"].values
    deltas   = df_spear["Divergence (Δ)"].values

    fig, ax = plt.subplots(figsize=(8, 8))

    # Colour encodes divergence magnitude: green = agreement, red = divergence
    sc = ax.scatter(rho_ht_v, rho_tgt,
                    c=deltas, cmap='RdYlGn_r',
                    s=80, edgecolors='grey', linewidths=0.5, zorder=3)
    plt.colorbar(sc, ax=ax, label='Divergence (Δ)', shrink=0.8)

    lim = max(abs(rho_tgt).max(), abs(rho_ht_v).max()) + 0.05
    ax.plot([-lim, lim], [-lim, lim],
            color='steelblue', linestyle='--', linewidth=1.2,
            label='Perfect agreement (ρ_target ≈ ρ_HT)')
    ax.axhline(0, color='grey', linewidth=0.5, linestyle=':')
    ax.axvline(0, color='grey', linewidth=0.5, linestyle=':')

    # Annotate only the top 15% most divergent variables
    from adjustText import adjust_text
    threshold = np.percentile(deltas, 85)
    texts = [
        ax.text(rx, ry, name, fontsize=8.5, color='#333333')
        for name, rx, ry, d in zip(names_s, rho_ht_v, rho_tgt, deltas)
        if d >= threshold
    ]
    adjust_text(texts, ax=ax,
                arrowprops=dict(arrowstyle='-', color='#aaaaaa', lw=0.6),
                expand=(1.3, 1.5),
                force_text=(0.3, 0.5))

    ax.set_xlabel("ρ with Heart Team Decision", fontsize=12)
    ax.set_ylabel("ρ with Target (Ground Truth)", fontsize=12)
    ax.set_title("Spearman Correlation: Ground Truth vs Heart Team Decision",
                 fontweight='bold', fontsize=12, pad=12)
    ax.legend(fontsize=9)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig('Spearman_Scatter.png', dpi=300, bbox_inches='tight')
    plt.show()

# ── Figure B — Grouped horizontal bars: ρ_target vs ρ_HT per variable ────────
# Variables sorted by |Δ| descending (highest divergence at the top).
# Translucent bars indicate non-significant correlations (p ≥ 0.05).
if len(df_spear) > 0:
    df_plot   = df_spear.sort_values("Divergence (Δ)", ascending=True)
    labels_b  = df_plot["Clinical variable"].values
    rho_tgt_b = df_plot["ρ with ETIQUETA (Target)"].values
    rho_ht_b  = df_plot["ρ with HT Decision"].values
    p_tgt_b   = df_plot["p-value Target"].values
    p_ht_b    = df_plot["p-value HT"].values

    n     = len(labels_b)
    fig, ax = plt.subplots(figsize=(10, max(10, n * 0.45)))

    y      = np.arange(n)
    height = 0.35

    bars1 = ax.barh(y + height / 2, rho_tgt_b, height=height,
                    color='#1f77b4', edgecolor='white', linewidth=0.4,
                    label='ρ with Ground Truth')
    bars2 = ax.barh(y - height / 2, rho_ht_b,  height=height,
                    color='#ff7f0e', edgecolor='white', linewidth=0.4,
                    label='ρ with HT Decision')

    # Make non-significant bars translucent to draw attention to robust correlations
    for bar, p in zip(bars1, p_tgt_b):
        if p >= ALPHA:
            bar.set_alpha(0.3)
    for bar, p in zip(bars2, p_ht_b):
        if p >= ALPHA:
            bar.set_alpha(0.3)

    ax.text(0.02, 0.02,
            '* Translucent bars indicate non-significant\n  correlations (p ≥ 0.05)',
            transform=ax.transAxes, fontsize=10, fontweight='bold',
            ha='left', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#cccccc', alpha=0.9))

    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels_b, fontsize=8.5)
    ax.set_xlabel("Spearman correlation coefficient (ρ)", fontsize=11)
    ax.set_title("Spearman Correlation — Ground Truth vs HT Decision",
                 fontsize=20, fontweight='bold', pad=12)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig('Spearman_Grouped_Bars.png', dpi=300, bbox_inches='tight')
    plt.show()

# ════════════════════════════════════════════════════════════════════════════
# EXPORT ALL SHEETS TO EXCEL
# ════════════════════════════════════════════════════════════════════════════
print(f"\nSaving all sheets to '{OUTPUT_FILE}'...")

with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    df_desc.to_excel( writer, sheet_name='Descriptive_Table',       index=False)
    df_kw.to_excel(   writer, sheet_name='Inter_Cohort_Comparison', index=False)
    df_pval.to_excel( writer, sheet_name='P_Values',                index=False)
    df_spear.to_excel(writer, sheet_name='Spearman_Correlation',    index=False)

print("Sheets written: Descriptive_Table | Inter_Cohort_Comparison | P_Values | Spearman_Correlation")

# ════════════════════════════════════════════════════════════════════════════
# FIGURE C — INTRA-CLASS P-VALUE BAR CHART
# ─────────────────────────────────────────────────────────────────────────────
# One horizontal bar per variable, computed on the full Valladolid 2023 cohort.
# Colour: red if p < 0.05 (significant difference between classes), blue otherwise.
# Dashed red line at α = 0.05 for reference.
# ════════════════════════════════════════════════════════════════════════════

names_pval  = []
values_pval = []

for label, col, vtype in VARIABLES:
    if vtype is None:
        continue
    _, p_num = intraclass_pvalue(df_global, col, vtype)
    if not np.isnan(p_num):
        names_pval.append(label)
        values_pval.append(p_num)

if values_pval:
    # Reverse order so the first variable appears at the top of the chart
    names_plot  = names_pval[::-1]
    values_plot = values_pval[::-1]
    colours     = ['#d62728' if p < ALPHA else '#1f77b4' for p in values_plot]

    fig, ax = plt.subplots(figsize=(10, max(10, len(names_plot) * 0.38)))
    y = np.arange(len(names_plot))

    ax.barh(y, values_plot, color=colours, edgecolor='white', linewidth=0.5, height=0.7)
    ax.axvline(x=ALPHA, color='red', linestyle='dashed', linewidth=1.5)
    ax.text(ALPHA + 0.002, len(names_plot) - 0.5,
            f'Threshold p = {ALPHA}',
            color='red', fontsize=9, fontweight='bold', va='top', ha='left')
    ax.set_yticks(y)
    ax.set_yticklabels(names_plot, fontsize=9)
    ax.set_xlabel('p-value', fontsize=11)
    ax.set_xlim(0, 1.08)
    ax.set_title('Intra-class p-values by clinical variable',
                 fontsize=13, fontweight='bold', pad=12)
    ax.grid(axis='x', linestyle='--', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig('PValues_by_Variable.png', dpi=300, bbox_inches='tight')
    plt.show()

    n_sig = sum(1 for p in values_pval if p < ALPHA)
    print(f"  Significant variables (p < {ALPHA}): {n_sig}/{len(values_pval)}")

# ════════════════════════════════════════════════════════════════════════════
# FIGURE D — TRAIN / VAL PARTITION VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────
# 2×2 grid showing the distribution of the 4 stratification control variables
# across the train and validation subsets.
#   · Continuous (Age)  → side-by-side boxplots
#   · Binary            → percentage bar charts with count annotations
# P-values are read directly from the 'Validation_Split' sheet generated by
# IAvsHT_Validation_Split.py to ensure consistency with the published table.
# Colour coding: green = p ≥ 0.05 (balanced), red = p < 0.05 (imbalanced).
# ════════════════════════════════════════════════════════════════════════════

# Load p-values from the validation sheet (pre-computed by IAvsHT_Validation_Split.py)
try:
    df_vsplit = pd.read_excel(FILE_VLD_2023, sheet_name='Validation_Split')
    pval_dict = {
        str(v).strip().upper(): float(p)
        for v, p in zip(df_vsplit['Variable'], df_vsplit['p-value'])
    }
    print("  Partition p-values loaded from 'Validation_Split'.")
except Exception as e:
    print(f"  WARNING: could not read 'Validation_Split' ({e}). "
          f"Run IAvsHT_Validation_Split.py first.")
    pval_dict = {}

# Variables used for stratification during split creation
STRATIFICATION_VARS = [
    ("Age",                         "EDAD",                         "cont"),
    ("Critical preoperative status","ESTADO PREOPERATORIO CRÍTICO",  "bin"),
    ("Aortic + valve surgery",      "CIRUGIA DE AORTA + VALVULAR",  "bin"),
    ("Bivalvular + coronary surgery","BIVALVULAR + CORONARIO",       "bin"),
]

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
axes      = axes.flatten()
COLOURS   = ['#4C72B0', '#d62728']   # Blue = Train, Red = Val

for ax, (label, col, vtype) in zip(axes, STRATIFICATION_VARS):

    col_t = find_col(df_train, col)
    col_v = find_col(df_val,   col)

    if col_t is None or col_v is None:
        ax.set_visible(False)
        continue

    s_train = pd.to_numeric(df_train[col_t], errors='coerce').dropna()
    s_val   = pd.to_numeric(df_val[col_v],   errors='coerce').dropna()

    # Retrieve pre-computed p-value — exact match first, then partial
    pv = pval_dict.get(col_t.strip().upper(), np.nan)
    if np.isnan(pv):
        for k, v in pval_dict.items():
            if col.strip().upper() in k or k in col.strip().upper():
                pv = v
                break

    p_label = (f'p = {pv:.4f}' if not np.isnan(pv) and pv >= 0.001
               else ('p < 0.001' if not np.isnan(pv) else 'p = N/A'))
    p_colour = '#2ca02c' if not np.isnan(pv) and pv >= ALPHA else '#d62728'

    if vtype == "cont":
        # Continuous variable: side-by-side boxplots
        bp = ax.boxplot(
            [s_train.values, s_val.values],
            patch_artist=True, widths=0.45,
            medianprops=dict(color='white', linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            flierprops=dict(marker='o', markersize=3, alpha=0.4)
        )
        for patch, colour in zip(bp['boxes'], COLOURS):
            patch.set_facecolor(colour)
            patch.set_alpha(0.8)
        ax.set_xticks([1, 2])
        ax.set_xticklabels([f'Train\n(N={len(s_train)})', f'Val\n(N={len(s_val)})'], fontsize=9)
        ax.set_ylabel('Years', fontsize=9)
    else:
        # Binary variable: percentage bars with count annotations
        pct_t = (s_train == 1).mean() * 100
        pct_v = (s_val   == 1).mean() * 100
        n_t   = int((s_train == 1).sum())
        n_v   = int((s_val   == 1).sum())

        bars = ax.bar([0, 1], [pct_t, pct_v],
                      color=COLOURS, edgecolor='white', linewidth=0.5, width=0.45)
        for bar, pct, n in zip(bars, [pct_t, pct_v], [n_t, n_v]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f'{pct:.1f}%\n(n={n})',
                    ha='center', va='bottom', fontsize=8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f'Train\n(N={len(s_train)})', f'Val\n(N={len(s_val)})'], fontsize=9)
        ax.set_ylabel('Positive patients (%)', fontsize=9)
        ax.set_ylim(0, max(pct_t, pct_v) * 1.4 + 3)

    ax.set_title(label, fontsize=11, fontweight='bold', pad=6)
    ax.text(0.97, 0.97, p_label,
            transform=ax.transAxes, fontsize=9, color=p_colour,
            ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=p_colour, alpha=0.85))
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)

fig.suptitle('Distribution of stratification variables across Train and Validation sets',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('Verification_TrainVal_Partition.png', dpi=300, bbox_inches='tight')
plt.show()

# ════════════════════════════════════════════════════════════════════════════
# SHEET 5 — GLOBAL CLASSIFIER RANKING (Borda count)
# ─────────────────────────────────────────────────────────────────────────────
# Position-based scoring across 6 individual rankings:
#   Local methodology, Mixed methodology, and LOCO × 4 cohorts.
# Scoring: 12 pts → 1st place, 11 pts → 2nd, ..., 1 pt → 12th.
# Classifiers are sorted by total accumulated points (descending).
# ════════════════════════════════════════════════════════════════════════════
print("\nComputing global classifier ranking (Borda count)...")

# ── Helpers for robust column name handling ───────────────────────────────────

def get_col(df, candidates):
    """Return the first column name from candidates found in df, or raise KeyError."""
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of {candidates} found in {list(df.columns)}")


def normalise(df):
    """Rename 'Clasificador'→'Model' and 'AUC-ROC'/'AUC'→'AUC' for consistency."""
    col_model = get_col(df, ['Modelo', 'Clasificador'])
    col_auc   = get_col(df, ['AUC', 'AUC-ROC'])
    return df.rename(columns={col_model: 'Model', col_auc: 'AUC'})


def ranking_from_rows(df):
    """
    Build a {model_name: rank_position} dict from the row order of the DataFrame.
    Rows where AUC is non-numeric (e.g. section headers) are excluded.
    """
    df = df.copy()
    df['Model'] = df['Model'].astype(str).str.strip()
    df = df[pd.to_numeric(df['AUC'], errors='coerce').notna()].reset_index(drop=True)
    return {row['Model']: idx + 1 for idx, row in df.iterrows()}


# ── Load and normalise ranking sheets ─────────────────────────────────────────
df_local  = normalise(pd.read_excel(RANKING_EXCEL, sheet_name='Local'))
df_mixed = normalise(pd.read_excel(RANKING_EXCEL, sheet_name='Mixed'))
df_loco   = normalise(pd.read_excel(RANKING_EXCEL, sheet_name='LOCO'))

ranking_local  = ranking_from_rows(df_local)
ranking_mixed = ranking_from_rows(df_mixed)

# LOCO sheet: section headers are rows where AUC is non-numeric
LOCO_COHORTS = ['Valladolid (2023)', 'Valladolid (2025)', 'Granada', 'Salamanca']

df_loco['AUC']   = pd.to_numeric(df_loco['AUC'], errors='coerce')
df_loco['Model'] = df_loco['Model'].astype(str).str.strip()

# Group LOCO rows by cohort (rows whose 'Model' value matches a cohort name are headers)
loco_groups = {c: [] for c in LOCO_COHORTS}
current     = None
for _, row in df_loco.iterrows():
    if row['Model'] in LOCO_COHORTS:
        current = row['Model']
    elif current is not None and not pd.isna(row['AUC']):
        loco_groups[current].append(row)

loco_rankings = {
    cohort: {row['Model']: idx + 1 for idx, row in pd.DataFrame(rows).reset_index(drop=True).iterrows()}
    for cohort, rows in loco_groups.items()
}

# ── Validate: all models present in all rankings ──────────────────────────────
all_models = set(ranking_local) | set(ranking_mixed) | set().union(*[set(r) for r in loco_rankings.values()])
for name, rk in [('Local', ranking_local), ('Mixed', ranking_mixed),
                 *[(f'LOCO {c}', loco_rankings[c]) for c in LOCO_COHORTS]]:
    missing = all_models - set(rk)
    if missing:
        raise ValueError(f"Models missing from '{name}' ranking: {sorted(missing)}")

all_models = sorted(all_models)

# ── Compute Borda scores ──────────────────────────────────────────────────────
N_MODELS = 12   # Total number of classifiers evaluated

RANKING_COLS = {
    'Local':          ranking_local,
    'Mixed':          ranking_mixed,
    'LOCO V23':       loco_rankings['Valladolid (2023)'],
    'LOCO V25':       loco_rankings['Valladolid (2025)'],
    'LOCO Granada':   loco_rankings['Granada'],
    'LOCO Salamanca': loco_rankings['Salamanca'],
}

rows_borda = []
for model in all_models:
    row   = {'Classifier': model}
    total = 0
    for col_name, rk in RANKING_COLS.items():
        pos = rk.get(model, N_MODELS)       # Default to last place if not found
        pts = N_MODELS - pos + 1
        row[col_name] = pts
        total        += pts
    row['Total'] = total
    rows_borda.append(row)

df_borda = (pd.DataFrame(rows_borda)
              .sort_values('Total', ascending=False)
              .reset_index(drop=True))
df_borda.insert(0, 'Position', range(1, len(df_borda) + 1))

# ── Write to Excel ────────────────────────────────────────────────────────────
with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    df_borda.to_excel(writer, sheet_name='Global_Ranking', index=False)

print(f"\n✓ Sheet 'Global_Ranking' added to {OUTPUT_FILE}")
print(df_borda.to_string(index=False))