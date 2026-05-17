##############################################################################
# IAvsHT — Mixed (Pooled) Training and Validation Strategy
# ─────────────────────────────────────────────────────────────────────────────
# Partitioning strategy: The four independent datasets are pooled together
# (N=1164) and iteratively split into a 70% Train, 15% Validation, and 15% 
# Internal Test set. The split is strictly stratified by four clinical 
# variables to ensure statistical balance (Welch/Chi2 p-values > 0.05).
#
# Data flow:
#   Pooled Cohort (N=1164)
#        ├── Train (70%, N≈815)
#        ├── Val   (15%, N≈175)
#        └── Test  (15%, N≈174) — Evaluated globally and broken down by center
#
# Pipeline (Steps 1–7):
#   1. Imports, constants and shared function definitions
#   2. Data loading: pooling + iterative stratified split
#   3. Preprocessing: zero-variance removal → collinearity pruning →
#      EDAD/EDAD>75 handling → StandardScaler → FCBF feature selection (tol=2.2)
#   4. Hyperparameter optimisation (RandomizedSearchCV, 200 iter, 5-fold CV)
#      with JSON cache invalidated automatically by pipeline fingerprint
#   5. Classifier ranking → top-3 → Platt calibration → Voting Ensemble →
#      model selection (AUC primary, Bal.Acc@Youden tiebreaker) →
#      Youden threshold optimisation
#   6. Test evaluation: confusion matrices + ROC curves (overall and per center)
#   7. SHAP explainability: KernelExplainer, K-Means background,
#      feature importance rankings + beeswarm plots (train and per center)
#
# Output directory : IAvsHT_mixed/
# Figures (PNG)    :
#   fcbf_tolerance_mixed.png               fcbf_su_ranking_mixed.png
#   matrices_comparison_multicentre_mixed.png
#   matrices_by_center_origin_mixed.png    roc_curves_mixed.png
#   shap_ranking_train_mixed.png           shap_beeswarm_train_mixed.png
#   shap_ranking_test_mixed.png            shap_beeswarm_test_mixed.png
# Excel            : IAvsHT_mixed.xlsx
##############################################################################

# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — IMPORTS, CONSTANTS AND SHARED FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

# ── Standard library ─────────────────────────────────────────────────────────
import os
import json
import hashlib
import json as _json
import re
import ast
import warnings

# ── Third-party ──────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
from scipy import stats

# ── scikit-learn ─────────────────────────────────────────────────────────────
from sklearn.calibration           import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble              import (AdaBoostClassifier, BaggingClassifier,
                                           ExtraTreesClassifier,
                                           GradientBoostingClassifier,
                                           RandomForestClassifier,
                                           VotingClassifier)
from sklearn.linear_model          import LogisticRegression
from sklearn.metrics               import (accuracy_score, balanced_accuracy_score,
                                           cohen_kappa_score, ConfusionMatrixDisplay,
                                           confusion_matrix, f1_score,
                                           matthews_corrcoef, mutual_info_score,
                                           precision_score, recall_score,
                                           roc_auc_score, roc_curve)
from sklearn.model_selection       import RandomizedSearchCV
from sklearn.neighbors             import KNeighborsClassifier
from sklearn.neural_network        import MLPClassifier
from sklearn.preprocessing         import StandardScaler
from sklearn.svm                   import SVC
from sklearn.tree                  import DecisionTreeClassifier

# ── Imbalanced-learn / XGBoost ───────────────────────────────────────────────
from imblearn.ensemble import RUSBoostClassifier
from xgboost           import XGBClassifier

# Suppress known non-critical warnings that clutter the output
warnings.filterwarnings('ignore', message='l1_ratio parameter is only used')
warnings.filterwarnings('ignore', message='Clustering metrics expects discrete values')
warnings.filterwarnings('ignore', message='One or more of the test scores are non-finite')

# ── Output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR = 'IAvsHT_mixed'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Information-theoretic helpers (used by FCBF) ─────────────────────────────

def entropy(vec):
    """Shannon entropy of a discrete vector (bits)."""
    _, counts = np.unique(vec, return_counts=True)
    probs = counts / len(vec)
    return -np.sum(probs * np.log2(probs + 1e-12))


def symmetrical_uncertainty(x, y):
    """
    Symmetrical Uncertainty (SU) between two discrete variables.
    SU(X,Y) = 2 · MI(X;Y) / (H(X) + H(Y))  ∈ [0, 1]
    """
    h_x = entropy(x)
    h_y = entropy(y)
    if (h_x + h_y) == 0:
        return 0.0
    mi_bits = mutual_info_score(x, y) / np.log(2)
    return mi_bits / (h_x + h_y)


def fcbf(X, y, threshold=0.0, tolerancia=1.0):
    """Fast Correlation-Based Filter — relaxed variant."""
    n_samples, n_features = X.shape

    su_target = []
    for i in range(n_features):
        su = symmetrical_uncertainty(X[:, i], y)
        if su >= threshold:
            su_target.append((su, i))
    su_target.sort(reverse=True, key=lambda t: t[0])

    candidates       = su_target[:]
    selected_indices = []

    while candidates:
        best_su, best_idx = candidates[0]
        selected_indices.append(best_idx)
        candidates.pop(0)

        remaining  = []
        pivot_vals = X[:, best_idx]
        for su_cand, idx_cand in candidates:
            su_between = symmetrical_uncertainty(pivot_vals, X[:, idx_cand])
            if su_between < su_cand * tolerancia:
                remaining.append((su_cand, idx_cand))
        candidates = remaining

    return sorted(selected_indices)


# ── Clinical performance metrics ──────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_prob=None, name=''):
    """Compute 11 clinical performance metrics from ground truth and predictions."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sens    = recall_score(y_true, y_pred, zero_division=0)
    spec    = tn / (tn + fp) if (tn + fp) > 0 else 0
    acc     = accuracy_score(y_true, y_pred)
    ppv     = precision_score(y_true, y_pred, zero_division=0)
    npv     = tn / (tn + fn) if (tn + fn) > 0 else 0
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1      = f1_score(y_true, y_pred, zero_division=0)
    auc_val = roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan
    kappa   = cohen_kappa_score(y_true, y_pred)
    mcc     = matthews_corrcoef(y_true, y_pred)

    return {
        'Model':         name,
        'Sensitivity':   round(sens,    4),
        'Specificity':   round(spec,    4),
        'Accuracy':      round(acc,     4),
        'PPV':           round(ppv,     4),
        'NPV':           round(npv,     4),
        'AUC-ROC':       round(auc_val, 4),
        'Bal. Accuracy': round(bal_acc, 4),
        'F1-Score':      round(f1,      4),
        'Kappa':         round(kappa,   4),
        'MCC':           round(mcc,     4),
    }


# ── JSON serialisation helpers for sklearn estimators ────────────────────────

def _estimator_to_str(estimator):
    cls    = type(estimator).__name__
    params = estimator.get_params()
    parts  = ', '.join(f'{k}={repr(v)}' for k, v in sorted(params.items()) if v is not None)
    return f'__estimator__{cls}({parts})'

def _str_to_estimator(s):
    m = re.match(r'__estimator__(\w+)\((.*)\)', s, re.DOTALL)
    if not m:
        return s
    cls_name, params_str = m.group(1), m.group(2)
    cls_map = {'DecisionTreeClassifier': DecisionTreeClassifier}
    cls     = cls_map.get(cls_name)
    if cls is None:
        raise ValueError(f'Unknown estimator during deserialisation: {cls_name}')
    kwargs = {}
    if params_str.strip():
        for match in re.finditer(r'(\w+)=(\'(?:[^\'\\\\]|\\\\.)*\'|"(?:[^"\\\\]|\\\\.)*"|[^,]+)', params_str):
            k, raw_v = match.group(1).strip(), match.group(2).strip()
            try:
                kwargs[k] = ast.literal_eval(raw_v)
            except Exception:
                kwargs[k] = raw_v
    return cls(**kwargs)

def params_to_json(params):
    safe = {}
    for k, v in params.items():
        if v is None:
            safe[k] = '__None__'
        elif isinstance(v, tuple):
            safe[k] = list(v)
        elif hasattr(v, 'get_params'):
            safe[k] = _estimator_to_str(v)
        else:
            safe[k] = v
    return safe

def params_from_json(params):
    restored = {}
    for k, v in params.items():
        if v == '__None__':
            restored[k] = None
        elif k == 'hidden_layer_sizes' and isinstance(v, list):
            restored[k] = tuple(v)
        elif isinstance(v, str) and v.startswith('__estimator__'):
            restored[k] = _str_to_estimator(v)
        else:
            restored[k] = v
    return restored

def compute_fingerprint(features_list, param_spaces):
    def _s(v):
        return sorted([str(x) for x in v]) if isinstance(v, list) else str(v)
    def _normalise_space(v):
        if isinstance(v, list):
            merged = {}
            for d in v:
                for pk, pv in d.items():
                    merged.setdefault(pk, set()).update([str(x) for x in (pv if isinstance(pv, list) else [pv])])
            return {pk: sorted(pv) for pk, pv in merged.items()}
        return {pk: _s(pv) for pk, pv in v.items()}

    content = {
        'features':     sorted(features_list),
        'param_spaces': {k: _normalise_space(v) for k, v in param_spaces.items()},
    }
    return hashlib.md5(_json.dumps(content, sort_keys=True).encode()).hexdigest()[:12]


# ── SHAP beeswarm colour helper ───────────────────────────────────────────────

def get_beeswarm_colours(X_scaled, feat_idx, feature_name, X_norm, binary_features):
    if feature_name in binary_features:
        vals    = X_scaled[:, feat_idx]
        colours = np.where(vals > 0, '#d62728', '#1f77b4')  # red=1 (present), blue=0 (absent)
        return colours, True
    return X_norm[:, feat_idx], False


def draw_beeswarm(ax, shap_vals, X_class_sc, class_label, n_patients,
                  axis_annotation, binary_features, features_final, flip_sign=False):
    """Shared beeswarm panel drawing function."""
    if flip_sign:
        shap_vals = -shap_vals

    imp_cl  = np.abs(shap_vals).mean(axis=0)
    idx_ord = np.argsort(imp_cl)

    X_norm = np.zeros_like(X_class_sc)
    for j in range(X_class_sc.shape[1]):
        col = X_class_sc[:, j]
        rng = col.max() - col.min()
        X_norm[:, j] = (col - col.min()) / rng if rng > 1e-10 else 0.5

    y_pos = np.arange(len(idx_ord))
    for i, feat_idx in enumerate(idx_ord):
        sv     = shap_vals[:, feat_idx]
        fname  = features_final[feat_idx]
        jitter = np.random.normal(0, 0.08, size=len(sv))
        colours, is_binary = get_beeswarm_colours(X_class_sc, feat_idx, fname, X_norm, binary_features)
        
        if is_binary:
            ax.scatter(sv, y_pos[i] + jitter, c=colours,
                       alpha=0.85, s=50, edgecolors='white', linewidth=0.5)
        else:
            ax.scatter(sv, y_pos[i] + jitter, c=colours, cmap='RdYlBu_r', vmin=0, vmax=1,
                       alpha=0.85, s=50, edgecolors='white', linewidth=0.5)

    ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([features_final[i] for i in idx_ord], fontsize=9)
    ax.set_xlabel('SHAP value (Impact on prediction)', fontsize=10)
    ax.set_title(f'{class_label}\n(N={n_patients})', fontsize=11, fontweight='bold')
    ax.annotate(axis_annotation, xy=(0.5, -0.12), xycoords='axes fraction',
                ha='center', fontsize=9, style='italic')
    ax.grid(axis='x', linestyle='--', alpha=0.3)


# ── Youden-threshold Balanced Accuracy ───────────────────────────────────────

def bal_acc_at_youden(y_true, probs):
    fpr, tpr, thr = roc_curve(y_true, probs)
    j_idx   = np.argmax(tpr - fpr)
    thr_opt = thr[j_idx]
    preds   = (probs >= thr_opt).astype(int)
    return balanced_accuracy_score(y_true, preds), thr_opt


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — DATA LOADING (Pooled 70/15/15 iterative stratified split)
# ════════════════════════════════════════════════════════════════════════════

# ── 2.1 Load and concatenate all 4 datasets into a single pool ────────────────
def load_full_sheet(filepath, sheet):
    return pd.read_excel(filepath, sheet_name=sheet)

SOURCES = [
    ('IAvsHT_VALLADOLID_2023.xlsx', 'Train', 'Valladolid (2023)'),
    ('IAvsHT_VALLADOLID_2023.xlsx', 'Val',   'Valladolid (2023)'),
    ('IAvsHT_VALLADOLID_2025.xlsx', 0,       'Valladolid (2025)'),
    ('IAvsHT_GRANADA.xlsx',         0,       'Granada'),
    ('IAvsHT_SALAMANCA.xlsx',       0,       'Salamanca'),
]

parts = []
for filepath, sheet, center_name in SOURCES:
    df_part = load_full_sheet(filepath, sheet)
    df_part['_CENTRO'] = center_name   # Tracking origin for Step 6 breakdown
    parts.append(df_part)

df_pool = pd.concat(parts, ignore_index=True)
print(f'Total pooled patients: {len(df_pool)}')   # 1164

# ── 2.2 Iterative stratified splitting ────────────────────────────────────────
STRATIFY_VARS = [
    'EDAD', 'ESTADO PREOPERATORIO CRÍTICO', 
    'CIRUGIA DE AORTA + VALVULAR', 'BIVALVULAR + CORONARIO'
]

def test_balance(df_a, df_b, vars_estratif):
    """Welch p-values (continuous) / Chi² (binary) between two groups."""
    pvals = {}
    for v in vars_estratif:
        if df_a[v].nunique() <= 2:
            table = pd.crosstab(
                pd.concat([df_a[v], df_b[v]], ignore_index=True),
                pd.Series(['A'] * len(df_a) + ['B'] * len(df_b))
            )
            _, p, _, _ = stats.chi2_contingency(table)
        else:
            _, p = stats.ttest_ind(df_a[v].dropna(), df_b[v].dropna(), equal_var=False)
        pvals[v] = p
    return pvals

N_TOTAL  = len(df_pool)
N_TRAIN  = int(round(N_TOTAL * 0.70))            # ~815
N_VAL    = int(round(N_TOTAL * 0.15))            # ~175
N_TEST   = N_TOTAL - N_TRAIN - N_VAL             # ~174
P_THRESHOLD = 0.05
MAX_SEED    = 100_000
valid_seed  = None

for seed in range(MAX_SEED):
    rng      = np.random.default_rng(seed)
    idx_perm = rng.permutation(N_TOTAL)
    idx_tr = idx_perm[:N_TRAIN]
    idx_va = idx_perm[N_TRAIN:N_TRAIN + N_VAL]
    idx_te = idx_perm[N_TRAIN + N_VAL:]
    
    df_tr = df_pool.iloc[idx_tr]
    df_va = df_pool.iloc[idx_va]
    df_te = df_pool.iloc[idx_te]
    
    pvals_tv = test_balance(df_tr, df_va, STRATIFY_VARS)
    pvals_tt = test_balance(df_tr, df_te, STRATIFY_VARS)

    if (all(p > P_THRESHOLD for p in pvals_tv.values()) and
        all(p > P_THRESHOLD for p in pvals_tt.values())):
        valid_seed = seed
        idx_tr_final, idx_va_final, idx_te_final = idx_tr, idx_va, idx_te
        pvals_tv_final, pvals_tt_final = pvals_tv, pvals_tt
        break

if valid_seed is None:
    raise RuntimeError('No valid seed found. Reduce P_THRESHOLD or expand MAX_SEED.')

print(f'Valid seed found: {valid_seed}  | Train={N_TRAIN}  Val={N_VAL}  Test={N_TEST}')

# ── 2.3 Construct DataFrames and Numpy arrays ─────────────────────────────────
def split_Xy(df):
    """Splits X, y_target, y_HT and center label from an in-memory DataFrame."""
    df     = df.copy()
    center = df.pop('_CENTRO') if '_CENTRO' in df.columns else None
    y      = df.pop(df.columns[-1])
    y_ht   = df.pop('OBJETIVO HT')
    return df, y, y_ht, center

X_train_df, y_train_s, y_ht_train, _ = split_Xy(df_pool.iloc[idx_tr_final].reset_index(drop=True))
X_val_df,   y_val_s,   y_ht_val,   _ = split_Xy(df_pool.iloc[idx_va_final].reset_index(drop=True))
feature_names = X_train_df.columns.tolist()

X_train = X_train_df.to_numpy()
X_val   = X_val_df.to_numpy()
y_train = y_train_s.to_numpy()
y_val   = y_val_s.to_numpy()

# ── 2.4 data_tests: single entry 'Test_Interno' (with center labels) ──────────
X_ti_df, y_ti_s, y_ht_ti, center_ti = split_Xy(df_pool.iloc[idx_te_final].reset_index(drop=True))
data_tests = {
    'Test_Interno': {
        'X':      X_ti_df[feature_names].to_numpy(),
        'y':      y_ti_s.to_numpy(),
        'y_raw':  y_ht_ti,
        'center': center_ti.to_numpy(),
    }
}

# ── 2.5 Verification Block ────────────────────────────────────────────────────
print('\n' + '='*60)
print('STRATIFIED PARTITION VERIFICATION')
print('='*60)
print(f'\n{"Subset":<18} {"N":>5}  {"% class 1":>10}  {"% class 0":>10}')
print('-'*48)
for name, y_sub in [('Train', y_train), ('Val', y_val), ('Test_Interno', data_tests['Test_Interno']['y'])]:
    p1 = y_sub.mean() * 100
    print(f'{name:<18} {len(y_sub):>5}  {p1:>9.1f}%  {100-p1:>9.1f}%')

print(f'\nStratification P-values (seed={valid_seed}):')
print(f'\n  {"Variable":<42} {"Train vs Val":>13}  {"Train vs Test":>14}')
print('  ' + '-'*72)
for v in STRATIFY_VARS:
    pv_tv = pvals_tv_final[v]
    pv_tt = pvals_tt_final[v]
    mark_tv = '' if pv_tv > P_THRESHOLD else ' ⚠'
    mark_tt = '' if pv_tt > P_THRESHOLD else ' ⚠'
    print(f'  {v:<42} {pv_tv:>12.4f}{mark_tv}  {pv_tt:>12.4f}{mark_tt}')
print()


# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — PREPROCESSING
# ════════════════════════════════════════════════════════════════════════════

# ── 3.1 Remove zero-variance columns ─────────────────────────────────────────
std_dev      = np.std(X_train, axis=0)
zero_var_idx = np.where(std_dev == 0)[0]
if len(zero_var_idx) > 0:
    print(f'Removing {len(zero_var_idx)} zero-variance columns.')
    X_train = np.delete(X_train, zero_var_idx, axis=1)
    X_val   = np.delete(X_val,   zero_var_idx, axis=1)
    for k in data_tests:
        data_tests[k]['X'] = np.delete(data_tests[k]['X'], zero_var_idx, axis=1)
    for idx in sorted(zero_var_idx, reverse=True):
        del feature_names[idx]

# ── 3.2 Remove collinear columns (Pearson |r| > 0.95, computed on train) ─────
CORR_THRESHOLD = 0.95
df_tmp      = pd.DataFrame(X_train, columns=feature_names)
corr_matrix = df_tmp.corr().abs()
upper_tri   = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop_names = [c for c in upper_tri.columns if any(upper_tri[c] > CORR_THRESHOLD)]
if to_drop_names:
    print(f'Removing {len(to_drop_names)} collinear columns: {to_drop_names}')
    drop_idx = [feature_names.index(c) for c in to_drop_names]
    X_train  = np.delete(X_train, drop_idx, axis=1)
    X_val    = np.delete(X_val,   drop_idx, axis=1)
    for k in data_tests:
        data_tests[k]['X'] = np.delete(data_tests[k]['X'], drop_idx, axis=1)
    for idx in sorted(drop_idx, reverse=True):
        del feature_names[idx]

# ── 3.3 Special handling of EDAD and EDAD>75 ─────────────────────────────────
COL_EDAD     = 'EDAD'
COL_EDAD_BIN = 'EDAD>75'
idx_edad_forzada = feature_names.index(COL_EDAD) if COL_EDAD in feature_names else None

if COL_EDAD_BIN in feature_names:
    idx_bin = feature_names.index(COL_EDAD_BIN)
    X_train = np.delete(X_train, idx_bin, axis=1)
    X_val   = np.delete(X_val,   idx_bin, axis=1)
    for k in data_tests:
        data_tests[k]['X'] = np.delete(data_tests[k]['X'], idx_bin, axis=1)
    feature_names.pop(idx_bin)
    idx_edad_forzada = feature_names.index(COL_EDAD) if COL_EDAD in feature_names else None
    print(f'Excluded \'{COL_EDAD_BIN}\' from model (descriptive analysis only).')

# ── 3.4 StandardScaler ───────────────────────────────────────────────────────
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
for k in data_tests:
    data_tests[k]['X_sc'] = scaler.transform(data_tests[k]['X'])

# ── 3.5 FCBF feature selection ────────────────────────────────────────────────
FCBF_THRESHOLD = 0.001
X_train_disc = X_train_sc.copy()
for col_idx in range(X_train_disc.shape[1]):
    if col_idx == idx_edad_forzada:
        continue
    col_vals = X_train_disc[:, col_idx]
    if len(np.unique(col_vals)) > 2:
        X_train_disc[:, col_idx] = pd.cut(
            col_vals, bins=10, labels=False, duplicates='drop'
        ).astype(float)

# ── 3.5a Tolerance sweep: how N features and AUC change with FCBF tolerance ──
TOL_GRID = np.round(np.arange(0.30, 3.0, 0.1), 2)
curva_nfeatures, curva_auc_lr, curva_auc_rf = [], [], []

_rf_proxy = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0, class_weight='balanced')

for tol in TOL_GRID:
    if idx_edad_forzada is not None:
        _cols   = [i for i in range(X_train_disc.shape[1]) if i != idx_edad_forzada]
        _sel    = fcbf(X_train_disc[:, _cols], y_train, threshold=FCBF_THRESHOLD, tolerancia=tol)
        idx_tol = sorted([_cols[i] for i in _sel] + [idx_edad_forzada])
    else:
        idx_tol = fcbf(X_train_disc, y_train, threshold=FCBF_THRESHOLD, tolerancia=tol)
    curva_nfeatures.append(len(idx_tol))
    X_tr_tol = X_train_sc[:, idx_tol]
    X_va_tol = X_val_sc[:,   idx_tol]
    lr_tmp = LogisticRegression(C=0.1, max_iter=1000, class_weight='balanced', random_state=0)
    lr_tmp.fit(X_tr_tol, y_train)
    curva_auc_lr.append(roc_auc_score(y_val, lr_tmp.predict_proba(X_va_tol)[:, 1]))
    _rf_proxy.fit(X_tr_tol, y_train)
    curva_auc_rf.append(roc_auc_score(y_val, _rf_proxy.predict_proba(X_va_tol)[:, 1]))

COLOR_N, COLOR_LR, COLOR_RF = '#2c7bb6', '#d7191c', '#1a9641'
fig_fcbf, ax1 = plt.subplots(figsize=(10, 5))
ax1.set_xlabel('FCBF tolerance', fontsize=12)
ax1.set_xticks(np.arange(0.3, 3.0, 0.1))
ax1.set_xlim(0.3, 3.0)
ax1.set_ylabel('N features selected', color=COLOR_N, fontsize=12)
ax1.plot(TOL_GRID, curva_nfeatures, 'o-', color=COLOR_N, linewidth=2, label='N features')
ax1.tick_params(axis='y', labelcolor=COLOR_N)
ax1.set_ylim(0, max(curva_nfeatures) + 2)
ax2 = ax1.twinx()
ax2.set_ylabel('Validation AUC-ROC', fontsize=12)
ax2.plot(TOL_GRID, curva_auc_lr, 's--', color=COLOR_LR, linewidth=2, label='Logistic Regression')
ax2.plot(TOL_GRID, curva_auc_rf, '^--', color=COLOR_RF, linewidth=2, label='Random Forest')
ax2.set_ylim(0.5, 1.0)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=10)
plt.title('FCBF tolerance sweep: N features vs validation AUC', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fcbf_tolerance_mixed.png'), dpi=200, bbox_inches='tight')
plt.show()

# ── 3.5b Run definitive FCBF with chosen tolerance ───────────────────────────
FCBF_TOLERANCE = 2.2   # Adjusted tolerance for the mixed approach

if idx_edad_forzada is not None:
    cols_for_fcbf     = [i for i in range(X_train_disc.shape[1]) if i != idx_edad_forzada]
    idx_fcbf_local    = fcbf(X_train_disc[:, cols_for_fcbf], y_train,
                             threshold=FCBF_THRESHOLD, tolerancia=FCBF_TOLERANCE)
    selected_indices  = sorted([cols_for_fcbf[i] for i in idx_fcbf_local] + [idx_edad_forzada])
else:
    selected_indices  = sorted(fcbf(X_train_disc, y_train,
                                    threshold=FCBF_THRESHOLD, tolerancia=FCBF_TOLERANCE))

features_final = [feature_names[i] for i in selected_indices]
print(f'\nFCBF (tolerance={FCBF_TOLERANCE}): {len(features_final)} features selected:')
print(features_final)

# ── 3.5c Detect binary features on original (unscaled) X_train ───────────────
binary_features = set()
for fname in features_final:
    col_idx = feature_names.index(fname)
    vals    = np.unique(X_train[:, col_idx])
    if len(vals) == 2 and set(vals).issubset({0, 1}):
        binary_features.add(fname)
print(f'\nBinary features (detected on unscaled X_train): {sorted(binary_features)}')
print(f'Continuous features: {[f for f in features_final if f not in binary_features]}')

# ── 3.6 Symmetrical Uncertainty bar chart ────────────────────────────────────
su_items = []
for i in range(X_train_disc.shape[1]):
    su = symmetrical_uncertainty(X_train_disc[:, i], y_train)
    if su >= FCBF_THRESHOLD:
        su_items.append((su, feature_names[i], i in selected_indices))
su_items.sort(key=lambda t: t[0], reverse=True)

su_vals   = [t[0] for t in su_items]
su_names  = [t[1] for t in su_items]
su_colors = ['#d62728' if t[2] else '#4C72B0' for t in su_items]

fig_su, ax_su = plt.subplots(figsize=(9, max(8, len(su_vals) * 0.32)))
ax_su.barh(range(len(su_vals)), su_vals, color=su_colors, edgecolor='white', linewidth=0.4)
ax_su.set_yticks(range(len(su_names)))
ax_su.set_yticklabels(su_names, fontsize=7.5)
ax_su.invert_yaxis()
ax_su.set_xlabel('Symmetrical Uncertainty (SU)', fontsize=11)
ax_su.set_title('Individual feature relevance (SU with target)', fontsize=12, fontweight='bold')
ax_su.grid(axis='x', linestyle='--', alpha=0.4)
ax_su.legend(handles=[
    mpatches.Patch(color='#d62728', label=f'Selected by FCBF ({sum(t[2] for t in su_items)})'),
    mpatches.Patch(color='#4C72B0', label=f'Discarded as redundant ({sum(not t[2] for t in su_items)})'),
], fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fcbf_su_ranking_mixed.png'), dpi=200, bbox_inches='tight')
plt.show()

# ── 3.7 Apply feature selection to all sets ───────────────────────────────────
X_train_sel = X_train_sc[:, selected_indices]
X_val_sel   = X_val_sc[:,   selected_indices]
for k in data_tests:
    data_tests[k]['X_sel'] = data_tests[k]['X_sc'][:, selected_indices]


# ════════════════════════════════════════════════════════════════════════════
# STEP 4 — HYPERPARAMETER OPTIMISATION
# ════════════════════════════════════════════════════════════════════════════

# ── 4.1 Classifier factory ───────────────────────────────────────────────────
def make_classifier(name):
    return {
        'LR':              LogisticRegression(random_state=0, max_iter=1000),
        'LDA':             LinearDiscriminantAnalysis(),
        'KNN':             KNeighborsClassifier(),
        'SVM':             SVC(probability=True, random_state=0),
        'MLP':             MLPClassifier(random_state=0),
        'Random Forest':   RandomForestClassifier(random_state=0),
        'Extra Trees':     ExtraTreesClassifier(random_state=0),
        'Random Subspace': BaggingClassifier(estimator=DecisionTreeClassifier(),
                                max_features=0.5, bootstrap=False, random_state=0),
        'AdaBoost':        AdaBoostClassifier(random_state=0),
        'LogitBoost':      GradientBoostingClassifier(loss='log_loss', random_state=0),
        'RUSBoost':        RUSBoostClassifier(random_state=0),
        'XGBoost':         XGBClassifier(random_state=0, eval_metric='logloss'),
    }[name]

# ── 4.2 Hyperparameter search spaces ─────────────────────────────────────────
PARAM_SPACES = {
    'LR': [
        {
            'solver': ['saga'], 'penalty': ['l1', 'l2', 'elasticnet'],
            'l1_ratio': [0.5], 'class_weight': ['balanced', None],
        },
        {
            'solver': ['liblinear'], 'penalty': ['l1', 'l2'],
            'class_weight': ['balanced', None],
        },
    ],
    'LDA': {
        'solver':    ['lsqr', 'eigen'],
        'shrinkage': [None, 'auto', 0.1, 0.5, 0.9],
        'tol':       [1e-4, 1e-3, 1e-2],
    },
    'KNN': {
        'n_neighbors': [3, 5, 7, 11, 15],
        'weights':     ['uniform', 'distance'],
        'metric':      ['euclidean', 'manhattan', 'minkowski'],
        'p':           [1, 2],
    },
    'SVM': {
        'C':            [0.1, 1, 10, 50, 100],
        'gamma':        ['scale', 'auto', 0.1, 0.01, 0.001],
        'kernel':       ['rbf', 'linear', 'poly'],
        'class_weight': ['balanced', None],
    },
    'MLP': {
        'max_iter':           [1000, 1500],
        'hidden_layer_sizes': [(50,), (30, 30), (50, 25)],
        'activation':         ['relu', 'tanh'],
        'alpha':              [0.0001, 0.01, 0.1],
        'learning_rate_init': [0.001, 0.01],
    },
    'Random Forest': {
        'n_estimators':      [200, 500, 800],
        'max_depth':         [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf':  [2, 4, 6],
        'max_features':      ['sqrt', 'log2'],
        'class_weight':      ['balanced', 'balanced_subsample'],
    },
    'Extra Trees': {
        'n_estimators':     [200, 500, 800],
        'max_depth':        [5, 10, None],
        'min_samples_leaf': [2, 4, 6],
        'bootstrap':        [True, False],
        'class_weight':     ['balanced', 'balanced_subsample'],
    },
    'Random Subspace': {
        'n_estimators':                [100, 200, 500],
        'max_features':                [0.3, 0.5, 0.7],
        'max_samples':                 [0.7, 0.8, 1.0],
        'estimator__max_depth':        [3, 5, 10, None],
        'estimator__min_samples_leaf': [1, 2, 4],
        'estimator__class_weight':     ['balanced', None],
    },
    'AdaBoost': {
        'n_estimators':  [50, 100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.5, 1.0],
        'estimator':     [
            DecisionTreeClassifier(max_depth=1, class_weight='balanced'),
            DecisionTreeClassifier(max_depth=2, class_weight='balanced'),
            DecisionTreeClassifier(max_depth=3, class_weight='balanced'),
        ],
    },
    'LogitBoost': {
        'n_estimators':     [100, 200, 300, 500],
        'learning_rate':    [0.01, 0.05, 0.1, 0.2],
        'max_depth':        [2, 3, 4, 5],
        'min_samples_leaf': [2, 4, 6],
        'subsample':        [0.7, 0.8, 1.0],
        'max_features':     ['sqrt', 'log2', None],
    },
    'RUSBoost': {
        'n_estimators':      [50, 100, 200, 300],
        'learning_rate':     [0.01, 0.1, 0.5, 1.0],
        'sampling_strategy': ['auto', 0.5, 0.75, 1.0],
        'estimator':         [
            DecisionTreeClassifier(max_depth=1, class_weight='balanced'),
            DecisionTreeClassifier(max_depth=2, class_weight='balanced'),
            DecisionTreeClassifier(max_depth=3, class_weight='balanced'),
        ],
    },
    'XGBoost': {
        'n_estimators':     [100, 200, 300, 500],
        'learning_rate':    [0.01, 0.05, 0.1],
        'max_depth':        [3, 4, 5],
        'min_child_weight': [1, 3, 5],
        'gamma':            [0, 0.1, 0.2],
        'subsample':        [0.7, 0.8, 0.9],
        'colsample_bytree': [0.6, 0.8],
        'scale_pos_weight': [1, 3, 5],
        'reg_alpha':        [0, 0.01, 0.1, 1],
        'reg_lambda':       [1, 1.5, 2],
    },
}

# ── 4.3 Load or compute hyperparameters (JSON cache) ─────────────────────────
PARAMS_FILE     = os.path.join(OUTPUT_DIR, 'best_params_mixed.json')
cached_params   = {}
fingerprint_now = compute_fingerprint(features_final, PARAM_SPACES)

if os.path.exists(PARAMS_FILE):
    with open(PARAMS_FILE, 'r') as f:
        cache = json.load(f)
    fp_stored = cache.get('__fingerprint__', '')
    if fp_stored == fingerprint_now:
        cached_params = {k: params_from_json(v)
                         for k, v in cache.items() if not k.startswith('__')}
        print(f'\nHyperparameters loaded from cache (fingerprint OK: {fingerprint_now})')
        print(f'  → Skipping RandomizedSearchCV for: {list(cached_params.keys())}')
    else:
        print(f'\n⚠ Fingerprint changed ({fp_stored} → {fingerprint_now}) ')
        print('  → Cache invalidated, rerunning search.')
else:
    print(f'\n  \'{PARAMS_FILE}\' not found → RandomizedSearchCV will run.')
    print(f'  Fingerprint: {fingerprint_now}')

# ── 4.4 Optimise all 12 classifiers ──────────────────────────────────────────
CLASSIFIER_NAMES = [
    'LR', 'LDA', 'KNN', 'SVM', 'MLP',
    'Random Forest', 'Extra Trees', 'Random Subspace',
    'AdaBoost', 'LogitBoost', 'RUSBoost', 'XGBoost',
]

optimized_models = {}
new_params       = {}

print('\n--- STEP 4.4: HYPERPARAMETER OPTIMISATION (all 12 classifiers) ---')
for name in CLASSIFIER_NAMES:
    print(f'\n  [{name}]')
    clf = make_classifier(name)

    if name in cached_params:
        params = cached_params[name]
        print(f'    → Loading cached params: {params}')
        clf.set_params(**params)
        clf.fit(X_train_sel, y_train)
        best_clf         = clf
        new_params[name] = params

    elif name in PARAM_SPACES:
        search = RandomizedSearchCV(
            clf,
            PARAM_SPACES[name],
            n_iter=200,
            scoring='roc_auc',
            cv=5,
            n_jobs=-1,
            random_state=0,
        )
        search.fit(X_train_sel, y_train)
        best_clf         = search.best_estimator_
        new_params[name] = search.best_params_
        print(f'    → Best params found: {search.best_params_}')

    else:
        clf.fit(X_train_sel, y_train)
        best_clf = clf
        print('    → No search space defined; trained with defaults.')

    optimized_models[name] = best_clf

merged_params = {**cached_params, **new_params}
cache_out     = {'__fingerprint__': fingerprint_now}
cache_out.update({k: params_to_json(v) for k, v in merged_params.items()})
with open(PARAMS_FILE, 'w') as f:
    json.dump(cache_out, f, indent=2)
print(f'\nHyperparameters saved to \'{PARAMS_FILE}\' (fingerprint: {fingerprint_now}).')

# ── 4.5 Rank all 12 optimised classifiers on the validation set ──────────────
print('\n--- STEP 4.5: POST-OPTIMISATION RANKING (validation set) ---')

DELTA_TIEBREAK = 0.01
results = []
for name, clf in optimized_models.items():
    preds_val  = clf.predict(X_val_sel)
    probs_val  = clf.predict_proba(X_val_sel)[:, 1] if hasattr(clf, 'predict_proba') else None
    results.append(compute_metrics(y_val, preds_val, probs_val, name))

df_results = pd.DataFrame(results).sort_values('AUC-ROC', ascending=False).reset_index(drop=True)

ordered, pending = [], df_results.to_dict('records')
while pending:
    current = pending.pop(0)
    if pending and abs(current['AUC-ROC'] - pending[0]['AUC-ROC']) < DELTA_TIEBREAK:
        if pending[0]['Bal. Accuracy'] > current['Bal. Accuracy']:
            ordered.append(pending.pop(0))
            ordered.append(current)
        else:
            ordered.append(current)
    else:
        ordered.append(current)
df_results = pd.DataFrame(ordered).reset_index(drop=True)
print(df_results.to_string())


# ════════════════════════════════════════════════════════════════════════════
# STEP 5 — TOP-3 SELECTION, CALIBRATION AND VOTING ENSEMBLE
# ════════════════════════════════════════════════════════════════════════════

# ── 5.1 Select top-3 classifiers and calibrate individually ──────────────────
TOP_N = 3
top_models = df_results.head(TOP_N)
_top_display = top_models[['Model', 'AUC-ROC']]
print(f'\nTop {TOP_N} selected for Voting Ensemble:\n{_top_display}')

tuned_estimators  = []
calibrated_models = {}

for name in top_models['Model']:
    best_clf = optimized_models[name]
    tuned_estimators.append((name, best_clf))
    cal_model = CalibratedClassifierCV(estimator=best_clf, method='sigmoid', cv='prefit')
    cal_model.fit(X_val_sel, y_val)
    calibrated_models[name] = cal_model
    print(f'  → {name} calibrated on validation set.')

# ── 5.2 Build and calibrate the Voting Ensemble ───────────────────────────────
print('\nGenerating Voting Ensemble...')
voting_clf = VotingClassifier(estimators=tuned_estimators, voting='soft')
voting_clf.fit(X_train_sel, y_train)

calibrated_voting = CalibratedClassifierCV(estimator=voting_clf, method='sigmoid', cv='prefit')
calibrated_voting.fit(X_val_sel, y_val)

probs_val_voting     = calibrated_voting.predict_proba(X_val_sel)[:, 1]
auc_voting           = roc_auc_score(y_val, probs_val_voting)
best_individual_name = df_results.iloc[0]['Model']
probs_val_individual = calibrated_models[best_individual_name].predict_proba(X_val_sel)[:, 1]
best_individual_auc  = roc_auc_score(y_val, probs_val_individual)
print(f'Voting Ensemble calibrated AUC on Val: {auc_voting:.4f}')
print(f'Best individual calibrated ({best_individual_name}) AUC on Val: {best_individual_auc:.4f}')

# ── Model selection: AUC primary, Bal.Acc@Youden tiebreaker ──────────────────
bal_acc_voting,     thr_youden_voting     = bal_acc_at_youden(y_val, probs_val_voting)
bal_acc_individual, thr_youden_individual = bal_acc_at_youden(y_val, probs_val_individual)

if abs(auc_voting - best_individual_auc) < DELTA_TIEBREAK:
    print(f'  Tiebreaker (|ΔAUC|<{DELTA_TIEBREAK}) → Bal.Acc @ Youden threshold:')
    print(f'    Voting:     Bal.Acc = {bal_acc_voting:.4f}  (Youden = {thr_youden_voting:.4f})')
    print(f'    Individual: Bal.Acc = {bal_acc_individual:.4f}  (Youden = {thr_youden_individual:.4f})')
    if bal_acc_voting >= bal_acc_individual:
        final_model, final_name = calibrated_voting, 'Voting Ensemble'
    else:
        final_model, final_name = calibrated_models[best_individual_name], best_individual_name
elif auc_voting > best_individual_auc:
    final_model, final_name = calibrated_voting, 'Voting Ensemble'
else:
    final_model, final_name = calibrated_models[best_individual_name], best_individual_name
print(f'\nFinal model selected: {final_name}')

# ── 5.3 Youden threshold optimisation on the validation set ──────────────────
probs_val  = final_model.predict_proba(X_val_sel)[:, 1]
print(f'\nProbability statistics (validation set):')
print(f'  Min: {probs_val.min():.4f} | Mean: {probs_val.mean():.4f} | Max: {probs_val.max():.4f}')

fpr_val, tpr_val, thresholds_val = roc_curve(y_val, probs_val)
youden_idx = np.argmax(tpr_val - fpr_val)
final_thr  = thresholds_val[youden_idx]
best_se    = tpr_val[youden_idx]
best_sp    = 1 - fpr_val[youden_idx]

print(f'\n--- THRESHOLD OPTIMISATION (Youden index, validation set) ---')
print(f'Optimal clinical threshold (J): {final_thr:.4f}')
print(f'Expected sensitivity:           {best_se:.2%}')
print(f'Expected specificity:           {best_sp:.2%}')


# ════════════════════════════════════════════════════════════════════════════
# STEP 6 — EXTERNAL TEST EVALUATION (Overall and per Center)
# ════════════════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print('STEP 6 — EXTERNAL TEST RESULTS')
print('='*70)

n_tests = len(data_tests)
fig, axes = plt.subplots(nrows=n_tests, ncols=2, figsize=(10, 5 * n_tests))
if n_tests == 1:
    axes = np.array(axes).reshape(1, 2)
test_results = []

for i, (test_name, test_data) in enumerate(data_tests.items()):
    X_test = test_data['X_sel']
    y_test = test_data['y']
    y_ht   = test_data['y_raw']

    probs_test = final_model.predict_proba(X_test)[:, 1]
    preds_ia   = (probs_test >= final_thr).astype(int)
    m_ia = compute_metrics(y_test, preds_ia, probs_test, f'IA-{test_name}')
    test_results.append(m_ia)

    preds_ht = y_ht.values
    m_ht = compute_metrics(y_test, preds_ht, None, f'HT-{test_name}')
    test_results.append(m_ht)

    ConfusionMatrixDisplay.from_predictions(
        y_test, preds_ia, ax=axes[i, 0],
        cmap='Blues', colorbar=False, display_labels=['No surgery', 'Surgery']
    )
    axes[i, 0].set_title(f'AI on {test_name}', fontsize=10, fontweight='bold')

    ConfusionMatrixDisplay.from_predictions(
        y_test, preds_ht, ax=axes[i, 1],
        cmap='Oranges', colorbar=False, display_labels=['No surgery', 'Surgery']
    )
    axes[i, 1].set_title(f'Heart Team on {test_name}', fontsize=10, fontweight='bold')

plt.suptitle(
    f'Mixed training and validation strategy\n'
    f'Multicentre comparison: AI ({final_name}) vs Heart Team',
    fontsize=14, fontweight='bold', y=1.01
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'matrices_comparison_multicentre_mixed.png'),
            dpi=300, bbox_inches='tight')
plt.show()

# ── Confusion matrices broken down by center of origin (within Test_Interno) ──
test_centers  = data_tests['Test_Interno']['center']
ORDERED_CENTERS = ['Valladolid (2023)', 'Valladolid (2025)', 'Granada', 'Salamanca']
unique_centers  = [c for c in ORDERED_CENTERS if c in set(test_centers)]
n_centers       = len(unique_centers)
COLOURS_CENTER  = {'Valladolid (2023)': '#9467bd', 'Valladolid (2025)': '#1f77b4', 
                   'Granada': '#2ca02c', 'Salamanca': '#d62728'}

print('\n--- BREAKDOWN BY ORIGIN CENTER (Test_Interno) ---')
fig_c, axes_c = plt.subplots(nrows=n_centers, ncols=2, figsize=(12, 5 * n_centers))
if n_centers == 1:
    axes_c = np.array(axes_c).reshape(1, 2)

results_by_center = []
for j, center in enumerate(unique_centers):
    mask       = (test_centers == center)
    X_c        = data_tests['Test_Interno']['X_sel'][mask]
    y_c        = data_tests['Test_Interno']['y'][mask]
    y_raw_c    = data_tests['Test_Interno']['y_raw'].values[mask]
    
    probs_c    = final_model.predict_proba(X_c)[:, 1]
    preds_ia_c = (probs_c >= final_thr).astype(int)
    
    m_ia_c = compute_metrics(y_c, preds_ia_c, probs_c, f'IA-{center}')
    m_ht_c = compute_metrics(y_c, y_raw_c, None, f'HT-{center}')
    results_by_center.extend([m_ia_c, m_ht_c])
    
    print(f'  {center}: N={mask.sum()}  |  AI AUC={m_ia_c["AUC-ROC"]:.3f}  |  HT AUC={m_ht_c["AUC-ROC"]:.4f}')
    
    ConfusionMatrixDisplay.from_predictions(
        y_c, preds_ia_c, ax=axes_c[j, 0],
        cmap='Blues', colorbar=False, display_labels=['No surgery', 'Surgery']
    )
    axes_c[j, 0].set_title(f'AI on {center} (N={mask.sum()})', fontsize=10, fontweight='bold')
    
    ConfusionMatrixDisplay.from_predictions(
        y_c, y_raw_c, ax=axes_c[j, 1],
        cmap='Oranges', colorbar=False, display_labels=['No surgery', 'Surgery']
    )
    axes_c[j, 1].set_title(f'Heart Team on {center} (N={mask.sum()})', fontsize=10, fontweight='bold')

plt.suptitle(
    f'Mixed training and validation strategy\n'
    f'Multicentre comparison: AI ({final_name}) vs Heart Team',
    fontsize=14, fontweight='bold', y=1.01
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'matrices_by_center_origin_mixed.png'),
            dpi=300, bbox_inches='tight')
plt.show()

df_by_center = pd.DataFrame(results_by_center)
print('\n--- METRICS BY ORIGIN CENTER ---')
print(df_by_center.to_string(index=False))

# ── ROC curves by center of origin (within Test_Interno) ────────────────────
fig_roc_c, ax_roc_c = plt.subplots(figsize=(7, 6))
for center in unique_centers:
    mask      = (test_centers == center)
    X_c       = data_tests['Test_Interno']['X_sel'][mask]
    y_c       = data_tests['Test_Interno']['y'][mask]
    probs_c   = final_model.predict_proba(X_c)[:, 1]
    fpr_c, tpr_c, _ = roc_curve(y_c, probs_c)
    auc_c     = roc_auc_score(y_c, probs_c)
    color_c   = COLOURS_CENTER.get(center, '#7f7f7f')
    ax_roc_c.plot(fpr_c, tpr_c, color=color_c, linewidth=2, label=f'{center} (N={mask.sum()}, AUC={auc_c:.3f})')

ax_roc_c.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random classifier')
ax_roc_c.set_xlabel('1 − Specificity', fontsize=11)
ax_roc_c.set_ylabel('Sensitivity', fontsize=11)
ax_roc_c.set_title(
    f'Mixed strategy — Multicentre ROC curves ({final_name})', fontsize=12, fontweight='bold'
)
ax_roc_c.legend(loc='lower right', fontsize=10)
ax_roc_c.grid(linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curves_mixed.png'), dpi=200, bbox_inches='tight')
plt.show()

# ── Summary table and Excel export ───────────────────────────────────────────
df_test_summary = pd.DataFrame(test_results)
print('\n--- EXTERNAL TEST SUMMARY ---')
print(df_test_summary.to_string(index=False))

print('\nWriting results to Excel...')
try:
    EXCEL_PATH = os.path.join(OUTPUT_DIR, 'IAvsHT_mixed.xlsx')
    with pd.ExcelWriter(EXCEL_PATH, engine='openpyxl', mode='w') as writer:

        RANK_COLS = ['Model', 'AUC-ROC', 'Sensitivity', 'Specificity', 'Accuracy',
                     'PPV', 'NPV', 'Bal. Accuracy', 'F1-Score', 'Kappa', 'MCC']
        RANK_COLS = [c for c in RANK_COLS if c in df_results.columns]
        df_results[RANK_COLS].to_excel(writer, sheet_name='Classifier_Ranking', index=False)

        METRICS_ORDER = [
            ('Sensitivity', 'Sensitivity'), ('Specificity', 'Specificity'),
            ('Accuracy', 'Accuracy'),         ('PPV', 'PPV'),
            ('NPV', 'NPV'),                   ('F1-Score', 'F1-Score'),
            ('Bal. Accuracy', 'Bal. Accuracy'),('AUC-ROC', 'AUC-ROC'),
            ('Kappa', "Cohen's Kappa"),        ('MCC', 'MCC'),
        ]

        # Overall Internal Test Sheet
        df_ia_total = df_test_summary[df_test_summary['Model'].str.startswith('IA-')].copy()
        df_ia_total['Center'] = df_ia_total['Model'].str.replace('IA-', '', regex=False)
        centers_total = list(df_ia_total['Center'])
        
        pivot_rows_total = []
        for key, label in METRICS_ORDER:
            row = {'Metric': label}
            for center in centers_total:
                r = df_ia_total[df_ia_total['Center'] == center]
                row[center] = r[key].values[0] if len(r) > 0 and key in r.columns else ''
            pivot_rows_total.append(row)
        pd.DataFrame(pivot_rows_total).to_excel(writer, sheet_name='Overall_Test_Performance', index=False)

        # Breakdown by Center Sheet
        if 'results_by_center' in dir() or 'results_by_center' in locals():
            df_ia_c  = df_by_center[df_by_center['Model'].str.startswith('IA-')].copy()
            df_ia_c['Center'] = df_ia_c['Model'].str.replace('IA-', '', regex=False)
            centers_c = list(df_ia_c['Center'])
            
            pivot_rows_c = []
            for key, label in METRICS_ORDER:
                row = {'Metric': label}
                values = []
                for center in centers_c:
                    r   = df_ia_c[df_ia_c['Center'] == center]
                    val = r[key].values[0] if len(r) > 0 and key in r.columns else None
                    row[center] = val if val is not None else ''
                    if val is not None:
                        values.append(val)
                row['Mean ± SD'] = (f'{np.mean(values):.4f} ± {np.std(values):.4f}' if values else '')
                pivot_rows_c.append(row)
            pd.DataFrame(pivot_rows_c).to_excel(writer, sheet_name='Performance_By_Center', index=False)
            
    print(f'  Saved: {EXCEL_PATH}')
except Exception as e:
    print(f'  ERROR writing Excel: {e}')
    import traceback; traceback.print_exc()


# ════════════════════════════════════════════════════════════════════════════
# STEP 7 — SHAP EXPLAINABILITY
# ════════════════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print('STEP 7 — SHAP EXPLAINABILITY')
print('='*70)

def model_prob_surgery(X):
    """Wrapper for KernelExplainer: returns P(surgery) as a 1D array."""
    return final_model.predict_proba(X)[:, 1]

print('Building K-Means background for KernelExplainer...')
n_unique   = len(np.unique(X_train_sel, axis=0))
n_clusters = min(20, n_unique)
print(f'  Unique rows in Train: {n_unique} → using {n_clusters} K-Means clusters')
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', message='Clustering metrics expects discrete values')
    X_background = shap.kmeans(X_train_sel, n_clusters)

explainer = shap.KernelExplainer(model_prob_surgery, X_background)

# ── 7.1 Compute SHAP values on the training set ───────────────────────────────
print(f'Computing SHAP values on Train ({X_train_sel.shape[0]} patients)...')
shap_train    = explainer.shap_values(X_train_sel, nsamples=100)
imp_train     = np.abs(shap_train).mean(axis=0)
idx_train_asc = np.argsort(imp_train)

# ── Fig 1 — Global SHAP importance ranking (Train) ───────────────────────────
plt.figure(figsize=(9, 5))
plt.barh([features_final[i] for i in idx_train_asc], imp_train[idx_train_asc],
         color='#2c7bb6', edgecolor='white')
plt.xlabel('Mean absolute Shapley Values', fontsize=11)
plt.title(
    f'Mixed training and validation strategy\n'
    f'SHAP importance ranking — Training set\nModel: {final_name}',
    fontsize=15, fontweight='bold'
)
plt.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'shap_ranking_train_mixed.png'), dpi=200, bbox_inches='tight')
plt.show()

print('\nTop SHAP features (Train):')
for i in idx_train_asc[::-1]:
    print(f'  {features_final[i]:<45}: {imp_train[i]:.4f}')

# ── Fig 2 — Beeswarm on Train (1×2: class 0 | class 1) ───────────────────────
mask_c0_tr = (y_train == 0)
mask_c1_tr = (y_train == 1)
print(f'  Train — Class 0 (No surgery): {mask_c0_tr.sum()} patients')
print(f'  Train — Class 1 (Surgery):    {mask_c1_tr.sum()} patients')

fig2, axes2 = plt.subplots(1, 2, figsize=(16, 8))
draw_beeswarm(axes2[0], shap_train[mask_c0_tr], X_train_sel[mask_c0_tr],
              'Class 0 (No surgery)', mask_c0_tr.sum(),
              '← Favours "SURGERY" | Favours "NO SURGERY" →', 
              binary_features, features_final, flip_sign=True)
draw_beeswarm(axes2[1], shap_train[mask_c1_tr], X_train_sel[mask_c1_tr],
              'Class 1 (Surgery)', mask_c1_tr.sum(),
              '← Favours "NO SURGERY" | Favours "SURGERY" →', 
              binary_features, features_final, flip_sign=False)
plt.suptitle(
    f'Mixed training and validation strategy\n'
    f'SHAP beeswarm — Training set\nModel: {final_name}',
    fontsize=20, fontweight='bold', y=1.02
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'shap_beeswarm_train_mixed.png'), dpi=200, bbox_inches='tight')
plt.show()


# ── 7.2 Compute SHAP values per test center ──────────────────────────────────
print('\nComputing SHAP per origin center...')
shap_per_center = {}
for center in unique_centers:
    mask = (test_centers == center)
    X_c  = data_tests['Test_Interno']['X_sel'][mask]
    print(f'  {center} ({X_c.shape[0]} patients)...')
    shap_per_center[center] = explainer.shap_values(X_c, nsamples=100)

# ── Fig 3 — SHAP importance ranking per test center (2x2) ────────────────────
fig3, axes3 = plt.subplots(2, 2, figsize=(20, 14))
plt.rcParams['ytick.labelsize'] = 13

x_max = max(np.abs(sv).mean(axis=0).max() for sv in shap_per_center.values()) * 1.1

for ax, cname in zip(axes3.flatten(), unique_centers):
    shap_c  = shap_per_center[cname]
    imp_c   = np.abs(shap_c).mean(axis=0)
    idx_c   = np.argsort(imp_c)
    color_c = COLOURS_CENTER.get(cname, '#7f7f7f')
    n_pac   = (test_centers == cname).sum()
    
    ax.barh([features_final[i] for i in idx_c], imp_c[idx_c],
            color=color_c, edgecolor='white', height=0.5)
    ax.set_xlabel('Mean absolute Shapley Values', fontsize=13)
    ax.set_title(f'{cname} (N={n_pac})', fontsize=13, fontweight='bold')
    ax.set_xlim(0, x_max)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.tick_params(axis='x', labelsize=11)

plt.suptitle(
    f'Mixed training and validation strategy\n'
    f'SHAP importance ranking per test set\nModel: {final_name}',
    fontsize=16, fontweight='bold'
)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(os.path.join(OUTPUT_DIR, 'shap_ranking_test_mixed.png'), dpi=200, bbox_inches='tight')
plt.show()


# ── Fig 4 — Beeswarm per test center (N×2) ───────────────────────────────────
print(f'\nGenerating Beeswarm SHAP {n_centers}x2: origin centers (Class 0 vs Class 1)')
fig4, axes4 = plt.subplots(n_centers, 2, figsize=(16, 6 * n_centers))
if n_centers == 1:
    axes4 = axes4.reshape(1, 2)

for row, cname in enumerate(unique_centers):
    mask_center = (test_centers == cname)
    X_c         = data_tests['Test_Interno']['X_sel'][mask_center]
    y_c         = data_tests['Test_Interno']['y'][mask_center]
    shap_c      = shap_per_center[cname]
    
    mask_c0, mask_c1 = (y_c == 0), (y_c == 1)
    print(f'    {cname} — Class 0: {mask_c0.sum()} | Class 1: {mask_c1.sum()}')
    
    draw_beeswarm(axes4[row, 0], shap_c[mask_c0], X_c[mask_c0],
                  f'{cname}\nClass 0 (No surgery)', mask_c0.sum(),
                  '← Favours "SURGERY" | Favours "NO SURGERY" →', 
                  binary_features, features_final, flip_sign=True)
                  
    draw_beeswarm(axes4[row, 1], shap_c[mask_c1], X_c[mask_c1],
                  f'{cname}\nClass 1 (Surgery)', mask_c1.sum(),
                  '← Favours "NO SURGERY" | Favours "SURGERY" →', 
                  binary_features, features_final, flip_sign=False)

plt.suptitle(
    f'Mixed training and validation strategy\n'
    f'SHAP beeswarm per test set\nModel: {final_name}',
    fontsize=20, fontweight='bold', y=0.98
)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(OUTPUT_DIR, 'shap_beeswarm_test_mixed.png'), dpi=200, bbox_inches='tight')
plt.show()