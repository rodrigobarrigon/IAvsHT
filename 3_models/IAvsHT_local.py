##############################################################################
# IAvsHT — Local Training and Validation Strategy
# ─────────────────────────────────────────────────────────────────────────────
# Partitioning strategy: fixed train / validation split on a single centre
# (HCUV Valladolid 2023), with three fully independent external test cohorts.
#
# Data flow:
#   Train (N=551) ─┐
#                  ┤─ Valladolid 2023  (IAvsHT_VALLADOLID_2023.xlsx)
#   Val   (N=151) ─┘

#   Test  (N=150)   Valladolid 2025    (IAvsHT_VALLADOLID_2025.xlsx)
#   Test  (N=150)   Granada            (IAvsHT_GRANADA.xlsx)
#   Test  (N=162)   Salamanca          (IAvsHT_SALAMANCA.xlsx)
#
# Pipeline (Steps 1–7):
#   1. Imports, constants and shared function definitions
#   2. Data loading
#   3. Preprocessing: zero-variance removal → collinearity pruning →
#      EDAD/EDAD>75 handling → StandardScaler → FCBF feature selection
#   4. Hyperparameter optimisation (RandomizedSearchCV, 200 iter, 5-fold CV)
#      with JSON cache invalidated automatically by pipeline fingerprint
#   5. Classifier ranking → top-3 → Platt calibration → Voting Ensemble →
#      model selection (AUC primary, Bal.Acc@Youden tiebreaker) →
#      Youden threshold optimisation
#   6. External test evaluation: confusion matrices + ROC curves + Excel export
#   7. SHAP explainability: KernelExplainer, K-Means background,
#      feature importance rankings + beeswarm plots
#      (training set / test sets / correctly classified only)
#
# Output directory : IAvsHT_local/
# Figures (PNG)    :
#   fcbf_tolerance_local.png          fcbf_su_ranking_local.png
#   matrices_comparison_multicentre_local.png
#   roc_curves_local.png
#   shap_ranking_train_local.png      shap_beeswarm_train_local.png
#   shap_ranking_test_local.png       shap_beeswarm_test_local.png
#   shap_ranking_correct_local.png    shap_beeswarm_correct_local.png
# Excel            : IAvsHT_local.xlsx
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
OUTPUT_DIR = 'IAvsHT_local'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Information-theoretic helpers (used by FCBF) ─────────────────────────────

def entropy(vec):
    """
    Shannon entropy of a discrete vector (bits).
    H(X) = -Σ p(x) · log2(p(x))
    An epsilon of 1e-12 is added inside the log to prevent log(0).
    """
    _, counts = np.unique(vec, return_counts=True)   # unique values and their absolute frequencies
    probs = counts / len(vec)                         # convert counts to probabilities
    return -np.sum(probs * np.log2(probs + 1e-12))


def symmetrical_uncertainty(x, y):
    """
    Symmetrical Uncertainty (SU) between two discrete variables.
    SU(X,Y) = 2 · MI(X;Y) / (H(X) + H(Y))  ∈ [0, 1]
    SU = 1 means perfect functional dependence; SU = 0 means independence.
    MI is computed in nats (sklearn default) then converted to bits so
    the units match the Shannon entropies above.
    """
    h_x = entropy(x)
    h_y = entropy(y)
    if (h_x + h_y) == 0:
        return 0.0                                    # both variables are constant → SU undefined, return 0
    mi_bits = mutual_info_score(x, y) / np.log(2)    # mutual information in bits (base 2)
    return mi_bits / (h_x + h_y)                     # normalised to [0, 1]


def fcbf(X, y, threshold=0.0, tolerancia=1.0):
    """
    Fast Correlation-Based Filter — relaxed variant.

    Standard FCBF discards candidate i if SU(i, pivot) >= SU(i, target).
    This relaxed version only discards i if:
        SU(i, pivot) >= SU(i, target) * tolerancia

    Effect of the tolerance parameter:
        tolerancia = 1.0  →  original strict FCBF
        tolerancia > 1.0  →  more permissive (more features survive); useful
                              when multicentric generalisation is a priority

    Returns a sorted list of selected column indices.
    """
    n_samples, n_features = X.shape

    # ── Phase 1: rank all features by SU with the target ──────────────────────
    su_target = []
    for i in range(n_features):
        su = symmetrical_uncertainty(X[:, i], y)
        if su >= threshold:
            su_target.append((su, i))                 # keep only features above the minimum SU threshold
    su_target.sort(reverse=True, key=lambda t: t[0]) # highest relevance first

    # ── Phase 2: greedy redundancy elimination ────────────────────────────────
    candidates       = su_target[:]
    selected_indices = []

    while candidates:
        best_su, best_idx = candidates[0]             # the most relevant remaining feature becomes the pivot
        selected_indices.append(best_idx)
        candidates.pop(0)

        remaining  = []
        pivot_vals = X[:, best_idx]
        for su_cand, idx_cand in candidates:
            su_between = symmetrical_uncertainty(pivot_vals, X[:, idx_cand])
            # discard only if redundancy with the pivot exceeds the tolerance-adjusted threshold
            if su_between < su_cand * tolerancia:
                remaining.append((su_cand, idx_cand))
        candidates = remaining

    return sorted(selected_indices)                   # sorted for consistent downstream indexing


# ── Dataset loader ────────────────────────────────────────────────────────────

def load_sheet(filepath, sheet):
    """
    Read one Excel sheet and split it into (X, y, y_ht).
      X    : feature DataFrame (all columns except target and HT decision)
      y    : ground truth label (last column, named 'ETIQUETA')
      y_ht : Heart Team clinical decision (column 'OBJETIVO HT')
    """
    df   = pd.read_excel(filepath, sheet_name=sheet)
    y    = df.pop(df.columns[-1])   # ground truth is always the last column
    y_ht = df.pop('OBJETIVO HT')
    return df, y, y_ht


# ── Clinical performance metrics ──────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_prob=None, name=''):
    """
    Compute 11 clinical performance metrics from ground truth and predictions.
    y_prob (continuous probability scores) is required only for AUC-ROC;
    if not provided, AUC-ROC is set to NaN.
    All metrics are rounded to 4 decimal places.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sens    = recall_score(y_true, y_pred, zero_division=0)          # Se = TP / (TP+FN)
    spec    = tn / (tn + fp) if (tn + fp) > 0 else 0                 # Sp = TN / (TN+FP)
    acc     = accuracy_score(y_true, y_pred)
    ppv     = precision_score(y_true, y_pred, zero_division=0)        # PPV = TP / (TP+FP)
    npv     = tn / (tn + fn) if (tn + fn) > 0 else 0                 # NPV = TN / (TN+FN)
    bal_acc = balanced_accuracy_score(y_true, y_pred)                 # arithmetic mean of Se and Sp
    f1      = f1_score(y_true, y_pred, zero_division=0)
    auc_val = roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan
    kappa   = cohen_kappa_score(y_true, y_pred)                       # chance-corrected agreement
    mcc     = matthews_corrcoef(y_true, y_pred)                       # balanced metric using all 4 CM cells

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
# RandomizedSearchCV may return hyperparameters that include sklearn sub-estimator
# objects (e.g. DecisionTreeClassifier as the base estimator for AdaBoost).
# Standard JSON cannot serialise these objects, so we convert them to/from strings.

def _estimator_to_str(estimator):
    """
    Serialise an sklearn estimator to a human-readable string for JSON storage.
    Format: '__estimator__ClassName(param=value, ...)'
    Only non-None parameters are included to keep the string compact.
    """
    cls    = type(estimator).__name__
    params = estimator.get_params()
    parts  = ', '.join(f'{k}={repr(v)}' for k, v in sorted(params.items()) if v is not None)
    return f'__estimator__{cls}({parts})'


def _str_to_estimator(s):
    """
    Reconstruct an sklearn estimator from the string produced by _estimator_to_str.
    Extend cls_map if additional estimator types are needed.
    """
    m = re.match(r'__estimator__(\w+)\((.*)\)', s, re.DOTALL)
    if not m:
        return s                                                    # not a serialised estimator — return as-is
    cls_name, params_str = m.group(1), m.group(2)
    cls_map = {'DecisionTreeClassifier': DecisionTreeClassifier}
    cls     = cls_map.get(cls_name)
    if cls is None:
        raise ValueError(f'Unknown estimator during deserialisation: {cls_name}')
    kwargs = {}
    if params_str.strip():
        for match in re.finditer(
            r'(\w+)=(\'(?:[^\'\\\\]|\\\\.)*\'|"(?:[^"\\\\]|\\\\.)*"|[^,]+)', params_str
        ):
            k, raw_v = match.group(1).strip(), match.group(2).strip()
            try:
                kwargs[k] = ast.literal_eval(raw_v)
            except Exception:
                kwargs[k] = raw_v                                   # leave as string if not parseable
    return cls(**kwargs)


def params_to_json(params):
    """
    Convert a hyperparameter dict to a JSON-safe format:
      None          → '__None__'    (JSON has no None)
      tuple         → list           (JSON has no tuples)
      sklearn est.  → string         (via _estimator_to_str)
    """
    safe = {}
    for k, v in params.items():
        if v is None:
            safe[k] = '__None__'
        elif isinstance(v, tuple):
            safe[k] = list(v)
        elif hasattr(v, 'get_params'):                             # sklearn estimator object
            safe[k] = _estimator_to_str(v)
        else:
            safe[k] = v
    return safe


def params_from_json(params):
    """
    Restore a hyperparameter dict from its JSON-safe representation:
      '__None__'          → None
      hidden_layer_sizes  → tuple  (MLPClassifier requires a tuple)
      '__estimator__...'  → reconstructed sklearn estimator
    """
    restored = {}
    for k, v in params.items():
        if v == '__None__':
            restored[k] = None
        elif k == 'hidden_layer_sizes' and isinstance(v, list):
            restored[k] = tuple(v)                                  # MLP requires a tuple, not a list
        elif isinstance(v, str) and v.startswith('__estimator__'):
            restored[k] = _str_to_estimator(v)
        else:
            restored[k] = v
    return restored


# ── Pipeline fingerprint ──────────────────────────────────────────────────────

def compute_fingerprint(features_list, param_spaces):
    """
    Compute a 12-character MD5 fingerprint of the pipeline state:
    selected features + hyperparameter search spaces.

    The fingerprint is stored alongside the cached hyperparameters in the JSON
    file. If it changes (different FCBF features, or edited search spaces),
    the cache is invalidated and RandomizedSearchCV reruns from scratch.
    Handles param_spaces that are a dict or a list of dicts (e.g. LR).
    """
    def _s(v):
        return sorted([str(x) for x in v]) if isinstance(v, list) else str(v)

    def _normalise_space(v):
        """Merge a list of dicts into a single canonical dict for hashing."""
        if isinstance(v, list):
            merged = {}
            for d in v:
                for pk, pv in d.items():
                    merged.setdefault(pk, set()).update(
                        [str(x) for x in (pv if isinstance(pv, list) else [pv])]
                    )
            return {pk: sorted(pv) for pk, pv in merged.items()}
        return {pk: _s(pv) for pk, pv in v.items()}

    content = {
        'features':     sorted(features_list),
        'param_spaces': {k: _normalise_space(v) for k, v in param_spaces.items()},
    }
    return hashlib.md5(
        _json.dumps(content, sort_keys=True).encode()
    ).hexdigest()[:12]


# ── SHAP beeswarm colour helper ───────────────────────────────────────────────

def get_beeswarm_colours(X_scaled, feat_idx, feature_name, X_norm, binary_features):
    """
    Return (colours, is_binary) for one feature in a SHAP beeswarm scatter plot.

    Binary features (detected on unscaled X_train — see Step 3):
      · colour = '#d62728' (red)  if original value was 1 (condition present)
      · colour = '#1f77b4' (blue) if original value was 0 (condition absent)
      · Binarisation on scaled data: value > 0 → class 1; ≤ 0 → class 0
        (valid because StandardScaler centres the mean at 0)

    Continuous features:
      · colour = normalised feature value in [0, 1] → used with cmap='RdYlBu_r'

    IMPORTANT: binary detection must be done on the original unscaled X_train
    before StandardScaler is applied. The set `binary_features` is built in
    Step 3 for this reason — do not use the scaled values for this check.
    """
    if feature_name in binary_features:
        vals    = X_scaled[:, feat_idx]
        colours = np.where(vals > 0, '#d62728', '#1f77b4')  # red=1 (present), blue=0 (absent)
        return colours, True
    return X_norm[:, feat_idx], False


# ── Youden-threshold Balanced Accuracy ───────────────────────────────────────

def bal_acc_at_youden(y_true, probs):
    """
    Compute Balanced Accuracy evaluated at each model's own Youden threshold.

    The Youden index J = Sensitivity + Specificity − 1 is maximised along the
    ROC curve. The threshold that achieves this maximum is used as the decision
    boundary, and Balanced Accuracy is computed at that point.

    Why this matters for the Voting Ensemble tiebreaker (Step 5.2):
    After Platt sigmoid calibration, the probability scale is rescaled, so
    the natural decision boundary shifts away from 0.5. Comparing Bal.Acc at
    0.5 can unfairly favour the candidate whose sigmoid pushes scores closest
    to 0.5, regardless of actual discriminative power. Evaluating each
    candidate at its own Youden threshold removes this artefact.

    Returns: (balanced_accuracy, youden_threshold)
    """
    fpr, tpr, thr = roc_curve(y_true, probs)
    j_idx   = np.argmax(tpr - fpr)   # index of maximum Youden index
    thr_opt = thr[j_idx]
    preds   = (probs >= thr_opt).astype(int)
    return balanced_accuracy_score(y_true, preds), thr_opt


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — DATA LOADING
# ════════════════════════════════════════════════════════════════════════════

# ── Training and internal validation sets (Valladolid 2023) ──────────────────
# The split was generated by IAvsHT_Creation_Split.py using an iterative seed
# search that guarantees statistical balance across four control variables
# (age, critical preoperative status, bivalvular+coronary, aorta+valve).
X_train_df, y_train_s, y_ht_train = load_sheet('IAvsHT_VALLADOLID_2023.xlsx', 'Train')
X_val_df,   y_val_s,   y_ht_val   = load_sheet('IAvsHT_VALLADOLID_2023.xlsx', 'Val')

# Store column names before converting to numpy — required for FCBF and SHAP labels
feature_names = X_train_df.columns.tolist()

X_train = X_train_df.to_numpy()
X_val   = X_val_df.to_numpy()
y_train = y_train_s.to_numpy()
y_val   = y_val_s.to_numpy()

# ── External test cohorts ─────────────────────────────────────────────────────
# Each external cohort is read from the first sheet (index 0) of its Excel file.
# Column order is aligned to the training set to prevent silent feature mismatches.
EXTERNAL_SOURCES = {
    'Valladolid': 'IAvsHT_VALLADOLID_2025.xlsx',   # 150 patients
    'Granada':    'IAvsHT_GRANADA.xlsx',            # 150 patients
    'Salamanca':  'IAvsHT_SALAMANCA.xlsx',          # 162 patients
}

data_tests = {}
for name, path in EXTERNAL_SOURCES.items():
    X_ext_df, y_ext_s, y_raw_ext = load_sheet(path, 0)
    X_ext_df = X_ext_df[feature_names]              # enforce column order from training set
    data_tests[name] = {
        'X':     X_ext_df.to_numpy(),
        'y':     y_ext_s.to_numpy(),
        'y_raw': y_raw_ext,                         # Heart Team clinical decision (comparator)
    }

# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — PREPROCESSING
# ════════════════════════════════════════════════════════════════════════════

# ── 3.1 Remove zero-variance columns ─────────────────────────────────────────
# A feature with zero standard deviation carries no discriminative information.
# Removal is computed on train only to avoid any form of data leakage.
std_dev        = np.std(X_train, axis=0)
zero_var_idx   = np.where(std_dev == 0)[0]
if len(zero_var_idx) > 0:
    print(f'Removing {len(zero_var_idx)} zero-variance columns.')
    X_train = np.delete(X_train, zero_var_idx, axis=1)
    X_val   = np.delete(X_val,   zero_var_idx, axis=1)
    for k in data_tests:
        data_tests[k]['X'] = np.delete(data_tests[k]['X'], zero_var_idx, axis=1)
    for idx in sorted(zero_var_idx, reverse=True):
        del feature_names[idx]                       # keep feature_names in sync with the arrays

# ── 3.2 Remove collinear columns (Pearson |r| > 0.95, computed on train) ─────
# Near-perfectly correlated features carry redundant information and inflate
# apparent SHAP importance. Only one representative per collinear group is kept.
CORR_THRESHOLD = 0.95
df_tmp      = pd.DataFrame(X_train, columns=feature_names)
corr_matrix = df_tmp.corr().abs()
upper_tri   = corr_matrix.where(                     # upper triangle only (avoid double-counting pairs)
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)
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
# EDAD (continuous age) is forced into the final feature set regardless of FCBF
# output: age is the strongest single predictor in this domain and its removal
# would compromise clinical interpretability.
#
# EDAD>75 (binary age threshold) is excluded entirely from the model — it is a
# derived discretisation of EDAD that would introduce redundancy. It is retained
# only in the descriptive statistical analysis (IAvsHT_Statistical_Analysis.py).
COL_EDAD     = 'EDAD'
COL_EDAD_BIN = 'EDAD>75'

# Extract the EDAD column index BEFORE removing EDAD>75 (positions may shift)
idx_edad_forzada = feature_names.index(COL_EDAD) if COL_EDAD in feature_names else None

if COL_EDAD_BIN in feature_names:
    idx_bin = feature_names.index(COL_EDAD_BIN)
    X_train = np.delete(X_train, idx_bin, axis=1)
    X_val   = np.delete(X_val,   idx_bin, axis=1)
    for k in data_tests:
        data_tests[k]['X'] = np.delete(data_tests[k]['X'], idx_bin, axis=1)
    feature_names.pop(idx_bin)
    # Recalculate the EDAD index after removal — position may have shifted
    idx_edad_forzada = feature_names.index(COL_EDAD) if COL_EDAD in feature_names else None
    print(f'Excluded \'{COL_EDAD_BIN}\' from model (descriptive analysis only).')

# ── 3.4 StandardScaler ───────────────────────────────────────────────────────
# The scaler is fitted exclusively on the training set and applied (transform
# only) to val and all external cohorts, preventing any form of data leakage.
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)           # fit + transform on train
X_val_sc   = scaler.transform(X_val)                 # transform only — use train statistics
for k in data_tests:
    data_tests[k]['X_sc'] = scaler.transform(data_tests[k]['X'])

# ── 3.5 FCBF feature selection ────────────────────────────────────────────────
# FCBF is based on Symmetrical Uncertainty, which requires discrete inputs.
# A discretised copy of the scaled training set is created solely for the FCBF
# computation; the actual model and SHAP analysis use the continuous scaled values.
#
# EDAD is excluded from FCBF (it enters the model unconditionally) to prevent
# the algorithm from discarding other informative features as redundant with age.
FCBF_THRESHOLD = 0.001   # minimum SU with the target to enter the candidate pool

X_train_disc = X_train_sc.copy()
for col_idx in range(X_train_disc.shape[1]):
    if col_idx == idx_edad_forzada:
        continue                                      # EDAD: keep continuous, exclude from discretisation
    col_vals = X_train_disc[:, col_idx]
    if len(np.unique(col_vals)) > 2:                  # binary features are already discrete
        X_train_disc[:, col_idx] = pd.cut(
            col_vals, bins=10, labels=False, duplicates='drop'
        ).astype(float)                               # 10-bin equal-width binning via pandas.cut

# ── 3.5a Tolerance sweep: how N features and AUC change with FCBF tolerance ──
# Before committing to a tolerance value, we sweep a grid and plot:
#   · Left axis:  N features selected by FCBF
#   · Right axis: Validation AUC of two fast proxy classifiers (LR and RF)
# The elbow of the N-features curve (where adding more tolerance no longer
# increases feature count) guides the final choice of FCBF_TOLERANCIA.
TOL_GRID = np.round(np.arange(0.30, 3.0, 0.1), 2)
curva_nfeatures, curva_auc_lr, curva_auc_rf = [], [], []

# Random Forest proxy: fast AUC estimate before full hyperparameter optimisation
_rf_proxy = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0,
                                   class_weight='balanced')

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

# Plot: dual Y-axis (N features left, AUC right)
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
plt.savefig(os.path.join(OUTPUT_DIR, 'fcbf_tolerance_local.png'), dpi=200, bbox_inches='tight')
plt.show()

# ── 3.5b Run definitive FCBF with chosen tolerance ───────────────────────────
# The tolerance is fixed at the elbow of the N-features curve identified above.
# EDAD is excluded from FCBF and added back unconditionally afterwards.
FCBF_TOLERANCE = 0.6

if idx_edad_forzada is not None:
    cols_for_fcbf     = [i for i in range(X_train_disc.shape[1]) if i != idx_edad_forzada]
    idx_fcbf_local    = fcbf(X_train_disc[:, cols_for_fcbf], y_train,
                             threshold=FCBF_THRESHOLD, tolerancia=FCBF_TOLERANCE)
    # Translate local indices (without EDAD) back to global indices (with EDAD)
    selected_indices  = sorted([cols_for_fcbf[i] for i in idx_fcbf_local] + [idx_edad_forzada])
else:
    selected_indices  = sorted(fcbf(X_train_disc, y_train,
                                    threshold=FCBF_THRESHOLD, tolerancia=FCBF_TOLERANCE))

features_final = [feature_names[i] for i in selected_indices]
print(f'\nFCBF (tolerance={FCBF_TOLERANCE}): {len(features_final)} features selected:')
print(features_final)

# ── 3.5c Detect binary features on original (unscaled) X_train ───────────────
# Binary features must be identified before scaling because StandardScaler maps
# {0, 1} to arbitrary floats, making binary detection unreliable on scaled data.
# This set is used later in SHAP beeswarm plots: binary features receive
# categorical colouring (blue/red) instead of the continuous RdYlBu_r colormap,
# which would otherwise produce ambiguous intermediate yellow values.
binary_features = set()
for fname in features_final:
    col_idx = feature_names.index(fname)
    vals    = np.unique(X_train[:, col_idx])
    if len(vals) == 2 and set(vals).issubset({0, 1}):
        binary_features.add(fname)
print(f'\nBinary features (detected on unscaled X_train): {sorted(binary_features)}')
print(f'Continuous features: {[f for f in features_final if f not in binary_features]}')

# ── 3.6 Symmetrical Uncertainty bar chart ────────────────────────────────────
# Bar chart of individual SU (feature relevance with the target) for all features
# above FCBF_THRESHOLD, coloured by whether they were selected or discarded.
su_items = []
for i in range(X_train_disc.shape[1]):
    su = symmetrical_uncertainty(X_train_disc[:, i], y_train)
    if su >= FCBF_THRESHOLD:
        su_items.append((su, feature_names[i], i in selected_indices))
su_items.sort(key=lambda t: t[0], reverse=True)   # descending SU

su_vals   = [t[0] for t in su_items]
su_names  = [t[1] for t in su_items]
su_colors = ['#d62728' if t[2] else '#4C72B0' for t in su_items]  # red=selected, blue=discarded

fig_su, ax_su = plt.subplots(figsize=(9, max(8, len(su_vals) * 0.32)))  # dynamic height
ax_su.barh(range(len(su_vals)), su_vals, color=su_colors, edgecolor='white', linewidth=0.4)
ax_su.set_yticks(range(len(su_names)))
ax_su.set_yticklabels(su_names, fontsize=7.5)
ax_su.invert_yaxis()                               # highest SU at the top
ax_su.set_xlabel('Symmetrical Uncertainty (SU)', fontsize=11)
ax_su.set_title('Individual feature relevance (SU with target)', fontsize=12, fontweight='bold')
ax_su.grid(axis='x', linestyle='--', alpha=0.4)
ax_su.legend(handles=[
    mpatches.Patch(color='#d62728', label=f'Selected by FCBF ({sum(t[2] for t in su_items)})'),
    mpatches.Patch(color='#4C72B0', label=f'Discarded as redundant ({sum(not t[2] for t in su_items)})'),
], fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fcbf_su_ranking_local.png'), dpi=200, bbox_inches='tight')
plt.show()

# ── 3.7 Apply feature selection to all sets ───────────────────────────────────
# From this point on, only X_train_sel, X_val_sel and data_tests[k]['X_sel']
# are used for model training, validation and external evaluation.
X_train_sel = X_train_sc[:, selected_indices]
X_val_sel   = X_val_sc[:,   selected_indices]
for k in data_tests:
    data_tests[k]['X_sel'] = data_tests[k]['X_sc'][:, selected_indices]

# ════════════════════════════════════════════════════════════════════════════
# STEP 4 — HYPERPARAMETER OPTIMISATION
# ─────────────────────────────────────────────────────────────────────────────
# All 12 classifiers are optimised before any validation comparison.
# This ensures the Step 4.5 ranking is a fair contest between fully-tuned
# architectures rather than between default and optimised models.
#
# Caching strategy (JSON + fingerprint):
#   First run  : RandomizedSearchCV finds optimal params → saved to JSON.
#   Later runs : params loaded from JSON, model re-trained (~10–50× faster).
#   Cache miss : if FCBF features or search spaces change, the MD5 fingerprint
#                changes, the cache is invalidated, and the search reruns.
# ════════════════════════════════════════════════════════════════════════════

# ── 4.1 Classifier factory ───────────────────────────────────────────────────
# A factory function returns a fresh instance on every call, preventing
# RandomizedSearchCV from inheriting residual state from previous fits.

def make_classifier(name):
    """Return a freshly instantiated classifier for the given name."""
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
# LR uses a list of dicts (one per solver group) because different solvers
# support different penalty types; RandomizedSearchCV handles lists natively.

PARAM_SPACES = {
    'LR': [
        {   # saga: supports l1, l2 and elasticnet
            'solver': ['saga'], 'penalty': ['l1', 'l2', 'elasticnet'],
            'l1_ratio': [0.5], 'class_weight': ['balanced', None],
        },
        {   # liblinear: supports only l1 and l2
            'solver': ['liblinear'], 'penalty': ['l1', 'l2'],
            'class_weight': ['balanced', None],
        },
    ],
    'LDA': {
        'solver':    ['lsqr', 'eigen'],
        'shrinkage': [None, 'auto', 0.1, 0.5, 0.9],   # covariance regularisation (lsqr/eigen only)
        'tol':       [1e-4, 1e-3, 1e-2],
    },
    'KNN': {
        'n_neighbors': [3, 5, 7, 11, 15],              # odd values avoid ties
        'weights':     ['uniform', 'distance'],
        'metric':      ['euclidean', 'manhattan', 'minkowski'],
        'p':           [1, 2],                          # Minkowski order: 1=Manhattan, 2=Euclidean
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
        'alpha':              [0.0001, 0.01, 0.1],      # L2 regularisation
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
        'max_features':                [0.3, 0.5, 0.7],   # feature fraction per tree
        'max_samples':                 [0.7, 0.8, 1.0],   # sample fraction per tree
        'estimator__max_depth':        [3, 5, 10, None],
        'estimator__min_samples_leaf': [1, 2, 4],
        'estimator__class_weight':     ['balanced', None],
    },
    'AdaBoost': {
        'n_estimators':  [50, 100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.5, 1.0],          # contribution of each weak learner
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
        'subsample':        [0.7, 0.8, 1.0],              # stochastic boosting sample fraction
        'max_features':     ['sqrt', 'log2', None],
    },
    'RUSBoost': {
        'n_estimators':      [50, 100, 200, 300],
        'learning_rate':     [0.01, 0.1, 0.5, 1.0],
        'sampling_strategy': ['auto', 0.5, 0.75, 1.0],   # majority-class undersampling ratio
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
        'min_child_weight': [1, 3, 5],                    # leaf regularisation
        'gamma':            [0, 0.1, 0.2],                # minimum loss reduction to split
        'subsample':        [0.7, 0.8, 0.9],
        'colsample_bytree': [0.6, 0.8],
        'scale_pos_weight': [1, 3, 5],                    # class imbalance correction
        'reg_alpha':        [0, 0.01, 0.1, 1],            # L1 regularisation
        'reg_lambda':       [1, 1.5, 2],                  # L2 regularisation
    },
}

# ── 4.3 Load or compute hyperparameters (JSON cache) ─────────────────────────
PARAMS_FILE     = os.path.join(OUTPUT_DIR, 'best_params_local.json')
cached_params   = {}
fingerprint_now = compute_fingerprint(features_final, PARAM_SPACES)

if os.path.exists(PARAMS_FILE):
    with open(PARAMS_FILE, 'r') as f:
        cache = json.load(f)
    fp_stored = cache.get('__fingerprint__', '')
    if fp_stored == fingerprint_now:
        # Pipeline unchanged: skip RandomizedSearchCV and reuse stored params
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

optimized_models = {}   # name → fitted estimator with best hyperparameters
new_params       = {}   # name → params found in this run (for cache update)

print('\n--- STEP 4.4: HYPERPARAMETER OPTIMISATION (all 12 classifiers) ---')
for name in CLASSIFIER_NAMES:
    print(f'\n  [{name}]')
    clf = make_classifier(name)                       # fresh instance to avoid residual state

    if name in cached_params:
        # Fast path: load from cache and refit on the selected feature set
        params = cached_params[name]
        print(f'    → Loading cached params: {params}')
        clf.set_params(**params)
        clf.fit(X_train_sel, y_train)
        best_clf         = clf
        new_params[name] = params

    elif name in PARAM_SPACES:
        # Search path: RandomizedSearchCV with 200 random combinations, 5-fold CV
        search = RandomizedSearchCV(
            clf,
            PARAM_SPACES[name],
            n_iter=200,          # number of random parameter combinations to evaluate
            scoring='roc_auc',   # optimise for discriminative ability
            cv=5,                # stratified 5-fold cross-validation on the training set
            n_jobs=-1,           # use all available CPU cores
            random_state=0,
        )
        search.fit(X_train_sel, y_train)
        best_clf         = search.best_estimator_
        new_params[name] = search.best_params_
        print(f'    → Best params found: {search.best_params_}')

    else:
        # No search space defined: train with sklearn defaults
        clf.fit(X_train_sel, y_train)
        best_clf = clf
        print('    → No search space defined; trained with defaults.')

    optimized_models[name] = best_clf

# Save all known params to JSON (merge cache + new results)
merged_params = {**cached_params, **new_params}
cache_out     = {'__fingerprint__': fingerprint_now}
cache_out.update({k: params_to_json(v) for k, v in merged_params.items()})
with open(PARAMS_FILE, 'w') as f:
    json.dump(cache_out, f, indent=2)
print(f'\nHyperparameters saved to \'{PARAMS_FILE}\' (fingerprint: {fingerprint_now}).')

# ── 4.5 Rank all 12 optimised classifiers on the validation set ──────────────
# Ranking is done only after all models are fully optimised, guaranteeing a
# fair comparison between architectures.
#
# Sorting criterion:
#   Primary  : AUC-ROC (descending)
#   Tiebreaker: if two adjacent models differ by less than DELTA_TIEBREAK in
#               AUC, the one with higher Balanced Accuracy is ranked first.
print('\n--- STEP 4.5: POST-OPTIMISATION RANKING (validation set) ---')

DELTA_TIEBREAK = 0.01
results = []
for name, clf in optimized_models.items():
    preds_val  = clf.predict(X_val_sel)
    probs_val  = clf.predict_proba(X_val_sel)[:, 1] if hasattr(clf, 'predict_proba') else None
    results.append(compute_metrics(y_val, preds_val, probs_val, name))

df_results = pd.DataFrame(results).sort_values('AUC-ROC', ascending=False).reset_index(drop=True)

# Pairwise tiebreaker pass
ordered, pending = [], df_results.to_dict('records')
while pending:
    current = pending.pop(0)
    if pending and abs(current['AUC-ROC'] - pending[0]['AUC-ROC']) < DELTA_TIEBREAK:
        # Tiebreaker: promote the model with higher Balanced Accuracy
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
# The top-3 classifiers by ranking (Step 4.5) are calibrated using Platt
# sigmoid scaling on the validation set (cv='prefit': the base estimator is
# already fitted; only the sigmoid layer is learnt from X_val_sel, y_val).
# Platt calibration corrects the probability scale without affecting the AUC
# (monotone transformation).
TOP_N = 3
top_models = df_results.head(TOP_N)
_top_display = top_models[['Model', 'AUC-ROC']]
print(f'\nTop {TOP_N} selected for Voting Ensemble:\n{_top_display}')

tuned_estimators  = []   # (name, estimator) pairs for VotingClassifier
calibrated_models = {}   # name → CalibratedClassifierCV

for name in top_models['Model']:
    best_clf = optimized_models[name]
    tuned_estimators.append((name, best_clf))
    cal_model = CalibratedClassifierCV(estimator=best_clf, method='sigmoid', cv='prefit')
    cal_model.fit(X_val_sel, y_val)
    calibrated_models[name] = cal_model
    print(f'  → {name} calibrated on validation set.')

# ── 5.2 Build and calibrate the Voting Ensemble ───────────────────────────────
# VotingClassifier with soft voting: each class receives the average of the
# predicted probabilities from the three base classifiers.
# When .fit() is called, sklearn internally clones the base estimators and
# re-trains them on X_train_sel — the already-fitted objects are not mutated.
print('\nGenerating Voting Ensemble...')
voting_clf = VotingClassifier(estimators=tuned_estimators, voting='soft')
voting_clf.fit(X_train_sel, y_train)

# Calibrate the Voting Ensemble using the same Platt procedure
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
# If |ΔAUC| < DELTA_TIEBREAK, a tiebreaker is activated.
# Each candidate is evaluated at its OWN Youden-optimal threshold rather than
# at the default 0.5. After Platt calibration, 0.5 is no longer the natural
# decision boundary, and comparing Bal.Acc at 0.5 can unfairly favour the
# candidate whose sigmoid compresses scores toward 0.5.
# See bal_acc_at_youden() in Step 1 for a full explanation.
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
# The default threshold of 0.5 is suboptimal in imbalanced clinical settings.
# The Youden index J = Sensitivity + Specificity − 1 is maximised across the
# full ROC curve to find the threshold that best balances both error types.
# This single threshold is then applied unchanged to all external test cohorts.
probs_val  = final_model.predict_proba(X_val_sel)[:, 1]
print(f'\nProbability statistics (validation set):')
print(f'  Min: {probs_val.min():.4f} | Mean: {probs_val.mean():.4f} | Max: {probs_val.max():.4f}')

fpr_val, tpr_val, thresholds_val = roc_curve(y_val, probs_val)
youden_idx = np.argmax(tpr_val - fpr_val)   # index of maximum Youden index
final_thr  = thresholds_val[youden_idx]     # Youden-optimal threshold
best_se    = tpr_val[youden_idx]
best_sp    = 1 - fpr_val[youden_idx]

print(f'\n--- THRESHOLD OPTIMISATION (Youden index, validation set) ---')
print(f'Optimal clinical threshold (J): {final_thr:.4f}')
print(f'Expected sensitivity:           {best_se:.2%}')
print(f'Expected specificity:           {best_sp:.2%}')

# ════════════════════════════════════════════════════════════════════════════
# STEP 6 — EXTERNAL TEST EVALUATION
# ════════════════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print('STEP 6 — EXTERNAL TEST RESULTS')
print('='*70)

n_tests = len(data_tests)
fig, axes = plt.subplots(nrows=n_tests, ncols=2, figsize=(12, 5 * n_tests))
if n_tests == 1:
    axes = axes.reshape(1, 2)
test_results = []

for i, (test_name, test_data) in enumerate(data_tests.items()):
    X_test = test_data['X_sel']
    y_test = test_data['y']
    y_ht   = test_data['y_raw']

    # ── AI predictions ──────────────────────────────────────────────────────
    probs_test = final_model.predict_proba(X_test)[:, 1]
    preds_ia   = (probs_test >= final_thr).astype(int)   # apply Youden threshold
    m_ia = compute_metrics(y_test, preds_ia, probs_test, f'IA-{test_name}')
    test_results.append(m_ia)

    # ── Heart Team predictions ───────────────────────────────────────────────
    preds_ht = y_ht.values
    m_ht = compute_metrics(y_test, preds_ht, None, f'HT-{test_name}')
    test_results.append(m_ht)

    # ── Confusion matrices: AI (blue, left) | HT (orange, right) ────────────
    ConfusionMatrixDisplay.from_predictions(
        y_test, preds_ia, ax=axes[i, 0],
        cmap='Blues', colorbar=False, display_labels=['No surgery', 'Surgery']
    )
    axes[i, 0].set_title(f'AI on {test_name} (N={len(y_test)})', fontsize=10, fontweight='bold')

    ConfusionMatrixDisplay.from_predictions(
        y_test, preds_ht, ax=axes[i, 1],
        cmap='Oranges', colorbar=False, display_labels=['No surgery', 'Surgery']
    )
    axes[i, 1].set_title(f'Heart Team on {test_name} (N={len(y_test)})', fontsize=10, fontweight='bold')

plt.suptitle(
    f'Local training and validation strategy\n'
    f'Multicentre comparison: AI ({final_name}) vs Heart Team',
    fontsize=14, fontweight='bold', y=1.01
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'matrices_comparison_multicentre_local.png'),
            dpi=200, bbox_inches='tight')
plt.show()

# ── ROC curves per external cohort ───────────────────────────────────────────
COLOURS_COHORT = {'Valladolid': '#1f77b4', 'Granada': '#2ca02c', 'Salamanca': '#d62728'}
fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
for test_name, test_data in data_tests.items():
    probs_roc = final_model.predict_proba(test_data['X_sel'])[:, 1]
    fpr_r, tpr_r, _ = roc_curve(test_data['y'], probs_roc)
    auc_r = roc_auc_score(test_data['y'], probs_roc)
    ax_roc.plot(fpr_r, tpr_r, color=COLOURS_COHORT.get(test_name, '#7f7f7f'), linewidth=2,
                label=f'{test_name}  (N={len(test_data["y"])}, AUC={auc_r:.3f})')
ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random classifier')
ax_roc.set_xlabel('1 − Specificity', fontsize=11)
ax_roc.set_ylabel('Sensitivity', fontsize=11)
ax_roc.set_title(
    f'Local strategy — Multicentre ROC curves ({final_name})', fontsize=12, fontweight='bold'
)
ax_roc.legend(loc='lower right', fontsize=10)
ax_roc.grid(linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curves_local.png'), dpi=200, bbox_inches='tight')
plt.show()

# ── Summary table and Excel export ───────────────────────────────────────────
df_test_summary = pd.DataFrame(test_results)
print('\n--- EXTERNAL TEST SUMMARY ---')
print(df_test_summary.to_string(index=False))

print('\nWriting results to Excel...')
try:
    EXCEL_PATH = os.path.join(OUTPUT_DIR, 'IAvsHT_local.xlsx')
    with pd.ExcelWriter(EXCEL_PATH, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:

        # Sheet: classifier ranking
        RANK_COLS = ['Model', 'AUC-ROC', 'Sensitivity', 'Specificity', 'Accuracy',
                     'PPV', 'NPV', 'Bal. Accuracy', 'F1-Score', 'Kappa', 'MCC']
        RANK_COLS = [c for c in RANK_COLS if c in df_results.columns]
        df_results[RANK_COLS].to_excel(writer, sheet_name='Classifier_Ranking', index=False)

        # Sheet: AI performance per external cohort (metrics as rows, cohorts as columns)
        METRICS_ORDER = [
            ('Sensitivity', 'Sensitivity'), ('Specificity', 'Specificity'),
            ('Accuracy', 'Accuracy'),         ('PPV', 'PPV'),
            ('NPV', 'NPV'),                   ('F1-Score', 'F1-Score'),
            ('Bal. Accuracy', 'Bal. Accuracy'),('AUC-ROC', 'AUC-ROC'),
            ('Kappa', "Cohen's Kappa"),        ('MCC', 'MCC'),
        ]
        df_ia = df_test_summary[df_test_summary['Model'].str.startswith('IA-')].copy()
        df_ia['Cohort'] = df_ia['Model'].str.replace('IA-', '', regex=False)
        centres = list(df_ia['Cohort'])

        pivot_rows = []
        for key, label in METRICS_ORDER:
            row = {'Metric': label}
            values = []
            for centre in centres:
                r   = df_ia[df_ia['Cohort'] == centre]
                val = r[key].values[0] if len(r) > 0 and key in r.columns else None
                row[centre] = val if val is not None else ''
                if val is not None:
                    values.append(val)
            row['Mean ± SD'] = (f'{np.mean(values):.4f} ± {np.std(values):.4f}' if values else '')
            pivot_rows.append(row)

        pd.DataFrame(pivot_rows).to_excel(writer, sheet_name='External_Test_Performance', index=False)
    print(f'  Saved: {EXCEL_PATH}')
except Exception as e:
    print(f'  ERROR writing Excel: {e}')
    import traceback; traceback.print_exc()

# ════════════════════════════════════════════════════════════════════════════
# STEP 7 — SHAP EXPLAINABILITY
# ─────────────────────────────────────────────────────────────────────────────
# KernelExplainer is used because the final model (calibrated Voting Ensemble)
# does not expose a tree structure, making TreeExplainer inapplicable.
# KernelExplainer is model-agnostic: it approximates Shapley values by sampling
# input perturbations and measuring the change in model output.
#
# Background: K-Means (≤ 20 centroids) compresses the training set into a
# compact representative summary, reducing computation while preserving
# the marginal distribution of each feature.
#
# nsamples=100 per instance: the number of Monte Carlo perturbations used
# to estimate each Shapley value. Higher values increase precision at the
# cost of computation time.
#
# Figures generated:
#   Fig 1: shap_ranking_train_local.png       — Global importance on train
#   Fig 2: shap_beeswarm_train_local.png      — Beeswarm on train (class 0 | class 1)
#   Fig 3: shap_ranking_test_local.png        — Importance per external cohort
#   Fig 4: shap_beeswarm_test_local.png       — Beeswarm per cohort (class 0 | class 1)
#   Fig 5: shap_ranking_correct_local.png    — Importance on correctly classified only
#   Fig 6: shap_beeswarm_correct_local.png   — Beeswarm on correctly classified only
# ════════════════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print('STEP 7 — SHAP EXPLAINABILITY')
print('='*70)


def model_prob_surgery(X):
    """Wrapper for KernelExplainer: returns P(surgery) as a 1D array."""
    return final_model.predict_proba(X)[:, 1]


# ── Build K-Means background ──────────────────────────────────────────────────
print('Building K-Means background for KernelExplainer...')
n_unique   = len(np.unique(X_train_sel, axis=0))
n_clusters = min(20, n_unique)                       # cap at 20 to balance speed and precision
print(f'  Unique rows in Train: {n_unique} → using {n_clusters} K-Means clusters')
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', message='Clustering metrics expects discrete values')
    X_background = shap.kmeans(X_train_sel, n_clusters)

explainer = shap.KernelExplainer(model_prob_surgery, X_background)

# ── 7.1 Compute SHAP values on the training set ───────────────────────────────
print(f'Computing SHAP values on Train ({X_train_sel.shape[0]} patients)...')
shap_train    = explainer.shap_values(X_train_sel, nsamples=100)
imp_train     = np.abs(shap_train).mean(axis=0)   # mean |SHAP| per feature
idx_train_asc = np.argsort(imp_train)              # ascending → most important at top in barh

# ── Fig 1 — Global SHAP importance ranking (Train) ───────────────────────────
plt.figure(figsize=(9, 5))
plt.barh([features_final[i] for i in idx_train_asc], imp_train[idx_train_asc],
         color='#2c7bb6', edgecolor='white')
plt.xlabel('Mean absolute Shapley Values', fontsize=11)
plt.title(
    f'Local training and validation strategy\n'
    f'SHAP importance ranking — Training set\nModel: {final_name}',
    fontsize=15, fontweight='bold'
)
plt.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'shap_ranking_train_local.png'), dpi=200, bbox_inches='tight')
plt.show()
print('\nTop SHAP features (Train):')
for i in idx_train_asc[::-1]:
    print(f'  {features_final[i]:<45}: {imp_train[i]:.4f}')


# ── Shared beeswarm panel drawing function ────────────────────────────────────
def draw_beeswarm(ax, shap_vals, X_class_sc, class_label, n_patients,
                  axis_annotation, flip_sign=False):
    """
    Draw one SHAP beeswarm panel (single class, single cohort) on `ax`.

    Parameters
    ----------
    ax              : matplotlib Axes to draw on
    shap_vals       : (N, F) SHAP values for this class subset
    X_class_sc      : (N, F) scaled feature matrix for this subset
    class_label     : string shown as the panel title
    n_patients      : number of patients in this subset
    axis_annotation : directional label placed below the X axis
    flip_sign       : if True, negate SHAP values so that positive = pushes
                      toward surgery; used for Class-0 panels to maintain a
                      consistent left=surgery / right=no-surgery convention.
    """
    if flip_sign:
        shap_vals = -shap_vals

    # Sort features by mean |SHAP| in this class (ascending → least at bottom)
    imp_cl  = np.abs(shap_vals).mean(axis=0)
    idx_ord = np.argsort(imp_cl)

    # Normalise scaled feature values to [0,1] per feature for the continuous colormap.
    # IMPORTANT: do NOT use these normalised values to detect binary features —
    # StandardScaler output no longer contains {0,1}. Use `binary_features` instead.
    X_norm = np.zeros_like(X_class_sc)
    for j in range(X_class_sc.shape[1]):
        col = X_class_sc[:, j]
        rng = col.max() - col.min()
        X_norm[:, j] = (col - col.min()) / rng if rng > 1e-10 else 0.5

    y_pos = np.arange(len(idx_ord))
    for i, feat_idx in enumerate(idx_ord):
        sv     = shap_vals[:, feat_idx]
        fname  = features_final[feat_idx]
        jitter = np.random.normal(0, 0.08, size=len(sv))   # vertical jitter to reduce overplotting
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


# ── Fig 2 — Beeswarm on Train (1×2: class 0 | class 1) ───────────────────────
mask_c0_tr = (y_train == 0)
mask_c1_tr = (y_train == 1)
print(f'  Train — Class 0 (No surgery): {mask_c0_tr.sum()} patients')
print(f'  Train — Class 1 (Surgery):    {mask_c1_tr.sum()} patients')

fig2, axes2 = plt.subplots(1, 2, figsize=(16, 8))
draw_beeswarm(axes2[0], shap_train[mask_c0_tr], X_train_sel[mask_c0_tr],
              'Class 0 (No surgery)', mask_c0_tr.sum(),
              '← Favours "SURGERY" | Favours "NO SURGERY" →', flip_sign=True)
draw_beeswarm(axes2[1], shap_train[mask_c1_tr], X_train_sel[mask_c1_tr],
              'Class 1 (Surgery)', mask_c1_tr.sum(),
              '← Favours "NO SURGERY" | Favours "SURGERY" →', flip_sign=False)
plt.suptitle(
    f'Local training and validation strategy\n'
    f'SHAP beeswarm — Training set\nModel: {final_name}',
    fontsize=20, fontweight='bold', y=1.02
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'shap_beeswarm_train_local.png'), dpi=200, bbox_inches='tight')
plt.show()

# ── 7.2 Compute SHAP values per external test cohort ─────────────────────────
print('\nComputing SHAP per external cohort...')
shap_per_cohort = {}
for cname, cdata in data_tests.items():
    print(f'  {cname} ({cdata["X_sel"].shape[0]} patients)...')
    shap_per_cohort[cname] = explainer.shap_values(cdata['X_sel'], nsamples=100)

# ── Fig 3 — SHAP importance ranking per test cohort ───────────────────────────
fig3, axes3 = plt.subplots(len(data_tests), 1, figsize=(8, 5 * len(data_tests)))
if len(data_tests) == 1:
    axes3 = [axes3]

# Unified X-axis scale for fair comparison across cohorts
x_max = max(np.abs(sv).mean(axis=0).max() for sv in shap_per_cohort.values()) * 1.1

for ax, (cname, shap_c) in zip(axes3, shap_per_cohort.items()):
    imp_c = np.abs(shap_c).mean(axis=0)
    idx_c = np.argsort(imp_c)
    ax.barh([features_final[i] for i in idx_c], imp_c[idx_c],
            color=COLOURS_COHORT.get(cname, '#7f7f7f'), edgecolor='white')
    ax.set_xlabel('Mean absolute Shapley Values', fontsize=10)
    ax.set_title(f'{cname} (N={data_tests[cname]["y"].shape[0]})', fontsize=11, fontweight='bold')
    ax.set_xlim(0, x_max)
    ax.grid(axis='x', linestyle='--', alpha=0.4)

plt.suptitle(
    f'Local training and validation strategy\n'
    f'SHAP importance ranking — Test sets\nModel: {final_name}',
    fontsize=15, fontweight='bold'
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'shap_ranking_test_local.png'), dpi=200, bbox_inches='tight')
plt.show()

# ── Fig 4 — Beeswarm per test cohort (3×2) ───────────────────────────────────
COHORTS_ORDERED = ['Valladolid', 'Granada', 'Salamanca']
fig4, axes4 = plt.subplots(3, 2, figsize=(16, 20))

for row, cname in enumerate(COHORTS_ORDERED):
    X_c    = data_tests[cname]['X_sel']
    y_c    = data_tests[cname]['y']
    shap_c = shap_per_cohort[cname]
    mask_c0, mask_c1 = (y_c == 0), (y_c == 1)
    print(f'    {cname} — Class 0: {mask_c0.sum()} | Class 1: {mask_c1.sum()}')
    draw_beeswarm(axes4[row, 0], shap_c[mask_c0], X_c[mask_c0],
                  f'{cname}\nClass 0 (No surgery)', mask_c0.sum(),
                  '← Favours "SURGERY" | Favours "NO SURGERY" →', flip_sign=True)
    draw_beeswarm(axes4[row, 1], shap_c[mask_c1], X_c[mask_c1],
                  f'{cname}\nClass 1 (Surgery)', mask_c1.sum(),
                  '← Favours "NO SURGERY" | Favours "SURGERY" →', flip_sign=False)

plt.suptitle(
    f'Local training and validation strategy\n'
    f'SHAP beeswarm — Test sets\nModel: {final_name}',
    fontsize=20, fontweight='bold', y=0.98
)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(OUTPUT_DIR, 'shap_beeswarm_test_local.png'), dpi=200, bbox_inches='tight')
plt.show()

# ── 7.3 SHAP on correctly classified patients only ────────────────────────────
# Comparing the SHAP profile of all test patients against only the correctly
# classified ones reveals where the model's uncertainty is concentrated.
# What disappears or changes when errors are excluded shows which feature
# combinations and value ranges drive classification mistakes.
print('\nComputing SHAP for correctly classified patients (Figs 5 & 6)...')

correct_by_cohort = {}
for cname, cdata in data_tests.items():
    X_c     = cdata['X_sel']
    y_c     = cdata['y']
    preds_c = (final_model.predict_proba(X_c)[:, 1] >= final_thr).astype(int)
    mask_ok = (y_c == preds_c)                       # True where prediction matches ground truth
    correct_by_cohort[cname] = {
        'mask_ok':  mask_ok,
        'X_ok':     X_c[mask_ok],
        'y_ok':     y_c[mask_ok],
        'shap_ok':  shap_per_cohort[cname][mask_ok],
        'n_ok':     mask_ok.sum(),
        'n_total':  len(y_c),
    }
    print(f'  {cname}: {mask_ok.sum()}/{len(y_c)} correctly classified')

# ── Fig 5 — Importance ranking on correctly classified patients ───────────────
fig5, axes5 = plt.subplots(len(data_tests), 1, figsize=(8, 5 * len(data_tests)))
if len(data_tests) == 1:
    axes5 = [axes5]

x_max_ok = max(
    np.abs(d['shap_ok']).mean(axis=0).max()
    for d in correct_by_cohort.values() if d['n_ok'] > 0
) * 1.1

for ax, (cname, d) in zip(axes5, correct_by_cohort.items()):
    if d['n_ok'] == 0:
        ax.text(0.5, 0.5, 'No correctly classified patients',
                ha='center', va='center', transform=ax.transAxes)
        continue
    imp_ok = np.abs(d['shap_ok']).mean(axis=0)
    idx_ok = np.argsort(imp_ok)
    ax.barh([features_final[i] for i in idx_ok], imp_ok[idx_ok],
            color=COLOURS_COHORT.get(cname, '#7f7f7f'), edgecolor='white')
    ax.set_xlabel('Mean absolute Shapley Values', fontsize=10)
    ax.set_title(
        f'{cname} — Correctly classified (N={d["n_ok"]}/{d["n_total"]})',
        fontsize=11, fontweight='bold'
    )
    ax.set_xlim(0, x_max_ok)
    ax.grid(axis='x', linestyle='--', alpha=0.4)

plt.suptitle(
    f'Local training and validation strategy\n'
    f'SHAP importance ranking — Correctly classified patients\nModel: {final_name}',
    fontsize=15, fontweight='bold'
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, 'shap_ranking_correct_local.png'), dpi=200, bbox_inches='tight')
plt.show()

# ── Fig 6 — Beeswarm on correctly classified patients (3×2) ──────────────────
fig6, axes6 = plt.subplots(3, 2, figsize=(16, 20))

for row, cname in enumerate(COHORTS_ORDERED):
    d       = correct_by_cohort[cname]
    X_ok    = d['X_ok']
    y_ok    = d['y_ok']
    shap_ok = d['shap_ok']
    mask_tn = (y_ok == 0)   # TN: ground truth 0, model predicted 0
    mask_tp = (y_ok == 1)   # TP: ground truth 1, model predicted 1
    print(f'    {cname} — TN: {mask_tn.sum()} | TP: {mask_tp.sum()}')

    draw_beeswarm(axes6[row, 0], shap_ok[mask_tn], X_ok[mask_tn],
                  f'{cname}\nClass 0 — No surgery (TN)', mask_tn.sum(),
                  '← Favours "SURGERY" | Favours "NO SURGERY" →', flip_sign=True)
    draw_beeswarm(axes6[row, 1], shap_ok[mask_tp], X_ok[mask_tp],
                  f'{cname}\nClass 1 — Surgery (TP)', mask_tp.sum(),
                  '← Favours "NO SURGERY" | Favours "SURGERY" →', flip_sign=False)

plt.suptitle(
    f'Local training and validation strategy\n'
    f'SHAP beeswarm — Correctly classified patients\nModel: {final_name}',
    fontsize=20, fontweight='bold', y=0.98
)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(OUTPUT_DIR, 'shap_beeswarm_correct_local.png'), dpi=200, bbox_inches='tight')
plt.show()
