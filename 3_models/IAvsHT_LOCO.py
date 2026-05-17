##############################################################################
# IAvsHT — Leave-One-Cohort-Out (LOCO) Training and Validation Strategy
# ─────────────────────────────────────────────────────────────────────────────
# Partitioning strategy: The pipeline loops 4 times. In each iteration (fold),
# one cohort is held out as the external test set. The remaining 3 cohorts 
# are concatenated and split into an internal 80% Train / 20% Validation set.
#
# Data flow (4 folds):
#   Iter 1: Test = Valladolid 2023 | Train/Val = Vall'25 + Granada + Salamanca
#   Iter 2: Test = Valladolid 2025 | Train/Val = Vall'23 + Granada + Salamanca
#   ... (repeats for all 4 centers)
#
# Pipeline (Steps 1–7 inside the loop):
#   1. Imports, constants and shared function definitions
#   2. Data loading: 80/20 train/val split of the 3 active cohorts.
#   3. Preprocessing: zero-variance removal → collinearity pruning →
#      EDAD/EDAD>75 handling → StandardScaler → FCBF feature selection.
#   4. Hyperparameter optimisation (RandomizedSearchCV, 200 iter, 5-fold CV).
#   5. Classifier ranking → top-3 → Platt calibration → Voting Ensemble →
#      model selection → Youden threshold optimisation.
#   6. Test evaluation: Metrics for the single left-out cohort.
#   7. SHAP explainability: Compute and accumulate SHAP values.
#
# Post-Loop Consolidation:
#   - Joint 4x2 Confusion Matrices (AI vs HT for all folds).
#   - Joint ROC curves (all 4 test cohorts in one plot).
#   - Consolidated SHAP Rankings (4x1) and Beeswarms (4x2) for Train and Test.
#   - Global Excel summary with standard deviations across folds.
#
# Output directory : IAvsHT_loco/
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

# ── Information-theoretic helpers (used by FCBF) ─────────────────────────────

def entropy(vec):
    """Shannon entropy of a discrete vector (bits)."""
    _, counts = np.unique(vec, return_counts=True)
    probs = counts / len(vec)
    return -np.sum(probs * np.log2(probs + 1e-12))


def symmetrical_uncertainty(x, y):
    """Symmetrical Uncertainty (SU) between two discrete variables in [0, 1]."""
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
    if not m: return s
    cls_name, params_str = m.group(1), m.group(2)
    cls_map = {'DecisionTreeClassifier': DecisionTreeClassifier}
    cls     = cls_map.get(cls_name)
    if cls is None: raise ValueError(f'Unknown estimator: {cls_name}')
    kwargs = {}
    if params_str.strip():
        for match in re.finditer(r'(\w+)=(\'(?:[^\'\\\\]|\\\\.)*\'|"(?:[^"\\\\]|\\\\.)*"|[^,]+)', params_str):
            k, raw_v = match.group(1).strip(), match.group(2).strip()
            try: kwargs[k] = ast.literal_eval(raw_v)
            except Exception: kwargs[k] = raw_v
    return cls(**kwargs)

def params_to_json(params):
    safe = {}
    for k, v in params.items():
        if v is None: safe[k] = '__None__'
        elif isinstance(v, tuple): safe[k] = list(v)
        elif hasattr(v, 'get_params'): safe[k] = _estimator_to_str(v)
        else: safe[k] = v
    return safe

def params_from_json(params):
    restored = {}
    for k, v in params.items():
        if v == '__None__': restored[k] = None
        elif k == 'hidden_layer_sizes' and isinstance(v, list): restored[k] = tuple(v)
        elif isinstance(v, str) and v.startswith('__estimator__'): restored[k] = _str_to_estimator(v)
        else: restored[k] = v
    return restored

def compute_fingerprint(features_list, param_spaces):
    """Creates an MD5 hash of the current features and hyperparameter space for cache invalidation."""
    def _s(v): return sorted([str(x) for x in v]) if isinstance(v, list) else str(v)
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


# ── Shared SHAP beeswarm drawing function ─────────────────────────────────────

def get_beeswarm_colours(X_scaled, feat_idx, feature_name, X_norm, binary_features):
    """Determines whether to color a SHAP point categorically (binary) or continuously."""
    if feature_name in binary_features:
        vals    = X_scaled[:, feat_idx]
        colours = np.where(vals > 0, '#d62728', '#1f77b4')  # red=1 (present), blue=0 (absent)
        return colours, True
    return X_norm[:, feat_idx], False


def draw_beeswarm(ax, shap_vals, X_class_sc, class_label, n_patients,
                  axis_annotation, binary_features, features_final, flip_sign=False):
    """Draws a customized SHAP beeswarm scatter plot on a given matplotlib axis."""
    if flip_sign:
        shap_vals = -shap_vals

    imp_cl  = np.abs(shap_vals).mean(axis=0)
    idx_ord = np.argsort(imp_cl)

    # Normalize continuous features for the colormap to ensure accurate color mapping
    X_norm = np.zeros_like(X_class_sc)
    for j in range(X_class_sc.shape[1]):
        col = X_class_sc[:, j]
        rng = col.max() - col.min()
        X_norm[:, j] = (col - col.min()) / rng if rng > 1e-10 else 0.5

    y_pos = np.arange(len(idx_ord))
    for i, feat_idx in enumerate(idx_ord):
        sv     = shap_vals[:, feat_idx]
        fname  = features_final[feat_idx]
        jitter = np.random.normal(0, 0.08, size=len(sv))  # Add vertical jitter to prevent overlapping
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


# ════════════════════════════════════════════════════════════════════════════
# LOCO CONFIGURATION & PREPARATION
# ════════════════════════════════════════════════════════════════════════════

OUTPUT_DIR = 'IAvsHT_loco'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the raw datasets into memory to avoid reading from disk on every fold
RAW_COHORTS = {
    'Valladolid (2023)': pd.concat([pd.read_excel('IAvsHT_VALLADOLID_2023.xlsx', 'Train'), 
                                    pd.read_excel('IAvsHT_VALLADOLID_2023.xlsx', 'Val')], ignore_index=True),
    'Valladolid (2025)': pd.read_excel('IAvsHT_VALLADOLID_2025.xlsx', 0),
    'Granada':           pd.read_excel('IAvsHT_GRANADA.xlsx', 0),
    'Salamanca':         pd.read_excel('IAvsHT_SALAMANCA.xlsx', 0),
}
cohort_names = list(RAW_COHORTS.keys())
print(f'Loaded cohorts: { {k: len(v) for k, v in RAW_COHORTS.items()} }')

def split_Xy_df(df):
    """Helper to pop the target variables from the DataFrame."""
    df   = df.copy()
    y    = df.pop(df.columns[-1])    # Ground truth (always the last column)
    y_ht = df.pop('OBJETIVO HT')     # Heart Team clinical decision
    return df, y, y_ht


# ── Accumulators to consolidate data across all 4 folds for post-loop reporting ──
loco_global_test_results = []   # List of metric dictionaries from all folds
loco_accumulated_preds   = {}   # Predictions and probabilities for joint matrices/ROC
loco_accumulated_ranking = {}   # Classifier rankings per fold for the final Excel sheet
loco_accumulated_shap    = {}   # SHAP values per fold for the final 4x2 consolidated plots


# ════════════════════════════════════════════════════════════════════════════
# LOCO MAIN LOOP (Runs 4 times)
# ════════════════════════════════════════════════════════════════════════════
for FOLD_IDX, cohort_test in enumerate(cohort_names):

    print('\n' + '█'*75)
    print(f'  LOCO FOLD {FOLD_IDX+1}/4  —  TEST SET: {cohort_test}')
    print('█'*75)

    FOLD_TAG = cohort_test.lower().replace(' ', '_')   # Clean string for file naming
    FOLD_DIR = OUTPUT_DIR                              # All outputs go to the root LOCO dir

    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 2 — DATA LOADING (Train/Val/Test distribution for this fold)
    # ─────────────────────────────────────────────────────────────────────────────

    # 2.1 Concatenate the 3 remaining cohorts to form the Train/Val pool
    df_train_val = pd.concat(
        [df for name, df in RAW_COHORTS.items() if name != cohort_test],
        ignore_index=True
    )
    
    # 2.2 Split the pool into 80% Train and 20% Validation 
    # (Fixed seed based on FOLD_IDX ensures the split is strictly reproducible)
    rng_fold     = np.random.default_rng(42 + FOLD_IDX)
    idx_perm     = rng_fold.permutation(len(df_train_val))
    n_train_loco = int(round(len(df_train_val) * 0.80))
    
    idx_tr = idx_perm[:n_train_loco]
    idx_va = idx_perm[n_train_loco:]

    X_train_df, y_train_s, y_ht_train = split_Xy_df(df_train_val.iloc[idx_tr].reset_index(drop=True))
    X_val_df,   y_val_s,   y_ht_val   = split_Xy_df(df_train_val.iloc[idx_va].reset_index(drop=True))

    feature_names = X_train_df.columns.tolist()
    X_train       = X_train_df.to_numpy()
    X_val         = X_val_df.to_numpy()
    y_train       = y_train_s.to_numpy()
    y_val         = y_val_s.to_numpy()

    print(f'  Train: {len(X_train)} pac. | Val: {len(X_val)} pac. | Test ({cohort_test}): {len(RAW_COHORTS[cohort_test])} pac.')

    # 2.3 The left-out cohort becomes the sole test set for this iteration
    X_test_df, y_test_s, y_ht_test = split_Xy_df(RAW_COHORTS[cohort_test])
    X_test_df = X_test_df[feature_names]
    data_tests = {
        cohort_test: {
            'X':     X_test_df.to_numpy(),
            'y':     y_test_s.to_numpy(),
            'y_raw': y_ht_test,
        }
    }

    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 3 — PREPROCESSING (Pipeline re-run from scratch on the new subsets)
    # ─────────────────────────────────────────────────────────────────────────────

    # ── 3.1 Remove zero-variance columns (computed strictly on Train) ────────────
    std_dev      = np.std(X_train, axis=0)
    zero_var_idx = np.where(std_dev == 0)[0]
    if len(zero_var_idx) > 0:
        print(f'Removing {len(zero_var_idx)} zero-variance columns.')
        X_train = np.delete(X_train, zero_var_idx, axis=1)
        X_val   = np.delete(X_val,   zero_var_idx, axis=1)
        for k in data_tests:
            data_tests[k]['X'] = np.delete(data_tests[k]['X'], zero_var_idx, axis=1)
        for index in sorted(zero_var_idx, reverse=True):
            del feature_names[index]

    # ── 3.2 Remove collinear columns (Pearson |r| > 0.95, computed on Train) ─────
    CORR_THRESHOLD = 0.95
    df_temp      = pd.DataFrame(X_train, columns=feature_names)
    corr_matrix  = df_temp.corr().abs()
    upper_tri    = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop_names = [col for col in upper_tri.columns if any(upper_tri[col] > CORR_THRESHOLD)]
    if to_drop_names:
        print(f'Removing {len(to_drop_names)} collinear columns: {to_drop_names}')
        to_drop_indices = [feature_names.index(col) for col in to_drop_names]
        X_train = np.delete(X_train, to_drop_indices, axis=1)
        X_val   = np.delete(X_val,   to_drop_indices, axis=1)
        for k in data_tests:
            data_tests[k]['X'] = np.delete(data_tests[k]['X'], to_drop_indices, axis=1)
        for index in sorted(to_drop_indices, reverse=True):
            del feature_names[index]

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

    # ── 3.4 StandardScaler (Fit on Train, Transform all) ─────────────────────────
    scaler_global = StandardScaler()
    X_train_sc    = scaler_global.fit_transform(X_train)
    X_val_sc      = scaler_global.transform(X_val)
    for k in data_tests:
        data_tests[k]['X_sc'] = scaler_global.transform(data_tests[k]['X'])

    # ── 3.5 Discretisation for FCBF Symmetrical Uncertainty ──────────────────────
    FCBF_THRESHOLD = 0.001
    X_train_disc   = X_train_sc.copy()
    for col_idx in range(X_train_disc.shape[1]):
        if col_idx == idx_edad_forzada:
            continue
        col_vals = X_train_disc[:, col_idx]
        if len(np.unique(col_vals)) > 2:
            X_train_disc[:, col_idx] = pd.cut(col_vals, bins=10, labels=False, duplicates='drop').astype(float)

    # ── 3.5b FCBF Feature Selection (Hardcoded Tolerance for LOCO) ───────────────
    FCBF_TOLERANCE = 1.0  # Strict tolerance: discards feature if redundancy >= its own relevance
    
    if idx_edad_forzada is not None:
        _cols_fcbf      = [j for j in range(X_train_disc.shape[1]) if j != idx_edad_forzada]
        _idx_fcbf_local = fcbf(X_train_disc[:, _cols_fcbf], y_train,
                               threshold=FCBF_THRESHOLD, tolerancia=FCBF_TOLERANCE)
        selected_indices = sorted([_cols_fcbf[j] for j in _idx_fcbf_local] + [idx_edad_forzada])
    else:
        selected_indices = sorted(fcbf(X_train_disc, y_train,
                                       threshold=FCBF_THRESHOLD, tolerancia=FCBF_TOLERANCE))
                                       
    features_final = [feature_names[i] for i in selected_indices]
    print(f'\nFCBF (tolerance={FCBF_TOLERANCE}): {len(features_final)} features selected:')
    print(features_final)

    # ── 3.5c Detect binary features on original (unscaled) X_train ───────────────
    # We must detect these before scaling to ensure {0,1} are recognized properly.
    binary_features = set()
    for _fname in features_final:
        _col_idx_orig = feature_names.index(_fname)
        _vals = np.unique(X_train[:, _col_idx_orig])
        if len(_vals) == 2 and set(_vals).issubset({0, 1}):
            binary_features.add(_fname)
    print(f'\nBinary features detected: {sorted(binary_features)}')
    print(f'Continuous features: {[f for f in features_final if f not in binary_features]}')

    # Apply the selected features to all active subsets in this fold
    X_train_sel = X_train_sc[:, selected_indices]
    X_val_sel   = X_val_sc[:,   selected_indices]
    for k in data_tests:
        data_tests[k]['X_sel'] = data_tests[k]['X_sc'][:, selected_indices]


    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 4 — HYPERPARAMETER OPTIMISATION
    # ─────────────────────────────────────────────────────────────────────────────

    def make_classifier(name):
        """Returns a fresh instance to avoid carrying over state between CV folds."""
        return {
            'LR':              LogisticRegression(random_state=0, max_iter=1000),
            'LDA':             LinearDiscriminantAnalysis(),
            'KNN':             KNeighborsClassifier(),
            'SVM':             SVC(probability=True, random_state=0),
            'MLP':             MLPClassifier(random_state=0),
            'Random Forest':   RandomForestClassifier(random_state=0),
            'Extra Trees':     ExtraTreesClassifier(random_state=0),
            'Random Subspace': BaggingClassifier(estimator=DecisionTreeClassifier(), max_features=0.5, bootstrap=False, random_state=0),
            'AdaBoost':        AdaBoostClassifier(random_state=0),
            'LogitBoost':      GradientBoostingClassifier(loss='log_loss', random_state=0),
            'RUSBoost':        RUSBoostClassifier(random_state=0),
            'XGBoost':         XGBClassifier(random_state=0, eval_metric='logloss'),
        }[name]

    PARAM_SPACES = {
        'LR': [
            {'solver': ['saga'], 'penalty': ['l1', 'l2', 'elasticnet'], 'l1_ratio': [0.5], 'class_weight': ['balanced', None]},
            {'solver': ['liblinear'], 'penalty': ['l1', 'l2'], 'class_weight': ['balanced', None]},
        ],
        'LDA': {'solver': ['lsqr', 'eigen'], 'shrinkage': [None, 'auto', 0.1, 0.5, 0.9], 'tol': [1e-4, 1e-3, 1e-2]},
        'KNN': {'n_neighbors': [3, 5, 7, 11, 15], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan', 'minkowski'], 'p': [1, 2]},
        'SVM': {'C': [0.1, 1, 10, 50, 100], 'gamma': ['scale', 'auto', 0.1, 0.01, 0.001], 'kernel': ['rbf', 'linear', 'poly'], 'class_weight': ['balanced', None]},
        'MLP': {'max_iter': [1000, 1500], 'hidden_layer_sizes': [(50,), (30, 30), (50, 25)], 'activation': ['relu', 'tanh'], 'alpha': [0.0001, 0.01, 0.1], 'learning_rate_init': [0.001, 0.01]},
        'Random Forest': {'n_estimators': [200, 500, 800], 'max_depth': [5, 10, 15, None], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [2, 4, 6], 'max_features': ['sqrt', 'log2'], 'class_weight': ['balanced', 'balanced_subsample']},
        'Extra Trees': {'n_estimators': [200, 500, 800], 'max_depth': [5, 10, None], 'min_samples_leaf': [2, 4, 6], 'bootstrap': [True, False], 'class_weight': ['balanced', 'balanced_subsample']},
        'Random Subspace': {'n_estimators': [100, 200, 500], 'max_features': [0.3, 0.5, 0.7], 'max_samples': [0.7, 0.8, 1.0], 'estimator__max_depth': [3, 5, 10, None], 'estimator__min_samples_leaf': [1, 2, 4], 'estimator__class_weight': ['balanced', None]},
        'AdaBoost': {'n_estimators': [50, 100, 200, 300], 'learning_rate': [0.01, 0.1, 0.5, 1.0], 'estimator': [DecisionTreeClassifier(max_depth=1, class_weight='balanced'), DecisionTreeClassifier(max_depth=2, class_weight='balanced'), DecisionTreeClassifier(max_depth=3, class_weight='balanced')]},
        'LogitBoost': {'n_estimators': [100, 200, 300, 500], 'learning_rate': [0.01, 0.05, 0.1, 0.2], 'max_depth': [2, 3, 4, 5], 'min_samples_leaf': [2, 4, 6], 'subsample': [0.7, 0.8, 1.0], 'max_features': ['sqrt', 'log2', None]},
        'RUSBoost': {'n_estimators': [50, 100, 200, 300], 'learning_rate': [0.01, 0.1, 0.5, 1.0], 'sampling_strategy': ['auto', 0.5, 0.75, 1.0], 'estimator': [DecisionTreeClassifier(max_depth=1, class_weight='balanced'), DecisionTreeClassifier(max_depth=2, class_weight='balanced'), DecisionTreeClassifier(max_depth=3, class_weight='balanced')]},
        'XGBoost': {'n_estimators': [100, 200, 300, 500], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 4, 5], 'min_child_weight': [1, 3, 5], 'gamma': [0, 0.1, 0.2], 'subsample': [0.7, 0.8, 0.9], 'colsample_bytree': [0.6, 0.8], 'scale_pos_weight': [1, 3, 5], 'reg_alpha': [0, 0.01, 0.1, 1], 'reg_lambda': [1, 1.5, 2]},
    }

    # Generate a unique cache file per fold to avoid crossing parameters
    PARAMS_FILE = os.path.join(OUTPUT_DIR, f'best_params_loco_{FOLD_TAG}.json')
    cached_params = {}
    fingerprint_now = compute_fingerprint(features_final, PARAM_SPACES)

    if os.path.exists(PARAMS_FILE):
        with open(PARAMS_FILE, 'r') as f:
            cache = json.load(f)
        if cache.get('__fingerprint__', '') == fingerprint_now:
            cached_params = {k: params_from_json(v) for k, v in cache.items() if not k.startswith('__')}
            print(f'\nHyperparameters loaded from cache (fingerprint OK)')
        else:
            print('\n⚠ Fingerprint changed → Cache invalidated, rerunning search.')
    else:
        print('\nCache not found → RandomizedSearchCV will run.')

    print('\n--- STEP 4.4: HYPERPARAMETER OPTIMISATION ---')
    CLASSIFIER_NAMES = ['LR', 'LDA', 'KNN', 'SVM', 'MLP', 'Random Forest', 'Extra Trees',
                        'Random Subspace', 'AdaBoost', 'LogitBoost', 'RUSBoost', 'XGBoost']
    
    optimized_models = {}
    new_params       = {}

    for name in CLASSIFIER_NAMES:
        print(f'\n  [{name}]')
        clf = make_classifier(name)
        
        if name in cached_params:
            params = cached_params[name]
            clf.set_params(**params)
            clf.fit(X_train_sel, y_train)
            best_clf = clf
            new_params[name] = params
            print('    → Loaded cached params.')
        elif name in PARAM_SPACES:
            search = RandomizedSearchCV(clf, PARAM_SPACES[name], n_iter=200, scoring='roc_auc',
                                        cv=5, n_jobs=-1, random_state=0)
            search.fit(X_train_sel, y_train)
            best_clf = search.best_estimator_
            new_params[name] = search.best_params_
            print(f'    → Best params found: {search.best_params_}')
        else:
            clf.fit(X_train_sel, y_train)
            best_clf = clf
            print('    → Trained with defaults.')
        
        optimized_models[name] = best_clf

    # Save cache
    merged_params = {**cached_params, **new_params}
    cache_out     = {'__fingerprint__': fingerprint_now}
    cache_out.update({k: params_to_json(v) for k, v in merged_params.items()})
    with open(PARAMS_FILE, 'w') as f:
        json.dump(cache_out, f, indent=2)

    # ── 4.5 Ranking based on Validation AUC ──────────────────────────────────────
    print('\n--- STEP 4.5: POST-OPTIMISATION RANKING (Validation Set) ---')
    results = []
    for name, clf in optimized_models.items():
        preds_val = clf.predict(X_val_sel)
        probs_val = clf.predict_proba(X_val_sel)[:, 1] if hasattr(clf, 'predict_proba') else None
        results.append(compute_metrics(y_val, preds_val, probs_val, name))

    df_results = pd.DataFrame(results).sort_values('AUC-ROC', ascending=False).reset_index(drop=True)
    
    # Tiebreaker: If AUC difference is < 0.01, rank by Balanced Accuracy instead
    DELTA_TIEBREAK = 0.01
    ordered, pending = [], df_results.to_dict('records')
    while pending:
        current = pending.pop(0)
        if pending and abs(current['AUC-ROC'] - pending[0]['AUC-ROC']) < DELTA_TIEBREAK:
            if pending[0]['Bal. Accuracy'] > current['Bal. Accuracy']:
                ordered.append(pending.pop(0))
                ordered.append(current)
            else: ordered.append(current)
        else: ordered.append(current)
    df_results = pd.DataFrame(ordered).reset_index(drop=True)
    print(df_results.to_string())


    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 5 — TOP-3 SELECTION, CALIBRATION AND VOTING ENSEMBLE
    # ─────────────────────────────────────────────────────────────────────────────
    
    TOP_N = 3
    top_models = df_results.head(TOP_N)
    print(f'\nTop {TOP_N} selected for Voting Ensemble:\n{top_models[["Model", "AUC-ROC"]]}')

    tuned_estimators  = []
    calibrated_models = {}

    # Calibrate individual top models via Platt scaling
    for name in top_models['Model']:
        best_clf = optimized_models[name]
        tuned_estimators.append((name, best_clf))
        cal_model = CalibratedClassifierCV(estimator=best_clf, method='sigmoid', cv='prefit')
        cal_model.fit(X_val_sel, y_val)
        calibrated_models[name] = cal_model

    # Build and calibrate the ensemble
    print('\nGenerating Voting Classifier...')
    voting_clf = VotingClassifier(estimators=tuned_estimators, voting='soft')
    voting_clf.fit(X_train_sel, y_train)
    
    calibrated_voting = CalibratedClassifierCV(estimator=voting_clf, method='sigmoid', cv='prefit')
    calibrated_voting.fit(X_val_sel, y_val)

    # ── Model Selection Logic ────────────────────────────────────────────────────
    probs_val_voting     = calibrated_voting.predict_proba(X_val_sel)[:, 1]
    auc_voting           = roc_auc_score(y_val, probs_val_voting)
    best_individual_name = df_results.iloc[0]['Model']
    probs_val_individual = calibrated_models[best_individual_name].predict_proba(X_val_sel)[:, 1]
    best_individual_auc  = roc_auc_score(y_val, probs_val_individual)

    def bal_acc_at_youden(y_true, probs):
        fpr, tpr, thr = roc_curve(y_true, probs)
        j_idx = np.argmax(tpr - fpr)
        preds = (probs >= thr[j_idx]).astype(int)
        return balanced_accuracy_score(y_true, preds), thr[j_idx]

    bal_acc_voting,     thr_voting     = bal_acc_at_youden(y_val, probs_val_voting)
    bal_acc_individual, thr_individual = bal_acc_at_youden(y_val, probs_val_individual)

    if abs(auc_voting - best_individual_auc) < DELTA_TIEBREAK:
        if bal_acc_voting >= bal_acc_individual:
            final_model, final_name = calibrated_voting, 'Voting Ensemble'
        else:
            final_model, final_name = calibrated_models[best_individual_name], best_individual_name
    elif auc_voting > best_individual_auc:
        final_model, final_name = calibrated_voting, 'Voting Ensemble'
    else:
        final_model, final_name = calibrated_models[best_individual_name], best_individual_name
    print(f'\nFinal model selected: {final_name}')

    # ── Youden threshold optimisation (evaluating on validation set) ─────────────
    probs_val = final_model.predict_proba(X_val_sel)[:, 1]
    fpr_val, tpr_val, thresholds_val = roc_curve(y_val, probs_val)
    youden_idx = np.argmax(tpr_val - fpr_val)
    final_thr  = thresholds_val[youden_idx]
    
    print(f'\nOptimal clinical threshold (J): {final_thr:.4f}')


    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 6 — EXTERNAL TEST (Left-Out Cohort Evaluation)
    # ─────────────────────────────────────────────────────────────────────────────
    print('\n' + '='*70)
    print(f'TEST RESULTS  —  fold: {cohort_test}')
    print('='*70)

    test_results = []
    for test_name, data in data_tests.items():
        X_test_act = data['X_sel']
        y_test_act = data['y']
        y_raw_act  = data['y_raw']
        
        probs_test = final_model.predict_proba(X_test_act)[:, 1]
        preds_ia   = (probs_test >= final_thr).astype(int)
        preds_ht   = y_raw_act.values
        
        m_ia = compute_metrics(y_test_act, preds_ia, probs_test, f'AI-{test_name}')
        m_ht = compute_metrics(y_test_act, preds_ht, None,       f'HT-{test_name}')
        test_results.extend([m_ia, m_ht])
        
        # Accumulate the predictions for the final joint plot outside the loop
        loco_accumulated_preds[test_name] = {
            'y_true':     y_test_act,
            'preds_ia':   preds_ia,
            'preds_ht':   preds_ht,
            'probs_ia':   probs_test,
            'final_name': final_name,
        }

    df_test_summary = pd.DataFrame(test_results)
    print(df_test_summary.to_string(index=False))

    # Save ranking to accumulator for global Excel report
    rank_cols = ['Model','AUC-ROC','Sensitivity','Specificity','Accuracy','PPV','NPV',
                 'Bal. Accuracy','F1-Score','Kappa','MCC']
    rank_cols = [c for c in rank_cols if c in df_results.columns]
    loco_accumulated_ranking[cohort_test] = df_results[rank_cols].copy()

    # Save metrics to global list
    for m in test_results:
        m['Fold'] = cohort_test
    loco_global_test_results.extend(test_results)


    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 7 — SHAP (Calculate and accumulate)
    # ─────────────────────────────────────────────────────────────────────────────
    # SHAP values are computationally expensive, so we calculate and store them 
    # here but plot them collectively after the loop.
    
    print('\n' + '='*70)
    print('STEP 7 — SHAP: Calculating and accumulating values')
    print('='*70)

    def model_prob_surgery(X):
        return final_model.predict_proba(X)[:, 1]

    n_unique   = len(np.unique(X_train_sel, axis=0))
    n_clusters = min(20, n_unique)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Clustering metrics expects discrete values')
        X_background = shap.kmeans(X_train_sel, n_clusters)
        
    explainer = shap.KernelExplainer(model_prob_surgery, X_background)

    print(f'Computing SHAP on Train ({X_train_sel.shape[0]} patients)...')
    shap_train = explainer.shap_values(X_train_sel, nsamples=100)

    X_test_sel = data_tests[cohort_test]['X_sel']
    y_test_sel = data_tests[cohort_test]['y']
    print(f'Computing SHAP on Test — {cohort_test} ({X_test_sel.shape[0]} patients)...')
    shap_test = explainer.shap_values(X_test_sel, nsamples=100)

    # Store for post-loop plotting
    loco_accumulated_shap[cohort_test] = {
        'shap_train':        shap_train,
        'X_train_sel':       X_train_sel,
        'y_train':           y_train,
        'shap_test':         shap_test,
        'X_test_sel':        X_test_sel,
        'y_test':            y_test_sel,
        'features_final':    features_final,
        'binary_features':   binary_features,
        'final_name':        final_name,
    }

# ════════════════════════════════════════════════════════════════════════════
# END OF LOCO LOOP
# ════════════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════════
# POST-LOOP: CONSOLIDATED SHAP FIGURES
# ════════════════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print('GENERATING CONSOLIDATED LOCO SHAP FIGURES')
print('='*70)

COLOURS_LOCO_SHAP = {
    'Valladolid (2023)': '#9467bd',
    'Valladolid (2025)': '#1f77b4',
    'Granada':           '#2ca02c',
    'Salamanca':         '#d62728',
}

folds_shap   = list(loco_accumulated_shap.keys())
n_folds_shap = len(folds_shap)

# ── FIG 1: Consolidated Ranking for all 4 Train Sets (2x2) ───────────────────
x_max_train = max(
    np.abs(loco_accumulated_shap[f]['shap_train']).mean(axis=0).max()
    for f in folds_shap
) * 1.1

fig1, axes1 = plt.subplots(2, 2, figsize=(20, 14))
plt.rcParams['ytick.labelsize'] = 13

for ax, fold_name in zip(axes1.flatten(), folds_shap):
    d          = loco_accumulated_shap[fold_name]
    imp        = np.abs(d['shap_train']).mean(axis=0)
    idx_asc    = np.argsort(imp)
    color_fold = COLOURS_LOCO_SHAP.get(fold_name, '#7f7f7f')
    n_train    = d['X_train_sel'].shape[0]
    
    ax.barh([d['features_final'][i] for i in idx_asc], imp[idx_asc],
            color=color_fold, edgecolor='white', height=0.5)
    ax.set_xlabel('Mean absolute Shapley Values', fontsize=13)
    ax.set_title(f'Train set without {fold_name} (N={n_train})', fontsize=13, fontweight='bold')
    ax.set_xlim(0, x_max_train)
    ax.grid(axis='x', linestyle='--', alpha=0.4)

plt.suptitle('Leave-One-Center-Out Methodology\nSHAP Importance Ranking across all Train sets',
             fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(os.path.join(OUTPUT_DIR, 'shap_ranking_train_loco.png'), dpi=200, bbox_inches='tight')
plt.show()

# ── FIG 2: Consolidated Beeswarm for all 4 Train Sets (4x2) ──────────────────
fig2, axes2 = plt.subplots(n_folds_shap, 2, figsize=(16, 6 * n_folds_shap))
if n_folds_shap == 1:
    axes2 = axes2.reshape(1, 2)

for row, fold_name in enumerate(folds_shap):
    d        = loco_accumulated_shap[fold_name]
    feat_bin = d['binary_features']
    feat_fin = d['features_final']
    shap_tr  = d['shap_train']
    X_tr     = d['X_train_sel']
    y_tr     = d['y_train']
    mask_c0  = (y_tr == 0)
    mask_c1  = (y_tr == 1)
    
    draw_beeswarm(
        axes2[row, 0], shap_tr[mask_c0], X_tr[mask_c0], 
        f'Train (test={fold_name})\nClass 0 (No surgery)', mask_c0.sum(),
        '← Favours "SURGERY" | Favours "NO SURGERY" →', feat_bin, feat_fin, flip_sign=True
    )
    draw_beeswarm(
        axes2[row, 1], shap_tr[mask_c1], X_tr[mask_c1], 
        f'Train (test={fold_name})\nClass 1 (Surgery)', mask_c1.sum(),
        '← Favours "NO SURGERY" | Favours "SURGERY" →', feat_bin, feat_fin, flip_sign=False
    )

plt.suptitle('Leave-One-Center-Out Methodology\nSHAP Beeswarm across all Train sets',
             fontsize=20, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'shap_beeswarm_train_loco.png'), dpi=200, bbox_inches='tight')
plt.show()

# ── FIG 3: Consolidated Ranking for all 4 Test Sets (2x2) ────────────────────
x_max_test = max(
    np.abs(loco_accumulated_shap[f]['shap_test']).mean(axis=0).max()
    for f in folds_shap
) * 1.1

fig3, axes3 = plt.subplots(2, 2, figsize=(20, 14))
for ax, fold_name in zip(axes3.flatten(), folds_shap):
    d          = loco_accumulated_shap[fold_name]
    imp        = np.abs(d['shap_test']).mean(axis=0)
    idx_asc    = np.argsort(imp)
    color_fold = COLOURS_LOCO_SHAP.get(fold_name, '#7f7f7f')
    n_test     = d['X_test_sel'].shape[0]
    
    ax.barh([d['features_final'][i] for i in idx_asc], imp[idx_asc],
            color=color_fold, edgecolor='white', height=0.5)
    ax.set_xlabel('Mean absolute Shapley Values', fontsize=13)
    ax.set_title(f'{fold_name} (N={n_test})', fontsize=13, fontweight='bold')
    ax.set_xlim(0, x_max_test)
    ax.grid(axis='x', linestyle='--', alpha=0.4)

plt.suptitle('Leave-One-Center-Out Methodology\nSHAP Importance Ranking across all Test sets',
             fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(os.path.join(OUTPUT_DIR, 'shap_ranking_test_loco.png'), dpi=200, bbox_inches='tight')
plt.show()

# ── FIG 4: Consolidated Beeswarm for all 4 Test Sets (4x2) ───────────────────
fig4, axes4 = plt.subplots(n_folds_shap, 2, figsize=(16, 6 * n_folds_shap))
if n_folds_shap == 1:
    axes4 = axes4.reshape(1, 2)

for row, fold_name in enumerate(folds_shap):
    d        = loco_accumulated_shap[fold_name]
    feat_bin = d['binary_features']
    feat_fin = d['features_final']
    shap_te  = d['shap_test']
    X_te     = d['X_test_sel']
    y_te     = d['y_test']
    mask_c0  = (y_te == 0)
    mask_c1  = (y_te == 1)
    
    draw_beeswarm(
        axes4[row, 0], shap_te[mask_c0], X_te[mask_c0], 
        f'{fold_name}\nClass 0 (No surgery)', mask_c0.sum(),
        '← Favours "SURGERY" | Favours "NO SURGERY" →', feat_bin, feat_fin, flip_sign=True
    )
    draw_beeswarm(
        axes4[row, 1], shap_te[mask_c1], X_te[mask_c1], 
        f'{fold_name}\nClass 1 (Surgery)', mask_c1.sum(),
        '← Favours "NO SURGERY" | Favours "SURGERY" →', feat_bin, feat_fin, flip_sign=False
    )

plt.suptitle('Leave-One-Center-Out Methodology\nSHAP Beeswarm across all Test sets',
             fontsize=20, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'shap_beeswarm_test_loco.png'), dpi=200, bbox_inches='tight')
plt.show()


# ════════════════════════════════════════════════════════════════════════════
# POST-LOOP: CONSOLIDATED MATRICES AND ROC CURVES
# ════════════════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print('GENERATING CONSOLIDATED LOCO MATRICES AND ROC CURVES')
print('='*70)

folds_order = list(loco_accumulated_preds.keys())
n_folds     = len(folds_order)

# ── Joint 4x2 Confusion Matrices ─────────────────────────────────────────────
fig_loco, axes_loco = plt.subplots(nrows=n_folds, ncols=2, figsize=(12, 5 * n_folds))
if n_folds == 1:
    axes_loco = axes_loco.reshape(1, 2)

for row, fold_name in enumerate(folds_order):
    d = loco_accumulated_preds[fold_name]
    
    ConfusionMatrixDisplay.from_predictions(
        d['y_true'], d['preds_ia'], ax=axes_loco[row, 0], 
        cmap='Blues', colorbar=False, display_labels=['No surgery', 'Surgery']
    )
    axes_loco[row, 0].set_title(f'AI on {fold_name} (N={len(d["y_true"])})', fontsize=10, fontweight='bold')
    
    ConfusionMatrixDisplay.from_predictions(
        d['y_true'], d['preds_ht'], ax=axes_loco[row, 1], 
        cmap='Oranges', colorbar=False, display_labels=['No surgery', 'Surgery']
    )
    axes_loco[row, 1].set_title(f'Heart Team on {fold_name} (N={len(d["y_true"])})', fontsize=10, fontweight='bold')

fig_loco.suptitle('Leave-One-Center-Out Methodology\nMulticentre Comparison: AI vs Heart Team',
                  fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'matrices_comparison_multicentre_loco.png'), dpi=300, bbox_inches='tight')
plt.show()

# ── Joint ROC Curve ──────────────────────────────────────────────────────────
fig_roc_loco, ax_roc_loco = plt.subplots(figsize=(7, 6))
for fold_name in folds_order:
    d = loco_accumulated_preds[fold_name]
    fpr_l, tpr_l, _ = roc_curve(d['y_true'], d['probs_ia'])
    auc_l   = roc_auc_score(d['y_true'], d['probs_ia'])
    color_l = COLOURS_LOCO_SHAP.get(fold_name, '#7f7f7f')
    
    ax_roc_loco.plot(fpr_l, tpr_l, color=color_l, linewidth=2, 
                     label=f'{fold_name} (N={len(d["y_true"])}, AUC={auc_l:.3f})')

ax_roc_loco.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random classifier')
ax_roc_loco.set_xlabel('1 − Specificity', fontsize=11)
ax_roc_loco.set_ylabel('Sensitivity', fontsize=11)
ax_roc_loco.set_title('Leave-One-Center-Out Methodology\nMulticentre ROC curves', fontsize=12, fontweight='bold')
ax_roc_loco.legend(loc='lower right', fontsize=10)
ax_roc_loco.grid(linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curves_loco.png'), dpi=200, bbox_inches='tight')
plt.show()


# ════════════════════════════════════════════════════════════════════════════
# POST-LOOP: EXCEL SUMMARY REPORT
# ════════════════════════════════════════════════════════════════════════════
print('\n' + '█'*75)
print('  CONSOLIDATED SUMMARY — LEAVE ONE COHORT OUT (4 folds)')
print('█'*75)

if loco_global_test_results:
    df_loco_global = pd.DataFrame(loco_global_test_results)
    
    df_ia_global = df_loco_global[df_loco_global['Model'].str.startswith('AI-')].copy()
    df_ht_global = df_loco_global[df_loco_global['Model'].str.startswith('HT-')].copy()

    METRICAS_NUM = ['Sensitivity','Specificity','Accuracy','PPV','NPV',
                    'AUC-ROC','Bal. Accuracy','F1-Score','Kappa','MCC']

    print('\n--- AI: metrics per fold ---')
    print(df_ia_global[['Fold','Model'] + [m for m in METRICAS_NUM if m in df_ia_global.columns]].to_string(index=False))

    print('\n--- Heart Team: metrics per fold ---')
    print(df_ht_global[['Fold','Model'] + [m for m in METRICAS_NUM if m in df_ht_global.columns]].to_string(index=False))

    print('\n--- AI: mean ± std across 4 folds ---')
    for m in METRICAS_NUM:
        if m in df_ia_global.columns:
            mu  = df_ia_global[m].mean()
            std = df_ia_global[m].std()
            print(f'  {m:<18}: {mu:.4f} ± {std:.4f}')

    print('\n--- Heart Team: mean ± std across 4 folds ---')
    for m in METRICAS_NUM:
        if m in df_ht_global.columns:
            mu  = df_ht_global[m].mean()
            std = df_ht_global[m].std()
            print(f'  {m:<18}: {mu:.4f} ± {std:.4f}')

    # ── Exporting to Excel ───────────────────────────────────────────────────
    FOLD_ORDER = ['Valladolid (2023)', 'Valladolid (2025)', 'Granada', 'Salamanca']
    METRICS_ORDER = [
        ('Sensitivity',  'Se'), ('Specificity', 'Sp'), ('Accuracy', 'Acc'),
        ('PPV', 'PPV'), ('NPV', 'NPV'), ('F1-Score', 'F1-Score'),
        ('Bal. Accuracy', 'Bal.Acc'), ('AUC-ROC', 'AUC'), ('Kappa', 'Kappa'), ('MCC', 'MCC'),
    ]
    EXCEL_RESUMEN = os.path.join(OUTPUT_DIR, 'IAvsHT_loco.xlsx')
    
    try:
        with pd.ExcelWriter(EXCEL_RESUMEN, engine='openpyxl', mode='w') as writer:
            SHEET_NAMES = {
                'Valladolid (2023)': 'Valladolid 2023',
                'Valladolid (2025)': 'Valladolid 2025',
                'Granada':           'Granada',
                'Salamanca':         'Salamanca',
            }
            
            # Write ranking sheets per fold
            for fold_name in FOLD_ORDER:
                if fold_name in loco_accumulated_ranking:
                    loco_accumulated_ranking[fold_name].to_excel(
                        writer, sheet_name=SHEET_NAMES[fold_name], index=False
                    )

            # Write the winning model summary across all folds
            winner_rows = []
            winner_cols = ['Metric'] + FOLD_ORDER + ['Mean ± std']
            for key, label in METRICS_ORDER:
                row = {'Metric': label}
                fold_values = []
                for fold_name in FOLD_ORDER:
                    row_ia = df_ia_global[df_ia_global['Fold'] == fold_name]
                    if len(row_ia) > 0 and key in row_ia.columns:
                        val = round(row_ia[key].values[0], 4)
                    else:
                        val = ''
                    row[fold_name] = val
                    if val != '': fold_values.append(val)
                
                if fold_values:
                    mu  = round(pd.Series(fold_values).mean(), 4)
                    std = round(pd.Series(fold_values).std(),  4)
                    row['Mean ± std'] = f'{mu} ± {std}'
                else:
                    row['Mean ± std'] = ''
                winner_rows.append(row)
                
            pd.DataFrame(winner_rows)[winner_cols].to_excel(
                writer, sheet_name='Winner_per_fold', index=False
            )

        print(f'\nLOCO summary exported: {EXCEL_RESUMEN}')
    except Exception as e:
        print(f'ERROR exporting summary: {e}')
        import traceback; traceback.print_exc()
else:
    print('  (No results accumulated — check the LOCO loop)')