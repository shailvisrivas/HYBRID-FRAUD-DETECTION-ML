# FILE 1 OF 3 - SETUP, CV TUNING, FINAL MODEL, HEADLINE RESULTS
# Run files 1, 2, 3 in order in the same Colab session (they share variables).
# ==========================================================
# HYBRID FRAUD DETECTION - 80/20 SPLIT WITH CROSS-VALIDATED TUNING
# Isolation Forest -> CatBoost -> ARDE (Adaptive Risk Decision Engine)
#
# PROTOCOL (state this in the paper):
#   1. Stratified 80/20 split of the labelled data (seed 42). The 20% test
#      set is NOT touched until the very last step.
#   2. Everything that is tuned is tuned by 5-fold CV INSIDE the 80%:
#        - CatBoost number of iterations (early stopping per fold, median used)
#        - classification threshold (max out-of-fold F1)
#        - ARDE LOW / HIGH cut-offs (max out-of-fold F1 / F0.5 of the ARDE score)
#      Inside every fold, encoders, z-score, amount cut-offs, Isolation Forest
#      and its min-max scaling are re-fitted on that fold's training rows only.
#   3. Final models are refitted on all dev rows (650) and scored once on test.
#
#   PART A  helpers (preprocessing, components, threshold search)
#   PART B  core run (CV tuning -> final fit -> test)
#   PART C  reports: headline, ARDE tiers, ablation, sensitivity, learned
#           weights, threshold stability, policy-target trade-off
#   PART D  baselines (LR, RF, CatBoost) on the SAME split and protocol
#   PART E  unseen transactions
#   PART F  optional multi-seed robustness (seeds fixed in advance)
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve
)
from catboost import CatBoostClassifier

# ----------------------------------------------------------
# CONFIG  (fix these BEFORE looking at test results)
# ----------------------------------------------------------
import os
DATA_PATH = next((p for p in ["Dataset/FraudFusion_Dataset.csv",
                              "/content/FraudFusion_Dataset.csv"] if os.path.exists(p)),
                 "Dataset/FraudFusion_Dataset.csv")
SEED = 42                       # headline run
N_FOLDS = 5
N_REPEATS = 2                   # repeated CV for out-of-fold scores
CLF_GRID = np.arange(0.30, 0.71, 0.01)
RUN_EXTRAS = True               # PART C extras
RUN_BASELINES = True            # PART D
N_SENSITIVITY_SAMPLES = 200
N_STABILITY_BOOT = 200
N_TEST_BOOT = 1000
MULTI_SEEDS = [7, 123, 2024, 99]   # fixed in advance; headline run is seed 42; never pick seeds by test score
CV_RECALL_TARGET = 0.95         # policy-target trade-off table only
CV_PRECISION_TARGET = 0.90

CAT = ["Merchant", "Device", "Location", "Payment_Mode"]
BASE = CAT + ["Amount", "Log_Amount", "Amount_ZScore", "Hour", "Night", "Weekend"]
FULL = BASE + ["Anomaly_Score"]
COMPONENTS = ["Fraud_Probability", "Anomaly_Score", "Amount_Risk", "Time_Risk"]

# Domain-informed design prior, NOT learned.
ARDE_WEIGHTS = {"Fraud_Probability": 0.55, "Anomaly_Score": 0.25,
                "Amount_Risk": 0.10, "Time_Risk": 0.10}

pd.set_option("display.width", 220)


# ==========================================================
# PART A - HELPERS
# ==========================================================

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df = df.dropna(how="all")
    df = df.dropna(subset=["Amount", "Merchant", "Device", "Location",
                           "Payment_Mode", "Transaction_Timestamp"])
    df["Transaction_Timestamp"] = pd.to_datetime(
        df["Transaction_Timestamp"], format="mixed", dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Transaction_Timestamp"])
    df["Hour"] = df["Transaction_Timestamp"].dt.hour
    df["Weekend"] = (df["Transaction_Timestamp"].dt.dayofweek >= 5).astype(int)
    df["Night"] = ((df["Hour"] >= 22) | (df["Hour"] <= 5)).astype(int)   # 22:00-05:59
    df["Log_Amount"] = np.log1p(df["Amount"])
    return df[df["Fraud"].notna()].copy(), df[df["Fraud"].isna()].copy()


def fit_prep(train_raw):
    """Everything learned from the training rows only."""
    st = {"classes": {c: sorted(train_raw[c].astype(str).unique()) for c in CAT},
          "mu": float(train_raw["Amount"].mean())}
    sd = float(train_raw["Amount"].std())
    st["sd"] = sd if sd > 0 else 1.0
    q1, q3 = train_raw["Amount"].quantile([0.25, 0.75])
    st["cut"] = {"high": float(q3 + 1.5 * (q3 - q1)),
                 "med": float(train_raw["Amount"].median())}
    return st


def apply_prep(raw, st):
    out = raw.copy()
    for c in CAT:
        mapping = {k: i for i, k in enumerate(st["classes"][c])}
        out[c] = out[c].astype(str).map(mapping).fillna(len(mapping)).astype(int)
    out["Amount_ZScore"] = (out["Amount"] - st["mu"]) / st["sd"]
    return out[BASE].copy()


def fit_iso(Xb, seed):
    iso = IsolationForest(n_estimators=300, contamination=0.05,
                          random_state=seed).fit(Xb)
    raw = -iso.decision_function(Xb)
    return {"iso": iso, "lo": raw.min(), "den": max(raw.max() - raw.min(), 1e-8)}


def iso_score(ist, Xb):
    return np.clip((-ist["iso"].decision_function(Xb) - ist["lo"]) / ist["den"], 0, 1)


def fold_data(dev, tr, va, seed):
    """Prep + Isolation Forest fitted on dev.iloc[tr] only."""
    a, b = dev.iloc[tr], dev.iloc[va]
    st = fit_prep(a)
    Xa, Xb = apply_prep(a, st), apply_prep(b, st)
    ist = fit_iso(Xa, seed)
    Xa["Anomaly_Score"] = iso_score(ist, Xa)
    Xb["Anomaly_Score"] = iso_score(ist, Xb)
    return Xa, Xb, st


def make_components(p, anom, amount, hour, cut, index=None):
    amount, hour = np.asarray(amount, float), np.asarray(hour, int)
    return pd.DataFrame({
        "Fraud_Probability": np.asarray(p, float),
        "Anomaly_Score": np.asarray(anom, float),
        "Amount_Risk": np.where(amount >= cut["high"], 1.0,
                       np.where(amount >= cut["med"], 0.60, 0.20)),
        "Time_Risk": np.where((hour >= 22) | (hour <= 5), 1.0, 0.0),
    }, index=index)


def make_cb(n_iter, seed):
    return CatBoostClassifier(
        iterations=n_iter, learning_rate=0.03, depth=7, l2_leaf_reg=5,
        random_strength=1, bagging_temperature=1, loss_function="Logloss",
        eval_metric="AUC", auto_class_weights="Balanced",
        random_seed=seed, verbose=False)


def tune_iterations(dev, y, seed, cols):
    """CV early stopping per fold; median best iteration is used afterwards."""
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
    best = []
    for tr, va in skf.split(dev, y):
        Xa, Xb, _ = fold_data(dev, tr, va, seed)
        m = make_cb(600, seed)
        m.fit(Xa[cols], y[tr], eval_set=(Xb[cols], y[va]),
              use_best_model=True, early_stopping_rounds=50)
        best.append(int(m.best_iteration_) + 1)
    return int(np.median(best)), best


def cb_fp(n_iter, seed, cols):
    def f(Xa, ya, Xb):
        return make_cb(n_iter, seed).fit(Xa[cols], ya).predict_proba(Xb[cols])[:, 1]
    return f


def sk_fp(factory, cols):
    def f(Xa, ya, Xb):
        return factory().fit(Xa[cols], ya).predict_proba(Xb[cols])[:, 1]
    return f


def oof_predictions(dev, y, seed, fit_predict, repeats=N_REPEATS):
    """Out-of-fold ARDE components for every dev row (averaged over repeats).
    Fixed model settings, no early stopping on held-out rows."""
    n = len(dev)
    acc = {c: np.zeros(n) for c in COMPONENTS}
    for r in range(repeats):
        skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed + r)
        for tr, va in skf.split(dev, y):
            Xa, Xb, st = fold_data(dev, tr, va, seed)
            comp = make_components(fit_predict(Xa, y[tr], Xb),
                                   Xb["Anomaly_Score"], Xb["Amount"], Xb["Hour"],
                                   st["cut"])
            for c in COMPONENTS:
                acc[c][va] += comp[c].to_numpy() / repeats
    return pd.DataFrame(acc)


def pick_clf_threshold(y, p):
    best_f, best_t = -1.0, 0.5
    for t in CLF_GRID:
        f = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f > best_f + 1e-12:
            best_f, best_t = f, float(t)
    return best_t, best_f


def calculate_arde_scores(comp, weights):
    w = np.array([weights[c] for c in comp.columns])
    return np.clip(comp.to_numpy(float) @ w, 0.0, 1.0)


def _renorm(cols):
    tot = sum(ARDE_WEIGHTS[c] for c in cols)
    return {c: ARDE_WEIGHTS[c] / tot for c in cols}


def _sweep(s, y, cand):
    """Precision, recall, F1, F0.5 of (score >= t) for every t in cand."""
    pred = s[None, :] >= cand[:, None]
    pos = (y == 1)
    tp = (pred & pos[None, :]).sum(1).astype(float)
    fp = (pred & ~pos[None, :]).sum(1).astype(float)
    fn = pos.sum() - tp

    def div(a, b):
        return np.divide(a, b, out=np.zeros_like(a), where=b > 0)
    return (div(tp, tp + fp), tp / max(pos.sum(), 1),
            div(2 * tp, 2 * tp + fp + fn),
            div(1.25 * tp, 1.25 * tp + 0.25 * fn + fp))


def optimise_arde_thresholds(comp, y, weights):
    """LOW = argmax F1 of (score >= t); HIGH = argmax F0.5 with t > LOW."""
    y = np.asarray(y).astype(int)
    s = calculate_arde_scores(comp, weights)
    cand = np.unique(np.concatenate(([0.0], s, [1.0])))
    prec, rec, f1, f05 = _sweep(s, y, cand)
    i = np.lexsort((-cand, rec, f1))[-1]          # ties: higher recall, lower t
    low = float(cand[i])
    hi_idx = np.where(cand > low)[0]
    if len(hi_idx) == 0:
        return {"low": low, "high": low, "elev_f1": float(f1[i]), "high_f05": 0.0}
    j = hi_idx[np.lexsort((cand[hi_idx], rec[hi_idx], prec[hi_idx], f05[hi_idx]))[-1]]
    return {"low": low, "high": float(cand[j]),
            "elev_f1": float(f1[i]), "high_f05": float(f05[j])}


def target_thresholds(s, y, recall_target, precision_target):
    """LOW = highest cut-off with recall >= target; HIGH = lowest cut-off above
    LOW with precision >= target (fallback max F0.5). Midpoints between scores."""
    s = np.asarray(s, float)
    y = np.asarray(y).astype(int)
    cand = np.unique(s)
    prec, rec, _, f05 = _sweep(s, y, cand)

    def mid(i):
        return float((cand[i - 1] + cand[i]) / 2) if i > 0 else float(cand[0])
    i_low = np.where(rec >= recall_target)[0].max()
    ok = np.where((np.arange(len(cand)) > i_low) & (prec >= precision_target))[0]
    if len(ok):
        high = mid(ok.min())
    else:
        idx = np.arange(i_low + 1, len(cand))
        high = mid(idx[np.argmax(f05[idx])]) if len(idx) else mid(i_low)
    return mid(i_low), max(high, mid(i_low))


def assign_tiers(s, low, high):
    return np.where(s >= high, "HIGH", np.where(s >= low, "MODERATE", "LOW"))


def row_metrics(y, s, low, high):
    e, h = (s >= low).astype(int), (s >= high).astype(int)
    return {"ROC-AUC": roc_auc_score(y, s), "PR-AUC": average_precision_score(y, s),
            "Elev P": precision_score(y, e, zero_division=0),
            "Elev R": recall_score(y, e, zero_division=0),
            "Elev F1": f1_score(y, e, zero_division=0),
            "HIGH P": precision_score(y, h, zero_division=0),
            "HIGH R": recall_score(y, h, zero_division=0),
            "HIGH F0.5": fbeta_score(y, h, beta=0.5, zero_division=0),
            "Elev flagged": float(e.mean()), "HIGH flagged": float(h.mean())}


def boot_ci(y, s, fn, n_boot=N_TEST_BOOT, seed=SEED):
    """Stratified bootstrap 95% CI on test metrics; thresholds held fixed."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    draws = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        draws.append(fn(y[idx], s[idx]))
    d = pd.DataFrame(draws)
    return d.quantile(0.025), d.quantile(0.975)


def clf_metrics(y, p, t):
    pr = (p >= t).astype(int)
    return {"Accuracy": accuracy_score(y, pr),
            "Precision": precision_score(y, pr, zero_division=0),
            "Recall": recall_score(y, pr, zero_division=0),
            "F1": f1_score(y, pr, zero_division=0),
            "ROC-AUC": roc_auc_score(y, p),
            "PR-AUC": average_precision_score(y, p)}


# ==========================================================
# PART B - CORE RUN
# ==========================================================

LAB, UNSEEN = load_data()
print("Labelled rows:", len(LAB), "| unseen rows:", len(UNSEEN))


def run_core(seed):
    y_all = LAB["Fraud"].astype(int)
    dev, test = train_test_split(LAB, test_size=0.20, stratify=y_all, random_state=seed)
    dev, test = dev.reset_index(drop=True), test.reset_index(drop=True)
    y_dev = dev["Fraud"].astype(int).to_numpy()
    y_test = test["Fraud"].astype(int).to_numpy()

    # ---- tuning inside dev only ----
    n_iter, fold_iters = tune_iterations(dev, y_dev, seed, FULL)
    oof = oof_predictions(dev, y_dev, seed, cb_fp(n_iter, seed, FULL))
    clf_thr, clf_oof_f1 = pick_clf_threshold(y_dev, oof["Fraud_Probability"].to_numpy())
    arde_t = optimise_arde_thresholds(oof, y_dev, ARDE_WEIGHTS)

    # ---- final fit on all dev rows, single look at test ----
    st = fit_prep(dev)
    Xd, Xt = apply_prep(dev, st), apply_prep(test, st)
    ist = fit_iso(Xd, seed)
    Xd["Anomaly_Score"], Xt["Anomaly_Score"] = iso_score(ist, Xd), iso_score(ist, Xt)
    model = make_cb(n_iter, seed).fit(Xd[FULL], y_dev)
    p_test = model.predict_proba(Xt[FULL])[:, 1]
    comp_test = make_components(p_test, Xt["Anomaly_Score"], Xt["Amount"],
                                Xt["Hour"], st["cut"])
    return dict(seed=seed, dev=dev, test=test, y_dev=y_dev, y_test=y_test,
                n_iter=n_iter, fold_iters=fold_iters, oof=oof, clf_thr=clf_thr,
                clf_oof_f1=clf_oof_f1, arde_t=arde_t, st=st, ist=ist, Xd=Xd, Xt=Xt,
                model=model, p_test=p_test, comp_test=comp_test)


def headline(R):
    c = clf_metrics(R["y_test"], R["p_test"], R["clf_thr"])
    s = calculate_arde_scores(R["comp_test"], ARDE_WEIGHTS)
    a = row_metrics(R["y_test"], s, R["arde_t"]["low"], R["arde_t"]["high"])
    return c, a


R = run_core(SEED)
y_dev, y_test = R["y_dev"], R["y_test"]
print(f"\nDev rows {len(y_dev)} ({y_dev.sum()} frauds) | test rows {len(y_test)} "
      f"({y_test.sum()} frauds)")
print(f"CatBoost iterations per fold {R['fold_iters']} -> using {R['n_iter']}")
print(f"Classification threshold (OOF F1 {R['clf_oof_f1']:.4f}): {R['clf_thr']:.2f}")
print(f"ARDE cut-offs from OOF: LOW {R['arde_t']['low']:.4f} | HIGH {R['arde_t']['high']:.4f}")
print(f"Amount cut-offs (dev): median Rs {R['st']['cut']['med']:.2f}, "
      f"high Rs {R['st']['cut']['high']:.2f} | time rule 22:00-05:59")

clf_res, arde_res = headline(R)
pred = (R["p_test"] >= R["clf_thr"]).astype(int)
print("\n===== FINAL TEST PERFORMANCE: CatBoost P_f at CV threshold =====")
for k, v in clf_res.items():
    print(f"{k:10s}: {v * 100:.2f}%" if k == "Accuracy" else f"{k:10s}: {v:.4f}")
print("\n", classification_report(y_test, pred, zero_division=0))
cm = confusion_matrix(y_test, pred)
print("Confusion matrix\n", cm)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["GENUINE", "FRAUD"], yticklabels=["GENUINE", "FRAUD"])
plt.xlabel("Predicted Class"); plt.ylabel("Actual Class"); plt.title("Confusion Matrix")
plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=300); plt.show()

fpr, tpr, _ = roc_curve(y_test, R["p_test"])
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, lw=2, label=f"AUC = {clf_res['ROC-AUC']:.3f}")
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve"); plt.legend(); plt.grid(True)
plt.tight_layout(); plt.savefig("roc_curve.png", dpi=300); plt.show()

fi = pd.DataFrame({"Feature": FULL, "Importance": R["model"].get_feature_importance()}
                  ).sort_values("Importance", ascending=False)
print("\nFeature importance\n", fi.to_string(index=False))


