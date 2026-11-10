# ==========================================================
# FILE 3 OF 3 - BASELINES, UNSEEN TRANSACTIONS, MULTI-SEED
# Run AFTER files 1 and 2 in the same session.
# ==========================================================
# ==========================================================
# PART D - BASELINES ON THE SAME SPLIT AND PROTOCOL
# ==========================================================

def make_lr():
    num = [c for c in BASE if c not in CAT]
    ct = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
                            ("num", StandardScaler(), num)])
    return make_pipeline(ct, LogisticRegression(C=1.0, max_iter=2000,
                                                class_weight="balanced"))


def make_rf():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                  class_weight="balanced", random_state=SEED, n_jobs=-1)


def evaluate_baseline(name, fit_predict):
    dev, test = R["dev"], R["test"]
    oof_p = oof_predictions(dev, y_dev, SEED, fit_predict)["Fraud_Probability"].to_numpy()
    thr, _ = pick_clf_threshold(y_dev, oof_p)
    p = fit_predict(R["Xd"], y_dev, R["Xt"])
    m = clf_metrics(y_test, p, thr)
    lo, hi = boot_ci(y_test, p, lambda y, s: clf_metrics(y, s, thr))
    return {"Model": name, "Threshold": thr, **m,
            "F1 95% CI": f"[{lo['F1']:.3f}, {hi['F1']:.3f}]",
            "AUC 95% CI": f"[{lo['ROC-AUC']:.3f}, {hi['ROC-AUC']:.3f}]"}


class LSTMClassifier:
    """Small LSTM for tabular rows (each row = a sequence of length 1).
    Same one-hot + scaling as Logistic Regression; fixed 60 epochs, balanced
    class weights, no early stopping on held-out data."""
    def __init__(self, seed=SEED, epochs=60, units=32):
        self.seed, self.epochs, self.units = seed, epochs, units

    def fit(self, X, y):
        import tensorflow as tf
        tf.keras.utils.set_random_seed(self.seed)
        num = [c for c in X.columns if c not in CAT]
        self.ct = ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
             ("num", StandardScaler(), num)], sparse_threshold=0)
        Z = self.ct.fit_transform(X).astype("float32")[:, None, :]
        y = np.asarray(y).astype(int)
        cw = {0: len(y) / (2 * (y == 0).sum()), 1: len(y) / (2 * (y == 1).sum())}
        self.m = tf.keras.Sequential([
            tf.keras.layers.Input(shape=Z.shape[1:]),
            tf.keras.layers.LSTM(self.units),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation="sigmoid")])
        self.m.compile(optimizer="adam", loss="binary_crossentropy")
        self.m.fit(Z, y, epochs=self.epochs, batch_size=32, class_weight=cw, verbose=0)
        return self

    def predict_proba(self, X):
        Z = self.ct.transform(X).astype("float32")[:, None, :]
        p = self.m.predict(Z, verbose=0).ravel()
        return np.column_stack([1 - p, p])


if RUN_BASELINES:
    # ---- TABLE 1: comparison with other algorithms (paper's comparison table)
    comp_rows = []
    for name, fp in [("Logistic Regression", sk_fp(make_lr, BASE)),
                     ("Random Forest", sk_fp(make_rf, BASE)),
                     ("LSTM", sk_fp(LSTMClassifier, BASE))]:
        try:
            comp_rows.append(evaluate_baseline(name, fp))
        except ImportError:
            print(f"{name} skipped: TensorFlow not installed.")
    comp_rows.append(evaluate_baseline("Hybrid (CatBoost + Isolation Forest, P_f)",
                                       cb_fp(R["n_iter"], SEED, FULL)))
    print("\n===== COMPARISON TABLE (test; same split, CV-tuned threshold) =====")
    print(pd.DataFrame(comp_rows).round(4).to_string(index=False))

    # ---- TABLE 2: ablation of the anomaly feature (own components) ----------
    n_it_nb, _ = tune_iterations(R["dev"], y_dev, SEED, BASE)
    abl_rows = [
        evaluate_baseline("CatBoost without anomaly feature", cb_fp(n_it_nb, SEED, BASE)),
        evaluate_baseline("CatBoost + Isolation Forest (hybrid)", cb_fp(R["n_iter"], SEED, FULL)),
    ]
    print("\n===== ABLATION: DOES THE ANOMALY FEATURE HELP? (test) =====")
    print(pd.DataFrame(abl_rows).round(4).to_string(index=False))
    print("ARDE is a tier system on top of P_f, not a new classifier; its rows are "
          "in the ablation table of file 2.")


# ==========================================================
# PART E - UNSEEN TRANSACTIONS (actions chosen by risk tier only)
# ==========================================================

ACTIONS = {
    "HIGH": "Require additional authentication or block, per the payment system's policy.",
    "MODERATE": "Require secondary authentication before completing the transaction.",
    "LOW": "Approve. No significant combined risk indicators.",
}

print("\n")
if UNSEEN.empty:
    print("No unseen transactions available.")
else:
    Xu = apply_prep(UNSEEN, R["st"])
    Xu["Anomaly_Score"] = iso_score(R["ist"], Xu)
    pu = R["model"].predict_proba(Xu[FULL])[:, 1]
    cu = make_components(pu, Xu["Anomaly_Score"], Xu["Amount"], Xu["Hour"],
                         R["st"]["cut"], index=UNSEEN.index)
    su = calculate_arde_scores(cu, ARDE_WEIGHTS)
    tiers = assign_tiers(su, LOW_T, HIGH_T)
    for k, (idx, row) in enumerate(UNSEEN.iterrows(), 1):
        i = k - 1
        reasons = [f"CatBoost fraud probability {pu[i] * 100:.2f}%.",
                   f"Amount Rs {row['Amount']:,.2f} -> amount risk {cu['Amount_Risk'].iloc[i]:.2f}.",
                   "Night-time window (22:00-05:59)." if cu["Time_Risk"].iloc[i] else
                   "Outside the night-risk window.",
                   f"Anomaly score {cu['Anomaly_Score'].iloc[i]:.2f}."]
        print("=" * 70)
        print(f"TRANSACTION {k}/{len(UNSEEN)}  ID: {row.get('Transaction_ID', idx)}")
        print(f"Risk score {su[i] * 100:.2f}% | Tier {tiers[i]}")
        for r_ in reasons:
            print(" -", r_)
        print("Action:", "\n".join(textwrap.wrap(ACTIONS[tiers[i]], 66)))


# ==========================================================
# PART F - OPTIONAL MULTI-SEED ROBUSTNESS (headline stays seed 42)
# ==========================================================

if MULTI_SEEDS:
    recs = []
    for sd_ in MULTI_SEEDS:
        c_, a_ = headline(run_core(sd_))
        recs.append({"seed": sd_, "Accuracy": c_["Accuracy"], "Precision": c_["Precision"],
                     "Recall": c_["Recall"], "F1": c_["F1"], "ROC-AUC": c_["ROC-AUC"],
                     "ARDE Elev F1": a_["Elev F1"], "ARDE HIGH F0.5": a_["HIGH F0.5"]})
    d = pd.DataFrame(recs)
    print("\n===== MULTI-SEED (different splits; each fully CV-tuned) =====")
    print(d.round(4).to_string(index=False))
    print(d.drop(columns="seed").agg(["mean", "std"]).T.round(4))

print("\nAnalysis completed.")
