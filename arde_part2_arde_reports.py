# ==========================================================
# FILE 2 OF 3 - ARDE REPORTS
# Run AFTER file 1 in the same session (uses R, y_dev, y_test, COMPONENTS ...).
# Contains: tier table, ablation with CIs, weight sensitivity, learned
# weights, threshold stability, policy-target trade-off.
# ==========================================================
# ==========================================================
# PART C - ARDE REPORTS
# ==========================================================

LOW_T, HIGH_T = R["arde_t"]["low"], R["arde_t"]["high"]
test_scores = calculate_arde_scores(R["comp_test"], ARDE_WEIGHTS)
test_tiers = assign_tiers(test_scores, LOW_T, HIGH_T)


def tier_table(comp, y, low, high, weights=ARDE_WEIGHTS):
    s = calculate_arde_scores(comp, weights)
    d = pd.DataFrame({"Tier": assign_tiers(s, low, high), "y": y})
    out = d.groupby("Tier")["y"].agg(["count", "sum", "mean"])
    out.columns = ["Transactions", "Frauds", "Fraud rate"]
    out["Share of all"] = out["Transactions"] / len(d)
    return out.reindex(["LOW", "MODERATE", "HIGH"]).fillna(0)


print("\n===== ARDE TIER TABLE (test) =====")
print(tier_table(R["comp_test"], y_test, LOW_T, HIGH_T).round(4))
print("\nARDE test metrics:", {k: round(v, 4) for k, v in arde_res.items()})
print("(Compare with P_f alone in the ablation table below.)")


def evaluate_config(name, cols, weights, fixed=None, ci=True):
    s_test = calculate_arde_scores(R["comp_test"][cols], weights)
    if fixed is None:
        t = optimise_arde_thresholds(R["oof"][cols], y_dev, weights)
        low, high = t["low"], t["high"]
    else:
        low, high = fixed
    row = {"Configuration": name, "t_low": low, "t_high": high,
           **row_metrics(y_test, s_test, low, high)}
    if ci:
        lo, hi = boot_ci(y_test, s_test, lambda y, s: row_metrics(y, s, low, high))
        row["Elev F1 95% CI"] = f"[{lo['Elev F1']:.3f}, {hi['Elev F1']:.3f}]"
        row["HIGH F0.5 95% CI"] = f"[{lo['HIGH F0.5']:.3f}, {hi['HIGH F0.5']:.3f}]"
    return row


if RUN_EXTRAS:
    rows = [evaluate_config("Full ARDE", COMPONENTS, ARDE_WEIGHTS)]
    for c in COMPONENTS:
        rest = [x for x in COMPONENTS if x != c]
        rows.append(evaluate_config(f"Without {c}", rest, _renorm(rest)))
    rows.append(evaluate_config("P_f only (no ARDE)", ["Fraud_Probability"],
                                {"Fraud_Probability": 1.0}))
    rows.append(evaluate_config("Full ARDE, fixed 0.40/0.70", COMPONENTS,
                                ARDE_WEIGHTS, fixed=(0.40, 0.70)))
    print("\n===== ABLATION (test; cut-offs tuned on out-of-fold dev scores) =====")
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    # ---- Dirichlet weight sensitivity ----
    rng = np.random.default_rng(SEED)
    base = np.array([ARDE_WEIGHTS[c] for c in COMPONENTS])
    res = []
    for _ in range(N_SENSITIVITY_SAMPLES):
        wd = dict(zip(COMPONENTS, rng.dirichlet(50 * base)))
        t = optimise_arde_thresholds(R["oof"], y_dev, wd)
        s = calculate_arde_scores(R["comp_test"], wd)
        res.append(row_metrics(y_test, s, t["low"], t["high"]))
    print("\n===== WEIGHT SENSITIVITY (Dirichlet around 0.55/0.25/0.10/0.10) =====")
    print(pd.DataFrame(res)[["ROC-AUC", "PR-AUC", "Elev F1", "HIGH F0.5"]]
          .agg(["mean", "std", "min", "max"]).T.round(4))

    # ---- learned weights ----
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(R["oof"][COMPONENTS], y_dev)
    coef = np.clip(lr.coef_[0], 0, None)
    print("\n===== LEARNED (logistic regression on OOF dev) vs HAND-SET WEIGHTS =====")
    print(pd.DataFrame({"ARDE weight": base, "LogReg coef": lr.coef_[0],
                        "LogReg relative weight": coef / max(coef.sum(), 1e-12)},
                       index=COMPONENTS).round(4))

    # ---- threshold stability ----
    rng = np.random.default_rng(SEED)
    out = []
    for _ in range(N_STABILITY_BOOT):
        idx = rng.integers(0, len(y_dev), len(y_dev))
        if y_dev[idx].sum() == 0:
            continue
        t = optimise_arde_thresholds(R["oof"].iloc[idx], y_dev[idx], ARDE_WEIGHTS)
        out.append((t["low"], t["high"]))
    print("\n===== THRESHOLD STABILITY (bootstrap of dev OOF scores) =====")
    print(pd.DataFrame(out, columns=["t_low", "t_high"])
          .agg(["mean", "std", "min", "max"]).T.round(4))

    # ---- policy-target trade-off (NOT the main design) ----
    s_oof = calculate_arde_scores(R["oof"], ARDE_WEIGHTS)
    pol = []
    for rt in (0.90, 0.95, 0.98):
        for pt in (0.85, 0.90, 0.95):
            lo_, hi_ = target_thresholds(s_oof, y_dev, rt, pt)
            m = row_metrics(y_test, test_scores, lo_, hi_)
            pol.append({"recall_target": rt, "precision_target": pt, "t_low": lo_,
                        "t_high": hi_, "Elev R": m["Elev R"], "Elev P": m["Elev P"],
                        "Elev flagged": m["Elev flagged"], "HIGH P": m["HIGH P"],
                        "HIGH R": m["HIGH R"]})
    print("\n===== POLICY-TARGET TRADE-OFF (test) =====")
    print(pd.DataFrame(pol).round(4).to_string(index=False))


