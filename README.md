# A Hybrid Framework for Digital Payment Fraud Detection

Isolation Forest + CatBoost + an Adaptive Risk Decision Engine (ARDE) for digital payment fraud detection and risk-tier decision support.

## Abstract

The framework combines unsupervised anomaly detection (Isolation Forest), supervised fraud classification (CatBoost, with the anomaly score as an extra feature) and ARDE, which combines the fraud probability, anomaly score, transaction amount and time into a risk score, a LOW / MODERATE / HIGH tier, and a recommended action.

## Important correction

An earlier version of this repository (and of the paper) tuned CatBoost early stopping and the classification threshold on the test set. Those results (97.55% accuracy) were optimistic and are **not valid**; they have been removed. The code and results here use the corrected protocol below.

## Evaluation protocol

* Stratified 80/20 split of the 813 labelled transactions (seed 42): 650 development rows (96 frauds) and 163 test rows (24 frauds).
* All tuning uses 5-fold cross-validation (2 repeats) inside the development set: CatBoost iterations, the classification threshold (range 0.30 to 0.70, maximum F1) and the ARDE cut-offs (LOW = maximum F1, HIGH = maximum F0.5, on out-of-fold scores).
* Encoders, amount statistics, amount cut-offs, the Isolation Forest and its scaling are fitted on training folds only.
* Final models are trained on all development rows and scored once on the test set.
* Four unlabelled transactions are used only for the demonstration in `arde_part3_baselines_unseen.py`.

## Code (run in order, same session)

| File | Purpose |
| --- | --- |
| `arde_part1_core.py` | Setup, CV tuning, final model, headline test results, figures |
| `arde_part2_arde_reports.py` | ARDE tiers, ablation with bootstrap CIs, weight sensitivity, learned weights, threshold stability, policy-target trade-off |
| `arde_part3_baselines_unseen.py` | Baselines (Logistic Regression, Random Forest, LSTM), anomaly-feature comparison, unseen transactions, multi-seed check |

The dataset is read from `Dataset/FraudFusion_Dataset.csv` (or `/content/FraudFusion_Dataset.csv` in Colab). Install the packages in `requirements.txt` (TensorFlow is needed only for the LSTM baseline). `MULTI_SEEDS` in part 1 holds the extra random splits (7, 123, 2024, 99) that were fixed in advance; the headline run is seed 42.

## Results (seed 42, test set)

| Metric | CatBoost + Isolation Forest (threshold 0.39) |
| --- | ---: |
| Accuracy | 96.32% |
| Precision | 0.821 |
| Recall | 0.958 |
| F1-score | 0.885 |
| ROC-AUC | 0.990 |

Confusion matrix: 134 genuine and 23 fraudulent transactions correct, 5 false alarms, 1 missed fraud (`results/confusion_matrix.png`).

Comparison on the same split and protocol: Logistic Regression F1 0.700, LSTM F1 0.727, Random Forest F1 0.842, hybrid F1 0.885 (see the paper for all metrics and intervals).

**Read these numbers with care.** The test set has only 24 frauds, so each missed fraud changes recall by about 4 points. Seed 42 is the most favourable of five random splits: over five splits the hybrid reached mean accuracy 93.0% and mean F1 0.77 (SD 0.09).

## ARDE findings

* On the test set the HIGH tier holds 24 transactions, 20 of them fraudulent (83.3%); the LOW tier holds 137 with 2 frauds. The MODERATE tier is very small (2 transactions).
* ARDE does **not** improve detection over the CatBoost fraud probability alone (alert F1 0.880 vs 0.902, overlapping intervals). It is a transparent decision-support layer that adds graded tiers, explanations and recommended actions.
* The ARDE weights (0.55, 0.25, 0.10, 0.10) are fixed design choices, not learned. The tier cut-offs (about 0.43 and 0.54) are chosen from out-of-fold scores.
* The Isolation Forest feature shows no measurable benefit on this dataset (CatBoost without it scores slightly higher on the test split, with overlapping intervals).

## Technologies

Python, pandas, NumPy, scikit-learn, CatBoost, TensorFlow (LSTM baseline), Matplotlib, Seaborn.

## Limitations

* The dataset is synthetic; Merchant is the most important feature, which suggests the generator ties fraud to merchants. Real fraud also shows adversarial drift and temporal velocity patterns that a synthetic generator cannot reproduce.
* Small test set (24 frauds); results depend on the split.
* The cut-offs are noisy (bootstrap SD about 0.04 for LOW and 0.06 for HIGH).
* ARDE's contextual terms overlap with CatBoost inputs (night indicator, amount) and add transparency rather than new evidence. The dataset has no user-history, device-change or link-click features, so the context-aware benefit of ARDE cannot be measured here.

## Future work

* Evaluation on real transaction data with user-context features (device change, payee novelty, velocity)
* Monitoring for drift and periodic retraining
* Integration with real-time payment systems
