# FraudGuard: A Hybrid Framework for Digital Payment Fraud Detection

## Abstract

FraudFusion++ is a machine learning-based fraud detection framework designed for digital payment systems. The framework combines unsupervised anomaly detection using Isolation Forest, supervised fraud classification using CatBoost, and an Adaptive Risk Decision Engine (ARDE) for intelligent transaction risk assessment.

## Research Problem

Traditional fraud detection systems often struggle with:
- New fraud patterns
- Highly imbalanced transaction data
- False positive predictions

This research proposes a hybrid approach to improve fraud detection capability.

## Proposed Framework

FraudFusion++ consists of three major components:

1. Isolation Forest
   - Detects unusual transaction behaviour
   - Generates anomaly scores

2. CatBoost Classifier
   - Learns fraud patterns from labelled transactions
   - Classifies transactions as fraud/genuine

3. Adaptive Risk Decision Engine
   - Combines model output and transaction behaviour
   - Generates final risk decisions

## Dataset Features

| Feature | Description |
|-|-|
| Amount | Transaction value |
| Merchant | Transaction receiver |
| Device | Device information |
| Location | Transaction location |
| Payment Mode | UPI/Card/etc |
| Transaction Time | Transaction timestamp |
| Fraud | Target label |

## Results

Accuracy: 97.86%

Evaluation Metrics:
- Precision
- Recall
- F1-score
- ROC-AUC

## Technologies Used

Python  
Pandas  
NumPy  
Scikit-learn  
CatBoost  
Matplotlib  

## Research Contribution

- Hybrid anomaly detection and classification framework
- Risk-based decision mechanism
- Behaviour-based fraud analysis
