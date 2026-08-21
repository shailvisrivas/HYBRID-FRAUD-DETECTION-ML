A Hybrid Framework for Digital Payment Fraud Detection

## Abstract

A Hybrid Fraud Detection framework is designed for digital payment systems. The framework combines unsupervised anomaly detection using Isolation Forest, supervised fraud classification using CatBoost, and an Adaptive Risk Decision Engine (ARDE) for intelligent transaction risk assessment.

## Research Problem

Existing studies have explored fraud classification, anomaly detection, hybrid learning, and risk management, but these capabilities are often considered separately. Simply predicting whether a transaction is fraudulent may not be enough in a practical payment system, where it is also important to identify unusual behaviour, assess transaction risk, and determine an appropriate action. Our framework addresses this gap by combining Isolation Forest, CatBoost, and an Adaptive Risk Decision Engine (ARDE) to provide risk-based decision support beyond simple fraud prediction.

## Proposed Framework

It consists of three major components:

### Isolation Forest

* Detects unusual transaction behaviour
* Identifies anomalous transactions
* Generates normalized anomaly scores

### CatBoost Classifier

* Learns fraud patterns from labelled transactions
* Uses transaction features along with anomaly-related information
* Classifies transactions as fraud or genuine

### Adaptive Risk Decision Engine (ARDE)

* Combines fraud probability and anomaly score with transaction risk factors
* Considers transaction amount and timing during risk assessment
* Determines the overall risk level and recommended action

## Dataset

The research uses a synthetic digital payment transaction dataset created for experimental evaluation. The dataset simulates transaction behaviour for fraud detection research and does not contain real customer or financial information.

## Results

The proposed framework achieved an accuracy of **97.55%**.

### Evaluation Metrics

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 97.55% |
| Precision | 88.46% |
| Recall    | 95.83% |
| F1-Score  | 92.00% |
| ROC-AUC   | 98.95% |

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* CatBoost
* Matplotlib

## Research Contribution

* Combines **Isolation Forest** for anomaly detection and **CatBoost** for fraud classification.
* Uses **ARDE** to combine fraud probability, anomaly score, transaction amount, and timing for risk assessment.
* Extends fraud detection from simple **fraud/genuine prediction** to **risk-based decision support** with recommended actions.

## Limitations

* The framework is evaluated using a synthetic dataset.
* Periodic threshold tuning may be required for the risk decision mechanism.
* Combining multiple detection techniques introduces slightly higher computational complexity.
* Performance may vary when applied to real-world transaction data.

## Future Work

* Evaluation using larger real-world datasets
* Continuous monitoring of model performance
* Detection of evolving fraud patterns
* Integration with real-time digital payment systems
* Exploration of additional machine learning techniques
