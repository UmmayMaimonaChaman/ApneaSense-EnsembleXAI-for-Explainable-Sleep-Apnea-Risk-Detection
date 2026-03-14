# ApneaSense: EnsembleXAI for Explainable Sleep-Apnea Risk Detection
💤An end-to-end **machine learning pipeline** for predicting **sleep apnea risk** using lifestyle, physiological, and cardiovascular indicators. 
The project integrates **ensemble learning, statistical analysis, and explainable AI (XAI)** to build interpretable predictive models.


![Python](https://img.shields.io/badge/Python-3.10-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)
![Explainable AI](https://img.shields.io/badge/XAI-SHAP-purple)
![Status](https://img.shields.io/badge/Project-Research%20Prototype-yellow)

---

# 📌 Project Overview

Sleep apnea is a common but underdiagnosed disorder associated with cardiovascular complications and reduced quality of life. This project explores how **machine learning models can assist in early risk detection** using accessible health indicators.

Key components include:

- Statistical feature significance testing
- Class imbalance correction (SMOTE)
- Multiple machine learning models
- Ensemble learning techniques
- Explainable AI interpretation (SHAP + permutation importance)

The final system achieved:

**ROC-AUC ≈ 0.71** for apnea risk prediction.

---

# 🧠 Machine Learning Workflow

```mermaid
flowchart TD

A[Raw Lifestyle & Diagnostic Datasets] --> B[Data Cleaning & Preprocessing]

B --> C[Feature Engineering]
C --> D[Statistical Significance Testing]

D --> E[Dataset Alignment & Merge]

E --> F[Train Test Split]

F --> G1[Logistic Regression]
F --> G2[Random Forest]
F --> G3[XGBoost]

G1 --> H[Cross Validation]
G2 --> H
G3 --> H

H --> I[Class Imbalance Handling SMOTE]

I --> J[Model Training]

J --> K[Model Evaluation]

K --> L1[Individual Model Metrics]
K --> L2[Ensemble Models]

L2 --> M1[Soft Voting]
L2 --> M2[Weighted Late Fusion]

M1 --> N[Best Model Selection]
M2 --> N

N --> O[Explainable AI Analysis]

O --> P1[Feature Importance]
O --> P2[SHAP Interpretation]
O --> P3[Permutation Importance]

P1 --> Q[Clinical Insights]
P2 --> Q
P3 --> Q
