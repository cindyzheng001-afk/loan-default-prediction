# Loan Default Prediction

A machine learning pipeline to predict loan defaults using 2.2M LendingClub loans (2007–2018).

## Business Problem
Lenders lose billions annually to loan defaults. This project builds an explainable ML model that predicts default risk before a loan is issued, with a business-focused threshold simulator showing the dollar impact of different risk tolerances.

## Results
| Model | AUC Score | Default Recall |
|-------|-----------|----------------|
| Logistic Regression | 0.7041 | 9% |
| Random Forest + SMOTE | 0.6905 | 38% |
| XGBoost + SMOTE | 0.7131 | 11% |

**Optimal threshold (0.30):** Saves $384M while maintaining reasonable approval rates.

## Key Findings
- Loan grade is the strongest single predictor (6.7% default at Grade A vs 51.1% at Grade G)
- SHAP analysis revealed loan purpose is the #1 feature — debt consolidation loans carry the highest systemic risk
- Lowering decision threshold from 0.50 to 0.30 saves an additional $294M but rejects 39,000 more good loans

## Tech Stack
- **Python** — pandas, numpy, scikit-learn, xgboost, shap, imbalanced-learn
- **Visualization** — matplotlib, seaborn
- **Dashboard** — Streamlit
- **Dataset** — LendingClub Accepted Loans 2007–2018 (Kaggle)

## Project Structure
loan-default-prediction/
├── notebooks/
│   └── 01_eda.ipynb
├── dashboard/
│   └── app.py
├── models/
│   └── xgb_model.pkl
├── reports/
│   ├── default_rate_by_grade.png
│   ├── default_rate_by_income.png
│   ├── default_rate_by_dti.png
│   ├── default_rate_by_state.png
│   ├── shap_feature_importance.png
│   ├── shap_beeswarm.png
│   ├── shap_waterfall.png
│   └── threshold_simulator.png
└── sql/

## How to Run
pip install pandas numpy scikit-learn xgboost shap imbalanced-learn streamlit joblib

cd dashboard
streamlit run app.py

## Author
Cindy Zheng | Data Analytics Capstone 2025