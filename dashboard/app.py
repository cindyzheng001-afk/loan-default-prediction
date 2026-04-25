import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap

# ── PAGE CONFIG ──
st.set_page_config(
    page_title="Loan Default Predictor",
    page_icon="🏦",
    layout="wide"
)

# ── LOAD MODEL ──
@st.cache_resource
def load_model():
    model = joblib.load(r'C:\Users\czhen\Desktop\loan-default-prediction\models\xgb_model.pkl')
    return model

xgb_model = load_model()

# ── HEADER ──
st.title("🏦 Loan Default Prediction Dashboard")
st.markdown("**Built with XGBoost + SHAP | LendingClub Dataset (2.2M loans)**")
st.divider()

# ── SIDEBAR INPUTS ──
st.sidebar.header("📋 Loan Application Input")

loan_amnt = st.sidebar.slider("Loan Amount ($)", 1000, 40000, 15000, step=500)
int_rate = st.sidebar.slider("Interest Rate (%)", 5.0, 30.0, 13.0, step=0.5)
grade = st.sidebar.selectbox("Loan Grade", ["A","B","C","D","E","F","G"])
annual_inc = st.sidebar.slider("Annual Income ($)", 20000, 300000, 75000, step=5000)
dti = st.sidebar.slider("Debt-to-Income Ratio", 0.0, 50.0, 15.0, step=0.5)
fico = st.sidebar.slider("FICO Score", 580, 850, 700, step=5)
emp_length = st.sidebar.slider("Employment Length (years)", 0, 10, 5)
term = st.sidebar.radio("Loan Term", [36, 60])
home_ownership = st.sidebar.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
purpose = st.sidebar.selectbox("Loan Purpose", [
    "debt_consolidation", "credit_card", "home_improvement",
    "other", "major_purchase", "medical", "small_business",
    "vacation", "moving", "house", "wedding", "educational",
    "renewable_energy"
])
threshold = st.sidebar.slider("Decision Threshold", 0.10, 0.60, 0.30, step=0.05)

# ── FEATURE ENGINEERING ──
grade_map = {"A":1,"B":2,"C":3,"D":4,"E":5,"F":6,"G":7}
grade_encoded = grade_map[grade]
high_utilization = 1 if dti > 75 else 0

input_data = {
    'loan_amnt': loan_amnt, 'int_rate': int_rate,
    'installment': loan_amnt / term, 'term_months': term,
    'grade_encoded': grade_encoded, 'annual_inc': annual_inc,
    'dti': dti, 'fico_avg': fico, 'revol_util': dti * 2,
    'high_utilization': high_utilization, 'open_acc': 10,
    'pub_rec': 0, 'emp_length_clean': emp_length,
    'home_ownership_OTHER': 1 if home_ownership == 'OTHER' else 0,
    'home_ownership_OWN': 1 if home_ownership == 'OWN' else 0,
    'home_ownership_RENT': 1 if home_ownership == 'RENT' else 0,
    'purpose_credit_card': 1 if purpose == 'credit_card' else 0,
    'purpose_debt_consolidation': 1 if purpose == 'debt_consolidation' else 0,
    'purpose_home_improvement': 1 if purpose == 'home_improvement' else 0,
    'purpose_house': 1 if purpose == 'house' else 0,
    'purpose_major_purchase': 1 if purpose == 'major_purchase' else 0,
    'purpose_medical': 1 if purpose == 'medical' else 0,
    'purpose_moving': 1 if purpose == 'moving' else 0,
    'purpose_other': 1 if purpose == 'other' else 0,
    'purpose_small_business': 1 if purpose == 'small_business' else 0,
    'purpose_vacation': 1 if purpose == 'vacation' else 0,
    'purpose_wedding': 1 if purpose == 'wedding' else 0,
}

input_df = pd.DataFrame([input_data])

# ── PREDICTION ──
prob = xgb_model.predict_proba(input_df)[0][1]
decision = "❌ DENY" if prob >= threshold else "✅ APPROVE"
color = "red" if prob >= threshold else "green"

# ── MAIN PANEL ──
col1, col2, col3 = st.columns(3)
col1.metric("Default Probability", f"{prob:.1%}")
col2.metric("Decision Threshold", f"{threshold:.0%}")
col3.metric("Decision", decision)

st.divider()

# ── THRESHOLD SIMULATOR ──
st.subheader("📊 Threshold Business Impact")
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    | Metric | Value |
    |--------|-------|
    | Default Probability | {prob:.1%} |
    | Decision | {decision} |
    | Threshold | {threshold:.0%} |
    | Loan Amount | ${loan_amnt:,} |
    | Grade | {grade} |
    | FICO Score | {fico} |
    """)

with col2:
    # Gauge chart
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.barh(['Risk Score'], [prob], color=color, height=0.3)
    ax.barh(['Risk Score'], [1 - prob], left=[prob], 
             color='lightgray', height=0.3)
    ax.axvline(x=threshold, color='black', linestyle='--', 
                label=f'Threshold ({threshold:.0%})')
    ax.set_xlim(0, 1)
    ax.set_title('Default Risk Gauge')
    ax.legend()
    st.pyplot(fig)

st.divider()
st.markdown("*Built by Cindy Zheng | Capstone Project 2025*")