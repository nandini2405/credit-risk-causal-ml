"""
app.py  —  Streamlit Dashboard for Credit Risk Causal ML
---------------------------------------------------------
Run: streamlit run app.py
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Causal ML",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .risk-low    { color: #27ae60; font-size: 2rem; font-weight: bold; }
    .risk-medium { color: #f39c12; font-size: 2rem; font-weight: bold; }
    .risk-high   { color: #e74c3c; font-size: 2rem; font-weight: bold; }
    .metric-card { background: #1e1e2e; padding: 1rem; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar — Applicant Inputs ────────────────────────────────────────────────
st.sidebar.title("🏦 Applicant Details")

loan_amnt    = st.sidebar.slider("Loan Amount ($)", 1000, 40000, 15000, step=500)
term         = st.sidebar.selectbox("Term (months)", [36, 60])
int_rate     = st.sidebar.slider("Interest Rate (%)", 5.0, 30.0, 13.5, step=0.1)
annual_inc   = st.sidebar.number_input("Annual Income ($)", 10000, 300000, 60000, step=1000)
dti          = st.sidebar.slider("Debt-to-Income Ratio", 0.0, 60.0, 18.5, step=0.5)
fico_score   = st.sidebar.slider("FICO Score", 580, 850, 700, step=5)
emp_length   = st.sidebar.slider("Employment Length (yrs)", 0, 10, 5)
grade_num    = st.sidebar.selectbox("Grade", [1,2,3,4,5,6,7],
                                     format_func=lambda x: ["A","B","C","D","E","F","G"][x-1])
delinq_2yrs  = st.sidebar.number_input("Delinquencies (2yrs)", 0, 20, 0)
pub_rec      = st.sidebar.number_input("Public Records", 0, 10, 0)
revol_bal    = st.sidebar.number_input("Revolving Balance ($)", 0, 100000, 12000, step=500)
revol_util   = st.sidebar.slider("Revolving Utilization (%)", 0.0, 100.0, 45.0)
home_ownership = st.sidebar.selectbox("Home Ownership", [0,1,2],
                                        format_func=lambda x: ["RENT","OWN","MORTGAGE"][x])

# Derived features
installment       = round((loan_amnt * (int_rate/100/12)) /
                          (1 - (1 + int_rate/100/12)**(-term)), 2)
payment_to_income = round(installment / (annual_inc / 12 + 1e-6), 4)
loan_to_income    = round(loan_amnt / (annual_inc + 1e-6), 4)
credit_utilization= round(revol_bal / (annual_inc + 1e-6), 4)

payload = {
    "loan_amnt": loan_amnt, "funded_amnt": loan_amnt,
    "term": term, "int_rate": int_rate, "installment": installment,
    "emp_length": float(emp_length), "home_ownership": home_ownership,
    "annual_inc": annual_inc, "verification_status": 1, "purpose": 2,
    "dti": dti, "delinq_2yrs": float(delinq_2yrs), "fico_score": float(fico_score),
    "inq_last_6mths": 1.0, "open_acc": 10.0, "pub_rec": float(pub_rec),
    "revol_bal": float(revol_bal), "revol_util": revol_util, "total_acc": 20.0,
    "initial_list_status": 1, "application_type": 0, "grade_num": grade_num,
    "payment_to_income": payment_to_income, "loan_to_income": loan_to_income,
    "credit_utilization": credit_utilization,
}


# ── Main panel ────────────────────────────────────────────────────────────────
st.title("🏦 Credit Risk — Causal ML Dashboard")
st.caption("Powered by XGBoost/LightGBM/CatBoost + DoWhy Causal Inference + SHAP Explainability")

col_pred, col_gauge, col_factors = st.columns([1.2, 1.5, 1.8])

if st.sidebar.button("🔍 Predict Default Risk", use_container_width=True):
    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
        resp.raise_for_status()
        result = resp.json()

        prob      = result["default_probability"]
        tier      = result["risk_tier"]
        score     = result["risk_score"]
        factors   = result["top_3_causal_factors"]
        interp    = result["interpretation"]

        # ── Probability metric ────────────────────────────────────────────────
        with col_pred:
            st.markdown("### Prediction")
            color = {"LOW": "#27ae60","MEDIUM": "#f39c12",
                     "HIGH": "#e74c3c","VERY HIGH": "#c0392b"}.get(tier, "#888")
            st.markdown(f"<h2 style='color:{color}'>{tier} RISK</h2>", unsafe_allow_html=True)
            st.metric("Default Probability", f"{prob:.1%}")
            st.metric("Risk Score",          score)
            st.info(interp)

        # ── Gauge chart ───────────────────────────────────────────────────────
        with col_gauge:
            st.markdown("### Risk Gauge")
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=round(prob * 100, 1),
                number={"suffix": "%"},
                title={"text": "Default Probability"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": color},
                    "steps": [
                        {"range": [0,  20], "color": "#d5f5e3"},
                        {"range": [20, 45], "color": "#fef9e7"},
                        {"range": [45, 65], "color": "#fde8d8"},
                        {"range": [65,100], "color": "#fadbd8"},
                    ],
                    "threshold": {"line": {"color": "red", "width": 3},
                                  "thickness": 0.8, "value": 50},
                }
            ))
            fig.update_layout(height=280, margin=dict(t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

        # ── SHAP factors ──────────────────────────────────────────────────────
        with col_factors:
            st.markdown("### Top Causal Factors (SHAP)")
            if factors and "error" not in factors[0]:
                names  = [f["feature"] for f in factors]
                values = [f["shap_value"] for f in factors]
                colors = ["#e74c3c" if v > 0 else "#27ae60" for v in values]

                fig2 = go.Figure(go.Bar(
                    x=values, y=names, orientation="h",
                    marker_color=colors,
                    text=[f"{v:+.4f}" for v in values],
                    textposition="outside"
                ))
                fig2.update_layout(
                    title="SHAP Values (red=increases risk, green=decreases)",
                    xaxis_title="SHAP Value",
                    height=280,
                    margin=dict(t=40, b=10)
                )
                st.plotly_chart(fig2, use_container_width=True)

                for f in factors:
                    direction_icon = "🔴" if f["shap_value"] > 0 else "🟢"
                    st.write(f"{direction_icon} **{f['feature']}** → {f['direction']} "
                             f"(SHAP: {f['shap_value']:+.4f})")
            else:
                st.warning("SHAP factors unavailable. Run explain.py first.")

    except requests.exceptions.ConnectionError:
        st.error("❌ API not reachable. Make sure the FastAPI server is running:\n"
                 "`uvicorn api.main:app --reload --port 8000`")
    except Exception as e:
        st.error(f"Error: {e}")

else:
    # ── Placeholder panels ────────────────────────────────────────────────────
    with col_pred:
        st.markdown("### Prediction")
        st.info("👈 Fill in applicant details and click **Predict**.")

    with col_gauge:
        st.markdown("### Risk Gauge")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=0,
            number={"suffix": "%"},
            title={"text": "Default Probability"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#ccc"},
                   "steps": [
                       {"range": [0,  20], "color": "#d5f5e3"},
                       {"range": [20, 45], "color": "#fef9e7"},
                       {"range": [45, 65], "color": "#fde8d8"},
                       {"range": [65,100], "color": "#fadbd8"},
                   ]}
        ))
        fig.update_layout(height=280, margin=dict(t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_factors:
        st.markdown("### Top Causal Factors (SHAP)")
        st.info("Prediction factors will appear here after you click Predict.")


# ── Bottom — About section ────────────────────────────────────────────────────
st.divider()
with st.expander("ℹ️ About this Project"):
    st.markdown("""
**Tech Stack:** XGBoost · LightGBM · CatBoost · DoWhy · SHAP · MLflow · FastAPI · Streamlit · Docker

**Key differentiators:**
- Uses **causal inference** (DoWhy backdoor adjustment) to identify *why* someone defaults, not just *that* they will
- Per-applicant **SHAP waterfall explanations** — interpretable AI, compliant with fair lending requirements (ECOA)
- **KS Statistic** tracked alongside AUC-ROC — standard credit risk evaluation metric
- **Optuna** hyperparameter tuning across 3 gradient-boosted models — best model auto-selected by MLflow

**Dataset:** LendingClub (2M+ loan records, Kaggle)
    """)
