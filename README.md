# 🏦 Credit Risk Causal ML

> **Predict *why* borrowers default — not just *that* they will.**
> Combining Gradient Boosting + Causal Inference (DoWhy) + SHAP Explainability into a production-ready credit risk system.

---

## 🎯 Project Highlights

| Dimension | Detail |
|---|---|
| **Dataset** | LendingClub (100K sample from 2M+ loan records) |
| **Models** | XGBoost · LightGBM · CatBoost (best auto-selected via MLflow) |
| **Tuning** | Optuna TPE — 30 trials per model |
| **Causal Inference** | DoWhy — backdoor ATE estimation per financial variable |
| **Explainability** | SHAP global summaries + per-applicant waterfall plots |
| **Serving** | FastAPI REST API with `/predict` endpoint |
| **Dashboard** | Streamlit — real-time risk gauge + SHAP factor visualization |
| **Tracking** | MLflow — all experiments logged |
| **Deploy** | Docker + Railway / AWS EC2 |

---

## 📁 Project Structure

```
credit-risk-causal-ml/
├── data/
│   └── processed/          ← preprocessed CSVs (generated)
├── models/                 ← saved model artifacts (generated)
│   └── plots/              ← SHAP plots (generated)
├── src/
│   ├── preprocess.py       ← data cleaning + feature engineering
│   ├── causal_model.py     ← DoWhy DAG + ATE estimation
│   ├── train.py            ← model training + Optuna + MLflow
│   └── explain.py          ← SHAP global + local explanations
├── api/
│   └── main.py             ← FastAPI prediction server
├── app.py                  ← Streamlit dashboard
├── run_pipeline.py         ← master script (runs everything)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 🚀 Setup & Run on Your Machine

### Step 1 — Clone / Download
```bash
git clone https://github.com/YOUR_USERNAME/credit-risk-causal-ml.git
cd credit-risk-causal-ml
```

### Step 2 — Create virtual environment
```bash
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Download Dataset
1. Go to: https://www.kaggle.com/datasets/wordsforthewise/lending-club
2. Download `accepted_2007_to_2018Q4.csv.gz`
3. Extract and rename to `lending_club_raw.csv`
4. Place it in the `data/` folder

### Step 5 — Run full pipeline
```bash
python run_pipeline.py --input data/lending_club_raw.csv --trials 30
```

This runs all 4 steps automatically:
1. ✅ Data preprocessing + feature engineering
2. ✅ Causal DAG construction + ATE estimation (DoWhy)
3. ✅ Model training — XGBoost, LightGBM, CatBoost (Optuna tuned)
4. ✅ SHAP explainability + plot generation

---

## 🔬 Run Steps Individually

```bash
# Step 1 — Preprocess only
python src/preprocess.py --input data/lending_club_raw.csv

# Step 2 — Causal analysis only
python src/causal_model.py

# Step 3 — Train only
python src/train.py

# Step 4 — SHAP explanations only
python src/explain.py
```

---

## 🌐 Start the API

```bash
uvicorn api.main:app --reload --port 8000
```

**Test it:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "loan_amnt": 15000, "funded_amnt": 15000,
    "term": 36, "int_rate": 13.5, "installment": 507.0,
    "emp_length": 5.0, "home_ownership": 1, "annual_inc": 60000,
    "verification_status": 1, "purpose": 2, "dti": 18.5,
    "delinq_2yrs": 0.0, "fico_score": 700.0, "inq_last_6mths": 1.0,
    "open_acc": 10.0, "pub_rec": 0.0, "revol_bal": 12000.0,
    "revol_util": 45.0, "total_acc": 20.0, "initial_list_status": 1,
    "application_type": 0, "grade_num": 3, "payment_to_income": 0.10,
    "loan_to_income": 0.25, "credit_utilization": 0.20
}'
```

**API Docs:** http://localhost:8000/docs

---

## 📊 Start the Streamlit Dashboard

```bash
# Make sure API is running first, then:
streamlit run app.py
```
Open: http://localhost:8501

---

## 📈 View MLflow Experiments

```bash
mlflow ui
```
Open: http://localhost:5000

---

## 🐳 Docker Deploy

```bash
# Build and start both API + Dashboard
docker-compose up --build

# API:       http://localhost:8000
# Dashboard: http://localhost:8501
```

---

## ☁️ Deploy to Railway (Free)

1. Push code to GitHub
2. Go to railway.app → New Project → Deploy from GitHub
3. Add environment variable: `MODEL_DIR=models`
4. Railway auto-detects Dockerfile and deploys

---

## 🎯 Target Metrics

| Metric | Target | Why it matters |
|---|---|---|
| AUC-ROC | ≥ 0.92 | Overall discrimination power |
| F1-Score (Default) | ≥ 0.78 | Handles class imbalance |
| KS Statistic | ≥ 0.45 | Industry standard for credit scoring |
| PR-AUC | ≥ 0.80 | Precision-Recall tradeoff on imbalanced data |

---

## 💡 What Makes This Different

1. **Causal inference, not just correlation** — DoWhy estimates *causal* impact of DTI, interest rate, FICO on default, controlling for confounders via backdoor adjustment.

2. **Regulatory-ready explainability** — SHAP per-applicant explanations satisfy fair lending requirements (ECOA, Fair Credit Reporting Act).

3. **KS Statistic** — Standard credit risk metric rarely seen in academic ML projects but universally used in production banking.

4. **Production architecture** — FastAPI + Docker + MLflow + Railway mirrors real fintech deployment stacks.

---

## 🛠️ Tech Stack

`Python 3.11` · `XGBoost` · `LightGBM` · `CatBoost` · `Optuna` · `DoWhy` ·
`SHAP` · `MLflow` · `FastAPI` · `Streamlit` · `Plotly` · `Docker` · `Railway`
