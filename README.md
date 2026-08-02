# Credit Risk Prediction

A machine learning project to predict loan default risk using the LendingClub dataset.

## Problem Statement

When a bank or lending platform issues a loan, it needs to assess the risk that the borrower will default (fail to repay). This project builds a binary classification model to predict whether a loan will be **"Charged Off"** (default) or **"Fully Paid"**, based on borrower and loan attributes.

## Dataset

- **Source:** [LendingClub Loan Data (Kaggle)](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
- **Used:** 50,000 records (sampled from 2M+ available), 19 selected features out of 151 original columns

## Project Workflow

1. **Data Loading & Exploration** — Reviewed all 151 columns, identified relevant features
2. **Target Definition** — Created binary target: `1` = Charged Off, `0` = Fully Paid
3. **Feature Selection** — Selected 19 features relevant to credit risk (loan amount, interest rate, income, FICO score, DTI, etc.)
4. **Data Cleaning** — Handled missing values (mode/median imputation), removed outliers (invalid DTI and income values)
5. **Exploratory Data Analysis (EDA)** — Visualized loan amount distribution, default rate by grade, feature correlations, default rate by loan purpose, and interest rate/DTI vs default
6. **Encoding** — Label Encoding for categorical features
7. **Train-Test Split** — 80/20 split, stratified by target
8. **Modeling**
   - Logistic Regression (baseline, with feature scaling)
   - XGBoost (with `scale_pos_weight` for class imbalance)
9. **Class Imbalance Handling** — Applied SMOTE on training data
10. **Hyperparameter Tuning** — GridSearchCV over `n_estimators`, `max_depth`, `learning_rate`
11. **Evaluation** — Accuracy, AUC-ROC, Precision/Recall/F1
12. **Feature Importance** — Identified top predictors of default
13. **Sample Predictions** — Demonstrated model output on test samples

## Results

| Model | AUC-ROC | Notes |
|---|---|---|
| Logistic Regression | 0.726 | High accuracy, low recall on defaulters |
| XGBoost (scale_pos_weight) | 0.664 | Improved recall, lower AUC |
| XGBoost + SMOTE + GridSearchCV | **0.722** | Best balance; tuned with `learning_rate=0.05, max_depth=5, n_estimators=200` |

**Top predictive features:** Loan term, interest rate, loan grade, home ownership, DTI

## Key Insight

Cross-validation AUC during hyperparameter tuning was 0.897, but test AUC was 0.722. This gap occurs because SMOTE-balanced training data makes cross-validation folds easier than the real-world imbalanced test set — a useful lesson in evaluating models fairly.

## Tech Stack

Python, Pandas, NumPy, Scikit-learn (Logistic Regression, GridSearchCV), XGBoost, imbalanced-learn (SMOTE), Matplotlib, Seaborn

## Setup Instructions

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
2. Extract `accepted_2007_to_2018Q4.csv.gz` from the downloaded zip
3. Upload this file to your Google Drive (root "My Drive" folder)
4. Open `Credit_Risk_project.ipynb` in Google Colab
5. Run cells in order — the first cell will prompt Google Drive authorization

## Limitations & Future Improvements

- Trained on a 50,000-row sample (out of 2M+ available records) due to computational constraints
- Used 19 manually selected features out of 151 available columns
- AUC-ROC of ~0.72 indicates room for improvement; production credit models typically achieve 0.75–0.85+
- Class imbalance (~20% defaults) limits recall on the minority class
- Future work: SHAP for detailed explainability, additional features (credit history depth), training on the full dataset, ensemble/stacking methods

## Key Learning

Initially, SMOTE was applied before GridSearchCV, causing CV AUC (0.897) to be 
much higher than test AUC (0.722). After fixing with `imblearn.pipeline`, 
CV AUC (0.715) and test AUC (0.723) are now aligned, giving an honest estimate 
of model performance on real-world imbalanced data.
