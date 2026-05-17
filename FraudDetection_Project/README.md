# 🛡️ Real-Time Fraud Detection System with Explainable AI

**Domain:** AI & Data Analytics | **Level:** Advanced | **Week 4 — Capstone**

---

## 📁 Project Structure

```
FraudDetection_Project/
├── analysis.ipynb          ← Main Jupyter Notebook (Tasks 1–8)
├── data/
│   ├── train_transaction.csv
│   └── train_identity.csv
├── dashboard/
│   ├── app.py              ← Streamlit multi-page dashboard
│   └── model.pkl           ← Trained LightGBM model + scaler
├── charts/                 ← All saved visualisation charts
│   ├── class_imbalance.png
│   ├── txn_amt_distribution.png
│   ├── correlation_heatmap.png
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   ├── pr_curves.png
│   ├── threshold_f1.png
│   ├── shap_waterfall_confirmed_fraud.png
│   ├── shap_waterfall_borderline.png
│   ├── shap_waterfall_legitimate.png
│   ├── shap_dependence.png
│   ├── shap_vs_model_importance.png
│   ├── risk_tier_comparison.png
│   ├── risk_tier_donut.png
│   ├── fraud_rate_by_hour.png
│   ├── pr_optimal_threshold.png
│   └── interactive_scatter.html
├── model_comparison.png    ← Model metrics comparison bar chart
├── shap_summary.png        ← Global SHAP summary plot
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Download Dataset
1. Go to [Kaggle IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/data)
2. Download `train_transaction.csv` and `train_identity.csv`
3. Place both files inside the `data/` folder

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Jupyter Notebook
```bash
jupyter notebook analysis.ipynb
```
Run all cells in order. Each task is clearly labelled with Markdown headings.

### 4. Run Streamlit Dashboard
```bash
cd dashboard
streamlit run app.py
```

---

## 🌐 Live Dashboard

**Streamlit Community Cloud URL:**  
> [https://your-username-fraud-detection.streamlit.app](https://your-username-fraud-detection.streamlit.app)

*(Deploy steps: push repo to GitHub → go to share.streamlit.io → connect repo → set `dashboard/app.py` as entrypoint)*

---

## ✅ Tasks Completed

| Task | Description | Status |
|------|-------------|--------|
| Task 1 | Data Loading, Merging & EDA | ✅ Complete |
| Task 2 | Preprocessing, SMOTE & Feature Engineering | ✅ Complete |
| Task 3 | Model Training, Comparison & Threshold Optimization | ✅ Complete |
| Task 4 | SHAP Explainable AI | ✅ Complete |
| Task 5 | Risk Segmentation & Fraud Pattern Analysis | ✅ Complete |
| Task 6 | Streamlit Multi-Page Dashboard | ✅ Complete |
| Task 7 | Visualizations (7 charts + interactive) | ✅ Complete |
| Task 8 | Insights & Business Recommendations | ✅ Complete |

---

## 🤖 Models Used

| Model | Type |
|-------|------|
| LightGBM | Gradient Boosted Trees (Primary) |
| XGBoost | Gradient Boosted Trees (Comparison) |
| Isolation Forest | Unsupervised Anomaly Detection |

**Best Model:** LightGBM tuned with Optuna (30 trials)  
**Metric Focus:** PR-AUC (most meaningful for class-imbalanced fraud)

---

## 📊 Key Results

- Severe class imbalance handled via SMOTE (sampling_strategy=0.30)
- Threshold optimized using F1-Score maximization
- Top fraud signals: TransactionAmt, HourOfDay, Vesta V-features
- Estimated annual savings: ~$122M for a mid-size bank

---

## 🛠️ Tools & Libraries

`Python 3.x` · `Pandas` · `NumPy` · `LightGBM` · `XGBoost` · `Scikit-learn`  
`imbalanced-learn (SMOTE)` · `SHAP` · `Optuna` · `Plotly` · `Streamlit` · `Matplotlib` · `Seaborn`
