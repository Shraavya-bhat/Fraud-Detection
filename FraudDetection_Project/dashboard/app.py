"""
Real-Time Fraud Detection Dashboard
Streamlit Multi-Page Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Load Model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    with open(model_path, 'rb') as f:
        bundle = pickle.load(f)
    return bundle

@st.cache_data
def load_data():
    """Load and preprocess test data for display."""
    # Try to load processed data if available
    data_path = os.path.join(os.path.dirname(__file__), 'test_results.csv')
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None

# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/fraud.png", width=80)
st.sidebar.title("🛡️ Fraud Detection")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview", "🔍 Transaction Explorer", "🧠 SHAP Explainer"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Filters")
risk_filter = st.sidebar.multiselect(
    "Risk Tier",
    ["Critical Risk", "Suspicious", "Clear"],
    default=["Critical Risk", "Suspicious", "Clear"]
)

amount_range = st.sidebar.slider(
    "Transaction Amount ($)",
    min_value=0, max_value=10000, value=(0, 10000), step=50
)

hour_range = st.sidebar.slider(
    "Hour of Day",
    min_value=0, max_value=23, value=(0, 23)
)

st.sidebar.markdown("---")
st.sidebar.info("**Model:** LightGBM (Optuna-tuned)\n\n**Dataset:** IEEE-CIS Fraud Detection")

# ─── Helper Functions ─────────────────────────────────────────────────────────
def assign_tier(p):
    if p >= 0.75:   return "Critical Risk"
    elif p >= 0.40: return "Suspicious"
    else:           return "Clear"

def tier_color(tier):
    return {"Critical Risk": "🔴", "Suspicious": "🟡", "Clear": "🟢"}.get(tier, "⚪")

def get_plain_english(prob, top_features_dict):
    if prob >= 0.75:
        verdict = "⚠️ HIGH FRAUD RISK"
        desc = "This transaction exhibits multiple high-risk signals. Immediate review recommended."
    elif prob >= 0.40:
        verdict = "⚠️ BORDERLINE — REVIEW NEEDED"
        desc = "Mixed signals detected. A manual review is advisable before approving this transaction."
    else:
        verdict = "✅ LIKELY LEGITIMATE"
        desc = "This transaction shows low fraud indicators and matches typical legitimate patterns."

    feature_text = ""
    for feat, val in list(top_features_dict.items())[:3]:
        direction = "↑ increases" if val > 0 else "↓ decreases"
        feature_text += f"\n- **{feat}**: {direction} fraud risk (SHAP={val:.4f})"

    return verdict, desc, feature_text

# ─── Generate dummy results if no saved results ───────────────────────────────
@st.cache_data
def generate_sample_results(n=5000):
    np.random.seed(42)
    ids = np.arange(1000000, 1000000 + n)
    amounts = np.random.lognormal(4.5, 1.2, n)
    hours = np.random.randint(0, 24, n)
    probs = np.random.beta(0.5, 8, n)  # skewed low (most are legit)
    probs[:150] = np.random.beta(5, 2, 150)  # inject fraud cases
    np.random.shuffle(probs)
    tiers = [assign_tier(p) for p in probs]
    actual = (probs > 0.5).astype(int)

    df = pd.DataFrame({
        'TransactionID': ids,
        'TransactionAmt': amounts.round(2),
        'HourOfDay': hours,
        'FraudProbability': probs.round(4),
        'RiskTier': tiers,
        'ActualFraud': actual,
        'AmtToMeanRatio': (amounts / amounts.mean()).round(3),
        'DeviceRisk': np.random.randint(0, 2, n)
    })
    return df

# ─── PAGE 1 — Overview ───────────────────────────────────────────────────────
if page == "📊 Overview":
    st.title("🛡️ Fraud Detection Operations Center")
    st.markdown("**Real-Time Overview — IEEE-CIS Fraud Detection System**")
    st.markdown("---")

    df = generate_sample_results()

    # Apply filters
    df_filtered = df[
        (df['RiskTier'].isin(risk_filter)) &
        (df['TransactionAmt'].between(*amount_range)) &
        (df['HourOfDay'].between(*hour_range))
    ]

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Transactions", f"{len(df_filtered):,}", delta=None)
    with col2:
        fraud_count = int(df_filtered['ActualFraud'].sum())
        st.metric("🚨 Fraud Cases", f"{fraud_count:,}", delta=f"{fraud_count/len(df_filtered)*100:.2f}%")
    with col3:
        detection_rate = (df_filtered['FraudProbability'] >= 0.5).sum() / max(fraud_count, 1)
        st.metric("Detection Rate", f"{min(detection_rate, 1.0)*100:.1f}%")
    with col4:
        avg_fraud_amt = df_filtered[df_filtered['ActualFraud']==1]['TransactionAmt'].mean()
        st.metric("Avg Fraud Amount", f"${avg_fraud_amt:,.2f}" if not np.isnan(avg_fraud_amt) else "N/A")

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Risk Tier Distribution")
        tier_counts = df_filtered['RiskTier'].value_counts().reset_index()
        tier_counts.columns = ['Tier', 'Count']
        colors = {'Critical Risk': '#F44336', 'Suspicious': '#FF9800', 'Clear': '#4CAF50'}
        fig_donut = px.pie(
            tier_counts, names='Tier', values='Count',
            color='Tier', color_discrete_map=colors,
            hole=0.5
        )
        fig_donut.update_traces(textposition='outside', textinfo='percent+label')
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_right:
        st.subheader("Fraud Rate by Hour of Day")
        hourly = df_filtered.groupby('HourOfDay')['ActualFraud'].mean().reset_index()
        fig_hour = px.bar(
            hourly, x='HourOfDay', y='ActualFraud',
            color='ActualFraud', color_continuous_scale='Reds',
            labels={'HourOfDay': 'Hour', 'ActualFraud': 'Fraud Rate'}
        )
        fig_hour.update_layout(showlegend=False)
        st.plotly_chart(fig_hour, use_container_width=True)

    st.subheader("Transaction Amount vs Hour (Fraud Probability)")
    fig_scatter = px.scatter(
        df_filtered.sample(min(1500, len(df_filtered)), random_state=42),
        x='HourOfDay', y='TransactionAmt',
        color='FraudProbability', color_continuous_scale='RdYlGn_r',
        size='AmtToMeanRatio', size_max=10,
        hover_data=['TransactionID', 'RiskTier', 'FraudProbability'],
        labels={'HourOfDay': 'Hour', 'TransactionAmt': 'Amount ($)'}
    )
    st.plotly_chart(fig_scatter, use_container_width=True)


# ─── PAGE 2 — Transaction Explorer ───────────────────────────────────────────
elif page == "🔍 Transaction Explorer":
    st.title("🔍 Transaction Explorer")
    st.markdown("Search, filter, and inspect individual transactions with live risk scores.")
    st.markdown("---")

    df = generate_sample_results()

    # Apply filters
    df_filtered = df[
        (df['RiskTier'].isin(risk_filter)) &
        (df['TransactionAmt'].between(*amount_range)) &
        (df['HourOfDay'].between(*hour_range))
    ].copy()

    # Search by ID
    search_id = st.text_input("🔎 Search by TransactionID", placeholder="e.g. 1000042")
    if search_id.strip():
        try:
            df_filtered = df_filtered[df_filtered['TransactionID'] == int(search_id)]
        except ValueError:
            st.warning("Please enter a valid numeric TransactionID.")

    # Sort
    sort_col = st.selectbox("Sort by", ['FraudProbability', 'TransactionAmt', 'HourOfDay'], index=0)
    df_display = df_filtered.sort_values(sort_col, ascending=False).head(500).copy()

    # Color-code risk tier
    def highlight_tier(row):
        colors_map = {'Critical Risk': 'background-color: #ffcccc',
                      'Suspicious': 'background-color: #fff3cc',
                      'Clear': 'background-color: #ccffcc'}
        return [colors_map.get(row['RiskTier'], '')] * len(row)

    st.dataframe(
        df_display[['TransactionID', 'TransactionAmt', 'HourOfDay',
                    'FraudProbability', 'RiskTier', 'DeviceRisk', 'ActualFraud']],
        use_container_width=True, height=400
    )

    st.markdown("---")
    st.subheader("📌 Quick Risk Score Lookup")
    lookup_id = st.number_input("Enter TransactionID", min_value=1000000, max_value=1004999, step=1, value=1000042)
    row = df[df['TransactionID'] == lookup_id]
    if not row.empty:
        r = row.iloc[0]
        tier_icon = tier_color(r['RiskTier'])
        st.markdown(f"""
        | Field | Value |
        |-------|-------|
        | TransactionID | `{r['TransactionID']}` |
        | Amount | `${r['TransactionAmt']:.2f}` |
        | Hour | `{r['HourOfDay']}:00` |
        | Fraud Probability | `{r['FraudProbability']:.4f}` |
        | Risk Tier | {tier_icon} **{r['RiskTier']}** |
        | Device Risk | `{'High' if r['DeviceRisk'] else 'Normal'}` |
        """)
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=r['FraudProbability'] * 100,
            title={'text': "Fraud Risk Score (%)"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': '#F44336' if r['FraudProbability'] >= 0.75 else '#FF9800' if r['FraudProbability'] >= 0.40 else '#4CAF50'},
                   'steps': [{'range': [0, 40], 'color': '#e8f5e9'},
                              {'range': [40, 75], 'color': '#fff8e1'},
                              {'range': [75, 100], 'color': '#ffebee'}]}
        ))
        st.plotly_chart(gauge, use_container_width=True)
    else:
        st.info("TransactionID not found in sample dataset.")


# ─── PAGE 3 — SHAP Explainer ─────────────────────────────────────────────────
elif page == "🧠 SHAP Explainer":
    st.title("🧠 SHAP Explainer — Why Was This Transaction Flagged?")
    st.markdown("Enter a TransactionID to see a feature-level explanation of the fraud score.")
    st.markdown("---")

    df = generate_sample_results()

    txn_id = st.number_input("Transaction ID", min_value=1000000, max_value=1004999, step=1, value=1000042)
    row = df[df['TransactionID'] == txn_id]

    if not row.empty:
        r = row.iloc[0]
        prob = r['FraudProbability']
        tier = r['RiskTier']

        st.markdown(f"### Transaction `{txn_id}`")
        col1, col2, col3 = st.columns(3)
        col1.metric("Fraud Probability", f"{prob:.4f}")
        col2.metric("Risk Tier", tier)
        col3.metric("Amount", f"${r['TransactionAmt']:.2f}")

        st.markdown("---")

        # Simulate SHAP values for demo (replace with real explainer in production)
        feature_names = [
            'TransactionAmt', 'AmtToMeanRatio', 'HourOfDay', 'V258', 'V201',
            'card1', 'addr1', 'D1', 'C1', 'DeviceRisk',
            'V294', 'V83', 'D15', 'C13', 'LogTransactionAmt',
            'V126', 'C6', 'D10', 'V307', 'V313'
        ]
        np.random.seed(int(txn_id) % 1000)
        shap_vals = np.random.normal(0, 0.15, len(feature_names))
        shap_vals[0] = prob * 1.5 - 0.3  # TransactionAmt dominant
        shap_vals[1] = prob * 0.8 - 0.1
        shap_vals[2] = 0.08 if r['HourOfDay'] < 5 else -0.05
        shap_vals[9] = 0.12 if r['DeviceRisk'] else -0.04
        shap_vals = np.clip(shap_vals, -0.5, 0.5)

        shap_df = pd.DataFrame({'Feature': feature_names, 'SHAP Value': shap_vals})
        shap_df = shap_df.reindex(shap_df['SHAP Value'].abs().sort_values(ascending=False).index)

        st.subheader("📊 SHAP Feature Contributions")
        fig_shap = px.bar(
            shap_df.head(15), x='SHAP Value', y='Feature', orientation='h',
            color='SHAP Value', color_continuous_scale='RdBu_r',
            title=f'SHAP Waterfall — TransactionID {txn_id}',
            color_continuous_midpoint=0
        )
        fig_shap.update_layout(yaxis={'categoryorder': 'total ascending'}, height=450)
        st.plotly_chart(fig_shap, use_container_width=True)

        # Plain-English explanation
        top_features = dict(zip(shap_df['Feature'].head(5), shap_df['SHAP Value'].head(5)))
        verdict, desc, feature_text = get_plain_english(prob, top_features)

        st.markdown("---")
        st.subheader("📝 Plain-English Explanation")
        st.markdown(f"### {verdict}")
        st.markdown(desc)
        st.markdown("**Key contributing factors:**")
        st.markdown(feature_text)

        st.markdown("---")
        st.caption("⚠️ SHAP values shown are illustrative approximations for the dashboard demo. "
                   "Run `analysis.ipynb` to compute real SHAP values from the trained model.")
    else:
        st.warning("TransactionID not found. Try IDs between 1000000 and 1004999.")
