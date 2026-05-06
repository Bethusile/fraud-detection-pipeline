import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import joblib
import os

load_dotenv()

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="shield",
    layout="wide"
)

DB_URL = (
    f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
    f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
)

@st.cache_resource
def get_engine():
    return create_engine(DB_URL)

@st.cache_data(ttl=5)
def load_transactions():
    engine = get_engine()
    query = text("""
        SELECT * FROM transactions
        ORDER BY processed_at DESC
        LIMIT 2000
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn)

@st.cache_data(ttl=5)
def load_summary():
    engine = get_engine()
    query = text("""
        SELECT
            COUNT(*) as total,
            SUM(is_fraud_predicted) as fraud_count,
            SUM(transaction_amt) as total_amt,
            SUM(CASE WHEN is_fraud_predicted = 1 THEN transaction_amt ELSE 0 END) as fraud_amt,
            AVG(fraud_probability) as avg_probability
        FROM transactions
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn).iloc[0]


st.markdown("""
<style>
[data-testid="stHeader"] {
    background: #ff007f !important;
    background-image: none !important;
    height: 4px !important;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
[data-testid="stDecoration"] {
    background: #ff007f !important;
    background-image: none !important;
}
[data-testid="stHeader"] {
    background: transparent !important;
    border-bottom: none !important;
}
</style>
""", unsafe_allow_html=True)

def main():
    st.title("Real-Time Fraud Detection Pipeline")
    st.caption("Live transaction monitoring powered by Kafka + XGBoost + PostgreSQL")

    st.sidebar.header("Controls")
    auto_refresh = st.sidebar.checkbox("Auto-refresh (5s)", value=False)
    if auto_refresh:
        import time
        time.sleep(5)
        st.rerun()

    if st.sidebar.button("Refresh now"):
        st.cache_data.clear()
        st.rerun()

    threshold = st.sidebar.slider(
        "Fraud probability threshold", 0.0, 1.0, 0.5, 0.05
    )

    try:
        df = load_transactions()
        summary = load_summary()
    except Exception as e:
        st.error(f"Database connection error: {e}")
        st.stop()

    if df.empty:
        st.warning("No transactions yet. Run the producer and processor first.")
        st.stop()

    total = int(summary["total"])
    fraud_count = int(summary["fraud_count"])
    total_amt = float(summary["total_amt"] or 0)
    fraud_amt = float(summary["fraud_amt"] or 0)
    fraud_rate = fraud_count / total * 100 if total > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total transactions", f"{total:,}")
    col2.metric("Fraud detected", f"{fraud_count:,}", delta=f"{fraud_rate:.1f}% rate", delta_color="inverse")
    col3.metric("Total volume", f"${total_amt:,.0f}")
    col4.metric("Fraud volume", f"${fraud_amt:,.0f}", delta_color="inverse")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Fraud vs legitimate transactions")
        pie_data = pd.DataFrame({
            "Status": ["Legitimate", "Fraud"],
            "Count": [total - fraud_count, fraud_count]
        })
        fig_pie = px.pie(
            pie_data, values="Count", names="Status",
            color="Status",
            color_discrete_map={"Legitimate": "#1D9E75", "Fraud": "#E24B4A"}
        )
        fig_pie.update_layout(margin=dict(t=0, b=0))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.subheader("Transaction amount distribution")
        fig_hist = px.histogram(
            df, x="transaction_amt",
            color=df["is_fraud_predicted"].map({0: "Legitimate", 1: "Fraud"}),
            nbins=50,
            color_discrete_map={"Legitimate": "#1D9E75", "Fraud": "#E24B4A"},
            labels={"color": "Status", "transaction_amt": "Amount ($)"}
        )
        fig_hist.update_layout(margin=dict(t=0, b=0), bargap=0.1)
        st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Fraud probability distribution")
    fig_prob = px.histogram(
        df, x="fraud_probability", nbins=50,
        color_discrete_sequence=["#534AB7"],
        labels={"fraud_probability": "Fraud probability"}
    )
    fig_prob.add_vline(x=threshold, line_dash="dash", line_color="red",
                       annotation_text=f"Threshold: {threshold}")
    fig_prob.update_layout(margin=dict(t=0, b=0))
    st.plotly_chart(fig_prob, use_container_width=True)

    st.subheader("SHAP feature importance")
    try:
        shap_data = joblib.load("model/shap_data.pkl")
        shap_vals = shap_data["values"]
        feature_names = ["TransactionAmt", "ProductCD", "card4",
                         "card6", "P_emaildomain", "TransactionDT"]
        importance = pd.DataFrame({
            "Feature": feature_names,
            "Mean |SHAP|": abs(shap_vals).mean(axis=0)
        }).sort_values("Mean |SHAP|", ascending=True)

        fig_shap = px.bar(
            importance, x="Mean |SHAP|", y="Feature",
            orientation="h",
            color_discrete_sequence=["#534AB7"],
            title="Which features drive fraud predictions most?"
        )
        fig_shap.update_layout(margin=dict(t=40, b=0))
        st.plotly_chart(fig_shap, use_container_width=True)
    except Exception:
        st.info("SHAP data not available.")

    st.subheader("Recent flagged transactions")
    fraud_df = df[df["fraud_probability"] >= threshold].copy()
    fraud_df = fraud_df[[
        "transaction_id", "transaction_amt", "product_cd",
        "card4", "card6", "fraud_probability", "processed_at"
    ]].rename(columns={
        "transaction_id": "TX ID",
        "transaction_amt": "Amount ($)",
        "product_cd": "Product",
        "card4": "Card type",
        "card6": "Card category",
        "fraud_probability": "Fraud probability",
        "processed_at": "Processed at"
    })
    fraud_df["Fraud probability"] = fraud_df["Fraud probability"].apply(lambda x: f"{x:.2%}")
    st.dataframe(fraud_df, use_container_width=True, height=300)

    st.subheader("All recent transactions")
    display_df = df[[
        "transaction_id", "transaction_amt", "is_fraud_predicted",
        "fraud_probability", "processed_at"
    ]].copy()
    display_df["fraud_probability"] = display_df["fraud_probability"].apply(lambda x: f"{x:.2%}")
    display_df["is_fraud_predicted"] = display_df["is_fraud_predicted"].map({0: "Legitimate", 1: "FRAUD"})
    st.dataframe(display_df, use_container_width=True, height=300)

if __name__ == "__main__":
    main()
