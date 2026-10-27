import streamlit as st
import pandas as pd
from ml_pipeline import (
    CSV_FILE,
    load_trained_bundle,
    predict_any_stock,
    predict_from_manual_input,
    train_and_save_models
)

st.set_page_config(
    page_title="Stock Price Predictor | AIML Mini Project",
    page_icon="📈",
    layout="wide"
)

# --- CUSTOM STYLING FOR TOP BANNER & CARDS ---
st.markdown("""
<style>
.team-banner {
    background: linear-gradient(90deg, #1e3a8a 0%, #2563eb 100%);
    color: white;
    padding: 18px 24px;
    border-radius: 12px;
    margin-bottom: 20px;
}
.team-banner h2 { color: white !important; margin: 0 0 6px 0; font-size: 26px; }
.team-badges { display: flex; gap: 15px; flex-wrap: wrap; margin-top: 10px; }
.member-pill {
    background: rgba(255, 255, 255, 0.18);
    padding: 6px 14px;
    border-radius: 8px;
    font-weight: 600;
    font-size: 14px;
}
</style>

<div class="team-banner">
    <h2>📈 Stock Price Prediction System (Pre-Trained ML Pipeline)</h2>
    <div>AIML Mini Project — Trained on Fixed CSV Dataset | Predicts Any Stock Without Retraining</div>
    <div class="team-badges">
        <div class="member-pill">👤 Krishan Kumar (Roll No: 4333)</div>
        <div class="member-pill">👤 Lokendra Singh (Roll No: 4334)</div>
        <div class="member-pill">👤 Priyanshu Mishra (Roll No: 4345)</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR: TEAM INFO & STOCK INPUT ---
with st.sidebar:
    st.header("👨‍💻 Project Team")
    st.markdown("""
    * **Krishan Kumar** — `Roll No: 4333`
    * **Lokendra Singh** — `Roll No: 4334`
    * **Priyanshu Mishra** — `Roll No: 4345`
    """)
    st.markdown("**📘 Submitted to:** Vaishali Ingale")
    st.divider()

    st.header("🔍 Test Any Stock")
    popular_stocks = {
        "TCS (NSE)": "TCS.NS",
        "Reliance Industries (NSE)": "RELIANCE.NS",
        "Infosys (NSE)": "INFY.NS",
        "HDFC Bank (NSE)": "HDFCBANK.NS",
        "Tata Motors (NSE)": "TATAMOTORS.NS",
        "State Bank of India (NSE)": "SBIN.NS",
        "Wipro (NSE)": "WIPRO.NS",
        "ITC (NSE)": "ITC.NS",
        "Apple Inc. (NASDAQ)": "AAPL",
        "Custom Ticker...": "CUSTOM"
    }
    choice = st.selectbox("Quick Select Stock:", list(popular_stocks.keys()))
    if popular_stocks[choice] == "CUSTOM":
        ticker = st.text_input("Enter Any Yahoo Finance Ticker:", "ZOMATO.NS")
    else:
        ticker = popular_stocks[choice]

    st.caption("✅ Models are **already trained** on `stock_dataset.csv`. Selecting a stock only runs inference (`model.predict()`).")

    if st.button("🔄 Re-Train Models on CSV (Admin)", help="Only click if you edited stock_dataset.csv"):
        with st.spinner("Training models once on stock_dataset.csv..."):
            train_and_save_models()
        st.success("Models trained & saved to trained_models.pkl!")

# Load the pre-trained model bundle & static CSV
bundle = load_trained_bundle()
csv_df = pd.read_csv(CSV_FILE)

# --- 4 ORGANIZED TABS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 1. Live Stock Predictor (Any Stock)",
    "📂 2. Static Training Dataset (CSV) & EDA",
    "📊 3. Model Evaluation & GridSearchCV",
    "🧮 4. Manual Price Calculator"
])

# =====================================================================
# TAB 1: LIVE PREDICTION USING PRE-TRAINED MODELS (NO RETRAINING)
# =====================================================================
with tab1:
    st.subheader(f"Instant Prediction for `{ticker}` (Using Pre-Trained Models)")
    pred_data = predict_any_stock(ticker)

    if pred_data is None:
        st.error(f"Could not fetch recent market features for `{ticker}`. Please check the ticker symbol.")
    else:
        last_close = pred_data["last_close"]
        pred_lr = pred_data["pred_lr"]
        pred_rf = pred_data["pred_rf"]

        diff_lr = pred_lr - last_close
        diff_rf = pred_rf - last_close

        # Top Metric Cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Latest Closing Price", f"₹{last_close:,.2f}")
        c2.metric("Linear Regression Prediction", f"₹{pred_lr:,.2f}", f"{diff_lr:+.2f}")
        c3.metric("Optimized Random Forest Prediction", f"₹{pred_rf:,.2f}", f"{diff_rf:+.2f}")
        signal = "🟢 BULLISH (BUY)" if diff_rf > 0 else "🔴 BEARISH (SELL/HOLD)"
        c4.metric("AI Trend Signal", signal, f"{(diff_rf / last_close) * 100:+.2f}%")

        st.divider()

        col_chart, col_feats = st.columns([2, 1])
        with col_chart:
            st.markdown(f"**Recent Price & Moving Averages (`{ticker}`)**")
            st.line_chart(pred_data["history_df"][["Close", "MA5", "MA10"]])
        with col_feats:
            st.markdown("**Today's Input Features Passed to Model**")
            st.dataframe(pred_data["latest_row"].round(2), use_container_width=True)

# =====================================================================
# TAB 2: STATIC CSV DATASET & EXPLORATORY DATA ANALYSIS (PR5)
# =====================================================================
with tab2:
    st.subheader("📂 Fixed Training Dataset (`stock_dataset.csv`)")
    st.write(
        f"Our models were trained **once** on `{CSV_FILE}` containing **{len(csv_df)} rows** "
        f"and **{len(csv_df.columns)} columns** across benchmark stocks (`TCS.NS`, `INFY.NS`, `RELIANCE.NS`, `HDFCBANK.NS`)."
    )

    col_dl, col_info = st.columns([1, 2])
    with col_dl:
        with open(CSV_FILE, "rb") as f:
            st.download_button(
                label="⬇️ Download stock_dataset.csv",
                data=f,
                file_name="stock_dataset.csv",
                mime="text/csv",
                type="primary"
            )
    with col_info:
        st.info("💡 This physical file `stock_dataset.csv` is also saved directly in your project folder for offline inspection.")

    st.markdown("#### Preview of `stock_dataset.csv` (First 100 Rows)")
    st.dataframe(csv_df.head(100), use_container_width=True)

    st.markdown("#### Exploratory Data Analysis (EDA) — Statistical Summary")
    st.dataframe(bundle["eda_summary"], use_container_width=True)

# =====================================================================
# TAB 3: MODEL PERFORMANCE & OPTIMIZATION (PR6 & PR11)
# =====================================================================
with tab3:
    st.subheader("📊 Pre-Trained Model Evaluation (80% Train / 20% Test Split on CSV)")
    metrics_df = pd.DataFrame(bundle["metrics"]).T
    st.table(metrics_df)

    st.success(f"**Best Hyperparameters Found via GridSearchCV:** `{bundle['best_params']}`")

    st.subheader("📈 Actual vs. Predicted Prices on CSV Test Set")
    test_chart_df = pd.DataFrame({
        "Actual Close Price": bundle["y_test"],
        "Linear Regression Predicted": bundle["lr_preds"],
        "Optimized Random Forest Predicted": bundle["rf_opt_preds"]
    })
    st.line_chart(test_chart_df)

# =====================================================================
# TAB 4: MANUAL PRICE CALCULATOR (TEST ANY CUSTOM NUMBERS)
# =====================================================================
with tab4:
    st.subheader("🧮 Predict Any Stock Manually (Custom Feature Inputs)")
    st.write("Enter any stock's figures below to test the pre-trained models without internet:")

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        m_open = st.number_input("Open Price (₹)", value=1500.0)
        m_high = st.number_input("High Price (₹)", value=1530.0)
    with mc2:
        m_low = st.number_input("Low Price (₹)", value=1490.0)
        m_close = st.number_input("Close Price (₹)", value=1520.0)
    with mc3:
        m_ma5 = st.number_input("5-Day Moving Avg (MA5)", value=1510.0)
        m_ma10 = st.number_input("10-Day Moving Avg (MA10)", value=1500.0)
        m_vol = st.number_input("Volume", value=2500000.0)

    if st.button("Predict Next Day Close", type="primary"):
        p_lr, p_rf = predict_from_manual_input(m_open, m_high, m_low, m_close, m_vol, m_ma5, m_ma10)
        rc1, rc2 = st.columns(2)
        rc1.success(f"**Linear Regression Prediction:** ₹{p_lr:,.2f}")
        rc2.success(f"**Optimized Random Forest Prediction:** ₹{p_rf:,.2f}")