import streamlit as st
from datetime import date
import pandas as pd
import matplotlib.pyplot as plt
import os
from lstm_model import create_sequences, predict_future, load_saved, get_test_predictions

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="Nike Stock Predictor",
    layout="wide",
    page_icon="👟",
)

# ----------------- CUSTOM CSS -----------------
st.markdown("""
    <style>
        /* Background & font */
        body {
            background-color: #0D0D0D;
            color: #FFFFFF;
            font-family: 'Montserrat', sans-serif;
        }
        /* Title */
        .title-logo {
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 30px;
        }
        .title-logo img {
            margin-right: 15px;
            height: 70px;
        }
        h1 {
            font-weight: 800;
            background: linear-gradient(90deg, #FF3B3B, #E62020);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #111111 !important;
            padding: 20px;
        }
        /* Buttons */
        .stButton>button {
            background: linear-gradient(135deg, #E62020, #FF3B3B);
            color: white;
            border-radius: 12px;
            padding: 0.6em 1.2em;
            font-weight: bold;
            border: none;
            transition: 0.3s;
        }
        .stButton>button:hover {
            transform: scale(1.05);
            background: linear-gradient(135deg, #FF5555, #E62020);
        }
        /* Tables */
        .stDataFrame div[data-testid="stDataFrameContainer"] {
            background-color: #1A1A1A;
            color: white;
            border-radius: 12px;
        }
        /* Metrics */
        [data-testid="stMetricValue"] {
            color: #FF3B3B;
            font-size: 22px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------- HEADER -----------------
# ----------------- HEADER -----------------
st.markdown("""
<div class="title-logo">
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a6/Logo_NIKE.svg/512px-Logo_NIKE.svg.png" style="height:70px;">
    <h1>Nike Stock Predictor</h1>
</div>
""", unsafe_allow_html=True)


ticker = "NKE"
st.markdown(f"**Ticker:** `{ticker}`")

# ----------------- SIDEBAR INPUTS -----------------
st.sidebar.header("⚙️ Configuration")
time_steps = st.sidebar.slider(
    "⏳ Time steps (sequence length)",
    min_value=10, max_value=200, value=60
)
future_days = st.sidebar.slider(
    "📈 Days to predict (business days)",
    min_value=1, max_value=90, value=30
)

# ----------------- MAIN TABS -----------------
tab1, tab2 = st.tabs(["📊 Historical Prediction", "🔮 Future Prediction"])

# ----------------- HISTORICAL PREDICTION -----------------
with tab1:
    st.subheader("Historical Model Performance")
    if st.button("Show Trained vs Predicted Chart", key="hist_chart"):
        try:
            dates, actual, predicted = get_test_predictions(ticker, time_steps=time_steps)
        except FileNotFoundError as e:
            st.error(f"{e}\nPlease run data_prep_analysis.py and train the model first.")
        else:
            fig, ax = plt.subplots(figsize=(14, 7))
            ax.plot(dates, actual, label="Actual", color="#FFFFFF", linewidth=2)
            ax.plot(dates, predicted, linestyle="--", label="Predicted", color="#E62020", linewidth=2)
            ax.set_facecolor("#0D0D0D")
            ax.set_title(f"{ticker} — Trained vs Predicted Prices", color="#FFFFFF", fontsize=16)
            ax.set_xlabel("Date", color="#FFFFFF")
            ax.set_ylabel("Adjusted Close Price", color="#FFFFFF")
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.legend(facecolor="#1A1A1A")
            st.pyplot(fig)

            mse = ((actual - predicted) ** 2).mean()
            rmse = mse ** 0.5
            st.metric("📉 RMSE on Historical Data", f"{rmse:.4f}")

# ----------------- FUTURE PREDICTIONS -----------------
with tab2:
    st.subheader("Future Price Forecast")
    if st.button("Predict Future with Saved Model", key="future_pred"):
        try:
            model, scaler = load_saved(ticker)
        except FileNotFoundError as e:
            st.error(f"{e}\nPlease run data_prep_analysis.py and train the model first.")
        else:
            df = pd.read_csv(os.path.join("data", f"{ticker}_cleaned.csv"), index_col=0, parse_dates=True)
            prices = df["Adj Close"].values.reshape(-1, 1)
            scaled = scaler.transform(prices)
            X, y = create_sequences(scaled, time_steps=time_steps)

            last_seq = X[-1]
            future_pred = predict_future(model, last_seq, n_steps=int(future_days), scaler=scaler)

            last_date = df.index[-1]
            future_dates = []
            current = last_date
            while len(future_dates) < int(future_days):
                current += pd.Timedelta(days=1)
                if current.weekday() < 5:
                    future_dates.append(current)

            # Plot chart
            fig, ax = plt.subplots(figsize=(14, 7))
            ax.plot(df.index, df["Adj Close"], label="Actual", color="#FFFFFF", linewidth=2)
            ax.plot(pd.to_datetime(future_dates), future_pred.flatten(), linestyle='--',
                    label="Future Prediction", color="#E62020", linewidth=2)
            ax.set_facecolor("#0D0D0D")
            ax.set_title(f"{ticker} — Actual vs Future Prediction", color="#FFFFFF", fontsize=16)
            ax.set_xlabel("Date", color="#FFFFFF")
            ax.set_ylabel("Adjusted Close Price", color="#FFFFFF")
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.legend(facecolor="#1A1A1A")
            st.pyplot(fig)

            out_df = pd.DataFrame({"Date": future_dates, "Predicted_Close": future_pred.flatten()})
            out_df["Date"] = pd.to_datetime(out_df["Date"]).dt.date

            st.markdown("### 📊 Future Predictions Table")
            st.dataframe(out_df, use_container_width=True)

# ----------------- FOOTER -----------------
st.markdown("---")
with st.expander("ℹ️ Notes"):
    st.markdown("""
    - This app uses preprocessed **Nike (NKE)** stock data only.  
    - The LSTM model is loaded from the `models/` folder.  
    - Predictions are for educational & visualization purposes, **not financial advice**.  
    """)
