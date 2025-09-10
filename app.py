import streamlit as st
from datetime import date
import pandas as pd
import matplotlib.pyplot as plt
import os

from lstm_model import create_sequences, predict_future, load_saved, get_test_predictions

# ----------------- PAGE CONFIG -----------------
st.set_page_config(page_title="Nike Stock Predictor", layout="wide", page_icon=":nike:")

# ----------------- CUSTOM CSS -----------------
st.markdown("""
    <style>
        body {
            background-color: #111111;
            color: #FFFFFF;
            font-family: 'Arial', sans-serif;
        }
        .stButton>button {
            background-color: #E62020;
            color: white;
            border-radius: 10px;
            padding: 0.5em 1em;
            font-weight: bold;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #FF3B3B;
            color: white;
        }
        .stDataFrame div[data-testid="stDataFrameContainer"] {
            background-color: #222222;
            color: white;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #FFFFFF;
        }
        .title-logo {
            display: flex;
            align-items: center;
            margin-bottom: 20px;
        }
        .title-logo img {
            margin-right: 15px;
            height: 60px;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------- HEADER WITH LOGO -----------------
st.markdown("""
<div class="title-logo">
    <img src="https://upload.wikimedia.org/wikipedia/commons/a/a6/Logo_NIKE.svg" width="80">
    <h1>Nike Stock Predictor</h1>
</div>
""", unsafe_allow_html=True)

ticker = "NKE"
st.markdown(f"**Ticker:** {ticker}")

# ----------------- USER INPUTS -----------------
col1, col2 = st.columns([1, 2])

with col1:
    time_steps = st.number_input(
        "⏳ Time steps (sequence length)", 
        min_value=10, max_value=200, value=60,
        help="Number of previous days model will use to predict the next day"
    )
    future_days = st.number_input(
        "📈 Days to predict (business days)", 
        min_value=1, max_value=90, value=30,
        help="How many business days into the future to predict"
    )

# ----------------- TABS -----------------
tab1, tab2 = st.tabs(["Historical Prediction", "Future Prediction"])

# ----------------- HISTORICAL PREDICTION -----------------
with tab1:
    if st.button("Show Trained vs Predicted Chart", key="hist_chart"):
        try:
            dates, actual, predicted = get_test_predictions(ticker, time_steps=time_steps)
        except FileNotFoundError as e:
            st.error(f"{e}\nPlease run data_prep_analysis.py and train the model first.")
        else:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(dates, actual, label="Actual", color="#FFFFFF")
            ax.plot(dates, predicted, linestyle="--", label="Predicted", color="#E62020")
            ax.set_facecolor("#111111")
            ax.set_title(f"{ticker} — Trained vs Predicted Prices", color="#FFFFFF")
            ax.set_xlabel("Date", color="#FFFFFF")
            ax.set_ylabel("Adjusted Close Price", color="#FFFFFF")
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.legend(facecolor="#222222")
            st.pyplot(fig)

            mse = ((actual - predicted) ** 2).mean()
            rmse = mse ** 0.5
            st.markdown(f"**RMSE on historical data:** <span style='color:#E62020;'>{rmse:.4f}</span>", unsafe_allow_html=True)

# ----------------- FUTURE PREDICTIONS -----------------
with tab2:
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

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df.index, df["Adj Close"], label="Actual", color="#FFFFFF")
            ax.plot(pd.to_datetime(future_dates), future_pred.flatten(), linestyle='--', label="Future Prediction", color="#E62020")
            ax.set_facecolor("#111111")
            ax.set_title(f"{ticker} — Actual vs Future Prediction", color="#FFFFFF")
            ax.set_xlabel("Date", color="#FFFFFF")
            ax.set_ylabel("Adjusted Close Price", color="#FFFFFF")
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.legend(facecolor="#222222")
            st.pyplot(fig)

            out_df = pd.DataFrame({"Date": future_dates, "Predicted_Close": future_pred.flatten()})
            out_df["Date"] = pd.to_datetime(out_df["Date"]).dt.date
            st.subheader("📊 Future Predictions")
            st.dataframe(out_df, use_container_width=True)

st.markdown("---")
st.markdown(
    "**Notes:** This app uses preprocessed NKE data only. The LSTM model is loaded from the `models/` folder. "
    "Future predictions are based on the saved model."
)
