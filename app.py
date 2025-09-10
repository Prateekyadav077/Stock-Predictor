"""
Streamlit app for Nike Stock Predictor — LSTM demo
Uses preprocessed/trained data and saved model for predictions.
Run with: streamlit run app.py
"""

import streamlit as st
from datetime import date
import pandas as pd
import matplotlib.pyplot as plt
import os

from lstm_model import create_sequences, predict_future, load_saved, get_test_predictions

st.set_page_config(page_title="Nike Stock Predictor", layout="wide")

# ----------------- APP TITLE -----------------
st.title("Nike Stock Predictor")
ticker = "NKE"
st.markdown(f"**Ticker:** {ticker}")

# ----------------- USER INPUTS -----------------
time_steps = st.number_input("Time steps (sequence length)", min_value=10, max_value=200, value=60)
future_days = st.number_input("Days to predict (business days)", min_value=1, max_value=90, value=30)

# ----------------- HISTORICAL PREDICTION -----------------
if st.button("Show Trained vs Predicted Chart"):
    try:
        dates, actual, predicted = get_test_predictions(ticker, time_steps=time_steps)
    except FileNotFoundError as e:
        st.error(f"{e}\nPlease run data_prep_analysis.py and train the model first.")
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(dates, actual, label="Actual (Training/Test Data)")
        ax.plot(dates, predicted, linestyle="--", label="Predicted (Model)")
        ax.set_title(f"{ticker} — Trained vs Predicted Prices")
        ax.set_xlabel("Date")
        ax.set_ylabel("Adjusted Close Price")
        ax.legend()
        st.pyplot(fig)

        # Calculate simple accuracy metrics
        mse = ((actual - predicted) ** 2).mean()
        rmse = mse ** 0.5
        st.markdown(f"**RMSE on historical data:** {rmse:.4f}")

# ----------------- FUTURE PREDICTIONS -----------------
if st.button("Predict Future with Saved Model"):
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

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df.index, df["Adj Close"], label="Actual (Training Data)")
        ax.plot(pd.to_datetime(future_dates), future_pred.flatten(), linestyle='--', label="Future Prediction")
        ax.set_title(f"{ticker} — Actual vs Future Pred")
        ax.set_xlabel("Date")
        ax.set_ylabel("Adjusted Close Price")
        ax.legend()
        st.pyplot(fig)

        out_df = pd.DataFrame({"Date": future_dates, "Predicted_Close": future_pred.flatten()})
        out_df["Date"] = pd.to_datetime(out_df["Date"]).dt.date
        st.subheader("Future Predictions")
        st.dataframe(out_df)

st.markdown("---")
st.markdown(
    "**Notes:** This app uses preprocessed NKE data only. The LSTM model is loaded from the `models/` folder. "
    "Future predictions are based on the saved model."
)
