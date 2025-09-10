"""Streamlit app to run data download, show EDA, train LSTM and show future predictions.
Run with: streamlit run app.py
"""

import streamlit as st
from datetime import date
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from data_prep_analysis import fetch_and_clean, save_clean_csv
from lstm_model import create_sequences, build_model, predict_future, train_and_save, load_saved

st.set_page_config(page_title="Stock LSTM Demo", layout="wide")

st.title("Stock Prediction — LSTM demo")

col1, col2 = st.columns([2, 1])
with col1:
    ticker = st.text_input("Ticker (Yahoo)", value="NKE")
    start_date = st.date_input("Start date", value=date(2018, 1, 1))
    end_date = st.date_input("End date", value=date.today())
    time_steps = st.number_input("Time steps (sequence length)", min_value=10, max_value=200, value=60)
    epochs = st.number_input("Epochs", min_value=1, max_value=200, value=10)
    future_days = st.number_input("Days to predict (business days)", min_value=1, max_value=90, value=30)

with col2:
    st.write("Model training will use the downloaded adjusted close prices.\nIf you already ran data_prep_analysis.py and trained the model, the app will attempt to load model files from the `models/` folder.")

if st.button("Download & Prepare Data"):
    with st.spinner("Downloading and preparing data..."):
        df = fetch_and_clean(ticker.upper(), start_date.isoformat(), end_date.isoformat())
        out = save_clean_csv(df, ticker.upper())
        st.success(f"Data saved to {out} — {len(df)} rows")
        st.dataframe(df.tail())

if st.button("Train LSTM (quick)"):
    with st.spinner("Training model — this may take a while locally"):
        df = fetch_and_clean(ticker.upper(), start_date.isoformat(), end_date.isoformat())
        prices = df["Adj Close"].values.reshape(-1, 1)
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        s = scaler.fit_transform(prices)
        X, y = create_sequences(s, time_steps=time_steps)
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        model = build_model(time_steps=time_steps)
        model.fit(X_train, y_train, epochs=int(epochs), batch_size=32, validation_split=0.1, verbose=1)
        st.success("Training completed (session model only). You can now predict future using this session model.")

        last_seq = X[-1]
        future_pred = predict_future(model, last_seq, n_steps=int(future_days), scaler=scaler)

        last_date = df.index[-1]
        future_dates = []
        current = last_date
        while len(future_dates) < int(future_days):
            current = current + pd.Timedelta(days=1)
            if current.weekday() < 5:
                future_dates.append(current)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df.index, df["Adj Close"], label="Actual")
        ax.plot(pd.to_datetime(future_dates), future_pred.flatten(), linestyle='--', label="Future Pred")
        ax.set_title(f"{ticker.upper()} — Actual vs Future Pred")
        ax.legend()
        st.pyplot(fig)

        out_df = pd.DataFrame({"Date": future_dates, "Predicted_Close": future_pred.flatten()})
        out_df["Date"] = pd.to_datetime(out_df["Date"]).dt.date
        st.dataframe(out_df)

if st.button("Load Saved Model & Predict"):
    try:
        model, scaler = load_saved(ticker.upper())
    except FileNotFoundError as e:
        st.error(str(e))
    else:
        df = pd.read_csv(os.path.join("data", f"{ticker.upper()}_cleaned.csv"), index_col=0, parse_dates=True)
        prices = df["Adj Close"].values.reshape(-1, 1)
        scaled = scaler.transform(prices)
        X, y = create_sequences(scaled, time_steps=time_steps)
        last_seq = X[-1]
        future_pred = predict_future(model, last_seq, n_steps=int(future_days), scaler=scaler)

        last_date = df.index[-1]
        future_dates = []
        current = last_date
        while len(future_dates) < int(future_days):
            current = current + pd.Timedelta(days=1)
            if current.weekday() < 5:
                future_dates.append(current)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df.index, df["Adj Close"], label="Actual")
        ax.plot(pd.to_datetime(future_dates), future_pred.flatten(), linestyle='--', label="Future Pred (saved model)")
        ax.set_title(f"{ticker.upper()} — Actual vs Future Pred (saved model)")
        ax.legend()
        st.pyplot(fig)

        out_df = pd.DataFrame({"Date": future_dates, "Predicted_Close": future_pred.flatten()})
        out_df["Date"] = pd.to_datetime(out_df["Date"]).dt.date
        st.dataframe(out_df)

st.markdown("---")
st.markdown("**Notes:** This demo trains a local LSTM on adjusted close prices only. For production: tune hyperparameters, add features, keep a held-out test set, and store models to disk.")
