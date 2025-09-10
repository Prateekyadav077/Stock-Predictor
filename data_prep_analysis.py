"""data_prep_analysis.py

Fetches historical stock data from Yahoo Finance, cleans it, computes a few indicators,
and saves cleaned data to CSV.

Usage:
    python data_prep_analysis.py --ticker NKE --start 2018-01-01 --end 2024-12-31
"""
import argparse
from datetime import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import os

def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Compute RSI using a simple Wilder's smoothing approximation."""
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(window=window, min_periods=window).mean()
    ma_down = down.rolling(window=window, min_periods=window).mean()
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def fetch_and_clean(ticker: str, start: str, end: str) -> pd.DataFrame:
    # Set auto_adjust=False to keep 'Adj Close' column
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker} between {start} and {end}")
    df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Fill small gaps
    df = df.ffill().bfill()

    # Feature engineering
    df["Returns"] = df["Adj Close"].pct_change()
    df["LogRet"] = np.log(df["Adj Close"]) - np.log(df["Adj Close"].shift(1))
    df["MA_10"] = df["Adj Close"].rolling(window=10).mean()
    df["MA_50"] = df["Adj Close"].rolling(window=50).mean()
    df["STD_20"] = df["Adj Close"].rolling(window=20).std()
    df["RSI_14"] = compute_rsi(df["Adj Close"], window=14)

    df = df.dropna()
    return df


def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = df[["Adj Close", "Returns", "LogRet"]].describe()
    return stats

def save_clean_csv(df: pd.DataFrame, ticker: str, out_dir: str = "data") -> str:
    os.makedirs(out_dir, exist_ok=True)
    
    # Keep only numeric columns that LSTM will use
    numeric_cols = ["Adj Close"]  # Only keep what the model needs
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=numeric_cols)
    
    filename = os.path.join(out_dir, f"{ticker}_cleaned.csv")
    df.to_csv(filename)
    return filename


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="NKE", help="Ticker to download (default: NKE)")
    parser.add_argument("--start", default="2018-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=datetime.today().strftime("%Y-%m-%d"), help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    df = fetch_and_clean(args.ticker, args.start, args.end)
    print(f"Downloaded and cleaned {len(df)} rows for {args.ticker}")
    print(summary_stats(df))
    out = save_clean_csv(df, args.ticker)
    print(f"Saved cleaned data to {out}")

if __name__ == "__main__":
    main()
