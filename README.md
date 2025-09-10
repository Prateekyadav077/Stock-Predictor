# Stock Prediction Project (Nike example)

This repo contains four files demonstrating a simple pipeline: download & clean data, train an LSTM, a Streamlit front-end, and this README.

## Files created
- `data_prep_analysis.py` — download from Yahoo Finance (yfinance), clean and save cleaned CSV to `data/`.
- `lstm_model.py` — build/train an LSTM from the cleaned CSV and save model+scaler to `models/`.
- `app.py` — Streamlit demo to run the pipeline locally and visualize predictions.
- `README.md` — this file.

## Quick start

1. Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
# or
pip install yfinance pandas numpy scikit-learn matplotlib streamlit tensorflow joblib
```

2. Prepare data (example for Nike):

```bash
python data_prep_analysis.py --ticker NKE --start 2018-01-01 --end 2024-12-31
```

3. Train model (quick):

```bash
python lstm_model.py --ticker NKE --time_steps 60 --epochs 20
```

4. Run the Streamlit app:

```bash
streamlit run app.py
```

## Notes and security
- The project is a demo: it uses only adjusted close prices. For better predictions add features (volume, indicators, external data), tune hyperparameters and evaluate properly.
- Do not commit any private tokens (ngrok tokens, API keys) to public repos.
- Training requires TensorFlow which may need a GPU for faster runs. The app trains models locally if you press the training button.
