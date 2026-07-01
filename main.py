"""
FBI Time Series Forecasting
============================
Forecasting Facebook (FB / META) stock closing prices using a stacked
LSTM (Long Short-Term Memory) network.

Pipeline stages:
    1. Data Collection    -> load_data()
    2. Preprocessing      -> preprocess_data()
    3. Model Building     -> build_lstm_model()
    4. Training            -> train_model()
    5. Evaluation           -> evaluate_model()
    6. Visualization        -> plot_predictions()

Run:
    python main.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TICKER = "META"          # Facebook is now listed as META on NASDAQ
DATA_PATH = "data/FB.csv"  # local CSV fallback (Date, Open, High, Low, Close, Volume)
SEQUENCE_LENGTH = 60      # number of past days used to predict the next day
TRAIN_SPLIT = 0.8
EPOCHS = 25
BATCH_SIZE = 32
OUTPUT_DIR = "outputs"


# ---------------------------------------------------------------------------
# 1. Data Collection
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """
    Loads historical stock data.
    Priority: local CSV (data/FB.csv) if present, else download via yfinance.
    Expected columns: Date, Open, High, Low, Close, Volume
    """
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
        print(f"Loaded local dataset: {DATA_PATH} ({len(df)} rows)")
    else:
        import yfinance as yf
        print(f"Local dataset not found. Downloading {TICKER} data from Yahoo Finance...")
        df = yf.download(TICKER, start="2015-01-01", auto_adjust=True)
        df.reset_index(inplace=True)
        os.makedirs("data", exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
        print(f"Downloaded and cached to {DATA_PATH} ({len(df)} rows)")

    df = df.sort_values("Date").reset_index(drop=True)
    df = df.dropna()
    return df


# ---------------------------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------------------------
def create_sequences(data: np.ndarray, seq_len: int):
    """Converts a 1-D scaled price array into supervised (X, y) sequences."""
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def preprocess_data(df: pd.DataFrame):
    """
    - Uses the 'Close' price column.
    - Scales values to [0, 1] using MinMaxScaler.
    - Splits into train/test sets.
    - Builds sliding-window sequences for the LSTM input.
    """
    close_prices = df[["Close"]].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    split_idx = int(len(scaled_data) * TRAIN_SPLIT)
    train_data = scaled_data[:split_idx]
    # include the last SEQUENCE_LENGTH points of train in test so the first
    # test sequence has enough history
    test_data = scaled_data[split_idx - SEQUENCE_LENGTH:]

    X_train, y_train = create_sequences(train_data, SEQUENCE_LENGTH)
    X_test, y_test = create_sequences(test_data, SEQUENCE_LENGTH)

    # LSTM expects input shape: (samples, timesteps, features)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    print(f"Train sequences: {X_train.shape}, Test sequences: {X_test.shape}")
    return X_train, y_train, X_test, y_test, scaler, split_idx


# ---------------------------------------------------------------------------
# 3. Model Building
# ---------------------------------------------------------------------------
def build_lstm_model(input_shape) -> Sequential:
    """
    Stacked LSTM architecture:
        LSTM(50, return_sequences=True) -> Dropout
        LSTM(50, return_sequences=False) -> Dropout
        Dense(25) -> Dense(1)
    """
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.summary()
    return model


# ---------------------------------------------------------------------------
# 4. Training
# ---------------------------------------------------------------------------
def train_model(model, X_train, y_train):
    early_stop = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1
    )
    return history


# ---------------------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------------------
def evaluate_model(model, X_test, y_test, scaler):
    """
    Predicts on the test set, inverse-transforms back to real price scale,
    and computes RMSE, MAE, and R^2.
    """
    predictions_scaled = model.predict(X_test)

    predictions = scaler.inverse_transform(predictions_scaled)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
    mae = mean_absolute_error(y_test_actual, predictions)
    r2 = r2_score(y_test_actual, predictions)

    print("\n--- Evaluation Metrics ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"R^2 : {r2:.4f}")

    return predictions, y_test_actual, {"RMSE": rmse, "MAE": mae, "R2": r2}


# ---------------------------------------------------------------------------
# 6. Visualization
# ---------------------------------------------------------------------------
def plot_predictions(df, split_idx, y_test_actual, predictions):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dates = df["Date"].values[split_idx:]
    dates = dates[:len(y_test_actual)]

    plt.figure(figsize=(12, 6))
    plt.plot(dates, y_test_actual, label="Actual Close Price", color="steelblue")
    plt.plot(dates, predictions, label="Predicted Close Price", color="orangered")
    plt.title(f"{TICKER} Stock Price Prediction (LSTM)")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "prediction_plot.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved prediction plot to {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    df = load_data()
    X_train, y_train, X_test, y_test, scaler, split_idx = preprocess_data(df)

    model = build_lstm_model(input_shape=(X_train.shape[1], 1))
    train_model(model, X_train, y_train)

    predictions, y_test_actual, metrics = evaluate_model(model, X_test, y_test, scaler)
    plot_predictions(df, split_idx, y_test_actual, predictions)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save(os.path.join(OUTPUT_DIR, "lstm_model.h5"))
    print(f"Model saved to {OUTPUT_DIR}/lstm_model.h5")


if __name__ == "__main__":
    main()
