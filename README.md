# FBI Time Series Forecasting

Forecasting Facebook (FB / META) stock closing prices using a stacked LSTM
(Long Short-Term Memory) network, based on the project specification
outlined in `FBI Time series forecasting.pptx`.

## 1. Objective

- Analyze historical FB/META stock trends.
- Build a deep learning (LSTM) model to forecast the closing price.
- Evaluate model performance using standard regression metrics.
- Visualize actual vs. predicted prices.

## 2. Dataset

| Field  | Description                          |
|--------|---------------------------------------|
| Date   | Trading date                          |
| Open   | Opening price                         |
| High   | Highest price of the day              |
| Low    | Lowest price of the day               |
| Close  | Closing price (target variable)       |
| Volume | Number of shares traded               |

**Source:** Yahoo Finance (fetched automatically via the `yfinance` library
if `data/FB.csv` is not already present). Facebook Inc. now trades under
the ticker `META`; this is configured in `main.py` via the `TICKER`
variable.

If you have your own dataset (e.g., a Kaggle CSV), place it at
`data/FB.csv` with the columns above and the script will use it directly
instead of downloading.

## 3. Methodology

1. **Data Collection** — Load CSV or download via Yahoo Finance API.
2. **Preprocessing**
   - Use the `Close` price column.
   - Scale values to the range [0, 1] using `MinMaxScaler`.
   - Convert the scaled series into supervised sequences using a sliding
     window of 60 trading days (`SEQUENCE_LENGTH = 60`) to predict day 61.
   - Split chronologically into 80% train / 20% test (no shuffling, since
     order matters in time series).
3. **Model Architecture** (stacked LSTM)

   ```
   LSTM(50, return_sequences=True)
   Dropout(0.2)
   LSTM(50, return_sequences=False)
   Dropout(0.2)
   Dense(25, activation='relu')
   Dense(1)
   ```

   Optimizer: Adam · Loss: Mean Squared Error

4. **Training** — 25 epochs, batch size 32, with `EarlyStopping` on
   training loss (patience = 5) to avoid unnecessary overfitting.
5. **Evaluation** — Predictions are inverse-transformed back to the
   original price scale and compared against actual prices using:
   - **RMSE** (Root Mean Squared Error)
   - **MAE** (Mean Absolute Error)
   - **R² Score**
6. **Visualization** — A line plot of actual vs. predicted closing prices
   is saved to `outputs/prediction_plot.png`.

## 4. Project Structure

```
FBI_Time_Series_Forecasting/
├── data/
│   └── FB.csv              # dataset (auto-downloaded on first run if absent)
├── outputs/
│   ├── lstm_model.h5       # trained model (generated after running)
│   └── prediction_plot.png # actual vs predicted plot (generated after running)
├── main.py                 # complete pipeline: load -> preprocess -> train -> evaluate -> plot
├── requirements.txt        # Python dependencies
└── README.md
```

## 5. How to Run

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the pipeline
python main.py
```

Console output will show training progress and final metrics
(RMSE, MAE, R²). The trained model and prediction plot are saved to the
`outputs/` directory.

## 6. Notes on Design Choices

- The pipeline is intentionally kept in a **single script** (`main.py`)
  with clearly separated functions per stage, rather than split across
  multiple modules, to keep the project simple to run and review.
- `EarlyStopping` is used instead of a fixed epoch count alone, which is a
  standard practice worth mentioning in a placement interview when
  discussing how you prevent overfitting in a training loop.
- The 80/20 split is **chronological, not random** — this is essential in
  time series problems to avoid data leakage (the model must never see
  future data during training).

## 7. Possible Extensions

- Compare LSTM against a stacked RNN or GRU baseline.
- Add technical indicators (moving averages, RSI, MACD) as additional
  input features instead of using `Close` price alone.
- Deploy the trained model with a simple Streamlit frontend for
  interactive forecasting (similar to the TSLA prediction project).
