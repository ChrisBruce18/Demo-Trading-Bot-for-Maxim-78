import json
import time
from datetime import datetime
import pandas as pd
import numpy as np

from strategies import (
    PredictiveEnsemble,
    SimpleKalman,
    EWMA_Volatility,
    precision_weighted_ensemble,
    size_from_vol
)

############################
# CONFIG
############################

with open("config.json", "r") as f:
    cfg = json.load(f)

SYMBOL = cfg.get("SYMBOL", "BTCUSDT")
LOG_FILE = cfg.get("LOG_FILE", "logs/trade_log.csv")
SLEEP_SECONDS = cfg.get("SLEEP_SECONDS", 5)

############################
# SETUP LOG FILE
############################

import os
if not os.path.exists("logs"):
    os.makedirs("logs")

if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=['timestamp', 'price', 'side', 'strategy', 
'confidence', 'qty']).to_csv(LOG_FILE, index=False)

############################
# LOAD HISTORICAL / LIVE DATA
############################

def load_json_data():
    """Loads the historical candles file."""
    try:
        with open("data.json", "r") as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except Exception as e:
        print("JSON load failed:", e)
        return pd.DataFrame()

def fetch_latest_price(df):
    """Returns the most recent close price."""
    return float(df["close"].iloc[-1])

############################
# INDICATORS
############################

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_indicators(df):
    df["MA_short"] = df["close"].rolling(20).mean()
    df["MA_long"] = df["close"].rolling(50).mean()
    df["EMA_10"] = df["close"].ewm(span=10).mean()
    df["EMA_30"] = df["close"].ewm(span=30).mean()
    df["RSI"] = compute_rsi(df["close"])
    df["returns"] = df["close"].pct_change()
    return df

############################
# SIGNAL GENERATION
############################

def generate_signal(df):
    """Simple MA cross + RSI filter."""
    if len(df) < 60:
        return None, 0.0

    ma_cross_buy = (
        (df["MA_short"].iloc[-2] < df["MA_long"].iloc[-2])
        and (df["MA_short"].iloc[-1] > df["MA_long"].iloc[-1])
    )

    ma_cross_sell = (
        (df["MA_short"].iloc[-2] > df["MA_long"].iloc[-2])
        and (df["MA_short"].iloc[-1] < df["MA_long"].iloc[-1])
    )

    if ma_cross_buy and df["RSI"].iloc[-1] < 70:
        return "buy", 0.65

    if ma_cross_sell and df["RSI"].iloc[-1] > 30:
        return "sell", 0.65

    return None, 0.0

############################
# ORDER EXECUTION
############################

def execute_trade(side, confidence, price, strategy):
    qty = size_from_vol(confidence, price)

    commentary = (
        f"{datetime.now()} - Predicted {side.upper()} "
        f"(conf {confidence:.2f}) qty {qty:.6f} @ {price:.2f} via 
{strategy}"
    )
    print(commentary)

    # Log to CSV
    log_df = pd.DataFrame([[datetime.now(), price, side, strategy, 
confidence, qty]],
                          columns=['timestamp', 'price', 'side', 
'strategy', 'confidence', 'qty'])
    log_df.to_csv(LOG_FILE, mode='a', header=False, index=False)

    # Placeholder for actual trade execution hook
    # Replace with API call to Gemini or other exchange
    return {
        "status": "ready_to_send",
        "side": side,
        "price": price,
        "qty": qty,
        "strategy": strategy
    }

############################
# MAIN LOOP
############################

def main():
    df = load_json_data()
    if df.empty:
        print("No data loaded. Exiting.")
        return

    df = compute_indicators(df)
    ensemble_model = PredictiveEnsemble()

    while True:
        try:
            price = fetch_latest_price(df)

            # Model predictions
            prediction, conf = ensemble_model.predict(df)

            # Technical signal
            tech_side, tech_conf = generate_signal(df)

            final_side = None
            final_conf = 0.0

            if prediction in ["buy", "sell"]:
                final_side = prediction
                final_conf = conf

            if tech_side is not None and tech_conf > final_conf:
                final_side = tech_side
                final_conf = tech_conf

            if final_side:
                execute_trade(
                    side=final_side,
                    confidence=final_conf,
                    price=price,
                    strategy="ensemble+signals"
                )

            time.sleep(SLEEP_SECONDS)

        except KeyboardInterrupt:
            print("Stopped by user.")
            break
        except Exception as e:
            print("Runtime error:", e)
            time.sleep(5)

if __name__ == "__main__":
    main()


