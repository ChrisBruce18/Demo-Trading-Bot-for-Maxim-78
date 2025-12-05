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
TRADE_INTERVAL = cfg.get("TRADE_INTERVAL", 30)  # seconds between checks

# create logs dir if missing
import os
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# ensure log file exists
if not os.path.exists(LOG_FILE):
    pd.DataFrame(
        columns=['timestamp', 'price', 'side', 'strategy', 'confidence', 
'qty']
    ).to_csv(LOG_FILE, index=False)

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
    if len(df) < 60:
        return None, 0.0

    ma_cross_buy = (
        df["MA_short"].iloc[-2] < df["MA_long"].iloc[-2]
        and df["MA_short"].iloc[-1] > df["MA_long"].iloc[-1]
    )

    ma_cross_sell = (
        df["MA_short"].iloc[-2] > df["MA_long"].iloc[-2]
        and df["MA_short"].iloc[-1] < df["MA_long"].iloc[-1]
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

    # log trade
    pd.DataFrame(
        [[datetime.now(), price, side, strategy, float(confidence), 
float(qty)]],
        columns=['timestamp', 'price', 'side', 'strategy', 'confidence', 
'qty']
    ).to_csv(LOG_FILE, mode='a', header=False, index=False)

    # placeholder for live API call: integrate Gemini / Binance API here
    # return actual order result as needed
    return {"status": "logged", "qty": qty, "price": price, "side": side, 
"strategy": strategy}

############################
# MAIN LOOP
############################

def main():
    df = load_json_data()
    if df.empty:
        print("No data loaded. Exiting.")
        return

    df = compute_indicators(df)

    # initialize models
    ensemble_model = PredictiveEnsemble()
    kalman_model = SimpleKalman(q_level=1e-4, q_trend=1e-4, r=1.0)
    vol_estimator = EWMA_Volatility(span=20)

    # initial training
    try:
        ensemble_model.train_initial(df["close"].values)
    except Exception:
        pass

    kalman_model.initialize(df["close"].iloc[0])

    print("Live bot started. Running with historical data and API hooks.")

    while True:
        try:
            price = fetch_latest_price(df)

            # model predictions
            preds = ensemble_model.predict(df["close"].values)
            preds_list = []
            if preds is not None:
                mu_var_bayes, mu_var_sgd = preds
                preds_list.append(mu_var_bayes)
                preds_list.append(mu_var_sgd)

            # Kalman update
            level, trend, obs_var = 
kalman_model.update(df["close"].iloc[-1])
            kalman_mu = trend / max(level, 1e-8)
            kalman_var = obs_var
            preds_list.append((kalman_mu, kalman_var))

            # precision-weighted ensemble
            ensemble_mu, ensemble_var = 
precision_weighted_ensemble(preds_list)

            # volatility estimation
            ret = np.log(df["close"].iloc[-1] / df["close"].iloc[-2] + 
1e-12)
            est_vol = vol_estimator.update(ret)

            # technical signal
            tech_side, tech_conf = generate_signal(df)

            # final decision
            direction = "buy" if ensemble_mu > 0 else "sell"
            consensus = (tech_side == direction)

            confidence = max(0.0, min(1.0, 1.0 / (1.0 + ensemble_var)))
            qty = size_from_vol(0.02, est_vol, price, max_usd_alloc=1000)

            final_side = direction if consensus else None
            final_conf = confidence if consensus else 0.0

            if tech_side is not None and tech_conf > final_conf:
                final_side = tech_side
                final_conf = tech_conf

            if final_side and qty > 0:
                execute_trade(final_side, final_conf, price, 
strategy="PhD_Ensemble")
            else:
                print(
                    f"{datetime.now()} - No trade: consensus={consensus}, 
"
                    f"conf={confidence:.2f}, qty={qty:.6f}"
                )

            time.sleep(TRADE_INTERVAL)

        except KeyboardInterrupt:
            print("Stopped by user.")
            break
        except Exception as e:
            print("Runtime error:", e)
            time.sleep(5)


if __name__ == "__main__":
    main()


