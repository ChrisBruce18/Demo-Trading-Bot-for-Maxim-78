import json
import os
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from strategies import (
    PredictiveEnsemble,
    SimpleKalman,
    EWMA_Volatility,
    precision_weighted_ensemble,
    size_from_vol
)
from dashboard import update_dashboard  # assuming you have this module

############################
# CONFIG
############################

with open("config.json", "r") as f:
    cfg = json.load(f)

LOG_FILE = cfg.get("LOG_FILE", "logs/trade_log.csv")
SYMBOL = cfg.get("SYMBOL", "BTC/USDT")
TRADE_INTERVAL = cfg.get("TRADE_INTERVAL", 30)  # seconds between checks
MAX_USD_ALLOC = cfg.get("MAX_USD_ALLOC", 1000)
TARGET_VOL = cfg.get("TARGET_VOL", 0.02)

# create logs dir
if not os.path.exists("logs"):
    os.makedirs("logs")

# ensure log header exists
if not os.path.exists(LOG_FILE):
    pd.DataFrame(
        columns=['timestamp','price','side','strategy','confidence','qty']
    ).to_csv(LOG_FILE, index=False)

############################
# DATA LOADING
############################

def load_json_data():
    """Load historical candle data from JSON."""
    try:
        with open("data.json", "r") as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except Exception as e:
        print("JSON load failed:", e)
        return pd.DataFrame()

def fetch_latest_price(df):
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
    df["signal"] = None
    return df

############################
# SIGNAL GENERATION
############################

def ma_crossover_signal(df, short=5, long=20):
    if len(df) < long + 2:
        return None
    df = df.copy()
    df['MA_short'] = df['close'].rolling(short).mean()
    df['MA_long'] = df['close'].rolling(long).mean()
    if df['MA_short'].iloc[-2] < df['MA_long'].iloc[-2] and 
df['MA_short'].iloc[-1] > df['MA_long'].iloc[-1]:
        return 'buy'
    elif df['MA_short'].iloc[-2] > df['MA_long'].iloc[-2] and 
df['MA_short'].iloc[-1] < df['MA_long'].iloc[-1]:
        return 'sell'
    return None

############################
# ORDER EXECUTION
############################

def execute_trade(side, price, strategy, confidence, qty):
    commentary = (
        f"{datetime.now()} - Predicted {side.upper()} "
        f"(conf {confidence:.2f}) qty {qty:.6f} @ {price:.2f} via 
{strategy}"
    )
    print(commentary)

    log = pd.DataFrame([[datetime.now(), price, side, strategy, 
float(confidence), float(qty)]],
                       
columns=['timestamp','price','side','strategy','confidence','qty'])
    log.to_csv(LOG_FILE, mode='a', header=False, index=False)

    try:
        # Placeholder for actual live API call
        print("Live trade executed (replace with API call).")
        return {"status": "live_order_sent"}
    except Exception as e:
        print("Trade execution error:", str(e))
        return None

############################
# MAIN LOOP
############################

def main():
    df_master = load_json_data()
    if df_master.empty:
        print("No data loaded. Exiting.")
        return

    df_master = compute_indicators(df_master)

    # initialize models
    ensemble = PredictiveEnsemble(window=40)
    kalman = SimpleKalman(q_level=1e-4, q_trend=1e-4, r=1.0)
    vol_est = EWMA_Volatility(span=20)

    try:
        ensemble.train_initial(df_master['close'].values)
    except Exception:
        pass
    kalman.initialize(df_master['close'].iloc[0])

    print("Live trading bot started.")

    while True:
        try:
            last_price = float(df_master['close'].iloc[-1])
            price_now = last_price  # Replace with real-time fetch from 
API

            # MA signal
            ma_sig = ma_crossover_signal(df_master)

            # ensemble prediction
            ensemble.online_update(df_master['close'].values)
            preds = ensemble.predict(df_master['close'].values)
            preds_list = []
            if preds is not None:
                mu_var_bayes, mu_var_sgd = preds
                preds_list.append(mu_var_bayes)
                preds_list.append(mu_var_sgd)

            # Kalman update
            level, trend, obs_var = 
kalman.update(df_master['close'].iloc[-1])
            kalman_mu = trend / max(level, 1e-8)
            kalman_var = obs_var
            preds_list.append((kalman_mu, kalman_var))

            # ensemble aggregation
            ensemble_mu, ensemble_var = 
precision_weighted_ensemble(preds_list)

            # estimated volatility
            ret = np.log(df_master['close'].iloc[-1] / 
df_master['close'].iloc[-2] + 1e-12)
            est_vol = vol_est.update(ret)

            # direction & consensus
            direction = 'buy' if ensemble_mu > 0 else 'sell'
            consensus = (ma_sig == direction)

            # confidence mapping
            confidence = max(0.0, min(1.0, 1.0 / (1.0 + ensemble_var)))

            # position sizing
            qty = size_from_vol(TARGET_VOL, est_vol, price_now, 
max_usd_alloc=MAX_USD_ALLOC)

            final_signal = None
            commentary = None
            if consensus and confidence > 0.25 and qty > 0:
                final_signal = direction
                commentary = execute_trade(final_signal, price_now, 
'PhD_Ensemble', confidence, qty)
            else:
                commentary = f"{datetime.now()} - No trade: 
consensus={consensus}, conf={confidence:.2f}, qty={qty:.6f}"

            df_master.at[df_master.index[-1], 'signal'] = final_signal

            # update dashboard
            update_dashboard(df_master, commentary)

            time.sleep(TRADE_INTERVAL)

        except KeyboardInterrupt:
            print("Stopped by user.")
            break
        except Exception as e:
            print("Runtime error:", e)
            time.sleep(5)

if __name__ == "__main__":
    main()

