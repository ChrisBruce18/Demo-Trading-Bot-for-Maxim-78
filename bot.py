import json
import time
from datetime import datetime, timedelta
import os
import numpy as np
import pandas as pd

from strategies import (
    PredictiveEnsemble,
    SimpleKalman,
    EWMA_Volatility,
    precision_weighted_ensemble,
    size_from_vol
)
from dashboard import update_dashboard

############################
# CONFIG
############################

with open("config.json", "r") as f:
    cfg = json.load(f)

DEMO_MODE = cfg.get("DEMO_MODE", True)
SYMBOL = cfg.get("SYMBOL", "BTCUSDT")
TIMEFRAME = cfg.get("TIMEFRAME", "1m")
LIMIT = cfg.get("LIMIT", 200)
TRADE_INTERVAL = cfg.get("TRADE_INTERVAL", 30)  # seconds
LOG_FILE = cfg.get("LOG_FILE", "logs/trade_log.csv")

# Ensure logs directory exists
if not os.path.exists("logs"):
    os.makedirs("logs")

# Ensure CSV header
if not os.path.exists(LOG_FILE):
    
pd.DataFrame(columns=['timestamp','price','side','strategy','confidence','qty']).to_csv(LOG_FILE, 
index=False)

############################
# LOAD HISTORICAL / DEMO DATA
############################

def load_json_data():
    try:
        with open("data.json", "r") as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except Exception as e:
        print("JSON load failed:", e)
        return pd.DataFrame()

def fetch_latest_price(df):
    return float(df["close"].iloc[-1])

def generate_demo_data(length=LIMIT, start_price=50000.0):
    rng = np.random.default_rng(seed=42)
    price = start_price + np.cumsum(rng.normal(scale=40.0, size=length))
    timestamps = [datetime.now() - 
timedelta(seconds=TRADE_INTERVAL*(length - i)) for i in range(length)]
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': price + rng.normal(scale=5.0, size=length),
        'high': price + rng.normal(scale=10.0, size=length),
        'low': price - rng.normal(scale=10.0, size=length),
        'close': price,
        'volume': rng.integers(1, 10, size=length)
    })
    return df

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

    ma_cross_buy = df["MA_short"].iloc[-2] < df["MA_long"].iloc[-2] and 
df["MA_short"].iloc[-1] > df["MA_long"].iloc[-1]
    ma_cross_sell = df["MA_short"].iloc[-2] > df["MA_long"].iloc[-2] and 
df["MA_short"].iloc[-1] < df["MA_long"].iloc[-1]

    if ma_cross_buy and df["RSI"].iloc[-1] < 70:
        return "buy", 0.65
    if ma_cross_sell and df["RSI"].iloc[-1] > 30:
        return "sell", 0.65

    return None, 0.0

############################
# TRADE EXECUTION
############################

def execute_trade(side, price, strategy, confidence, qty):
    commentary = f"{datetime.now()} - Predicted {side.upper()} (conf 
{confidence:.2f}) qty {qty:.6f} @ {price:.2f} via {strategy}"
    print(commentary)

    log = pd.DataFrame([[datetime.now(), price, side, strategy, 
float(confidence), float(qty)]],
                       
columns=['timestamp','price','side','strategy','confidence','qty'])
    log.to_csv(LOG_FILE, mode='a', header=False, index=False)

    if DEMO_MODE:
        print("DEMO_MODE=True → Trade NOT sent.")
        return {"status": "demo", "qty": qty, "price": price, "side": 
side, "strategy": strategy}

    try:
        # Placeholder for live execution
        print("Executing live order (placeholder).")
        return {"status": "live_order_sent"}
    except Exception as e:
        print("Trade execution error:", e)
        return None

############################
# MAIN BOT
############################

def main():
    if DEMO_MODE:
        df_master = generate_demo_data()
    else:
        df_master = load_json_data()
        if df_master.empty:
            raise RuntimeError("No data loaded and DEMO_MODE=False")

    df_master['signal'] = None

    # Initialize models
    ensemble = PredictiveEnsemble(window=40)
    kalman = SimpleKalman(q_level=1e-4, q_trend=1e-4, r=1.0)
    vol_est = EWMA_Volatility(span=20)

    try:
        ensemble.train_initial(df_master['close'].values)
    except Exception:
        pass

    kalman.initialize(df_master['close'].iloc[0])

    print("Bot started. DEMO_MODE =", DEMO_MODE)

    while True:
        try:
            # New candle simulation for demo
            if DEMO_MODE:
                last_time = df_master['timestamp'].iloc[-1]
                new_time = last_time + 
pd.Timedelta(seconds=TRADE_INTERVAL)
                last_price = float(df_master['close'].iloc[-1])
                new_price = last_price + np.random.normal(scale=40.0)
                new_row = {
                    'timestamp': new_time,
                    'open': new_price + np.random.normal(scale=5.0),
                    'high': new_price + abs(np.random.normal(scale=10.0)),
                    'low': new_price - abs(np.random.normal(scale=10.0)),
                    'close': new_price,
                    'volume': int(max(1, np.random.poisson(5)))
                }
                df_master = pd.concat([df_master.iloc[1:], 
pd.DataFrame([new_row])], ignore_index=True)

            # Compute indicators
            df_master = compute_indicators(df_master)

            # Signals
            ma_sig = generate_signal(df_master)[0]

            # Ensemble updates
            ensemble.online_update(df_master['close'].values)
            preds = ensemble.predict(df_master['close'].values)

            preds_list = []
            if preds is not None:
                mu_var_bayes, mu_var_sgd = preds
                preds_list.append(mu_var_bayes)
                preds_list.append(mu_var_sgd)

            level, trend, obs_var = 
kalman.update(df_master['close'].iloc[-1])
            kalman_mu = trend / max(level, 1e-8)
            kalman_var = obs_var
            preds_list.append((kalman_mu, kalman_var))

            ensemble_mu, ensemble_var = 
precision_weighted_ensemble(preds_list)

            ret = np.log(df_master['close'].iloc[-1] / 
df_master['close'].iloc[-2] + 1e-12)
            est_vol = vol_est.update(ret)

            direction = 'buy' if ensemble_mu > 0 else 'sell'
            consensus = (ma_sig == direction)
            confidence = max(0.0, min(1.0, 1.0 / (1.0 + ensemble_var)))

            target_vol = 0.02
            price_now = float(df_master['close'].iloc[-1])
            qty = size_from_vol(target_vol, est_vol, price_now, 
max_usd_alloc=1000)

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

            # Dashboard
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

