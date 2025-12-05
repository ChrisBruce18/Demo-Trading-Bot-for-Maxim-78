# bot.py
import time
import os
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# ------------------------------
# LOAD CONFIG FROM JSON
# ------------------------------
CONFIG_FILE = 'config.json'
if not os.path.exists(CONFIG_FILE):
    raise FileNotFoundError(f"Missing configuration file: {CONFIG_FILE}")

with open(CONFIG_FILE, 'r') as f:
    config = json.load(f)

DEMO_MODE = config.get('demo_mode', True)
SYMBOL = config.get('symbol', 'BTC/USDT')
TIMEFRAME = config.get('timeframe', '1m')
LIMIT = config.get('limit', 200)
TRADE_INTERVAL = config.get('trade_interval', 30)
LOG_FILE = config.get('log_file', 'logs/trade_log.csv')
MAX_USD_ALLOC = config.get('max_usd_alloc', 1000)
TARGET_VOL = config.get('target_vol', 0.02)

# ------------------------------
# LOGGING SETUP
# ------------------------------
if not os.path.exists('logs'):
    os.makedirs('logs')

if not os.path.exists(LOG_FILE):
    pd.DataFrame(
        columns=['timestamp','price','side','strategy','confidence','qty']
    ).to_csv(LOG_FILE, index=False)

# ------------------------------
# IMPORT DASHBOARD & STRATEGIES
# ------------------------------
from dashboard import update_dashboard
from strategies import (
    PredictiveEnsemble,
    SimpleKalman,
    EWMA_Volatility,
    precision_weighted_ensemble,
    size_from_vol
)

# ------------------------------
# DEMO DATA GENERATOR
# ------------------------------
def generate_demo_data(length=LIMIT, start_price=50000.0):
    rng = np.random.default_rng(seed=42)
    price = start_price + np.cumsum(rng.normal(scale=40.0, size=length))
    timestamps = [
        datetime.now() - timedelta(seconds=TRADE_INTERVAL*(length - i))
        for i in range(length)
    ]
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': price + rng.normal(scale=5.0, size=length),
        'high': price + rng.normal(scale=10.0, size=length),
        'low': price - rng.normal(scale=10.0, size=length),
        'close': price,
        'volume': rng.integers(1, 10, size=length)
    })
    return df

# ------------------------------
# STRATEGY HELPERS
# ------------------------------
def ma_crossover_signal(df, short=5, long=20):
    if len(df) < long + 2:
        return None
    df = df.copy()
    df['MA_short'] = df['close'].rolling(short).mean()
    df['MA_long'] = df['close'].rolling(long).mean()
    if (df['MA_short'].iloc[-2] < df['MA_long'].iloc[-2] and
        df['MA_short'].iloc[-1] > df['MA_long'].iloc[-1]):
        return 'buy'
    elif (df['MA_short'].iloc[-2] > df['MA_long'].iloc[-2] and
          df['MA_short'].iloc[-1] < df['MA_long'].iloc[-1]):
        return 'sell'
    return None

def execute_trade(side, price, strategy, confidence, qty):
    commentary = (
        f"{datetime.now()} - Predicted {side.upper()} "
        f"(conf {confidence:.2f}) qty {qty:.6f} @ {price:.2f} via 
{strategy}"
    )
    print(commentary)
    log_df = pd.DataFrame([[datetime.now(), price, side, strategy, 
float(confidence), float(qty)]],
                          
columns=['timestamp','price','side','strategy','confidence','qty'])
    log_df.to_csv(LOG_FILE, mode='a', header=False, index=False)
    return commentary

# ------------------------------
# MAIN INITIALIZATION
# ------------------------------
if DEMO_MODE:
    df_master = generate_demo_data()
else:
    raise RuntimeError("DEMO_MODE=False requires exchange fetch code.")

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

print(f"Bot started. DEMO_MODE={DEMO_MODE}, Symbol={SYMBOL}")

# ------------------------------
# MAIN LOOP
# ------------------------------
while True:
    try:
        # Simulate new candle
        last_time = df_master['timestamp'].iloc[-1]
        new_time = last_time + pd.Timedelta(seconds=TRADE_INTERVAL)
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

        # Compute signals
        ma_sig = ma_crossover_signal(df_master)

        # Update ensemble
        ensemble.online_update(df_master['close'].values)
        preds = ensemble.predict(df_master['close'].values)
        preds_list = []

        if preds is not None:
            mu_var_bayes, mu_var_sgd = preds
            preds_list.append(mu_var_bayes)
            preds_list.append(mu_var_sgd)

        # Kalman filter update
        level, trend, obs_var = kalman.update(df_master['close'].iloc[-1])
        kalman_mu = trend / max(level, 1e-8)
        kalman_var = obs_var
        preds_list.append((kalman_mu, kalman_var))

        # Compute ensemble
        ensemble_mu, ensemble_var = 
precision_weighted_ensemble(preds_list)

        # Estimate volatility
        ret = np.log(df_master['close'].iloc[-1] / 
df_master['close'].iloc[-2] + 1e-12)
        est_vol = vol_est.update(ret)

        # Determine consensus & confidence
        direction = 'buy' if ensemble_mu > 0 else 'sell'
        consensus = (ma_sig == direction)
        confidence = max(0.0, min(1.0, 1.0 / (1.0 + ensemble_var)))

        # Position sizing
        price_now = float(df_master['close'].iloc[-1])
        qty = size_from_vol(TARGET_VOL, est_vol, price_now, 
max_usd_alloc=MAX_USD_ALLOC)

        # Execute or skip trade
        if consensus and confidence > 0.25 and qty > 0:
            final_signal = direction
            commentary = execute_trade(
                final_signal,
                price_now,
                'PhD_Ensemble',
                confidence,
                qty
            )
        else:
            final_signal = None
            commentary = (
                f"{datetime.now()} - No trade: consensus={consensus}, "
                f"conf={confidence:.2f}, qty={qty:.6f}"
            )

        df_master.at[df_master.index[-1], 'signal'] = final_signal

        # Update dashboard
        update_dashboard(df_master, commentary)

        # Wait until next interval
        time.sleep(TRADE_INTERVAL)

    except KeyboardInterrupt:
        print("Stopped by user.")
        break
    except Exception as e:
        print("Runtime error:", e)
        time.sleep(5)

