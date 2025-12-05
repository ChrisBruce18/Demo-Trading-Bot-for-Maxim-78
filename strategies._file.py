# bot.py
import time
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# demo vs live toggle
DEMO_MODE = True

# local config (adjust as needed)
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1m'
LIMIT = 200  # number of candles in buffer
TRADE_INTERVAL = 30  # seconds between checks
LOG_FILE = 'logs/trade_log.csv'

# create logs dir
if not os.path.exists('logs'):
    os.makedirs('logs')

# ensure header present
if not os.path.exists(LOG_FILE):
    
pd.DataFrame(columns=['timestamp','price','side','strategy','confidence','qty']).to_csv(LOG_FILE,index=False)

# import dashboard & strategies
from dashboard import update_dashboard
from strategies import PredictiveEnsemble, SimpleKalman, EWMA_Volatility, 
precision_weighted_ensemble, size_from_vol

# ------------------------------
# Synthetic data generator for demo mode
# ------------------------------
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

# ------------------------------
# Strategy helpers (MA crossover kept simple)
# ------------------------------
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

def execute_trade(side, price, strategy, confidence, qty):
    commentary = f"{datetime.now()} - Predicted {side.upper()} (conf 
{confidence:.2f}) qty {qty:.6f} @ {price:.2f} via {strategy}"
    print(commentary)
    log = pd.DataFrame([[datetime.now(), price, side, strategy, 
float(confidence), float(qty)]],
                       
columns=['timestamp','price','side','strategy','confidence','qty'])
    log.to_csv(LOG_FILE, mode='a', header=False, index=False)
    return commentary

# ------------------------------
# Main initialization
# ------------------------------
if DEMO_MODE:
    df_master = generate_demo_data()
else:
    # placeholder for real exchange fetch; user will add ccxt/Gemini 
integration
    raise RuntimeError("DEMO_MODE=False: integrate exchange fetch code or 
set DEMO_MODE=True for now.")

df_master['signal'] = None

# initialize models (global for the run)
ensemble = PredictiveEnsemble(window=40)
kalman = SimpleKalman(q_level=1e-4, q_trend=1e-4, r=1.0)
vol_est = EWMA_Volatility(span=20)

# train initial when enough data
try:
    ensemble.train_initial(df_master['close'].values)
except Exception:
    pass
kalman.initialize(df_master['close'].iloc[0])

print("Demo bot started (DEMO_MODE=True). Running with synthetic data.")

# main loop
while True:
    try:
        # simulate new candle
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

        # compute signals
        ma_sig = ma_crossover_signal(df_master)
        # update ensemble online
        ensemble.online_update(df_master['close'].values)
        preds = ensemble.predict(df_master['close'].values)
        # in case ensemble not ready
        preds_list = []
        if preds is not None:
            mu_var_bayes, mu_var_sgd = preds
            preds_list.append(mu_var_bayes)
            preds_list.append(mu_var_sgd)
        # kalman update
        level, trend, obs_var = kalman.update(df_master['close'].iloc[-1])
        kalman_mu = trend / max(level, 1e-8)
        kalman_var = obs_var
        preds_list.append((kalman_mu, kalman_var))

        # compute ensemble
        ensemble_mu, ensemble_var = 
precision_weighted_ensemble(preds_list)

        # estimated vol (log returns)
        ret = np.log(df_master['close'].iloc[-1] / 
df_master['close'].iloc[-2] + 1e-12)
        est_vol = vol_est.update(ret)

        # direction and consensus
        direction = 'buy' if ensemble_mu > 0 else 'sell'
        consensus = (ma_sig == direction)

        # confidence mapping (lower ensemble_var -> higher confidence)
        confidence = max(0.0, min(1.0, 1.0 / (1.0 + ensemble_var)))

        # position sizing
        target_vol = 0.02  # tune based on timeframe
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

        # update dashboard
        update_dashboard(df_master, commentary)

        time.sleep(TRADE_INTERVAL)  # wait before next candle
    except KeyboardInterrupt:
        print("Stopped by user.")
        break
    except Exception as e:
        print("Runtime error:", e)
        time.sleep(5)

