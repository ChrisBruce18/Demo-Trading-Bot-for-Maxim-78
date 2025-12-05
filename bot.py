import json
import time
from datetime import datetime
import pandas as pd
import numpy as np
import os

from strategies import (
    PredictiveEnsemble,
    SimpleKalman,
    EWMA_Volatility,
    precision_weighted_ensemble,
    size_from_vol
)

############################
# CONFIG LOADER
############################

class Config:
    def __init__(self, path="config.json"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found")
        with open(path, "r") as f:
            self.cfg = json.load(f)
        self.exchange = self.cfg.get("exchange", {})
        self.trade_settings = self.cfg.get("trade_settings", {})

    @property
    def name(self):
        return self.exchange.get("name")

    @property
    def api_key(self):
        return self.exchange.get("api_key")

    @property
    def api_secret(self):
        return self.exchange.get("api_secret")

    @property
    def symbols(self):
        return self.exchange.get("symbols", [])

    @property
    def timeframe(self):
        return self.exchange.get("timeframe", "1m")

    @property
    def limit(self):
        return self.exchange.get("limit", 200)

    @property
    def sandbox(self):
        return self.exchange.get("sandbox", True)

    @property
    def trade_interval(self):
        return self.trade_settings.get("trade_interval", 30)

    @property
    def log_file(self):
        return self.trade_settings.get("log_file", "logs/trade_log.csv")

    @property
    def max_usd_allocation(self):
        return self.trade_settings.get("max_usd_allocation", 1000)

    @property
    def target_vol(self):
        return self.trade_settings.get("target_vol", 0.02)

    @property
    def sleep_seconds(self):
        return self.trade_settings.get("sleep_seconds", 5)

cfg = Config()

# Ensure logs directory exists
os.makedirs(os.path.dirname(cfg.log_file), exist_ok=True)
if not os.path.exists(cfg.log_file):
    pd.DataFrame(
        columns=['timestamp','price','side','strategy','confidence','qty']
    ).to_csv(cfg.log_file, index=False)

############################
# DATA FUNCTIONS
############################

def load_json_data(file="data.json"):
    """Load historical candle data from JSON."""
    try:
        with open(file, "r") as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except Exception as e:
        print("JSON load failed:", e)
        return pd.DataFrame()

def fetch_latest_price(df):
    """Get the most recent close price."""
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
    """MA crossover + RSI filter."""
    if len(df) < 60:
        return None, 0.0

    ma_cross_buy = (
        df["MA_short"].iloc[-2] < df["MA_long"].iloc[-2] and
        df["MA_short"].iloc[-1] > df["MA_long"].iloc[-1]
    )
    ma_cross_sell = (
        df["MA_short"].iloc[-2] > df["MA_long"].iloc[-2] and
        df["MA_short"].iloc[-1] < df["MA_long"].iloc[-1]
    )

    if ma_cross_buy and df["RSI"].iloc[-1] < 70:
        return "buy", 0.65
    if ma_cross_sell and df["RSI"].iloc[-1] > 30:
        return "sell", 0.65
    return None, 0.0

############################
# TRADE EXECUTION
############################

def execute_trade(side, confidence, price, strategy):
    qty = size_from_vol(cfg.target_vol, price, 
max_usd_alloc=cfg.max_usd_allocation)
    commentary = (
        f"{datetime.now()} - Predicted {side.upper()} "
        f"(conf {confidence:.2f}) qty {qty:.6f} @ {price:.2f} via 
{strategy}"
    )
    print(commentary)

    # Log trade
    log = pd.DataFrame(
        [[datetime.now(), price, side, strategy, float(confidence), 
float(qty)]],
        columns=['timestamp','price','side','strategy','confidence','qty']
    )
    log.to_csv(cfg.log_file, mode='a', header=False, index=False)
    return commentary

############################
# MAIN LOOP
############################

def main():
    df = load_json_data()
    if df.empty:
        print("No data loaded. Exiting.")
        return

    df = compute_indicators(df)

    # Initialize models
    ensemble = PredictiveEnsemble(window=40)
    kalman = SimpleKalman(q_level=1e-4, q_trend=1e-4, r=1.0)
    vol_est = EWMA_Volatility(span=20)

    try:
        ensemble.train_initial(df['close'].values)
    except Exception:
        pass
    kalman.initialize(df['close'].iloc[0])

    print(f"Live bot started for {cfg.name.upper()}, symbols: 
{cfg.symbols}")

    while True:
        try:
            price = fetch_latest_price(df)

            # Ensemble model
            ensemble.online_update(df['close'].values)
            preds = ensemble.predict(df['close'].values)
            preds_list = []
            if preds:
                mu_var_bayes, mu_var_sgd = preds
                preds_list.append(mu_var_bayes)
                preds_list.append(mu_var_sgd)
            level, trend, obs_var = kalman.update(price)
            kalman_mu = trend / max(level, 1e-8)
            kalman_var = obs_var
            preds_list.append((kalman_mu, kalman_var))

            # Weighted ensemble
            ensemble_mu, ensemble_var = 
precision_weighted_ensemble(preds_list)

            # Volatility estimate
            ret = np.log(price / df['close'].iloc[-2] + 1e-12)
            est_vol = vol_est.update(ret)

            # Technical signal
            tech_side, tech_conf = generate_signal(df)

            # Consensus & confidence
            direction = 'buy' if ensemble_mu > 0 else 'sell'
            consensus = (tech_side == direction)
            confidence = max(0.0, min(1.0, 1.0 / (1.0 + ensemble_var)))

            qty = size_from_vol(cfg.target_vol, est_vol, price, 
max_usd_alloc=cfg.max_usd_allocation)

            final_signal = None
            commentary = None
            if consensus and confidence > 0.25 and qty > 0:
                final_signal = direction
                commentary = execute_trade(final_signal, confidence, 
price, 'Ensemble+Signal')
            else:
                commentary = f"{datetime.now()} - No trade: 
consensus={consensus}, conf={confidence:.2f}, qty={qty:.6f}"

            # Update dataframe (append latest candle)
            df = df.append({'timestamp': datetime.now(), 'close': price}, 
ignore_index=True)

            time.sleep(cfg.trade_interval)

        except KeyboardInterrupt:
            print("Stopped by user.")
            break
        except Exception as e:
            print("Runtime error:", e)
            time.sleep(5)

if __name__ == "__main__":
    main()


