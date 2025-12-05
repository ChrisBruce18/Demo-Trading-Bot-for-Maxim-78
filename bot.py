import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime
from sklearn.linear_model import LinearRegression
from dashboard import update_dashboard

# ------------------------------
# CONFIGURATION
# ------------------------------
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1m'
LIMIT = 100
TRADE_AMOUNT = 0.001

API_KEY = 'YOUR_BINANCE_TESTNET_API_KEY'
API_SECRET = 'YOUR_BINANCE_TESTNET_API_SECRET'

exchange = ccxt.binance({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
    'options': {'defaultType': 'future'},
})
exchange.set_sandbox_mode(True)

# ------------------------------
# LOGGING
# ------------------------------
import os
if not os.path.exists('logs'):
    os.makedirs('logs')
LOG_FILE = 'logs/trade_log.csv'
if not os.path.exists(LOG_FILE):
    
pd.DataFrame(columns=['timestamp','price','side','strategy','confidence']).to_csv(LOG_FILE,index=False)

# ------------------------------
# STRATEGY FUNCTIONS
# ------------------------------
def fetch_ohlcv():
    data = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=LIMIT)
    df = pd.DataFrame(data, 
columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def ma_crossover_signal(df, short=5, long=20):
    df['MA_short'] = df['close'].rolling(short).mean()
    df['MA_long'] = df['close'].rolling(long).mean()
    if df['MA_short'].iloc[-2] < df['MA_long'].iloc[-2] and 
df['MA_short'].iloc[-1] > df['MA_long'].iloc[-1]:
        return 'buy'
    elif df['MA_short'].iloc[-2] > df['MA_long'].iloc[-2] and 
df['MA_short'].iloc[-1] < df['MA_long'].iloc[-1]:
        return 'sell'
    return None

def predictive_poly_signal(df, windo

