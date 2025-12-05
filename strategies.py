# strategies.py
import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge, SGDRegressor
from sklearn.preprocessing import StandardScaler

class EWMA_Volatility:
    def __init__(self, span=20):
        self.alpha = 2 / (span + 1)
        self.var = None

    def update(self, ret):
        if self.var is None:
            self.var = ret**2
        else:
            self.var = (1 - self.alpha) * self.var + self.alpha * (ret**2)
        return float(np.sqrt(self.var))

class SimpleKalman:
    def __init__(self, q_level=1e-5, q_trend=1e-5, r=1.0):
        self.x = np.array([[0.0], [0.0]])  # [level, trend]
        self.P = np.eye(2) * 1.0
        self.F = np.array([[1.0, 1.0],
                           [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        self.Q = np.diag([q_level, q_trend])
        self.R = np.array([[r]])
        self.initialized = False

    def initialize(self, first_price):
        self.x = np.array([[first_price], [0.0]])
        self.initialized = True

    def update(self, price):
        if not self.initialized:
            self.initialize(price)
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        y = np.array([[price]]) - (self.H @ x_pred)
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        self.x = x_pred + K @ y
        self.P = (np.eye(2) - K @ self.H) @ P_pred
        level = float(self.x[0])
        trend = float(self.x[1])
        obs_var = float(S)
        return level, trend, obs_var

class PredictiveEnsemble:
    def __init__(self, window=40):
        self.window = window
        self.scaler = StandardScaler()
        self.bayes = BayesianRidge(compute_score=True)
        self.sgd = SGDRegressor(max_iter=1000, tol=1e-3)
        self.trained = False

    def featurize(self, closes):
        closes = np.asarray(closes)
        rocs = np.concatenate([[0.0], np.diff(closes) / closes[:-1]])
        ma5 = 
pd.Series(closes).rolling(5).mean().fillna(method='bfill').values
        ma20 = 
pd.Series(closes).rolling(20).mean().fillna(method='bfill').values
        vol = 
pd.Series(closes).pct_change().rolling(10).std().fillna(0).values
        X = np.vstack([rocs, (closes - ma5) / (ma5 + 1e-8), (ma5 - ma20) / 
(ma20 + 1e-8), vol]).T
        return X

    def train_initial(self, closes):
        X = self.featurize(closes)
        y = np.concatenate([np.diff(closes) / closes[:-1], [0.0]])
        self.scaler.fit(X)
        Xs = self.scaler.transform(X)
        # fit on full history (cheap for demo)
        self.bayes.fit(Xs, y)
        # initialize sgd
        try:
            self.sgd.partial_fit(Xs, y)
        except Exception:
            # sgd might require 2D shape; try fit if not yet initialized
            self.sgd.fit(Xs, y)
        self.trained = True

    def predict(self, closes):
        if not self.trained:
            return None, None
        X = self.featurize(closes[-self.window:])
        Xs = self.scaler.transform(X)
        x_recent = Xs[-1].reshape(1, -1)
        mu_bayes = float(self.bayes.predict(x_recent)[0])
        # crude predictive variance approx from BayesianRidge (alpha_ ~ 
noise precision)
        try:
            var_bayes = 1.0 / (self.bayes.alpha_ + 1e-9)
        except Exception:
            var_bayes = 1.0
        mu_sgd = float(self.sgd.predict(x_recent)[0])
        var_sgd = 1.0
        return (mu_bayes, var_bayes), (mu_sgd, var_sgd)

    def online_update(self, closes):
        if not self.trained and len(closes) >= max(self.window, 50):
            self.train_initial(closes)
            return
        if len(closes) < self.window:
            return
        X = self.featurize(closes[-self.window:])
        y = np.concatenate([np.diff(closes[-self.window:]) / 
closes[-self.window:-1], [0.0]])
        Xs = self.scaler.transform(X)
        try:
            self.sgd.partial_fit(Xs, y)
            self.bayes.fit(Xs, y)
        except Exception:
            pass

def precision_weighted_ensemble(preds_vars):
    mus = np.array([p for p, v in preds_vars], dtype=float)
    vars_ = np.array([v for p, v in preds_vars], dtype=float)
    vars_[vars_ <= 1e-8] = 1e-8
    precs = 1.0 / vars_
    mu = (mus * precs).sum() / precs.sum()
    var = 1.0 / precs.sum()
    return float(mu), float(var)

def size_from_vol(target_vol, est_vol, price, max_usd_alloc=1000):
    if est_vol <= 0:
        return 0.0
    # dollar exposure target scaled by estimate
    dollar_risk = max_usd_alloc * (target_vol / (est_vol + 1e-9))
    qty = dollar_risk / max(price, 1e-8)
    return float(qty)

