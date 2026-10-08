"""
learner_nn.py -- the engine's "learn from mistakes" brain.  Two parts, both small, both honest.

1) MISTAKE MEMORY (nearest neighbours).  "Is this setup a copy of setups that already lost?"
   Every resolved trade (and shadow trade) is a point in feature space.  For a new candidate we look at its closest past setups;
   if most of them LOST (more than the engine's overall loss rate) the candidate loses probability, if most WON it gains a little.
   This is exactly "the last trade failed -> don't repeat that mistake", and it works from a handful of trades.

2) TINY NEURAL NETWORK (pure numpy, one hidden layer).  Learns non-linear combinations of the factors / entry-quality numbers that the
   per-factor model cannot see.  It is NOT used blindly: it is trained on the older 70% of the history and tested on the newest 30%;
   it only gets a vote (NN_BLEND) if it beats a plain base-rate guess out-of-sample.  With too little data it stays OFF by itself.
   (TensorFlow/LSTM was deliberately NOT used: ~100 trades cannot train a sequence model without memorising them, and TensorFlow is
   a ~500 MB install that can break a free Streamlit deployment.  This network trains in milliseconds.)

learned_adjustment() returns a log-odds delta: + = more likely to win, - = repeat of a mistake.
"""
from __future__ import annotations

import math
import time

import numpy as np

NUM_KEYS = ["rsi", "ema20_dist_atr", "last_candle_atr", "run_6c_atr", "streak", "atr", "rr", "hour", "vix", "pcr_per15", "strategy_score"]
NN_MIN_ROWS = 30             # below this the network stays off
NN_HIDDEN = 8
NN_EPOCHS = 350
NN_LR = 0.05
NN_L2 = 0.03
NN_BLEND = 0.5               # share of the network's opinion used once it has proven itself out-of-sample
NN_MIN_IMPROVE = 0.01        # holdout log-loss must beat the base-rate guess by this much
NN_MAX_DELTA = 0.5
RECENCY_HALF_LIFE = 60       # rows
MEM_K = 7
MEM_MIN_ROWS = 8
MEM_REF_PRIOR_N = 12.0
MEM_MAX_PENALTY = 0.35
MEM_MAX_BONUS = 0.15
_CACHE = {"key": None, "ts": 0.0, "bundle": None}
CACHE_TTL_S = 240


def _logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _num(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else float("nan")
    except Exception:
        return float("nan")


def vectorize(flags, feats, keys):
    """flags: dict of booleans; feats: dict of numbers (or None for old trades).  -> 1-D float array (NaN = missing numeric)."""
    f = feats or {}
    x = [1.0 if (flags or {}).get(k) else 0.0 for k in keys]
    x += [_num(f.get(k)) if feats else float("nan") for k in NUM_KEYS]
    x.append(_num(f.get("is_buy")) if (feats and f.get("is_buy") is not None) else 0.5)
    x.append(1.0 if feats else 0.0)
    return np.array(x, dtype=float)


def _standardizer(X, n_flag):
    """Per-column mean/std for the numeric block (columns n_flag .. n_flag+len(NUM_KEYS)); NaN -> 0 after scaling."""
    mu = np.zeros(X.shape[1]); sd = np.ones(X.shape[1])
    for j in range(n_flag, n_flag + len(NUM_KEYS)):
        col = X[:, j]
        ok = col[~np.isnan(col)]
        if len(ok) >= 3:
            mu[j] = ok.mean(); sd[j] = max(ok.std(), 1e-6)
    return mu, sd


def _apply(X, mu, sd):
    Z = (X - mu) / sd
    return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)


class TinyNet:
    def __init__(self, d, h=NN_HIDDEN, seed=7):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.3, (d, h)); self.b1 = np.zeros(h)
        self.W2 = rng.normal(0, 0.3, (h, 1)); self.b2 = np.zeros(1)

    def _fwd(self, X):
        H = np.tanh(X @ self.W1 + self.b1)
        z = (H @ self.W2 + self.b2).ravel()
        return H, 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def predict(self, X):
        return self._fwd(X)[1]

    def fit(self, X, y, w, epochs=NN_EPOCHS, lr=NN_LR, l2=NN_L2):
        w = w / max(w.sum(), 1e-9)
        params = [self.W1, self.b1, self.W2, self.b2]
        m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
        for t in range(1, epochs + 1):
            H, p = self._fwd(X)
            dz = ((p - y) * w).reshape(-1, 1)                       # d(weighted logloss)/dz
            gW2 = H.T @ dz + l2 * self.W2; gb2 = dz.sum(0)
            dH = (dz @ self.W2.T) * (1 - H ** 2)
            gW1 = X.T @ dH + l2 * self.W1; gb1 = dH.sum(0)
            for i, g in enumerate((gW1, gb1, gW2, gb2)):             # Adam
                m[i] = 0.9 * m[i] + 0.1 * g; v[i] = 0.999 * v[i] + 0.001 * g * g
                params[i] -= lr * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
        return self


def _wlogloss(p, y, w):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-(w * (y * np.log(p) + (1 - y) * np.log(1 - p))).sum() / max(w.sum(), 1e-9))


def _prepare(rows, keys):
    """rows: list of dicts {flags, label, weight, feats} OLDEST first -> X, y, w"""
    X = np.vstack([vectorize(r["flags"], r.get("feats"), keys) for r in rows])
    y = np.array([float(r["label"]) for r in rows])
    n = len(rows)
    rec = np.array([0.5 ** ((n - 1 - i) / RECENCY_HALF_LIFE) for i in range(n)])
    w = np.array([float(r.get("weight", 1.0)) for r in rows]) * (0.35 + 0.65 * rec)
    return X, y, w


def train(rows, keys, force=False):
    """-> bundle dict.  Cached for a few minutes (training is cheap but runs inside the 30 s refresh)."""
    key = (len(rows), round(sum(r["label"] for r in rows), 3), round(sum(float(r.get("weight", 1.0)) for r in rows), 3))
    now = time.time()
    if not force and _CACHE["key"] == key and now - _CACHE["ts"] < CACHE_TTL_S and _CACHE["bundle"] is not None:
        return _CACHE["bundle"]
    bundle = {"valid": False, "n": len(rows), "reason": f"needs {NN_MIN_ROWS} labelled trades (have {len(rows)})",
              "keys": list(keys), "base_rate": None, "net": None, "mu": None, "sd": None, "improve": None}
    try:
        if len(rows) >= 1:
            X, y, w = _prepare(rows, keys)
            bundle["base_rate"] = float((y * w).sum() / w.sum())
            bundle.update({"X": X, "y": y, "w": w})
        if len(rows) >= NN_MIN_ROWS:
            X, y, w = _prepare(rows, keys)
            cut = int(len(rows) * 0.7)
            n_flag = len(keys)
            mu, sd = _standardizer(X[:cut], n_flag)
            net = TinyNet(X.shape[1]).fit(_apply(X[:cut], mu, sd), y[:cut], w[:cut])
            base_tr = float((y[:cut] * w[:cut]).sum() / w[:cut].sum())
            p_nn = net.predict(_apply(X[cut:], mu, sd))
            ll_nn = _wlogloss(p_nn, y[cut:], w[cut:])
            ll_base = _wlogloss(np.full(len(p_nn), base_tr), y[cut:], w[cut:])
            improve = ll_base - ll_nn
            bundle["improve"] = round(improve, 4)
            if improve >= NN_MIN_IMPROVE:
                mu2, sd2 = _standardizer(X, n_flag)
                bundle.update({"valid": True, "net": TinyNet(X.shape[1]).fit(_apply(X, mu2, sd2), y, w), "mu": mu2, "sd": sd2,
                               "reason": f"validated out-of-sample (beats base-rate guess by {improve:.3f} log-loss on the newest {len(rows) - cut} trades)"})
            else:
                bundle["reason"] = f"not used: did not beat a base-rate guess out-of-sample (gain {improve:+.3f}); it keeps learning as trades arrive"
    except Exception as exc:                                         # learning must never break trading
        bundle.update({"valid": False, "reason": f"error: {type(exc).__name__}"})
    _CACHE.update({"key": key, "ts": now, "bundle": bundle})
    return bundle


def mistake_memory(x, bundle):
    """Nearest-neighbour check.  -> {"penalty_logit": float (+ = penalty), "note": str, "n": int}"""
    out = {"penalty_logit": 0.0, "note": "", "n": 0}
    try:
        X, y, w = bundle.get("X"), bundle.get("y"), bundle.get("w")
        if X is None or len(y) < MEM_MIN_ROWS:
            out["note"] = f"mistake memory: needs {MEM_MIN_ROWS} labelled setups"
            return out
        n_flag = len(bundle["keys"])
        mu, sd = _standardizer(X, n_flag)
        Z = _apply(X, mu, sd); z = _apply(x.reshape(1, -1), mu, sd)[0]
        d = np.sqrt(((Z - z) ** 2).sum(axis=1) / Z.shape[1])
        idx = np.argsort(d)[:MEM_K]
        wk = w[idx] / (1.0 + d[idx])
        loss_k = float(((1 - y[idx]) * wk).sum() / wk.sum())
        loss_all = float(((1 - y) * w).sum() / w.sum())
        # Reference loss rate: the engine's own overall rate, shrunk toward a neutral 50% while there are few trades.  Without the
        # shrink, a history of ONLY losses would make "my neighbours lost" look average (100% vs 100%) and nothing would be learned.
        n_eff = float(w.sum())
        ref = (n_eff * loss_all + MEM_REF_PRIOR_N * 0.5) / (n_eff + MEM_REF_PRIOR_N)
        delta = loss_k - ref
        pen = max(-MEM_MAX_BONUS, min(MEM_MAX_PENALTY, 1.2 * delta))
        if len(idx) >= 3 and all(y[i] == 0 for i in idx[:3]):
            pen = min(MEM_MAX_PENALTY, pen + 0.10)
        losses = int((y[idx] == 0).sum())
        out.update({"penalty_logit": round(pen, 3), "n": int(len(idx)),
                    "note": (f"mistake memory: {losses} of the {len(idx)} most similar past setups lost "
                             f"(engine overall loses {100 * loss_all:.0f}%)")})
    except Exception:
        out["note"] = "mistake memory unavailable"
    return out


def learned_adjustment(flags, feats, keys, rows):
    """Main entry.  rows = labelled history (oldest first).  -> {"delta_logit", "nn": {...}, "memory": {...}, "note"}"""
    bundle = train(rows, keys)
    x = vectorize(flags, feats, keys)
    mem = mistake_memory(x, bundle)
    nn_delta, nn_p = 0.0, None
    if bundle.get("valid") and bundle.get("net") is not None:
        try:
            z = _apply(x.reshape(1, -1), bundle["mu"], bundle["sd"])
            nn_p = float(bundle["net"].predict(z)[0])
            nn_delta = NN_BLEND * (_logit(nn_p) - _logit(bundle["base_rate"]))
            nn_delta = max(-NN_MAX_DELTA, min(NN_MAX_DELTA, nn_delta))
        except Exception:
            nn_delta = 0.0
    delta = nn_delta - mem["penalty_logit"]
    parts = []
    if nn_p is not None:
        parts.append(f"neural net says {100 * nn_p:.0f}% for this setup ({nn_delta:+.2f})")
    if mem["note"]:
        parts.append(mem["note"] + (f" ({-mem['penalty_logit']:+.2f})" if mem["penalty_logit"] else ""))
    return {"delta_logit": round(delta, 3), "nn": {"valid": bool(bundle.get("valid")), "p": nn_p, "reason": bundle.get("reason"),
                                                    "improve": bundle.get("improve"), "n": bundle.get("n")},
            "memory": mem, "note": " | ".join(parts)}
