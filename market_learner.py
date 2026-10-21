"""
market_learner.py -- what the recorded market teaches (v46, purely additive, ADVISORY only).

1. PATTERN MEMORY  ~20 named situations ("naya day-low bana kar neeche wick", "RSI oversold + volume spike" ...).  For every
   one we count, from the recorded candles, what price did in the next 60 min: +25 pts first / -25 pts first / neither.  So the
   system knows, from DATA, which situations really lead to a bounce / a fall and which are noise.
2. FORECASTER  a small logistic-regression + neural-network ensemble that reads the whole feature row and answers
       * P(price touches +25 before -25)           ("dir25")
       * P(+50 before -50, if a 50 pt move comes)  ("dir50")
       * P(a 50-pt move happens at all in 60 min)  ("mv50")
   Every model is WALK-FORWARD tested: trained on the older 75% of the days, scored on the newest 25% it never saw.  It is only
   marked TRUSTED when it beats the "always guess the majority" baseline there.  Untrusted = shown as information, never as advice.
3. SCORECARD  forecasts made LIVE are stored with the candle; once the outcome is known they are scored for real.

Nothing here opens, blocks or changes a trade (FORECAST_GATE stays "off" until the scorecard proves it).
"""
from __future__ import annotations

import logging
import math
import warnings

import numpy as np

import market_memory as mm

logger = logging.getLogger(__name__)

FORECAST_GATE = "off"          # "off" = advisory only.  (A later version can switch this to "soft" once the live scorecard is good.)
MIN_ROWS = {"dir25": 80, "dir50": 60, "mv50": 100}
MIN_VAL = 120                  # test rows (all folds together) needed before a model may be called trusted
TRUST_AUC = 0.58               # mean AUC over the folds
TRUST_MIN_FOLD_AUC = 0.52      # ... and even the worst fold must be above chance
TRUST_ACC_EDGE = 0.02
PATTERN_MIN_N = 12
EXT_MIN_ROWS = 150


def _g(f, k):
    v = f.get(k)
    try:
        v = float(v)
        return v if not math.isnan(v) else float("nan")
    except Exception:
        return float("nan")


# ----------------------------------------------------------------------------- 1. patterns
# (key, Hinglish label, side hint, condition(f))   side hint: "UP" = usually a bullish situation, "DN" = bearish, "-" = neutral
PATTERNS = [
    ("sweep_low", "Naya day-low bana kar neeche lambi wick (stop-hunt)", "UP",
     lambda f: _g(f, "lower_wick") >= 0.5 and _g(f, "range_atr") >= 1.0 and _g(f, "dist_daylow_atr") <= 0.3),
    ("sweep_high", "Naya day-high bana kar upar lambi wick (stop-hunt)", "DN",
     lambda f: _g(f, "upper_wick") >= 0.5 and _g(f, "range_atr") >= 1.0 and _g(f, "dist_dayhigh_atr") <= 0.3),
    ("at_day_low_after_fall", "Din ke low par, din bhar ~0.6%+ gira hua", "-",
     lambda f: _g(f, "day_pos") <= 0.12 and _g(f, "day_move_pct") <= -0.6),
    ("at_day_high_after_rise", "Din ke high par, din bhar ~0.6%+ chadha hua", "-",
     lambda f: _g(f, "day_pos") >= 0.88 and _g(f, "day_move_pct") >= 0.6),
    ("rsi_oversold", "RSI 30 ya neeche", "UP", lambda f: _g(f, "rsi") <= 30),
    ("rsi_overbought", "RSI 70 ya upar", "DN", lambda f: _g(f, "rsi") >= 70),
    ("stretched_below_ema", "Price EMA20 se 2 ATR neeche (khicha hua)", "UP", lambda f: _g(f, "ema20_dist_atr") <= -2.0),
    ("stretched_above_ema", "Price EMA20 se 2 ATR upar (khicha hua)", "DN", lambda f: _g(f, "ema20_dist_atr") >= 2.0),
    ("vol_spike_green", "Volume spike + badi green candle", "UP", lambda f: _g(f, "vol_ratio") >= 1.8 and _g(f, "candle_atr") >= 0.8),
    ("vol_spike_red", "Volume spike + badi red candle", "DN", lambda f: _g(f, "vol_ratio") >= 1.8 and _g(f, "candle_atr") <= -0.8),
    ("above_vwap_trend", "VWAP aur EMA20 dono ke upar (uptrend)", "UP", lambda f: _g(f, "vwap_dist_atr") >= 0.5 and _g(f, "ema20_dist_atr") > 0),
    ("below_vwap_trend", "VWAP aur EMA20 dono ke neeche (downtrend)", "DN", lambda f: _g(f, "vwap_dist_atr") <= -0.5 and _g(f, "ema20_dist_atr") < 0),
    ("orb_breakout_up", "Opening range ke upar", "UP", lambda f: _g(f, "or_pos") == 1.0),
    ("orb_breakdown", "Opening range ke neeche", "DN", lambda f: _g(f, "or_pos") == -1.0),
    ("round_number", "50 ke round number ke bilkul paas", "-", lambda f: _g(f, "round50_dist") <= 6),
    ("streak_red4", "Lagatar 4+ red candles", "-", lambda f: _g(f, "streak") <= -4),
    ("streak_green4", "Lagatar 4+ green candles", "-", lambda f: _g(f, "streak") >= 4),
    ("late_session", "Din ka aakhri hissa (13:45 ke baad)", "-", lambda f: _g(f, "mins_since_open") >= 270),
    ("prev_day_high_zone", "Pichle din ke high ke 0.5 ATR ke andar", "DN", lambda f: abs(_g(f, "dist_pdh_atr")) <= 0.5),
    ("prev_day_low_zone", "Pichle din ke low ke 0.5 ATR ke andar", "UP", lambda f: abs(_g(f, "dist_pdl_atr")) <= 0.5),
    ("at_put_wall", "Put OI wall ke bilkul paas (live data)", "UP", lambda f: -12 <= _g(f, "put_wall_dist") <= 6 and _g(f, "put_wall_share") >= 0.6),
    ("at_call_wall", "Call OI wall ke bilkul paas (live data)", "DN", lambda f: -6 <= _g(f, "call_wall_dist") <= 12 and _g(f, "call_wall_share") >= 0.6),
]
_PAT_LABEL = {k: t for k, t, _s, _fn in PATTERNS}


def pattern_label(key):
    return _PAT_LABEL.get(key, key)


def active_patterns(f: dict):
    out = []
    for k, _t, _s, fn in PATTERNS:
        try:
            if fn(f):
                out.append(k)
        except Exception:
            pass
    return out


def _decided(snaps, level="f25"):
    return [s for s in snaps if s.get("y") and s["y"].get(level) in ("UP", "DN", "NONE")]


def pattern_stats(snaps, level="f25"):
    """[{key,text,hint,n,up,dn,none,lift_up,lift_dn}] for patterns with enough samples, strongest edge first."""
    rows = _decided(snaps, level)
    if not rows:
        return [], {}
    tot = len(rows)
    base = {"up": sum(1 for s in rows if s["y"][level] == "UP") / tot, "dn": sum(1 for s in rows if s["y"][level] == "DN") / tot, "n": tot}
    acc = {k: [0, 0, 0, 0] for k, *_ in PATTERNS}     # n, up, dn, none
    for s in rows:
        for k in active_patterns(s["f"]):
            a = acc[k]
            a[0] += 1
            a[1 if s["y"][level] == "UP" else 2 if s["y"][level] == "DN" else 3] += 1
    out = []
    for k, text, hint, _fn in PATTERNS:
        n, u, dn, no = acc[k]
        if n < PATTERN_MIN_N:
            continue
        out.append({"key": k, "text": text, "hint": hint, "n": n, "up": u / n, "dn": dn / n, "none": no / n,
                    "lift_up": u / n - base["up"], "lift_dn": dn / n - base["dn"]})
    out.sort(key=lambda z: -max(abs(z["lift_up"]), abs(z["lift_dn"])))
    return out, base


# ----------------------------------------------------------------------------- 2. forecaster
def _matrix(rows, feats):
    X = np.array([[_g(s["f"], k) for k in feats] for s in rows], dtype=float)
    return X


def _impute_fit(X):
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    Xi = np.where(np.isnan(X), med, X)
    mu, sd = Xi.mean(0), Xi.std(0) + 1e-9
    return med, mu, sd


def _transform(X, pack):
    med, mu, sd = pack
    Xi = np.where(np.isnan(X), med, X)
    return np.clip((Xi - mu) / sd, -5, 5)


def _fit(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    pack = _impute_fit(X)
    Xs = _transform(X, pack)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr = LogisticRegression(C=0.3, max_iter=600).fit(Xs, y)
        nn = MLPClassifier(hidden_layer_sizes=(12,), alpha=0.05, max_iter=400, random_state=7).fit(Xs, y)
    return {"pack": pack, "lr": lr, "nn": nn}


def _proba(m, X):
    Xs = _transform(X, m["pack"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return 0.5 * m["lr"].predict_proba(Xs)[:, 1] + 0.5 * m["nn"].predict_proba(Xs)[:, 1]


def _target(rows, name):
    if name == "dir25":
        sel = [s for s in rows if s["y"].get("f25") in ("UP", "DN")]
        return sel, np.array([1 if s["y"]["f25"] == "UP" else 0 for s in sel])
    if name == "dir50":
        sel = [s for s in rows if s["y"].get("f50") in ("UP", "DN")]
        return sel, np.array([1 if s["y"]["f50"] == "UP" else 0 for s in sel])
    sel = [s for s in rows if s["y"].get("f50") in ("UP", "DN", "NONE", "AMB")]
    return sel, np.array([0 if s["y"]["f50"] == "NONE" else 1 for s in sel])


def _train_one(rows, name, feats):
    """-> (model|None, report).  Rolling-origin test: 3 expanding-window folds over the newest half of the data (each fold is trained
    only on the PAST, with a 12-candle purge gap because neighbouring candles share look-ahead), then a final refit on everything.
    Trusted only if the average AUC is clearly above chance, the WORST fold is still above chance, and accuracy beats the
    'always guess the training-set majority' baseline.  Neighbouring candles overlap, so the bar is deliberately strict."""
    sel, y = _target(rows, name)
    rep = {"n": len(sel), "need": MIN_ROWS[name], "trusted": False, "ready": False}
    if len(sel) < MIN_ROWS[name] or len(set(y.tolist())) < 2:
        return None, rep
    X = _matrix(sel, feats)
    n = len(sel)
    gap = 12
    try:
        from sklearn.metrics import roc_auc_score
        edges = [int(n * 0.5), int(n * 0.667), int(n * 0.833), n]
        P, Y, aucs, base_hits = [], [], [], []
        for a, b in zip(edges[:-1], edges[1:]):
            tr = slice(0, max(a - gap, 0))
            if a - gap < 40 or b - a < 10 or len(set(y[tr].tolist())) < 2 or len(set(y[a:b].tolist())) < 2:
                continue
            m0 = _fit(X[tr], y[tr])
            p = _proba(m0, X[a:b])
            maj = 1 if y[tr].mean() >= 0.5 else 0
            P += p.tolist()
            Y += y[a:b].tolist()
            aucs.append(float(roc_auc_score(y[a:b], p)))
            base_hits += [int(v == maj) for v in y[a:b]]
        if aucs:
            P, Y = np.array(P), np.array(Y)
            acc = float(((P >= 0.5) == (Y == 1)).mean())
            base = float(np.mean(base_hits))
            rep.update({"val_n": int(len(Y)), "val_acc": round(acc, 3), "val_base_acc": round(base, 3), "val_auc": round(float(np.mean(aucs)), 3),
                        "val_auc_min": round(float(np.min(aucs)), 3), "folds": len(aucs), "val_brier": round(float(((P - Y) ** 2).mean()), 4),
                        "trusted": bool(len(Y) >= MIN_VAL and len(aucs) >= 2 and np.mean(aucs) >= TRUST_AUC and np.min(aucs) >= TRUST_MIN_FOLD_AUC
                                        and acc >= base + TRUST_ACC_EDGE)})
        m = _fit(X, y)
        rep["ready"] = True
        return m, rep
    except Exception:
        logger.exception("forecaster training failed (ignored): %s", name)
        return None, rep


_MODELS: dict = {"key": None, "bundle": None, "running": False}


def _build(lab):
    bundle = {"n_labelled": len(lab), "base": {}, "ext": {}, "ext_n": 0}
    for name in MIN_ROWS:
        bundle["base"][name] = _train_one(lab, name, mm.BASE_FEATS)
    ext_rows = [s for s in lab if s["f"].get("pcr") is not None or s["f"].get("put_wall_dist") is not None or s["f"].get("call_wall_dist") is not None]
    bundle["ext_n"] = len(ext_rows)
    if len(ext_rows) >= EXT_MIN_ROWS:
        feats = mm.BASE_FEATS + mm.EXT_FEATS
        for name in MIN_ROWS:
            bundle["ext"][name] = _train_one(ext_rows, name, feats)
    return bundle


def train_all(snaps, block=False):
    """Train in a BACKGROUND thread (the 30-s page never waits) whenever ~10 more labelled candles exist.
    Returns the newest finished bundle, or None while the very first training is still running."""
    import threading
    lab = [s for s in snaps if s.get("y")][-3000:]     # newest 3000 labelled candles (~40 days) keep training fast
    key = (len(lab) // 10, (lab[-1]["ts"][:10] if lab else ""))
    if _MODELS["key"] == key and _MODELS["bundle"] is not None:
        return _MODELS["bundle"]
    if _MODELS["running"] and not block:
        return _MODELS["bundle"]

    def job():
        _MODELS["running"] = True
        try:
            _MODELS.update({"bundle": _build(lab), "key": key})
        except Exception:
            logger.exception("market_learner training job failed (ignored)")
        finally:
            _MODELS["running"] = False

    if block:
        job()
    else:
        threading.Thread(target=job, daemon=True).start()
    return _MODELS["bundle"]


def forecast(snaps, f_now: dict, block=False):
    """Forecast for the current feature row.  Always returns a dict; 'ready' False when there is too little data."""
    out = {"ready": False, "trusted": False, "lean": "NEUTRAL", "msg": "", "reports": {}}
    try:
        bundle = train_all(snaps, block=block)
        if bundle is None:
            out["msg"] = "Forecaster pehli baar seekh raha hai (background me chal raha hai) -- agle refresh par dikhega."
            return out
        out["n_labelled"] = bundle["n_labelled"]
        base_m, base_rep = bundle["base"]["dir25"]
        out["reports"] = {n: bundle["base"][n][1] for n in bundle["base"]}
        if base_m is None:
            out["msg"] = f"Data abhi kam hai: {base_rep['n']}/{base_rep['need']} labelled candles. Har naye candle ke saath seekhta rahega."
            return out
        # choose the model set: the extended one (OI/PCR/VIX/breadth ...) only if it validated better than the base one
        use = "base"
        sets = bundle["base"]
        ext = bundle["ext"].get("dir25") if bundle["ext"] else None
        if ext and ext[0] is not None and ext[1].get("trusted") and ext[1].get("val_auc", 0) >= base_rep.get("val_auc", 0):
            use, sets = "ext", bundle["ext"]
        feats = mm.BASE_FEATS + (mm.EXT_FEATS if use == "ext" else [])
        x = np.array([[_g(f_now, k) for k in feats]], dtype=float)
        res = {}
        for name in ("dir25", "dir50", "mv50"):
            m, rep = sets.get(name, (None, {}))
            if m is not None:
                res[name] = float(_proba(m, x)[0])
        out["ready"] = True
        out["model_used"] = use
        out["p_up25"] = res.get("dir25")
        out["p_up50"] = res.get("dir50")
        out["p_move50"] = res.get("mv50")
        out["trusted"] = bool(sets["dir25"][1].get("trusted"))
        out["reports"] = {n: sets[n][1] for n in sets}
        p = out["p_up25"]
        out["lean"] = "UP" if p >= 0.58 else "DOWN" if p <= 0.42 else "NEUTRAL"
        out["patterns"] = active_patterns(f_now)
        out["msg"] = ("Validated (naye, na dekhe hue data par is se behtar nikla)" if out["trusted"]
                      else "Abhi UNTRUSTED -- sirf jaankari, validation me baseline se behtar sabit nahi hua")
    except Exception:
        logger.exception("forecast failed (ignored)")
        out["msg"] = "forecast error (ignored)"
    return out


def trade_view(fc: dict, direction: str):
    """How does the forecast see a proposed BUY/SELL?  -> {'verdict': SUPPORTS|AGAINST|NEUTRAL|UNTRUSTED|NO_DATA, 'p_dir': float|None}"""
    if not fc or not fc.get("ready") or fc.get("p_up25") is None:
        return {"verdict": "NO_DATA", "p_dir": None}
    p = fc["p_up25"] if str(direction).upper() == "BUY" else 1.0 - fc["p_up25"]
    if not fc.get("trusted"):
        return {"verdict": "UNTRUSTED", "p_dir": p}
    return {"verdict": "SUPPORTS" if p >= 0.56 else "AGAINST" if p <= 0.44 else "NEUTRAL", "p_dir": p}


# ----------------------------------------------------------------------------- 3. live scorecard
def scorecard(snaps):
    """Real out-of-sample proof: forecasts stored LIVE vs what the market then did."""
    rows = [s for s in snaps if s.get("fc") and s.get("y") and s["fc"].get("p_up25") is not None and s["y"].get("f25") in ("UP", "DN")]
    if not rows:
        return {"n": 0}
    hits = [((s["fc"]["p_up25"] >= 0.5) == (s["y"]["f25"] == "UP")) for s in rows]
    conf = [(h, s) for h, s in zip(hits, rows) if abs(s["fc"]["p_up25"] - 0.5) >= 0.08]
    trusted = [h for h, s in zip(hits, rows) if s["fc"].get("trusted")]
    return {"n": len(rows), "acc": sum(hits) / len(hits),
            "n_conf": len(conf), "acc_conf": (sum(1 for h, _ in conf if h) / len(conf)) if conf else None,
            "n_trusted": len(trusted), "acc_trusted": (sum(trusted) / len(trusted)) if trusted else None}
