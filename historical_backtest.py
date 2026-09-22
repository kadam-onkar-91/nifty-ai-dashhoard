"""Leakage-safe historical validation for the NIFTY AI ML component.

This module deliberately does NOT fabricate option-chain/order-flow/news history.
It validates the OHLCV+indicator ML component from a user-supplied historical
CSV using expanding walk-forward splits and a label embargo. A full end-to-end
trade-decision replay requires historical snapshots for every dashboard input.
"""
from app_logging import get_logger
logger = get_logger(__name__)

import numpy as np
import pandas as pd
from ml_engine import _build_features, _build_ensemble, _ensemble_predict_proba, apply_triple_barrier, MIN_CONFIDENCE
from sklearn.metrics import accuracy_score, confusion_matrix


def _normalize_ohlcv(df):
    if df is None or df.empty:
        raise ValueError("Historical dataset is empty")
    x = df.copy()
    rename = {c: c.strip().lower() for c in x.columns}
    x.rename(columns=rename, inplace=True)
    aliases = {
        "datetime": "Timestamp", "date": "Timestamp", "time": "Timestamp",
        "open": "Open", "high": "High", "low": "Low", "close": "Close",
        "volume": "Volume", "vol": "Volume"
    }
    x.rename(columns={c: aliases.get(c, c) for c in x.columns}, inplace=True)
    required = ["Open", "High", "Low", "Close"]
    missing = [c for c in required if c not in x.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if "Timestamp" in x.columns:
        x["Timestamp"] = pd.to_datetime(x["Timestamp"], errors="coerce")
        x = x.dropna(subset=["Timestamp"]).set_index("Timestamp")
    else:
        x.index = pd.RangeIndex(len(x))
    for c in ["Open", "High", "Low", "Close"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    if "Volume" not in x.columns:
        x["Volume"] = 0.0
    x["Volume"] = pd.to_numeric(x["Volume"], errors="coerce").fillna(0.0)
    x = x.dropna(subset=required).sort_index()
    if x.index.duplicated().any():
        x = x[~x.index.duplicated(keep="last")]
    return x


def run_ml_walk_forward(df, n_folds=5, holding_period=5, min_train=80):
    x = _normalize_ohlcv(df)
    if len(x) < max(min_train + holding_period + 20, 160):
        return {"status": "NOT_ENOUGH_DATA", "rows": len(x), "trades": 0,
                "note": "Need more historical candles for a meaningful walk-forward test."}
    work, feature_cols = _build_features(x)
    work["Barrier_Label"] = apply_triple_barrier(work["Close"], holding_period=holding_period)
    work.loc[work.index[-holding_period:], "Barrier_Label"] = np.nan
    clean = work.dropna(subset=feature_cols + ["Barrier_Label"])
    if len(clean) < min_train + 40:
        return {"status": "NOT_ENOUGH_DATA", "rows": len(clean), "trades": 0,
                "note": "Not enough clean labeled candles after indicator warm-up."}

    X, y = clean[feature_cols], clean["Barrier_Label"].astype(int)
    total = len(clean)
    fold_size = total // (n_folds + 1)
    preds, actuals, confidences = [], [], []
    for i in range(1, n_folds + 1):
        train_end = fold_size * i
        test_end = fold_size * (i + 1) if i < n_folds else total
        train_cut = train_end - holding_period
        if train_cut < min_train or test_end <= train_end:
            continue
        Xtr, ytr = X.iloc[:train_cut], y.iloc[:train_cut]
        Xte, yte = X.iloc[train_end:test_end], y.iloc[train_end:test_end]
        if ytr.nunique() < 2 or Xte.empty:
            continue
        proba, classes = _ensemble_predict_proba(_build_ensemble(), Xtr, ytr, Xte)
        if proba is None:
            continue
        for j, row in enumerate(proba):
            idx = int(np.argmax(row))
            conf = float(row[idx])
            pred = int(classes[idx])
            if pred != 0 and conf < MIN_CONFIDENCE:
                pred = 0
            preds.append(pred); actuals.append(int(yte.iloc[j])); confidences.append(conf)

    if not preds:
        return {"status": "NOT_ENOUGH_DATA", "rows": len(clean), "trades": 0,
                "note": "No valid out-of-sample folds were produced."}
    directional = [(p, a) for p, a in zip(preds, actuals) if p != 0]
    wins = sum(p == a for p, a in directional)
    losses = len(directional) - wins
    return {
        "status": "OK", "rows": len(clean), "oos_predictions": len(preds),
        "directional_trades": len(directional), "wins": wins, "losses": losses,
        "win_rate_pct": round(100 * wins / len(directional), 2) if directional else None,
        "accuracy_all_labels_pct": round(100 * accuracy_score(actuals, preds), 2),
        "trade_rate_pct": round(100 * len(directional) / len(preds), 2),
        "avg_confidence_pct": round(100 * float(np.mean(confidences)), 2),
        "confusion_matrix": confusion_matrix(actuals, preds, labels=[-1, 0, 1]).tolist(),
        "leakage_controls": ["future-unavailable final labels removed", "walk-forward expanding train", "holding-period embargo at train/test boundary", "causal feature filling only"],
        "note": "This validates only the OHLCV/ML component. It is not a full options/order-flow/news replay."
    }
