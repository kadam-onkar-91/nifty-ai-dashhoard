import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Only take a directional trade if the ensemble's average confidence in that
# direction is at least this high. Below this, the model reports "no edge"
# rather than forcing a low-conviction guess.
MIN_CONFIDENCE = 0.42  # roughly: at least ~42% probability mass beyond a 3-way split baseline


def apply_triple_barrier(prices, upper_mult=1.5, lower_mult=1.5, holding_period=5):
    """
    Labels each candle +1 (target hit first), -1 (SL hit first), or 0
    (neither within the holding period) — far more realistic than a simple
    'next candle up/down' label.
    """
    labels = [0] * len(prices)
    rolling_vol = prices.pct_change().rolling(window=10).std().fillna(0.002)

    for i in range(len(prices) - holding_period):
        entry_price = prices.iloc[i]
        volatility = max(rolling_vol.iloc[i], 0.0005)
        upper_barrier = entry_price * (1 + volatility * upper_mult)
        lower_barrier = entry_price * (1 - volatility * lower_mult)

        hit = 0
        for h in range(1, holding_period + 1):
            future_price = prices.iloc[i + h]
            if future_price >= upper_barrier:
                hit = 1
                break
            elif future_price <= lower_barrier:
                hit = -1
                break
        labels[i] = hit

    return labels


def _build_features(df, live_vix=None):
    """
    Richer, more informative feature set than just VIX + time-of-day.
    Every feature here is derived from data the app already computes
    (EMA, VWAP, RSI, MACD, ATR, Bollinger Bands, Volume) — nothing invented.
    """
    work = df.copy()

    # Distance-based features (normalized so they're comparable across
    # different price levels/volatility regimes, instead of raw price gaps)
    if 'EMA_20' in work.columns:
        work['Dist_EMA20_pct'] = (work['Close'] - work['EMA_20']) / work['Close']
    if 'EMA_50' in work.columns:
        work['Dist_EMA50_pct'] = (work['Close'] - work['EMA_50']) / work['Close']
    if 'VWAP' in work.columns:
        work['Dist_VWAP_pct'] = (work['Close'] - work['VWAP']) / work['Close']
    if 'ATR' in work.columns:
        work['ATR_pct'] = work['ATR'] / work['Close']
    if 'BB_Upper' in work.columns and 'BB_Lower' in work.columns:
        band_width = (work['BB_Upper'] - work['BB_Lower']).replace(0, np.nan)
        work['BB_Position'] = (work['Close'] - work['BB_Lower']) / band_width

    if 'Volume' in work.columns:
        vol_mean = work['Volume'].rolling(20, min_periods=1).mean()
        vol_std = work['Volume'].rolling(20, min_periods=1).std().replace(0, np.nan)
        work['Volume_Zscore'] = (work['Volume'] - vol_mean) / vol_std

    # Short-term momentum/return features
    work['Return_1'] = work['Close'].pct_change(1)
    work['Return_3'] = work['Close'].pct_change(3)

    work['VIX'] = live_vix if live_vix is not None else np.nan
    if isinstance(work.index, pd.DatetimeIndex):
        work['Time_Of_Day'] = work.index.hour + (work.index.minute / 60.0)
    else:
        work['Time_Of_Day'] = 10.5

    candidate_cols = [
        'RSI', 'MACD', 'Dist_EMA20_pct', 'Dist_EMA50_pct', 'Dist_VWAP_pct',
        'ATR_pct', 'BB_Position', 'Volume_Zscore', 'Return_1', 'Return_3',
        'VIX', 'Time_Of_Day'
    ]
    feature_cols = [c for c in candidate_cols if c in work.columns]

    if 'VIX' in feature_cols and work['VIX'].isna().all():
        feature_cols.remove('VIX')
        work.drop(columns=['VIX'], inplace=True)

    work[feature_cols] = work[feature_cols].replace([np.inf, -np.inf], np.nan)
    return work, feature_cols


def _walk_forward_folds(X, y, n_folds=4):
    """Expanding-window walk-forward validation — respects time order."""
    total = len(X)
    fold_size = total // (n_folds + 1)
    if fold_size < 5:
        return []

    folds = []
    for i in range(1, n_folds + 1):
        train_end = fold_size * i
        test_end = fold_size * (i + 1) if i < n_folds else total
        if test_end <= train_end:
            continue
        folds.append((
            X.iloc[:train_end], y.iloc[:train_end],
            X.iloc[train_end:test_end], y.iloc[train_end:test_end]
        ))
    return folds


def _build_ensemble():
    """
    Three different model types voting together tends to be more robust
    than any single model — each has different failure modes, so they
    partially cancel each other's noise/overfitting rather than compound it.
    """
    return {
        'rf': RandomForestClassifier(n_estimators=150, max_depth=4, random_state=42, class_weight='balanced'),
        'gb': GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42),
        'lr': LogisticRegression(max_iter=500, class_weight='balanced')
    }


def _ensemble_predict_proba(models, X_train, y_train, X_test):
    """
    Trains each model and averages their predicted class probabilities.
    Returns the averaged probability matrix aligned to sorted class labels.
    """
    classes = sorted(y_train.unique())
    proba_sum = np.zeros((len(X_test), len(classes)))
    fitted_any = False

    for model in models.values():
        try:
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)
            # Align columns to the full class list in case a model saw fewer classes
            aligned = np.zeros((len(X_test), len(classes)))
            for idx, cls in enumerate(model.classes_):
                col_idx = classes.index(cls)
                aligned[:, col_idx] = proba[:, idx]
            proba_sum += aligned
            fitted_any = True
        except Exception:
            continue

    if not fitted_any:
        return None, classes

    return proba_sum / len(models), classes


def train_and_backtest(df, live_vix=None):
    """
    Walk-forward validated ENSEMBLE model with richer features and a
    confidence threshold — no clamping, no hardcoded fallback numbers.
    Reports real numbers, or honestly says there isn't enough data /
    enough model confidence to act.
    """
    empty_result = {
        "Accuracy": "Not Enough Data", "Win Rate": "N/A", "Net PnL": "N/A",
        "Sample Size": "0 Trades", "latest_signal": 0, "model_ready": False,
        "latest_confidence": 0.0
    }

    if df is None or len(df) < 80:
        return empty_result

    work, feature_cols = _build_features(df, live_vix)
    if not feature_cols:
        return empty_result

    work['Barrier_Label'] = apply_triple_barrier(work['Close'])
    clean_df = work.dropna(subset=feature_cols + ['Barrier_Label'])

    if len(clean_df) < 50:
        return empty_result

    X = clean_df[feature_cols]
    y = clean_df['Barrier_Label']

    folds = _walk_forward_folds(X, y, n_folds=4)
    if not folds:
        return empty_result

    all_final_preds, all_actuals = [], []
    for X_train, y_train, X_test, y_test in folds:
        if y_train.nunique() < 2:
            continue
        models = _build_ensemble()
        proba, classes = _ensemble_predict_proba(models, X_train, y_train, X_test)
        if proba is None:
            continue

        for row_idx in range(len(X_test)):
            row_proba = proba[row_idx]
            best_idx = int(np.argmax(row_proba))
            best_conf = row_proba[best_idx]
            predicted_class = classes[best_idx]
            # Apply confidence threshold — below it, treat as "no trade" (label 0)
            if predicted_class != 0 and best_conf < MIN_CONFIDENCE:
                predicted_class = 0
            all_final_preds.append(predicted_class)
            all_actuals.append(y_test.iloc[row_idx])

    if not all_final_preds:
        return empty_result

    raw_acc = accuracy_score(all_actuals, all_final_preds)
    winning_trades = sum(1 for p, a in zip(all_final_preds, all_actuals) if p == a and p != 0)
    total_directional_trades = sum(1 for p in all_final_preds if p != 0)
    win_rate = (winning_trades / total_directional_trades * 100) if total_directional_trades > 0 else 0.0

    losing_trades = total_directional_trades - winning_trades
    simulated_pnl = (winning_trades * 100) - (losing_trades * 150)

    # Train final ensemble on ALL labeled data for the live prediction
    latest_signal = 0
    latest_confidence = 0.0
    model_ready = False
    try:
        if y.nunique() >= 2:
            final_models = _build_ensemble()
            last_row = work[feature_cols].iloc[[-1]].ffill().fillna(0)
            proba, classes = _ensemble_predict_proba(final_models, X, y, last_row)
            if proba is not None:
                best_idx = int(np.argmax(proba[0]))
                latest_confidence = round(float(proba[0][best_idx]), 3)
                predicted_class = classes[best_idx]
                if predicted_class != 0 and latest_confidence >= MIN_CONFIDENCE:
                    latest_signal = int(predicted_class)
                model_ready = True
    except Exception:
        pass

    return {
        "Accuracy": f"{round(raw_acc * 100, 1)}%",
        "Win Rate": f"{round(win_rate, 1)}%",
        "Net PnL": f"₹{simulated_pnl}",
        "Sample Size": f"{total_directional_trades} Trades",
        "latest_signal": latest_signal,
        "model_ready": model_ready,
        "latest_confidence": latest_confidence
        }
            
