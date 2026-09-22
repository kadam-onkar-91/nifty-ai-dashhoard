"""Structured global-to-NIFTY research layer.

This module does not invent market data. It consumes the already-fetched global
market/news tables and turns them into an auditable macro context for the
NIFTY decision engine. It is deliberately secondary to domestic price,
structure, breadth, options and order-flow evidence.
"""
from app_logging import get_logger
logger = get_logger(__name__)


def _pct(df, name):
    try:
        row = df[df["Global Market / Asset"] == name]
        if row.empty:
            return None
        v = float(row["Change (%)"].iloc[0])
        return v if v == v else None
    except Exception:
        return None


def _price(df, name):
    try:
        row = df[df["Global Market / Asset"] == name]
        if row.empty:
            return None
        v = float(row["Latest Price"].iloc[0])
        return v if v > 0 else None
    except Exception:
        return None


def build_global_research(df_markets, df_news=None, india_news_sentiment=None):
    """Return an auditable macro/global context for NIFTY.

    No single global variable is allowed to override the domestic NIFTY
    structure. The output is evidence + a bounded directional bias only.
    """
    df = df_markets
    if df is None or getattr(df, "empty", True):
        return {
            "status": "UNAVAILABLE",
            "directional_bias": "UNKNOWN",
            "strength": "NONE",
            "risk_regime": "UNKNOWN",
            "evidence": [],
            "conflicts": [],
            "data_points": 0,
            "warning": "Global market table unavailable; global research is not used as a trade trigger.",
        }

    equity_names = [
        "S&P 500 (US)", "Nasdaq Composite (US)", "Dow Jones (US)",
        "Nikkei 225 (Japan)", "Shanghai Composite (China)",
        "Hang Seng (Hong Kong)", "KOSPI (South Korea)",
        "FTSE 100 (UK)", "DAX (Germany)", "CAC 40 (France)",
        "ASX 200 (Australia)", "Straits Times (Singapore)",
    ]
    equity_changes = [_pct(df, n) for n in equity_names]
    equity_changes = [x for x in equity_changes if x is not None]

    positives = sum(x > 0 for x in equity_changes)
    negatives = sum(x < 0 for x in equity_changes)
    avg_equity = sum(equity_changes) / len(equity_changes) if equity_changes else None

    vix = _price(df, "India VIX")
    dxy = _pct(df, "US Dollar Index (DXY)")
    us10y = _pct(df, "US 10-Year Treasury Yield")
    crude = _pct(df, "Crude Oil (WTI)")
    gold = _pct(df, "Gold")
    usdinr = _pct(df, "USD/INR")

    bull = 0
    bear = 0
    evidence = []
    conflicts = []

    # Broad global equity risk appetite: secondary evidence only.
    if positives >= negatives + 3:
        bull += 2
        evidence.append(f"Global equities broadly risk-on ({positives} positive vs {negatives} negative markets).")
    elif negatives >= positives + 3:
        bear += 2
        evidence.append(f"Global equities broadly risk-off ({negatives} negative vs {positives} positive markets).")
    elif equity_changes:
        evidence.append(f"Global equities mixed ({positives} positive / {negatives} negative); no strong global equity impulse.")

    # India VIX: higher volatility is risk context, not an automatic sell signal.
    if vix is not None:
        if vix >= 20:
            bear += 1
            evidence.append(f"India VIX elevated at {vix:.2f}; risk/volatility is elevated.")
        elif vix < 14:
            bull += 1
            evidence.append(f"India VIX relatively calm at {vix:.2f}.")
        else:
            evidence.append(f"India VIX is {vix:.2f}; neither extreme is detected by this rule.")

    # USD pressure is normally a macro headwind for risk assets; keep it as a
    # small contextual factor rather than a standalone trade trigger.
    if dxy is not None:
        if dxy > 0.25:
            bear += 1
            evidence.append(f"DXY is firmer ({dxy:+.2f}%), a potential risk-asset headwind.")
        elif dxy < -0.25:
            bull += 1
            evidence.append(f"DXY is softer ({dxy:+.2f}%), a potential relief factor for risk assets.")

    if us10y is not None:
        if us10y > 0.35:
            bear += 1
            evidence.append(f"US 10Y yield is rising ({us10y:+.2f}% daily move); rates are a potential headwind.")
        elif us10y < -0.35:
            bull += 1
            evidence.append(f"US 10Y yield is falling ({us10y:+.2f}% daily move); rates pressure is easing.")

    if crude is not None:
        if crude > 1.0:
            bear += 1
            evidence.append(f"WTI is up {crude:+.2f}%; higher oil can matter for India's imported energy cost.")
        elif crude < -1.0:
            bull += 1
            evidence.append(f"WTI is down {crude:+.2f}%; oil pressure is easing.")

    if usdinr is not None:
        if usdinr > 0.25:
            bear += 1
            evidence.append(f"USD/INR is higher ({usdinr:+.2f}%), indicating rupee pressure.")
        elif usdinr < -0.25:
            bull += 1
            evidence.append(f"USD/INR is lower ({usdinr:+.2f}%), indicating rupee relief.")

    # Gold is not a direct NIFTY signal. Its move is retained as risk/defensive
    # evidence only, preventing the common mistake of treating gold strength as
    # an automatic equity sell signal.
    if gold is not None and gold > 1.0:
        evidence.append(f"Gold is strong ({gold:+.2f}%); this is treated as defensive/macro evidence, not an automatic NIFTY sell signal.")

    if india_news_sentiment:
        s = str(india_news_sentiment).upper()
        if "BULLISH" in s:
            bull += 1
            evidence.append("India news sentiment is classified bullish by the live RSS heuristic.")
        elif "BEARISH" in s:
            bear += 1
            evidence.append("India news sentiment is classified bearish by the live RSS heuristic.")
        else:
            evidence.append("India news sentiment is mixed/neutral.")

    if bull >= bear + 3:
        bias, strength = "BUY", "STRONG"
    elif bear >= bull + 3:
        bias, strength = "SELL", "STRONG"
    elif bull > bear:
        bias, strength = "BUY", "MODERATE"
    elif bear > bull:
        bias, strength = "SELL", "MODERATE"
    else:
        bias, strength = "NEUTRAL", "WEAK"

    if positives >= negatives + 3 and negatives >= positives + 3:
        conflicts.append("Internal global breadth calculation conflict.")
    if dxy is not None and us10y is not None and dxy > 0.25 and us10y < -0.35:
        conflicts.append("USD and Treasury-yield signals are pulling in different macro directions.")
    if vix is not None and vix >= 20 and bias == "BUY":
        conflicts.append("High VIX conflicts with the otherwise bullish global score; domestic confirmation must be strong.")

    return {
        "status": "AVAILABLE",
        "directional_bias": bias,
        "strength": strength,
        "risk_regime": "RISK_ON" if bull >= bear + 2 else "RISK_OFF" if bear >= bull + 2 else "MIXED",
        "score": int(bull - bear),
        "bull_points": int(bull),
        "bear_points": int(bear),
        "data_points": len(equity_changes),
        "metrics": {
            "global_equity_avg_change_pct": round(avg_equity, 3) if avg_equity is not None else None,
            "india_vix": vix,
            "dxy_change_pct": dxy,
            "us10y_change_pct": us10y,
            "wti_change_pct": crude,
            "gold_change_pct": gold,
            "usd_inr_change_pct": usdinr,
            "global_equity_positive_count": positives,
            "global_equity_negative_count": negatives,
        },
        "evidence": evidence[:12],
        "conflicts": conflicts[:8],
        "news_source": "Google News RSS heuristic" if df_news is not None else "Not supplied",
        "note": "Global research is contextual evidence. NIFTY price structure, breadth, options, order flow and risk gates remain primary.",
    }
