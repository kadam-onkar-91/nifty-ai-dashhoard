"""Named strategy library + adaptive ensemble for the NIFTY intraday engine.

The strategy layer is intentionally independent from the existing AI/self-learning
research engine.  A strategy can produce an entry candidate, but the final trade
requires the AI research direction to agree with that strategy direction.

The module uses only completed bars for signals and chronological walk-forward
validation.  It does not claim that a backtest guarantees future performance.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

STRATEGY_VERSION = "v3_32_strategy_pattern_ensemble_live_ready"
# Walk-forward stats use the same risk convention as the live engine so the
# displayed strategy statistics are not based on a different hidden R:R model.
WF_SL_POINTS = 20.0
WF_RR = 1.5
DEFAULT_HORIZON = 12
MIN_WF_SAMPLES = 20
MIN_WF_RESOLVED_FOR_RATE = 10
VERIFIED_WF_SAMPLES = 30
RECENT_SETUP_BARS = 6  # a short execution window; prevents a valid setup disappearing between refreshes

STRATEGY_META = {
    "modified_920_orb": {"name": "Modified 9:20 ORB", "type": "directional", "regimes": {"TREND_UP", "TREND_DOWN", "HIGH_VOL", "RANGE"}},
    "classic_30m_orb": {"name": "Classic 30-Min ORB", "type": "directional", "regimes": {"TREND_UP", "TREND_DOWN", "HIGH_VOL"}},
    "ema_trap_fakeout": {"name": "EMA Trap / Fakeout Scalping", "type": "directional", "regimes": {"TREND_UP", "TREND_DOWN", "HIGH_VOL", "RANGE"}},
    "oi_writer_zones": {"name": "Option-Seller OI Zones", "type": "directional", "regimes": {"TREND_UP", "TREND_DOWN", "RANGE", "HIGH_VOL"}},
    "brahmastra": {"name": "Brahmastra Triple Confirmation", "type": "directional", "regimes": {"TREND_UP", "TREND_DOWN", "HIGH_VOL"}},
    "expiry_iron_condor": {"name": "Expiry Iron Condor", "type": "neutral", "regimes": {"RANGE", "LOW_VOL"}},
    "daily_liquidity_levels": {"name": "Daily Liquidity Levels", "type": "directional", "regimes": {"TREND_UP", "TREND_DOWN", "RANGE", "HIGH_VOL"}},
    "previous_day_liquidity_sweep": {"name": "Previous-Day Liquidity Sweep", "type": "directional", "regimes": {"TREND_UP", "TREND_DOWN", "RANGE", "HIGH_VOL"}},
    "structure_supply_demand": {"name": "Price Action Supply/Demand", "type": "directional", "regimes": {"TREND_UP", "TREND_DOWN", "RANGE"}},
    # New unique strategies supplied by the user. Duplicates of existing strategies
    # are intentionally NOT registered again (PDH/PDL sweep, ORB+VWAP+Volume and
    # VWAP Pullback were already represented by the existing library).
    "opening_drive": {"name": "Opening Drive", "type": "directional", "regimes": {"TREND_UP", "TREND_DOWN", "HIGH_VOL"}},
    "initial_balance_breakout": {"name": "Initial Balance Breakout (IBB)", "type": "directional", "regimes": {"TREND_UP", "TREND_DOWN", "HIGH_VOL"}},
    "gap_and_go": {"name": "Gap-and-Go", "type": "directional", "regimes": {"TREND_UP", "TREND_DOWN", "HIGH_VOL"}},
    "gap_fade_fill": {"name": "Gap-Fade / Gap-Fill", "type": "directional", "regimes": {"RANGE", "LOW_VOL", "TREND_UP", "TREND_DOWN"}},
    "fibonacci_price_action": {"name": "Fibonacci Retracement + Price Action", "type": "directional", "regimes": {"TREND_UP", "TREND_DOWN", "RANGE"}},
    "pivot_reversal": {"name": "Pivot Point Reversal", "type": "directional", "regimes": {"RANGE", "TREND_UP", "TREND_DOWN"}},
    "rsi_divergence_reversal": {"name": "RSI Divergence Reversal", "type": "directional", "regimes": {"RANGE", "HIGH_VOL"}},
    "bb_squeeze_breakout": {"name": "Bollinger Band Squeeze Breakout", "type": "directional", "regimes": {"HIGH_VOL", "TREND_UP", "TREND_DOWN"}},
    "vwap_mean_reversion": {"name": "VWAP Mean-Reversion", "type": "directional", "regimes": {"RANGE", "LOW_VOL"}},
    "liquidity_sweep_fvg": {"name": "Liquidity Sweep + FVG Reversal", "type": "directional", "regimes": {"RANGE", "TREND_UP", "TREND_DOWN", "HIGH_VOL"}},
    "orb_vwap_volume_retest": {"name": "ORB + VWAP + Volume + Retest", "type": "directional", "regimes": {"TREND_UP", "TREND_DOWN", "HIGH_VOL"}},
    "vwap_pullback_structure": {"name": "VWAP Pullback + Market Structure", "type": "directional", "regimes": {"TREND_UP", "TREND_DOWN"}},
    "opening_range_pd_confluence": {"name": "Opening Range + Previous-Day Level Confluence", "type": "directional", "regimes": {"TREND_UP", "TREND_DOWN", "HIGH_VOL", "RANGE"}},
    "cpr_breakout_vwap_volume": {"name": "CPR Breakout + VWAP + Volume", "type": "directional", "regimes": {"TREND_UP", "TREND_DOWN", "HIGH_VOL"}},
    "market_structure_break_retest": {"name": "Market Structure Break + Retest", "type": "directional", "regimes": {"TREND_UP", "TREND_DOWN"}},
    "vwap_sr_rejection": {"name": "VWAP + Support/Resistance Rejection", "type": "directional", "regimes": {"RANGE", "TREND_UP", "TREND_DOWN"}},
    "ema2050_vwap_pullback": {"name": "EMA 20/50 Pullback + VWAP", "type": "directional", "regimes": {"TREND_UP", "TREND_DOWN"}},
    "inside_bar_vwap_volume": {"name": "Inside-Bar Breakout + VWAP + Volume", "type": "directional", "regimes": {"TREND_UP", "TREND_DOWN", "HIGH_VOL"}},
    "bb_mean_reversion_vwap": {"name": "Bollinger Band Mean Reversion + VWAP", "type": "directional", "regimes": {"RANGE", "LOW_VOL"}},
    # Chart-pattern strategies supplied later by the user. These are distinct
    # from the existing ORB/VWAP/liquidity strategies above and are evaluated
    # from the same live Upstox OHLCV bars + locally calculated indicators.
    "bull_flag_breakout": {"name": "Bull Flag Breakout", "type": "directional", "regimes": {"TREND_UP", "HIGH_VOL"}},
    "inverse_head_shoulders": {"name": "Inverse Head & Shoulders", "type": "directional", "regimes": {"TREND_UP", "RANGE", "HIGH_VOL"}},
    "ascending_triangle": {"name": "Ascending Triangle", "type": "directional", "regimes": {"TREND_UP", "RANGE", "HIGH_VOL"}},
    "double_bottom": {"name": "Double Bottom / W-Pattern", "type": "directional", "regimes": {"TREND_UP", "RANGE"}},
    "cup_handle": {"name": "Cup and Handle", "type": "directional", "regimes": {"TREND_UP", "RANGE"}},
    "descending_triangle": {"name": "Descending Triangle", "type": "directional", "regimes": {"TREND_DOWN", "RANGE", "HIGH_VOL"}},
    "falling_wedge": {"name": "Falling Wedge", "type": "directional", "regimes": {"TREND_UP", "TREND_DOWN", "RANGE"}},
    "head_shoulders": {"name": "Head and Shoulders", "type": "directional", "regimes": {"TREND_DOWN", "RANGE", "HIGH_VOL"}},
    "symmetrical_triangle": {"name": "Symmetrical Triangle Breakout", "type": "directional", "regimes": {"TREND_UP", "TREND_DOWN", "RANGE", "HIGH_VOL"}},
    "rectangle_breakout": {"name": "Box / Rectangle Range Breakout", "type": "directional", "regimes": {"TREND_UP", "TREND_DOWN", "RANGE", "HIGH_VOL"}},
}



def _num(v, default=np.nan):
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h = pd.to_numeric(df["High"], errors="coerce")
    l = pd.to_numeric(df["Low"], errors="coerce")
    c = pd.to_numeric(df["Close"], errors="coerce")
    prev = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-prev).abs(), (l-prev).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=max(5, n//2)).mean()


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    for c in ("Open", "High", "Low", "Close", "Volume"):
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce")
    if "Volume" not in x.columns:
        x["Volume"] = 0.0
    x["ATR"] = _atr(x)
    x["EMA20"] = x["Close"].ewm(span=20, adjust=False).mean()
    x["EMA50"] = x["Close"].ewm(span=50, adjust=False).mean()
    x["EMA9"] = x["Close"].ewm(span=9, adjust=False).mean()
    x["RET5"] = x["Close"].pct_change(5)
    x["VOL20"] = x["Volume"].rolling(20, min_periods=5).mean()
    x["VWAP"] = x["VWAP"] if "VWAP" in x.columns else ((x["High"]+x["Low"]+x["Close"])/3*x["Volume"]).cumsum()/x["Volume"].replace(0,np.nan).cumsum()
    x["RSI14"] = _rsi(x["Close"])
    x["BB_MID"] = x["Close"].rolling(20, min_periods=10).mean()
    x["BB_STD"] = x["Close"].rolling(20, min_periods=10).std(ddof=0)
    x["BB_UPPER"] = x["BB_MID"] + 2.0*x["BB_STD"]
    x["BB_LOWER"] = x["BB_MID"] - 2.0*x["BB_STD"]
    x["BB_WIDTH"] = (x["BB_UPPER"]-x["BB_LOWER"])/x["BB_MID"].replace(0,np.nan)
    x["BB_WIDTH_PCTL"] = x["BB_WIDTH"].rolling(80, min_periods=20).rank(pct=True)
    x["SUPERTREND_DIR"] = _supertrend(x, length=20, factor=2.0)
    return x


def _times(x: pd.DataFrame) -> pd.Series:
    try:
        idx = pd.to_datetime(x.index)
        return pd.Series(idx, index=x.index).dt.time
    except Exception:
        return pd.Series(pd.NaT, index=x.index)


def _session_range(x: pd.DataFrame, start_hm: str, end_hm: str, *, completed_only: bool = False):
    try:
        ts = pd.to_datetime(x.index)
        if len(ts) == 0: return None
        # ORB/IBB/opening-drive ranges must be built from the completed opening
        # window only. The old implementation included the CURRENT breakout
        # candle inside the range, making conditions such as Close > range High
        # mathematically impossible until much later (and sometimes forever).
        current_day = ts[-1].date()
        start = pd.Timestamp(start_hm).time()
        end = pd.Timestamp(end_hm).time()
        if completed_only and ts[-1].time() <= end:
            return None
        m = pd.Series((ts.date == current_day) & (ts.time >= start) & (ts.time <= end), index=x.index)
        y = x.loc[m.values]
        if y.empty:
            return None
        return float(y["High"].max()), float(y["Low"].min())
    except Exception:
        return None


def _candle_confirmation(r, p) -> int:
    o, c, h, l = map(_num, (r["Open"], r["Close"], r["High"], r["Low"]))
    po, pc = _num(p["Open"]), _num(p["Close"])
    body = abs(c-o); rng = max(h-l, 1e-9)
    lower = min(o,c)-l; upper = h-max(o,c)
    bull_engulf = pc < po and c > o and c >= po and o <= pc
    bear_engulf = pc > po and c < o and o >= pc and c <= po
    hammer = lower >= max(2*body, 0.25*rng) and c >= o
    shooting = upper >= max(2*body, 0.25*rng) and c <= o
    return 1 if bull_engulf or hammer else -1 if bear_engulf or shooting else 0


def _prev_day_levels(x: pd.DataFrame):
    try:
        ts = pd.to_datetime(x.index)
        d = pd.Series(ts.date, index=x.index)
        days = list(pd.unique(d.dropna()))
        if len(days) < 2:
            return None
        prev = days[-2]
        y = x.loc[d == prev]
        if y.empty:
            return None
        return float(y["High"].max()), float(y["Low"].min()), float(y["Open"].iloc[0]), float(y["Close"].iloc[-1])
    except Exception:
        return None


def _rsi(x: pd.Series, n=14):
    d = x.diff(); up = d.clip(lower=0).rolling(n).mean(); dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0,np.nan)
    return 100 - 100/(1+rs)


def _macd(x: pd.Series):
    fast=x.ewm(span=12,adjust=False).mean(); slow=x.ewm(span=26,adjust=False).mean(); line=fast-slow; sig=line.ewm(span=9,adjust=False).mean()
    return line, sig


def _supertrend(x: pd.DataFrame, length: int = 20, factor: float = 2.0):
    """Return Supertrend direction using completed OHLC bars only.

    +1 = bullish, -1 = bearish.  This is calculated locally from the same
    live Upstox OHLC dataframe; it never invents an external indicator value.
    """
    h=x["High"].astype(float); l=x["Low"].astype(float); c=x["Close"].astype(float)
    prev=c.shift(1)
    tr=pd.concat([(h-l).abs(), (h-prev).abs(), (l-prev).abs()], axis=1).max(axis=1)
    atr=tr.rolling(length, min_periods=length).mean()
    hl2=(h+l)/2.0
    upper=hl2+factor*atr; lower=hl2-factor*atr
    final_upper=upper.copy(); final_lower=lower.copy(); direction=pd.Series(index=x.index, dtype=float)
    if len(x): direction.iloc[0]=1
    for i in range(1,len(x)):
        if not np.isfinite(atr.iloc[i]):
            direction.iloc[i]=direction.iloc[i-1]; continue
        if np.isfinite(final_upper.iloc[i-1]) and c.iloc[i-1] <= final_upper.iloc[i-1]:
            final_upper.iloc[i]=min(upper.iloc[i], final_upper.iloc[i-1])
        else:
            final_upper.iloc[i]=upper.iloc[i]
        if np.isfinite(final_lower.iloc[i-1]) and c.iloc[i-1] >= final_lower.iloc[i-1]:
            final_lower.iloc[i]=max(lower.iloc[i], final_lower.iloc[i-1])
        else:
            final_lower.iloc[i]=lower.iloc[i]
        prev_dir=direction.iloc[i-1]
        if prev_dir >= 0:
            direction.iloc[i] = -1 if c.iloc[i] < final_lower.iloc[i] else 1
        else:
            direction.iloc[i] = 1 if c.iloc[i] > final_upper.iloc[i] else -1
    return direction


def _oi_signal(option_chain: Any, price: float) -> int:
    try:
        oc = option_chain
        if oc is None or getattr(oc, "empty", True) or "Strike" not in oc.columns:
            return 0
        oc = oc.copy()
        for c in ("Strike","Call OI","Put OI","Call OI Change","Put OI Change"):
            if c in oc.columns: oc[c]=pd.to_numeric(oc[c],errors="coerce")
        near = oc[(oc["Strike"] >= price-150) & (oc["Strike"] <= price+150)]
        if near.empty: return 0
        call = float(near["Call OI"].sum()) if "Call OI" in near.columns else 0
        put = float(near["Put OI"].sum()) if "Put OI" in near.columns else 0
        call_ch = float(near["Call OI Change"].sum()) if "Call OI Change" in near.columns else 0
        put_ch = float(near["Put OI Change"].sum()) if "Put OI Change" in near.columns else 0
        # Writer-zone interpretation is used as evidence, not certainty.
        if put_ch > max(abs(call_ch)*1.15, 0) and put >= call*0.85: return 1
        if call_ch > max(abs(put_ch)*1.15, 0) and call >= put*0.85: return -1
        if put > call*1.35: return 1
        if call > put*1.35: return -1
    except Exception:
        return 0
    return 0



def _prev_session_open_gap(x):
    try:
        ts=pd.to_datetime(x.index); days=pd.Series(ts.date,index=x.index); uniq=list(pd.unique(days.dropna()))
        if len(uniq)<2: return None
        cur,prev=uniq[-1],uniq[-2]
        curm=(days==cur).values; prevm=(days==prev).values
        c=x.loc[curm]; p=x.loc[prevm]
        if c.empty or p.empty: return None
        return float(c["Open"].iloc[0]), float(p["Close"].iloc[-1]), float(p["High"].max()), float(p["Low"].min())
    except Exception: return None


def _pivot_levels(x):
    lv=_prev_day_levels(x)
    if not lv: return None
    h,l,o,c=lv; p=(h+l+c)/3.0
    return p, 2*p-l, 2*p-h, p-(h-l), p+(h-l)


def _last_swing(x, look=20):
    y=x.iloc[-look-1:-1] if len(x)>look+1 else x.iloc[:-1]
    if len(y)<5: return None
    hi=float(y["High"].max()); lo=float(y["Low"].min())
    ih=int(y["High"].values.argmax()); il=int(y["Low"].values.argmin())
    return hi,lo,ih,il


def _fvg_reversal(x, direction):
    if len(x)<4: return False
    a=x.iloc[-3]; b=x.iloc[-2]; r=x.iloc[-1]
    atr=_num(r["ATR"],0)
    if direction>0:
        # bullish 3-candle imbalance: current/third low above first high,
        # followed by a pullback that rejects the gap and closes bullish.
        gap_low=_num(a["High"]); gap_high=_num(r["Low"])
        return gap_high>gap_low and _num(b["Low"])<=gap_high and _num(r["Close"])>_num(r["Open"]) and (_num(r["Close"])-_num(r["Open"]))>=0.20*atr
    gap_low=_num(r["High"]); gap_high=_num(a["Low"])
    return gap_high>gap_low and _num(b["High"])>=gap_low and _num(r["Close"])<_num(r["Open"]) and (_num(r["Open"])-_num(r["Close"]))>=0.20*atr

def _lin_slope(values) -> float:
    """Normalized linear slope; used only for tolerant chart-shape detection."""
    a=np.asarray(values,dtype=float)
    if len(a)<3 or not np.isfinite(a).all(): return 0.0
    xx=np.arange(len(a),dtype=float)
    return float(np.polyfit(xx,a,1)[0])


def _pattern_pivots(x: pd.DataFrame, look: int = 36, span: int = 2):
    """Return simple confirmed local highs/lows from completed candles."""
    y=x.iloc[-look:].copy()
    if len(y)<12: return [], []
    highs=[]; lows=[]
    for i in range(span, len(y)-span):
        h=float(y['High'].iloc[i]); l=float(y['Low'].iloc[i])
        if h>=float(y['High'].iloc[i-span:i+span+1].max()): highs.append((i,h))
        if l<=float(y['Low'].iloc[i-span:i+span+1].min()): lows.append((i,l))
    return highs,lows


def _pattern_breakout(x: pd.DataFrame, level: float, direction: int, atr: float, volume=True) -> bool:
    r=x.iloc[-1]; p=x.iloc[-2]
    c=float(r['Close']); o=float(r['Open']); body=abs(c-o)
    vol_ok=float(r['Volume'])>=1.05*max(float(r['VOL20']),1.0) if volume else True
    if direction>0:
        return c>level+0.03*atr and c>o and body>=0.20*atr and vol_ok
    return c<level-0.03*atr and c<o and body>=0.20*atr and vol_ok


def _triangle_shape(x: pd.DataFrame, direction: str, look: int = 24):
    y=x.iloc[-look:]
    if len(y)<16: return None
    hs=y['High'].rolling(3,center=True).max(); ls=y['Low'].rolling(3,center=True).min()
    hi_idx=[i for i in range(1,len(y)-1) if y['High'].iloc[i]>=y['High'].iloc[i-1:i+2].max()]
    lo_idx=[i for i in range(1,len(y)-1) if y['Low'].iloc[i]<=y['Low'].iloc[i-1:i+2].min()]
    if len(hi_idx)<2 or len(lo_idx)<2: return None
    hvals=[float(y['High'].iloc[i]) for i in hi_idx[-4:]]; lvals=[float(y['Low'].iloc[i]) for i in lo_idx[-4:]]
    hslope=_lin_slope(hvals); lslope=_lin_slope(lvals)
    atr=max(float(y['ATR'].iloc[-1]),1e-9)
    if direction=='ascending' and abs(hslope)<=0.35*atr and lslope>0.08*atr:
        return max(hvals), hslope, lslope
    if direction=='descending' and abs(lslope)<=0.35*atr and hslope<-0.08*atr:
        return min(lvals), hslope, lslope
    if direction=='symmetric' and hslope<-0.08*atr and lslope>0.08*atr:
        return (max(hvals),min(lvals)), hslope, lslope
    return None


def _signal_rules(x: pd.DataFrame, name: str, option_chain=None) -> int:
    if len(x) < 60: return 0
    r=x.iloc[-1]; p=x.iloc[-2]; close=_num(r["Close"]); high=_num(r["High"]); low=_num(r["Low"]); atr=_num(r["ATR"],0)
    if not math.isfinite(close) or atr<=0: return 0
    ema20=_num(r["EMA20"]); ema50=_num(r["EMA50"]); vwap=_num(r["VWAP"])
    body=abs(close-_num(r["Open"])); vol_ok=_num(r["Volume"],0)>=1.15*max(_num(r["VOL20"],0),1)

    if name == "modified_920_orb":
        rr=_session_range(x,"09:15:00","09:20:00", completed_only=True)
        if not rr: return 0
        rh,rl=rr; conf=_candle_confirmation(r,p)
        # Modified ORB: breakout -> retest of the broken 9:20 boundary ->
        # completed confirmation candle. No blind first-break entry.
        recent=x.iloc[-6:-1]
        up_break=(float(recent["High"].max())>rh) and low <= rh+0.25*atr and close>rh and conf==1
        dn_break=(float(recent["Low"].min())<rl) and high >= rl-0.25*atr and close<rl and conf==-1
        return 1 if up_break else -1 if dn_break else 0

    if name == "classic_30m_orb":
        rr=_session_range(x,"09:15:00","09:45:00", completed_only=True)
        if not rr: return 0
        rh,rl=rr
        if close>rh and (vol_ok or body>=0.5*atr): return 1
        if close<rl and (vol_ok or body>=0.5*atr): return -1
        return 0

    if name == "ema_trap_fakeout":
        prev_hi=float(x["High"].iloc[-6:-1].max()); prev_lo=float(x["Low"].iloc[-6:-1].min())
        # Break above resistance then fail and reclaim EMA = long trap reversal.
        if _num(p["High"])>prev_hi and close>ema20 and _num(r["Close"])>_num(r["Open"]) and body>=0.25*atr: return 1
        if _num(p["Low"])<prev_lo and close<ema20 and _num(r["Close"])<_num(r["Open"]) and body>=0.25*atr: return -1
        return 0

    if name == "oi_writer_zones":
        return _oi_signal(option_chain, close)

    if name == "brahmastra":
        macd,sig=_macd(x["Close"]); rsi=_rsi(x["Close"])
        st_dir=_num(r.get("SUPERTREND_DIR"), 0)
        vwap_near=abs(close-vwap) <= 0.35*atr
        bull=(st_dir > 0 and vwap_near and macd.iloc[-1]>sig.iloc[-1] and macd.iloc[-2]<=sig.iloc[-2] and rsi.iloc[-1]>=50)
        bear=(st_dir < 0 and vwap_near and macd.iloc[-1]<sig.iloc[-1] and macd.iloc[-2]>=sig.iloc[-2] and rsi.iloc[-1]<=50)
        return 1 if bull else -1 if bear else 0

    if name == "expiry_iron_condor":
        # Non-directional: never produces a BUY/SELL trigger. It is reported as
        # a RANGE strategy candidate only and therefore cannot by itself open a directional trade.
        return 0

    if name == "daily_liquidity_levels":
        lv=_prev_day_levels(x)
        if not lv: return 0
        dh,dl,do,dc=lv; conf=_candle_confirmation(r,p)
        # Daily liquidity level can be previous-day high, low, or opening level.
        daily_open_hit=(low<=do<=high)
        if (low<dl or daily_open_hit) and close>max(dl, do-0.20*atr) and conf==1: return 1
        if (high>dh or daily_open_hit) and close<min(dh, do+0.20*atr) and conf==-1: return -1
        return 0

    if name == "previous_day_liquidity_sweep":
        lv=_prev_day_levels(x)
        if not lv: return 0
        dh,dl,do,dc=lv; conf=_candle_confirmation(r,p)
        prev_hi=_num(p["High"]); prev_lo=_num(p["Low"])
        # Sweep previous-day extreme, then require a confirming candle whose
        # high/low takes out the prior candle in the reversal direction.
        if low<dl and close>_num(r["Open"]) and high>prev_hi and conf==1: return 1
        if high>dh and close<_num(r["Open"]) and low<prev_lo and conf==-1: return -1
        return 0

    if name == "structure_supply_demand":
        hh=float(x["High"].iloc[-20:-3].max()); ll=float(x["Low"].iloc[-20:-3].min())
        bull_structure=close>ema50 and close>hh
        bear_structure=close<ema50 and close<ll
        # Demand/supply retest + minimum 2.5R structural room.
        recent_high=float(x["High"].iloc[-20:-1].max()); recent_low=float(x["Low"].iloc[-20:-1].min())
        long_risk=max(close-(ema20-0.35*atr), 0.25*atr); short_risk=max((ema20+0.35*atr)-close, 0.25*atr)
        long_reward=max(recent_high-close, 0.0); short_reward=max(close-recent_low, 0.0)
        if bull_structure and low<=ema20+0.35*atr and close>_num(p["Close"]) and long_reward/max(long_risk,1e-9)>=2.5: return 1
        if bear_structure and high>=ema20-0.35*atr and close<_num(p["Close"]) and short_reward/max(short_risk,1e-9)>=2.5: return -1
        return 0
    if name == "structure_supply_demand":
        hh=float(x["High"].iloc[-20:-3].max()); ll=float(x["Low"].iloc[-20:-3].min())
        bull_structure=close>ema50 and close>hh; bear_structure=close<ema50 and close<ll
        recent_high=float(x["High"].iloc[-20:-1].max()); recent_low=float(x["Low"].iloc[-20:-1].min())
        long_risk=max(close-(ema20-0.35*atr), 0.25*atr); short_risk=max((ema20+0.35*atr)-close, 0.25*atr)
        if bull_structure and low<=ema20+0.35*atr and close>_num(p["Close"]) and (recent_high-close)/max(long_risk,1e-9)>=2.5: return 1
        if bear_structure and high>=ema20-0.35*atr and close<_num(p["Close"]) and (close-recent_low)/max(short_risk,1e-9)>=2.5: return -1
        return 0

    if name == "opening_drive":
        rr=_session_range(x,"09:15:00","09:30:00", completed_only=True)
        if not rr: return 0
        rh,rl=rr; op=_num(x.loc[p.name]["Open"]) if p.name in x.index else _num(p["Open"])
        # Directional opening drive: first 15m closes near its extreme and the current candle continues it.
        first=x.iloc[-min(len(x),20):]
        if close>rh and close>vwap and body>=0.35*atr and close>=high-0.25*atr: return 1
        if close<rl and close<vwap and body>=0.35*atr and close<=low+0.25*atr: return -1
        return 0

    if name == "initial_balance_breakout":
        rr=_session_range(x,"09:15:00","10:15:00", completed_only=True)
        if not rr: return 0
        rh,rl=rr
        if close>rh and close>vwap and body>=0.35*atr: return 1
        if close<rl and close<vwap and body>=0.35*atr: return -1
        return 0

    if name == "gap_and_go":
        g=_prev_session_open_gap(x)
        if not g: return 0
        op,pc,ph,pl=g; gap=op-pc
        if abs(gap)<0.35*atr: return 0
        if gap>0 and close>op and close>vwap and body>=0.20*atr: return 1
        if gap<0 and close<op and close<vwap and body>=0.20*atr: return -1
        return 0

    if name == "gap_fade_fill":
        g=_prev_session_open_gap(x)
        if not g: return 0
        op,pc,ph,pl=g; gap=op-pc
        if abs(gap)<0.35*atr: return 0
        # Price re-enters the prior close zone after failing to hold the gap.
        if gap>0 and low<=pc and close>pc and close<_num(p["Close"]): return 1
        if gap<0 and high>=pc and close<pc and close>_num(p["Close"]): return -1
        return 0

    if name == "fibonacci_price_action":
        sw=_last_swing(x,30)
        if not sw: return 0
        hi,lo,ih,il=sw; rng=hi-lo
        if rng<=0.8*atr: return 0
        up=ih<il
        fib618=hi-0.618*rng if up else lo+0.618*rng
        fib50=hi-0.50*rng if up else lo+0.50*rng
        conf=_candle_confirmation(r,p)
        near=abs(close-fib618)<=0.30*atr or abs(close-fib50)<=0.30*atr
        if up and near and close>ema20 and conf==1: return 1
        if not up and near and close<ema20 and conf==-1: return -1
        return 0

    if name == "pivot_reversal":
        pv=_pivot_levels(x)
        if not pv: return 0
        p0,r1,s1,s2,r2=pv; conf=_candle_confirmation(r,p)
        if low<=s1+0.20*atr and close>s1 and conf==1: return 1
        if high>=r1-0.20*atr and close<r1 and conf==-1: return -1
        return 0

    if name == "rsi_divergence_reversal":
        rsi=x["RSI14"]
        if len(x)<25: return 0
        # Simple confirmed 5-bar divergence, deliberately requiring price confirmation.
        prev=rsi.iloc[-8:-3]; prevc=x["Close"].iloc[-8:-3]
        conf=_candle_confirmation(r,p)
        if float(x["Low"].iloc[-1])<float(prevc.min()) and float(rsi.iloc[-1])>float(prev.min()) and rsi.iloc[-1]<40 and conf==1: return 1
        if float(x["High"].iloc[-1])>float(prevc.max()) and float(rsi.iloc[-1])<float(prev.max()) and rsi.iloc[-1]>60 and conf==-1: return -1
        return 0

    if name == "bb_squeeze_breakout":
        bw=_num(r["BB_WIDTH_PCTL"]); prev_bw=_num(p["BB_WIDTH_PCTL"])
        if not math.isfinite(bw): return 0
        if min(bw,prev_bw)>0.25: return 0
        if close>_num(r["BB_UPPER"]) and body>=0.35*atr and close>vwap: return 1
        if close<_num(r["BB_LOWER"]) and body>=0.35*atr and close<vwap: return -1
        return 0

    if name == "vwap_mean_reversion":
        dist=close-vwap
        if dist < -1.2*atr and close>_num(p["Close"]) and _num(r["RSI14"])<38: return 1
        if dist > 1.2*atr and close<_num(p["Close"]) and _num(r["RSI14"])>62: return -1
        return 0

    if name == "liquidity_sweep_fvg":
        lv=_prev_day_levels(x); conf=_candle_confirmation(r,p)
        if lv:
            dh,dl,do,dc=lv
            if low<dl and close>dl and conf==1 and _fvg_reversal(x,1): return 1
            if high>dh and close<dh and conf==-1 and _fvg_reversal(x,-1): return -1
        return 0

    if name == "orb_vwap_volume_retest":
        rr=_session_range(x,"09:15:00","09:30:00", completed_only=True)
        if not rr: return 0
        rh,rl=rr; recent=x.iloc[-6:-1]; conf=_candle_confirmation(r,p)
        broke_up=float(recent["High"].max())>rh; broke_dn=float(recent["Low"].min())<rl
        if broke_up and low<=rh+0.25*atr and close>rh and close>vwap and vol_ok and conf==1: return 1
        if broke_dn and high>=rl-0.25*atr and close<rl and close<vwap and vol_ok and conf==-1: return -1
        return 0

    if name == "vwap_pullback_structure":
        trend_up=ema20>ema50 and close>ema50; trend_dn=ema20<ema50 and close<ema50
        conf=_candle_confirmation(r,p)
        if trend_up and low<=vwap+0.25*atr and close>vwap and conf==1: return 1
        if trend_dn and high>=vwap-0.25*atr and close<vwap and conf==-1: return -1
        return 0

    if name == "opening_range_pd_confluence":
        rr=_session_range(x,"09:15:00","09:30:00", completed_only=True); lv=_prev_day_levels(x)
        if not rr or not lv: return 0
        rh,rl=rr; dh,dl,do,dc=lv; conf=_candle_confirmation(r,p)
        bull=(close>rh and abs(close-dh)<=0.45*atr and close>vwap and conf==1)
        bear=(close<rl and abs(close-dl)<=0.45*atr and close<vwap and conf==-1)
        return 1 if bull else -1 if bear else 0

    if name == "cpr_breakout_vwap_volume":
        pv=_pivot_levels(x)
        if not pv: return 0
        p0,r1,s1,s2,r2=pv
        # CPR width is approximated from prior-day range; narrow CPR = compression.
        lv=_prev_day_levels(x); conf=_candle_confirmation(r,p)
        if not lv: return 0
        h,l,o,c=lv; tc=max(p0,(h+l)/2); bc=min(p0,(h+l)/2)
        if close>tc and close>vwap and vol_ok and conf==1: return 1
        if close<bc and close<vwap and vol_ok and conf==-1: return -1
        return 0

    if name == "market_structure_break_retest":
        sw=_last_swing(x,15); conf=_candle_confirmation(r,p)
        if not sw: return 0
        hi,lo,_,_=sw
        # prior candle performs the break; current candle retests and closes back with confirmation.
        if _num(p["Close"])>hi and low<=hi+0.25*atr and close>hi and conf==1: return 1
        if _num(p["Close"])<lo and high>=lo-0.25*atr and close<lo and conf==-1: return -1
        return 0

    if name == "vwap_sr_rejection":
        conf=_candle_confirmation(r,p)
        sw=_last_swing(x,20)
        if not sw: return 0
        hi,lo,_,_=sw
        if abs(close-vwap)<=0.35*atr and low<=lo+0.35*atr and close>vwap and conf==1: return 1
        if abs(close-vwap)<=0.35*atr and high>=hi-0.35*atr and close<vwap and conf==-1: return -1
        return 0

    if name == "ema2050_vwap_pullback":
        conf=_candle_confirmation(r,p)
        if ema20>ema50 and close>ema50 and abs(close-vwap)<=0.45*atr and low<=ema20+0.35*atr and conf==1: return 1
        if ema20<ema50 and close<ema50 and abs(close-vwap)<=0.45*atr and high>=ema20-0.35*atr and conf==-1: return -1
        return 0

    if name == "inside_bar_vwap_volume":
        if _num(p["High"])>=_num(x["High"].iloc[-3]) or _num(p["Low"])<=_num(x["Low"].iloc[-3]): return 0
        parent_hi=_num(x["High"].iloc[-3]); parent_lo=_num(x["Low"].iloc[-3])
        if close>parent_hi and close>vwap and vol_ok: return 1
        if close<parent_lo and close<vwap and vol_ok: return -1
        return 0

    if name == "bb_mean_reversion_vwap":
        conf=_candle_confirmation(r,p)
        if low<=_num(r["BB_LOWER"]) and close>_num(r["BB_LOWER"]) and close>=vwap-0.8*atr and conf==1: return 1
        if high>=_num(r["BB_UPPER"]) and close<_num(r["BB_UPPER"]) and close<=vwap+0.8*atr and conf==-1: return -1
        return 0

    # ----------------------- chart-pattern ensemble -----------------------
    # Every pattern below requires a completed-candle confirmation. The
    # detector is deliberately tolerant (ATR-normalized) so a valid setup is
    # not lost because of a tiny tick difference, while the final AI layer
    # still has to agree before a directional trade can be opened.
    ph, pl = _pattern_pivots(x, 40, 2)
    recent = x.iloc[-40:]
    avg_atr=max(float(r["ATR"]),1e-9)

    if name == "bull_flag_breakout":
        if len(x)<30: return 0
        pole=x.iloc[-30:-12]
        pole_move=float(pole["Close"].iloc[-1]-pole["Close"].iloc[0])
        flag=x.iloc[-12:-1]
        if pole_move < 1.8*avg_atr: return 0
        fs=_lin_slope(flag["Close"].values)
        flag_range=float(flag["High"].max()-flag["Low"].min())
        flag_top=float(flag["High"].max())
        if fs<0 and flag_range<=max(3.0*avg_atr,abs(pole_move)*0.55) and _pattern_breakout(x,flag_top,1,avg_atr,True): return 1
        return 0

    if name == "inverse_head_shoulders":
        lows=pl[-5:]
        if len(lows)<3: return 0
        a,b,c=lows[-3:]
        tol=0.65*avg_atr
        shoulders_sim=abs(a[1]-c[1])<=tol
        head_deeper=b[1] < min(a[1],c[1])-0.30*avg_atr
        left_peak=float(recent["High"].iloc[a[0]:b[0]+1].max())
        right_peak=float(recent["High"].iloc[b[0]:c[0]+1].max())
        neckline=(left_peak+right_peak)/2.0
        if shoulders_sim and head_deeper and _pattern_breakout(x,neckline,1,avg_atr,False): return 1
        return 0

    if name == "ascending_triangle":
        sh=_triangle_shape(x,'ascending',28)
        if sh and _pattern_breakout(x,float(sh[0]),1,avg_atr,True): return 1
        return 0

    if name == "double_bottom":
        lows=pl[-5:]
        if len(lows)<2: return 0
        a,b=lows[-2:]
        if b[0]-a[0]<4: return 0
        if abs(a[1]-b[1])>0.70*avg_atr: return 0
        neckline=float(recent["High"].iloc[a[0]:b[0]+1].max())
        if b[1] > a[1]+0.75*avg_atr: return 0
        if _pattern_breakout(x,neckline,1,avg_atr,False): return 1
        return 0

    if name == "cup_handle":
        if len(recent)<30: return 0
        q=max(len(recent)//3,8)
        left=float(recent["High"].iloc[:q].max())
        mid_idx=int(recent["Low"].iloc[q:2*q].values.argmin())+q
        mid=float(recent["Low"].iloc[mid_idx])
        right=float(recent["High"].iloc[2*q:].max())
        rim=(left+right)/2.0
        depth=rim-mid
        handle=recent.iloc[2*q:-1]
        if depth<1.2*avg_atr or abs(left-right)>0.9*avg_atr or handle.empty: return 0
        handle_low=float(handle["Low"].min())
        if handle_low < rim-0.55*depth: return 0
        if _pattern_breakout(x,rim,1,avg_atr,True): return 1
        return 0

    if name == "descending_triangle":
        sh=_triangle_shape(x,'descending',28)
        if sh and _pattern_breakout(x,float(sh[0]),-1,avg_atr,True): return -1
        return 0

    if name == "falling_wedge":
        y=recent.iloc[:-1]
        hs=_lin_slope(y["High"].values); ls=_lin_slope(y["Low"].values)
        width_start=float(y["High"].iloc[0]-y["Low"].iloc[0]); width_end=float(y["High"].iloc[-1]-y["Low"].iloc[-1])
        # Both boundaries slope down, with the lower boundary falling faster
        # (convergence). Current close must break the upper boundary.
        if hs < -0.03*avg_atr and ls < -0.03*avg_atr and ls < hs and width_end < 0.80*max(width_start,avg_atr):
            upper=float(y["High"].iloc[-1]) + hs
            if _pattern_breakout(x,upper,1,avg_atr,True): return 1
        return 0

    if name == "head_shoulders":
        highs=ph[-5:]
        if len(highs)<3: return 0
        a,b,c=highs[-3:]
        tol=0.75*avg_atr
        shoulders_sim=abs(a[1]-c[1])<=tol
        head_higher=b[1] > max(a[1],c[1])+0.30*avg_atr
        left_trough=float(recent["Low"].iloc[a[0]:b[0]+1].min())
        right_trough=float(recent["Low"].iloc[b[0]:c[0]+1].min())
        neckline=(left_trough+right_trough)/2.0
        if shoulders_sim and head_higher and _pattern_breakout(x,neckline,-1,avg_atr,False): return -1
        return 0

    if name == "symmetrical_triangle":
        sh=_triangle_shape(x,'symmetric',28)
        if sh:
            upper,lower=sh[0]
            if _pattern_breakout(x,float(upper),1,avg_atr,True): return 1
            if _pattern_breakout(x,float(lower),-1,avg_atr,True): return -1
        return 0

    if name == "rectangle_breakout":
        y=recent.iloc[:-1]
        if len(y)<18: return 0
        top=float(y["High"].max()); bot=float(y["Low"].min()); rng=top-bot
        if rng<1.5*avg_atr or rng>8.0*avg_atr: return 0
        # A real box needs repeated contacts on both sides and a compressed
        # middle; avoid calling a single trend candle a rectangle.
        top_hits=int((y["High"]>=top-0.18*avg_atr).sum())
        bot_hits=int((y["Low"]<=bot+0.18*avg_atr).sum())
        mid=(top+bot)/2.0
        if top_hits<2 or bot_hits<2: return 0
        if _pattern_breakout(x,top,1,avg_atr,True): return 1
        if _pattern_breakout(x,bot,-1,avg_atr,True): return -1
        return 0
    return 0


def _regime(x: pd.DataFrame) -> str:
    if len(x)<60: return "UNKNOWN"
    r=x.iloc[-1]; atr_pct=_num(r["ATR"])/max(abs(_num(r["Close"])),1e-9)*100
    slope=(_num(x["EMA20"].iloc[-1])-_num(x["EMA20"].iloc[-10]))/max(abs(_num(r["Close"])),1e-9)
    spread=abs(_num(r["EMA20"])-_num(r["EMA50"])) / max(abs(_num(r["Close"])),1e-9)
    if spread>0.0015 and abs(slope)>0.00025: return "TREND_UP" if slope>0 else "TREND_DOWN"
    if atr_pct>0.35: return "HIGH_VOL"
    if atr_pct<0.16: return "LOW_VOL"
    return "RANGE"


def _walk_forward(name: str, x: pd.DataFrame, start: int = 100, horizon: int = DEFAULT_HORIZON) -> Dict[str, Any]:
    """Leakage-safe walk-forward using the LIVE engine's fixed 20-point SL / 1.5R TP.

    A signal is generated only from data available at bar i. Future bars are used
    solely to resolve the outcome. If TP and SL are both touched in the same bar,
    the outcome is marked ambiguous rather than assuming the favourable order.
    """
    if len(x) < start + horizon + 5 or name == "oi_writer_zones":
        return {
            "samples": 0, "resolved_samples": 0, "wins": 0, "losses": 0,
            "ambiguous": 0, "unresolved": 0, "win_rate": None,
            "expectancy": None, "profit_factor": None,
            "validation_status": "INSUFFICIENT",
            "note": "Insufficient compatible historical data" if name != "oi_writer_zones"
                    else "OI history not supplied to this price-only backtest"
        }

    outcomes = []
    ambiguous = 0
    unresolved = 0
    for i in range(start, len(x) - horizon):
        hist = x.iloc[:i + 1]
        sig = _signal_rules(hist, name)
        if not sig:
            continue
        entry = _num(x["Close"].iloc[i])
        if not math.isfinite(entry):
            continue
        tp = entry + (WF_SL_POINTS * WF_RR if sig > 0 else -WF_SL_POINTS * WF_RR)
        sl = entry - (WF_SL_POINTS if sig > 0 else -WF_SL_POINTS)
        future = x.iloc[i + 1:i + 1 + horizon]
        resolved = None
        for _, bar in future.iterrows():
            bh = _num(bar["High"]); bl = _num(bar["Low"])
            if not math.isfinite(bh) or not math.isfinite(bl):
                continue
            if sig > 0:
                hit_tp = bh >= tp; hit_sl = bl <= sl
            else:
                hit_tp = bl <= tp; hit_sl = bh >= sl
            if hit_tp and hit_sl:
                ambiguous += 1
                resolved = 0.0
                break
            if hit_tp:
                resolved = 1.0
                break
            if hit_sl:
                resolved = -1.0
                break
        if resolved is None:
            unresolved += 1
            outcomes.append(0.0)
        else:
            outcomes.append(resolved)

    total = len(outcomes)
    wins = int(sum(v > 0 for v in outcomes))
    losses = int(sum(v < 0 for v in outcomes))
    resolved_samples = wins + losses
    win_rate = round(100 * wins / max(resolved_samples, 1), 1) if resolved_samples >= MIN_WF_RESOLVED_FOR_RATE else None
    a = np.asarray(outcomes, float) if outcomes else np.asarray([], float)
    # Expectancy is expressed in R: +1R for TP, -1R for SL, 0 for unresolved/ambiguous.
    expectancy = round(float(a.mean()), 3) if len(a) else None
    profit_factor = round(float(wins / max(losses, 1)), 2) if losses else (float("inf") if wins else None)
    if resolved_samples >= VERIFIED_WF_SAMPLES:
        validation_status = "VERIFIED"
    elif resolved_samples >= MIN_WF_RESOLVED_FOR_RATE:
        validation_status = "PROVISIONAL"
    else:
        validation_status = "INSUFFICIENT"
    return {
        "samples": total, "resolved_samples": resolved_samples, "wins": wins, "losses": losses,
        "ambiguous": ambiguous, "unresolved": unresolved, "win_rate": win_rate,
        "expectancy": expectancy, "profit_factor": profit_factor,
        "validation_status": validation_status,
        "sl_points": WF_SL_POINTS, "rr": WF_RR, "horizon_bars": horizon,
        "note": ("Verified historical sample" if validation_status == "VERIFIED" else
                  "Provisional historical sample" if validation_status == "PROVISIONAL" else
                  "Not enough resolved samples for a trustworthy win-rate display")
    }


def _recent_strategy_signal(x: pd.DataFrame, name: str, option_chain=None, window: int = RECENT_SETUP_BARS):
    """Find the most recent completed-bar strategy signal.

    The old engine looked only at x.iloc[-1]. A valid ORB/retest/pullback can
    therefore disappear on the next candle before the 30-second dashboard
    refresh and never reach the AI comparison layer. This helper keeps the
    setup actionable for a short, controlled window without turning a past
    signal into an all-day permission to trade.
    """
    if name == "oi_writer_zones":
        sig = _signal_rules(x, name, option_chain=option_chain)
        return sig, 0
    n = min(max(int(window), 1), max(len(x) - 60, 1))
    start = max(60, len(x) - n)
    latest_sig = 0
    latest_age = None
    for i in range(start, len(x)):
        hist = x.iloc[:i+1]
        sig = _signal_rules(hist, name, option_chain=None)
        if sig:
            latest_sig = sig
            latest_age = len(x) - 1 - i
    return latest_sig, (latest_age if latest_sig else None)


def _context_vote(name, direction, regime, sr_context, level_prediction, order_flow, mtf, ai_analysis):
    score=0.0; meta=STRATEGY_META[name]
    if regime in meta["regimes"]: score+=22
    text=(str(sr_context)+" "+str(level_prediction)).lower()
    if name in ("daily_liquidity_levels","previous_day_liquidity_sweep") and any(k in text for k in ("liquidity","sweep","support","resistance")): score+=18
    if name=="modified_920_orb" and "opening" in text: score+=10
    if name=="classic_30m_orb" and "opening" in text: score+=10
    if name=="brahmastra" and "vwap" in text: score+=10
    if name=="structure_supply_demand" and any(k in text for k in ("supply","demand")): score+=16
    if direction==1 and "BUYING PRESSURE" in str(order_flow).upper(): score+=10
    if direction==-1 and "SELLING PRESSURE" in str(order_flow).upper(): score+=10
    mt=str(mtf).lower()
    if direction==1 and "bullish" in mt: score+=8
    if direction==-1 and "bearish" in mt: score+=8
    # IMPORTANT: do not feed the independent AI decision back into the
    # strategy score. The final layer compares strategy direction vs AI
    # direction; keeping this score independent prevents circular confirmation.
    return min(score,100.0)


def analyze(df: pd.DataFrame, *, sr_context=None, level_prediction=None, order_flow=None,
            mtf=None, ai_analysis=None, trade_history=None, option_chain=None,
            option_chain_live: bool = False) -> Dict[str,Any]:
    if df is None or len(df)<80:
        return {"version":STRATEGY_VERSION,"status":"INSUFFICIENT_DATA","has_setup":False,"direction":"WAIT","score":0.0,"strategies":{}}
    x=_prepare(df).dropna(subset=["Close","ATR"]); regime=_regime(x); results={}; candidates=[]
    for key,meta in STRATEGY_META.items():
        live_option_chain = option_chain if (key == "oi_writer_zones" and option_chain_live) else None
        sig, signal_age = _recent_strategy_signal(x, key, option_chain=live_option_chain)
        wf=_walk_forward(key,x)
        raw=50.0
        if wf.get("samples",0)>=MIN_WF_SAMPLES:
            raw += max(-18,min(28,(wf.get("expectancy") or 0)*24))
            if wf.get("profit_factor") is not None: raw += max(-8,min(10,(wf["profit_factor"]-1)*5))
            # Recent/current fit receives a modest weight; never overrides setup evidence.
            if wf.get("win_rate") is not None and wf["win_rate"]>=55: raw+=5
        ctx=_context_vote(key,sig,regime,sr_context,level_prediction,order_flow,mtf,ai_analysis)
        # A strategy signal is the primary qualification. Context/walk-forward
        # statistics rank its quality but cannot erase a genuine live setup.
        current=min(100,max(0,0.68*raw+0.32*ctx))
        if sig and meta["type"]=="directional": candidates.append((current,key,sig))
        _source_note = "LIVE_UPSTOX_OPTION_CHAIN" if key == "oi_writer_zones" and option_chain_live else ("LIVE_UPSTOX_PRICE_DATA" if key != "oi_writer_zones" else "BLOCKED_SIMULATED_OPTION_CHAIN")
        validation_status = wf.get("validation_status", "INSUFFICIENT")
        results[key]={"name":meta["name"],"type":meta["type"],"signal":"BUY" if sig>0 else "SELL" if sig<0 else "NONE","current_score":round(current,1),"walk_forward":wf,"regime_fit":regime in meta["regimes"],"formed_now":bool(sig and signal_age == 0),"active_setup":bool(sig),"live_ready":bool(sig and meta["type"]=="directional"),"validation_status":validation_status,"signal_age_bars":signal_age,"formed_at":str(x.index[-1-signal_age]) if sig and signal_age is not None else None,"data_source":_source_note}
    buy=sorted([(s,n) for s,n,d in candidates if d>0],reverse=True); sell=sorted([(s,n) for s,n,d in candidates if d<0],reverse=True)
    buy_score=buy[0][0] if buy else 0; sell_score=sell[0][0] if sell else 0
    direction="BUY" if buy_score>sell_score else "SELL" if sell_score>buy_score else "WAIT"
    best=max(buy_score,sell_score); opposing=min(buy_score,sell_score) if buy and sell else 0; edge=best-opposing
    # The named strategy rules themselves are the trigger. A valid signal may
    # score below an arbitrary composite threshold simply because some context
    # modules are neutral/unknown. Requiring the old 52+ score here was a major
    # cause of "large move but no trade". Keep a tiny edge requirement only when
    # both directions are active, so one genuine strategy can reach the AI
    # comparison layer while contradictory strategy signals still remain visible.
    has_setup=bool(candidates) and (edge >= 1.0)
    selected=[n for _,n in (buy if direction=="BUY" else sell)[:3]] if has_setup else []
    active_names=[STRATEGY_META[n]["name"] for _,n,d in candidates if d==(1 if direction=="BUY" else -1)]
    return {"version":STRATEGY_VERSION,"status":"READY","has_setup":has_setup,"direction":direction if has_setup else "WAIT","score":round(best,1),"edge":round(edge,1),"regime":regime,"selected_strategies":selected,"active_strategy_count":len(active_names),"buy_score":round(buy_score,1),"sell_score":round(sell_score,1),"strategies":results,"formed_at":(results[selected[0]].get("formed_at") if selected else None),"reason":(f"{direction} setup: {', '.join(active_names[:3])}" if has_setup else "No named strategy setup is active in the current execution window.")}
