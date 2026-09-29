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

STRATEGY_VERSION = "v2_named_strategy_library_walkforward"
DEFAULT_HORIZON = 6
MIN_WF_SAMPLES = 18

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
    x["SUPERTREND_DIR"] = _supertrend(x, length=20, factor=2.0)
    return x


def _times(x: pd.DataFrame) -> pd.Series:
    try:
        idx = pd.to_datetime(x.index)
        return pd.Series(idx, index=x.index).dt.time
    except Exception:
        return pd.Series(pd.NaT, index=x.index)


def _session_range(x: pd.DataFrame, start_hm: str, end_hm: str):
    try:
        ts = pd.to_datetime(x.index)
        if len(ts) == 0: return None
        # ORB is defined on the CURRENT trading session only, never by mixing
        # the 09:15-09:20 (or 09:15-09:45) ranges from previous days.
        current_day = ts[-1].date()
        start = pd.Timestamp(start_hm).time()
        end = pd.Timestamp(end_hm).time()
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


def _signal_rules(x: pd.DataFrame, name: str, option_chain=None) -> int:
    if len(x) < 60: return 0
    r=x.iloc[-1]; p=x.iloc[-2]; close=_num(r["Close"]); high=_num(r["High"]); low=_num(r["Low"]); atr=_num(r["ATR"],0)
    if not math.isfinite(close) or atr<=0: return 0
    ema20=_num(r["EMA20"]); ema50=_num(r["EMA50"]); vwap=_num(r["VWAP"])
    body=abs(close-_num(r["Open"])); vol_ok=_num(r["Volume"],0)>=1.15*max(_num(r["VOL20"],0),1)

    if name == "modified_920_orb":
        rr=_session_range(x,"09:15:00","09:20:00")
        if not rr: return 0
        rh,rl=rr; conf=_candle_confirmation(r,p)
        # Modified ORB: breakout -> retest of the broken 9:20 boundary ->
        # completed confirmation candle. No blind first-break entry.
        recent=x.iloc[-6:-1]
        up_break=(float(recent["High"].max())>rh) and low <= rh+0.25*atr and close>rh and conf==1
        dn_break=(float(recent["Low"].min())<rl) and high >= rl-0.25*atr and close<rl and conf==-1
        return 1 if up_break else -1 if dn_break else 0

    if name == "classic_30m_orb":
        rr=_session_range(x,"09:15:00","09:45:00")
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


def _walk_forward(name: str, x: pd.DataFrame, start: int=100, horizon: int=DEFAULT_HORIZON) -> Dict[str,Any]:
    if len(x)<start+horizon+5 or name=="oi_writer_zones":
        return {"samples":0,"win_rate":None,"expectancy":None,"profit_factor":None,"note":"Insufficient compatible historical data" if name!="oi_writer_zones" else "OI history not supplied to this price-only backtest"}
    outcomes=[]
    for i in range(start,len(x)-horizon):
        hist=x.iloc[:i+1]; sig=_signal_rules(hist,name)
        if not sig: continue
        entry=_num(x["Close"].iloc[i]); atr=max(_num(x["ATR"].iloc[i]),1e-9); future=x.iloc[i+1:i+1+horizon]
        if sig>0:
            r=(_num(future["High"].max())-entry)/atr; adverse=(entry-_num(future["Low"].min()))/atr
        else:
            r=(entry-_num(future["Low"].min()))/atr; adverse=(_num(future["High"].max())-entry)/atr
        outcomes.append(1.0 if r>=1 and adverse<1 else -1.0 if adverse>=1 and r<1 else 0.0)
    if not outcomes: return {"samples":0,"win_rate":None,"expectancy":None,"profit_factor":None}
    a=np.asarray(outcomes,float); wins=(a>0).sum(); losses=(a<0).sum(); gw=a[a>0].sum(); gl=abs(a[a<0].sum())
    return {"samples":int(len(a)),"win_rate":round(100*wins/max(wins+losses,1),1),"expectancy":round(float(a.mean()),3),"profit_factor":round(float(gw/max(gl,1e-9)),2) if gl else None}


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
        sig=_signal_rules(x,key,option_chain=(option_chain if (key != "oi_writer_zones" or option_chain_live) else None))
        wf=_walk_forward(key,x)
        raw=50.0
        if wf.get("samples",0)>=MIN_WF_SAMPLES:
            raw += max(-18,min(28,(wf.get("expectancy") or 0)*24))
            if wf.get("profit_factor") is not None: raw += max(-8,min(10,(wf["profit_factor"]-1)*5))
            # Recent/current fit receives a modest weight; never overrides setup evidence.
            if wf.get("win_rate") is not None and wf["win_rate"]>=55: raw+=5
        ctx=_context_vote(key,sig,regime,sr_context,level_prediction,order_flow,mtf,ai_analysis)
        current=min(100,max(0,0.58*raw+0.42*ctx))
        if sig and meta["type"]=="directional": candidates.append((current,key,sig))
        _source_note = "LIVE_UPSTOX_OPTION_CHAIN" if key == "oi_writer_zones" and option_chain_live else ("LIVE_UPSTOX_PRICE_DATA" if key != "oi_writer_zones" else "BLOCKED_SIMULATED_OPTION_CHAIN")
        results[key]={"name":meta["name"],"type":meta["type"],"signal":"BUY" if sig>0 else "SELL" if sig<0 else "NONE","current_score":round(current,1),"walk_forward":wf,"regime_fit":regime in meta["regimes"],"formed_now":bool(sig),"formed_at":str(x.index[-1]) if sig else None,"data_source":_source_note}
    buy=sorted([(s,n) for s,n,d in candidates if d>0],reverse=True); sell=sorted([(s,n) for s,n,d in candidates if d<0],reverse=True)
    buy_score=buy[0][0] if buy else 0; sell_score=sell[0][0] if sell else 0
    direction="BUY" if buy_score>sell_score else "SELL" if sell_score>buy_score else "WAIT"
    best=max(buy_score,sell_score); opposing=min(buy_score,sell_score) if buy and sell else 0; edge=best-opposing
    # One genuine named strategy is enough to create a candidate; AI must agree later.
    has_setup=best>=58 and edge>=5
    selected=[n for _,n in (buy if direction=="BUY" else sell)[:3]] if has_setup else []
    return {"version":STRATEGY_VERSION,"status":"READY","has_setup":has_setup,"direction":direction if has_setup else "WAIT","score":round(best,1),"edge":round(edge,1),"regime":regime,"selected_strategies":selected,"buy_score":round(buy_score,1),"sell_score":round(sell_score,1),"strategies":results,"formed_at":str(x.index[-1]) if has_setup else None,"reason":(f"{direction} setup: {', '.join(STRATEGY_META[n]['name'] for n in selected)}" if has_setup else "No named strategy has a sufficiently strong current setup; no entry candidate.")}
