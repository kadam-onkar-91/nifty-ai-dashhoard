"""Multi-stock research/analysis engine.

Design goals:
- One common pipeline for NSE equities.
- Upstox live data when authenticated; Yahoo Finance only as an explicitly
  labelled fallback.
- No fabricated fundamentals/news. Missing fields remain unavailable.
- NIFTY context is used as market regime context, not as a forced trade signal.
- Equity options are not assumed to exist; option-chain fields are marked N/A.
"""
from app_logging import get_logger
logger = get_logger(__name__)
import urllib.parse
import requests
import feedparser
import pandas as pd
import numpy as np
import yfinance as yf

import indicators
import support_resistance
import smart_money
import liquidity_engine
import market_regime
import cpr_pivot
import sniper_setup
import position_sizing
import risk_engine
import backtest_engine
import monte_carlo
import drift_detection
import database

# Broad liquid NSE universe. The list is a fallback; the Upstox instrument
# master is still used to resolve the actual NSE instrument key.
NSE_STOCKS = [
"RELIANCE","HDFCBANK","ICICIBANK","BHARTIARTL","TCS","INFY","ITC","LT","SBIN","AXISBANK",
"KOTAKBANK","HINDUNILVR","BAJFINANCE","M&M","MARUTI","SUNPHARMA","HCLTECH","WIPRO","TITAN","ULTRACEMCO",
"ADANIENT","ADANIPORTS","NTPC","POWERGRID","ONGC","COALINDIA","TATASTEEL","JSWSTEEL","HINDALCO","VEDL",
"TATAMOTORS","EICHERMOT","HEROMOTOCO","BAJAJ-AUTO","ASIANPAINT","NESTLEIND","BRITANNIA","CIPLA","DRREDDY","DIVISLAB",
"APOLLOHOSP","BEL","HAL","INDIGO","TRENT","ZOMATO","JIOFIN","SHRIRAMFIN","BAJAJFINSV","INDUSINDBK",
"BANKBARODA","PNB","CANBK","IDFCFIRSTB","FEDERALBNK","YESBANK","RPOWER","JPPOWER","HCC","TRIDENT",
"IRFC","IREDA","RVNL","NHPC","SAIL","IOC","BPCL","GAIL","DLF","PIDILITIND","SIEMENS","ABB","DABUR",
"GODREJCP","AMBUJACEM","ACC","BHEL","LICI","DMART","ZYDUSLIFE"
]

SECTOR_INDEX = {
    "Banking": "^NSEBANK", "Financial Services": "^NSEBANK", "IT": "^CNXIT",
    "Pharma": "^CNXPHARMA", "Healthcare": "^CNXPHARMA", "Auto": "^CNXAUTO",
    "Metal": "^CNXMETAL", "FMCG": "^CNXFMCG", "Energy": "^CNXENERGY",
    "Oil & Gas": "^CNXENERGY", "Realty": "^CNXREALTY", "Media": "^CNXMEDIA"
}


def _instrument_map():
    """Return NSE equity metadata from Upstox's official instrument master.

    Metadata is kept together so the same selected symbol can drive market
    data, ISIN-based fundamentals and option eligibility without guessing an
    ISIN from a third-party source.
    """
    try:
        url = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.csv.gz"
        d = pd.read_csv(url, compression="gzip")
        if "tradingsymbol" not in d.columns or "instrument_key" not in d.columns:
            return {}
        if "segment" in d.columns:
            d = d[d["segment"].astype(str).str.upper().eq("NSE_EQ")]
        elif "instrument_type" in d.columns:
            d = d[d["instrument_type"].astype(str).str.upper().isin(["EQ", "EQUITY"])]
        out = {}
        for _, row in d.iterrows():
            sym = str(row.get("tradingsymbol", "")).strip().upper()
            if not sym:
                continue
            out[sym] = {
                "instrument_key": row.get("instrument_key"),
                "isin": row.get("isin") or row.get("isin_code"),
                "exchange": row.get("exchange", "NSE"),
                "instrument_type": row.get("instrument_type", "EQ"),
            }
        return out
    except Exception:
        logger.exception("Upstox instrument master unavailable")
        return {}


def _upstox_candles(access_token, instrument_key, interval="5", days_back=30):
    if not access_token or not instrument_key:
        return None
    headers = {"Accept":"application/json", "Authorization":f"Bearer {access_token}"}
    key = urllib.parse.quote(instrument_key, safe="")
    candles=[]
    try:
        from datetime import datetime, timedelta
        to_date=(datetime.now()-timedelta(days=1)).strftime("%Y-%m-%d")
        from_date=(datetime.now()-timedelta(days=days_back)).strftime("%Y-%m-%d")
        u=f"https://api.upstox.com/v3/historical-candle/{key}/minutes/{interval}/{to_date}/{from_date}"
        r=requests.get(u,headers=headers,timeout=8)
        if r.ok and r.json().get("status")=="success": candles += r.json().get("data",{}).get("candles",[])
    except Exception: pass
    try:
        u=f"https://api.upstox.com/v3/historical-candle/intraday/{key}/minutes/{interval}"
        r=requests.get(u,headers=headers,timeout=8)
        if r.ok and r.json().get("status")=="success": candles += r.json().get("data",{}).get("candles",[])
    except Exception: pass
    if not candles: return None
    df=pd.DataFrame(candles,columns=['Timestamp','Open','High','Low','Close','Volume','OI'])
    df['Timestamp']=pd.to_datetime(df['Timestamp']); df.drop_duplicates('Timestamp',inplace=True)
    df.sort_values('Timestamp',inplace=True); df.set_index('Timestamp',inplace=True)
    return df


def _yf_candles(yf_symbol, period="30d", interval="5m"):
    try:
        df=yf.download(yf_symbol,period=period,interval=interval,progress=False,auto_adjust=False)
        if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
        return df.dropna(how="all") if df is not None else None
    except Exception:
        return None


def _prepare(df):
    if df is None or df.empty: return None
    df=df.copy()
    for c in ['Open','High','Low','Close','Volume']:
        if c not in df.columns: return None
        df[c]=pd.to_numeric(df[c],errors='coerce')
    df.dropna(subset=['Open','High','Low','Close'],inplace=True)
    df['EMA_20']=df['Close'].ewm(span=20,adjust=False).mean()
    df['EMA_50']=df['Close'].ewm(span=50,adjust=False).mean()
    tr=pd.concat([df['High']-df['Low'],(df['High']-df['Close'].shift()).abs(),(df['Low']-df['Close'].shift()).abs()],axis=1).max(axis=1)
    df['ATR']=tr.ewm(span=14,adjust=False).mean()
    try: df['RSI']=indicators.calculate_rsi(df,14).fillna(50)
    except Exception: df['RSI']=50.0
    try: df['MACD'],df['MACD_Signal'],df['MACD_Hist']=indicators.calculate_macd(df)
    except Exception: df['MACD']=df['MACD_Signal']=df['MACD_Hist']=0.0
    df.ffill(inplace=True); df.fillna(0,inplace=True)
    return df



def _news(symbol, company_name=None, limit=6):
    q=f"{company_name or symbol} stock India NSE news"
    try:
        url=f"https://news.google.com/rss/search?q={urllib.parse.quote(q)}&hl=en-IN&gl=IN&ceid=IN:en"
        feed=feedparser.parse(url)
        rows=[]
        for e in feed.entries[:limit]:
            rows.append({'title':e.get('title',''), 'published':e.get('published',''), 'source':e.get('source',{}).get('title','') if isinstance(e.get('source'),dict) else '', 'link':e.get('link','')})
        return rows, ('AVAILABLE' if rows else 'UNAVAILABLE')
    except Exception:
        return [], 'UNAVAILABLE'


NIFTY50_CONSTITUENTS = [
"ADANIENT","ADANIPORTS","APOLLOHOSP","ASIANPAINT","AXISBANK","BAJAJ-AUTO","BAJAJFINSV","BAJFINANCE","BHARTIARTL","BEL",
"CIPLA","COALINDIA","DRREDDY","EICHERMOT","ETERNAL","GRASIM","HCLTECH","HDFCBANK","HDFCLIFE","HEROMOTOCO",
"HINDALCO","HINDUNILVR","ICICIBANK","INDUSINDBK","INFY","ITC","JIOFIN","JSWSTEEL","KOTAKBANK","LT",
"M&M","MARUTI","NESTLEIND","NTPC","ONGC","POWERGRID","RELIANCE","SBILIFE","SBIN","SHRIRAMFIN",
"SUNPHARMA","TATACONSUM","TATAMOTORS","TATASTEEL","TCS","TECHM","TITAN","TRENT","ULTRACEMCO","WIPRO"
]


def _get_nifty50_constituents():
    """Try the official Nifty Indices constituent CSV, then use the bundled fallback."""
    urls=[
        'https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv',
        'https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv?download=1'
    ]
    headers={'User-Agent':'Mozilla/5.0','Accept':'text/csv,*/*'}
    for url in urls:
        try:
            r=requests.get(url,headers=headers,timeout=8)
            if r.ok and r.text and 'Symbol' in r.text:
                import io
                d=pd.read_csv(io.StringIO(r.text))
                col=next((c for c in d.columns if str(c).strip().lower()=='symbol'),None)
                if col:
                    vals=[str(x).strip().upper() for x in d[col].dropna().tolist() if str(x).strip()]
                    if len(vals)>=45:
                        return vals[:50], 'Nifty Indices official constituent CSV'
        except Exception:
            pass
    return NIFTY50_CONSTITUENTS, 'Bundled NIFTY 50 fallback list'


def _nifty50_fundamentals(limit=50):
    """Build a transparent NIFTY 50 constituent fundamental snapshot.

    Yahoo does not expose the official live NIFTY 50 weight table through this
    engine, so aggregates are equal-constituent averages/coverage metrics,
    explicitly labelled as such rather than pretending to be index-weighted.
    """
    fields=['marketCap','trailingPE','forwardPE','priceToBook','returnOnEquity','profitMargins','revenueGrowth','earningsGrowth','debtToEquity','dividendYield']
    constituents, constituent_source = _get_nifty50_constituents()
    rows=[]
    for sym in constituents[:limit]:
        try:
            info=yf.Ticker(sym+'.NS').info or {}
            row={'symbol':sym}
            for f in fields: row[f]=info.get(f)
            rows.append(row)
        except Exception:
            rows.append({'symbol':sym})
    if not rows:
        return {'status':'UNAVAILABLE','source':'Yahoo Finance','coverage':0,'constituents':0}
    out={'status':'AVAILABLE','source':'Yahoo Finance constituent snapshots','method':'Equal-constituent aggregate; not official index-weighted fundamentals','constituents':len(constituents),'constituent_source':constituent_source,'coverage':sum(1 for r in rows if any(r.get(f) is not None for f in fields))}
    for f in fields:
        vals=[float(r[f]) for r in rows if isinstance(r.get(f),(int,float)) and np.isfinite(r[f])]
        out[f+'_mean']=float(np.mean(vals)) if vals else None
        out[f+'_coverage']=len(vals)
    # A simple regime description is informational only; it is not a trade signal.
    pe=out.get('trailingPE_mean'); rg=out.get('revenueGrowth_mean'); eg=out.get('earningsGrowth_mean')
    out['regime_note']='Fundamental data available' if any(v is not None for v in (pe,rg,eg)) else 'Fundamental data unavailable'
    return out


def _nifty_news(limit=8):
    q='NIFTY 50 NSE India stock market index news'
    try:
        url=f"https://news.google.com/rss/search?q={urllib.parse.quote(q)}&hl=en-IN&gl=IN&ceid=IN:en"
        feed=feedparser.parse(url)
        rows=[]
        for e in feed.entries[:limit]:
            rows.append({'title':e.get('title',''),'published':e.get('published',''),'source':e.get('source',{}).get('title','') if isinstance(e.get('source'),dict) else '','link':e.get('link','')})
        return rows, ('AVAILABLE' if rows else 'UNAVAILABLE')
    except Exception:
        return [], 'UNAVAILABLE'


# Public wrappers -- used directly by the main NIFTY 50 dashboard in app.py
# (the underscore-prefixed versions above were only ever consumed inside this
# module, for the per-stock "NIFTY 50 context" tab; these are the same real
# data, just callable without reaching into a "private" name from outside).
def get_nifty50_news(limit=8):
    """Live NIFTY 50 / NSE India market news (Google News RSS), same feed the
    per-stock dashboard's News+Context tab already uses."""
    return _nifty_news(limit=limit)


def get_nifty50_fundamentals(limit=50):
    """Equal-constituent aggregate fundamentals (mean P/E, P/B, ROE, margins,
    growth, debt/equity, dividend yield) across the real NIFTY 50 constituent
    list. Explicitly labelled as an aggregate, not an official index-weighted
    number -- same real data the per-stock dashboard already computes but
    never displayed anywhere until now."""
    return _nifty50_fundamentals(limit=limit)


# ---------------------------------------------------------------------
# Two soft confluence factors for ai_trade_decision.py / trade_learning.py
# -- NIFTY 50-specific news and fundamentals, folded into the SAME weighted
# confluence score as every other factor (order flow, VWAP, regime, etc.),
# not a separate gate. Like every other soft factor here, neither can
# create a trade by itself -- they can only nudge/confirm a setup that is
# already qualifying on price/structure, and both get logged into
# trade_learning's per-factor win-rate history so the engine learns over
# time whether NIFTY-specific news/fundamentals actually predicted wins in
# THIS user's own resolved trades.
_N50_POS_WORDS = ['surge', 'jump', 'gain', 'growth', 'rally', 'positive', 'boost', 'up',
                  'high', 'deal', 'record', 'outperform', 'upgrade', 'beat', 'rebound']
_N50_NEG_WORDS = ['fall', 'drop', 'slump', 'crash', 'loss', 'inflation', 'war', 'tension',
                  'negative', 'down', 'crisis', 'sanction', 'sell-off', 'selloff', 'downgrade',
                  'miss', 'default', 'plunge']


def get_nifty50_news_sentiment(news_rows=None, limit=10):
    """Same simple positive/negative keyword heuristic as global_news.py
    (disclosed as such, not full AI sentiment) -- but scored specifically
    over live NIFTY 50 / NSE headlines, not the generic 'India' region feed
    the engine used before. Pass the already-fetched get_nifty50_news() rows
    to avoid a duplicate RSS call; if None, fetches fresh (fast, RSS-only)."""
    if news_rows is None:
        news_rows, _ = _nifty_news(limit=limit)
    if not news_rows:
        return {'label': 'UNAVAILABLE', 'positive': 0, 'negative': 0, 'headlines_scored': 0}
    pos = neg = 0
    for row in news_rows:
        title = str((row or {}).get('title', '')).lower()
        if any(w in title for w in _N50_POS_WORDS):
            pos += 1
        if any(w in title for w in _N50_NEG_WORDS):
            neg += 1
    if pos > neg:
        label = 'BULLISH'
    elif neg > pos:
        label = 'BEARISH'
    else:
        label = 'NEUTRAL'
    return {'label': label, 'positive': pos, 'negative': neg, 'headlines_scored': len(news_rows)}


def get_nifty50_fundamentals_bias(fund=None):
    """Turns the NIFTY 50 constituent-aggregate fundamentals (already
    computed by get_nifty50_fundamentals) into a soft, slow-moving
    growth/quality tilt. Fundamentals move over weeks/quarters, not
    minutes -- this can only agree-or-not with an intraday technical setup,
    never drive one. Pass the cached dict from get_nifty50_fundamentals();
    this does NOT re-fetch (that call pulls 50 stocks and is slow)."""
    if not isinstance(fund, dict) or fund.get('status') != 'AVAILABLE':
        return {'label': 'UNAVAILABLE', 'score': 0}
    score = 0
    eg = fund.get('earningsGrowth_mean')
    rg = fund.get('revenueGrowth_mean')
    roe = fund.get('returnOnEquity_mean')
    if eg is not None:
        score += 1 if eg > 0 else -1
    if rg is not None:
        score += 1 if rg > 0 else -1
    if roe is not None:
        score += 1 if roe > 0.15 else (-1 if roe < 0.08 else 0)
    if score >= 2:
        label = 'BULLISH'
    elif score <= -2:
        label = 'BEARISH'
    else:
        label = 'NEUTRAL'
    return {'label': label, 'score': score}


def _index_context():
    try:
        d=yf.download('^NSEI',period='5d',interval='1d',progress=False,auto_adjust=False)
        if isinstance(d.columns,pd.MultiIndex): d.columns=d.columns.get_level_values(0)
        c=d['Close'].dropna()
        return {'nifty_last':float(c.iloc[-1]),'nifty_change_pct':float((c.iloc[-1]/c.iloc[-2]-1)*100) if len(c)>1 else None,'source':'Yahoo Finance'}
    except Exception: return {'nifty_last':None,'nifty_change_pct':None,'source':'UNAVAILABLE'}


def _sector_context(sector):
    idx=SECTOR_INDEX.get(sector)
    if not idx: return {'status':'UNAVAILABLE','reason':'No mapped sector index'}
    try:
        d=yf.download(idx,period='5d',interval='1d',progress=False,auto_adjust=False)
        if isinstance(d.columns,pd.MultiIndex): d.columns=d.columns.get_level_values(0)
        c=d['Close'].dropna()
        return {'status':'AVAILABLE','index':idx,'last':float(c.iloc[-1]),'change_pct':float((c.iloc[-1]/c.iloc[-2]-1)*100) if len(c)>1 else None}
    except Exception: return {'status':'UNAVAILABLE','index':idx}


def _upstox_json(url, access_token, params=None, timeout=8):
    if not access_token:
        return None
    try:
        r = requests.get(url, params=params, headers={"Accept": "application/json", "Authorization": f"Bearer {access_token}"}, timeout=timeout)
        data = r.json()
        return data if r.ok and data.get("status") == "success" else None
    except Exception:
        return None


def _upstox_fundamentals(isin, access_token):
    """Use Upstox's ISIN-based Company Fundamentals APIs when authenticated."""
    if not isin or not access_token:
        return {"status": "UNAVAILABLE", "source": "Upstox Company Fundamentals", "reason": "Upstox login/ISIN unavailable"}
    base = "https://api.upstox.com/v2/fundamentals"
    endpoints = {
        "profile": f"{base}/{isin}/profile",
        "ratios": f"{base}/{isin}/key-ratios",
        "income_statement": f"{base}/{isin}/income-statement",
        "balance_sheet": f"{base}/{isin}/balance-sheet",
        "cash_flow": f"{base}/{isin}/cash-flow",
        "shareholdings": f"{base}/{isin}/share-holdings",
        "corporate_actions": f"{base}/{isin}/corporate-actions",
        "competitors": f"{base}/{isin}/competitors",
    }
    out = {"status": "AVAILABLE", "source": "Upstox Company Fundamentals", "isin": isin}
    got = 0
    for name, url in endpoints.items():
        data = _upstox_json(url, access_token, timeout=8)
        if data and "data" in data:
            out[name] = data["data"]
            got += 1
    if got == 0:
        return {"status": "UNAVAILABLE", "source": "Upstox Company Fundamentals", "isin": isin,
                "reason": "Upstox fundamentals endpoints returned no usable data"}
    return out


def _fundamentals(symbol, access_token=None, isin=None):
    """Selected-stock fundamentals: Upstox first; Yahoo only as labelled fallback."""
    up = _upstox_fundamentals(isin, access_token)
    if up.get("status") == "AVAILABLE":
        profile = up.get("profile") or {}
        ratios = up.get("ratios") or []
        ratio_map = {str(x.get("name")): x for x in ratios if isinstance(x, dict)}
        return {
            "status": "AVAILABLE", "source": "Upstox Company Fundamentals", "isin": isin,
            "company_profile": profile.get("company_profile"),
            "sector": profile.get("sector"),
            "sector_market_cap_inr": profile.get("sector_market_cap_inr"),
            "ratios": ratios,
            "ratio_map": ratio_map,
            "income_statement": up.get("income_statement"),
            "balance_sheet": up.get("balance_sheet"),
            "cash_flow": up.get("cash_flow"),
            "shareholdings": up.get("shareholdings"),
            "corporate_actions": up.get("corporate_actions"),
            "competitors": up.get("competitors"),
        }
    try:
        info = yf.Ticker(symbol + ".NS").info or {}
        keys = ['longName','sector','industry','marketCap','trailingPE','forwardPE','priceToBook','returnOnEquity','profitMargins','revenueGrowth','earningsGrowth','debtToEquity','dividendYield','beta','52WeekChange']
        out = {k: info.get(k) for k in keys}
        out.update({'symbol': symbol, 'source': 'Yahoo Finance fallback', 'status': 'AVAILABLE' if any(v is not None for v in out.values()) else 'UNAVAILABLE'})
        return out
    except Exception as e:
        return {'symbol': symbol, 'source': 'Yahoo Finance fallback', 'status': 'UNAVAILABLE', 'reason': type(e).__name__}


def _upstox_quote_depth(instrument_key, access_token):
    if not instrument_key or not access_token:
        return {"status": "UNAVAILABLE", "source": "Upstox market quote"}
    data = _upstox_json("https://api.upstox.com/v2/market-quote/quotes", access_token,
                        params={"instrument_key": instrument_key}, timeout=6)
    if not data:
        return {"status": "UNAVAILABLE", "source": "Upstox market quote"}
    match = next(iter((data.get("data") or {}).values()), {})
    depth = match.get("depth") or {}
    buys = depth.get("buy") or []
    sells = depth.get("sell") or []
    bq = sum(float(x.get("quantity", 0) or 0) for x in buys)
    sq = sum(float(x.get("quantity", 0) or 0) for x in sells)
    total = bq + sq
    imbalance = ((bq - sq) / total * 100) if total else None
    return {
        "status": "AVAILABLE", "source": "Upstox Live Market Quote",
        "last_price": match.get("last_price"), "prev_close": match.get("ohlc", {}).get("close"),
        "volume": match.get("volume"), "total_buy_qty": bq, "total_sell_qty": sq,
        "imbalance_pct": round(imbalance, 2) if imbalance is not None else None,
        "pressure": "BUYING" if imbalance is not None and imbalance > 15 else "SELLING" if imbalance is not None and imbalance < -15 else "BALANCED",
        "best_bid": buys[0].get("price") if buys else None, "best_ask": sells[0].get("price") if sells else None,
    }


def _upstox_stock_option_chain(instrument_key, access_token, live_price):
    """Fetch real stock option-chain/OI/IV/Greeks when Upstox exposes it."""
    if not instrument_key or not access_token or live_price is None:
        return {"status":"UNAVAILABLE", "source":"Upstox Option Chain"}
    contracts = _upstox_json("https://api.upstox.com/v2/option/contract", access_token,
                             params={"instrument_key": instrument_key}, timeout=8)
    rows = contracts.get("data") if contracts else None
    if not rows:
        return {"status":"UNAVAILABLE", "source":"Upstox Option Chain", "reason":"No option contracts returned for this stock"}
    expiries = sorted({str(x.get("expiry")) for x in rows if x.get("expiry")})
    if not expiries:
        return {"status":"UNAVAILABLE", "source":"Upstox Option Chain", "reason":"No expiry returned"}
    chain = _upstox_json("https://api.upstox.com/v2/option/chain", access_token,
                         params={"instrument_key": instrument_key, "expiry_date": expiries[0]}, timeout=8)
    data = chain.get("data") if chain else None
    if not data:
        return {"status":"UNAVAILABLE", "source":"Upstox Option Chain", "expiry": expiries[0], "reason":"Option chain returned no rows"}
    parsed=[]
    for r in data:
        strike=r.get("strike_price")
        if strike is None: continue
        ce=r.get("call_options") or {}; pe=r.get("put_options") or {}
        cm=ce.get("market_data") or {}; pm=pe.get("market_data") or {}
        cg=ce.get("option_greeks") or {}; pg=pe.get("option_greeks") or {}
        parsed.append({
            "Strike": float(strike),
            "CE LTP": cm.get("ltp"), "CE OI": cm.get("oi"), "CE IV": cg.get("iv"), "CE Delta": cg.get("delta"),
            "PE LTP": pm.get("ltp"), "PE OI": pm.get("oi"), "PE IV": pg.get("iv"), "PE Delta": pg.get("delta"),
            "PCR": r.get("pcr"),
        })
    parsed.sort(key=lambda x: abs(x["Strike"] - live_price))
    return {"status":"AVAILABLE", "source":"Upstox Live Option Chain", "expiry": expiries[0],
            "underlying_spot": data[0].get("underlying_spot_price") if data else live_price,
            "rows": parsed[:17]}


def _stock_mtf(df):
    """Multi-timeframe structure derived only from the selected stock's candles."""
    out = {}
    rules = {"5M": "5min", "15M": "15min", "30M": "30min", "1H": "1h", "4H": "4h", "1D": "1D"}
    for label, rule in rules.items():
        try:
            agg = {"Open":"first", "High":"max", "Low":"min", "Close":"last", "Volume":"sum"}
            x = df.resample(rule).agg(agg).dropna()
            if len(x) < 15:
                out[label] = {"status":"DATA_UNAVAILABLE"}
                continue
            e20 = x["Close"].ewm(span=20, adjust=False).mean().iloc[-1]
            e50 = x["Close"].ewm(span=50, adjust=False).mean().iloc[-1]
            last = float(x["Close"].iloc[-1])
            trend = "Bullish" if last > e20 > e50 else "Bearish" if last < e20 < e50 else "Mixed"
            ms = smart_money.detect_market_structure(x)
            out[label] = {"status":"OK", "trend":trend, "last":last, "structure":ms[0].get("Market Event") if ms else "Unknown", "bars":len(x)}
        except Exception:
            out[label] = {"status":"DATA_UNAVAILABLE"}
    available=[v["trend"] for v in out.values() if v.get("status")=="OK"]
    bulls=sum(1 for x in available if x=="Bullish"); bears=sum(1 for x in available if x=="Bearish")
    out["alignment"] = "Bullish" if bulls > bears and bulls >= 3 else "Bearish" if bears > bulls and bears >= 3 else "Mixed/Neutral"
    return out


def _technical_context(df, live, atr):
    v = df["Volume"].astype(float)
    pv = (df["Close"] * v).cumsum()
    vv = v.cumsum().replace(0, np.nan)
    vwap = float((pv / vv).iloc[-1]) if vv.iloc[-1] else None
    fvg, obs, sweeps = smart_money.detect_smc_zones(df.tail(250))
    structure = smart_money.detect_market_structure(df.tail(250))
    candle = smart_money.detect_candlestick_pattern(df)
    liq = liquidity_engine.build_liquidity_map(df, live, atr)
    return {"vwap": vwap, "smc": {"fvg": fvg, "order_blocks": obs, "liquidity_sweeps": sweeps},
            "market_structure": structure, "candlestick": candle, "liquidity": liq,
            "mtf": _stock_mtf(df)}


def analyze_stock(symbol, access_token=None):
    symbol=symbol.upper().strip().replace('.NS','')
    if symbol not in NSE_STOCKS:
        # Still allow a valid NSE symbol entered manually.
        allowed=True
    else: allowed=True
    if not allowed: return {'status':'INVALID','symbol':symbol}

    inst_meta=_instrument_map().get(symbol, {})
    inst=inst_meta.get("instrument_key")
    df=_upstox_candles(access_token,inst) if inst else None
    data_source='Upstox Live' if df is not None and not df.empty else 'Yahoo Finance fallback'
    if df is None or df.empty: df=_yf_candles(symbol+'.NS')
    df=_prepare(df)
    if df is None or df.empty: return {'status':'DATA_UNAVAILABLE','symbol':symbol,'message':'No usable OHLCV data from Upstox or Yahoo Finance.'}

    live=float(df['Close'].iloc[-1]); atr=float(df['ATR'].iloc[-1])
    levels=support_resistance.get_key_levels(df)
    supports=[x for x in levels.get('supports',[]) if x<live]
    resistances=[x for x in levels.get('resistances',[]) if x>live]
    support=max(supports) if supports else None; resistance=min(resistances) if resistances else None
    ema20=float(df['EMA_20'].iloc[-1]); ema50=float(df['EMA_50'].iloc[-1]); rsi=float(df['RSI'].iloc[-1])
    trend='BULLISH' if live>ema20>ema50 else 'BEARISH' if live<ema20<ema50 else 'MIXED'
    momentum='OVERBOUGHT/EXTENDED' if rsi>=70 else 'OVERSOLD/WEAK' if rsi<=30 else 'NEUTRAL'
    fund=_fundamentals(symbol, access_token=access_token, isin=inst_meta.get("isin")); company=(fund.get("longName") or symbol)
    if fund.get("source") == "Upstox Company Fundamentals":
        company = symbol
    news,status=_news(symbol,company)
    sector=fund.get('sector') or fund.get('industry') or 'Unknown'; nifty=_index_context(); sec=_sector_context(sector); nifty_fund=_nifty50_fundamentals(); nifty_news,nifty_news_status=_nifty_news()
    quote=_upstox_quote_depth(inst, access_token)
    technical=_technical_context(df, live, atr)
    option_chain=_upstox_stock_option_chain(inst, access_token, live)
    bullish=0; bearish=0
    if trend=='BULLISH': bullish+=2
    elif trend=='BEARISH': bearish+=2
    if live>ema20: bullish+=1
    else: bearish+=1
    if 45<=rsi<=65: bullish+=1 if trend=='BULLISH' else 0; bearish+=1 if trend=='BEARISH' else 0
    if rsi>=70: bearish+=1
    if rsi<=30: bullish+=1
    if support and (live-support)<=max(0.5*atr, live*0.003): bullish+=1
    if technical.get('vwap') is not None and live > technical['vwap']: bullish+=1
    elif technical.get('vwap') is not None and live < technical['vwap']: bearish+=1
    if technical.get('mtf',{}).get('alignment') == 'Bullish': bullish+=2
    elif technical.get('mtf',{}).get('alignment') == 'Bearish': bearish+=2
    if quote.get('pressure') == 'BUYING': bullish+=1
    elif quote.get('pressure') == 'SELLING': bearish+=1
    if resistance and (resistance-live)<=max(0.5*atr, live*0.003): bearish+=1
    if nifty.get('nifty_change_pct') is not None:
        if nifty['nifty_change_pct']>0.25: bullish+=1
        elif nifty['nifty_change_pct']<-0.25: bearish+=1
    if sec.get('change_pct') is not None:
        if sec['change_pct']>0.5: bullish+=1
        elif sec['change_pct']<-0.5: bearish+=1
    decision='NO TRADE'
    if bullish>=bearish+2 and trend=='BULLISH' and rsi<70: decision='BULLISH BIAS'
    elif bearish>=bullish+2 and trend=='BEARISH' and rsi>30: decision='BEARISH BIAS'
    return {'status':'OK','symbol':symbol,'company':company,'data_source':data_source,'live_price':live,'atr':atr,
            'ema20':ema20,'ema50':ema50,'rsi':rsi,'trend':trend,'momentum':momentum,'support':support,'resistance':resistance,
            'fundamentals':fund,'news':news,'news_status':status,'sector':sector,'sector_context':sec,'nifty_context':nifty,'nifty_fundamentals':nifty_fund,'nifty_news':nifty_news,'nifty_news_status':nifty_news_status,
            'bullish_score':bullish,'bearish_score':bearish,'decision':decision,'timestamp':str(df.index[-1]),
            'instrument_key':inst,'isin':inst_meta.get('isin'),'quote':quote,'technical_context':technical,'option_chain':option_chain,
            'data_quality': {'market_data': data_source, 'fundamentals': fund.get('source'), 'company_news': status, 'nifty_context': nifty.get('source')}}


# =====================================================================
# FULL DASHBOARD MODE -- everything below this line is what powers the
# "Select Your Stock" primary dashboard (sidebar), which mirrors the
# main NIFTY 50 dashboard's own engines/factors/weights, driven by the
# SELECTED STOCK's own real data. See analyze_stock_full() at the
# bottom -- that is the single entry point app.py calls.
# =====================================================================

def _stock_df_for_engines(symbol, access_token=None):
    """Re-fetch + prepare OHLCV for `symbol` with a session VWAP column
    added, for use by the additional engines below (regime, CPR/pivots,
    sniper setup, the institutional AI engine). Mirrors analyze_stock()'s
    own fetch exactly so results stay consistent with the Overview/
    Technical tabs built from analyze_stock()."""
    symbol = symbol.upper().strip().replace('.NS', '')
    inst_meta = _instrument_map().get(symbol, {})
    inst = inst_meta.get("instrument_key")
    df = _upstox_candles(access_token, inst) if inst else None
    data_source = 'Upstox Live' if df is not None and not df.empty else 'Yahoo Finance fallback'
    if df is None or df.empty:
        df = _yf_candles(symbol + '.NS')
    df = _prepare(df)
    if df is None or df.empty:
        return None, data_source, inst, inst_meta
    v = df['Volume'].astype(float)
    if v.sum() > 0:
        pv = (df['Close'] * v).cumsum()
        vv = v.cumsum().replace(0, np.nan)
        df['VWAP'] = (pv / vv).fillna(df['Close'])
    else:
        df['VWAP'] = df['Close']
    return df, data_source, inst, inst_meta


def _stock_option_chain_for_sr(option_chain_result):
    """Adapts this engine's stock option-chain schema (CE OI / PE OI) into
    the 'Call OI' / 'Put OI' schema support_resistance.py and
    sniper_setup.py expect, so those engines work UNMODIFIED for a stock's
    OWN option chain when that stock actually has listed F&O. Most NSE
    equities don't -- when that's the case this returns an empty frame and
    every dependent factor honestly reports unavailable, exactly like it
    already does for Nifty when option data is missing. Nothing is ever
    filled in with Nifty's own option chain."""
    empty = pd.DataFrame(columns=["Strike", "Call OI", "Put OI", "PCR"])
    if not option_chain_result or option_chain_result.get("status") != "AVAILABLE":
        return empty
    rows = option_chain_result.get("rows") or []
    if not rows:
        return empty
    out = pd.DataFrame(rows)
    out = out.rename(columns={"CE OI": "Call OI", "PE OI": "Put OI"})
    for c in ["Call OI", "Put OI", "PCR"]:
        if c not in out.columns:
            out[c] = 0.0
    return out


def _flow_footprint_from_quote(quote):
    """Stock-level analogue of Nifty's FII/DII footprint, fed into the
    exact same calculate_institutional_flow_score() the Nifty dashboard
    uses. Nifty's FII/DII number is derived from INDEX option OI -- a
    concept that only exists for the index -- so it has no literal
    per-stock equivalent; the stock's OWN real-time resting buy/sell
    imbalance (live Upstox depth) is the closest honest real-data
    substitute for 'institutional flow' at the single-stock level."""
    if not quote or quote.get("status") != "AVAILABLE":
        return "Neutral (order-book data unavailable)"
    pressure = quote.get("pressure", "BALANCED")
    if pressure == "BUYING":
        return "BULLISH (stock order-book buying pressure)"
    if pressure == "SELLING":
        return "BEARISH (stock order-book selling pressure)"
    return "Neutral (balanced order-book)"


def _breadth_status_from_sector(sector_context):
    """Stock-level analogue of Nifty-50 internal breadth: this stock's own
    mapped SECTOR index change, real data, same STRONG/WEAK vocabulary the
    scoring engine already understands."""
    if not sector_context or sector_context.get("status") != "AVAILABLE":
        return "Neutral (sector index data unavailable)"
    chg = sector_context.get("change_pct")
    if chg is None:
        return "Neutral"
    if chg >= 0.8: return "STRONG / POSITIVE sector breadth"
    if chg >= 0.2: return "Positive sector breadth"
    if chg <= -0.8: return "WEAK / NEGATIVE sector breadth"
    if chg <= -0.2: return "Negative sector breadth"
    return "Neutral sector breadth"


def _sector_correlation_note(trend, sector_context):
    """Stock-level analogue of Nifty's Bank Nifty / Sensex divergence
    check. Here the correlated benchmark is the stock's OWN sector index
    (same sector map already used elsewhere in this engine), which plays
    the same 'does the broader group confirm this move' role that Bank
    Nifty plays specifically for the Nifty index."""
    if not sector_context or sector_context.get("status") != "AVAILABLE" or sector_context.get("change_pct") is None:
        return None
    chg = sector_context["change_pct"]
    if trend == "BULLISH" and chg > 0.2:
        return "CONFIRMED -- sector index also trending up with this stock"
    if trend == "BEARISH" and chg < -0.2:
        return "CONFIRMED -- sector index also trending down with this stock"
    if (trend == "BULLISH" and chg < -0.2) or (trend == "BEARISH" and chg > 0.2):
        return "DIVERGENCE WARNING -- sector index is moving the opposite way to this stock"
    return None


def analyze_stock_full(symbol, access_token=None, ai_engine=None, global_avg_change=0.0,
                        sentiment_score="Neutral", capital=100000.0, risk_pct=1.0, live_vix=None):
    """
    The FULL, dashboard-grade version of analyze_stock(): same base
    research (Overview / Technical+SMC/ICT / Options+Order Flow /
    Fundamentals / News from analyze_stock) PLUS every extra engine the
    main NIFTY 50 dashboard runs -- Market Regime, CPR/Pivots/Opening
    Range, the institutional Hybrid AI scoring engine (IDENTICAL weights:
    20% technical / 25% SMC / 15% macro / 40% flow), Sniper Setup,
    Position Sizing, Risk Engine, real Backtest, Monte Carlo and Drift
    Detection -- all computed from THIS symbol's own real candles and
    THIS symbol's own trade history (never mixed with Nifty's or another
    stock's).

    A few Nifty-only inputs genuinely have no per-stock equivalent because
    they are derived from INDEX derivatives specifically (index option
    PCR/Max Pain, FII/DII index-option footprint, Nifty futures volume
    used as an index volume proxy, Bank Nifty/Sensex AS the correlated
    instrument). For those this uses the closest real, honestly-labelled
    per-stock substitute instead (this stock's OWN option chain when it
    has listed F&O, this stock's OWN real traded volume -- equities have
    real volume, unlike the index, so no proxy is even needed --, this
    stock's OWN order-book imbalance, and its mapped SECTOR index as the
    correlated benchmark). Same method, same weights, real data for the
    selected stock, nothing fabricated and nothing borrowed from Nifty.
    """
    base = analyze_stock(symbol, access_token)
    if base.get("status") != "OK":
        return base
    symbol = base["symbol"]

    df, _ds2, inst, inst_meta = _stock_df_for_engines(symbol, access_token)
    if df is None or df.empty:
        base["full_dashboard_status"] = "PARTIAL -- extended engines need more candle history"
        return base

    live_price = base["live_price"]
    atr = base["atr"]
    tc = base.get("technical_context") or {}
    df_oc_adapted = _stock_option_chain_for_sr(base.get("option_chain"))

    # ---- Market Regime (same engine as Nifty, this stock's own df) ----
    try:
        regime = market_regime.detect_regime(df, live_vix=live_vix)
        regime_guidance = market_regime.regime_adjusted_guidance(regime)
    except Exception:
        logger.exception("Stock market regime failed")
        regime = {"primary": "UNKNOWN", "volatility": "UNKNOWN", "structure": "UNKNOWN", "adx": None,
                  "plus_di": None, "minus_di": None, "atr_percentile": None, "vix": None, "expiry_regime": False}
        regime_guidance = "Regime unavailable."
    is_choppy = "CHOPPY" in str(regime.get("structure", "")).upper() or "RANGE" in str(regime.get("structure", "")).upper()

    # ---- CPR / Pivots / Opening Range (same engine, this stock's own df) ----
    try:
        pivots = cpr_pivot.calculate_pivots(df)
        cpr_pos = cpr_pivot.cpr_position(live_price, pivots) if pivots.get("status") == "OK" else "N/A"
        orb = cpr_pivot.opening_range(df)
    except Exception:
        logger.exception("Stock CPR/pivots failed")
        pivots, cpr_pos, orb = {"status": "UNAVAILABLE"}, "N/A", {"status": "UNAVAILABLE"}

    # ---- Institutional Hybrid AI engine -- SAME model + SAME weights as Nifty ----
    ai_result = None
    if ai_engine is not None:
        try:
            ms = tc.get("market_structure") or []
            smc_event = ms[0].get("Market Event", "Neutral Structure") if ms else "Neutral Structure"
            fii_footprint = _flow_footprint_from_quote(base.get("quote"))
            breadth_status = _breadth_status_from_sector(base.get("sector_context"))
            corr_note = _sector_correlation_note(base.get("trend"), base.get("sector_context"))
            ai_result = ai_engine.analyze(
                live_price=live_price, df=df, df_option_chain=df_oc_adapted, smc_data=smc_event,
                sentiment_score=sentiment_score, global_avg_change=global_avg_change,
                fii_footprint=fii_footprint, breadth_status=breadth_status, is_choppy=is_choppy,
                banknifty_correlation_note=corr_note
            )
        except Exception:
            logger.exception("Stock AI engine analyze failed")
            ai_result = None

    # ---- Sniper Setup (same engine, this stock's own VWAP/CPR/SMC/OI) ----
    try:
        cpr_info = None
        if pivots.get("status") == "OK":
            cpr_info = {"pivot": pivots["pivot"], "cpr_top": pivots["tc"], "cpr_bottom": pivots["bc"],
                        "pdh": pivots["pdh"], "pdl": pivots["pdl"]}
        sniper = sniper_setup.generate_sniper_setup(
            live_price=live_price, live_vwap=float(df["VWAP"].iloc[-1]), cpr_info=cpr_info,
            fvg=tc.get("smc", {}).get("fvg", []), ob=tc.get("smc", {}).get("order_blocks", []),
            df_option_chain=df_oc_adapted, atr=atr, candle_pattern=tc.get("candlestick"),
            poc_level=None, volume_is_real=True
        )
    except Exception:
        logger.exception("Stock sniper setup failed")
        sniper = None

    # ---- Position Sizing + Risk Engine + Backtest + Monte Carlo + Drift ----
    # All built from THIS symbol's own logged paper-trade history only.
    entry_price = ai_result["entry_price"] if ai_result else live_price
    sl = ai_result["stop_loss"] if ai_result else (live_price - atr)
    try:
        position_size_result = position_sizing.calculate_position_size(
            capital=capital, risk_per_trade_pct=risk_pct, entry_price=entry_price, stop_loss_price=sl)
    except Exception:
        logger.exception("Stock position sizing failed")
        position_size_result = {"status": "UNAVAILABLE"}
    try:
        closed_trades = database.fetch_all_closed_trades(symbol=symbol)
    except Exception:
        logger.exception("Stock closed trades fetch failed")
        closed_trades = []
    try:
        risk_state = risk_engine.evaluate_risk_state(closed_trades, capital=capital)
    except Exception:
        logger.exception("Stock risk engine failed")
        risk_state = {"status": "UNAVAILABLE", "block_reasons": []}
    try:
        backtest_report = backtest_engine.generate_backtest_report(closed_trades)
    except Exception:
        logger.exception("Stock backtest failed")
        backtest_report = {"status": "UNAVAILABLE"}
    try:
        monte_carlo_report = monte_carlo.run_monte_carlo(closed_trades)
    except Exception:
        logger.exception("Stock monte carlo failed")
        monte_carlo_report = {"status": "UNAVAILABLE"}
    try:
        drift_report = drift_detection.check_for_drift(closed_trades)
    except Exception:
        logger.exception("Stock drift detection failed")
        drift_report = {"status": "NOT_ENOUGH_DATA"}

    base.update({
        "full_dashboard_status": "OK",
        "regime": regime, "regime_guidance": regime_guidance, "is_choppy": is_choppy,
        "pivots": pivots, "cpr_position": cpr_pos, "opening_range": orb,
        "ai_result": ai_result, "sniper": sniper,
        "position_size": position_size_result, "risk_state": risk_state,
        "backtest_report": backtest_report, "monte_carlo_report": monte_carlo_report,
        "drift_report": drift_report, "trade_history_count": len(closed_trades),
    })
    return base
