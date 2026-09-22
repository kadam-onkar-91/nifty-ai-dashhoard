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


def _upstox_candles(access_token, instrument_key, interval="5", days_back=5):
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


def _yf_candles(yf_symbol, period="5d", interval="5m"):
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
            if len(x) < 20:
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
