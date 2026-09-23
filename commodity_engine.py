"""Gold/Silver intelligence engine.

Commodity mode intentionally mirrors the multi-stock pipeline but replaces
company fundamentals with physical/macro commodity fundamentals.

No fabricated fundamentals/news. Missing fields are UNAVAILABLE.
Preferred symbols: GOLD (XAU/USD) and SILVER (XAG/USD). The engine can also
be pointed at an Indian/MCX Yahoo symbol through the asset profile.
"""
from app_logging import get_logger
logger = get_logger(__name__)
import urllib.parse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
try:
    import feedparser
except Exception:
    feedparser = None
try:
    import yfinance as yf
except Exception:
    yf = None

import indicators
import support_resistance
import smart_money
import liquidity_engine
import market_regime

COMMODITIES = {
    "GOLD": {
        "name": "Gold",
        "spot_symbol": "XAUUSD=X",
        "futures_symbol": "GC=F",
        "asset_class": "PRECIOUS_METAL",
        "unit": "USD/troy oz",
        "macro_drivers": ["DXY", "US real yields", "US rates", "inflation", "central-bank demand", "ETF flows", "geopolitical risk", "mine supply", "recycling", "jewellery demand", "Indian demand"],
        "news_query": "gold bullion XAU USD central banks Fed yields inflation geopolitics India demand",
    },
    "SILVER": {
        "name": "Silver",
        "spot_symbol": "XAGUSD=X",
        "futures_symbol": "SI=F",
        "asset_class": "PRECIOUS_METAL",
        "unit": "USD/troy oz",
        "macro_drivers": ["DXY", "US real yields", "US rates", "inflation", "industrial demand", "solar demand", "electronics demand", "mine supply", "recycling", "ETF flows", "gold-silver ratio", "geopolitical risk"],
        "news_query": "silver bullion XAG USD industrial demand solar supply deficit Fed yields India",
    },
}



XAU_API_URL = "https://xaus.com/api/v1/spot?compact=1"
XAU_INTRADAY_URL = "https://xaus.com/api/v1/intraday"

def _xaus_spot(asset):
    """Free keyless XAU/XAG spot feed. Returns real observed values with freshness metadata.
    XAUS documents XAU/XAG spot as indicative mid-market rates; never labels them as
    broker-executable quotes.
    """
    symbol = "xau" if asset == "GOLD" else "xag"
    try:
        r = requests.get(XAU_API_URL, timeout=8)
        r.raise_for_status()
        payload = r.json()
        block = payload.get(symbol) or {}
        price = block.get("price")
        state = payload.get("data_state") or {}
        if price is None:
            return None
        return {
            "price": float(price),
            "currency": block.get("currency", "USD"),
            "unit": block.get("unit", "oz"),
            "data_state": state,
            "updated_at": payload.get("updated_at"),
            "source": "XAUS keyless spot API",
        }
    except Exception as exc:
        logger.warning("XAUS spot failed for %s: %s", asset, exc)
        return None

def _xaus_intraday(asset, hours=48):
    """Build OHLCV-like bars from the free XAUS 2-minute recorded intraday series."""
    symbol = "xau" if asset == "GOLD" else "xag"
    try:
        r = requests.get(XAU_INTRADAY_URL, params={"symbol": symbol, "hours": max(1, min(int(hours), 48))}, timeout=10)
        r.raise_for_status()
        payload = r.json()
        rows = payload.get("data") or payload.get("series") or payload.get("points") or []
        parsed = []
        for row in rows:
            if isinstance(row, dict):
                ts = row.get("timestamp") or row.get("time") or row.get("t")
                price = row.get("price") or row.get("close") or row.get("p")
                vol = row.get("volume") or row.get("v") or 0
            elif isinstance(row, (list, tuple)) and len(row) >= 2:
                ts, price = row[0], row[1]; vol = row[2] if len(row) > 2 else 0
            else:
                continue
            if ts is None or price is None:
                continue
            parsed.append((pd.to_datetime(ts, unit="ms" if isinstance(ts, (int, float)) and ts > 10_000_000_000 else None, errors="coerce", utc=True), float(price), float(vol or 0)))
        if len(parsed) < 20:
            return None
        d = pd.DataFrame(parsed, columns=["Timestamp", "price", "Volume"]).dropna(subset=["Timestamp", "price"]).sort_values("Timestamp").set_index("Timestamp")
        # XAUS intraday is a price series, not exchange OHLCV. Build transparent bars from
        # observed prices; volume is kept at zero because no fake volume is created.
        out = pd.DataFrame(index=d.index)
        out["Open"] = d["price"].shift(1).fillna(d["price"])
        out["High"] = d["price"].combine(out["Open"], max)
        out["Low"] = d["price"].combine(out["Open"], min)
        out["Close"] = d["price"]
        out["Volume"] = 0.0
        out["DataQuality"] = "SPOT_OBSERVED_NO_EXCHANGE_VOLUME"
        return out
    except Exception as exc:
        logger.warning("XAUS intraday failed for %s: %s", asset, exc)
        return None

def _live_spot_snapshot(asset):
    snap = _xaus_spot(asset)
    if snap:
        return snap
    return None

def _upstox_search_mcx_contract(access_token, asset):
    """Resolve the current/next MCX futures contract without hardcoding expiry."""
    if not access_token:
        return None
    query = "GOLD" if asset == "GOLD" else "SILVER"
    headers = {"Accept": "application/json", "Authorization": f"Bearer {access_token}"}
    base = "https://api.upstox.com/v2/instruments/search"
    for expiry in ("current_month", "next_month"):
        try:
            r = requests.get(base, headers=headers, params={
                "query": query, "exchanges": "MCX", "segments": "FO",
                "instrument_types": "FUT", "expiry": expiry,
                "page_number": 1, "records": 30,
            }, timeout=8)
            if not r.ok:
                continue
            rows = r.json().get("data") or []
            rows = [x for x in rows if x.get("segment") == "MCX_FO" and x.get("instrument_type") == "FUT"]
            if rows:
                rows.sort(key=lambda x: str(x.get("expiry") or "9999-99-99"))
                return rows[0]
        except Exception as exc:
            logger.warning("MCX contract search failed for %s: %s", asset, exc)
    return None


def _upstox_candles(access_token, instrument_key, interval=5, days_back=10):
    if not access_token or not instrument_key:
        return None
    headers = {"Accept": "application/json", "Authorization": f"Bearer {access_token}"}
    key = urllib.parse.quote(instrument_key, safe="")
    from datetime import timedelta
    now = datetime.now()
    to_date = now.strftime("%Y-%m-%d")
    from_date = (now - timedelta(days=days_back)).strftime("%Y-%m-%d")
    rows = []
    try:
        u = f"https://api.upstox.com/v3/historical-candle/{key}/minutes/{int(interval)}/{to_date}/{from_date}"
        r = requests.get(u, headers=headers, timeout=10)
        if r.ok and r.json().get("status") == "success":
            rows = r.json().get("data", {}).get("candles", [])
    except Exception as exc:
        logger.warning("MCX historical candles failed: %s", exc)
    if not rows:
        try:
            u = f"https://api.upstox.com/v3/historical-candle/intraday/{key}/minutes/{int(interval)}"
            r = requests.get(u, headers=headers, timeout=10)
            if r.ok and r.json().get("status") == "success":
                rows = r.json().get("data", {}).get("candles", [])
        except Exception as exc:
            logger.warning("MCX intraday candles failed: %s", exc)
    if not rows:
        return None
    out = []
    for row in rows:
        if len(row) < 6:
            continue
        out.append([row[0], row[1], row[2], row[3], row[4], row[5], row[6] if len(row) > 6 else 0])
    if not out:
        return None
    df = pd.DataFrame(out, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume", "Open Interest"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df.set_index("Timestamp", inplace=True)
    return df.sort_index()


def _upstox_quote(access_token, instrument_key):
    if not access_token or not instrument_key:
        return None
    try:
        r = requests.get(
            "https://api.upstox.com/v3/market-quote/quotes",
            headers={"Accept": "application/json", "Authorization": f"Bearer {access_token}"},
            params={"instrument_key": instrument_key}, timeout=8,
        )
        if not r.ok:
            return None
        data = r.json().get("data") or {}
        # Upstox keys may use ':' even though requests use '|'.
        item = data.get(instrument_key.replace("|", ":")) or data.get(instrument_key)
        if not item:
            item = next(iter(data.values()), None)
        return item
    except Exception as exc:
        logger.warning("MCX quote failed: %s", exc)
        return None

def _download(symbol, period="10d", interval="5m"):
    # Generic fallback for macro symbols. Commodity assets use _download_commodity.
    if yf is None:
        return None
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df is None or df.empty:
            return None
        return df.dropna(how="all")
    except Exception as exc:
        logger.warning("Commodity price download failed for %s: %s", symbol, exc)
        return None

def _download_commodity(asset, period="10d", interval="5m"):
    # PRIMARY: Yahoo Finance COMEX futures (GC=F / SI=F) -- free, keyless,
    # real, live-updating, and has enough intraday history for 15m/1h
    # resampling. Futures track spot extremely closely (a small, usually
    # tiny carry basis) and are the standard global reference price.
    #
    # The earlier primary feed here (a "xaus.com" endpoint) does not
    # appear to be a real, reachable API -- it was returning no usable
    # data on every call, which is exactly why 15m/1h/4h/1d and the
    # Gold/Silver ratio were always showing UNAVAILABLE/N/A. Replaced
    # rather than kept as a fallback, since a dead endpoint that always
    # errors adds latency for zero benefit.
    futures_symbol = COMMODITIES[asset].get("futures_symbol")
    if futures_symbol:
        df = _download(futures_symbol, period=period, interval=interval)
        if df is not None and not df.empty:
            return df, "YAHOO_FINANCE_COMEX_FUTURES"
    # Secondary fallback: Yahoo Finance forex-style spot cross symbol.
    symbol = COMMODITIES[asset]["spot_symbol"]
    df = _download(symbol, period=period, interval=interval)
    if df is not None and not df.empty:
        return df, "YAHOO_FINANCE_SPOT_FALLBACK"
    return None, "UNAVAILABLE"


def _download_commodity_long(asset, period="60d", interval="1h"):
    """Supplementary longer-history feed used for the 4H/1D rows of the
    multi-timeframe table. Yahoo Finance's hourly COMEX futures series
    comfortably covers 60 days (well within its ~730-day cap for 1h data),
    which is real, live-updating exchange data -- used for nothing except
    the higher-timeframe trend read."""
    futures_symbol = COMMODITIES[asset].get("futures_symbol")
    if futures_symbol:
        df = _download(futures_symbol, period=period, interval=interval)
        if df is not None and not df.empty:
            return df, "YAHOO_FINANCE_COMEX_FUTURES_HOURLY"
    symbol = COMMODITIES[asset]["spot_symbol"]
    df = _download(symbol, period=period, interval=interval)
    if df is not None and not df.empty:
        return df, "YAHOO_FINANCE_HOURLY_HISTORY"
    return None, "UNAVAILABLE"


def _prepare(df):
    if df is None or df.empty:
        return None
    df = df.copy()
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c not in df.columns:
            return None
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)
    if df.empty:
        return None
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs(),
    ], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(span=14, adjust=False).mean()
    try:
        df["RSI"] = indicators.calculate_rsi(df, 14).fillna(50)
    except Exception:
        df["RSI"] = 50.0
    try:
        df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = indicators.calculate_macd(df)
    except Exception:
        df["MACD"] = df["MACD_Signal"] = df["MACD_Hist"] = 0.0
    df.ffill(inplace=True)
    df.fillna(0, inplace=True)
    return df


def _last_change(symbol, period="10d", interval="1d"):
    d = _download(symbol, period=period, interval=interval)
    if d is None or d.empty or "Close" not in d:
        return {"status": "UNAVAILABLE", "symbol": symbol}
    c = pd.to_numeric(d["Close"], errors="coerce").dropna()
    return {
        "status": "AVAILABLE",
        "symbol": symbol,
        "last": float(c.iloc[-1]),
        "change_pct": float((c.iloc[-1] / c.iloc[-2] - 1) * 100) if len(c) > 1 else None,
        "timestamp": str(c.index[-1]),
        "source": "Yahoo Finance",
    }


def _macro_snapshot():
    # These are context inputs, not a claim that every feed is currently live.
    symbols = {
        "dxy": "DX-Y.NYB",
        "us10y": "^TNX",
        "oil": "CL=F",
        "copper": "HG=F",
        "sp500": "^GSPC",
    }
    out = {}
    for key, sym in symbols.items():
        out[key] = _last_change(sym, period="10d", interval="1d")
    # Gold/silver ratio is computed from spot closes when both are available.
    gold_snap = _xaus_spot("GOLD")
    silver_snap = _xaus_spot("SILVER")
    gold = {"status":"AVAILABLE", "last":gold_snap["price"], "source":gold_snap["source"]} if gold_snap else _last_change(COMMODITIES["GOLD"]["futures_symbol"], period="10d", interval="1d")
    silver = {"status":"AVAILABLE", "last":silver_snap["price"], "source":silver_snap["source"]} if silver_snap else _last_change(COMMODITIES["SILVER"]["futures_symbol"], period="10d", interval="1d")
    if gold.get("status") == "AVAILABLE" and silver.get("status") == "AVAILABLE" and silver.get("last"):
        out["gold_silver_ratio"] = {
            "status": "AVAILABLE",
            "value": gold["last"] / silver["last"],
            "source": "Calculated from Yahoo Finance spot prices",
        }
    else:
        out["gold_silver_ratio"] = {"status": "UNAVAILABLE"}
    return out


def _commodity_fundamentals(asset):
    """Return structural commodity fundamentals, not company-style metrics."""
    a = COMMODITIES[asset]
    if asset == "GOLD":
        factors = {
            "central_bank_demand": "WATCH",
            "investment_etf_flows": "WATCH",
            "jewellery_demand": "WATCH",
            "mine_supply": "WATCH",
            "recycling_supply": "WATCH",
            "technology_demand": "WATCH",
            "indian_demand": "WATCH",
            "real_yield_sensitivity": "HIGH",
            "usd_sensitivity": "HIGH",
            "geopolitical_safe_haven_sensitivity": "HIGH",
        }
        sources = ["World Gold Council research", "market macro feeds"]
    else:
        factors = {
            "industrial_demand": "WATCH",
            "solar_demand": "WATCH",
            "electronics_demand": "WATCH",
            "mine_supply": "WATCH",
            "recycling_supply": "WATCH",
            "investment_etf_flows": "WATCH",
            "indian_demand": "WATCH",
            "gold_silver_ratio": "WATCH",
            "real_yield_sensitivity": "HIGH",
            "usd_sensitivity": "HIGH",
            "industrial_cycle_sensitivity": "HIGH",
        }
        sources = ["CME/market research", "market macro feeds"]
    return {
        "status": "STRUCTURAL_AVAILABLE",
        "asset": asset,
        "asset_class": a["asset_class"],
        "method": "Commodity fundamentals: physical supply/demand + macro opportunity-cost and risk drivers. This is not company financial data.",
        "drivers": a["macro_drivers"],
        "factor_monitor": factors,
        "sources": sources,
    }


def _news(asset, limit=10):
    q = COMMODITIES[asset]["news_query"]
    if feedparser is None:
        return [], "UNAVAILABLE_DEPENDENCY"
    try:
        url = f"https://news.google.com/rss/search?q={urllib.parse.quote(q)}&hl=en-IN&gl=IN&ceid=IN:en"
        feed = feedparser.parse(url)
        rows = []
        for e in feed.entries[:limit]:
            rows.append({
                "title": e.get("title", ""),
                "published": e.get("published", ""),
                "source": e.get("source", {}).get("title", "") if isinstance(e.get("source"), dict) else "",
                "link": e.get("link", ""),
            })
        return rows, ("AVAILABLE" if rows else "UNAVAILABLE")
    except Exception as exc:
        logger.warning("Commodity news failed for %s: %s", asset, exc)
        return [], "UNAVAILABLE"


def _technical(df, live_override=None):
    live = float(live_override) if live_override is not None else float(df["Close"].iloc[-1])
    atr = float(df["ATR"].iloc[-1])
    levels = support_resistance.get_key_levels(df)
    supports = [x for x in levels.get("supports", []) if x < live]
    resistances = [x for x in levels.get("resistances", []) if x > live]
    support = max(supports) if supports else None
    resistance = min(resistances) if resistances else None
    ema20 = float(df["EMA_20"].iloc[-1]); ema50 = float(df["EMA_50"].iloc[-1]); rsi = float(df["RSI"].iloc[-1])
    trend = "BULLISH" if live > ema20 > ema50 else "BEARISH" if live < ema20 < ema50 else "MIXED"
    momentum = "OVERBOUGHT/EXTENDED" if rsi >= 70 else "OVERSOLD/WEAK" if rsi <= 30 else "NEUTRAL"
    return {
        "live_price": live, "atr": atr, "ema20": ema20, "ema50": ema50, "rsi": rsi,
        "trend": trend, "momentum": momentum, "support": support, "resistance": resistance,
        "macd_hist": float(df["MACD_Hist"].iloc[-1]), "timestamp": str(df.index[-1]),
    }


def analyze_commodity(asset, period="10d", interval="5m", access_token=None):
    asset = str(asset).upper().strip()
    if asset not in COMMODITIES:
        return {"status": "INVALID", "asset": asset}
    profile = COMMODITIES[asset]

    # PRIMARY MODE: global Forex/spot precious-metal instruments XAU/USD and XAG/USD.
    # MCX is intentionally kept separate and is never silently mixed into the spot series.
    # This makes a displayed price such as 4,310.53 unambiguously a USD/troy-oz spot quote.
    df_raw, feed_source = _download_commodity(asset, period=period, interval=interval)
    live_snap = _live_spot_snapshot(asset)
    _pair = "XAU/USD" if asset == "GOLD" else "XAG/USD"
    market_source = (f"COMEX FUTURES (spot proxy) {_pair}" if feed_source == "YAHOO_FINANCE_COMEX_FUTURES"
                      else f"GLOBAL SPOT {_pair}")
    source_meta = {
        "market": "GLOBAL_SPOT",
        "symbol": profile["spot_symbol"],
        "quote_convention": "USD per troy ounce",
        "primary": True,
        "feed": feed_source,
        "live_spot": live_snap,
        "mcx_mode": "SEPARATE_OPTIONAL",
    }

    # Optional MCX metadata is exposed separately for users who explicitly want Indian
    # futures. It is never used to overwrite the XAU/USD or XAG/USD price series.
    mcx_contract = _upstox_search_mcx_contract(access_token, asset)
    if mcx_contract:
        source_meta["mcx_reference"] = {
            "instrument_key": mcx_contract.get("instrument_key"),
            "trading_symbol": mcx_contract.get("trading_symbol"),
            "expiry": mcx_contract.get("expiry"),
            "lot_size": mcx_contract.get("lot_size"),
            "tick_size": mcx_contract.get("tick_size"),
        }

    df = _prepare(df_raw)
    if df is None or df.empty:
        return {
            "status": "DATA_UNAVAILABLE", "asset": asset,
            "message": "No usable commodity price data from the configured live/spot sources.",
            "data_source": "UNAVAILABLE", "market_source": market_source, "contract": source_meta,
        }
    tech = _technical(df, live_override=(live_snap.get("price") if live_snap else None))
    macro = _macro_snapshot()
    fundamentals = _commodity_fundamentals(asset)
    news, news_status = _news(asset)

    # Reuse the same structural engines used elsewhere in the dashboard.
    try:
        fvgs, order_blocks, sweeps = smart_money.detect_smc_zones(df)
        smc_structure = smart_money.detect_market_structure(df)
        smc = {"status": "AVAILABLE", "fvgs": fvgs, "order_blocks": order_blocks, "liquidity_sweeps": sweeps, "market_structure": smc_structure}
    except Exception as exc:
        smc = {"status": "UNAVAILABLE", "reason": type(exc).__name__}
    try:
        liquidity = liquidity_engine.build_liquidity_map(df, tech["live_price"], tech["atr"])
        liquidity_target = liquidity_engine.nearest_liquidity_target(liquidity, tech["live_price"])
        liquidity["nearest_target"] = liquidity_target
    except Exception as exc:
        liquidity = {"status": "UNAVAILABLE", "reason": type(exc).__name__}
    try:
        regime = market_regime.detect_regime(df, live_vix=None)
    except Exception as exc:
        regime = {"primary": "UNAVAILABLE", "volatility": "UNAVAILABLE", "structure": "UNAVAILABLE", "reason": type(exc).__name__}

    # Local multi-timeframe stack. 15m/1h come from the same short price
    # history as the rest of this analysis; 4h/1d need more calendar span
    # than the 48-hour free feed can ever hold, so they resample a
    # supplementary longer Yahoo history fetched just for this table.
    mtf = {}
    try:
        df_long, _df_long_source = _download_commodity_long(asset)
        short_rules = (("15m", "15min"), ("1h", "1h"))
        long_rules = (("4h", "4h"), ("1d", "1D"))
        for label, rule in short_rules:
            x = df.resample(rule).agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
            if len(x) >= 25:
                c = float(x["Close"].iloc[-1]); e20 = float(x["Close"].ewm(span=20, adjust=False).mean().iloc[-1]); e50 = float(x["Close"].ewm(span=50, adjust=False).mean().iloc[-1])
                mtf[label] = {"status":"AVAILABLE", "trend":"BULLISH" if c > e20 > e50 else "BEARISH" if c < e20 < e50 else "MIXED", "close":c}
            else:
                mtf[label] = {"status":"UNAVAILABLE", "reason":"Insufficient resampled history"}
        for label, rule in long_rules:
            src = df_long if (df_long is not None and not df_long.empty) else df
            x = src.resample(rule).agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
            if len(x) >= 20:
                c = float(x["Close"].iloc[-1]); e20 = float(x["Close"].ewm(span=20, adjust=False).mean().iloc[-1]); e50 = float(x["Close"].ewm(span=50, adjust=False).mean().iloc[-1])
                mtf[label] = {"status":"AVAILABLE", "trend":"BULLISH" if c > e20 > e50 else "BEARISH" if c < e20 < e50 else "MIXED", "close":c,
                              "source": _df_long_source if src is df_long else "SHORT_HISTORY_FALLBACK"}
            else:
                mtf[label] = {"status":"UNAVAILABLE", "reason":"Insufficient resampled history"}
    except Exception as exc:
        mtf = {"status":"UNAVAILABLE", "reason":type(exc).__name__}

    # Context score only. It is deliberately not a guaranteed trade signal.
    bullish = bearish = 0
    if tech["trend"] == "BULLISH": bullish += 2
    elif tech["trend"] == "BEARISH": bearish += 2
    if tech["rsi"] >= 70: bearish += 1
    elif tech["rsi"] <= 30: bullish += 1

    dxy = macro.get("dxy", {})
    if dxy.get("status") == "AVAILABLE":
        if dxy.get("change_pct", 0) > 0.3: bearish += 1
        elif dxy.get("change_pct", 0) < -0.3: bullish += 1
    us10y = macro.get("us10y", {})
    if us10y.get("status") == "AVAILABLE":
        if us10y.get("change_pct", 0) > 1.0: bearish += 1
        elif us10y.get("change_pct", 0) < -1.0: bullish += 1

    bias = "NO TRADE / MIXED" if abs(bullish - bearish) < 2 else ("BULLISH BIAS" if bullish > bearish else "BEARISH BIAS")

    # Structural trade candidate: only expose a level when a real structural
    # support/resistance exists. It is a candidate for the AI decision layer,
    # not an automatic order and not a guaranteed prediction.
    trade_candidate = {"status": "NO_VALIDATED_SETUP"}
    if bias == "BULLISH BIAS" and tech.get("support") is not None:
        entry = tech["live_price"]
        stop = float(tech["support"] - max(0.35 * tech["atr"], 0.05 * tech["atr"]))
        risk = entry - stop
        if risk > 0 and tech.get("resistance") is not None and tech["resistance"] > entry:
            target = float(tech["resistance"])
            trade_candidate = {"status":"CANDIDATE", "direction":"BUY", "entry_reference":entry, "stop_loss":stop, "target":target, "rr":round((target-entry)/risk,2) if risk else None, "reason":"Bullish structure + real support below price; AI must revalidate full context before any trade."}
    elif bias == "BEARISH BIAS" and tech.get("resistance") is not None:
        entry = tech["live_price"]
        stop = float(tech["resistance"] + max(0.35 * tech["atr"], 0.05 * tech["atr"]))
        risk = stop - entry
        if risk > 0 and tech.get("support") is not None and tech["support"] < entry:
            target = float(tech["support"])
            trade_candidate = {"status":"CANDIDATE", "direction":"SELL", "entry_reference":entry, "stop_loss":stop, "target":target, "rr":round((entry-target)/risk,2) if risk else None, "reason":"Bearish structure + real resistance above price; AI must revalidate full context before any trade."}

    return {
        "status": "OK", "asset": asset, "name": profile["name"],
        "asset_class": profile["asset_class"], "unit": profile["unit"],
        "data_source": market_source,
        "contract": source_meta,
        "open_interest": float(df["Open Interest"].iloc[-1]) if "Open Interest" in df.columns else None,
        "technical": tech, "fundamentals": fundamentals, "macro_context": macro,
        "smc_context": smc, "liquidity_context": liquidity, "market_regime": regime, "multi_timeframe": mtf,
        "news": news, "news_status": news_status, "trade_candidate": trade_candidate,
        "bullish_score": bullish, "bearish_score": bearish, "decision": bias,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
