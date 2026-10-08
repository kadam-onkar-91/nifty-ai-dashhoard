"""
oi_buildup.py -- "is fresh money (OI) backing this move, or is it running out of fuel?"

Uses the option chain the app already loads.  Upstox gives every strike's OI AND yesterday's closing OI (prev_oi), so the OI added
TODAY near the price is known:
    Call OI added  = call writers (sellers) are capping the market            -> bearish fuel
    Put  OI added  = put writers are defending the market                      -> bullish fuel
    Call OI cut    = call writers running (short covering)                    -> bullish
    Put  OI cut    = put writers running (long unwinding)                     -> bearish
Net fuel = (change in Put OI - change in Call OI) / total OI, inside +-BAND points of the price.

Combined with the price direction this is the classic "buildup" read:
    price down + bearish fuel  -> SHORT BUILDUP   (the fall has fresh sellers: SELL is backed)
    price down + bullish fuel  -> fall is being absorbed (put writers adding / calls unwinding): SELLING HERE RISKS A BOUNCE
    price up   + bullish fuel  -> LONG BUILDUP    (BUY is backed)
    price up   + bearish fuel  -> rally is being sold into: BUYING HERE RISKS A REJECTION

NOTE: this is an OPTION-chain OI read (a proxy).  True futures OI needs the futures instrument key and is not used here.
Returns a soft penalty / small bonus only (never a hard block), and stays neutral when prev-OI data is missing (mock chain).
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

BAND = 300.0                # strikes within this many points of the price
MIN_NET_FUEL = 0.02         # net fuel below 2% of total OI = no clear bias
AGAINST_PENALTY = 0.20      # logit cost when OI flow is against the trade
CONFIRM_BONUS = 0.10        # logit bonus when OI flow backs the trade (subtracted from the penalty)
PRICE_MOVE_MIN_PCT = 0.30   # |day change| below this = price direction unknown / flat


def assess(df_chain, live_price, direction, price_change_pct=None):
    out = {"available": False, "bias": "NEUTRAL", "label": "OI buildup: no data", "d_call": 0.0, "d_put": 0.0,
           "net_fuel": 0.0, "confirms": False, "against": False, "penalty_logit": 0.0, "flow": 0}
    try:
        if not isinstance(df_chain, pd.DataFrame) or df_chain.empty or direction not in ("BUY", "SELL"):
            return out
        need = {"Strike", "Call OI", "Put OI", "Call Prev OI", "Put Prev OI"}
        if not need.issubset(df_chain.columns):
            return out
        d = df_chain[(df_chain["Strike"] - float(live_price)).abs() <= BAND]
        if d.empty:
            return out
        call_now, put_now = float(d["Call OI"].sum()), float(d["Put OI"].sum())
        call_prev, put_prev = float(d["Call Prev OI"].sum()), float(d["Put Prev OI"].sum())
        if call_prev <= 0 or put_prev <= 0 or (call_now + put_now) <= 0:
            return out                                               # prev OI not provided (e.g. simulated chain)
        d_call, d_put = call_now - call_prev, put_now - put_prev
        net = (d_put - d_call) / (call_now + put_now)
        bias = "BUY" if net >= MIN_NET_FUEL else "SELL" if net <= -MIN_NET_FUEL else "NEUTRAL"
        out.update({"available": True, "bias": bias, "d_call": round(d_call), "d_put": round(d_put), "net_fuel": round(net, 4)})

        move = None
        if price_change_pct is not None:
            pc = float(price_change_pct)
            move = "UP" if pc >= PRICE_MOVE_MIN_PCT else "DOWN" if pc <= -PRICE_MOVE_MIN_PCT else None

        if move == "DOWN" and bias == "SELL":
            label = "SHORT BUILDUP: the fall has fresh call-writing behind it (selling is backed)"
        elif move == "DOWN" and bias == "BUY":
            label = "Fall is being ABSORBED: put writers adding / calls unwinding (selling here risks a bounce)"
        elif move == "UP" and bias == "BUY":
            label = "LONG BUILDUP: the rally has fresh put-writing behind it (buying is backed)"
        elif move == "UP" and bias == "SELL":
            label = "Rally is being SOLD INTO: call writers adding / puts unwinding (buying here risks a rejection)"
        elif bias == "NEUTRAL":
            label = "OI flow is balanced today (no clear writer pressure)"
        else:
            label = f"OI flow today favours {'buyers' if bias == 'BUY' else 'sellers'} (put/call OI change {d_put:+,.0f}/{d_call:+,.0f})"
        out["label"] = label

        if bias == direction:
            out["confirms"], out["flow"] = True, 1
            out["penalty_logit"] = -CONFIRM_BONUS
        elif bias != "NEUTRAL":
            out["against"], out["flow"] = True, -1
            out["penalty_logit"] = AGAINST_PENALTY
    except Exception:
        logger.exception("oi_buildup.assess failed (ignored)")
    return out
