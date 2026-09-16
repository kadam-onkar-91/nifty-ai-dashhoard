import requests
import urllib.parse
import market_data

"""
PHASE 3 -- ORDER FLOW / MARKET MICROSTRUCTURE ENGINE
------------------------------------------------------
Honesty note up front: true tick-by-tick Cumulative Volume Delta (CVD)
needs a live WebSocket tick feed classifying every trade as buyer- or
seller-initiated. This app polls a REST snapshot every ~30s (Streamlit
fragment refresh), so it CANNOT produce real tick-level CVD -- and per
this project's own rule ("never fabricate CVD/order-flow values"), it
does not pretend to.

What IS real and available: Upstox's free market-quote/quotes endpoint
(the same one this app already calls for LTP/OHLC) returns a `depth`
object with the top-5 resting Buy and Sell orders -- real, live order
book data, no paid Level-2 subscription required. From this we compute
a genuine, live **Order Book Imbalance (OBI)**: how much more resting
buy quantity vs sell quantity sits in the top 5 levels right now. This
is a real, standard order-flow metric (just not the same thing as CVD),
so it's reported honestly under its own name rather than mislabeled.

Note the underlying instrument: the NIFTY spot index itself has no
order book at all (an index isn't directly tradable). So this reads
depth from the near-month NIFTY FUTURES contract instead -- the same
instrument market_data.py already uses for real Volume, for the same
reason (futures are genuinely, continuously traded).
"""


def get_order_flow_snapshot(access_token):
    """
    Returns a dict describing the current live Order Book Imbalance on
    the near-month NIFTY futures contract. On any failure (no token, no
    futures key resolvable, no depth in the response, market closed with
    an empty book), returns a DATA_UNAVAILABLE status instead of a
    fabricated number.
    """
    out = {
        "status": "DATA_UNAVAILABLE", "reason": None,
        "instrument": None, "total_buy_qty": None, "total_sell_qty": None,
        "imbalance_pct": None, "pressure": None, "best_bid": None, "best_ask": None,
        "spread": None,
    }
    if not access_token:
        out["reason"] = "No Upstox access token -- login required for live order book data."
        return out

    fut_key = market_data.get_nifty_futures_instrument_key()
    if not fut_key:
        out["reason"] = "Could not resolve the near-month NIFTY futures instrument_key from Upstox's instrument master."
        return out
    out["instrument"] = fut_key

    try:
        encoded_key = urllib.parse.quote(fut_key, safe="|")
        url = f"https://api.upstox.com/v2/market-quote/quotes?instrument_key={encoded_key}"
        headers = {"Accept": "application/json", "Authorization": f"Bearer {access_token}"}
        res = requests.get(url, headers=headers, timeout=6)
        res_json = res.json()
        if res_json.get("status") != "success":
            out["reason"] = f"Upstox API did not return success: {res_json.get('status')}"
            return out
        data = res_json.get("data", {})
        match = next(iter(data.values()), None)
        if match is None:
            out["reason"] = "Upstox returned no quote data for the futures instrument."
            return out

        depth = match.get("depth", {})
        buy_levels = depth.get("buy", [])
        sell_levels = depth.get("sell", [])
        if not buy_levels and not sell_levels:
            out["reason"] = "Upstox response had no depth data (market likely closed or feed not entitled)."
            return out

        total_buy_qty = sum(float(lv.get("quantity", 0) or 0) for lv in buy_levels)
        total_sell_qty = sum(float(lv.get("quantity", 0) or 0) for lv in sell_levels)
        if total_buy_qty == 0 and total_sell_qty == 0:
            out["reason"] = "Order book is empty on both sides right now (no resting orders)."
            return out

        total = total_buy_qty + total_sell_qty
        imbalance = (total_buy_qty - total_sell_qty) / total if total > 0 else 0.0

        best_bid = next((lv.get("price") for lv in buy_levels if lv.get("price")), None)
        best_ask = next((lv.get("price") for lv in sell_levels if lv.get("price")), None)
        spread = (best_ask - best_bid) if (best_bid and best_ask) else None

        if imbalance > 0.15:
            pressure = "BUYING PRESSURE (Order Book) 🟢"
        elif imbalance < -0.15:
            pressure = "SELLING PRESSURE (Order Book) 🔴"
        else:
            pressure = "BALANCED ORDER BOOK ⚪"

        out.update({
            "status": "OK",
            "total_buy_qty": int(total_buy_qty), "total_sell_qty": int(total_sell_qty),
            "imbalance_pct": round(imbalance * 100, 1), "pressure": pressure,
            "best_bid": best_bid, "best_ask": best_ask,
            "spread": round(spread, 2) if spread is not None else None,
        })
        return out
    except Exception as e:
        out["reason"] = f"Order flow fetch failed: {type(e).__name__}: {e}"
        return out
