"""TradingView-style LIVE 1-minute candle chart for NIFTY 50.

* 1-minute candles (Upstox intraday feed), last ~3 hours shown, zoom/pan stays put.
* The candle that is forming RIGHT NOW moves up/down every few seconds from the live LTP
  (its High / Low stretch as price moves) and a new candle opens when the minute changes.
* Own tiny auto-refresh (TICK_SECONDS) that only fetches the live price -- it does NOT re-run the
  heavy dashboard pipeline.  While the market is closed there is no timer: the last session is shown.

Switch off any time with  TICK_SECONDS = 0  (the dashboard then keeps its old chart).
Pure presentation: no signal / scoring / trading logic is touched.
"""
from __future__ import annotations

import time as _time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import market_data
import market_status
import upstox_auth

TICK_SECONDS = 2        # live refresh speed while market is OPEN. 0 = feature off (old chart is used)
MAX_CANDLES = 240       # candles kept in memory / on chart
VISIBLE_CANDLES = 90    # candles visible when the chart first opens
CANDLE_TTL = 20         # seconds before 1-minute history is re-fetched from broker

UP, DOWN = "#26A69A", "#EF5350"


def enabled() -> bool:
    return bool(TICK_SECONDS and TICK_SECONDS > 0)


# ----------------------------------------------------------------------------- data
@st.cache_data(ttl=CANDLE_TTL, show_spinner=False)
def _history_1m(token: str):
    return market_data.fetch_candles_for_timeframe(token, "minutes", "1", days_back=4)


@st.cache_data(ttl=1, show_spinner=False)
def _ltp(token: str):
    return market_data._fetch_upstox_ltp(token)


def _to_naive_ist(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert("Asia/Kolkata").tz_localize(None)
    df.index = idx
    return df[["Open", "High", "Low", "Close", "Volume"]].astype(float)


def apply_live_tick(df: pd.DataFrame, ltp: float, now_naive: pd.Timestamp, forming: dict | None):
    """Merge the live price into the 1-minute frame.

    Returns (df, forming).  `forming` remembers the running High/Low of the current minute so the
    candle keeps stretching tick after tick even between broker history refreshes.
    """
    minute = now_naive.floor("min")
    prev_close = float(df["Close"].iloc[-1]) if len(df) and df.index[-1] < minute else None

    if not forming or forming.get("minute") != minute:
        o = prev_close if prev_close is not None else ltp
        forming = {"minute": minute, "o": o, "h": max(o, ltp), "l": min(o, ltp)}
    else:
        forming["h"] = max(forming["h"], ltp)
        forming["l"] = min(forming["l"], ltp)

    df = df.copy()
    if minute in df.index:                      # broker already has a (partial) candle for this minute
        row = df.loc[minute]
        df.loc[minute, "Open"] = float(row["Open"])
        df.loc[minute, "High"] = max(float(row["High"]), forming["h"])
        df.loc[minute, "Low"] = min(float(row["Low"]), forming["l"])
        df.loc[minute, "Close"] = ltp
    else:                                       # brand-new candle just opened
        df.loc[minute] = [forming["o"], forming["h"], forming["l"], ltp, 0.0]
    return df.sort_index(), forming


def _session_vwap(df: pd.DataFrame):
    """Per-day VWAP. Index has no real volume, so fall back to an equal-weight average (flagged)."""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    day = df.index.normalize()
    if df["Volume"].sum() > 0:
        pv = (tp * df["Volume"]).groupby(day).cumsum()
        v = df["Volume"].groupby(day).cumsum().replace(0, float("nan"))
        return (pv / v).ffill(), True
    return tp.groupby(day).expanding().mean().reset_index(level=0, drop=True).sort_index(), False


# ----------------------------------------------------------------------------- figure
def build_figure(df: pd.DataFrame, ltp: float | None, poc: float | None):
    d = df.tail(MAX_CANDLES)
    ema = d["Close"].ewm(span=20, adjust=False).mean()
    vwap, real_vol = _session_vwap(d)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"], name="Nifty 50 · 1m",
        increasing=dict(line=dict(color=UP, width=1), fillcolor=UP),
        decreasing=dict(line=dict(color=DOWN, width=1), fillcolor=DOWN),
        whiskerwidth=0.4))
    fig.add_trace(go.Scatter(x=d.index, y=ema, mode="lines", name="EMA 20",
                             line=dict(color="#2962FF", width=1.2), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=d.index, y=vwap, mode="lines",
                             name="VWAP" if real_vol else "VWAP (avg proxy)",
                             line=dict(color="#E040FB", width=1.8, dash="dot"), hoverinfo="skip"))

    if poc:
        fig.add_hline(y=poc, line_width=1.2, line_color="#FFEA00", opacity=0.7,
                      annotation_text=f"POC {poc:,.0f}", annotation_position="top left",
                      annotation_font_color="#FFEA00")
    if ltp:
        last_up = d["Close"].iloc[-1] >= d["Open"].iloc[-1]
        col = UP if last_up else DOWN
        fig.add_hline(y=ltp, line_width=1, line_dash="dot", line_color=col,
                      annotation_text=f" {ltp:,.2f} ", annotation_position="right",
                      annotation_font=dict(color="#fff", size=12), annotation_bgcolor=col)

    vis = d.tail(VISIBLE_CANDLES)
    lo, hi = float(vis["Low"].min()), float(vis["High"].max())
    pad = max((hi - lo) * 0.08, 5.0)
    x_end = d.index[-1] + pd.Timedelta(minutes=6)

    fig.update_layout(
        template="plotly_dark", height=520, uirevision="nifty-live-1m",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=4, r=4, t=36, b=8), hovermode="x unified", dragmode="pan",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, font=dict(size=11)),
    )
    fig.update_xaxes(
        range=[vis.index[0], x_end], tickformat="%H:%M", nticks=6, showgrid=True,
        gridcolor="rgba(255,255,255,.05)", rangebreaks=[
            dict(bounds=["sat", "mon"]),                  # weekends
            dict(bounds=[15.5, 9.25], pattern="hour"),    # overnight (NSE 09:15 - 15:30)
        ])
    fig.update_yaxes(range=[lo - pad, hi + pad], side="right", showgrid=True,
                     gridcolor="rgba(255,255,255,.06)", tickformat=",.0f")
    return fig


# ----------------------------------------------------------------------------- UI
def _header(ltp, prev_day_close, is_open):
    if ltp is None:
        return
    chg = (ltp - prev_day_close) if prev_day_close else 0.0
    pct = (chg / prev_day_close * 100) if prev_day_close else 0.0
    col = UP if chg >= 0 else DOWN
    arrow = "▲" if chg >= 0 else "▼"
    secs = 60 - pd.Timestamp.now(tz="Asia/Kolkata").second
    badge = ('<span class="lc-live"><i></i>LIVE</span>' if is_open
             else '<span class="lc-live off">MARKET CLOSED</span>')
    timer = f'<span class="lc-t">candle closes in {secs}s</span>' if is_open else ""
    st.markdown(f"""
<style>
.lc-bar{{display:flex;flex-wrap:wrap;align-items:baseline;gap:6px 14px;margin:2px 0 6px}}
.lc-px{{font:600 clamp(26px,5vw,40px)/1 'JetBrains Mono',monospace;letter-spacing:-.02em}}
.lc-ch{{font:500 15px 'JetBrains Mono',monospace}}
.lc-t{{font:500 12px 'JetBrains Mono',monospace;color:#8b93ad}}
.lc-live{{display:inline-flex;align-items:center;gap:6px;font:600 11px 'JetBrains Mono',monospace;
  letter-spacing:.12em;color:#22e6a8;padding:4px 10px;border:1px solid rgba(34,230,168,.4);border-radius:999px}}
.lc-live i{{width:7px;height:7px;border-radius:50%;background:#22e6a8;animation:lcp 1.4s infinite}}
.lc-live.off{{color:#ff8a9b;border-color:rgba(255,93,122,.4)}}
@keyframes lcp{{50%{{opacity:.25}}}}
</style>
<div class="lc-bar"><span class="lc-px">{ltp:,.2f}</span>
<span class="lc-ch" style="color:{col}">{arrow} {chg:+,.2f} ({pct:+.2f}%)</span>{badge}{timer}</div>
""", unsafe_allow_html=True)


def _body(was_open: bool):
    state = market_status.get_market_status()["state"]
    is_open = state == "OPEN"
    if is_open != was_open:          # market just opened/closed -> re-pick refresh speed once
        st.rerun()

    token = st.session_state.get("access_token")
    if not token:
        try:
            token = upstox_auth._stored_token()      # shared login token (no UI side effects)
        except Exception:
            token = None
    if not token:
        st.info("⏳ Live chart: broker login hone ka wait... (dashboard login hote hi chalu ho jaayega)")
        return

    hist = _history_1m(token)
    if hist is None or len(hist) == 0:
        st.warning("1-minute candles abhi broker se nahi mile -- thodi der mein dobara try hoga.")
        return

    df = _to_naive_ist(hist)
    ltp = None
    if is_open:
        ltp = _ltp(token)
        if ltp:
            now_naive = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)
            df, st.session_state["_lc_forming"] = apply_live_tick(
                df, float(ltp), now_naive, st.session_state.get("_lc_forming"))
    if ltp is None:
        ltp = float(df["Close"].iloc[-1])

    last_day = df.index[-1].normalize()
    prev = df[df.index.normalize() < last_day]
    prev_day_close = float(prev["Close"].iloc[-1]) if len(prev) else None

    _header(ltp, prev_day_close, is_open)
    st.plotly_chart(
        build_figure(df, ltp, st.session_state.get("_live_poc")),
        width="stretch", key="live_candle_chart",
        config={"displaylogo": False, "scrollZoom": True,
                "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"]})
    st.caption("1-minute candles · Upstox live feed · chalti hui candle har "
               f"{TICK_SECONDS}s mein update hoti hai" if is_open else
               "1-minute candles · Upstox · market band hai, aakhri session dikh raha hai")


def render() -> bool:
    """Draw the live chart. Returns False if disabled (caller then keeps its old chart)."""
    if not enabled():
        return False
    try:
        is_open = market_status.get_market_status()["state"] == "OPEN"
        frag = st.fragment(run_every=TICK_SECONDS if is_open else None)(_body)
        frag(is_open)
        return True
    except Exception:
        import logging
        logging.getLogger(__name__).exception("live_chart failed; falling back to old chart")
        return False
