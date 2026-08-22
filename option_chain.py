import math
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta


def get_valid_expiry(access_token=None, instrument_key="NSE_INDEX|Nifty 50"):
    today = datetime.now()
    days_ahead = 3 - today.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    next_thursday = today + timedelta(days=days_ahead)
    return next_thursday.strftime('%Y-%m-%d')


# -------------------------------------------------------------------
# REAL Black-Scholes Greeks (used only for the SIMULATED fallback chain
# below, when the live Upstox option chain isn't available). Previously
# every single strike showed the exact same hardcoded Vega=10.5,
# Delta=0.5, Theta=-5.2, IV=14.5% regardless of strike or moneyness --
# that was fake. This computes real Greeks from the actual spot price,
# real time-to-expiry, and the live India VIX as the IV estimate.
# -------------------------------------------------------------------
def _norm_pdf(x):
    return math.exp(-x * x / 2.0) / math.sqrt(2 * math.pi)


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_greeks(S, K, T, sigma, r=0.065, option_type="call"):
    """Black-Scholes Greeks. T = years to expiry, sigma = IV as a decimal."""
    if T <= 0:
        T = 1.0 / 365.0
    if sigma <= 0:
        sigma = 0.12
    try:
        d1 = (math.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        if option_type == "call":
            delta = _norm_cdf(d1)
            theta = (-(S * _norm_pdf(d1) * sigma) / (2 * math.sqrt(T))
                     - r * K * math.exp(-r * T) * _norm_cdf(d2)) / 365.0
        else:
            delta = _norm_cdf(d1) - 1.0
            theta = (-(S * _norm_pdf(d1) * sigma) / (2 * math.sqrt(T))
                     + r * K * math.exp(-r * T) * _norm_cdf(-d2)) / 365.0
        vega = S * _norm_pdf(d1) * math.sqrt(T) / 100.0
        return round(delta, 4), round(theta, 2), round(vega, 2)
    except Exception:
        return 0.5, -5.0, 10.0


@st.cache_data(ttl=120, show_spinner=False)
def _get_estimated_iv():
    """Real India VIX from Yahoo Finance, used as the IV estimate instead
    of a hardcoded 14.5% for every strike."""
    try:
        vix_hist = yf.Ticker("^INDIAVIX").history(period="2d")
        if not vix_hist.empty:
            return float(vix_hist['Close'].iloc[-1]) / 100.0
    except Exception:
        pass
    return 0.13  # sane fallback only if VIX itself is unreachable


def get_mock_option_chain(spot=24250.0):
    """
    SIMULATED fallback chain -- used when there's no live Upstox login/data.
    Greeks are now real Black-Scholes math (live spot, live VIX-based IV,
    real time-to-expiry). Open Interest is still a modelled estimate (no
    public free real-time OI source exists without a broker connection) --
    this is disclosed to the user via the "SIMULATED" source flag returned
    by generate_option_chain_data(), not hidden.
    """
    sigma = _get_estimated_iv()
    expiry_str = get_valid_expiry()
    try:
        expiry_dt = datetime.strptime(expiry_str, '%Y-%m-%d')
        T = max((expiry_dt - datetime.now()).days, 1) / 365.0
    except Exception:
        T = 7 / 365.0

    strikes = [spot - 200, spot - 150, spot - 100, spot - 50, spot, spot + 50, spot + 100, spot + 150, spot + 200]
    parsed_rows = []
    for strike in strikes:
        strike = round(strike / 50) * 50
        if abs(strike - spot) < 30:
            opt_type = "ATM"
            call_oi = 100000
            put_oi = 100000
        elif strike < spot:
            opt_type = "ITM"
            call_oi = int(10000 + (spot - strike) * 100)
            put_oi = int(20000 + (spot - strike) * 150)
        else:
            opt_type = "OTM"
            call_oi = int(20000 + (strike - spot) * 150)
            put_oi = int(10000 + (strike - spot) * 100)

        delta, theta, vega = _bs_greeks(spot, strike, T, sigma, option_type="call")

        parsed_rows.append({
            "Strike": strike,
            "Type": opt_type,
            "Call OI": call_oi,
            "Vega": vega,
            "Put OI": put_oi,
            "IV (%)": round(sigma * 100, 2),
            "Delta": delta,
            "Theta": theta,
            "PCR": round(put_oi / call_oi, 2)
        })
    return pd.DataFrame(parsed_rows)


def generate_option_chain_data(current_price=None):
    """
    Returns (df_option_chain, source) where source is "LIVE" (real Upstox
    option chain with real OI/Greeks) or "SIMULATED" (Black-Scholes
    estimate, disclosed -- see get_mock_option_chain). Previously this
    silently returned a mock table with no way for the UI to tell the
    difference from live data.
    """
    possible_keys = ['access_token', 'UPSTOX_ACCESS_TOKEN', 'token', 'upstox_token', 'auth_token']
    access_token = None

    for key in possible_keys:
        if key in st.session_state and st.session_state[key]:
            access_token = st.session_state[key]
            break

    spot = current_price if current_price else 24250.0

    if not access_token:
        return get_mock_option_chain(spot), "SIMULATED"

    instrument_key = "NSE_INDEX|Nifty 50"
    expiry_date = get_valid_expiry(access_token, instrument_key)

    url = "https://api.upstox.com/v2/option/chain"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {access_token}"
    }
    params = {
        "instrument_key": instrument_key,
        "expiry_date": expiry_date
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=8)
        if response.status_code == 200:
            res_json = response.json()
            data_list = res_json.get('data', [])

            if data_list:
                parsed_rows = []
                for item in data_list:
                    strike = item.get('strike_price', 0)

                    call_options = item.get('call_options', {})
                    call_market = call_options.get('market_data', {})
                    call_greeks = call_options.get('option_greeks', {})
                    call_oi = call_market.get('oi', 0)

                    put_options = item.get('put_options', {})
                    put_market = put_options.get('market_data', {})
                    put_oi = put_market.get('oi', 0)

                    if abs(strike - spot) < 30:
                        opt_type = "ATM"
                    elif strike < spot:
                        opt_type = "ITM"
                    else:
                        opt_type = "OTM"

                    pcr = round(put_oi / call_oi, 2) if call_oi > 0 else 0.0

                    parsed_rows.append({
                        "Strike": strike,
                        "Type": opt_type,
                        "Call OI": call_oi,
                        "Vega": call_greeks.get('vega', 0.0),
                        "Put OI": put_oi,
                        "IV (%)": call_greeks.get('iv', 0.0),
                        "Delta": call_greeks.get('delta', 0.0),
                        "Theta": call_greeks.get('theta', 0.0),
                        "PCR": pcr
                    })

                df = pd.DataFrame(parsed_rows)
                if not df.empty:
                    return df.sort_values(by="Strike"), "LIVE"
    except Exception:
        pass

    # Live API failed (e.g. weekend, market closed, token expired) -->
    # fall back to the disclosed simulated chain, never silently.
    return get_mock_option_chain(spot), "SIMULATED"


def calculate_max_pain(df_option_chain):
    try:
        if df_option_chain is None or df_option_chain.empty:
            return 24250.0
        df = df_option_chain.reset_index(drop=True)
        strikes = df['Strike'].values
        call_oi = df['Call OI'].values if 'Call OI' in df.columns else np.zeros(len(strikes))
        put_oi = df['Put OI'].values if 'Put OI' in df.columns else np.zeros(len(strikes))

        total_pain = {}
        for expiry_price in strikes:
            pain = 0
            for i, strike in enumerate(strikes):
                if expiry_price > strike:
                    pain += (expiry_price - strike) * call_oi[i]
                if expiry_price < strike:
                    pain += (strike - expiry_price) * put_oi[i]
            total_pain[expiry_price] = pain

        if not total_pain:
            return 24250.0
        return float(min(total_pain, key=total_pain.get))
    except Exception:
        return 24250.0


def get_fii_dii_fo_footprint(df_option_chain):
    try:
        if df_option_chain is None or df_option_chain.empty:
            return "NEUTRAL", 1.0
        df = df_option_chain.reset_index(drop=True)
        total_call_oi = df['Call OI'].sum() if 'Call OI' in df.columns else 1
        total_put_oi = df['Put OI'].sum() if 'Put OI' in df.columns else 0
        pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 1.0

        if pcr > 1.2:
            footprint = "BULLISH (FIIs Writing Puts / Accumulating Longs)"
        elif pcr < 0.8:
            footprint = "BEARISH (FIIs Writing Calls / Hedging Short)"
        else:
            footprint = "NEUTRAL / RANGE-BOUND (Balanced Institutional Positioning)"
        return footprint, round(pcr, 2)
    except Exception:
        return "NEUTRAL", 1.0
