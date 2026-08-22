import requests
import pandas as pd
import numpy as np
import streamlit as st

def get_nearest_expiry(access_token, instrument_key="NSE_INDEX|Nifty 50"):
    """
    Fetches the available expiry dates from Upstox and returns the nearest one.
    """
    url = "https://api.upstox.com/v2/option/chain/get-expiry-dates"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {access_token}"
    }
    params = {
        "instrument_key": instrument_key
    }
    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            res = response.json()
            expiry_dates = res.get('data', [])
            if expiry_dates:
                # Pehli (sabse kareeb wali) expiry date return kar do
                return expiry_dates[0].get('expiry_date')
    except Exception as e:
        print(f"Error fetching expiry dates: {e}")
    return None

def generate_option_chain_data(current_price=None):
    """
    Fetches real-time Option Chain data directly from Upstox API v2 using dynamic expiry date.
    """
    possible_keys = ['access_token', 'UPSTOX_ACCESS_TOKEN', 'token', 'upstox_token', 'auth_token']
    access_token = None
    
    for key in possible_keys:
        if key in st.session_state and st.session_state[key]:
            access_token = st.session_state[key]
            break
            
    if not access_token:
        st.warning("⚠️ Access Token missing in session state!")
        return pd.DataFrame()
        
    instrument_key = "NSE_INDEX|Nifty 50"
    
    # 1. Pehle nearest expiry date nikalo
    expiry_date = get_nearest_expiry(access_token, instrument_key)
    if not expiry_date:
        st.error("❌ Could not fetch expiry date from Upstox.")
        return pd.DataFrame()
        
    # 2. Ab option chain API ko hit karo expiry date ke sath
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
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            res_json = response.json()
            data_list = res_json.get('data', [])
            
            if not data_list:
                return pd.DataFrame()
                
            parsed_rows = []
            for item in data_list:
                strike = item.get('strike_price')
                
                call_options = item.get('call_options', {})
                call_market = call_options.get('market_data', {})
                call_greeks = call_options.get('option_greeks', {})
                call_oi = call_market.get('oi', 0)
                
                put_options = item.get('put_options', {})
                put_market = put_options.get('market_data', {})
                put_greeks = put_options.get('option_greeks', {})
                put_oi = put_market.get('oi', 0)
                
                pcr = round(put_oi / call_oi, 2) if call_oi > 0 else 0.0
                
                parsed_rows.append({
                    "Strike": strike,
                    "Call OI": call_oi,
                    "Put OI": put_oi,
                    "IV (%)": call_greeks.get('iv', 0.0),
                    "Delta": call_greeks.get('delta', 0.0),
                    "Theta": call_greeks.get('theta', 0.0),
                    "Vega": call_greeks.get('vega', 0.0),
                    "PCR": pcr
                })
                
            df = pd.DataFrame(parsed_rows)
            if not df.empty:
                return df.sort_values(by="Strike").set_index("Strike")
        else:
            st.error(f"❌ Upstox API Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        st.error(f"❌ Exception occurred: {e}")
        
    return pd.DataFrame()

def calculate_max_pain(df_option_chain):
    try:
        if df_option_chain is None or df_option_chain.empty:
            return 0.0
        df = df_option_chain.reset_index() if 'Strike' not in df_option_chain.columns else df_option_chain
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
            return 0.0
        return float(min(total_pain, key=total_pain.get))
    except Exception:
        return 0.0

def get_fii_dii_fo_footprint(df_option_chain):
    try:
        if df_option_chain is None or df_option_chain.empty:
            return "NEUTRAL", 1.0
        df = df_option_chain.reset_index() if 'Call OI' not in df.columns else df_option_chain
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
