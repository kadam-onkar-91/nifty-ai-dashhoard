import requests
import pandas as pd
import numpy as np
import streamlit as st

def fetch_live_option_chain(instrument_key="NSE_INDEX|Nifty 50"):
    """
    Fetches real-time Option Chain data directly from Upstox API v2
    using the active access token from Streamlit session state.
    """
    # Session state se active access token fetch karna
    access_token = st.session_state.get('access_token') or st.session_state.get('UPSTOX_ACCESS_TOKEN')
    
    if not access_token:
        # Agar token nahi mila to warning dikhayega
        return pd.DataFrame()
        
    url = "https://api.upstox.com/v2/option/chain"
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
            res_json = response.json()
            data_list = res_json.get('data', [])
            
            if not data_list:
                return pd.DataFrame()
                
            parsed_rows = []
            for item in data_list:
                strike = item.get('strike_price')
                
                # Call side data
                call_options = item.get('call_options', {})
                call_market = call_options.get('market_data', {})
                call_greeks = call_options.get('option_greeks', {})
                call_oi = call_market.get('oi', 0)
                
                # Put side data
                put_options = item.get('put_options', {})
                put_market = put_options.get('market_data', {})
                put_greeks = put_options.get('option_greeks', {})
                put_oi = put_market.get('oi', 0)
                
                # PCR calculation
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
            print(f"Upstox API Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"Error fetching option chain: {e}")
        
    return pd.DataFrame()

def calculate_max_pain(df_option_chain):
    """
    Calculates the Max Pain strike price where option buyers experience maximum loss.
    """
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
            
        max_pain_strike = min(total_pain, key=total_pain.get)
        return float(max_pain_strike)
    except Exception:
        return 0.0

def get_fii_dii_fo_footprint(df_option_chain):
    """
    Estimates institutional FII/DII bias based on PCR and OI buildup.
    """
    try:
        if df_option_chain is None or df_option_chain.empty:
            return "NEUTRAL", 1.0
            
        df = df_option_chain.reset_index() if 'Call OI' not in df_option_chain.columns else df_option_chain
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
