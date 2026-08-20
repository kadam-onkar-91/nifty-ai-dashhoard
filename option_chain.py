import pandas as pd
import numpy as np

def generate_option_chain_data(current_price):
    if not current_price:
        return pd.DataFrame()
    
    atm_strike = round(current_price / 50) * 50
    strikes = [atm_strike + (i * 50) for i in range(-5, 6)]
    
    data = []
    for strike in strikes:
        dist = abs(atm_strike - strike) / 50
        
        call_oi = int(120000 / (dist + 1.2)) if strike >= atm_strike else int(60000 / (dist + 1.5))
        put_oi = int(120000 / (dist + 1.2)) if strike <= atm_strike else int(60000 / (dist + 1.5))
        
        iv = round(14.5 + dist * 0.4, 2)
        
        if strike == atm_strike: delta = 0.50
        elif strike > atm_strike: delta = round(max(0.05, 0.50 - (dist * 0.09)), 2)
        else: delta = round(min(0.95, 0.50 + (dist * 0.09)), 2)
            
        theta = round(-12.5 + dist, 2)
        vega = round(10.5 - (dist * 0.5), 2)
        pcr = round(put_oi / call_oi, 2) if call_oi > 0 else 0
        
        tag = "ATM" if strike == atm_strike else ("ITM" if strike < atm_strike else "OTM")
        
        data.append({
            "Strike": strike,
            "Type": tag,
            "Call OI": call_oi,
            "Put OI": put_oi,
            "IV (%)": iv,
            "Delta": delta,
            "Theta": theta,
            "Vega": vega,
            "PCR": pcr
        })
        
    return pd.DataFrame(data).set_index("Strike")

def calculate_max_pain(df_option_chain):
    """
    Calculates the Max Pain strike price where option buyers experience maximum loss.
    """
    try:
        if df_option_chain.empty:
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
            
        max_pain_strike = min(total_pain, key=total_pain.get)
        return float(max_pain_strike)
    except Exception:
        return 0.0

def get_fii_dii_fo_footprint(df_option_chain):
    """
    Estimates institutional FII/DII bias based on PCR and OI buildup.
    """
    try:
        if df_option_chain.empty:
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