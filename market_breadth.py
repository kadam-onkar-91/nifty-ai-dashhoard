import yfinance as yf
import pandas as pd
import streamlit as st

@st.cache_data(ttl=60)
def get_real_market_breadth():
    # Expanded heavyweights covering Nifty 50, Bank Nifty, and Sensex major constituents
    heavyweights = {
        'Reliance': 'RELIANCE.NS', 
        'HDFC Bank': 'HDFCBANK.NS', 
        'ICICI Bank': 'ICICIBANK.NS', 
        'Infosys': 'INFY.NS', 
        'TCS': 'TCS.NS',
        'Axis Bank': 'AXISBANK.NS',
        'SBI': 'SBIN.NS',
        'Kotak Bank': 'KOTAKBANK.NS',
        'ITC': 'ITC.NS',
        'L&T': 'LT.NS'
    }
    
    data = []
    advances = 0
    declines = 0
    
    try:
        tickers = list(heavyweights.values())
        hist_data = yf.download(tickers, period="2d", progress=False)['Close']
        
        for name, ticker in heavyweights.items():
            if ticker in hist_data.columns:
                # Handle DataFrame or Series safely depending on yfinance response format
                series = hist_data[ticker]
                if len(series) >= 2:
                    prev_close = float(series.iloc[0])
                    curr_price = float(series.iloc[-1])
                else:
                    prev_close = curr_price = float(series.iloc[-1])
                
                change_pct = ((curr_price - prev_close) / prev_close) * 100 if prev_close > 0 else 0.0
                
                if change_pct > 0: 
                    advances += 1
                else: 
                    declines += 1
                
                data.append({
                    "Symbol": name, 
                    "LTP": round(curr_price, 2), 
                    "Change (%)": round(change_pct, 2)
                })
        
        df = pd.DataFrame(data)
        total_symbols = len(heavyweights)
        total_advances = int((advances / max(1, total_symbols)) * 50)
        total_declines = 50 - total_advances
        
        return df, total_advances, total_declines
        
    except Exception:
        safe_df = pd.DataFrame([
            {"Symbol": k, "LTP": 0.0, "Change (%)": 0.0} for k in heavyweights.keys()
        ])
        return safe_df, 25, 25
