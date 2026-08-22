import pandas as pd
import yfinance as yf

def get_global_market_indices():
    """
    Fetches real-time prices and percentage changes for global stock markets 
    and macro indicators with official country flags/logos.
    """
    indices_data = [
        {"name": "S&P 500 (US)", "ticker": "^GSPC", "code": "us"},
        {"name": "Nasdaq Composite (US)", "ticker": "^IXIC", "code": "us"},
        {"name": "Dow Jones (US)", "ticker": "^DJI", "code": "us"},
        {"name": "Nikkei 225 (Japan)", "ticker": "^N225", "code": "jp"},
        {"name": "Shanghai Composite (China)", "ticker": "000001.SS", "code": "cn"},
        {"name": "Hang Seng (Hong Kong)", "ticker": "^HSI", "code": "hk"},
        {"name": "KOSPI (South Korea)", "ticker": "^KS11", "code": "kr"},
        {"name": "FTSE 100 (UK)", "ticker": "^FTSE", "code": "gb"},
        {"name": "DAX (Germany)", "ticker": "^GDAXI", "code": "de"},
        {"name": "CAC 40 (France)", "ticker": "^FCHI", "code": "fr"},
        {"name": "ASX 200 (Australia)", "ticker": "^AXJO", "code": "au"},
        {"name": "Straits Times (Singapore)", "ticker": "^STI", "code": "sg"},
        {"name": "India VIX", "ticker": "^INDIAVIX", "code": "in"},
        {"name": "Crude Oil (WTI)", "ticker": "CL=F", "code": "oil"},
        {"name": "USD/INR", "ticker": "USDINR=X", "code": "in"}
    ]
    
    data = []
    
    for item in indices_data:
        name = item["name"]
        ticker = item["ticker"]
        code = item["code"]
        
        if code == "oil":
            flag_url = "https://img.icons8.com/color/48/oil-industry.png"
        else:
            flag_url = f"https://flagcdn.com/w40/{code}.png"
            
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="2d")
            if not hist.empty and len(hist) >= 1:
                current_price = hist['Close'].iloc[-1]
                prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                change_pct = ((current_price - prev_close) / prev_close) * 100
                
                status = "Bullish 🟢" if change_pct >= 0 else "Bearish 🔴"
                
                data.append({
                    "Logo": flag_url,
                    "Global Market / Asset": name,
                    "Latest Price": round(current_price, 2),
                    "Change (%)": round(change_pct, 2),
                    "Status": status
                })
            else:
                data.append({
                    "Logo": flag_url,
                    "Global Market / Asset": name,
                    "Latest Price": 0.0,
                    "Change (%)": 0.0,
                    "Status": "Neutral 🟡"
                })
        except Exception:
            data.append({
                "Logo": flag_url,
                "Global Market / Asset": name,
                "Latest Price": 0.0,
                "Change (%)": 0.0,
                "Status": "Neutral 🟡"
            })
            
    df_markets = pd.DataFrame(data)
    return df_markets

def get_global_market_summary(df_markets):
    """
    Calculates an automatic Global Market Sentiment Score based on live data.
    """
    if df_markets.empty:
        return "Neutral / Mixed 🟡", 0.0
    
    bullish_count = len(df_markets[df_markets['Status'].str.contains('Bullish')])
    bearish_count = len(df_markets[df_markets['Status'].str.contains('Bearish')])
    avg_change = df_markets['Change (%)'].mean()
    
    if bullish_count >= bearish_count + 4:
        score = "🚀 Strong Bullish"
    elif bullish_count > bearish_count:
        score = "🟢 Mild Bullish"
    elif bearish_count >= bullish_count + 4:
        score = "🚨 Strong Bearish"
    elif bearish_count > bullish_count:
        score = "🔴 Mild Bearish"
    else:
        score = "🟡 Neutral / Sideways"
        
    return score, round(avg_change, 2)
