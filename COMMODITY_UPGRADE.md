# Gold & Silver Upgrade — v27

Added to the v26 Multi-Stock Intelligence base:

- GOLD (XAU/USD) and SILVER (XAG/USD) asset profiles.
- Same core technical philosophy as equities: EMA/RSI/MACD, structural S/R, SMC/ICT, liquidity map, market regime and local multi-timeframe stack.
- Commodity-specific fundamentals instead of company PE/EPS: physical supply/demand, central-bank demand, investment/ETF flows, industrial demand, solar/electronics demand for silver, mine/recycling supply, Indian demand, real-yield/rate sensitivity, USD sensitivity, geopolitical risk and gold/silver ratio.
- Cross-market macro context: DXY, US 10Y yield proxy, crude oil, copper and S&P 500.
- Commodity-specific news feed with source/date/link fields.
- Conservative structural trade candidate generation only when real support/resistance exists; it is not an automatic order and must be revalidated by the AI decision layer.
- Commodity snapshot is handed to the AI chat context when the user analyzes Gold/Silver.
- Persistent Supabase migration: `SUPABASE_COMMODITY_MIGRATION.sql` for setups, learning profiles, fundamentals cache and news cache.
- Gold/Silver learning is isolated from NIFTY and individual-stock learning.

## Data integrity

No fake prices, fundamentals or news are inserted. If a source is unavailable, the corresponding field is marked unavailable. The default commodity price source is Yahoo Finance XAU/USD and XAG/USD spot. For Indian/MCX execution, an exchange-specific live feed/instrument mapping should be connected before treating the displayed spot data as an MCX contract quote.
