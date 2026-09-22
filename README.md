
## Multi-Stock AI Intelligence (v26 upgrade)

This package adds a common NSE equity analysis pipeline while preserving the existing NIFTY dashboard and `ai_trade_setups` table.

### Stock analysis
- Select an NSE symbol from the Multi-Stock AI Intelligence panel.
- Upstox Live OHLCV is used when an authenticated Upstox session can resolve the NSE instrument.
- Yahoo Finance is an explicitly labelled fallback when Upstox data is unavailable.
- Technical analysis includes EMA20/EMA50, ATR, RSI, momentum, nearby support/resistance and trend state.
- Fundamental snapshot uses Yahoo Finance company fundamentals when available; missing values are not fabricated.
- Company news uses live RSS headlines and displays the publication/source data returned by the feed.
- Sector and NIFTY context are included.
- Equity option-chain data is not assumed; NIFTY option analytics remain in the original NIFTY engine.

### Supabase preservation
Your existing `SUPABASE_URL` and `SUPABASE_KEY` do not change. Existing `ai_trade_setups` is not dropped or recreated.

The file `SUPABASE_MULTISTOCK_MIGRATION.sql` contains only additive `CREATE TABLE IF NOT EXISTS` statements for stock-specific history/cache. Run it once in the same Supabase project if you want persistent stock-specific setup/history/cache. The old NIFTY learning data remains untouched.

### No-fake-data rule
Every live/fallback source is labelled. If a fundamental/news field cannot be obtained, it is reported as unavailable rather than filled with a guessed value.


## NIFTY 50 context upgrade
The Multi-Stock panel now includes a dedicated NIFTY 50 fundamental context and dedicated NIFTY 50 news feed. Fundamental aggregates are explicitly labelled as constituent-level aggregates; when the official Nifty Indices constituent CSV is reachable, the current constituent list is fetched dynamically, otherwise the bundled fallback list is used. The aggregate is not presented as an official index-weighted fundamental metric. News remains source/date-labelled and missing data is shown as unavailable.

Run `SUPABASE_MULTISTOCK_MIGRATION.sql` once if you want the NIFTY 50 fundamental/news snapshots persisted in the existing Supabase project. It is additive and does not drop or recreate existing tables.

## Gold & Silver Commodity Intelligence

The dashboard now supports `GOLD` (XAU/USD) and `SILVER` (XAG/USD) as isolated commodity profiles. They use the same technical/risk/learning philosophy as the equity engine, but commodity fundamentals are different from company fundamentals.

Tracked commodity drivers include supply/demand, central-bank and investment flows, industrial demand (especially for silver), mine/recycling supply, DXY, US yields/rates, inflation, geopolitics and the gold/silver ratio. News is sourced through RSS and unavailable fields remain unavailable rather than being fabricated.

Run `SUPABASE_COMMODITY_MIGRATION.sql` once to add persistent commodity setup/history/cache tables. Existing NIFTY and stock tables are not dropped or recreated.

### Free XAU/USD + XAG/USD live spot feed
The commodity engine uses the free keyless XAUS spot/intraday API as the primary global-spot source for XAU/USD and XAG/USD. The feed is an indicative mid-market reference, not an executable broker quote. The engine records source/freshness metadata and falls back to Yahoo Finance only when the primary source is unavailable. TradingView is not used as a backend market-data API because TradingView states that it does not provide a general market-data API. MCX remains a separate optional reference and is never silently mixed into XAU/USD or XAG/USD.


## v29 Global Research + Custom Stock Selection
The NIFTY decision engine now receives a structured global research layer built from the live/fallback global market table and India/global news sentiment. It considers broad global equity breadth, India VIX, DXY, US 10Y yield, WTI, USD/INR and Gold as contextual evidence. It never lets a single macro variable override validated NIFTY structure, options, breadth, order flow or risk gates. Strong opposite global research can veto a setup.

The Multi-Stock panel supports typing an NSE symbol directly and adding it to a custom watchlist, instead of being limited to the bundled universe.
