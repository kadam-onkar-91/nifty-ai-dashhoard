# NIFTY AI Multi-Stock Intelligence — v29 GLOBAL RESEARCH + CUSTOM STOCK SELECTION

## Final additions
- XAU/USD (Gold) and XAG/USD (Silver) primary global spot mode.
- Free keyless XAUS spot feed with freshness metadata.
- Free XAUS recorded intraday series for commodity technical analysis.
- Yahoo Finance remains a secondary fallback when the primary feed is unavailable.
- No fake prices; unavailable data remains unavailable.
- XAU/XAG spot is never silently replaced by MCX futures.
- MCX contract metadata remains a separate optional reference when Upstox access is available.
- Commodity fundamentals/news/macro/SMC/liquidity/MTF/risk/learning remain isolated by asset.
- TradingView is not treated as a backend market-data API.

## Validation
- Python compilation: PASS
- Regression tests: 12/12 PASS
- Multi-stock universe smoke tests: PASS

## v29 upgrades
- Structured global-to-NIFTY research layer consumes the existing global market/news tables.
- Global research is mandatory in the NIFTY master context before a trade decision is accepted.
- Strong opposite global research can veto a setup; global context cannot blindly flip a domestic signal.
- New `global_research_aligned` learning factor is persisted with trade snapshots.
- Custom NSE symbol search + add/remove watchlist in the Multi-Stock UI.
- NIFTY domestic structure, options, breadth, order flow, S/R and risk remain primary evidence.
