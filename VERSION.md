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

## v17 -- OI bounce-or-break verdict + drought fixes (ai_trade_decision.py, trade_learning.py)
- NEW `_level_verdict`: for each real OI/strong S/R level the engine reads the full-data consensus for BOTH the bounce side and
  the break side, and reports state (FAR / APPROACHING / TESTING / BOUNCE_CONFIRMED / BREAK_CONFIRMED) + lean. Shown as
  `LEVEL VERDICT` lines in the Market View and stored in `entry_meta` for learning.
- NEW gate: a candidate (bounce, rejection, breakdown, breakout, probes included) is skipped when the data agrees with the
  OPPOSITE side by BREAK_LEAN_MARGIN (8 pct-pts) or more.
- Break trades only after a confirmed break (unchanged fake-break check) -- never before it.
- BUG FIX: confirmed break of an OI wall could never trade (oi_against veto fired on the very wall that was broken).
- BUG FIX: EMA20/VWAP/POC and weak 0.5-strength round numbers no longer count as a "wall" that trims the target below 1.5R.
- Mid-air continuation ("trend continuation, no fresh level") is blocked (CONT_REQUIRE_LEVEL).
- Drought easing enabled (DROUGHT_MAX_STEPS 0 -> 2) and now counted in MARKET minutes (no night/weekend false droughts).
- Bad-form penalty expires after FORM_STALE_HOURS (30h); probes stay on under bad form but only at strong OI walls.
- Tests: `python -m unittest tests_level_verdict` (7 tests). tests.py still has one older synthetic test
  (`test_fall_then_stall_produces_a_trade`, no OI/zones supplied) that also failed BEFORE these changes.
