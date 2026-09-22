# NIFTY AI — Testing & Backtest Notes

## What was hardened
- Removed the market-data `bfill()` look-ahead path; causal `ffill()` is used instead.
- ML triple-barrier labels at the end of a dataset are treated as unknown, not as neutral labels.
- Walk-forward ML validation uses a holding-period embargo at the train/test boundary.
- Self-learning factor key is consistent: `mtf_stack_aligned`.
- `data_quality_pass` is now captured in the decision factor snapshot.
- Entry quality is stricter: validated level reaction must be >= 65%, major directional conflicts block, and entries too far from the structural level are rejected.
- ATR-only target fallback is disabled for high-quality AI setups; a structural/ladder target must provide the required R:R.
- Added `historical_backtest.py` for leakage-safe OHLCV/ML walk-forward validation.
- Added Streamlit CSV upload under the backtest section.

## Historical CSV format
Required columns:
- `Timestamp` (optional but recommended)
- `Open`
- `High`
- `Low`
- `Close`
- `Volume` (optional; missing volume is treated as 0 for the ML component)

## Important limitation
The historical ML test is **not** a full historical replay of the complete trading tool. A true end-to-end replay needs synchronized historical snapshots for option chain/OI/IV, futures order flow, breadth, global markets, news, SMC/liquidity state, etc. The project does not contain those historical snapshots, so they are not fabricated.

## Validation performed on this package
- Python compile check: PASS
- Regression tests: 8/8 PASS
- Synthetic historical walk-forward smoke test: PASS

## Entry-location correction (v20)
- BUY at support requires a confirmed support rejection and keeps the entry close to the support zone; it does not chase a bounce after price has already travelled away.
- BUY at resistance is blocked on a first touch/first breakout. It now requires a completed close above resistance followed by a retest-and-hold.
- SELL at resistance requires confirmed rejection.
- SELL at support requires a completed close below support followed by a retest-and-rejection.
- The trade-decision engine now forces the level side to match the current S/R location instead of allowing a bullish signal to select a resistance ladder level while price is actually at support (or the reverse).
- This is a loss-reduction filter, not a guarantee of profitability.
