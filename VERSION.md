# NIFTY AI -- v40: the engine learns from its mistakes (shadow trades + mistake memory + self-validated neural net)

Goal: stop the "many gates -> no trades -> no learning -> no trades" dead-lock WITHOUT taking more losing real trades, and let the
engine see "this looks like the setups that already lost".  Nothing about the real-trade path was loosened except the items listed.

- **Shadow trades (`shadow_trades.py`)**: every setup that was fully formed but BLOCKED (edge bar, segment pause, Gemini reject, local-fallback
  block) is followed virtually with the real SL / target / time-stop rules. No money, never shown as a trade. Resolved shadows feed the learners
  at half weight. Dashboard expander "Shadow trades" shows per gate: how many blocked setups would have won -> tells you if a gate is RIGHT
  (e.g. Gemini) or too strict. Persistent storage: run `SUPABASE_SHADOW_MIGRATION.sql` once (otherwise local file, lost on restart).
- **Mistake memory + tiny neural net (`learner_nn.py`, pure numpy)**: each new candidate is compared with past setups. If most of its closest
  neighbours lost, it loses probability (and gains a little if they won) -- shown as "Learned from past trades: ...". A one-hidden-layer
  network learns factor/entry-quality combinations, but it is trained on the older 70% and tested on the newest 30% and is only used if it
  beats a base-rate guess out-of-sample; with < 30 labelled trades it stays off. (TensorFlow/LSTM not used: too little data, huge install.)
- **Gate diet**: EV bar 0.08 -> 0.05; the "bad form raises the bar" rule removed (it froze the engine); dry-spell relief after 120 min (was 180).
  Unchanged on purpose: Gemini stays the final block; session window; same-signal rule; cooldown 10 min; entry-quality penalties (soft).
- Calibration now uses real (current logic) + shadow trades (weight 0.5); edge_model accepts per-row weights.
- Every logged trade stores `_features` (RSI, EMA distance, ATR, R:R, hour, VIX, PCR slope, strategy score) for the learners.
- Tests: 92 pass.

# NIFTY AI -- v39: why Gemini looked "unavailable", and a fallback that no longer chases

- **Cause of the false "Gemini nahi mila":** my v37.1 background-review change shortened the Gemini HTTP timeouts (40/25 s -> 25/15 s) and put a
  45 s deadline on the whole review. A normal, slow-but-working Gemini call (70k chars of context + Google Search) then timed out, every key "failed",
  and the result was reported as UNAVAILABLE -> the local fallback approved trades. Quota and keys were fine. Restored 40/25 s and a 150 s deadline
  (the review runs in a background thread, so the page never waits).
- **Fallback only for REAL quota exhaustion** (every attempt = 429). Timeout / 5xx / 404 -> no trade on that refresh, Gemini is asked again after 45 s.
- **New `entry_quality.py`** (price-action sense for the engine AND the fallback): flags chasing (RSI extreme, outside the Bollinger band, far from EMA20,
  entry right after an impulse candle, price already ran 3 ATR in 30 min, many same-colour candles) and credits real confirmations (liquidity sweep,
  rejection wick, pullback + resumption). In the main path it is a probability penalty (EV decides); in the LOCAL FALLBACK it is a strict block.
- **Local fallback is stricter:** >= 5 direct confirmations, EV bar +0.05, and it blocks a chase unless a sweep/rejection/pullback confirms it.
- **Targets are now realistic:** capped at 2.2R and at the distance price can plausibly travel inside the holding window (0.7 x ATR x sqrt(bars), never below 1.5R).
  (The last losing trade had a 3R / 84 pt target on a 28 pt stop = about a whole 2-hour range.)
- **New `pcr_velocity.py`:** slope of PCR over the last 20 min (rising = put writing = bullish, falling = bearish); learned factor + small penalty when against the trade.
- New learning factors: entry_not_extended, entry_confirmed_bounce, pcr_velocity_aligned.
- Tests: 77 pass (1 skipped without streamlit).

# NIFTY AI -- v38.2: page stuck on "loading" -- slow sources can no longer freeze it

- Symptom: page shows "Stop", sections greyed (stale) and nothing refreshes. Greyed = the run has not yet reached those sections, so
  the script was blocked EARLIER, in the data-fetch part (about 20 sources called one after another every 30 s).
- New `safe_io.py`: `guarded()` waits at most `timeout` s for a source; on timeout/error it returns the last good value (or an
  "unavailable" default) and keeps loading in the background, no duplicate jobs. Applied to: market breadth (heavyweights + Nifty 50),
  Nifty change %, Bank Nifty/Sensex.
- New step line at the top of the dashboard: while loading it shows "Loading: <step>", afterwards "Page Xs me load hua | slowest: ...".
  If the page ever hangs again, that line says which step it is stuck on.
- Found + fixed by tests: a deadlock in the first version of guarded() (lock re-entered by a done-callback of an already-finished job).
- Tests: 61 (1 skipped when streamlit is not installed).

# NIFTY AI -- v38.1: one setup, one trade

- After a trade hit its target the engine DID run a full fresh research pass for the next trade -- but a strategy signal stays "alive" for ~6 candles
  (30 min), so the SAME already-played signal could open another trade on the very next refresh (buying again at the top of the move that just paid).
  Reproduced and fixed: a new same-direction trade now needs a signal formed AFTER the previous same-direction trade was opened
  (`trade_learning.signal_already_traded`). Applies after a WIN and after a LOSS. Opposite direction is unaffected. Shown in the gate table as "Same setup already traded".
- Tests: 56 pass.

# NIFTY AI -- v38: target at the OI wall + fixes (no new entry gates)

- **BUG:** the OI-wall target never worked. The parser looked for lower-case `strike`/`oi` keys but the real chain has `Strike`/`Call OI`/`Put OI`,
  so it found no rows and every target silently fell back to ladder/ATR. Fixed (`_oi_walls`, `_row_oi`).
- **Target = just IN FRONT of the nearest significant OI wall** (call-OI wall above for BUY, put-OI wall below for SELL; significant = OI >= 60% of the
  strongest wall in range; 4 pts or 0.25 ATR before the wall, because price bounces AT the wall). Cap 3R. A strong wall closer than 1.2R does not block the
  trade -- it only lowers the win probability (soft). The reason is shown under the trade ("🎯 Target ... sits just before the XXXXX OI wall").
- **Calibration no longer punished by old logic:** the win-rate calibration now uses only trades made by the CURRENT logic version; before this, 30 trades
  from the old logic (23% wins) would have pulled every new trade's probability down and frozen the engine. Factor lifts still learn from all history (capped smaller: 0.6).
  Penalties (conflicts, wall ahead, choppy) now always count, even before there is a track record.
- **Time stop (60 min):** a trade that has not gone 0.35R in our favour after 60 min is closed (EXPIRED, marked-to-market, still teaches the learner). Before, a dead trade
  blocked every new setup for up to 2 hours (one open trade at a time). Best/worst excursion now uses candle highs/lows.
- Fixed: "bad form" staleness compared IST trade times with the UTC server clock.
- Stop-loss unchanged (20-28 pts; structure-aware widening is capped at 28).
- `python -m unittest tests_edge_v36` -> 50 pass.

# NIFTY AI -- v37.4: fallback restored + every trade records WHO approved it

- `GEMINI_FAIL_OPEN = True` again (same as the version that traded well): explicit Gemini REJECT = no trade; Gemini technically
  unreachable -> the strict local fallback review decides (>=4 direct confirmations, conflicts limit, R:R >= 1.5, positive expectancy).
- Every trade stores `_gemini` = {reviewer: GEMINI | LOCAL_FALLBACK, status, model, reason, gemini_error, at}.
  Recent AI Setups has a Gemini column: "Gemini ne approve kiya" / "Gemini nahi mila -> local fallback" / "record nahi (purana trade)".

# NIFTY AI -- v37.3: Gemini report is stored with every trade

- Each approved trade now stores the Gemini review that approved it (status, model, search used, confidence, reason, time) in its
  `factors_json["_gemini"]`. "Recent AI Setups" has a new **Gemini** column, plus an expander with the full Gemini reason per trade.
- Trades logged before this version do NOT have it ("not recorded (old trade)") -- the old code never saved the verdict.

# NIFTY AI -- v37.2: no Gemini approval = no trade, ever

- `GEMINI_FAIL_OPEN = False`: previously, if Gemini was technically unreachable (quota/5xx), a local fallback review could approve a
  trade WITHOUT Gemini. Now an unavailable Gemini also means no trade.

# NIFTY AI -- v37.1: page no longer hangs on Gemini

- **Cause found:** the Gemini final review ran INSIDE the 30-second dashboard script. On a slow/overloaded Gemini it could try up to 30
  key/model combinations at ~65 s each, so the page sat on "Stop" and nothing loaded.
- **Fix:** the review now runs in a background thread (`ai_trade_decision.gemini_review_async`). The page renders at once; while the
  verdict is pending there is no trade (Gemini stays the final block) and the result is picked up on the next 30 s refresh.
  The whole review also has a hard 45 s deadline (`REVIEW_DEADLINE_S`, `gemini_pool.run(deadline_s=...)`) and shorter HTTP timeouts (25 s / 15 s).
- Files changed: `gemini_pool.py`, `ai_trade_decision.py`, `app.py` (+ tests). `python -m unittest tests_edge_v36` -> 33 pass.

# NIFTY AI -- v37 (final): Gemini stays the final block + fewer, lighter gates (on top of v36)

- **Gemini is the FINAL BLOCK again.** Anything other than an approval = no trade (no probe, no reduced size). Its reports were right.
- Kept ONLY the data bug fix: Gemini used to receive the OLDEST candles (first 18000 chars of a 500-candle oldest-first table).
  It now gets the latest 90 candles, labelled "LAST row = newest". Its judging rules were not changed.
- Lighter gates (v36 had added too many): same-direction cooldown 20 -> 10 min; the 3-loss pause is OFF by default
  (`trade_learning.LOSS_STREAK_PAUSE = None`). Quality is now carried by the learned expectancy bar, not by extra gates.
- Structure-aware stop-loss (a parameter, not a gate): if the validated S/R zone's far edge is beyond the base 20-28 pt stop,
  the stop goes just past that edge (+0.25 ATR), capped at `MAX_SL_PTS` = 32. Fewer "stopped by the normal wick" losses.
- New expander "Kaun sa gate kitni baar roka": counts, per 5-min candle, which gate stopped a setup, so tuning is done on evidence.
- Tests: `python -m unittest tests_edge_v36 -v` -> 29 pass.

# NIFTY AI Multi-Stock Intelligence — v36 CALIBRATED EDGE

## v36 -- accuracy / quality upgrade of the AI Trade Decision Engine
**Honest scope:** none of this can promise a win rate. The project has no historical option-chain/OI snapshots, so the
full engine cannot be back-tested. v36 therefore (1) fixes real bugs that were corrupting results and learning,
(2) replaces the arbitrary confidence number with a calibrated, self-correcting one, and (3) adds protections that
cut repeated/avoidable losses. Measure it on your own resolved trades.

### New: `edge_model.py`
- Win probability = conservative rule prior -> per-factor learned log-odds lift (shrunk, clamped) -> reliability
  correction toward the engine's REAL win rate once >= 15 resolved trades exist.
- Expectancy in R: `EV = p*RR - (1-p)`. The engine trades only if EV >= `MIN_EV_R` (0.08). A farther target is NOT
  treated as bigger edge (probability is scaled by reward:risk).
- With no track record the model never claims more than 50%.

### Bugs fixed
- SQLite fallback: every refresh overwrote `factors_json` with only the excursion meta -> all stored factor flags were wiped.
- Trades were resolved only against the 30-second snapshot price: SL/target touches between refreshes were missed.
  Now candle highs/lows since entry are scanned; a candle touching both SL and target counts as LOSS (pessimistic).
- Timestamps/expiry used server time (UTC on Streamlit Cloud). Now IST. Trades are squared off at 15:20 IST and a trade
  that survives overnight is closed at the last price seen in its own session, never a gap-open price.
- EXPIRED trades were silently dropped from win rate and learning. They now teach the learner by mark-to-market R
  (>= +0.5R win, <= -0.5R loss).
- `segment_gate`, `recent_performance` (self-learning "block proven losers") were defined but never called, and no trade
  stored the `_entry_meta` they need. Now wired in.
- `global_research_aligned`, `nifty50_news_aligned`, `nifty50_fundamentals_aligned` were learning factors that were never
  computed. Now computed. Strong opposite global research now vetoes (was documented in v29, not implemented).
- `local_fallback_review` rule used the old confidence number; now uses expectancy.

### New protections / frequency controls (all constants at the top of `ai_trade_decision.py`)
- Entry window 09:25-14:45 IST (`NO_ENTRY_BEFORE/AFTER`).
- 20 min cooldown after a stop-out in the same direction; 45 min pause after 3 straight losses (`trade_learning.entry_cooldown`).
- Target capped at `MAX_TARGET_R` (2.0R) -- a 140 pt OI wall almost never fills inside the 2h holding window.
- Dry-spell relief: after ~3h with no setup the EV bar eases by 0.06R (never below break-even) so the engine cannot freeze itself.
- Choppy volatility, major conflicts and S/R caveats now reduce probability instead of being ignored.

### Tests
- `python -m unittest tests_edge_v36 -v` : 22 tests, all pass.
- `tests.py` (3 failures/errors) and `tests_level_verdict.py` (7) were ALREADY failing before v36: they target a v17 engine
  (`_level_verdict`, `_ist_now`, `df=` argument, BOUNCE_BUY playbooks) that is not in this package. Not changed.

---

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
