## v55b -- gate judged by its own record (evidence relief) + Gemini verdict in the same refresh
- 30-Oct evidence from the dashboard: setups refused by the calibrated-edge gate (shadow rows labelled "No positive edge (EV)" + the pre-v54 mislabelled "S/R location / confirmation", ~90 resolved) won ~48 % at R:R ~2, while the trades the gate let through won 28 %. The gate was keeping out winners.
- `EVIDENCE_RELIEF`: for strategy-led entries the EV bar and win-floor stand aside WHILE shadow_trades.edge_gate_record() (n>=30) has a 90 % Wilson lower bound above break-even + 2 pts. It switches itself off if >=8 real trades taken under it (flag `evidence_relief`) win less than break-even-5 pts. All other gates (knife, cooldown, confluence, entry distance, SL<=28, Gemini) unchanged. Kill switch `ai_trade_decision.EVIDENCE_RELIEF=False`.
- Gemini review: the verdict is awaited up to `REVIEW_WAIT_S` (14 s) inside the same refresh instead of always arriving one 30 s refresh (often a full candle) late ("Gemini review pending" blocked 29 candles on 30-Oct). Set `REVIEW_WAIT_S = 0` for the old behaviour. Gemini's veto itself is kept (its rejects won 38 % vs approvals 34 %: no edge either way yet).
- Not fixable in code: the engine saw no candles before 10:00 on 30-Oct (two of the four 50+ pt moves were before that) -- keep the keeper/app running from 09:15 (KEEPER_SETUP.md).

## v55 -- live-dominant entries judged by their own record + edge-blocked setups shadow-tracked
- The "No positive edge" gate was charging every candidate (even fully aligned live-dominant BUYs, e.g. 29-Oct +147 pts) with the engine's old 25-29 % late-chase record.
  An aligned, not-extended, live-dominant trend entry is now its own entry type (`live_dom_entry`): calibrated on its OWN real+shadow rows; with fewer than 6 it has a neutral prior (<=50 %) and learned drags count 25 %. The EV bar, SL/target rules, entry-distance gate and Gemini review are unchanged. Once it has its own losing record it is stopped automatically.
- shadow_trades: gap / open / daily caps are now per blocking gate (daily 30, 10 per gate) so the edge/EV gate is finally tracked and learns whether it blocks winners.
- Kill switch: `ai_trade_decision.LIVE_DOM_ENTRIES = False` restores v54. LOGIC_VERSION not bumped.
- gate_log.missed_moves + dashboard timeline: every >=70 pt move that was NOT traded is listed with the gates that held the matching direction back (tests_v55_missed_moves.py).
- tests_v55_live_dom.py (5 tests); v42/v54 test histories now tag the lane's own record.

# v54b -- 28-Oct: V-reversal BUY refused by lagging conflicts  [screenshots 09:48-09:55: 5 live BUY strategies, consensus 88 %, live +52, still "7 direct confirmations; 2 major conflicts"]
- The confluence gate vetoed any entry with 2 of 3 (regime / MTF / order flow) against it.  Those are LAGGING reads: right after a gap-down capitulation (22,740 -> 22,550) and the V-bounce they still say bearish.
- `CONFLICT_RELIEF`: when the live tape + broad strategy agreement dominate (momentum-lane test: live >= 45, >= 3 live strategies, consensus >= 80 %, Model B not STRONG_OPPOSED) or the entry is a confirmed turning-point
  reaction, 2 conflicts are only a probability cost (0.20 each, already charged) -- all 3 still block.  Reaction entries are also exempt from the "opposite transition" flag.  Edge bar / floor / Gemini / SL unchanged.
- The single 100-pt opening candle (09:15-09:20) can not be traded by any rule (entries start 09:25) -- the catchable part is the capitulation low + bounce afterwards.
- Tests: 3 new in tests_v54_reaction (20 total) + all earlier suites pass.  NOT measured on live data.

# v54 -- TURNING-POINT (REACTION) ENTRIES: resistance-rejection SELL / support-bounce BUY were structurally impossible  [27-Oct: "itna bada move aaya, support/resistance se reverse, ek bhi trade nahi"]
What the 27-Oct screenshots + a replay of that tape through the code showed (Nifty 22,820 spike top -> 22,681 -> 22,672; engine took only one BUY):
- The strategy layer DID produce SELL signals during the fall; the refusals came later ("win chance 14.3 %", "entry too far from the validated level").
- ROOT CAUSE 1 (strategy_engine._signal_validity): every signal whose direction is against the live tape (EMA/VWAP/momentum) is blocked as "stale / against the move".  A reversal signal
  (upthrust at resistance, spring at support) fires exactly while the live tape is STILL with the old move, so every reversal was blocked the moment it formed.  The engine could only
  enter after the new move was established = late, extended = the "immediate reversal" losses (47 cases, 19 % wins on the dashboard).
- ROOT CAUSE 2 (ai_trade_decision): "AI research must point the same way" -- the AI vote (order flow / regime / MTF / breadth) lags at a turn by definition -- and the knife gate.
- ROOT CAUSE 3: the engine's overall SELL record (taken mostly into bounces/bottoms) was charged to every new SELL, including a clean rejection at a validated level.
- Also found: a continuation BUY at a 6-ATR spike top that has just printed a rejection candle was NOT blocked (the exhaustion guard only looked 0.7 ATR from the extreme).
Changes (each has a kill switch; SL <= 28 pts, target / OI-wall logic, cooldown, same-zone guard, Gemini final review, no-trailing rule UNCHANGED):
1. `strategy_engine.reaction_confirmed` + `REVERSAL_STRATEGIES` (spring/upthrust, VWAP+S/R rejection, pivot reversal, RSI divergence, liquidity sweep+FVG, PDH/PDL sweep, EMA trap, double bottom,
   H&S, inverse H&S): such a signal passes the "against the live tape" block when the turn is CONFIRMED -- price within 1.6 ATR of the 24-bar high/low after a >= 1.5 ATR move AND a rejection wick
   with the turn.  Continuation strategies stay blocked exactly as before.  Kill switch `REACTION_ENTRIES`.
2. `_exhaustion_reversal` BLOW-OFF branch: after a >= 3 ATR move, the extreme bar + rejection + close back through the middle of the previous (spike) candle is a spring/upthrust even with no old
   swing to sweep (a vertical spike has none).  Kill switch `EXH_BLOWOFF`.
3. Exhaustion guard tier: after a >= 3.5 ATR move the "near the extreme" distance is 1.6 ATR (was 0.7) -> no continuation BUY under a freshly rejected spike top / SELL over a rejected capitulation low.
4. `ai_trade_decision`: a confirmed reaction at the MATCHING zone (SELL at AT/NEAR_RESISTANCE, BUY at AT/NEAR_SUPPORT) is exempt from the "AI vote must agree" lock (no rebase to the lagging AI side) and
   from the knife gate.  Wrong zone (selling INTO support) = still blocked.  Kill switch `REACTION_ENTRIES`, `REACTION_NEEDS_LOCATION`.
5. `trade_learning.compute_edge(entry_type="reaction")`: a reaction is calibrated on REACTION trades only; until 6 of them have resolved it gets the neutral (capped 50 %) prior, and the learned penalties
   (mistake memory / NN / tags / weak segment) count 25 % (`REACTION_LEARNED_DRAG_FACTOR`).  The EV bar, win-chance floor and Gemini still decide, and once reaction trades exist their OWN record rules
   (tests: a reaction type with a losing record is stopped).  Ordinary / chasing entries keep the full record -- they do NOT get easier.  Kill switch `trade_learning.ENTRY_TYPE_AWARE`.
6. Gemini prompt rule 2e: for reaction_entry=true the old trend on the live tape is expected and not "chasing"; it must still SEE the rejection at that level, otherwise REJECT.
7. TRANSPARENCY: every "No positive edge" refusal now lists where the probability went ("Probability drags: global/macro -0.15, entry quality +0.30, repeat-mistake tags +0.20 ...") and how many
   comparable trades it was calibrated on -> the next review does not need guessing.  (+ = hurts)
8. BUG FIX: the gate diagnostics counted every "No positive edge" refusal as "S/R location / confirmation" (the text "break-even needs ..." matched the wrong pattern).
- NOT changed on purpose: late continuation entries (a SELL 6 ATR into a fall) are still judged by the full record -- the dashboard's own post-mortem says chases lose; the aim is to be there at the TURN or the RETEST, not to chase.
- Tests: tests_v54_reaction (17) + all earlier suites pass (tests_level_verdict was already failing before).  Effect on win rate is NOT measured on live data: reaction trades are a new, unproven trade type --
  they will be small in number, are logged with lane="reaction", and are judged separately from the first resolved trade.

# v53 -- UNUSED DASHBOARD DATA NOW COUNTS: sector breadth, volume POC, FII option footprint, IV skew, IV percentile, drift report, Monte-Carlo risk  [27-Oct: "factors ka data use me lao"]
Audit: of the 46 dashboard sections only the RAW ones were read by the engine.  Sector breadth, POC, FII footprint, IV skew/percentile, drift and Monte-Carlo were computed and SHOWN,
but never changed a decision (some only reached Gemini as text).
- NEW `data_confluence.py` (pure, kill switch `ENABLED`): each source -> one signed vote for/against the proposed direction; net of weighted votes gives a CAPPED nudge
  (penalty up to +0.18 logit, bonus up to -0.06) -- it can never create a trade or touch SL/target.  Missing data = no vote.  Engine drift adds +0.06 (be pickier when the recent record is worse than baseline).
- The votes are stored as factor flags (`sector_breadth_aligned`, `poc_aligned`, `fii_footprint_aligned`, `iv_skew_aligned`, `model_not_drifting`, `risk_of_ruin_low`, `iv_not_extreme`)
  and have their own group in the Trade Post-Mortem ("Extra data") -> the engine measures whether each one REALLY helps on your trades.  Misleading one? set `data_confluence.VOTES_ON[...] = False`.
- No new dashboard text: the screen looks exactly as before (the data effect works silently).
- NOT added to ALL_FACTOR_KEYS on purpose (would change the trained edge model / NN inputs); they are learned through the post-mortem scoreboard first.
- Tests: tests_v53_data_confluence (7) + all earlier suites pass.  Effect on win rate NOT measured on live data.

# v52b -- SUPPORT-SPRING BOUNCE + NO SELLING THE BOTTOM + LOCATION-MISTAKE LEARNING AMPLIFIER  [27-Oct: "support se bounce, fall ke baad bhi trade nahi, aur neeche SELL ban gaya"]
1. NEW strategy `spring_reclaim_reversal` (strategy_engine, #41): after a real fall (>= 2 ATR off the 24-bar high) price sweeps/tests the prior swing low, the last completed candle
   rejects it (lower wick / bullish engulf) and closes back above the level -> BUY.  Mirror at resistance after a rally -> SELL (upthrust).  Pure price action, works from day 1;
   walk-forward / shadow / post-mortem then rate it like any other strategy.  Kill switch `EXHAUSTION_REVERSAL`.
2. NEW EXHAUSTION GUARD (`_signal_validity`): SELL blocked when price already fell >= 2.5 ATR, sits within 0.7 ATR of the low AND the last candles show a rejection wick (BUY mirrored at the top).
   A clean breakdown with no rejection still passes.  Kill switch `EXHAUSTION_GUARD`.
3. LEARNING AMPLIFIER (trade_postmortem + market_brain): each LOSS is checked for "SOLD_THE_BOTTOM / BOUGHT_THE_TOP" (candles before entry).  2+ such losses in the last 12 decided trades
   make the guard stricter (2.5 -> 2.15 -> 1.8 ATR, never lower); the tag appears in the Trade Post-Mortem failure-mode table.  So a repeated mistake is punished harder next time, not just recorded.
- SL / target / cooldown / same-zone guard / Gemini review UNCHANGED.  Tests: tests_v52_spring (7) + tests_v52_location_learning (3) + tests_v52_reclaim (5) + all older suites pass.
- NOT measured on live/backtest data.  Strategy count is now 41 (UI text still says 40 in places).

# v52 -- V-RECOVERY DEADLOCK FIXED: engine no longer sits out a dip-and-reclaim move  [from the 27-Oct screenshots]
27-Oct: Nifty 22,810 -> 22,650 (support/EMA zone) -> 22,767.  Named Strategy Library showed Opening Drive BUY 44.8 (min 38) but "BLOCKED - BUY: price already dropped 1.37 ATR
off the high", while the SELL strategies were blocked as "stale/against the move" -> NO side was tradable for the whole recovery.  Root cause: `strategy_engine._signal_validity`
measured the drop against the 12-bar high even after price had already bounced back above EMA9/EMA20 with live tape UP (+34 < the old 40 bar).
- NEW `RECLAIM_EXCEPTION` (strategy_engine): a BUY passes the retrace block when price is above EMA9 + EMA20 + VWAP, 3-bar momentum is up, live tape >= +15 with the signal,
  and it is already >= 0.7 ATR off the 12-bar low (SELL mirrored).  Signals against the live tape (<= -30) are STILL blocked; chase penalty unchanged.  Kill switch: RECLAIM_EXCEPTION = False.
- Shadow evidence reviewed (27-Oct): Gemini rejected 22 -> 36 % win (block was right), Self-learning paused 12 -> 33 % (right), S/R location 66 -> 48.5 % (gate too broad; it is already a soft penalty on the strategy path).
- SL / target / cooldown / same-zone guard / Gemini review UNCHANGED.  Tests: tests_v52_reclaim (5 new) + all earlier suites pass.  NOT measured on live data.

# v51b -- GLOBAL VETO no longer freezes the START of a big move  [from the 21-Oct screenshots: "Global research is STRONGLY BUY against this SELL"]
Global research reads yesterday's close, so the veto fires most exactly when a new move begins (the live lane needs score >= 45 + 80 % consensus, which only exists
LATER in the move).  The engine's own post-mortem also shows global alignment does not predict wins (aligned 28 % vs not aligned 30 %, "farq saabit nahi").
- `GLOBAL_VETO_EARLY_RELIEF`: the veto becomes a 0.20 penalty as soon as live price action >= 25 in the trade direction + >= 2 live strategies agree + opposite consensus <= 30 %
  + no severe chase.  The edge bar, win-chance floor, cooldowns and Gemini still decide.  Flat tape (< 25) = veto still blocks.  Tests: 3 new (16 in tests_v51).

# v51 -- BIG MOVES NO LONGER MISSED: side-aware win-rate, live tape outvotes stale global data, opening V-reversal, Upstox analytics token  [from the 26-Oct review]
What the 26-Oct screenshots + the code showed (the engine took ZERO trades while Nifty went 22,433 -> 22,631+):
- 09:31 the feed was `YAHOO_FALLBACK_NON_TRADABLE` (nobody had logged in to Upstox after the 03:30 token expiry) -> "no named strategy" -> no trade possible.
- 09:35-09:45 "Opening range forming": v50 allows an entry before 09:45 only when Model B is STRONG_ALIGNED.  Global data was bearish at the open, so the
  V-reversal BUY was impossible by rule.
- 10:04 five BUY strategies live, consensus BUY 100 vs SELL 0, live price action +90 -- and still "No positive edge: win chance 26.9 %".  Root cause in
  `edge_model.estimate_edge`: the RELIABILITY CORRECTION subtracts (model's average prediction - the engine's real win rate) from EVERY candidate.  The real record
  (67 trades, ~25 % wins, almost all SELLs taken into opening whipsaws) was therefore charged to a BUY that had nothing to do with it.  A fully aligned BUY was rated 34 %.
  Fewer trades -> no new BUY data -> the drag never healed.  Same side-blindness in the mistake-memory ("your nearest neighbours lost" -- the neighbours were SELLs).
- Evidence the filters are not discriminating: blocked ("shadow") setups won 42-48 % vs 25 % for the real trades; Gemini-approved trades won 37 % vs 42 % for the ones it rejected.
Changes (every one has a kill switch; SL / target / cooldown / same-zone guard / Gemini review / post-mortem UNCHANGED):
1. SIDE-AWARE CALIBRATION (`trade_learning.SIDE_AWARE_CALIBRATION`): a BUY is calibrated on the BUY record, a SELL on the SELL record.  A side with < 4 trades of its own is not
   punished for the other side.  Rows stored without features (very old) count for both sides.  A side whose OWN record is bad is still stopped (tests).
2. SIDE-AWARE MISTAKE MEMORY (`learner_nn.SIDE_AWARE_MEMORY`): BUY compared with past BUYs, SELL with past SELLs (needs 8 same-side rows, else no verdict).
3. LIVE DOMINANCE (`ai_trade_decision.LIVE_DOMINANCE_RELIEF`): when live price action >= 45, >= 3 live strategies, consensus >= 80 % (the old `_momentum_lane_ok`) all point the
   same way, Model B only has to be "not STRONGLY opposed" (before: had to AGREE) for the momentum lane, the learned-penalty caps and the opening-hour EV bar.  The lane still
   needs win chance >= 36 % and can waive at most 0.10 R, so a real losing record is NOT bypassed.  STRONG_OPPOSED gets no relief.
4. OPENING V-REVERSAL (`ai_trade_decision.OPENING_LIVE_EXCEPTION`): 09:25-09:45 an entry is allowed without Model B STRONG_ALIGNED when the 5-min structure has ALREADY turned in the
   trade direction (trend-transition confirmed) + live dominance + Model B not opposed + engine win chance >= 38 %.
5. UPSTOX ANALYTICS TOKEN (`upstox_auth._analytics_token`): free, read-only, valid 1 YEAR -- add `UPSTOX_ANALYTICS_TOKEN = "..."` to Streamlit secrets and the live feed is on all day
   without the daily login (normal login still works and is preferred when present).  A 401 from the LTP call now clears a dead token immediately.
- Each new trade stores `_live_dominance` {live_dom, relief_used, opening_exception} so the effect can be audited afterwards.
- Tests: tests_v51_live_dominance (13 new) + all earlier suites pass.  (tests_level_verdict was already failing before v51 -- untouched.)
- NOT verified: no live/backtest market data was available here, so the effect on win rate is NOT measured.  Watch the first days in the Trade Post-Mortem / Shadow panels.

# v50b -- global veto no longer fires on every refresh: it steps aside (becomes a 0.20 penalty) while live price action + strategies strongly confirm the trade (`GLOBAL_VETO_RELIEF`). It still blocks when the live market does NOT confirm.

# v50 -- TWO MODELS TOGETHER: Model A (structure + strategies) + Model B (macro / global-led, the old v41 idea)  [from the 22-Oct 5-straight-loss review]
What the diff old(v41) -> v49b showed: v42/v43 loosened the engine in four places at once -- (1) STRONG opposite global research became a 0.20 penalty
(v29..v41: hard veto), (2) a "momentum lane" waived the learned win-rate bar completely, (3) the learned penalties (mistake memory/NN, repeat-mistake tags)
were capped at 0.30 / 0.20, (4) coarse segments no longer block.  Result on 22-Oct: 56 resolved trades, 16W/40L (28.6 %), 42 "immediate reversal" cases
with 16.7 % wins, and three trades taken at 33-44 % honest win chance.
- NEW `macro_confluence.py` (Model B): global research (strong/moderate) + world-equity average + India news + Nifty-50 news + fundamentals + breadth +
  order flow + PCR velocity + VIX -> signed score -> STRONG_ALIGNED / ALIGNED / NEUTRAL / OPPOSED / STRONG_OPPOSED.  Pure, no I/O, kill switch `ENABLED`.
- Global research STRONG against the trade = VETO again (`ai_trade_decision.GLOBAL_VETO`).  The v43 fix is kept: STRONG needs the world EQUITY markets behind it,
  so VIX/DXY/yield/crude points alone can not create a false STRONG and freeze a trending market (the 21-Oct case).
- Agreement of both models earns leniency; disagreement gets the old strict behaviour:
    * Model B agrees -> small probability bonus (-0.08 / -0.15 logit), the v42 caps on learned penalties stay, momentum lane is possible.
    * Model B opposes -> penalty (+0.15 / +0.30 logit), learned penalties applied in FULL (v41 behaviour), no lane.
- Momentum lane tightened: needs Model B ALIGNED, win chance >= 36 % (`LANE_MIN_WIN_PROB`), and may waive at most 0.10 R of EV (`LANE_MAX_EV_SHORTFALL`).
- Opening range: before 09:45 only a trade with Model B STRONG_ALIGNED (global research strong with it); 09:45-10:00 the EV bar is +0.05 R unless Model B agrees.
  (Market autopsy: "opening volatility" = #1 turn reason, 11 of 31 big moves; two of the five losses were opening SELLs into a V-reversal.)
- UI: one line under every trade shows both models and whether the lane was used.  Each stored trade has `macro_model_aligned` and `_macro`.
- Tests: tests_v50_two_models (10 new); tests_edge_v36 / v42 / v43 updated to the new rules (veto restored, lane needs both models).
- NOT changed: SL, target logic, cooldown + same-zone guard (v49), Gemini final review, post-mortem, Market Brain, gate log.

# v49b -- Google Search visibility (diagnostics only; no trading logic changed)
Finding (22-Oct screenshot): ALL 9 stored approvals say "no search" -> the grounded (Google Search) call inside the review fails every time and silently
falls back to plain JSON.  The cause is not visible from the screenshot, so now: each new trade stores WHY the search call failed (`search_error`) and whether the
separate Google-grounded news brief (which is added to every review prompt) was available and really grounded (`news_brief`); both show under each trade in
"Har trade ke peeche Gemini ka report".  Gemini scorecard also lists approvals model-wise (jeet/total).

# v49 -- stop the "same failed zone, same side" re-entry + opening whipsaw  [small, narrow; SL / target / other gates UNCHANGED]
From the 5 straight losses (10-21 13:53, 14:10; 10-22 09:41, 09:57, 10:33):
- SAME-ZONE GUARD (`trade_learning.ZONE_GUARD`, kill switch = False): after a same-side STOP-OUT, no new same-side entry for 60 min unless price has
  moved >= 0.5 x that trade's risk beyond the failed entry in our direction.  Two of the five losses were exactly this (SELL 22683.8 -> 22687.1,
  SELL 22636.1 -> 22629.1, each stopped again).  Reason text starts with "Cooldown", so it is counted under the Cooldown gate.
- Gemini's review packet now also states: trades stopped out today, clock hour, day move %, position in the day's range; prompt rule 2d asks it to
  reject a same-zone repeat and to be extra strict before 10:00.  (Google Search was already used first, plain-JSON fallback only if the grounded call fails;
  each stored trade shows "Google Search ON / no search".)
- New risk tag `opening_whipsaw` (09:30-10:00).  Like every tag it only costs probability after the user's own trades prove it bad (>= 6 weighted samples, win < 30%).
- Tests: 175 pass (5 new).

# v48 -- TRADE POST-MORTEM (har trade ka "kyon" seekhna)  [purely additive; trading logic, gates, SL, target UNCHANGED]
New: `trade_postmortem.py`, `tests_postmortem_v48.py`.  Touched (small, guarded): `trade_learning.py` (+ read-only `get_resolved_full`),
`market_memory.py` (+ `load_pm_days`; day autopsies ignore `pm-` rows), `market_brain.py` (post-mortem panel + Gemini note).
- Every closed trade gets a review: best/worst move in R, minutes held, SL size vs ATR, and for a LOSS what happened in the next 90 min
  (stop-hunt then target reached / wrong direction / whipsaw / never worked / worked then failed) + a Hinglish story.
- Factor & group scoreboard (global, news, Nifty50 fundamentals, strategy/structure, option-chain flow, entry quality): win rate when aligned
  vs not aligned, Wilson-tested, "data kam" until >= 8 trades each side.  Pure-noise test: <= 1 false verdict / 40.
- Gemini's review prompt gets a <= 440-char note (last losses' reasons + proven reliable / misleading groups).  Advisory only.
- One stored row per day (`pm-YYYY-MM-DD`).  Kill switch: `trade_postmortem.ENABLED = False`.
- It does NOT change the engine probability: the engine already learns every factor via trade_learning / learner_nn / Market Brain.
- Tests: 166 pass across all suites (11 new).
- v48b: (1) WARNING-COUNT LOSS FILTER: counts how many signal groups (global / news / fundamentals / strategy / flow / entry quality) argued against a
  trade at entry.  If the user's OWN closed trades PROVE (Wilson, >= 10 trades in both buckets) that "half or more groups against" trades win less, a new
  candidate in that state gets a PENALTY of 0.10 logit (~2.5 pct pts).  Penalty only, never a bonus, inert until proven, switches itself off if the
  evidence disappears.  Kill switch `trade_postmortem.WARN_GATE = "off"`.  (2) MISSED-MOVE REVIEW: for every 50+ pt move the engine did not trade,
  the gate that was answering at its start (from gate_log) + gate-wise counts.  Advisory.  Tests: 170 pass.

# v47 -- WHOLE-TOOL DAILY UPGRADE LOOP  [additive; every part bounded, fail-safe, with a kill switch]
New: `market_adaptive.py`, `tests_market_brain_v47.py`.  Touched (small, guarded): `market_memory.py` (records which named strategies are live),
`market_brain.py` (Upgrade Report panel), `strategy_engine.py` (one fallback), `ai_trade_decision.py` (Gemini memory block + earned nudge).
- BUG FOUND: `strategy_engine` called `trade_learning.get_strategy_performance()`, which does not exist -> strategies never learned from live
  outcomes (the AttributeError was swallowed).  Now the Market Brain strategy scorecard feeds the same hook: per-strategy "episodes" scored
  against a random candle, shrunk toward neutral, needs >=10 episodes, effect on a strategy score is a few points at most (engine clamp +/-8).
- Gemini memory: its review prompt gets a <=1500-char block (its own approve/reject track record, active patterns with real stats, forecaster
  status, weak/strong strategies, big-move catch rate).  Advisory: prompt says it never overrides the live packet.
- Earned engine nudge (`market_adaptive.ENGINE_ADJUST = "auto"`, set "off" to disable): +/-0.15 logit max (~3.5 pct pts).  Forecaster part
  needs a TRUSTED model AND >=30 live validated forecasts with >=55% right; pattern part needs ~100 candles and a 99% edge after deflating for
  overlapping candles.  Pure-noise test: 0 false nudges / 24.  Inert until then; switches itself off if evidence disappears.
- Daily upgrade log: one `stats-YYYY-MM-DD` row per day (stored in `market_autopsy`) -> trend table in "Roz ka Upgrade Report", plus plain-language
  suggestions (never auto-applied).  Hard gates, SL/target rules, cooldown, trade tables: UNCHANGED.
- Tests: 12 new, all pass.  The 7 failures in `tests_level_verdict` already existed before v46/v47 (identical on the original zip).

# v46 -- MARKET BRAIN (roz seekhne wala dimaag)  [purely additive; no trade logic changed]
New files: `market_memory.py` (recorder), `market_learner.py` (patterns + forecaster), `market_autopsy.py`, `market_brain.py` (panel),
`SUPABASE_MARKET_BRAIN_MIGRATION.sql` (run once).  `app.py`: one import + one guarded call above the Shadow-trades expander.
- Recorder: every completed 5-min candle -> feature snapshot (price/volume/RSI/EMA/VWAP/day position/opening range, + live OI walls, PCR,
  VIX, breadth, global mood, strategy state).  60 min later the outcome is added (max up/down, which of +/-25 and +/-50 pts was hit FIRST).
  First run back-fills the candles the app already has, so learning starts on day one.
- Autopsy: zig-zag swings for every day; for each reversal the data-visible reasons (stop-hunt wick, prev-day level, volume spike, RSI/EMA
  stretch, reaction candle, VWAP, round number, OI wall) and whether the engine caught the move.  Finished days are stored permanently.
- Pattern memory: ~20 named situations with their real 60-min outcome statistics.
- Forecaster: logistic + small neural net ensemble.  P(+25 first), P(+50 first), P(50-pt move in 60 min).  Rolling-origin tested on the past
  only; "trusted" needs avg AUC >= 0.58, worst fold >= 0.52 and a win over the majority baseline (0/6 false-trusted on pure noise in tests).
  Trains in a background thread; live forecasts are stored and scored once their outcome is known (Scorecard).
- ADVISORY ONLY (`market_learner.FORECAST_GATE = "off"`): it does not open, block or resize any trade.  Switch to a soft penalty only after
  the scorecard shows it helps on the engine's own trades.

# NIFTY AI -- v43: global research can no longer block a falling market; v42 restored; 5-min chart behind a switch

- **Why "no trade" on 21-Oct:** every candle 10:50-11:10 had SELL strategies live and price action at -55, but the gate said
  "Global research is STRONGLY BUY against this SELL".  That "STRONG BUY" came from macro points (calm VIX, soft DXY, falling yields, oil down)
  while the global tracker itself read "Mild Bearish" (4 up / 6 down).  It was also a HARD veto that ran BEFORE Gemini, so Gemini's Google-Search
  research never even got asked.
- **Fix 1 (`global_research.py`):** "STRONG" now needs the global EQUITY markets to back it (>= 2 more up than down and a positive/negative average).
  Macro-only points reach MODERATE at most.
- **Fix 2 (`ai_trade_decision.py`):** a strong opposite global read is now a probability penalty (`GLOBAL_OPPOSE_PENALTY` 0.20), not a block.
  Gemini (with Google Search, plus a cached news brief) judges the global picture.
- **v42 was missing from the repo (only tests + tick_guard.py were there).  Restored to the tests' spec:**
  momentum-capture lane (live score >= 45, strategy consensus >= 80%, >= 3 live strategies, no severe chase -> the history-driven win-rate bar is
  waived; Gemini review, 28-pt stop, cooldowns still apply), learned-drag cap 0.30 / tag-penalty cap 0.20, OI target picker (first wall that pays,
  closer wall = small penalty), `trade_learning.resolve_tick` (SL/target on the live tick, started by `tick_guard` from app.py), Gemini news brief.
- **Self-learning segments:** a whole grade / side / hour no longer hard-blocks (soft penalty instead); only playbook / playbook+grade / probe
  can block, and only trades from the CURRENT logic version count.
- **UI:** live chart is 5-minute and sits behind a one-line "Live 5-min chart dikhao" switch (OFF = not fetched at all);
  "Named Strategy Library" is always collapsed to one line.
- Tests: `python -m unittest tests_edge_v36 tests_v37_quality tests_v41_visibility tests_v42_momentum_tick tests_v43_global_chart` -> 143 pass.

# NIFTY AI -- v41: "why no trade?" is now answered from disk (no trading rule changed)

- **Problem:** a big morning move and no trade, but the only evidence was a per-session table that restarts on every page reload (the keeper robot reloads
  every 30 min), plus a generic "No named strategy..." sentence. It was impossible to tell a gate from "the engine was not even running".
- **New `gate_log.py`:** every 5-min candle the engine's verdict (gate name, direction, price, strategy scores, reason, data source) is written to a local
  SQLite file (`gate_log.db`, 14 days, git-ignored). New expander "Aaj ka engine timeline" shows today's range, trades taken, a per-gate count, the last 60
  candles, and a warning when the engine was not running (late start / holes > 15 min). A big move + zero trades now says which of the two it was.
- **"No named strategy" now explains itself:** data source if it is not UPSTOX_LIVE, how many live strategies exist, how many the live price action blocked
  (and the first reason), best BUY/SELL score vs the minimum. Text is only appended, so gate classification is unchanged.
- No entry/exit/target/gate/learning logic was touched. Tests: `python -m unittest tests_v41_visibility` (8); with the older suites 122 pass.
  (`tests_level_verdict.py` is a legacy file that already failed before v41: it calls an old API.)

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
