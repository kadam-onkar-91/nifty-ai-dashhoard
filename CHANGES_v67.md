# v67 — speed + lock fix (calculation / trade logic: ZERO change)
1. database is locked: every sqlite connect has timeout=30, WAL mode + busy_timeout (core_infra, market_intel, stock_suite).
2. 'N/A' in Global Sentiment table no longer crashes the gradient styling (app.py, stock_suite).
3. SPEED -- walk-forward statistics (41 strategies, ~75 s CPU) now run in a separate low-priority process instead of a thread that fought the page for the GIL (strategies.pym `_wf_offload`; falls back to the old thread path on any problem).
4. SPEED -- each historical bar's strategy signal is remembered with a checksum of all history before it; a refresh recomputes only new bars (77 s -> 0.5 s, results verified bit-identical to the uncached run).
5. SPEED -- the two market-breadth network calls run at the same time (app.py).
Tests: 319 OK.  Revert: put back v66 files.
