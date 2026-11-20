# v68 — crash fix + SL 28 cap + faster load (trade logic / calculation: ZERO change)
Zip me v66 (part 2-4) aur v67 ki CHANGES likhi thi par unka code files me tha hi nahi -- wo ab laga diya.
1. StreamlitInvalidLayoutContextError: Nifty fragment ke andar sidebar ("Upstox Live Authentication") likhne se crash. Fix: app.py full run me slot claim + core_infra.pym `get_upstox_access_token` ab safe (panel skip ho sakta hai, login/token kabhi nahi rukta).
2. Nifty SL: MIN 24 / MAX 28 pts (pehle 30-45) -- ai_brain.pym. Structure-aware widening bhi 28 se capped.
3. Stock SL: base 1.2 x ATR, ceiling 1.5 x ATR, floor 0.15 % / cap 0.6 % of price (stock_suite.pym `scale`).
4. "database is locked": har sqlite connect timeout=30, WAL + busy_timeout.
5. 'N/A' gradient crash: tables ke number columns pehle numeric banaye jaate hain.
6. SPEED: breadth (2 calls), global news+markets, Nifty change ab run ke shuru me ek saath start hote hain (safe_io.start) -- total time = sabse slow source, sum nahi.
7. Log me naye [STEP] lines (Gate log, Market brain, Track record, News render) -- agli baar exact dikhega kaun sa hissa slow hai.
Revert: purani files wapas.
