# v66 — sirf kaam ki news/events (aane wale + abhi asar chalu), har stock / Nifty ke apne, crash fix
Sirf 2 file badli hain: `news_intel.py` aur `tests_all.pym`. Engine, gates, SL/target, loading, refresh -- kuch nahi chheda.

1. CRASH FIX (asli wajah): `news_intel.fetch_events` `ai_trade_decision.REVIEW_MODELS` maangta tha jo ab hai hi nahi -> Gemini wala events thread har baar mar jata tha, isliye Nifty page par sirf expiry/US-open (fixed) dikhte the. Ab safe default models.
2. "HO CHUKA" band: har render par har row check hoti hai -- jo event nikal chuka aur jiska market reaction poora ho gaya, wo gayab. Dikhta sirf: (a) AANE WALA, (b) "abhi asar chalu" (jaise market band hone ke baad aayi news, jiska reaction agle 09:15 + 1 ghante tak baaki). Impact 1-2 wali rows nahi dikhti.
3. IMPACT + KAB: har row me Impact (MEDIUM/BADA/BAHUT BADA), countdown ("2 ghante 10 min baad"), "Market me asar kab/kahan" (09:15 gap / market chalu me turant / expiry ka aakhri ghanta), kitni der asar, kyun, headline ka sabut. Time ke hisaab se sort (live pehle, phir jo sabse pehle aane wala).
4. HAR PAGE KI APNI NEWS: stock = us stock ke 6 targeted Google-News searches (company / aane wali date: results-board-record date-AGM / orders-deals / SEBI-court-rating / sector). Nifty 50 = 8 index-level searches (RBI-policy, Fed/US data, FII-DII, crude-dollar, India CPI-GDP, heavyweights results, expiry-NSE, geopolitics). Pehle NVIDIA (Gemini nahi); NVIDIA fail ho to Gemini+Google; wo bhi fail ho to rule-based headline reader -- table khaali nahi rehta.
5. FRESH + FAST: sab kuch background thread me (page sirf dict padhta hai -> loading/refresh ka time bilkul same). Headlines har 10 min fir se padhi jaati hain (RSS, free); model sirf tab dubara bulaya jata hai jab headlines badli hon (ya 3 ghante baad) -> quota bachta hai. Naya check chalte waqt purani rows dikhti rehti hain (spinner se khaali nahi).
6. Tests: tests_v66_best_events (8 naye). Poora suite: 314 OK.
Revert: purani news_intel.py wapas daal do (v65).

## v66 (part 2) — sidebar/fragment error fix
7. `StreamlitInvalidLayoutContextError` (log me 3 baar): Nifty fragment ke andar sidebar me "Upstox Live Authentication" likhte waqt run beech me mar jata tha. Fix: (a) full run me `st.sidebar.empty()` se slot claim (app.py), (b) `get_upstox_access_token` ab us ek run ke liye sidebar-widgets chhod kar token logic poora chala deta hai (core_infra.pym) -- login/token kabhi nahi rukta. Test: TestV66SidebarFragment.
Poora suite: 315 tests.
Files badli: news_intel.py, tests_all.pym, core_infra.pym, app.py.

## v66 (part 3) — stop-loss maximum 28 points
8. Nifty ka SL ab kabhi 28 point se bada nahi (pehle 45 tak jata tha). Stop = 1.5 x ATR, 24 se 28 ke beech clamp; structure-aware widening bhi isi 28 se capped. Target ab bhi kam se kam 1.5R (28 pt SL -> ~42 pt target). Baaki engine/gates bilkul same. Sirf `ai_brain.pym` ke 3 constants badle (MIN_SL_PTS 24, MAX_SL_PTS 28).
   Stocks ka SL apne price ke hisaab se alag scale hota hai (stock_suite) -- usko nahi chheda.

## v66 (part 4) — stock SL: Nifty jaisa, stock ke ATR ke hisaab se
9. Stock ka SL ab Nifty ke naye 24-28 pt jaisa hi ATR-ratio par: base stop 1.2 x ATR, ceiling 1.5 x ATR (structure-aware widening iske upar nahi jaati). Pehle floor price ka 0.25 % tha (shant stock me Nifty ke ~0.11 % se 2x bada SL) -- ab floor 0.15 % (round-trip cost ka 3x), aur ek ceiling 0.6 % price ki, taaki volatile spike me bada SL na bane. Target kam se kam 1.5R, baaki engine same.
   Example: Rs 1000 stock, ATR 2.0 -> SL Rs 2.4 se Rs 3.0 (pehle Rs 2.6 se Rs 3.8).
   File badli: stock_suite.pym. Tests: TestV66StockStopScale. Poora suite 319 OK.
