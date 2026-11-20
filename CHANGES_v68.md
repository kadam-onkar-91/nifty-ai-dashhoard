# v68 — SL/target chhota, fundamentals + news fix, fast refresh (logs ke hisaab se)

Zip me CHANGES_v66/v67 likhe the par unka code (SL 28, sidebar crash fix, sqlite timeout) files me tha hi nahi -- wo sab ab lag gaya hai.

1. **Nifty SL/Target**: SL = 1.5 x ATR, 24 se 26 pt ke beech (kabhi 28 se bada nahi). Target kam se kam 1.5R aur kabhi 40 pt se zyada nahi (`MAX_TARGET_PTS`). Pehle SL 30-45 aur target 100 pt tak jata tha. (`ai_brain.pym`)
2. **Stock SL**: stock ke apne ATR ke hisaab se -- base 1.2 x ATR, ceiling 1.5 x ATR, floor price ka 0.15 %, upar se 0.6 %. Nifty ka 40-pt cap stocks par nahi lagta. (`stock_suite.pym`)
3. **Fundamentals "unavailable" ki wajah**: Streamlit Cloud par Yahoo `.info` har call par `401 Invalid Crumb` deta hai (log me 100+ baar har refresh) -- 50 stocks + banks + stock page sab wahi call karte the. Naya `fundamentals_hub.py`: disk/memory cache (12 ghante), Yahoo circuit-breaker (2 fail -> 30 min band), fallback screener.in, har number ke saath source + umar. Bank-fundamentals, Nifty-50 aggregate aur stock fundamentals teeno isi se.
4. **Loading/refresh fast**: Nifty-50 fundamentals ab page ko block nahi karte (background + server start par prewarm; pehle 45 sec tak ruka rehta tha). Market breadth ke 2 calls ek saath (pehle 2 x 20 sec ek ke baad ek). Sidebar crash (`StreamlitInvalidLayoutContextError`) fix. sqlite "database is locked" fix (timeout 30). Walk-forward SL live SL se match.
5. **News**: Nifty news ab 3 market queries ek saath, latest pehle. Naya "📰 Badi khabar -> market par asar" table: headline se turant (bina API) -- impact level, asar kab, kitni der. Purana events/impact table (NVIDIA/Gemini) pehle jaisa.
6. Tests: 325 OK (naye: SL/target cap, fundamentals_hub parser/breaker/cache, headline impact).
Revert: purani files wapas daal do, `fundamentals_hub.py` hata do.
