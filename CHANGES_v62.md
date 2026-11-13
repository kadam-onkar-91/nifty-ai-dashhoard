# v62 — kya badla (sirf selection + stop; baaki logic same)
1. Nifty SL: 0.9xATR (20-28 pt) -> 1.5xATR clamp 30-45 pt (ai_brain: SL_ATR_MULT / MIN_SL_PTS / MAX_SL_PTS). Target kam se kam 1.5R rehta hai. Stocks ka SL pehle jaisa (1.3-1.9 ATR).
2. (v63) Win-chance floors wapas purane (.36/.34/.38) -- koi extra tightening nahi, taaki move ke baad trade na rukey.
3. Lane scorecard (trade_core): momentum/tape/evidence/live_dom/reaction lane ka apna real record (>=8 trades, win < 34%) -> lane pause. 'base' lane kabhi pause nahi hoti. Har stock ka apna record.
4. BAD_FORM_EV_EXTRA 0.0 -> 0.08 (kharab form me bar thoda upar; 30h baad khud expire).
Revert: constants ko purani value pe karo (SL_ATR_MULT=0.9, MIN 20, MAX 28; LANE_GUARD=False).
