# v65 — loading + har stock ki apni news/events (baaki engine/gates/SL/target bilkul same)

1. LOADING: keep-open "robot tab" ab URL me `?robot=1` ke saath khulta hai (keep_open.py). Jab aap Nifty page par kaam kar rahe ho, robot tab sirf halka "standby" page dikhata hai (IST heartbeat ke saath) -- pehle dono tab poora heavy pipeline ek saath chalate the (log me har step 2-5x slow). Jab aap stock page load kar rahe ho, robot tab us load ke khatam hone tak (max 75 s) ruk jata hai, phir Nifty engine chalata hai. Aap jaate hi robot tab engine wapas poora chalata hai.
   Stock fragment ab "run start / abort" bhi batata hai (watchlist robot ko pata chalta hai ki stock page abhi load ho raha hai).
2. NEWS/EVENTS (stock page): ab har stock ki apni research. 5 alag Google-News RSS searches (company / results-board / orders-deals / SEBI-court-promoter / sector) -> asli dated headlines -> NVIDIA NIM model (Gemini nahi) unhe "ho chuka" aur "aane wala" events me badalta hai, "asar kab se" aur "kitni der" ke saath. Har row ke saath headline ka sabut; jis row ka sabut nahi, uspar "(AI andaza)". Nifty ke generic events (expiry/US open) ab stock par nahi dikhte. Automatic, background me, har stock ke liye 2 ghante me ek baar; button se dobara.
   Zaroori: Streamlit Secrets me NVIDIA_API_KEY (ya NVIDIA_API_KEY_1..5) -- wahi jo chat box ke liye hai. Key na ho to purana Gemini+Google fallback chalta hai (stock ke naam ke saath).
3. Tests: tests_v65_stock_events (4 naye). Poora suite: 306 OK.
Revert: keep_open.py me robot=1 wali 2 line hata do -> standby band.
