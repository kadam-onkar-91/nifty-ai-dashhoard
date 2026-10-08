# Mobile band, tab chalta rahe (free "robot tab")

Streamlit tabhi chalta hai jab koi browser tab khula ho. GitHub Actions (free) market ke time ek headless Chrome me tumhari app
khol ke rakhta hai -- jaise tum phone pe website khule rakhte ho.

## Ek baar setup (5 minute)
1. Ye 3 cheezein repo me upload karo (folders ke saath):
   - `.github/workflows/keep_app_awake.yml`
   - `.github/keeper/keep_open.py`
   - `KEEPER_SETUP.md` (optional)
   (GitHub me Add file -> Create new file -> naam me `.github/workflows/keep_app_awake.yml` likho, content paste karo.)
2. Repo -> **Settings -> Secrets and variables -> Actions -> Variables tab -> New repository variable**
   - Name: `APP_URL`   Value: tumhari Streamlit app ka poora link (browser se copy karke)
   - (Optional) Name: `SKIP_DATES`  Value: market holidays, jaise `2026-11-08,2026-12-25`
3. Repo -> **Actions** tab -> "keep dashboard open in market hours" -> **Run workflow** (ek baar test).
   Logs me "tab alive" har minute dikhna chahiye. Test sirf kuch minute chalao, phir Cancel kar sakte ho.
4. Uske baad ye Mon-Fri 09:00 se 15:35 IST tak apne aap chalega.

## Roz ka kaam
- Subah market se pehle **ek baar** Upstox login (app kholo -> Login with Upstox -> PIN). Token server pe save hota hai,
  to robot tab ko bhi mil jaata hai. Token agle din ~3:30 AM pe expire hota hai.

## Zaroori dhyan
- **Free minutes:** public repo me unlimited. **Private repo me sirf 2000 min/mahina** milte hain, jabki ye ~8000 min lega.
  Private rakhna ho to ye free me nahi chalega (tab Oracle Cloud ka free VM use karo, batao to guide kar dunga).
- Public karne se pehle: repo me `trade_learning.db` ya koi secret/key to nahi? (secrets sirf `.streamlit/secrets.toml` / Streamlit
  Secrets me rakho, wo repo me nahi jaati.) Trade history Supabase me hai.
- GitHub ka schedule kabhi 5-15 minute late ho sakta hai. Subah 09:00 pe start hota hai, to 9:15 tak aam taur pe chal jaata hai.
- Agar kuch na chale: Actions -> run -> neeche "keeper-last-screenshot" download karke dekho robot ko kya dikha.
